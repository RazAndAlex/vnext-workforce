"""Persistent Codex/Claude runtime behind the host's session interface.

The presentation surface is an attachment. It does not own an agent, scheduler,
or provider process. Constructors never start a model turn.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .vnext_model_identity import identity_view
from .host_contract import RuntimeEvent, SessionRestoreRequest, SessionStartRequest, TerminalAttachment
from .vnext_orchestration import (
    AgentRecord, AgentRole, AgentStatus, EconomicPreset, EvidenceKind, ModelCard, ModelRegistry,
    OrchestrationControlPlane,
    ProtocolError,
    RegistryClaim,
    TERMINAL_STATUSES,
)
from .vnext_managed_session import ManagedSessionError, VNextManagedSession
from .vnext_runtime_types import (
    CREDENTIAL_ADJACENT_STDERR_PROVIDERS,
    ToolCallResult,
)
from .vnext_runtime_effects import RuntimeEffectJournal, runtime_effect_reader
from .vnext_report import closed_by_shutdown, provider_refused
from .vnext_scheduler import SchedulerCancelled, SchedulerError, SchedulerHooks, VNextScheduler
from .workforce_contracts import RunCancellation


PROVIDERS = {
    "codex": ("app-server", "user-codex-home"),
    "claude": ("claude-agent-sdk", "subscription"),
    "zai": (
        "claude-agent-sdk",
        "z.ai provider API key (~/.vnext/providers.json)",
    ),
    # The Codex runtime, pointed at a reseller's OpenAI-compatible endpoint by
    # a loopback bridge.  Same harness as "codex" and a different credential.
    "commandcode": (
        "app-server",
        "Command Code provider API key (~/.vnext/providers.json)",
    ),
    # A primary that already exists outside this service: its own client runs
    # its turns and its own permission system reviews its work.  vNext holds
    # the record and the child tree, and starts no process for it.
    "external": ("external-mcp", "client-owned"),
}

# Provider events can arrive after a native thread exists but before the
# managed binding is committed. Keep that short attribution race recoverable,
# but never allow a stale/foreign stream to grow without limit.
_PENDING_NATIVE_EVENT_LIMIT = 256

# What a provider error is called when vNext itself closed the stream it arrived
# on.  The provider's own fields travel with it, so nothing is lost; the name
# says who stopped the turn.
_STREAM_CLOSED_AT_SHUTDOWN = "stream_closed_at_shutdown"

# How long the last drain of a shutdown waits for a settled worker's final
# usage message.  A worker goes terminal about 1.1 s before the SDK result
# message that carries its tokens and its dollar figure, and a quit inside that
# gap used to cancel the reader and leave the outcome row null.  This has to fit
# inside the server's whole close grace (CLOSE_GRACE_SECONDS, 5.0 s) with room
# for the steps that follow, and it is spent only when a release is pending.
_FINAL_USAGE_WAIT_SECONDS = 3.0

# The reason a row carries when the wait itself was sound and the message simply
# did not come in time.  A fault has its own reason, naming what broke.
_USAGE_GRACE_EXPIRED = (
    "the close grace expired before this worker's final usage message arrived, "
    "so its tokens and cost are unknown"
)

# One clean retry absorbs a transient provider launch without turning every
# later manager call into another process launch when the provider is broken.
_ROOT_STARTUP_ATTEMPT_LIMIT = 2


def session_registry(request: SessionStartRequest) -> ModelRegistry:
    """Use explicit catalog choices; model names never determine authority."""
    models = request.catalog_config.get("models", [])
    if not isinstance(models, list):
        raise ValueError("catalog_config.models must be a list")
    entries = [dict(entry) for entry in models if isinstance(entry, Mapping)]
    if len(entries) != len(models):
        raise ValueError("every catalog model must be an object")
    primary = dict(request.primary)
    primary_model = primary.get("model") or primary.get("model_id")
    if not isinstance(primary_model, str) or not primary_model.strip():
        raise ValueError("primary.model is required")
    primary["model"] = primary_model
    # The host normally selects a catalog model by provider/model/effort rather
    # than copying its evidence claims into the primary selector. Preserve the
    # catalog card in that common path; an explicit primary claim list remains
    # a deliberate equality check below.
    if "claims" not in primary:
        catalog_primary = next(
            (entry for entry in entries
             if (entry.get("model") or entry.get("model_id")) == primary_model),
            None,
        )
        if catalog_primary is not None and "claims" in catalog_primary:
            primary["claims"] = catalog_primary["claims"]
    cards: dict[str, ModelCard] = {}
    for entry in [*entries, primary]:
        model = entry.get("model") or entry.get("model_id")
        provider = entry.get("provider")
        if not isinstance(model, str) or not model.strip() or provider not in PROVIDERS:
            raise ValueError("model catalog entries require a model and supported provider")
        harness, credential = PROVIDERS[provider]
        if entry.get("harness", harness) != harness:
            raise ValueError(f"unsupported harness for {provider}")
        # An external model is the client already holding this session open --
        # Claude Code itself.  vNext cannot start a second one, so a child
        # delegated to it can never run: it fails with "external primary
        # already has a thread" only after the manager has paid to spawn it.
        # Offering it for every role advertised a choice that is always wrong,
        # so the card claims the only role it can actually fill, and the
        # existing role-eligibility check refuses the rest by name.
        roles = frozenset({AgentRole.ROOT_MANAGER}) if provider == "external" else frozenset(AgentRole)
        card = ModelCard(model, roles, claims=_catalog_claims(entry), provider=provider,
                         harness=harness, credential_location=credential)
        if model in cards and cards[model] != card:
            raise ValueError(f"conflicting runtime selection for model {model}")
        cards[model] = card
    return ModelRegistry(list(cards.values()), [EconomicPreset(
        "session", frozenset(cards), frozenset(card.provider for card in cards.values())
    )])


def _catalog_claims(entry: Mapping[str, Any]) -> tuple[RegistryClaim, ...]:
    """Parse only provenance-labeled catalog claims supplied for this session."""

    raw_claims = entry.get("claims", [])
    if raw_claims is None:
        raw_claims = []
    if not isinstance(raw_claims, list):
        raise ValueError("model catalog claims must be a list")
    claims: list[RegistryClaim] = []
    for raw in raw_claims:
        if not isinstance(raw, Mapping):
            raise ValueError("each model catalog claim must be an object")
        try:
            kind = EvidenceKind(raw.get("kind"))
        except (TypeError, ValueError) as exc:
            raise ValueError("model catalog claim kind is invalid") from exc
        statement = raw.get("statement")
        if not isinstance(statement, str) or not statement.strip() or statement.strip() != statement:
            raise ValueError("model catalog claim statement is required")
        pointer = raw.get("evidence_pointer")
        if pointer is not None and (
            not isinstance(pointer, str) or not pointer.strip() or pointer.strip() != pointer
        ):
            raise ValueError("model catalog claim evidence_pointer is invalid")
        claims.append(RegistryClaim(kind, statement, pointer))
    return tuple(claims)


class _RootStartupAdapter:
    """Mark only exceptions raised by the root provider's start operation."""

    __slots__ = ("_adapter", "_failed")

    def __init__(self, adapter: Any, failed: Callable[[], None]) -> None:
        object.__setattr__(self, "_adapter", adapter)
        object.__setattr__(self, "_failed", failed)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._adapter, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__slots__:
            object.__setattr__(self, name, value)
        else:
            setattr(self._adapter, name, value)

    def start_thread(self, **kwargs: Any) -> Any:
        try:
            return self._adapter.start_thread(**kwargs)
        except Exception:
            self._failed()
            raise


class VNextRuntimeSession:
    def __init__(
        self, request: SessionStartRequest, emit: Callable[[RuntimeEvent], None], *,
        adapter_factories: Mapping[str, Callable[[], Any]] | None = None,
        remote_server_factory: Callable[..., Any] | None = None,
        terminal_process_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.request = request
        self.emit = emit
        self.workspace = Path(request.workspace).resolve()
        if not self.workspace.is_dir():
            raise ValueError("session workspace does not exist")
        self.registry = session_registry(request)
        self.control = OrchestrationControlPlane(self.registry)
        self.root = self.control.create_session(
            preset_id="session", root_model_id=str(request.primary.get("model") or request.primary["model_id"]),
            workspace=self.workspace, objective="", session_id=request.session_id,
            root_agent_id=request.primary_agent_id,
            task_contract={"effort": request.primary.get("effort", "high"),
                           "criteria": ["meet the user's required quality", "verify the actual result"],
                           "instructions": request.main_preset.get("instructions", "")},
        )
        self._factories = dict(adapter_factories or {})
        self._remote_server_factory = remote_server_factory
        self._terminal_process_factory = terminal_process_factory
        self._terminal_lock = threading.RLock()
        self._native_terminals: dict[str, dict[str, Any]] = {}
        self._runtime_ready = threading.Event()
        self._adapters: dict[str, Any] = {}
        self._initializing_adapters: dict[str, Any] = {}
        self._remote_codex_server: Any | None = None
        self._remote_cleanup_pending = False
        self._codex_mcp_relay: Any | None = None
        self._codex_mcp_relay_cleanup_pending = False
        self._commandcode_bridge: Any | None = None
        self._commandcode_home: Path | None = None
        self._startup_cleanup_errors: list[Exception] = []
        self._cursors: dict[int, Any] = {}
        # Raw events stay unchanged. Their adapter-local cursor travels beside
        # them while attribution waits for a late thread/turn registration.
        self._pending_native: dict[int, list[tuple[Mapping[str, Any], Any]]] = {}
        self._pending_native_drops: dict[int, int] = {}
        self._lock = threading.RLock()
        self._adapter_lock = threading.Lock()
        self._cancellation = RunCancellation()
        self._shutdown = threading.Event()
        self._managed: VNextManagedSession | None = None
        self._scheduler: VNextScheduler | None = None
        self._thread: threading.Thread | None = None
        self._collector: threading.Thread | None = None
        self._queued: list[tuple[str, str, str, str]] = []
        self._terminals: dict[str, str] = {}
        self._history: dict[str, list[dict[str, Any]]] = {}
        self._status = "ready"
        # Why the session failed, kept beside the status because an
        # external client is told only that its call did not work and
        # cannot read the event stream to find out why.
        self._failure = ""
        self._root_startup_attempts = 0
        # Whether the current attempt got the control tree standing before any
        # failure.  It separates a settled startup failure from one that landed
        # after the runtime was already usable.
        self._runtime_stood_up = False
        self._root_startup_retryable = False
        self._root_provider_start_failed = False
        self._root_startup_retry_lock = threading.Lock()
        self._close_error: Exception | None = None
        self._cleanup_base_status: str | None = None
        self._retryable_cleanup_steps: set[str] = set()
        self._nonretryable_cleanup_errors: list[Exception] = []
        self._cleanup_retry_requested = False
        self._restored_agents: dict[str, dict[str, Any]] = {}
        self._primary_resumed = False
        # Set when this close is the end of a run that went well: no worker was
        # still live when the client went away.  The scheduler still stops the
        # same way, because cancelling the root is how a live conversation is
        # released; the flag changes what the record says about it.
        self._clean_close = False

    def start(self) -> Mapping[str, Any]:
        self._record_agent(self.root)
        self.emit(RuntimeEvent("session.upsert", {"status": "ready", "primary_agent_id": self.root.agent_id}))
        if self.external_primary:
            # A provider-backed primary waits for a prompt to bring the
            # scheduler up.  An external one never receives a prompt: its first
            # act is a tool call from its own client, so the tree has to be
            # standing before that call can arrive.
            self._start_external_primary()
        return self.snapshot()

    @property
    def external_primary(self) -> bool:
        """Whether this session's root is a client vNext does not own."""

        return str(self.registry.cards[self.root.model_id].provider) == "external"

    def _start_external_primary(self, *, timeout: float = 30) -> None:
        with self._lock:
            if self._shutdown.is_set():
                raise RuntimeError("session runtime is closed")
            if self._thread is not None:
                return
            self._thread = threading.Thread(target=self._run, name="vnext-session", daemon=True)
            self._thread.start()
        if not self._runtime_ready.wait(timeout):
            raise RuntimeError("external primary runtime did not start")
        with self._lock:
            # The runtime signals ready once the control tree is standing, and
            # the root can still fail after that.  Reporting such a failure here
            # depends on which thread reaches the status field first, so it is
            # left to the client's next tool call, which is also where a
            # retryable root startup is retried.  A failure before the tree
            # stood up is settled and belongs to this call.
            if self._status == "failed" and not self._runtime_stood_up:
                raise RuntimeError(self._start_failure_reason())

    def begin_closing(self) -> None:
        """Refuse any further birth of a turn or worker in this tree.

        Sticky, and read under the session lock by the calls that would start
        one, so a child already running when the close began is refused on the
        provider handler's path as well as on the client's.
        """

        self.control.begin_closing()

    def external_tool_call(
        self, *, tool: str, arguments: Mapping[str, Any], timeout: float = 30
    ) -> ToolCallResult | Mapping[str, Any]:
        """Run one manager tool for the client that owns this session's root.

        The bind happens inside the scheduler loop, a moment after the runtime
        reports ready, so the first call from a fast client can arrive before
        the handler exists.  Wait for the binding rather than failing a race
        the caller cannot see or retry meaningfully.
        """

        if not self.external_primary:
            raise ValueError("session primary is not external")
        deadline = time.monotonic() + max(0.0, timeout)
        if self._status == "failed":
            self._retry_root_startup(timeout=max(0.0, deadline - time.monotonic()))
        adapter = self._adapters.get("external")
        if adapter is None:
            self._runtime_ready.wait(max(0.0, deadline - time.monotonic()))
            adapter = self._adapters.get("external")
        if adapter is None:
            if self._status == "failed":
                raise RuntimeError(self._start_failure_reason())
            raise RuntimeError("external primary adapter is not available")
        if not self._wait_external_adapter_bound(
            adapter, timeout=max(0.0, deadline - time.monotonic())
        ):
            if self._status == "failed":
                # The retry can fail while this call waits for its root binding.
                # Report that attempt's cause, not the closed adapter left by it.
                raise RuntimeError(self._start_failure_reason())
            raise RuntimeError("external primary is not bound to a control tree")
        return adapter.dispatch_external_call(tool=tool, arguments=arguments)

    def _wait_external_adapter_bound(self, adapter: Any, *, timeout: float) -> bool:
        """Wait in short slices so a failed retry returns its cause promptly."""

        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            # Ask the adapter before consulting the clock.  A retry that spent
            # the whole budget getting here, or a caller that passed zero,
            # otherwise reports a perfectly bound tree as unbound because the
            # stopwatch ran out first -- a healthy session refusing its own
            # tools.
            if adapter.wait_bound(0):
                return True
            # A closed adapter returns False from wait_bound before it reaches
            # the sleep inside it, so the slice below costs nothing and the loop
            # burns a whole core for the rest of the timeout.  Measured on the
            # external primary: wall=5.00s cpu=5.00s for one refused call, and
            # up to _MAX_CONCURRENT_STDIO_CALLS of them can be in flight while
            # the process is trying to exit inside its kill grace.  There is
            # nothing left to wait for either way.
            if getattr(adapter, "closed", False):
                return False
            if self._status == "failed":
                return False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            if adapter.wait_bound(min(0.05, remaining)):
                return True

    def _retry_root_startup(self, *, timeout: float) -> None:
        """Replace one cleanly-ended root startup with one fresh bounded attempt."""

        deadline = time.monotonic() + max(0.0, timeout)
        remaining = max(0.0, deadline - time.monotonic())
        if not self._root_startup_retry_lock.acquire(timeout=remaining):
            raise RuntimeError("a root startup retry is already in progress")
        try:
            # A concurrent caller may have completed the retry while this one
            # waited for the single retry owner.
            if self._status != "failed":
                return
            if (
                not self._root_startup_retryable
                or self._root_startup_attempts >= _ROOT_STARTUP_ATTEMPT_LIMIT
            ):
                raise RuntimeError(self._start_failure_reason())

            previous = self._thread
            if previous is not None:
                previous.join(timeout=max(0.0, deadline - time.monotonic()))
                if previous.is_alive():
                    raise RuntimeError(
                        "the failed root startup did not finish cleanup before the retry timeout"
                    )

            with self._lock:
                # Cleanup failure retains an owner that must not overlap a new
                # provider. Cancellation is likewise never a startup retry.
                if self._close_error is not None or self._cancellation.requested:
                    self._root_startup_retryable = False
                    raise RuntimeError(self._start_failure_reason())
                self._managed = None
                self._scheduler = None
                self._thread = None
                self._collector = None
                self._adapters.clear()
                self._initializing_adapters.clear()
                self._cursors.clear()
                self._pending_native.clear()
                self._pending_native_drops.clear()
                self._startup_cleanup_errors.clear()
                self._cleanup_base_status = None
                self._retryable_cleanup_steps.clear()
                self._nonretryable_cleanup_errors.clear()
                self._cleanup_retry_requested = False
                self._runtime_ready.clear()
                self._runtime_stood_up = False
                self._shutdown.clear()
                self._status = "ready"
                self._failure = ""
                self._root_startup_retryable = False
                self._thread = threading.Thread(
                    target=self._run, name="vnext-session", daemon=True
                )
                self._thread.start()

            if not self._runtime_ready.wait(max(0.0, deadline - time.monotonic())):
                raise RuntimeError("root startup retry did not respond before the call timeout")
            if self._status == "failed":
                raise RuntimeError(self._start_failure_reason())
        finally:
            self._root_startup_retry_lock.release()

    def _start_failure_reason(self) -> str:
        """Why this session is unusable, in terms its client can act on."""

        if not self._failure:
            return "the vNext session failed and recorded no reason"
        return f"the vNext session failed and cannot run tools: {self._failure}"

    def external_tools(self, *, timeout: float = 30) -> tuple[Mapping[str, Any], ...]:
        """The manager tool set as registered for this session's external root."""

        if not self.external_primary:
            raise ValueError("session primary is not external")
        self._runtime_ready.wait(timeout)
        adapter = self._adapters.get("external")
        if adapter is None or not adapter.wait_bound(timeout):
            raise RuntimeError("external primary is not bound to a control tree")
        return adapter.tools

    def restore(self, request: SessionRestoreRequest) -> Mapping[str, Any]:
        """Rebuild a quiescent control tree after a host restart.

        This deliberately performs no provider operation.  It preserves durable
        agent IDs and lineage so history and map replay stay coherent, while
        leaving provider controls unavailable until an adapter can re-attest a
        live thread.  A root that was mid-turn loses that turn and comes back
        ready; a running, waiting, or blocked child is not a quiescent tree
        and must use the provider reconciliation path instead.
        """

        if request.start != self.request:
            raise ValueError("restored session identity does not match this runtime")
        agents = [dict(item) for item in request.agents]
        dropped_turn_id = None
        terminal = {status.value for status in TERMINAL_STATUSES}
        saved_root = next((item for item in agents if item.get("agent_id") == self.root.agent_id), None)
        root_status = saved_root.get("status") if saved_root is not None else None
        root_terminal = root_status in terminal
        session_active = request.status not in {"ready", "idle", "completed", "cancelled", "failed"}
        # Decide from the root record.  The session row is persisted in a
        # separate transaction, so a crash can leave session=idle beside a
        # mid-turn root, or session=running beside a root already cancelled.
        active = saved_root is not None and not root_terminal and (
            saved_root.get("turn_id") is not None or root_status == "running" or session_active
        )
        if active or (session_active and root_terminal):
            # Any child still at work needs the provider reconciliation path.
            for item in agents:
                if item is saved_root:
                    continue
                if item.get("turn_id") is not None or item.get("status") not in terminal:
                    raise ValueError("active session requires provider reconciliation")
        if active:
            # Only a root that died mid-turn is recoverable here: its provider
            # process is gone, so the turn is dropped and the root comes back
            # ready on the same thread.  A terminal root is never revived.
            dropped_turn_id = saved_root.get("turn_id")
            saved_root["turn_id"] = None
            saved_root["status"] = "ready"
        self.control.restore_terminal_tree(self.root.session_id, agents)
        self._restored_agents = {str(item["agent_id"]): item for item in agents}
        self.root = self.control.sessions[self.root.session_id].agents[self.root.agent_id]
        for agent in self.control.sessions[self.root.session_id].agents.values():
            saved = self._restored_agents[agent.agent_id]
            if isinstance(saved.get("usage"), Mapping):
                agent.usage = dict(saved["usage"])
            if isinstance(saved.get("result"), Mapping):
                agent.result = dict(saved["result"])
        if active:
            self._status = "idle"
        elif session_active and root_terminal:
            # The root's own terminal status outranks a stale running row.
            self._status = str(root_status)
        else:
            self._status = request.status
        restoration: dict[str, Any] = {
            "state": "quiescent-tree-restored",
            "cursor": request.cursor,
            "provider_controls": "unavailable-pending-attestation",
        }
        if active:
            restoration["state"] = "interrupted-turn-restored"
            restoration["dropped_turn_id"] = dropped_turn_id
        views = []
        for agent in self.control.sessions[self.root.session_id].agents.values():
            view = self._agent_view(agent)
            view["capabilities"] = self._restored_agent_capabilities(agent)
            views.append(view)
        return {
            "status": self._status,
            "primary_agent_id": self.root.agent_id,
            "agents": views,
            "capabilities": self._restored_session_capabilities(),
            "restoration": restoration,
        }

    @staticmethod
    def _restored_session_capabilities() -> dict[str, str]:
        return {
            "prompt": "unavailable",
            "steer": "unavailable",
            "interrupt": "unavailable",
            "delegate": "unavailable",
            "fork": "unavailable",
            "tools": "unavailable",
            "skills": "unavailable",
            "settings": "unavailable",
            "native_terminal": "unavailable",
            "terminal_attach": "unavailable",
            "peer_message": "unavailable",
            "history": "available",
            "usage": "available",
        }

    def resume_primary(self) -> Mapping[str, Any]:
        """Open the saved root for a later prompt; never replay old work."""
        if self._shutdown.is_set() or self._cancellation.requested:
            raise ValueError("session runtime is closed")
        if self._primary_resumed:
            return self._resumed_snapshot()
        if self._shutdown.is_set() or self._thread is not None or not self._restored_agents:
            raise ValueError("primary resume requires a restored quiescent runtime")
        if self._status not in {"idle", "completed", "ready"}:
            raise ValueError("session is not resumable")
        if self.root.status not in {AgentStatus.READY, AgentStatus.COMPLETED}:
            raise ValueError("primary is not resumable")
        if any(agent.active_turn_id is not None or (
                agent.agent_id != self.root.agent_id and agent.status not in TERMINAL_STATUSES
        ) for agent in self.control.sessions[self.root.session_id].agents.values()):
            raise ValueError("nonterminal children require provider reconciliation")
        if self.registry.cards[self.root.model_id].provider not in {"claude", "codex"}:
            raise ValueError("provider cannot attest a fresh primary tool registration")
        saved = self._restored_agents[self.root.agent_id]
        if (not self.root.thread_id or not isinstance(saved.get("native_session_id"), str)
                or not saved["native_session_id"]):
            raise ValueError("saved primary lacks exact runtime and provider identities")
        if self.root.effort != self.request.primary.get("effort", "high"):
            raise ValueError("saved primary effort differs from session selection")
        scheduler = self._prepare_scheduler()
        # A native chat's root record says so, which keeps the user's text
        # verbatim after a host restart too.
        scheduler.bind_resumed_primary(provider_session=saved["native_session_id"],
                                       raw_user_prompts=saved.get("raw_user_prompts") is True)
        self._primary_resumed = True
        return self._resumed_snapshot()

    def resume_native_primary(self, native_session_id: str) -> Mapping[str, Any]:
        """Make a native Claude Code session this fresh session's root.

        The native id is both the runtime thread and the provider session, as
        in ``ClaudeCodeAdapter.resume_thread``, so the bridge's ``resume`` op
        takes it with no prior reservation.  The host has already checked that
        the session was written in this workspace.
        """
        if not isinstance(native_session_id, str) or not native_session_id:
            raise ValueError("native session resume requires a session id")
        with self._lock:
            if (self._shutdown.is_set() or self._thread is not None
                    or self._restored_agents or self._primary_resumed):
                raise ValueError("native session resume requires a fresh runtime")
        if self.registry.cards[self.root.model_id].provider != "claude":
            raise ValueError("native session resume is only available for Claude")
        self.control.attach_runtime_thread(self.root.agent_id, native_session_id)
        self._restored_agents = {self.root.agent_id: {
            "agent_id": self.root.agent_id, "native_session_id": native_session_id,
            "raw_user_prompts": True}}
        scheduler = self._prepare_scheduler()
        # The chat is the user's: each prompt lands in its transcript as their
        # line, so it carries their text alone.
        scheduler.bind_resumed_primary(provider_session=native_session_id, raw_user_prompts=True)
        self._primary_resumed = True
        self._status = "idle"
        return {**self._resumed_snapshot(),
                "restoration": {"state": "native-session-resumed",
                                "provider_controls": "primary-resume-configured"}}

    def _resumed_snapshot(self) -> Mapping[str, Any]:
        capabilities = self._restored_session_capabilities()
        for name in ("prompt", "interrupt", "delegate", "tools", "peer_message"):
            capabilities[name] = "available"
        return {**self.snapshot(), "capabilities": capabilities,
                "restoration": {"state": "primary-resumed", "provider_controls": "primary-resume-configured",
                                "active_task_reattachment": "unavailable"}}

    @staticmethod
    def _restored_agent_capabilities(_agent: AgentRecord) -> dict[str, str]:
        return {
            "prompt": "unavailable",
            "interrupt": "unavailable",
            "delegate": "unavailable",
            "peer_message": "unavailable",
            "native_terminal": "unavailable",
            "managed_terminal": "unavailable",
        }

    def _adapter(self, agent: AgentRecord) -> Any:
        card = self.registry.cards[agent.model_id]
        provider = str(card.provider)
        with self._adapter_lock:
            existing = self._adapters.get(provider)
            if existing is not None:
                return existing
            if provider in self._factories:
                adapter = self._factories[provider]()
            elif provider == "codex":
                adapter = self._codex_adapter(agent)
            elif provider == "commandcode":
                adapter = self._commandcode_adapter(agent)
            elif provider in {"claude", "zai"}:
                from .vnext_claude import ClaudeCodeAdapter, side_runtime_bridge_command
                side = self.request.config.get("claude_runtime")
                bridge_command = None
                if side:
                    # A side runtime from --update-runtimes: the bridge runs
                    # on that venv's Python, so it imports that SDK.
                    from .vnext_runtimes import verify_claude_runtime
                    verify_claude_runtime(side)
                    bridge_command = side_runtime_bridge_command(side)
                adapter = ClaudeCodeAdapter(workspace=str(self.workspace), provider=provider,
                    bridge_command=bridge_command,
                    permission_mode=str(self.request.config.get("claude_permission_mode", "default")),
                    tool_inputs=self.request.config.get("claude_tool_inputs") is True)
            elif provider == "external":
                from .vnext_external_primary import ExternalPrimaryAdapter
                adapter = ExternalPrimaryAdapter(
                    client=str(self.request.config.get("external_client", "claude-code")),
                    workspace=str(self.workspace),
                )
            else:
                raise ValueError(f"no runtime factory for {provider}")
            # Publish before initialize so a whole-session cancellation can
            # close an initializing provider instead of waiting blindly for
            # its startup timeout. The adapter lock still makes creation and
            # adoption single-writer operations.
            with self._lock:
                self._initializing_adapters[provider] = adapter
            try:
                initialized = adapter.initialize()
                side = self.request.config.get("claude_runtime")
                if provider in {"claude", "zai"} and side and isinstance(initialized, Mapping):
                    self.emit(RuntimeEvent("runtime.identity", {
                        "source": str(side.get("source") or "side"),
                        "harness": str(initialized.get("harness") or ""),
                        "sdk_version": str(initialized.get("sdk_version") or ""),
                        "python": str(side.get("python") or ""),
                        "cli_path": str(side.get("cli_path") or ""),
                    }, agent.agent_id))
                if provider == "codex" and self._restored_agents:
                    adapter.request("config/mcpServer/reload", {}, timeout=30)
                if self._scheduler is not None:
                    adapter.native_approval_handler = self._scheduler.review_approval
                self._adapters[provider] = adapter
                self._bind_native_observer(adapter)
                return adapter
            except BaseException as exc:
                # An adapter can spawn its provider process before initialize
                # raises. It is not in _adapters yet, so final cleanup cannot
                # discover it after this failure unless this path owns it.
                self._emit_provider_error(provider, agent, adapter, exc)
                self._close_failed_startup_adapter(adapter)
                if provider == "codex":
                    for close in (self._close_remote_codex_server, self._close_codex_mcp_relay,
                      self._close_commandcode_bridge):
                        try:
                            close()
                        except Exception as exc:
                            self._startup_cleanup_errors.append(exc)
                if provider == "commandcode":
                    try:
                        self._close_commandcode_bridge()
                    except Exception as exc:
                        self._startup_cleanup_errors.append(exc)
                raise
            finally:
                with self._lock:
                    self._initializing_adapters.pop(provider, None)

    def _emit_provider_error(
        self, provider: str, agent: AgentRecord, adapter: Any, exc: BaseException
    ) -> None:
        """Say why a provider would not start, in enough detail to act on.

        A session.error carried one sentence, and when that sentence was wrong
        -- an ImportError reported as a missing package that was installed --
        there was nothing else on disk to check it against. The traceback and
        the bridge's own stderr are the two things that would have named the
        cause, so they go in the run record beside the workspace. That file is
        local to this machine and this workspace; nothing here travels.
        """

        try:
            # Credential-adjacent stderr is never persisted in the session
            # run log; the constant says why, and the scheduler's
            # _report_binding_failure reads the same set.
            stderr = (
                ()
                if provider in CREDENTIAL_ADJACENT_STDERR_PROVIDERS
                else adapter.captured_stderr()
            )
        except Exception:
            stderr = ()
        payload = {
            "provider": provider,
            "model_id": agent.model_id,
            "role": agent.role.value,
            "error": str(exc),
            "kind": type(exc).__name__,
            "cause": f"{type(exc.__cause__).__name__}: {exc.__cause__}" if exc.__cause__ else None,
            "traceback": "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            ),
            "provider_stderr": list(stderr),
        }
        try:
            self.emit(RuntimeEvent("provider.error", payload, agent.agent_id))
        except Exception:
            # Observing a failure must never be what destroys it.
            pass

    def _codex_adapter(self, agent: AgentRecord) -> Any:
        from .live_runtime import resolve_session_runtime

        transport = self.request.config.get("codex_transport", "stdio")
        if self.request.config.get("codex_native_max_depth") is not None and transport != "websocket":
            raise ValueError("codex_native_max_depth requires the native WebSocket transport")
        codex_home = self._codex_home()
        if self._restored_agents:
            return self._resumed_codex_adapter(agent, transport=transport, codex_home=codex_home)
        if transport == "stdio":
            from .vnext_app_server import VNextAppServerAdapter
            executable, identity = resolve_session_runtime(self.request.config.get("codex_runtime"))
            self.emit(RuntimeEvent("runtime.identity", identity, agent.agent_id))
            return VNextAppServerAdapter(
                codex_executable=executable,
                codex_home=codex_home,
                workspace=self.workspace, client_name="vnext_host",
            )
        if transport != "websocket":
            raise ValueError("config.codex_transport must be 'stdio' or 'websocket'")
        from .vnext_remote_codex import RemoteCodexServer

        native_participation = self.request.config.get("codex_native_child_participation") is True
        factory = self._remote_server_factory or RemoteCodexServer
        server = factory(
            workspace=self.workspace,
            codex_home=codex_home,
            bind_host="127.0.0.1",
            require_auth=False,
            native_child_participation=native_participation,
            native_max_depth=self.request.config.get("codex_native_max_depth"),
            runtime_resolver=lambda: resolve_session_runtime(self.request.config.get("codex_runtime")),
        )
        with self._lock:
            self._remote_codex_server = server
        relay = None
        captured_adapter: Any | None = None

        def dispatch(tool: str, arguments: Mapping[str, Any], metadata: Mapping[str, Any]) -> ToolCallResult | Mapping[str, Any]:
            adapter = captured_adapter
            if adapter is None:
                raise RuntimeError("native Codex MCP relay dispatcher is not bound")
            handler = getattr(adapter, "dispatch_mcp_call", None)
            if not callable(handler):
                raise RuntimeError("native Codex adapter does not support relay dispatch")
            return handler(tool=tool, arguments=arguments, metadata=metadata)

        try:
            identity = server.start()
            if not isinstance(identity, Mapping):
                raise RuntimeError("remote Codex server returned malformed runtime identity")
            self.emit(RuntimeEvent("runtime.identity", dict(identity), agent.agent_id))
            overrides: Mapping[str, object] | None = None
            if native_participation:
                from .vnext_codex_mcp import CodexMcpRelay

                relay = CodexMcpRelay(tools=VNextScheduler.manager_tools(), dispatcher=dispatch,
                                      server_name="vnext_relay")
                with self._lock:
                    self._codex_mcp_relay = relay
                relay.start()
                overrides = relay.config_overrides("vnext_relay")
            if overrides is None:
                captured_adapter = server.connect(client_name="vnext_host")
            else:
                captured_adapter = server.connect(
                    client_name="vnext_host", mcp_config_overrides=overrides
                )
                captured_adapter.configure_manager_relay("vnext_relay")
            return captured_adapter
        except BaseException:
            # The shared server owns its adapter process. Close it before the
            # loopback relay, but make every owner attempt cleanup and retain a
            # failed one for the normal retry path.
            failures: list[Exception] = []
            for close in (self._close_remote_codex_server, self._close_codex_mcp_relay,
                      self._close_commandcode_bridge):
                try:
                    close()
                except Exception as exc:
                    failures.append(exc)
            if failures:
                self._startup_cleanup_errors.extend(failures)
            raise

    def _commandcode_adapter(self, agent: AgentRecord) -> Any:
        """Run the Codex app-server against a reseller's catalog.

        The harness is the one the "codex" provider already uses.  What changes
        is where its turns go: a loopback bridge holds the account key, rewrites
        each request into the shape the endpoint accepts, and hands Codex a
        throwaway bearer through the thread config map.

        The Codex home is private and nearly empty on purpose.  A user's own
        home carries MCP servers and plugins, and Codex would both start those
        processes and describe them to a provider that has no use for either.
        """

        from .live_runtime import resolve_session_runtime
        from .vnext_app_server import VNextAppServerAdapter
        from .vnext_commandcode import CommandCodeBridge
        from .vnext_provider_config import load_commandcode_provider

        if self.request.config.get("codex_transport", "stdio") != "stdio":
            raise ValueError("the Command Code provider runs over the stdio transport")
        credential = load_commandcode_provider()
        if credential is None:
            raise ValueError(
                "Command Code is unconfigured: add providers.commandcode.api_key to "
                "~/.vnext/providers.json"
            )
        bridge = CommandCodeBridge(api_key=credential.api_key)
        with self._lock:
            self._commandcode_bridge = bridge
        try:
            bridge.start()
            home = self._private_commandcode_home()
            executable, identity = resolve_session_runtime(self.request.config.get("codex_runtime"))
            self.emit(RuntimeEvent("runtime.identity", identity, agent.agent_id))
            return VNextAppServerAdapter(
                codex_executable=executable,
                codex_home=home,
                workspace=self.workspace,
                provider="commandcode",
                client_name="vnext_host",
                mcp_config_overrides=bridge.config_overrides("commandcode"),
            )
        except BaseException:
            try:
                self._close_commandcode_bridge()
            except Exception as exc:
                self._startup_cleanup_errors.append(exc)
            raise

    def _private_commandcode_home(self) -> Path:
        """Make a session-lifetime Codex home with nothing in it but settings."""

        if self._commandcode_home is not None:
            return self._commandcode_home
        base = Path(os.environ.get("LOCALAPPDATA") or tempfile.gettempdir())
        home = Path(tempfile.mkdtemp(prefix="vnext-commandcode-", dir=base)).resolve()
        (home / "config.toml").write_text(
            "check_for_update_on_startup = false\n"
            "include_apps_instructions = false\n"
            "include_collaboration_mode_instructions = false\n",
            encoding="utf-8",
        )
        self._commandcode_home = home
        return home

    def _close_commandcode_bridge(self) -> None:
        """Close the loopback bridge and discard the home it ran against."""

        with self._lock:
            bridge = self._commandcode_bridge
            home = self._commandcode_home
        try:
            if bridge is not None:
                bridge.close()
        finally:
            # The home holds three settings lines and nothing a retry needs.
            # It goes whether or not the bridge closed cleanly, because a
            # bridge that refuses to close would otherwise leave a directory
            # behind on every session that used this provider.
            if home is not None:
                shutil.rmtree(home, ignore_errors=True)
                with self._lock:
                    if self._commandcode_home is home:
                        self._commandcode_home = None
        # Only a close that returned clears the reference. The bridge is the
        # one object that can stop the server thread, so dropping it on
        # failure would leave the retry with nothing to call.
        with self._lock:
            if self._commandcode_bridge is bridge:
                self._commandcode_bridge = None

    def _resumed_codex_adapter(self, agent: AgentRecord, *, transport: str, codex_home: Path) -> Any:
        """Install the saved root's single relay before its provider starts.

        Per-thread resume overrides do not restore MCP on the pinned host.
        Process-local startup options avoid rewriting a user's saved config.
        """
        from .live_runtime import resolve_session_runtime
        from .vnext_app_server import VNextAppServerAdapter
        from .vnext_codex_mcp import CodexMcpRelay
        from .vnext_remote_codex import RemoteCodexServer

        if transport not in {"stdio", "websocket"}:
            raise ValueError("config.codex_transport must be 'stdio' or 'websocket'")
        captured_adapter = None

        def dispatch(tool: str, arguments: Mapping[str, Any], metadata: Mapping[str, Any]) -> ToolCallResult | Mapping[str, Any]:
            if captured_adapter is None:
                raise RuntimeError("resumed Codex MCP dispatcher is not bound")
            return captured_adapter.dispatch_mcp_call(tool=tool, arguments=arguments, metadata=metadata)

        relay = CodexMcpRelay(tools=VNextScheduler.manager_tools(), dispatcher=dispatch,
                              server_name="vnext_relay")
        self._codex_mcp_relay = relay
        try:
            relay.start()
            overrides = relay.config_overrides("vnext_relay")
            if transport == "stdio":
                executable, identity = resolve_session_runtime(self.request.config.get("codex_runtime"))
                captured_adapter = VNextAppServerAdapter(codex_executable=executable, codex_home=codex_home,
                    workspace=self.workspace, client_name="vnext_host", mcp_config_overrides=overrides,
                    mcp_startup_config_overrides=overrides)
            else:
                factory = self._remote_server_factory or RemoteCodexServer
                server = factory(workspace=self.workspace, codex_home=codex_home,
                    bind_host="127.0.0.1", require_auth=False,
                    native_child_participation=self.request.config.get("codex_native_child_participation") is True,
                    native_max_depth=self.request.config.get("codex_native_max_depth"),
                    mcp_startup_config_overrides=overrides,
                    runtime_resolver=lambda: resolve_session_runtime(self.request.config.get("codex_runtime")))
                self._remote_codex_server = server
                identity = server.start()
                captured_adapter = server.connect(client_name="vnext_host", mcp_config_overrides=overrides)
            captured_adapter.configure_resumed_root_relay("vnext_relay")
            self.emit(RuntimeEvent("runtime.identity", dict(identity), agent.agent_id))
            return captured_adapter
        except BaseException:
            if captured_adapter is not None:
                self._close_failed_startup_adapter(captured_adapter)
            for close in (self._close_remote_codex_server, self._close_codex_mcp_relay,
                      self._close_commandcode_bridge):
                try:
                    close()
                except Exception as exc:
                    self._startup_cleanup_errors.append(exc)
            raise

    def _codex_home(self) -> Path:
        """Resolve an explicitly configured isolated Codex home when present."""

        configured = self.request.config.get("codex_home")
        if configured is None:
            return Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
        if not isinstance(configured, str) or not configured:
            raise ValueError("config.codex_home must be an existing directory path")
        codex_home = Path(configured).resolve()
        if not codex_home.is_dir():
            raise ValueError("config.codex_home must be an existing directory path")
        return codex_home

    def _prepare_scheduler(self) -> VNextScheduler:
        if self._scheduler is not None:
            return self._scheduler
        if self._cancellation.requested:
            raise SchedulerCancelled("session cancelled during startup")
        try:
            adapter = self._adapter(self.root)
        except Exception:
            self._mark_root_provider_start_failure()
            raise
        root_adapter = _RootStartupAdapter(adapter, self._mark_root_provider_start_failure)
        effects = RuntimeEffectJournal(self.workspace)

        def choose(agent: AgentRecord) -> Any:
            if agent.agent_id == self.root.agent_id:
                chosen = root_adapter
            else:
                candidate = self._adapter(agent)
                # A child that resolves to the very adapter the root is using
                # must be handed the same object the root got.  Cleanup and
                # reconnect deduplicate adapters by ``id()``, and the raw
                # adapter and its start-failure wrapper are two ids for one
                # connection -- which closes that connection twice.
                chosen = root_adapter if candidate is adapter else candidate
            # The reader is chosen after the adapter, because a provider
            # registered through --provider has no built-in decoder and its
            # adapter is the thing that answers for its event shape.
            effects.bind_reader(
                agent.agent_id,
                runtime_effect_reader(
                    str(self.registry.cards[agent.model_id].provider), chosen
                ),
            )
            return chosen

        self._managed = VNextManagedSession(control=self.control, adapter=root_adapter,
            session_id=self.root.session_id, runtime_effects=effects, adapter_for_agent=choose)
        self._scheduler = VNextScheduler(managed=self._managed, root=self.root,
            cancellation=self._cancellation, hooks=SchedulerHooks(
                record_agent=self._record_agent, lifecycle=self._lifecycle,
                emit=lambda agent, state, code, message: self._record_agent(agent),
            ))
        return self._scheduler

    def _mark_root_provider_start_failure(self) -> None:
        with self._lock:
            self._root_provider_start_failed = True

    def _run(self) -> None:
        with self._lock:
            self._root_startup_attempts += 1
            self._root_provider_start_failed = False
            self._runtime_stood_up = False
        try:
            scheduler = self._prepare_scheduler()
            if self._cancellation.requested:
                raise SchedulerCancelled("session cancelled during startup")
            with self._lock:
                queued, self._queued = self._queued, []
            for initialized in list(self._adapters.values()):
                self._bind_native_observer(initialized)
            with self._lock:
                self._runtime_stood_up = True
            self._runtime_ready.set()
            for kind, agent_id, text, command_id in queued:
                if kind == "interrupt":
                    scheduler.interrupt_agent(agent_id)
                else:
                    self._deliver_prompt(agent_id, text, command_id)
            self._collector = threading.Thread(target=self._collect, name="vnext-native-events", daemon=True)
            self._collector.start()
            scheduler.run(keep_alive=True)
        except SchedulerCancelled:
            self._status = "completed" if self._clean_close else "cancelled"
            self._root_startup_retryable = False
        except Exception as exc:
            # Closing a provider which is still in initialize() normally makes
            # that call fail.  A requested whole-session cancellation is the
            # cause in that case, rather than a second startup failure.
            if self._cancellation.requested:
                self._status = "completed" if self._clean_close else "cancelled"
                self._root_startup_retryable = False
            else:
                with self._lock:
                    self._status = "failed"
                    self._failure = str(exc)
                    self._root_startup_retryable = (
                        self._root_provider_start_failed
                        and self._root_startup_attempts < _ROOT_STARTUP_ATTEMPT_LIMIT
                    )
                self.emit(RuntimeEvent("session.error", {
                    "error": str(exc),
                    "kind": type(exc).__name__,
                    "traceback": "".join(
                        traceback.format_exception(type(exc), exc, exc.__traceback__)
                    ),
                }))
        finally:
            self._runtime_ready.set()
            self._shutdown.set()
            terminal_failures = self._close_native_terminals()
            if self._collector is not None:
                self._collector.join(timeout=5)
                if self._collector.is_alive():
                    # The join has a deadline but had no verdict: a collector
                    # still running belongs to this attempt and still reads
                    # this attempt's adapter.  Starting a second attempt over
                    # it puts two collectors on one runtime, so the attempt is
                    # not retryable and the reason is recorded rather than
                    # inferred later from crossed event streams.
                    with self._lock:
                        self._root_startup_retryable = False
                        self._startup_cleanup_errors.append(
                            RuntimeError(
                                "runtime event collector did not stop within 5s; "
                                "this session cannot be restarted in place"
                            )
                        )
            failures: list[Exception] = [*self._startup_cleanup_errors, *terminal_failures]
            self._cleanup_base_status = self._status
            self._nonretryable_cleanup_errors = list(self._startup_cleanup_errors)
            self._retryable_cleanup_steps = {"native_terminals"} if terminal_failures else set()
            try:
                self._final_drain()
            except Exception as exc:
                failures.append(exc)
                self._nonretryable_cleanup_errors.append(exc)
            try:
                if self._managed is not None:
                    self._managed.close()
                else:
                    for adapter in list(self._adapters.values()):
                        try:
                            adapter.close()
                        except Exception as exc:
                            failures.append(exc)
                            self._retryable_cleanup_steps.add("adapters")
            except Exception as exc:
                failures.append(exc)
                self._retryable_cleanup_steps.add("managed")
            if self._remote_cleanup_pending:
                failures.append(RuntimeError("remote Codex server cleanup retry is required"))
                self._retryable_cleanup_steps.add("remote_codex_server")
            else:
                try:
                    self._close_remote_codex_server()
                except Exception as exc:
                    failures.append(exc)
                    self._retryable_cleanup_steps.add("remote_codex_server")
            if self._codex_mcp_relay_cleanup_pending:
                failures.append(RuntimeError("Codex MCP relay cleanup retry is required"))
                self._retryable_cleanup_steps.add("codex_mcp_relay")
            else:
                try:
                    self._close_codex_mcp_relay()
                except Exception as exc:
                    failures.append(exc)
                    self._retryable_cleanup_steps.add("codex_mcp_relay")
            try:
                self._close_commandcode_bridge()
            except Exception as exc:
                failures.append(exc)
                self._retryable_cleanup_steps.add("commandcode_bridge")
            if failures:
                self._record_cleanup_failure(failures)
            self.emit(RuntimeEvent("session.upsert", self._record_session_end()))

    def _record_cleanup_failure(
        self, failures: Sequence[Exception], *, retry_requested: bool = False
    ) -> None:
        cleanup_error = ExceptionGroup("runtime cleanup failed", list(failures))
        self._close_error = cleanup_error
        self._cleanup_retry_requested = retry_requested
        self._status = "failed"
        self._failure = str(cleanup_error)
        self._root_startup_retryable = False
        self.emit(RuntimeEvent("session.error", {"error": str(cleanup_error), "kind": "cleanup"}))

    def _retry_failed_cleanup(self) -> None:
        """Retry only cleanup owners retained after the first failed shutdown."""

        failures: list[Exception] = list(self._nonretryable_cleanup_errors)
        if "native_terminals" in self._retryable_cleanup_steps:
            terminal_failures = self._close_native_terminals()
            if terminal_failures:
                failures.extend(terminal_failures)
            else:
                self._retryable_cleanup_steps.discard("native_terminals")
        if "managed" in self._retryable_cleanup_steps:
            try:
                assert self._managed is not None
                self._managed.close()
            except Exception as exc:
                failures.append(exc)
            else:
                self._retryable_cleanup_steps.discard("managed")
        if "adapters" in self._retryable_cleanup_steps:
            adapter_failures: list[Exception] = []
            for adapter in list(self._adapters.values()):
                try:
                    adapter.close()
                except Exception as exc:
                    adapter_failures.append(exc)
            if adapter_failures:
                failures.extend(adapter_failures)
            else:
                self._retryable_cleanup_steps.discard("adapters")
        if "remote_codex_server" in self._retryable_cleanup_steps:
            try:
                self._close_remote_codex_server()
            except Exception as exc:
                failures.append(exc)
            else:
                self._retryable_cleanup_steps.discard("remote_codex_server")
        if "codex_mcp_relay" in self._retryable_cleanup_steps:
            try:
                self._close_codex_mcp_relay()
            except Exception as exc:
                failures.append(exc)
            else:
                self._retryable_cleanup_steps.discard("codex_mcp_relay")
        if "commandcode_bridge" in self._retryable_cleanup_steps:
            try:
                self._close_commandcode_bridge()
            except Exception as exc:
                failures.append(exc)
            else:
                self._retryable_cleanup_steps.discard("commandcode_bridge")
        if failures:
            self._record_cleanup_failure(failures, retry_requested=True)
            return
        self._close_error = None
        self._cleanup_retry_requested = False
        if self._cleanup_base_status is not None:
            self._status = self._cleanup_base_status
        self.emit(RuntimeEvent("session.upsert", {"status": self._status}))

    def prompt(self, agent_id: str, text: str, command_id: str) -> Mapping[str, Any]:
        if not text.strip():
            raise ValueError("prompt text is required")
        agent = self._require_agent(agent_id)
        if agent.agent_id == self.root.agent_id and self.external_primary:
            # Nothing reads this agent's inbox: its conversation belongs to the
            # client that owns it.  Accepting the prompt would queue mail no
            # model will ever see, which reads as delivery and is not.
            raise ValueError(
                "this session's primary is an external client; speak to it there"
            )
        if agent.agent_id != self.root.agent_id and agent.status in TERMINAL_STATUSES:
            raise ValueError("target agent is terminal")
        with self._lock:
            if self._shutdown.is_set():
                raise RuntimeError("session runtime is closed")
            if self._restored_agents and not self._primary_resumed:
                raise RuntimeError("restored primary requires provider attestation")
            if self._thread is None:
                if agent_id != self.root.agent_id:
                    raise ValueError("only the primary can receive the first session prompt")
                self._record_content(agent_id, None, "user", [{"type": "text", "text": text}], command_id=command_id)
                self.root.objective = text
                self.control.sessions[self.root.session_id].objective = text
                if self._primary_resumed:
                    self._deliver_prompt(agent_id, text, command_id)
                self._status = "running"
                self._thread = threading.Thread(target=self._run, name="vnext-session", daemon=True)
                self._thread.start()
                return {"delivery": "accepted", "agent_id": agent_id}
            if self._scheduler is None:
                self._record_content(agent_id, None, "user", [{"type": "text", "text": text}], command_id=command_id)
                self._queued.append(("prompt", agent_id, text, command_id))
                return {"delivery": "queued", "agent_id": agent_id}
        result = self._deliver_prompt(agent_id, text, command_id)
        self._record_content(agent_id, None, "user", [{"type": "text", "text": text}], command_id=command_id)
        return result

    def _deliver_prompt(self, agent_id: str, text: str, command_id: str) -> Mapping[str, Any]:
        scheduler = self._require_scheduler()
        if agent_id == self.root.agent_id:
            result = scheduler.message(text)
        else:
            result = scheduler.message_user(agent_id, text)
        delivery_event = "message.delivered" if result.get("delivery") == "delivered-into-active-turn" else "message.accepted"
        self.emit(RuntimeEvent(delivery_event, dict(result), agent_id, command_id=command_id))
        return result

    def interrupt(self, agent_id: str, turn_id: str | None, command_id: str) -> Mapping[str, Any]:
        agent = self._require_agent(agent_id)
        if turn_id is not None and agent.active_turn_id != turn_id:
            raise ValueError("turn is no longer active")
        with self._lock:
            if self._scheduler is None:
                if self._thread is None or self._shutdown.is_set():
                    return {"status": "not-running", "agent_id": agent_id}
                self._queued.append(("interrupt", agent_id, "", command_id))
                return {"status": "queued", "agent_id": agent_id}
        result = self._require_scheduler().interrupt_agent(agent_id)
        return dict(result or {"interrupted": True, "agent_id": agent_id})

    def compact(self, agent_id: str, command_id: str) -> Mapping[str, Any]:
        self._require_agent(agent_id)
        return self._require_scheduler().compact_agent(agent_id)

    def cancel(self, agent_ids: Sequence[str] | None, command_id: str) -> Mapping[str, Any]:
        if agent_ids is None:
            self._status = "completed" if self._clean_close else "cancelled"
            self._cancellation.request()
            if self._scheduler is not None:
                self._scheduler.cancel()
            if self._thread is None:
                self._shutdown.set()
            self._close_initializing_adapters()
            return {"status": "cancellation-requested"}
        scheduler = self._require_scheduler()
        for agent_id in agent_ids:
            self._require_agent(agent_id)
        # A stop the provider refused leaves that agent's turn running under a
        # record that now reads cancelled.  The manager-facing path reports it
        # by failing the call and naming the agents in `uninterrupted`; this
        # path threw every result away and always answered
        # `cancellation-requested`, so a client read a clean stop while the
        # work carried on underneath.  `status` is unchanged, because the
        # control plane really did cancel the subtree, and the refusal travels
        # in the same two keys the manager path uses.
        uninterrupted: list[dict[str, Any]] = []
        for agent_id in agent_ids:
            outcome = scheduler.cancel_agent(agent_id)
            refused = outcome.get("uninterrupted") if isinstance(outcome, Mapping) else None
            if refused:
                uninterrupted.extend(dict(entry) for entry in refused)
        result: dict[str, Any] = {"status": "cancellation-requested", "agents": list(agent_ids)}
        if uninterrupted:
            result["success"] = False
            result["uninterrupted"] = uninterrupted
        return result

    def peer_message(self, from_agent_id: str, to_agent_id: str, text: str, command_id: str) -> Mapping[str, Any]:
        self._require_agent(from_agent_id)
        self._require_agent(to_agent_id)
        return self._require_scheduler().send_message(sender_id=from_agent_id, target_id=to_agent_id, text=text)

    def resolve_approval(self, approval_id: str, decision: str, command_id: str) -> Mapping[str, Any]:
        return self._require_scheduler().resolve_approval(approval_id, decision, resolver="user")

    def snapshot(self) -> Mapping[str, Any]:
        session = self.control.sessions[self.root.session_id]
        with session.lock:
            agents = [self._agent_view(agent) for agent in session.agents.values()]
        terminals = [{"terminal_id": key, "agent_id": value["agent_id"], "mode": "native",
                      "state": value["state"], "attached": value["attached"]}
                     for key, value in list(self._native_terminals.items())]
        return {"status": self._status, "primary_agent_id": self.root.agent_id, "agents": agents,
                "terminals": terminals}

    def _agent_view(self, agent: AgentRecord) -> dict[str, Any]:
        card = self.registry.cards[agent.model_id]
        saved = self._restored_agents.get(agent.agent_id, {})
        native_session_id = saved.get("native_session_id")
        native_identity = saved.get("native_identity")
        adapter = self._adapters.get(str(card.provider))
        if agent.thread_id and adapter is not None:
            attestation = adapter.thread_identity_attestation(agent.thread_id)
            if attestation.get("bound") is True:
                native_session_id = attestation.get("provider_session")
                # Preserve provider-issued lineage without leaking arbitrary
                # adapter state or mistaking a runtime key for a session UUID.
                native_identity = {
                    key: attestation[key]
                    for key in ("origin", "native_agent_id", "native_task_id",
                                "parent_runtime_thread_id", "parent_native_turn_id",
                                "parent_tool_use_id", "parent_local_turn_reference",
                                "parent_turn_source", "parent_native_agent_id")
                    if isinstance(attestation.get(key), str) and attestation[key]
                } or None
        native_terminal = "available" if any(value["agent_id"] == agent.agent_id and value["state"] == "running"
                                            for value in list(self._native_terminals.values())) else "unavailable"
        capabilities = {
            "prompt": "available",
            "interrupt": "available",
            "delegate": "available",
            "peer_message": "available",
            "native_terminal": native_terminal,
            "managed_terminal": "available",
        }
        # An adopted provider child is not a regular scheduler-started agent.
        # Its adapter must attest an inbox and interrupt path before the
        # scheduler permits either operation, and it registers the manager
        # handler only when the child can actually use the coordination tools.
        # Report those observed limits instead of the generic defaults.
        scheduler = self._scheduler
        managed = self._managed
        if scheduler is not None:
            with scheduler._lock:
                delivery_contract = scheduler._native_child_delivery_contracts.get(agent.agent_id)
            if delivery_contract is not None:
                has_inbox = delivery_contract.get("context_messages") == "available"
                capabilities["prompt"] = "available" if has_inbox else "unavailable"
                capabilities["peer_message"] = "available" if has_inbox else "unavailable"
                capabilities["interrupt"] = (
                    "available"
                    if delivery_contract.get("interrupt") == "available"
                    else "unavailable"
                )
                handler = None
                if managed is not None:
                    with managed._lock:
                        handler = managed._handlers.get(agent.agent_id)
                capabilities["delegate"] = "available" if handler is not None else "unavailable"
        if saved and not (self._primary_resumed and agent.agent_id == self.root.agent_id):
            capabilities = self._restored_agent_capabilities(agent)
        view = {"agent_id": agent.agent_id, "parent_agent_id": agent.parent_agent_id,
            "role": agent.role.value, "model": agent.model_id, "provider": card.provider,
            "harness": card.harness, "status": agent.status.value, "native_session_id": native_session_id,
            "runtime_thread_id": agent.thread_id,
            "native_identity": native_identity,
            "effort": agent.effort, "turn_id": agent.active_turn_id,
            "approvals": agent.approvals,
            "usage": dict(agent.usage) or saved.get("usage"),
            "result": dict(agent.result) or saved.get("result", {}),
            "capabilities": capabilities}
        if saved.get("raw_user_prompts") is True:
            # Persisted on the root's record, so a restored runtime sees it.
            view["raw_user_prompts"] = True
        # Why an agent stopped is the one thing a blocked roster row has to
        # say.  The field was set on the record and dropped here, so a run
        # journal held 33 blocked workers and not one reason.  The key
        # appears only when there is a sentence, leaving every record for an
        # agent that was never blocked byte-identical to before.
        if isinstance(agent.blocker, str) and agent.blocker:
            view["blocker"] = agent.blocker
        view.update(identity_view(agent.model_identity))
        if (
            self._clean_close
            and agent.agent_id == self.root.agent_id
            and agent.status is AgentStatus.CANCELLED
        ):
            # Same reason as the terminal record: the roster row for the root
            # would otherwise contradict it.
            view["status"] = "completed"
        return view

    def _record_agent(self, agent: AgentRecord) -> None:
        self.emit(RuntimeEvent("agent.upsert", self._agent_view(agent), agent.agent_id))

    def _lifecycle(self, kind: str, agent: AgentRecord, data: Mapping[str, Any]) -> None:
        if kind == "agent_terminal" and self._clean_close and agent.agent_id == self.root.agent_id:
            # The root was cancelled to release it and its workers had all
            # finished, so the one line a reader looks at says the run ended
            # and who ended it.
            data = {**data, "status": "completed", "ended_by": "server close"}
        self._record_agent(agent)
        aliases = {"turn_started": "turn.started", "turn_completed": "turn.completed",
            "approval_requested": "approval.requested", "approval_resolved": "approval.resolved"}
        supplied_turn = data.get("turn_id")
        turn_id = supplied_turn if isinstance(supplied_turn, str) and supplied_turn else agent.active_turn_id
        self.emit(RuntimeEvent(aliases.get(kind, kind.replace("_", ".")), dict(data),
            agent.agent_id, turn_id))
        if (
            self._scheduler is not None
            and (kind == "agent_terminal" or (
                kind == "command_acknowledged"
                and data.get("command") == "cancel_agent"
                and data.get("status") == "cancelled"
            ))
        ):
            self._scheduler.queue_terminal_release(agent.agent_id)
        if kind == "conversation_idle" and not self._clean_close:
            # A close already under way has decided how this session ends; an
            # idle notice behind it would only overwrite that word.
            self._status = "idle"
            self.emit(RuntimeEvent("session.upsert", {"status": "idle"}))
        elif kind == "turn_started":
            self._status = "running"
            self.emit(RuntimeEvent("session.upsert", {"status": "running"}))

    def _require_agent(self, agent_id: str) -> AgentRecord:
        try:
            return self.control.sessions[self.root.session_id].agents[agent_id]
        except KeyError as exc:
            raise ValueError("agent does not belong to this session") from exc

    def _require_scheduler(self) -> VNextScheduler:
        if self._scheduler is None:
            raise RuntimeError("session runtime is starting; command is not yet available")
        return self._scheduler

    def history(self, agent_id: str, after: int = 0) -> Sequence[Mapping[str, Any]]:
        self._require_agent(agent_id)
        with self._lock:
            return list(self._history.get(agent_id, []))[after:]

    def _record_content(self, agent_id: str, turn_id: str | None, role: str,
                        blocks: list[dict[str, Any]], *, command_id: str | None = None) -> None:
        entry = {"role": role, "blocks": blocks, "turn_id": turn_id}
        with self._lock:
            self._history.setdefault(agent_id, []).append(entry)
        self.emit(RuntimeEvent("content.final", entry, agent_id, turn_id, command_id))

    def _collect(self) -> None:
        while not self._shutdown.wait(0.1):
            self._drain()

    def _final_drain(self) -> None:
        """The last drain of a shutdown, after the bills have come in.

        Closing the adapters stops the only readers a provider's final usage
        message can arrive through, so a quit taken right after a worker settled
        recorded the run with null tokens and a null price, and the report
        printed $0.00 for a worker that had really cost $0.238.  The counts were
        never missing: nothing waited for them.
        """

        self._settle_pending_worker_usage()
        self._drain()

    def _settle_pending_worker_usage(self) -> None:
        """Give each adapter the chance to collect the usage it still owes.

        An adapter with nothing outstanding is asked nothing, so a quit with no
        settled worker behind it costs exactly what it cost before.  A wait that
        runs out is said out loud: a reason the report can show beats a null
        nobody can explain.
        """

        with self._lock:
            adapters = list(self._adapters.values())
        ends_at = time.monotonic() + _FINAL_USAGE_WAIT_SECONDS
        for adapter in adapters:
            pending = getattr(adapter, "pending_usage_releases", None)
            collect = getattr(adapter, "await_pending_usage_releases", None)
            if not callable(pending) or not callable(collect):
                continue
            provider = str(getattr(adapter, "provider", ""))
            try:
                outstanding = tuple(pending())
            except Exception as exc:
                self._record_usage_wait_failure(provider, (), exc)
                continue
            if not outstanding:
                continue
            try:
                received = dict(collect(max(0.0, ends_at - time.monotonic())) or {})
            except Exception as exc:
                # Anything the adapter did not convert into a bridge rejection
                # arrives here as itself: a broken pipe, an EOF, an OSError from
                # outside the send path.  Each reservation still gets a reason
                # naming that cause, because a row with nothing in it reads as a
                # worker that cost nothing.  KeyboardInterrupt and CancelledError
                # are not Exception and end the close.
                self._record_usage_wait_failure(provider, outstanding, exc)
                continue
            reasons = self._adapter_usage_failures(adapter)
            unanswered = (
                *outstanding,
                *(thread_id for thread_id in received if thread_id not in outstanding),
            )
            for thread_id in unanswered:
                if received.get(thread_id):
                    continue
                self._emit_usage_unavailable(
                    provider, thread_id,
                    reasons.get(str(thread_id)) or _USAGE_GRACE_EXPIRED,
                )

    @staticmethod
    def _adapter_usage_failures(adapter: Any) -> dict[str, str]:
        """The named causes an adapter kept, when it keeps any."""

        failures = getattr(adapter, "usage_release_failures", None)
        if not callable(failures):
            return {}
        try:
            return {str(key): str(value) for key, value in dict(failures() or {}).items()}
        except Exception:
            return {}

    def _record_usage_wait_failure(
        self, provider: str, threads: tuple[str, ...], exc: BaseException,
    ) -> None:
        """Say what broke once, then give every waiting row the same cause."""

        self.emit(RuntimeEvent("telemetry.error", {
            "provider": provider,
            "step": "final_usage_wait",
            "error": str(exc),
        }))
        reason = (
            "the wait for this worker's final usage failed before the close grace "
            f"ran out: {type(exc).__name__}: {exc}"
        )
        for thread_id in threads:
            self._emit_usage_unavailable(provider, thread_id, reason)

    def _emit_usage_unavailable(self, provider: str, thread_id: object, reason: str) -> None:
        agent_id = (
            self._managed.agent_for_thread(str(thread_id))
            if self._managed is not None else None
        )
        self.emit(RuntimeEvent("usage.unavailable", {
            "provider": provider,
            "runtime_thread_id": str(thread_id),
            "reason": reason,
        }, agent_id))

    def _drain(self) -> None:
        from .vnext_runtime_projection import next_native_cursor
        with self._lock:
            adapters = list(self._adapters.values())
        for adapter in adapters:
            key = id(adapter)
            cursor = self._cursors.get(key)
            try:
                events = list(adapter.events_since(cursor))
                if events:
                    self._cursors[key] = next_native_cursor(str(adapter.provider), cursor, events)
                fresh = [
                    (event, self._native_event_cursor(adapter, cursor, event, offset))
                    for offset, event in enumerate(events)
                ]
            except Exception as exc:
                self.emit(RuntimeEvent("telemetry.error", {
                    "provider": str(adapter.provider),
                    "error": f"{type(exc).__name__}: {exc}",
                }))
                continue
            pending = self._pending_native.pop(key, []) + fresh
            unresolved: list[tuple[Mapping[str, Any], Any]] = []
            dropped = 0
            for event, event_cursor in pending:
                # One event that cannot be projected costs that event alone.
                # An exception here used to end the telemetry thread, and every
                # later usage and cost record of the session was lost with it.
                try:
                    dropped += self._route_native_event(adapter, event, event_cursor, unresolved)
                except Exception as exc:
                    self.emit(RuntimeEvent("telemetry.error", {
                        "provider": str(adapter.provider),
                        "error": f"{type(exc).__name__}: {exc}",
                    }))
            if unresolved:
                self._pending_native[key] = unresolved
            if dropped:
                total = self._pending_native_drops.get(key, 0) + dropped
                self._pending_native_drops[key] = total
                self.emit(
                    RuntimeEvent(
                        "telemetry.native_event_dropped",
                        {
                            "provider": str(adapter.provider),
                            "reason": "unattributed-capacity",
                            "dropped": dropped,
                            "total_dropped": total,
                            "retained": len(unresolved),
                        },
                    )
                )

    def _route_native_event(
        self,
        adapter: Any,
        event: Mapping[str, Any],
        event_cursor: Any,
        unresolved: list[tuple[Mapping[str, Any], Any]],
    ) -> int:
        """Publish one native event, or hold it; return 1 if it was dropped."""

        from .vnext_runtime_projection import accumulate_cost, project_native_event
        params = event.get("params", event)
        native = self._native_thread_id(params)
        agent_id = self._managed.agent_for_thread(str(native)) if self._managed and native else None
        if agent_id is None:
            if native:
                if len(unresolved) < _PENDING_NATIVE_EVENT_LIMIT:
                    unresolved.append((event, event_cursor))
                else:
                    return 1
            return 0
        if params.get("name") == "unsolicited_turn" and self._scheduler is not None:
            # No vNext turn waits on a turn the provider started by itself,
            # so this drain is the only reader that sees how it ended.
            self._scheduler.observe_unsolicited_turn(agent_id, params)
        native_turn = self._native_turn_id(params)
        if self._is_native_turn_started(event, params) and native_turn and self._scheduler is not None:
            try:
                self._scheduler.adopt_native_turn(
                    agent_id=agent_id,
                    provider=str(adapter.provider),
                    thread_id=str(native),
                    turn_id=str(native_turn),
                    cursor=event_cursor,
                )
            except (ManagedSessionError, ProtocolError, SchedulerError) as exc:
                self.emit(RuntimeEvent(
                    "telemetry.native_turn_refused",
                    {"provider": str(adapter.provider), "reason": str(exc)},
                    agent_id,
                ))
                return 0
        control_turn: str | None = None
        if native_turn is not None and self._managed is not None:
            control_turn = self._managed.control_turn_for_native(
                agent_id=agent_id,
                provider=str(adapter.provider),
                native_turn_id=str(native_turn),
            )
            # An adapter can emit before start_turn finishes binding
            # its handle. Retain the raw event rather than publishing
            # a native id that cannot join the control lifecycle.
            if control_turn is None:
                if len(unresolved) < _PENDING_NATIVE_EVENT_LIMIT:
                    unresolved.append((event, event_cursor))
                else:
                    return 1
                return 0
        for projected in project_native_event(str(adapter.provider), agent_id, event):
            if control_turn is not None:
                payload = dict(projected.payload)
                if params.get("turn_source") == "terminal-hook":
                    payload["local_turn_reference"] = str(native_turn)
                    payload["turn_source"] = "terminal-hook"
                elif params.get("native_runtime_thread_id"):
                    payload["native_task_id"] = str(native_turn)
                    payload["turn_source"] = "native-task"
                else:
                    payload["native_turn_id"] = str(native_turn)
                # Host history persists content payloads separately,
                # so carry the canonical event turn there as well.
                if projected.type == "content.final":
                    payload["turn_id"] = control_turn
                projected = RuntimeEvent(
                    projected.type,
                    payload,
                    projected.agent_id,
                    control_turn,
                    projected.command_id,
                )
            if projected.type == "provider.error":
                ours = self._stop_vnext_asked_for(agent_id, projected.payload)
                if ours is not None:
                    projected = RuntimeEvent(
                        "runtime.note",
                        {**projected.payload, "note": _STREAM_CLOSED_AT_SHUTDOWN,
                         "reason": ours},
                        projected.agent_id, projected.turn_id, projected.command_id,
                    )
            if projected.type == "usage.updated":
                session = self.control.sessions[self.root.session_id]
                with session.lock:
                    record = session.agents[agent_id]
                    # The count carries in the usage dict itself, which is the
                    # only per-agent store that survives the wholesale replace
                    # below and a restore from the saved snapshot.
                    calls = record.usage.get("provider_calls")
                    calls = calls + 1 if isinstance(calls, int) and not isinstance(calls, bool) else 1
                    usage = dict(projected.payload)
                    usage["provider_calls"] = calls
                    # Whoever reported it, the stored block is the agent's
                    # running total, so two children can be added together.
                    carried = accumulate_cost(record.usage.get("cost_tokens"),
                                              usage.get("cost_tokens"))
                    if carried is not None:
                        usage["cost_tokens"] = carried
                    record.usage = usage
                    # The emitted event is what the journal and the host read,
                    # so the count and the running total travel on it too.
                    projected = RuntimeEvent(projected.type, dict(usage),
                                             projected.agent_id, projected.turn_id,
                                             projected.command_id)
            self.emit(projected)
        return 0


    def _stop_vnext_asked_for(self, agent_id: str, payload: Mapping[str, Any]) -> str | None:
        """Why this provider error is vNext's own stop and not a fault.

        Quitting the server interrupts a Claude worker's still-open SDK stream.
        The SDK answers a result message with is_error true and terminal_reason
        aborted_streaming, for a turn nobody was waiting on any more.  Filed as
        provider.error against the finished worker, that one record made
        vnext-report print FAILED for a session where every worker had done its
        job, and made --failures useless as a watcher.

        Two readings say the stop was ours: the agent had already completed and
        the error has the abort shape a closed stream produces, or the session
        is closing and cannot start another turn.  An error on a turn that was
        genuinely working stays a provider error, and so does a provider's
        refusal that carries an HTTP status or a refusal code, such as a 401
        or authentication_failed, whether the worker
        had completed or the session was closing: those are the kinds worth
        paging anyone about.  The usage that arrives beside it
        is untouched, so a quit a millisecond after a worker settles still
        produces a priced row.
        """

        session = self.control.sessions.get(self.root.session_id)
        if session is None:
            return None
        with session.lock:
            record = session.agents.get(agent_id)
            status = record.status if record is not None else None
            closing = bool(getattr(session, "closing", False))
        if status == AgentStatus.COMPLETED and closed_by_shutdown(dict(payload)):
            return "agent-already-completed"
        if closing and not provider_refused(dict(payload)):
            return "session-closing"
        return None

    @staticmethod
    def _native_thread_id(params: Mapping[str, Any]) -> Any:
        turn = params.get("turn")
        turn = turn if isinstance(turn, Mapping) else {}
        return (
            params.get("native_runtime_thread_id")
            or params.get("threadId")
            or params.get("reservation_id")
            or params.get("thread_id")
            or turn.get("threadId")
        )

    @staticmethod
    def _native_turn_id(params: Mapping[str, Any]) -> Any:
        turn = params.get("turn")
        turn = turn if isinstance(turn, Mapping) else {}
        return params.get("turnId") or params.get("turn_reference") or turn.get("id")

    @staticmethod
    def _is_native_turn_started(event: Mapping[str, Any], params: Mapping[str, Any]) -> bool:
        return (
            event.get("method") == "turn/started" and isinstance(params.get("turn"), Mapping)
        ) or (
            params.get("name") == "terminal_turn_started"
            and params.get("turn_source") == "terminal-hook"
        ) or (
            params.get("name") == "native_child"
            and params.get("tracking") == "attested"
            and params.get("status") == "running"
            and bool(params.get("native_runtime_thread_id"))
        )

    @staticmethod
    def _native_event_cursor(adapter: Any, cursor: Any, event: Mapping[str, Any], offset: int) -> Any:
        event_cursor = event.get("cursor")
        if event_cursor is not None:
            return event_cursor
        # Codex adapters use an append-only, zero-based event index. Other
        # providers must provide their own cursor; external adoption only
        # accepts an observed Codex turn at this boundary.
        if str(getattr(adapter, "provider", "")) == "codex" and isinstance(cursor, int):
            return cursor + offset
        if str(getattr(adapter, "provider", "")) == "codex" and cursor is None:
            return offset
        return None

    def _close_initializing_adapters(self) -> None:
        """Ask startup adapters to stop without waiting under a runtime lock."""

        with self._lock:
            adapters = list(self._initializing_adapters.values())
        for adapter in adapters:
            self._close_failed_startup_adapter(adapter)
        for close in (self._close_remote_codex_server, self._close_codex_mcp_relay,
                      self._close_commandcode_bridge):
            try:
                close()
            except Exception as exc:
                self.emit(RuntimeEvent("session.error", {"error": str(exc), "kind": "startup-cleanup"}))

    def _close_failed_startup_adapter(self, adapter: Any) -> None:
        try:
            adapter.close()
        except Exception as exc:
            with self._lock:
                self._startup_cleanup_errors.append(exc)
            self.emit(RuntimeEvent("session.error", {"error": str(exc), "kind": "startup-cleanup"}))

    def _close_remote_codex_server(self) -> None:
        """Close an opt-in WebSocket server after every attached adapter detaches."""

        with self._lock:
            server = self._remote_codex_server
        if server is None:
            return
        try:
            cleanup = server.close()
            if cleanup is not None:
                residual = getattr(cleanup, "residual_count", None)
                errors = getattr(cleanup, "errors", ())
                if residual not in {0, None} or errors:
                    raise RuntimeError(
                        "remote Codex server cleanup was incomplete: "
                        f"residual_count={residual}, errors={tuple(errors or ())}"
                    )
        except Exception:
            # Keep the server instance reachable for a bounded retry. Dropping
            # it here would discard the only owner that can prove cleanup.
            self._remote_cleanup_pending = True
            raise
        else:
            self._remote_cleanup_pending = False
            with self._lock:
                if self._remote_codex_server is server:
                    self._remote_codex_server = None

    def _close_codex_mcp_relay(self) -> None:
        """Close the native-only relay after its shared Codex server."""

        with self._lock:
            relay = self._codex_mcp_relay
            server_remains = self._remote_codex_server is not None or self._remote_cleanup_pending
        if relay is None:
            return
        if server_remains:
            # The server remains the active owner of its adapter connection.
            # Closing the relay first can make an in-flight native provider
            # call look like a transport failure, so retain both owners until
            # the server's explicit cleanup retry succeeds.
            self._codex_mcp_relay_cleanup_pending = True
            raise RuntimeError("Codex MCP relay cleanup waits for shared server cleanup")
        try:
            relay.close()
        except Exception:
            # Retain the sole relay owner for an explicit cleanup retry.
            self._codex_mcp_relay_cleanup_pending = True
            raise
        else:
            self._codex_mcp_relay_cleanup_pending = False
            with self._lock:
                if self._codex_mcp_relay is relay:
                    self._codex_mcp_relay = None

    def _bind_native_observer(self, adapter: Any) -> None:
        observer = getattr(adapter, "set_native_child_observer", None)
        if callable(observer) and self._managed is not None and self._scheduler is not None:
            observer(self._scheduler.adopt_native_child,
                     parent_agent_resolver=self._managed.agent_for_thread)

    def attach_terminal(self, agent_id: str, terminal_id: str, command_id: str,
                        *, mode: str = "managed") -> TerminalAttachment:
        self._require_agent(agent_id)
        if mode == "native":
            return self._attach_native_terminal(agent_id, terminal_id)
        if mode != "managed":
            raise ValueError("terminal mode must be managed or native")
        with self._lock:
            if terminal_id in self._native_terminals:
                raise ValueError("terminal id already belongs to a native terminal")
            self._terminals[terminal_id] = agent_id
        return TerminalAttachment(terminal_id, agent_id, True, mode="managed",
            detail="Terminal commands use this existing session.")

    def _attach_native_terminal(self, agent_id: str, terminal_id: str) -> TerminalAttachment:
        from .native_terminal_process import NativeTerminalProcess
        agent = self._require_agent(agent_id)
        provider = str(self.registry.cards[agent.model_id].provider)
        if not agent.thread_id:
            raise ValueError("native resume requires a saved provider session; send the first chat prompt before attaching")
        if provider == "codex" and self.request.config.get("codex_transport", "stdio") != "websocket":
            raise ValueError("native Codex terminals require config.codex_transport=websocket at session creation")
        with self._terminal_lock:
            if terminal_id in self._terminals:
                raise ValueError("terminal id already belongs to a managed terminal")
            existing = self._native_terminals.get(terminal_id)
            if existing:
                if existing["agent_id"] != agent_id:
                    raise ValueError("terminal id belongs to another agent")
                if existing["state"] == "running":
                    existing["attached"] = True
                    return TerminalAttachment(terminal_id, agent_id, True, "native", "Reattached to running native terminal.")
                raise ValueError("terminal has exited; use a new terminal id")
            if any(item["agent_id"] == agent_id and item.get("lease_held")
                   for item in self._native_terminals.values()):
                raise ValueError("agent already has a native terminal; reattach using its terminal id")
            with self._lock:
                if self._shutdown.is_set():
                    raise RuntimeError("session runtime is closed")
            if not self._runtime_ready.wait(30) or self._shutdown.is_set():
                raise RuntimeError("runtime could not initialize for native terminal")
            scheduler = self._require_scheduler()
            scheduler.acquire_native_control_lease(agent_id)
            adapter = self._adapter(agent)
            entry: dict[str, Any] = {"agent_id": agent_id, "state": "starting", "attached": True,
                                     "process": None, "adapter": adapter, "provider": provider,
                                     "lease_held": True}
            self._native_terminals[terminal_id] = entry
            released_sdk = False
            try:
                if provider == "codex":
                    spec = adapter.terminal_launch(agent.thread_id)
                elif provider == "claude":
                    relay_factory = getattr(adapter, "terminal_relay_handler", None)
                    if not callable(relay_factory):
                        raise RuntimeError("Claude native terminal semantic relay is not available")
                    spec = adapter.begin_native_terminal(agent.thread_id,
                        relay_handler=relay_factory(agent.thread_id),
                        lease_dir=self.workspace / ".scratch" / "native-terminal-leases")
                    released_sdk = True
                else:
                    raise ValueError("provider has no native terminal adapter")
                factory = self._terminal_process_factory or NativeTerminalProcess
                process = factory([spec.executable, *spec.arguments], cwd=str(self.workspace),
                                  environment=spec.environment,
                                  on_event=lambda value: self._terminal_event(terminal_id, value))
                entry["process"] = process
                if provider == "claude":
                    adapter.set_native_terminal_interrupt(agent.thread_id, lambda: process.write("\x03"))
                entry["state"] = "running"
                self.emit(RuntimeEvent("terminal.started", {"terminal_id": terminal_id, "mode": "native"}, agent_id))
                self._record_agent(agent)
                return TerminalAttachment(terminal_id, agent_id, True, "native",
                                          "Service owns this terminal. Detaching its view leaves work running.")
            except BaseException:
                entry["state"] = "failed"
                try:
                    if entry["process"] is not None:
                        cleanup = entry["process"].close()
                        if cleanup.residual_count or cleanup.errors:
                            raise RuntimeError("native terminal process cleanup failed")
                    if released_sdk:
                        adapter.resume_after_native_terminal(agent.thread_id, terminal_stopped=True)
                except Exception as resume_error:
                    entry["error"] = str(resume_error)
                    raise
                else:
                    scheduler.release_native_control_lease(agent_id)
                    entry["lease_held"] = False
                raise

    def _terminal_event(self, terminal_id: str, value: Mapping[str, Any]) -> None:
        entry = self._native_terminals.get(terminal_id)
        if entry is None:
            return
        kind = value.get("type")
        if kind == "output":
            self.emit(RuntimeEvent("terminal.output", {"terminal_id": terminal_id, "data": value.get("data", "")}, entry["agent_id"]))
        elif kind == "error":
            self.emit(RuntimeEvent("terminal.error", {"terminal_id": terminal_id, "error": value.get("error")}, entry["agent_id"]))
        elif kind == "exit":
            # Reader must finish before OwnedProcess cleanup joins it.
            threading.Thread(target=self._finish_native_terminal, args=(terminal_id, dict(value)), daemon=True).start()

    def _finish_native_terminal(self, terminal_id: str, outcome: Mapping[str, Any]) -> None:
        with self._terminal_lock:
            entry = self._native_terminals.get(terminal_id)
            if not entry or not entry.get("lease_held"):
                return
            entry["state"] = "resuming"
            try:
                if entry["process"] is not None:
                    cleanup = entry["process"].close()
                    if cleanup.residual_count or cleanup.errors:
                        raise RuntimeError("native terminal process cleanup failed")
                if entry["provider"] == "claude" and not self._shutdown.is_set():
                    agent = self._require_agent(entry["agent_id"])
                    entry["adapter"].resume_after_native_terminal(agent.thread_id, terminal_stopped=True)
                entry["state"] = "exited"
                entry.pop("error", None)
                if self._scheduler:
                    self._scheduler.release_native_control_lease(entry["agent_id"])
                entry["lease_held"] = False
                self.emit(RuntimeEvent("terminal.exited", {"terminal_id": terminal_id, **outcome}, entry["agent_id"]))
            except Exception as exc:
                entry["state"] = "failed"
                entry["error"] = str(exc)
                self.emit(RuntimeEvent("terminal.error", {"terminal_id": terminal_id, "error": str(exc)}, entry["agent_id"]))

    def terminal_command(self, terminal_id: str, command_type: str, payload: Mapping[str, Any],
                         command_id: str) -> Mapping[str, Any]:
        with self._terminal_lock:
            entry = self._native_terminals.get(terminal_id)
            if not entry or entry["state"] != "running":
                raise ValueError("native terminal is not running")
            process = entry["process"]
            if command_type == "terminal_input":
                process.write(payload.get("data"))
            elif command_type == "terminal_resize":
                process.resize(payload.get("columns"), payload.get("rows"))
            elif command_type == "terminal_stop":
                self._finish_native_terminal(terminal_id, {"explicit_stop": True})
                if entry["state"] == "failed":
                    raise RuntimeError(entry["error"])
            else:
                raise ValueError("unknown terminal command")
        return {"accepted": True, "terminal_id": terminal_id}

    def _close_native_terminals(self) -> list[Exception]:
        failures: list[Exception] = []
        for terminal_id in list(self._native_terminals):
            self._finish_native_terminal(terminal_id, {"session_closed": True})
            entry = self._native_terminals[terminal_id]
            if entry.get("lease_held"):
                failures.append(RuntimeError(entry.get("error", "native terminal cleanup failed")))
        return failures

    def detach_terminal(self, terminal_id: str, command_id: str) -> Mapping[str, Any]:
        with self._terminal_lock:
            native = self._native_terminals.get(terminal_id)
            if native:
                native["attached"] = False
        with self._lock:
            self._terminals.pop(terminal_id, None)
        return {"detached": True, "work_continues": True}

    def _live_worker_ids(self) -> list[str]:
        """Every agent other than this session's root that can still move."""

        session = self.control.sessions[self.root.session_id]
        with session.lock:
            return [
                agent.agent_id
                for agent in session.agents.values()
                if agent.agent_id != self.root.agent_id
                and agent.status not in TERMINAL_STATUSES
            ]

    def _close_is_clean(self) -> bool:
        """Whether this close ends a run that finished its work.

        A client closing its pipe cancels the root, because that is how a live
        conversation is released, and the run record said `cancelled` for a
        session whose every worker had completed.  Three reviewers read a good
        run as a cancellation.  The question that separates the two cases is
        whether anything was still working when the client went away.
        """

        if self._status in {"failed", "cancelled"} or self._close_error is not None:
            return False
        if not self.external_primary:
            # An the host session with a provider-backed root is parked rather
            # than ended: its close releases a quiescent controller and keeps
            # the graph for a fresh one, which reopens the session as `idle`.
            # Those sessions keep today's words exactly
            # (tests/test_host_primary_resume_runtime.py).
            return False
        # Nothing here owns an external root's turns, so there is no root work
        # to cancel: the client simply left.
        return not self._live_worker_ids()

    def _record_session_end(self) -> dict[str, Any]:
        """The session's last word, decided by the close rather than by timing.

        The scheduler's own last move can land after the close has already read
        the tree as finished: on a real stdio run the idle notification arrived
        between the two and the record ended `idle`.  A cleanup that failed
        still owns the ending, because that failure is the news.
        """

        if self._clean_close and self._status != "failed":
            self._status = "completed"
            return {"status": "completed", "ended_by": "server close"}
        return {"status": self._status}

    def close(self) -> None:
        if not self._shutdown.is_set():
            scheduler = self._scheduler
            if self._thread is None and not self._initializing_adapters:
                # No scheduler ever ran (including configured primary resume).
                # Closing these process owners is not a user cancellation.
                self._shutdown.set()
            else:
                # Decided before the drain, so the cancellation underneath can
                # be recorded as the ending it is.  A worker still live at this
                # point keeps today's behaviour: it and the session are
                # cancelled, because that is what happened to its work.
                if self._close_is_clean():
                    self._clean_close = True
                    self._status = "completed"
                if scheduler is None or not scheduler.request_idle_shutdown():
                    self.cancel(None, "runtime-close")
        if self._thread is not None:
            self._thread.join(timeout=30)
            if self._thread.is_alive():
                raise RuntimeError("session runtime did not drain after cancellation")
        elif self._managed is not None:
            self._managed.close()
            self._close_remote_codex_server()
            self._close_codex_mcp_relay()
            self._close_commandcode_bridge()
            # A host may have recorded cleanup_failed after a prior close
            # attempt. Restore the original idle status once ownership drains.
            self.emit(RuntimeEvent("session.upsert", {"status": self._status}))
        else:
            for adapter in list(self._adapters.values()):
                adapter.close()
            self._close_remote_codex_server()
            self._close_codex_mcp_relay()
            self._close_commandcode_bridge()
        if self._close_error is not None:
            if self._cleanup_retry_requested:
                self._retry_failed_cleanup()
            else:
                self._cleanup_retry_requested = True
            if self._close_error is None:
                return
            raise RuntimeError("session runtime cleanup failed") from self._close_error


def runtime_session_factory(request: SessionStartRequest, emit: Callable[[RuntimeEvent], None]) -> VNextRuntimeSession:
    return VNextRuntimeSession(request, emit)
