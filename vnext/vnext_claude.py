"""Strict owned-process boundary for the optional Claude leaf runtime.

This module deliberately does not translate provider events into Codex effects,
invent correlation values, or host a provider client in-process.  The bridge is
the only process that imports the optional SDK; its JSONL contract carries only
provider-echoed identities.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .process_supervisor import OwnedProcess, ProcessCleanup
from .vnext_claude_effort import CLAUDE_ACCEPTED_EFFORTS
from .vnext_model_identity import remember, remember_claude_table
from .vnext_provider_config import PROVIDER_CONFIG_PATH, ZAI_ENDPOINT
try:
    from .vnext_diagnostics import (
        FailureCategory,
        classify_exception,
        classify_message,
        diagnostics_enabled,
        drain_stderr,
        record_failure,
    )
except Exception:  # pragma: no cover - exercised by deliberate injection only
    # An observability channel must not be able to take down the thing it
    # observes.  `vnext_diagnostics` is imported here at module level, so a
    # defect in it — a syntax error, a bad import, a module-level raise — used
    # to become an ImportError for `vnext_claude` and for every module that
    # imports `vnext_claude` in turn.  Measured: breaking the diagnostics
    # module took the suite from 204 tests to 121, with five modules failing
    # to import.
    #
    # Falling back to inert stubs keeps the runtime behaving exactly as it does
    # with diagnostics disarmed: nothing is classified, nothing is recorded.
    # The failure is silent HERE by design, because the only channel that could
    # report it is the one that just failed.  The test-side import in
    # tests/test_vnext_phase5e_live_canary.py is deliberately NOT made
    # defensive, so the defect is still caught loudly somewhere.
    class _InertCategories:
        """Stands in for ``FailureCategory`` with stable per-name sentinels.

        Call sites compare categories with ``is`` and with ``in``, so each name
        must yield one identical object every time rather than a fresh value.
        """

        def __init__(self) -> None:
            self._names: dict[str, str] = {}

        def __getattr__(self, name: str) -> str:
            if name.startswith("__"):
                raise AttributeError(name)
            return self._names.setdefault(name, name.lower())

    FailureCategory = _InertCategories()  # type: ignore[assignment]

    def classify_exception(exc: object) -> object:  # type: ignore[misc]
        del exc
        return FailureCategory.UNCLASSIFIED

    def classify_message(message: object) -> object:  # type: ignore[misc]
        del message
        return FailureCategory.UNCLASSIFIED

    def diagnostics_enabled() -> bool:  # type: ignore[misc]
        return False

    def drain_stderr(buffer: object) -> object:  # type: ignore[misc]
        del buffer
        return FailureCategory.UNCLASSIFIED

    def record_failure(*args: object, **kwargs: object) -> None:  # type: ignore[misc]
        del args, kwargs
        return None

from .vnext_runtime import RuntimePolicyError, require_workspace_policy
from .vnext_claude_terminal import (
    ClaudeTerminalError,
    ClaudeTerminalLease,
    ClaudeTerminalLeaseManager,
    ClaudeTerminalLaunchSpec,
    ClaudeTerminalRelayServer,
    RelayHandler,
)
from .vnext_runtime_types import (
    DynamicToolHandler,
    NativeChildBinding,
    NativeChildObservation,
    NativeChildObserver,
    NativeParentAgentResolver,
    RuntimeCleanup,
    RuntimePosture,
    ToolCallContext,
    ToolCallResult,
    TurnHandle,
)

def _native_canonical_model(value: object) -> bool:
    """Whether a string is a full provider model id rather than a catalog alias."""

    return isinstance(value, str) and value.startswith("claude-")


def _native_model_agrees(configured: object, observed: object) -> bool:
    """Whether an observed child model agrees with this thread's launch model.

    The catalog may name ``opus`` while the provider answers ``claude-opus-5``.
    An alias is a launch request rather than an observation, so it admits the
    canonical id the provider reports.  A configured canonical id still has to
    match exactly.  The bridge holds the same rule for the same reason; it runs
    in its own subprocess and shares no module with this adapter.  Agreement
    does not name the child: ``_native_catalog_model`` does.
    """

    if not isinstance(observed, str) or not observed:
        return False
    if not isinstance(configured, str) or not configured:
        return False
    if observed == configured:
        return True
    return not _native_canonical_model(configured) and _native_canonical_model(observed)


def _native_catalog_model(configured: str, observed: str) -> str:
    """The catalog name a native child runs under.

    Agreement above lets an alias through for any canonical id, so it cannot
    also decide the name: under an ``opus`` thread a subagent that answers
    ``claude-sonnet-4`` would be filed and priced as Opus.  The family is the
    first alphabetic part of the canonical id (``claude-opus-5`` is ``opus``,
    ``claude-3-5-sonnet-20241022`` is ``sonnet``).  The child runs under the
    configured alias when the family is that alias, and under its own family
    otherwise; a family the catalog lacks is then refused, with a reason, by
    the control plane that owns the catalog.
    """

    if observed == configured or not _native_canonical_model(observed):
        return configured
    if _native_canonical_model(configured):
        return configured
    parts = observed[len("claude-"):].split("-")
    family = next((part for part in parts if part.isalpha()), observed)
    return configured if family == configured else family



class ClaudeRuntimeError(RuntimeError):
    """Raised when the bridge cannot give an attested lifecycle result."""


_DEFAULT_BRIDGE_MODULE = "vnext.vnext_claude_bridge"

# The neutral effects this boundary may forward to the approval reviewer.
# `review_approval` in vnext_scheduler branches on exactly these four, and
# `_TOOL_EFFECTS` in vnext_claude_bridge emits exactly these four, so the
# three lists have to agree.  A field run found "read" and "network"
# missing here: a granted worker's Read of a path outside its workspace and
# its WebSearch both reached the reviewer with no effect at all, so the
# standing grant was never read and no approval was ever recorded.
_NEUTRAL_APPROVAL_EFFECTS = frozenset({"modify", "execute", "read", "network"})

# What a worker is told when no reviewer answered.  Naming the manager here
# was wrong twice over: the manager had not refused, and it had not been
# asked, so the worker reported a decision that was never taken.
_NO_REVIEWER_DECLINE_REASON = (
    "vNext declined this approval: no approval reviewer is attached to this worker"
)
_UNNAMED_DECLINE_REASON = "vNext declined this approval and the reviewer named no reason"

# How often a close re-asks the bridge whether a deferred release has
# drained.  The gap it covers was measured at about 1.1 s, so this is fine
# enough to add nothing noticeable and coarse enough to stay cheap.
_PENDING_USAGE_POLL_SECONDS = 0.05


def _owned_package_import_prelude() -> str:
    """Return fixed launcher code that gives this checkout import precedence.

    The bridge deliberately runs with the caller workspace as its current
    directory.  ``python -m`` would therefore be able to resolve an unrelated
    ``vnext`` checkout from that directory or an editable environment.
    Insert the directory that contains this adapter's package before importing
    the fixed bridge module.  This is a constant local path, never a caller
    value, and custom bridge commands bypass it unchanged.
    """

    package_root = str(Path(__file__).resolve().parent.parent)
    return f"import sys; sys.path.insert(0, {package_root!r}); "


def side_runtime_bridge_command(selection: Mapping[str, Any]) -> tuple[str, str, str]:
    """The bridge command for a side runtime installed by --update-runtimes.

    The bridge runs on the side venv's Python so it imports that venv's SDK.
    The two environment values tell it which SDK version to expect and which
    Claude CLI to start.
    """

    from .vnext_runtimes import CLI_PATH_ENV, SDK_VERSION_ENV

    setup = f"import os; os.environ[{SDK_VERSION_ENV!r}] = {str(selection['sdk_version'])!r}; "
    if selection.get("cli_path"):
        setup += f"os.environ[{CLI_PATH_ENV!r}] = {str(selection['cli_path'])!r}; "
    _python, flag, script = _owned_package_module_command(_DEFAULT_BRIDGE_MODULE)
    return (str(selection["python"]), flag, setup + script)


def _owned_package_module_command(module: str) -> tuple[str, str, str]:
    """Build the explicit, source-pinned command for one internal module."""

    return (
        sys.executable,
        "-c",
        "import runpy; "
        + _owned_package_import_prelude()
        + f"runpy.run_module({module!r}, run_name='__main__')",
    )


TerminalToolInvoker = Callable[[str, str, Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]
TerminalEventSink = Callable[[str, Mapping[str, Any]], Mapping[str, Any] | None]
TerminalInterrupt = Callable[[], None]


def _native_session_id(value: object) -> bool:
    return isinstance(value, str) and bool(value) and value != "default"


def _fatal_stage(message: object) -> str | None:
    """Classify adapter fatal state without leaking its message to receipts."""

    if not isinstance(message, str):
        return None
    if message.startswith("Claude bridge emitted") or message.startswith("Claude bridge stdout"):
        return "bridge-protocol"
    if message.startswith("Claude native child"):
        return "native-child-validation"
    if message.startswith("Claude native identity"):
        return "native-identity-validation"
    if message.startswith("Claude tool call"):
        return "tool-call-validation"
    if message.startswith("Claude permission"):
        return "permission-validation"
    return "adapter-validation"


def _fatal_code(message: object) -> str | None:
    """Return a fixed code for fatal boundaries that need live diagnosis."""

    fixed = {
        "Claude native child parent agent is not exactly resolved": "native-child-parent-unresolved",
        "Claude native child observer returned an invalid binding": "native-child-binding-invalid",
        "Claude native child identity conflicts with prior observation": "native-child-identity-conflict",
        "Claude native child task conflicts with prior observation": "native-child-task-conflict",
        "Claude native child terminal state conflicts with prior evidence": "native-child-terminal-conflict",
        "Claude native child tool call lacks exact task correlation": "native-child-tool-correlation-invalid",
        "Claude native child tool call is not attested": "native-child-tool-unattested",
    }
    return fixed.get(message) if isinstance(message, str) else None


def _validated_tool_definitions(tools: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return caller tool definitions the bridge is allowed to host.

    A manager delegates by calling these, so a malformed definition is a
    control-plane defect rather than a model mistake: reject it here, before a
    session is constructed, instead of discovering it as a missing tool.
    """

    definitions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for definition in tools:
        if not isinstance(definition, Mapping):
            raise ClaudeRuntimeError("Claude manager tool registration includes a malformed definition")
        name = definition.get("name")
        description = definition.get("description")
        schema = definition.get("inputSchema")
        if (
            not isinstance(name, str)
            or not name
            or name in seen
            or not isinstance(description, str)
            or not description
            or not isinstance(schema, Mapping)
            or schema.get("type") != "object"
            or not isinstance(schema.get("properties"), Mapping)
        ):
            raise ClaudeRuntimeError("Claude manager tool registration includes a malformed definition")
        seen.add(name)
        definitions.append(dict(definition))
    return definitions


def _definition_digest(definitions: Sequence[Mapping[str, Any]]) -> str | None:
    """Hash a tool payload exactly as the Codex boundary already does."""

    if not definitions:
        return None
    return hashlib.sha256(
        json.dumps(
            [dict(value) for value in definitions],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


class ClaudeCodeAdapter:
    """Provider leaf backed only by an :class:`OwnedProcess` JSONL bridge."""

    provider = "claude"
    harness = "claude-agent-sdk"
    _VERSION = 1

    def __init__(
        self,
        *,
        workspace: str,
        bridge_command: Sequence[str] | None = None,
        native_approval_handler: Callable[[str, Mapping[str, Any]], Mapping[str, Any]] | None = None,
        request_timeout_seconds: float = 15.0,
        permission_mode: str = "default",
        provider: str = "claude",
    ) -> None:
        started = time.monotonic()
        resolved = Path(workspace).resolve()
        if not resolved.is_dir():
            # Decision 3: the offending path is a fact about the caller's
            # machine and must not reach a receipt.  Only the label leaves.
            record_failure(
                FailureCategory.WORKSPACE_UNAVAILABLE,
                phase="adapter.init",
                step="resolve_workspace",
                duration_ms=(time.monotonic() - started) * 1000.0,
            )
            # The raised message and the recorded label both stay path-free.
            # The cause carries the offending path so a live traceback is
            # still debuggable; `__cause__` is local to the process and is
            # never what gets persisted or reported.
            raise ClaudeRuntimeError("workspace does not exist") from FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), str(resolved)
            )
        self._workspace = resolved
        if provider not in {"claude", "zai"}:
            raise ClaudeRuntimeError("Claude adapter provider is not supported by vNext")
        self.provider = provider
        self._endpoint_attestation: dict[str, Any] = {}
        if permission_mode not in {"default", "acceptEdits"}:
            raise ClaudeRuntimeError("Claude adapter permission mode is not supported by vNext")
        self._permission_mode = permission_mode
        # A default launch is source-pinned before it imports the bridge.  The
        # workspace stays the bridge CWD for its normal workspace contract,
        # while an explicit command remains wholly caller-controlled.
        self._command = (
            tuple(bridge_command)
            if bridge_command
            else _owned_package_module_command(_DEFAULT_BRIDGE_MODULE)
        )
        self._native_approval_handler = native_approval_handler
        self._timeout = request_timeout_seconds
        # start_thread is not a control request: it launches a Claude process
        # and waits for it to connect back.  On a loaded machine with several
        # projects delegating at once that took longer than the 15s every other
        # request gets, and the timeout was read as a dead provider and killed
        # the whole session.  Scale rather than pin, so a test that shortens the
        # request timeout still shortens this one.
        self._start_thread_timeout = request_timeout_seconds * 6
        self._owned: OwnedProcess | None = None
        self._condition = threading.Condition(threading.RLock())
        self._next_request = 1
        self._pending_requests: set[int] = set()
        self._responses: dict[int, Mapping[str, Any]] = {}
        # A diagnostics request is explicitly read-only.  If its local
        # timeout fires while the bridge is busy, a later matching response is
        # valid but no longer useful; retain only a bounded ID tombstone so it
        # cannot be mistaken for a forged response and kill the provider.
        self._expired_read_only_requests: deque[int] = deque(maxlen=64)
        self._late_read_only_response_count = 0
        # Every other request gets the same tombstone, for the same reason.
        # A turn that outruns the scheduler's bound is given up on locally
        # while the provider is still working; its answer then arrives for an
        # ID this adapter no longer holds.  Measured on 2026-09-18: a GLM turn
        # hit the 600s turn timeout, the late reply was read as a forged
        # response, and one worker's overrun killed the bridge its siblings
        # and the whole session were using.  An ID this adapter issued itself
        # is not a forgery; only one it never issued is.
        self._expired_requests: deque[int] = deque(maxlen=64)
        self._late_response_count = 0
        # Why a reservation's final usage was given up on, per reservation.  A
        # close reads this so the outcome row names the cause instead of
        # carrying a null nobody can explain.
        self._usage_release_failures: dict[str, str] = {}
        self._events: deque[Mapping[str, Any]] = deque(maxlen=2_048)
        # A runtime thread is a local reservation until a provider message
        # attests a native session.  Never reuse a provider value as this key.
        self._threads: dict[str, dict[str, Any]] = {}
        self._tool_registrations: dict[str, Mapping[str, Any]] = {}
        # A manager binds caller tools; a Worker binds none.  Both maps stay
        # empty for a Worker, which is what keeps the leaf gate intact.
        self._tool_handlers: dict[str, DynamicToolHandler] = {}
        self._tool_definitions: dict[str, list[dict[str, Any]]] = {}
        self._initialized = False
        self._turns: dict[tuple[str, str], TurnHandle] = {}
        self._reader_threads: list[threading.Thread] = []
        self._stderr: deque[str] = deque(maxlen=32)
        self._fatal: str | None = None
        self._closing = False
        self._runtime_cleanup: RuntimeCleanup | None = None
        # Kept by the adapter because the terminal relay must outlive a
        # detached browser UI.  The runtime owns the actual PTY process.
        self._terminal_leases: dict[str, tuple[ClaudeTerminalRelayServer, ClaudeTerminalLeaseManager, ClaudeTerminalLease]] = {}
        # Claude's terminal hooks do not expose an SDK turn handle.  The
        # service assigns a per-lease reference at UserPromptSubmit and marks
        # it terminal at Stop.  It is deliberately labelled as terminal-hook
        # evidence instead of being presented as a provider-native turn ID.
        self._terminal_turns: dict[tuple[str, str], dict[str, Any]] = {}
        self._terminal_interrupts: dict[str, TerminalInterrupt] = {}
        # Native Agent tasks are children of an SDK parent session, not
        # independently resumable Claude sessions.  Keep their attested
        # runtime key separate from the parent reservation and retain only
        # lifecycle facts the bridge correlated through its enrollment ledger.
        self._native_child_observer: NativeChildObserver | None = None
        self._native_parent_agent_resolver: NativeParentAgentResolver | None = None
        self._native_child_observations: dict[str, NativeChildObservation] = {}
        self._native_child_bindings: dict[str, NativeChildBinding] = {}
        # Why the control plane last refused to adopt a native child, read
        # once by the event that carried that child into the record.
        self._native_child_refusal: str | None = None
        self._native_child_tasks: dict[str, dict[str, Any]] = {}
        # A settled native task can leave the parent's sole SDK reader
        # unwinding for a short interval. Cache only the negative local
        # readiness result, so scheduler wake polling never turns into bridge
        # JSONL traffic while a native child is still active.
        self._turn_readiness: dict[str, tuple[int, bool, float]] = {}
        self._started_monotonic = started

    @property
    def native_approval_handler(self) -> Callable[[str, Mapping[str, Any]], Mapping[str, Any]] | None:
        return self._native_approval_handler

    @native_approval_handler.setter
    def native_approval_handler(self, value: Callable[[str, Mapping[str, Any]], Mapping[str, Any]] | None) -> None:
        self._native_approval_handler = value

    def _workspace_inside(self, workspace: str | None) -> Path | None:
        """Resolve a requested root, or return None when it is not ours.

        A private Worker root is a directory beneath the workspace this adapter
        connected to, so an equality check would refuse the very partition it is
        meant to allow.  Containment is the real rule: this adapter may only
        serve a node the connected posture already covers.  A path outside it,
        or one that escapes through a link or a parent segment, still returns
        None -- `resolve` is what makes that true rather than the string.

        The refusal itself belongs to each caller, so every raise site keeps a
        literal message the diagnostic scan can classify.
        """

        if workspace is None:
            return self._workspace
        resolved = Path(workspace).resolve()
        if resolved != self._workspace and self._workspace not in resolved.parents:
            return None
        return resolved

    def initialize(self, *, timeout: float = 30) -> Mapping[str, Any]:
        del timeout
        self._start_owned_bridge()
        payload = {"workspace": str(self._workspace)}
        if self.provider != "claude":
            payload["provider"] = self.provider
        result = self._request("initialize", payload)
        common_attested = (
            result.get("provider") == self.provider
            and result.get("harness") == self.harness
            and result.get("tool_support") == {"accepted": True, "requires_empty": False}
            and result.get("credential_override_rejected") is True
        )
        if self.provider == "claude":
            if not common_attested:
                self._record_failure(
                    FailureCategory.INITIALIZE_ATTESTATION_MISMATCH, step="initialize"
                )
                raise ClaudeRuntimeError("bridge initialization did not attest Claude capabilities")
        else:
            endpoint = result.get("endpoint_attestation")
            expected = {
                "endpoint_owner": "z.ai",
                "endpoint": ZAI_ENDPOINT,
                "endpoint_kind": "named-non-anthropic",
                "credential_owner": "z.ai provider",
                "credential_kind": "provider-api-key",
                "anthropic_subscription_credential": False,
            }
            if not common_attested or endpoint != expected:
                self._record_failure(
                    FailureCategory.INITIALIZE_ATTESTATION_MISMATCH, step="initialize"
                )
                raise ClaudeRuntimeError("bridge initialization did not attest z.ai capabilities")
            self._endpoint_attestation = dict(expected)
        self._initialized = True
        return result

    def start_thread(
        self,
        *,
        model: str,
        developer_instructions: str,
        tools: Sequence[Mapping[str, Any]],
        tool_handler: DynamicToolHandler | None = None,
        requested_posture: RuntimePosture,
        workspace: str | None = None,
        effort: str = "high",
        permission_mode: str | None = None,
    ) -> tuple[str, Mapping[str, Any]]:
        # Order matters: the pairing rule is checked before the definitions are
        # read, so a leaf that was handed tools it has no handler for is
        # refused as a leaf rather than as a schema mistake.
        if bool(tools) is not (tool_handler is not None):
            # A manager without a handler could delegate into a void; a handler
            # without tools is a binding the model can never reach.  Both are
            # control-plane defects, so neither is allowed to start a session.
            raise ClaudeRuntimeError("Claude tool registration requires exactly one caller handler")
        definitions = _validated_tool_definitions(tools)
        if not isinstance(requested_posture, RuntimePosture):
            raise ClaudeRuntimeError("Claude thread lacks a neutral requested posture")
        if not self._initialized:
            raise ClaudeRuntimeError("Claude bridge must initialize before starting a thread")
        if effort not in CLAUDE_ACCEPTED_EFFORTS:
            raise ClaudeRuntimeError("Claude thread effort is not supported by the SDK")
        permission_mode = self._permission_mode if permission_mode is None else permission_mode
        if permission_mode not in {"default", "acceptEdits"}:
            raise ClaudeRuntimeError("Claude thread permission mode is not supported by vNext")
        posture = requested_posture.as_dict()
        if (
            posture.get("workspace_writes") is not True
            or posture.get("network") not in {"restricted", "approval_gated"}
            or posture.get("approvals_requested") is not True
            or not isinstance(posture.get("reviewer"), str)
            or not posture["reviewer"]
            or posture.get("environment_ready") is not True
        ):
            raise ClaudeRuntimeError("Claude leaf cannot satisfy the requested neutral posture")
        thread_workspace = self._workspace_inside(workspace)
        if thread_workspace is None:
            raise ClaudeRuntimeError("Claude thread workspace differs from owned workspace")
        reservation_id = f"claude-reservation-{uuid.uuid4().hex}"
        result = self._request(
            "start_thread",
            {
                "workspace": str(thread_workspace),
                "tools": definitions,
                "developer_instructions": developer_instructions,
                "requested_posture": posture,
                "model": model,
                "effort": effort,
                "permission_mode": permission_mode,
                "reservation_id": reservation_id,
            },
            timeout_seconds=self._start_thread_timeout,
        )
        policy = result.get("policy")
        connection_evidence = result.get("connection_evidence")
        registration = result.get("tool_registration")
        effort_evidence = result.get("effort")
        reservation_echo = result.get("reservation_echo")
        if (
            reservation_echo != reservation_id
            or not isinstance(policy, Mapping)
            or connection_evidence != {"connected": True, "server_info_received": True}
            or effort_evidence != {"requested": effort, "applied": "sdk-option"}
            or result.get("permission_mode") != permission_mode
        ):
            raise ClaudeRuntimeError("Claude bridge did not attest a connected local reservation")
        try:
            require_workspace_policy(policy, approvals_reviewer=str(posture["reviewer"]))
        except RuntimePolicyError as exc:
            raise ClaudeRuntimeError("Claude bridge returned an invalid resolved posture") from exc
        resolved_posture = policy.get("posture")
        if not isinstance(resolved_posture, Mapping) or resolved_posture.get("network") != "approval_gated":
            raise ClaudeRuntimeError("Claude bridge did not resolve approval-gated network access")
        if (
            not isinstance(registration, Mapping)
            or registration.get("acknowledged") is not True
            or registration.get("model_id") != model
            or registration.get("tool_count") != len(definitions)
            or registration.get("tool_names") != [str(value["name"]) for value in definitions]
            or registration.get("definition_sha256") != _definition_digest(definitions)
            or registration.get("handler_registered") is not (tool_handler is not None)
        ):
            # For a Worker every expectation above degrades to the historical
            # empty registration: zero tools, no names, no digest, no handler.
            raise ClaudeRuntimeError("bridge did not attest the requested tool registration")
        if tool_handler is not None:
            self._tool_handlers[reservation_id] = tool_handler
            self._tool_definitions[reservation_id] = definitions
        self._threads[reservation_id] = {
            "policy": dict(policy),
            "provider_session": None,
            "binding_phase": "reserved",
            "generation": 0,
            "model": model,
            "effort": effort,
            "permission_mode": permission_mode,
            "developer_instructions": developer_instructions,
            "workspace": str(thread_workspace),
            "released": False,
            "active_turn": None,
        }
        self._tool_registrations[reservation_id] = dict(registration)
        self._note_model_identity(reservation_id, model, result.get("model_identity"))
        return reservation_id, policy

    def _note_model_identity(self, thread_id: str, alias: str, reported: object) -> None:
        """Keep the bridge's exact-model answer beside the alias it resolved."""

        state = self._threads.get(thread_id)
        if not isinstance(state, dict):
            return
        reported = reported if isinstance(reported, Mapping) else {}
        previous = state.get("model_identity")
        previous = previous if isinstance(previous, Mapping) else {}
        identity: dict[str, Any] = {
            "model_exact": reported.get("model_exact"),
            "model_exact_source": reported.get("model_exact_source"),
        }
        history = [value for value in reported.get("model_exact_history") or () if isinstance(value, str)]
        for value in [*(previous.get("model_exact_history") or ()), previous.get("model_exact")]:
            if isinstance(value, str) and value and value != identity["model_exact"] and value not in history:
                history.append(value)
        if history:
            identity["model_exact_history"] = history
        for key in ("model_ran", "model_ran_first"):
            if isinstance(previous.get(key), str):
                identity[key] = previous[key]
        state["model_identity"] = identity
        remember_claude_table(self.provider, reported.get("models"), (alias,) if alias else ())
        if alias:
            remember(self.provider, alias, reported.get("model_exact"), reported.get("model_exact_source"))

    def _note_model_ran(self, thread_id: object, ran: object, first: object = None) -> None:
        """Record the model id the worker's own replies carried."""

        state = self._threads.get(thread_id) if isinstance(thread_id, str) else None
        if not isinstance(state, dict) or not isinstance(ran, str) or not ran:
            return
        identity = state.setdefault("model_identity", {})
        identity["model_ran"] = ran
        identity.setdefault("model_ran_first", first if isinstance(first, str) and first else ran)

    def model_identity(self, thread_id: str) -> Mapping[str, Any]:
        """The exact model this thread asked for and the one that answered."""

        state = self._threads.get(thread_id)
        identity = state.get("model_identity") if isinstance(state, Mapping) else None
        return dict(identity) if isinstance(identity, Mapping) else {}

    def start_turn(
        self,
        thread_id: str,
        prompt: str,
        *,
        model: str = "",
        effort: str = "",
        approvals_reviewer: str = "auto_review",
        workspace: str | None = None,
        turn_timeout: float | None = None,
    ) -> TurnHandle:
        if self._workspace_inside(workspace) is None:
            raise ClaudeRuntimeError("Claude turn workspace differs from owned workspace")
        if thread_id not in self._threads:
            raise ClaudeRuntimeError("unknown provider thread")
        state = self._threads[thread_id]
        if thread_id in self._terminal_leases:
            raise ClaudeRuntimeError("Claude session is owned by an active native terminal")
        if state.get("active_turn") is not None:
            raise ClaudeRuntimeError("Claude thread already has an active turn")
        configured_model = state.get("model")
        configured_effort = state.get("effort")
        if model and model != configured_model:
            raise ClaudeRuntimeError("Claude turn model drifted from the connected thread")
        if effort and effort != configured_effort:
            raise ClaudeRuntimeError("Claude turn effort drifted from the connected thread")
        posture = state["policy"].get("posture")
        if not isinstance(posture, Mapping) or posture.get("reviewer") != approvals_reviewer:
            raise ClaudeRuntimeError("Claude turn reviewer drifted from the connected thread posture")
        previous_generation = int(state["generation"])
        generation = previous_generation + 1
        local_turn_id = f"claude-turn-{uuid.uuid4().hex}"
        # Publish the generation before the request.  The bridge may emit an
        # SDK identity event as soon as it acknowledges the turn, and the
        # reader thread must be able to bind it without a stale-generation
        # race.  Restore it only when the request itself is rejected.
        state["generation"] = generation
        try:
            result = self._request(
                "start_turn",
                {
                    "reservation_id": thread_id,
                    "turn_reference": local_turn_id,
                    "generation": generation,
                    "prompt": prompt,
                    "effort": configured_effort,
                    "turn_timeout": turn_timeout,
                },
            )
        except BaseException:
            state["generation"] = previous_generation
            raise
        turn_id, cursor = result.get("turn_id"), result.get("cursor")
        if result.get("reservation_echo") not in {None, thread_id} or result.get("turn_echo") not in {None, local_turn_id}:
            raise ClaudeRuntimeError("Claude bridge did not correlate the local turn")
        # Old deterministic fixtures echo a provider turn ID.  It is not a
        # session identity and is retained only as an opaque local turn value.
        if not isinstance(turn_id, str) or not turn_id:
            turn_id = local_turn_id
        if not isinstance(cursor, int):
            cursor = None
        handle = TurnHandle(thread_id=thread_id, turn_id=turn_id, cursor=cursor)
        self._turns[(thread_id, turn_id)] = handle
        state["active_turn"] = turn_id
        return handle

    def release_terminal_thread(
        self,
        thread_id: str,
        *,
        status: str = "completed",
        timeout_seconds: float | None = None,
    ) -> None:
        """Park one finished managed reservation without closing its siblings."""

        state = self._threads.get(thread_id)
        if not isinstance(state, dict) or state.get("released") is True:
            return
        if thread_id in self._terminal_leases or thread_id in self._native_child_tasks:
            return
        result = self._request(
            "release_agent",
            {"reservation_id": thread_id, "status": status},
            timeout_seconds=timeout_seconds,
        )
        if result.get("reservation_echo") != thread_id or result.get("terminal_owner") is not False:
            raise ClaudeRuntimeError("Claude agent release lacks exact reservation ownership")
        if result.get("released") is not True:
            # The bridge deferred the release and kept this reservation's SDK
            # client alive.  Recording the parent as released would send the
            # next ``cancel_native_child`` down its parent-released branch: the
            # child would be killed in the ledger only, and the live controller
            # that owns ``stop_task`` would never be asked to stop it.  The
            # adapter follows the release the bridge attests.
            #
            # A deferral is also the window the agent's bill arrives in: the
            # reader is still running because the SDK ``result`` message, which
            # carries ``usage`` and ``total_cost_usd``, has not come yet.  Name
            # it so a close can wait for it rather than cancel the reader.
            state["release_deferred"] = True
            state["release_status"] = status
            return
        state["released"] = True
        state["release_deferred"] = False
        self._turn_readiness.pop(thread_id, None)
        self._settle_native_children_of_released_parent(thread_id)

    def pending_usage_releases(self) -> tuple[str, ...]:
        """Reservations whose reader still owes the turn's final usage message.

        Reading the dictionary only, so a close can ask this and spend nothing
        when there is nothing outstanding -- which is the ordinary case.
        """

        return tuple(
            thread_id
            for thread_id, state in tuple(self._threads.items())
            if isinstance(state, dict)
            and state.get("release_deferred") is True
            and state.get("released") is not True
        )

    def await_pending_usage_releases(self, budget_seconds: float) -> dict[str, bool]:
        """Wait out the gap between a worker going terminal and its bill.

        Measured on 2026-10-01: ``agent.terminal`` is written about 1.1 s before
        the SDK ``result`` message that states the tokens and the dollar figure,
        and a quit inside that gap cancelled the reader.  Three quits with no
        pause lost the record; two with a pause kept it (207,933 tokens,
        $0.238).  Asking the bridge again is the wait: the reader releases
        itself once it has drained, and the usage event reaches this adapter on
        the same pipe before the release answer does.

        Returns one verdict per reservation: True once the reader has drained,
        False when the budget ran out first, which the caller records rather
        than leaving the row a silent null.
        """

        pending = self.pending_usage_releases()
        if not pending:
            return {}
        received = {thread_id: False for thread_id in pending}
        for thread_id in pending:
            self._usage_release_failures.pop(thread_id, None)
        given_up: set[str] = set()
        ends_at = time.monotonic() + max(0.0, float(budget_seconds))
        while True:
            outstanding = [
                thread_id for thread_id in pending
                if not received[thread_id] and thread_id not in given_up
            ]
            for position, thread_id in enumerate(outstanding):
                remaining = ends_at - time.monotonic()
                if remaining <= 0:
                    break
                state = self._threads.get(thread_id)
                if not isinstance(state, dict):
                    given_up.add(thread_id)
                    self._usage_release_failures[thread_id] = (
                        "the reservation had left this adapter before its final "
                        "usage message arrived"
                    )
                    continue
                # What is left is shared between the reservations still to be
                # asked on this pass.  One stalled reader used to be handed the
                # whole budget: the ready sibling behind it was never asked at
                # all, and its row was called missing while its bill was already
                # queued.  The poll interval is the floor, so a share that has
                # shrunk to nothing still buys one real attempt.
                share = max(remaining / (len(outstanding) - position), _PENDING_USAGE_POLL_SECONDS)
                try:
                    self.release_terminal_thread(
                        thread_id,
                        status=str(state.get("release_status") or "completed"),
                        timeout_seconds=min(self._timeout, remaining, share),
                    )
                except Exception as exc:
                    # Every fault leaves a reason behind, named after its cause.
                    # A bridge that cannot answer will not deliver the usage
                    # either, so asking it again spends the budget for nothing.
                    # KeyboardInterrupt and CancelledError are not Exception, so
                    # a close the operator stopped stops here as well.
                    given_up.add(thread_id)
                    self._usage_release_failures[thread_id] = (
                        "the Claude bridge failed while this worker's final usage was "
                        f"outstanding: {type(exc).__name__}: {exc}"
                    )
                    continue
                if state.get("released") is True:
                    received[thread_id] = True
            still_waiting = [
                thread_id for thread_id in pending
                if not received[thread_id] and thread_id not in given_up
            ]
            remaining = ends_at - time.monotonic()
            if not still_waiting or remaining <= 0:
                return received
            time.sleep(min(_PENDING_USAGE_POLL_SECONDS, remaining))

    def usage_release_failures(self) -> dict[str, str]:
        """Why each given-up reservation's final usage never arrived.

        A reservation whose wait only ran out of budget is absent: the caller
        already has a reason for that one.  Named faults are here so the outcome
        row says what went wrong rather than holding a silent null.
        """

        return dict(self._usage_release_failures)

    def _settle_native_children_of_released_parent(self, thread_id: str) -> str | None:
        """End the waits of native children whose parent's SDK reader is gone.

        The bridge keeps one SDK message reader per reservation, and a native
        child's terminal frame reaches vNext only through its parent's reader.
        Once that client is disconnected no frame can arrive, so a waiter would
        sit until its turn timeout -- 1800 s on 2026-09-30 -- and a cancel would
        ask the bridge for a controller that no longer exists.  A release the
        bridge deferred keeps its reader, and never reaches this.
        """

        reason = (
            "parent Claude reservation was released; its SDK reader can no longer deliver "
            "this native child's terminal frame"
        )
        with self._condition:
            orphaned = [
                task for task in self._native_child_tasks.values()
                if task.get("reservation_id") == thread_id and task.get("status") == "running"
            ]
            for task in orphaned:
                task["status"] = "killed"
                task["summary"] = reason
                task["settled_locally"] = True
            if not orphaned:
                return None
            self._condition.notify_all()
        return reason

    def _settle_native_children_on_close(self) -> str | None:
        """End every running native child's wait before the readers go away.

        Closing the adapter disconnects every bridge client and stops the SDK
        message readers, which are the only route a native child's terminal
        frame has into vNext.  Without this a waiter would sit out its whole
        turn timeout after shutdown for a frame nobody can send.
        """

        reason = (
            "Claude runtime closed; no SDK reader remains to deliver this native "
            "child's terminal frame"
        )
        with self._condition:
            orphaned = [
                task for task in self._native_child_tasks.values()
                if task.get("status") == "running"
            ]
            for task in orphaned:
                task["status"] = "killed"
                task["summary"] = reason
                task["settled_locally"] = True
            if not orphaned:
                return None
            self._condition.notify_all()
        return reason

    def _resume_released_thread(self, thread_id: str) -> None:
        state = self._threads.get(thread_id)
        if not isinstance(state, dict) or state.get("released") is not True:
            return
        session_id = state.get("provider_session")
        if _native_session_id(session_id):
            self.resume_attested_thread(
                runtime_thread=thread_id,
                provider_session=session_id,
                model=str(state["model"]),
                tool_handler=self._tool_handlers.get(thread_id),
                developer_instructions=state.get("developer_instructions"),
                approvals_reviewer=str(state["policy"]["posture"]["reviewer"]),
                workspace=state.get("workspace"),
                effort=str(state["effort"]),
                permission_mode=str(state["permission_mode"]),
            )
            return
        # A failed first turn may never attest a provider session. Recreate
        # its connected reservation under the same local handle for retry.
        result = self._request("restart_thread", {
            "reservation_id": thread_id,
            "workspace": state["workspace"],
            "tools": self._tool_definitions.get(thread_id, []),
            "developer_instructions": state.get("developer_instructions"),
            "requested_posture": RuntimePosture(
                workspace_writes=True, network="restricted", approvals_requested=True,
                reviewer=str(state["policy"]["posture"]["reviewer"]), environment_ready=True,
            ).as_dict(),
            "model": state["model"], "effort": state["effort"],
            "permission_mode": state["permission_mode"],
        })
        if result.get("reservation_echo") != thread_id or result.get("connection_evidence") != {"connected": True, "server_info_received": True}:
            raise ClaudeRuntimeError("Claude restart did not attest its local reservation")
        state["generation"] = 0
        state["released"] = False
        state["release_deferred"] = False

    def can_start_turn(self, thread_id: str) -> bool:
        """Return whether a parent reservation may safely begin another turn.

        Claude has one SDK message reader per reservation.  A native child's
        final lifecycle message can arrive after its parent's turn completed,
        so a new prompt is deferred until the bridge attests that this reader
        has drained. Other providers do not implement this optional seam.
        """

        with self._condition:
            state = self._threads.get(thread_id)
            if not isinstance(state, Mapping):
                raise ClaudeRuntimeError("unknown provider thread")
            if state.get("active_turn") is not None or thread_id in self._terminal_leases:
                return False
            generation = state.get("generation")
            if not isinstance(generation, int):
                raise ClaudeRuntimeError("Claude thread lacks a valid generation")
            # The initial prompt cannot overlap an earlier reader. Every
            # later generation must ask the bridge, even when no native child
            # was adopted: the primary terminal wakes its waiter before the
            # sole reader's ``finally`` clears bridge-local state.
            if generation == 0:
                return True
            child_tasks = [
                task for task in self._native_child_tasks.values()
                if task.get("reservation_id") == thread_id
            ]
            if any(task.get("status") == "running" for task in child_tasks):
                self._turn_readiness.pop(thread_id, None)
                return False
            released = state.get("released") is True
            if released:
                self._turn_readiness.pop(thread_id, None)
            cached = self._turn_readiness.get(thread_id)
            if (
                cached is not None
                and cached[0] == generation
                and cached[1] is False
                and time.monotonic() - cached[2] < 0.25
            ):
                return False
        if released:
            self._resume_released_thread(thread_id)
            return True
        result = self._request(
            "can_start_turn",
            {"reservation_id": thread_id, "generation": generation},
            timeout_seconds=min(self._timeout, 2.0),
            read_only=True,
        )
        ready = result.get("ready")
        if (
            result.get("reservation_echo") != thread_id
            or result.get("generation") != generation
            or not isinstance(ready, bool)
        ):
            raise ClaudeRuntimeError("Claude bridge readiness lacks exact local correlation")
        with self._condition:
            self._turn_readiness[thread_id] = (generation, ready, time.monotonic())
        return ready

    def wait_turn(
        self,
        handle: TurnHandle,
        *,
        timeout: float | None = None,
        timeout_seconds: float | None = None,
    ) -> Mapping[str, Any]:
        if timeout is not None and timeout_seconds is not None:
            raise ValueError("provide only one Claude wait timeout")
        native_child = self._wait_for_native_child_turn(
            handle,
            timeout_seconds=timeout if timeout is not None else timeout_seconds,
        )
        if native_child is not None:
            return native_child
        terminal = self._wait_for_terminal_turn(
            handle,
            timeout_seconds=timeout if timeout is not None else timeout_seconds,
        )
        if terminal is not None:
            return terminal
        try:
            result = self._request(
                "wait_turn",
                {"reservation_id": handle.thread_id, "turn_reference": handle.turn_id},
                timeout_seconds=timeout if timeout is not None else timeout_seconds,
            )
        finally:
            state = self._threads.get(handle.thread_id)
            if isinstance(state, dict) and state.get("active_turn") == handle.turn_id:
                state["active_turn"] = None
        if result.get("status") not in {"completed", "interrupted", "failed"}:
            raise ClaudeRuntimeError("unattested terminal result from Claude bridge")
        self._note_model_ran(handle.thread_id, result.get("model_ran"), result.get("model_ran_first"))
        if result.get("status") == "completed":
            state = self._threads.get(handle.thread_id)
            if not isinstance(state, Mapping) or not isinstance(state.get("provider_session"), str):
                raise ClaudeRuntimeError("Claude completion lacks attested native session identity")
        # The bridge terminal record is provider evidence.  Preserve its
        # usage/cost/session metadata for the runtime projection instead of
        # collapsing every successful Claude turn to a bare status.
        return dict(result)

    def steer(self, handle: TurnHandle, prompt: str) -> None:
        result = self._request(
            "steer",
            {"reservation_id": handle.thread_id, "turn_reference": handle.turn_id, "prompt": prompt},
        )
        if result.get("reservation_echo") not in {None, handle.thread_id}:
            raise ClaudeRuntimeError("Claude steer lacks local reservation correlation")
        if result.get("accepted") is True:
            return
        # The scheduler treats this as a request to queue for the next turn.
        # It must not record a delivery receipt for an SDK operation that does
        # not exist.
        if result.get("reason") == "native-steer-not-supported":
            raise ClaudeRuntimeError("active Claude steer is not supported; queue it for the next turn")
        raise ClaudeRuntimeError("Claude steer was not accepted")

    def interrupt(self, handle: TurnHandle) -> Mapping[str, Any]:
        with self._condition:
            child = dict(self._native_child_tasks.get(handle.thread_id, {}))
        if child:
            task_id = child.get("task_id")
            reservation = child.get("reservation_id")
            if handle.turn_id != task_id or not isinstance(reservation, str) or not isinstance(task_id, str):
                raise ClaudeRuntimeError("Claude native child interrupt lacks exact task correlation")
            result = self.stop_native_task(reservation, task_id)
            return {
                "interrupted": result.get("accepted") is True,
                "reservation_echo": reservation,
                "task_id": task_id,
                "turn_reference": task_id,
                "turn_source": "native-task",
            }
        with self._condition:
            terminal = self._terminal_turns.get((handle.thread_id, handle.turn_id))
            interrupt = self._terminal_interrupts.get(handle.thread_id)
            active = terminal is not None and terminal.get("status") == "running"
        if active:
            if interrupt is None:
                raise ClaudeRuntimeError("Claude native terminal has no interrupt controller")
            try:
                interrupt()
            except Exception as exc:
                raise ClaudeRuntimeError("Claude native terminal interrupt failed") from exc
            with self._condition:
                record = self._terminal_turns.get((handle.thread_id, handle.turn_id))
                if record is not None:
                    record["interrupt_requested"] = True
            return {
                "interrupted": True,
                "reservation_echo": handle.thread_id,
                "turn_reference": handle.turn_id,
                "turn_source": "terminal-hook",
            }
        result = self._request("interrupt", {"reservation_id": handle.thread_id, "turn_reference": handle.turn_id})
        if result.get("interrupted") is not True:
            raise ClaudeRuntimeError("Claude interrupt was not exactly correlated")
        state = self._threads.get(handle.thread_id)
        if isinstance(state, dict) and state.get("active_turn") == handle.turn_id:
            state["active_turn"] = None
        return result

    def begin_native_terminal(
        self,
        runtime_thread: str,
        *,
        relay_handler: RelayHandler,
        lease_dir: str | Path,
        executable: str = "claude",
    ) -> ClaudeTerminalLaunchSpec:
        """Hand an idle, attested Claude session to a service-owned CLI PTY.

        The scheduler must acquire its control lease before this method.  This
        adapter only makes the provider ownership transfer and starts the
        per-lease MCP/hook relay; it never starts a shell or terminal process.
        """

        state = self._threads.get(runtime_thread)
        if not isinstance(state, dict):
            raise ClaudeRuntimeError("Claude terminal handoff lacks a local runtime reservation")
        session_id = state.get("provider_session")
        if not _native_session_id(session_id):
            raise ClaudeRuntimeError("Claude terminal handoff requires an attested native session")
        if state.get("active_turn") is not None:
            raise ClaudeRuntimeError("cannot hand Claude session to terminal while an SDK turn is active")
        # The handoff asks the bridge for ``release_for_terminal``, which drops
        # this reservation's SDK client.  A native child's terminal frame only
        # reaches vNext through that client's reader, so a running child would
        # be stranded with no reader and its waiter would sit until the turn
        # timeout.  Refuse the handoff; the caller decides what to do with the
        # child.
        with self._condition:
            running_children = sum(
                1 for task in self._native_child_tasks.values()
                if task.get("reservation_id") == runtime_thread and task.get("status") == "running"
            )
        if running_children:
            raise ClaudeRuntimeError(
                "cannot hand Claude session to terminal while "
                f"{running_children} native child task(s) are running"
            )
        if runtime_thread in self._terminal_leases:
            raise ClaudeRuntimeError("Claude terminal handoff already has an active lease")
        self._resume_released_thread(runtime_thread)
        relay = ClaudeTerminalRelayServer(relay_handler)
        manager = ClaudeTerminalLeaseManager(lease_dir, relay)
        try:
            relay.start()
            lease = manager.acquire(
                runtime_thread_id=runtime_thread,
                native_session_id=session_id,
                model=str(state["model"]),
                effort=str(state["effort"]),
                permission_mode=str(state.get("permission_mode", self._permission_mode)),
                sdk_turn_active=False,
                executable=executable,
            )
            result = self._request(
                "release_for_terminal",
                {"reservation_id": runtime_thread, "session_id": session_id},
            )
            if result.get("reservation_echo") != runtime_thread or result.get("session_id") != session_id or result.get("released") is not True:
                raise ClaudeRuntimeError("Claude bridge did not attest idle terminal release")
        except BaseException:
            if "lease" in locals():
                try:
                    manager.release(lease.lease_id, terminal_stopped=True)
                except ClaudeTerminalError:
                    pass
            relay.close()
            raise
        self._terminal_leases[runtime_thread] = (relay, manager, lease)
        return lease.launch

    def native_terminal_relay_handler(
        self,
        runtime_thread: str,
        *,
        invoke_tool: TerminalToolInvoker,
        emit_event: TerminalEventSink,
        active_turn_reference: Callable[[], str | None] | None = None,
    ) -> RelayHandler:
        """Build the semantic relay used by a service-owned Claude terminal.

        ``invoke_tool`` is the scheduler's already-correlated dynamic-tool
        entry point.  ``emit_event`` persists terminal lifecycle evidence and
        may return a hook decision.  Both callbacks are injected so this
        provider adapter neither owns scheduler state nor invents a terminal
        turn identity.
        """

        if runtime_thread not in self._threads:
            raise ClaudeRuntimeError("Claude terminal relay lacks a local runtime reservation")

        def handler(event: Any) -> Mapping[str, Any]:
            if event.channel == "mcp":
                return self._handle_terminal_mcp(
                    runtime_thread,
                    event,
                    invoke_tool,
                    emit_event,
                    turn_reference=active_turn_reference() if active_turn_reference else None,
                )
            if event.channel == "hook":
                payload = dict(event.payload)
                payload.update({
                    "provider": self.provider,
                    "runtime_thread": runtime_thread,
                    "lease_id": event.lease_id,
                    "channel": "hook",
                })
                result = emit_event(runtime_thread, payload)
                # Hooks use a distinct Claude JSON result schema.  Keep an
                # injected explicit decision, otherwise continue so terminal
                # observation never becomes an accidental approval policy.
                return dict(result) if isinstance(result, Mapping) else {"continue": True}
            raise ClaudeRuntimeError("Claude terminal relay channel is unsupported")

        return handler

    def terminal_relay_handler(self, runtime_thread: str) -> RelayHandler:
        """Return a terminal relay with an explicit, lease-local turn seam.

        Claude hooks identify the submitted prompt and stop boundary but do
        not publish an SDK turn ID.  ``UserPromptSubmit`` therefore opens a
        vNext-generated reference whose ``turn_source`` remains
        ``terminal-hook`` throughout.  ``Stop`` completes that same reference.
        This creates an honest control boundary for the service-owned PTY; it
        never claims the reference was supplied by Claude's SDK.
        """

        if runtime_thread not in self._threads:
            raise ClaudeRuntimeError("Claude terminal relay lacks a local runtime reservation")

        def invoke(thread: str, tool: str, arguments: Mapping[str, Any], context: Mapping[str, Any]) -> Mapping[str, Any]:
            turn_reference = context.get("turn_reference")
            handler = self._tool_handlers.get(thread)
            if not isinstance(turn_reference, str):
                return {
                    "success": False,
                    "value": {"error": "native Claude terminal tool call arrived before prompt turn hook"},
                }
            if handler is None:
                return {
                    "success": False,
                    "value": {"error": "native Claude terminal has no configured vNext tool controller"},
                }
            request_id = context.get("request_id")
            lease_id = context.get("lease_id")
            call_id = f"terminal-mcp:{lease_id}:{request_id}"
            try:
                result = handler(
                    tool,
                    dict(arguments),
                    ToolCallContext(thread_id=thread, turn_id=turn_reference, call_id=call_id),
                )
            except Exception as exc:
                raise ClaudeRuntimeError("native Claude terminal tool controller failed") from exc
            if not isinstance(result, ToolCallResult):
                raise ClaudeRuntimeError("native Claude terminal tool controller returned an unsupported result")
            return result.as_dict()

        def emit(thread: str, value: Mapping[str, Any]) -> Mapping[str, Any] | None:
            item = dict(value)
            item.setdefault("reservation_id", thread)
            with self._condition:
                hook_name = item.get("hook_event_name")
                state = self._threads.get(thread)
                if isinstance(state, dict):
                    generation = state.get("generation")
                    if isinstance(generation, int):
                        item.setdefault("generation", generation)
                    session_id = state.get("provider_session")
                    if _native_session_id(session_id):
                        item.setdefault("provider_correlation", {"session": session_id})
                        item.setdefault("correlation_attested", True)
                active_turn = state.get("active_turn") if isinstance(state, dict) else None
                if hook_name == "UserPromptSubmit":
                    if not isinstance(active_turn, str):
                        active_turn = f"claude-terminal-turn-{uuid.uuid4().hex}"
                        if isinstance(state, dict):
                            state["active_turn"] = active_turn
                        self._terminal_turns[(thread, active_turn)] = {
                            "status": "running",
                            "source": "terminal-hook",
                        }
                    item.update({
                        "name": "terminal_turn_started",
                        "turn_reference": active_turn,
                        "turn_source": "terminal-hook",
                    })
                elif hook_name == "Stop" and isinstance(active_turn, str):
                    record = self._terminal_turns.get((thread, active_turn))
                    if record is not None:
                        record["status"] = "interrupted" if record.get("interrupt_requested") else "completed"
                    if isinstance(state, dict):
                        state["active_turn"] = None
                    item.update({
                        "name": "terminal_turn_completed",
                        "turn_reference": active_turn,
                        "turn_source": "terminal-hook",
                    })
                    self._condition.notify_all()
                else:
                    item.setdefault("name", "terminal_hook")
                    if isinstance(active_turn, str):
                        item.setdefault("turn_reference", active_turn)
                        item.setdefault("turn_source", "terminal-hook")
                previous = self._events[-1].get("cursor") if self._events else 0
                item["cursor"] = previous + 1 if isinstance(previous, int) else 1
                self._events.append(item)
            return None

        def active_turn_reference() -> str | None:
            with self._condition:
                state = self._threads.get(runtime_thread)
                value = state.get("active_turn") if isinstance(state, dict) else None
                return value if isinstance(value, str) else None

        return self.native_terminal_relay_handler(
            runtime_thread,
            invoke_tool=invoke,
            emit_event=emit,
            active_turn_reference=active_turn_reference,
        )

    def _handle_terminal_mcp(
        self,
        runtime_thread: str,
        event: Any,
        invoke_tool: TerminalToolInvoker,
        emit_event: TerminalEventSink,
        *,
        turn_reference: str | None = None,
    ) -> Mapping[str, Any]:
        payload = event.payload
        request_id = payload.get("id")
        method = payload.get("method")
        if not isinstance(method, str):
            return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32600, "message": "MCP method is required"}}
        if method == "initialize":
            return {"jsonrpc": "2.0", "id": request_id, "result": {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}, "serverInfo": {"name": "vnext", "version": "1"}}}
        if method == "tools/list":
            definitions = self._tool_definitions.get(runtime_thread, [])
            return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": [dict(value) for value in definitions]}}
        if method != "tools/call":
            return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": "MCP method is not supported"}}
        params = payload.get("params")
        name = params.get("name") if isinstance(params, Mapping) else None
        arguments = params.get("arguments", {}) if isinstance(params, Mapping) else {}
        names = {str(item.get("name")) for item in self._tool_definitions.get(runtime_thread, [])}
        if not isinstance(name, str) or name not in names or not isinstance(arguments, Mapping):
            return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32602, "message": "MCP tool invocation is invalid"}}
        context = {
            "provider": self.provider,
            "runtime_thread": runtime_thread,
            "lease_id": event.lease_id,
            "channel": "mcp",
            "request_id": request_id,
        }
        if turn_reference is not None:
            context["turn_reference"] = turn_reference
            context["turn_source"] = "terminal-hook"
        emit_event(runtime_thread, {"name": "terminal_mcp_tool", "tool": name, **context})
        try:
            answer = invoke_tool(runtime_thread, name, dict(arguments), context)
        except Exception as exc:
            return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32000, "message": f"vNext tool invocation failed: {type(exc).__name__}"}}
        if not isinstance(answer, Mapping) or not isinstance(answer.get("success"), bool) or not isinstance(answer.get("value"), Mapping):
            return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32000, "message": "vNext tool response is invalid"}}
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [{"type": "text", "text": json.dumps(answer["value"], ensure_ascii=False, separators=(",", ":"))}],
                "isError": not answer["success"],
            },
        }

    def set_native_terminal_interrupt(self, runtime_thread: str, callback: TerminalInterrupt) -> None:
        """Register the service PTY interrupt for the active terminal lease."""

        if runtime_thread not in self._terminal_leases:
            raise ClaudeRuntimeError("Claude native terminal interrupt lacks an active lease")
        if not callable(callback):
            raise TypeError("Claude native terminal interrupt must be callable")
        with self._condition:
            self._terminal_interrupts[runtime_thread] = callback

    def _wait_for_native_child_turn(
        self, handle: TurnHandle, *, timeout_seconds: float | None
    ) -> Mapping[str, Any] | None:
        """Wait for the provider's terminal task notification, not parent exit.

        A child task's native turn is its exact task ID.  Completion has to be
        delivered by the bridge before this returns: a parent ResultMessage is
        neither a child receipt nor evidence that its descendants drained.
        """

        with self._condition:
            task = self._native_child_tasks.get(handle.thread_id)
            if task is None:
                return None
            if task.get("task_id") != handle.turn_id:
                raise ClaudeRuntimeError("Claude native child wait lacks exact task correlation")
            deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
            while task.get("status") == "running":
                if self._fatal:
                    raise ClaudeRuntimeError(self._fatal)
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ClaudeRuntimeError("Claude native child wait timed out")
                self._condition.wait(remaining)
            terminal_status = task.get("status")
            status = {
                "completed": "completed",
                "failed": "failed",
                "stopped": "interrupted",
                "killed": "interrupted",
            }.get(terminal_status)
            if status is None:
                raise ClaudeRuntimeError("Claude native child has an unsupported terminal state")
            result: dict[str, Any] = {
                "status": status,
                "native_task_terminal": True,
                "task_id": handle.turn_id,
                "turn_reference": handle.turn_id,
                "turn_source": "native-task",
            }
            summary = task.get("summary")
            if isinstance(summary, str):
                result["summary"] = summary
            usage = task.get("usage")
            if isinstance(usage, Mapping):
                result["usage"] = dict(usage)
                result["usage_source"] = "native_task"
            return result

    def _wait_for_terminal_turn(
        self,
        handle: TurnHandle,
        *,
        timeout_seconds: float | None,
    ) -> Mapping[str, Any] | None:
        key = (handle.thread_id, handle.turn_id)
        with self._condition:
            record = self._terminal_turns.get(key)
            if record is None:
                return None
            deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
            while record.get("status") == "running":
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise ClaudeRuntimeError("Claude terminal turn timed out")
                self._condition.wait(remaining)
            status = record.get("status")
            if status not in {"completed", "interrupted", "failed"}:
                raise ClaudeRuntimeError("Claude terminal turn has an unattested status")
            state = self._threads.get(handle.thread_id)
            if isinstance(state, dict) and state.get("active_turn") == handle.turn_id:
                state["active_turn"] = None
            return {
                "status": status,
                "reservation_echo": handle.thread_id,
                "turn_reference": handle.turn_id,
                "turn_source": "terminal-hook",
            }

    def resume_after_native_terminal(
        self,
        runtime_thread: str,
        *,
        terminal_stopped: bool,
        tool_handler: DynamicToolHandler | None = None,
    ) -> Mapping[str, Any]:
        """Resume SDK ownership only after the service observes PTY exit."""

        held = self._terminal_leases.get(runtime_thread)
        state = self._threads.get(runtime_thread)
        if held is None or not isinstance(state, dict):
            raise ClaudeRuntimeError("Claude terminal resume lacks an active lease")
        relay, manager, lease = held
        if not terminal_stopped:
            raise ClaudeRuntimeError("cannot resume Claude SDK before terminal stop is confirmed")
        # A Ctrl+C path can terminate the PTY without a Claude Stop hook. Wake
        # the adopted terminal waiter before reacquiring the SDK so no control
        # turn remains falsely active during the owner transfer.
        self._settle_active_terminal_turn(runtime_thread, status="interrupted")
        try:
            result = self.resume_attested_thread(
                runtime_thread=runtime_thread,
                provider_session=lease.native_session_id,
                model=str(state["model"]),
                tool_handler=tool_handler if tool_handler is not None else self._tool_handlers.get(runtime_thread),
                approvals_reviewer=str(state["policy"]["posture"]["reviewer"]),
                workspace=str(self._workspace),
                effort=str(state["effort"]),
                permission_mode=str(state.get("permission_mode", "default")),
            )
            manager.release(
                lease.lease_id,
                terminal_stopped=True,
                resumed_native_session_id=lease.native_session_id,
            )
            return result
        except ClaudeTerminalError as exc:
            raise ClaudeRuntimeError("Claude terminal lease could not be released") from exc
        finally:
            # A failed SDK resume leaves no live terminal process.  Retain no
            # stale endpoint/token; a caller may start a fresh attested
            # recovery rather than accidentally reusing this lease.
            self._terminal_leases.pop(runtime_thread, None)
            self._terminal_interrupts.pop(runtime_thread, None)
            self._settle_active_terminal_turn(runtime_thread, status="interrupted")
            relay.close()

    def _settle_active_terminal_turn(self, runtime_thread: str, *, status: str) -> None:
        """Settle a hook turn when its owned PTY exits without a Stop hook."""

        with self._condition:
            state = self._threads.get(runtime_thread)
            active_turn = state.get("active_turn") if isinstance(state, dict) else None
            if isinstance(active_turn, str):
                record = self._terminal_turns.get((runtime_thread, active_turn))
                if record is not None and record.get("status") == "running":
                    record["status"] = status
                state["active_turn"] = None
            self._condition.notify_all()

    def stop_native_task(self, runtime_thread: str, task_id: str) -> Mapping[str, Any]:
        """Request stop for one SDK-observed native task.

        The provider attests only the task ID here.  It is not represented as
        a fully managed vNext child until the scheduler receives an attested
        child session and parent linkage.
        """

        if runtime_thread in self._terminal_leases:
            raise ClaudeRuntimeError("native Claude task control is unavailable while terminal owns the session")
        result = self._request(
            "stop_native_task",
            {"reservation_id": runtime_thread, "task_id": task_id},
        )
        if result.get("reservation_echo") != runtime_thread or result.get("task_id") != task_id or result.get("accepted") is not True:
            raise ClaudeRuntimeError("Claude native task stop was not exactly correlated")
        return result

    def cancel_native_child(self, native_child_thread_id: str) -> Mapping[str, Any]:
        """Request a stop for one running, ledger-attested native task."""

        with self._condition:
            task = dict(self._native_child_tasks.get(native_child_thread_id, {}))
        reservation = task.get("reservation_id")
        task_id = task.get("task_id")
        if not isinstance(reservation, str) or not isinstance(task_id, str):
            raise ClaudeRuntimeError("Claude native child cancel lacks an attested task")
        if task.get("status") != "running":
            return {"status": "not-running", "thread_id": native_child_thread_id}
        parent = self._threads.get(reservation)
        if isinstance(parent, Mapping) and parent.get("released") is True:
            # The bridge disconnected this reservation's SDK client, which owned
            # ``stop_task``. Asking for it returns "native task stop has no
            # active SDK controller" and the cancel is recorded as refused,
            # while the child is in fact beyond anyone's reach. Settle it here
            # so the cancel succeeds and the child's waiter stops.
            reason = self._settle_native_children_of_released_parent(reservation)
            return {
                "status": "parent-released",
                "thread_id": native_child_thread_id,
                "task_id": task_id,
                "reason": reason,
            }
        self.stop_native_task(reservation, task_id)
        return {
            "status": "interrupt-requested",
            "thread_id": native_child_thread_id,
            "task_id": task_id,
        }

    def set_native_child_observer(
        self,
        observer: NativeChildObserver | None,
        *,
        parent_agent_resolver: NativeParentAgentResolver | None = None,
    ) -> None:
        """Install the scheduler callback for bridge-attested native tasks.

        The bridge may report a task before the session runtime installs this
        callback.  Replay retained observations after installation; an
        observation is keyed by the provider's scoped runtime child key, so a
        replay cannot create a second vNext child for the same native task.
        """

        with self._condition:
            self._native_child_observer = observer
            self._native_parent_agent_resolver = parent_agent_resolver
            observations = tuple(self._native_child_observations.values())
        if observer is not None and parent_agent_resolver is not None:
            for observation in observations:
                self._adopt_native_child_observation(observation)

    def native_child_attestations(self) -> tuple[dict[str, Any], ...]:
        """Return content-safe facts for each ledger-attested native task."""

        with self._condition:
            values: list[dict[str, Any]] = []
            for runtime_thread, observation in self._native_child_observations.items():
                task = self._native_child_tasks.get(runtime_thread, {})
                binding = self._native_child_bindings.get(runtime_thread)
                values.append(
                    {
                        "provider": observation.provider,
                        "parent_thread_id": observation.parent_thread_id,
                        "native_child_thread_id": runtime_thread,
                        "native_child_id": observation.native_child_id,
                        "task_id": task.get("task_id"),
                        "status": task.get("status"),
                        "bound_agent_id": binding.agent_id if binding is not None else None,
                    }
                )
            return tuple(values)

    def _adopt_native_child_observation(self, observation: NativeChildObservation) -> str | None:
        """Bind one observed child to a vNext agent; say why when that is refused."""

        with self._condition:
            observer = self._native_child_observer
            resolver = self._native_parent_agent_resolver
            already_bound = observation.native_child_thread_id in self._native_child_bindings
        if observer is None or resolver is None or already_bound:
            return None
        expected_parent = resolver(observation.parent_thread_id)
        if expected_parent != observation.parent_agent_id:
            self._set_fatal("Claude native child parent agent is not exactly resolved")
            return None
        try:
            binding = observer(observation)
        except Exception as exc:
            self._record_failure(
                FailureCategory.IDENTITY_BINDING_FAILED,
                step="native_child_adopt",
                exception_type=type(exc).__name__,
            )
            return f"vNext could not adopt the child ({type(exc).__name__}: {str(exc)[:200]})"
        if (
            not isinstance(binding, NativeChildBinding)
            or binding.provider != self.provider
            or binding.native_thread_id != observation.native_child_thread_id
            or not binding.agent_id
        ):
            self._set_fatal("Claude native child observer returned an invalid binding")
            return None
        with self._condition:
            self._native_child_bindings[observation.native_child_thread_id] = binding
            if binding.tool_handler is not None:
                self._tool_handlers[observation.native_child_thread_id] = binding.tool_handler
            self._condition.notify_all()
        return None

    def _observe_native_child_event(self, event: Mapping[str, Any], cursor: int) -> bool:
        """Adopt only a bridge-ledger-attested child task.

        `native_runtime_thread_id` is scoped by the parent session.  A task ID
        is its only provider control turn; the parent SDK turn remains solely
        correlation evidence and is never reused as a child handle.
        """

        if event.get("name") != "native_child" or event.get("tracking") != "attested" or event.get("status") != "running":
            return False
        reservation = event.get("reservation_id")
        runtime_thread = event.get("native_runtime_thread_id")
        task_id = event.get("task_id")
        native_agent_id = event.get("native_agent_id")
        parent_tool_use_id = event.get("parent_tool_use_id")
        correlation = event.get("provider_correlation")
        if not all(
            isinstance(value, str) and value
            for value in (reservation, runtime_thread, task_id, native_agent_id, parent_tool_use_id)
        ):
            return False
        if event.get("turn_reference") != task_id:
            return False
        state = self._threads.get(reservation)
        if not isinstance(state, Mapping) or not isinstance(state.get("provider_session"), str):
            return False
        if runtime_thread != "claude-native:" + state["provider_session"] + ":" + native_agent_id:
            return False
        generation = event.get("generation")
        parent_turn = event.get("parent_turn_reference")
        if (
            not isinstance(generation, int)
            or not isinstance(parent_turn, str)
            or event.get("correlation_attested") is not True
            or not isinstance(correlation, Mapping)
            or correlation.get("session") != state.get("provider_session")
            or correlation.get("turn") != parent_turn
        ):
            return False
        parent_agent_resolver = self._native_parent_agent_resolver
        if parent_agent_resolver is None:
            return False
        parent_runtime = reservation
        parent_provider_turn = None
        native_parent = event.get("parent_native_agent_id")
        if native_parent is not None:
            if not isinstance(native_parent, str) or not native_parent or native_parent == native_agent_id:
                return False
            parent_runtime = "claude-native:" + state["provider_session"] + ":" + native_parent
            with self._condition:
                parent_task = self._native_child_tasks.get(parent_runtime)
                parent_binding = self._native_child_bindings.get(parent_runtime)
                if (
                    parent_task is None or parent_binding is None
                    or parent_task.get("reservation_id") != reservation
                    or parent_task.get("parent_session_id") != state["provider_session"]
                ):
                    return False
                parent_provider_turn = parent_task["task_id"]
                if (
                    event.get("parent_runtime_thread_id") != parent_runtime
                    or event.get("parent_native_task_id") != parent_provider_turn
                    or event.get("parent_native_turn_id") != parent_provider_turn
                ):
                    return False
        parent_agent_id = parent_agent_resolver(parent_runtime)
        if not isinstance(parent_agent_id, str) or not parent_agent_id:
            return False
        task_contract = event.get("task_contract")
        if (
            not isinstance(task_contract, Mapping)
            or task_contract.get("role") != "worker"
            or task_contract.get("role_source") != "vnext-native-task-mapping"
        ):
            return False
        objective = task_contract.get("objective")
        requested_model = task_contract.get("requested_model")
        observed_model = task_contract.get("observed_model")
        if observed_model is not None:
            # An observed model is accepted only from an attested bridge
            # source. The forwarded SDK AssistantMessage path is promoted by
            # the bridge only after its exact child identity join succeeds.
            if task_contract.get("observed_model_source") not in {
                "saved-child-assistant-message", "sdk-agent-metadata",
                "forwarded-child-assistant-message",
            }:
                self._native_child_refusal = (
                    f"the child's model came from {task_contract.get('observed_model_source')!r}, "
                    "which is not a source the bridge attests"
                )
                return False
            selected_model = observed_model
        else:
            selected_model = requested_model
        if not isinstance(objective, str) or not objective.strip():
            self._native_child_refusal = "the child's task contract has no objective"
            return False
        # Preserve requested aliases separately. Accept a saved child model
        # only with its explicit source, or a launch model this thread agrees
        # with. Never resolve an alias by copying the parent's model selection:
        # agreement runs the other way, from the provider's canonical answer
        # back to the alias that asked for it.
        if not _native_model_agrees(state.get("model"), selected_model):
            self._native_child_refusal = (
                f"the child answered on {selected_model!r}, which does not agree "
                f"with this thread's model {state.get('model')!r}"
            )
            return False
        # The vNext node is named by the catalog entry that admitted it. When
        # the thread launched on an alias such as ``opus`` and the provider
        # answers ``claude-opus-5``, the canonical id is not a catalog key: the
        # control plane refused it as an unknown model, the refusal was only a
        # diagnostics record, and every later event for the child was held
        # forever. The provider's own answer stays in the attested task
        # contract as ``observed_model``.  A child on another family keeps
        # its own family's name; see ``_native_catalog_model``.
        catalog_model = _native_catalog_model(state["model"], selected_model)
        observation = NativeChildObservation(
            provider=self.provider,
            parent_agent_id=parent_agent_id,
            parent_thread_id=parent_runtime,
            native_child_thread_id=runtime_thread,
            attested=True,
            delivery_contract={
                "history": "unavailable",
                "usage": "available",
                "interrupt": "available",
                "context_messages": "available",
                "dynamic_tools": "available",
            },
            native_child_id=native_agent_id,
            model_id=catalog_model,
            # The provider does not declare an organization role.  The bridge
            # explicitly maps a native Agent task to the vNext worker role and
            # preserves that mapping in the attested task contract.
            role="worker",
            effort=str(state.get("effort") or "high"),
            objective=objective,
            task_contract=dict(task_contract),
            source_cursor=cursor,
            # Root SDK turn references are bridge-local. Only a nested native
            # parent's task ID is a provider-issued control-turn identity.
            parent_native_turn_id=parent_provider_turn,
            capabilities={
                "native_task": "attested",
                "stop_task": "available",
                "dynamic_tools": "scoped",
            },
        )
        with self._condition:
            existing = self._native_child_observations.get(runtime_thread)
            if existing is not None and (
                existing.parent_thread_id != observation.parent_thread_id
                or existing.native_child_id != observation.native_child_id
                or existing.parent_native_turn_id != observation.parent_native_turn_id
            ):
                self._set_fatal("Claude native child identity conflicts with prior observation")
                return False
            if existing is None:
                self._native_child_observations[runtime_thread] = observation
            task = self._native_child_tasks.setdefault(runtime_thread, {})
            if task and task.get("task_id") != task_id:
                self._set_fatal("Claude native child task conflicts with prior observation")
                return False
            if task and not self._native_child_event_matches_task(event, task, task_id):
                return False
            if task.get("status") in {"completed", "failed", "stopped", "killed"}:
                return False
            task.update({
                "reservation_id": reservation,
                "task_id": task_id,
                "parent_session_id": state.get("provider_session"),
                "generation": generation,
                "parent_turn_reference": parent_turn,
                "parent_tool_use_id": parent_tool_use_id,
                "parent_native_agent_id": native_parent,
                "parent_native_task_id": parent_provider_turn,
                "parent_runtime_thread_id": parent_runtime,
            })
            task.setdefault("status", "running")
            task.setdefault("summary", None)
            self._turns[(runtime_thread, task_id)] = TurnHandle(
                thread_id=runtime_thread, turn_id=task_id, cursor=cursor
            )
            self._condition.notify_all()
        # A refusal used to reach only the diagnostics sink, and the child's
        # events then waited forever for an agent that never existed.
        refusal = self._adopt_native_child_observation(existing or observation)
        if refusal is not None:
            self._native_child_refusal = refusal
            return False
        return True

    def _record_native_child_usage(self, event: Mapping[str, Any]) -> bool:
        runtime_thread = event.get("native_runtime_thread_id")
        task_id = event.get("task_id")
        usage = event.get("usage")
        if not isinstance(runtime_thread, str) or not isinstance(task_id, str) or not isinstance(usage, Mapping):
            return False
        with self._condition:
            task = self._native_child_tasks.get(runtime_thread)
            if not self._native_child_event_matches_task(event, task, task_id):
                return False
            task["usage"] = dict(usage)
            self._condition.notify_all()
            return True

    def _record_native_child_completion(self, event: Mapping[str, Any]) -> bool:
        runtime_thread = event.get("native_runtime_thread_id")
        task_id = event.get("task_id")
        terminal_status = event.get("status")
        if (
            not isinstance(runtime_thread, str)
            or not isinstance(task_id, str)
            or terminal_status not in {"completed", "failed", "stopped", "killed"}
        ):
            return False
        with self._condition:
            task = self._native_child_tasks.get(runtime_thread)
            if not self._native_child_event_matches_task(event, task, task_id):
                return False
            previous = task.get("status")
            if previous in {"completed", "failed", "stopped", "killed"}:
                if previous != terminal_status:
                    if previous == "killed" and task.get("settled_locally") is True:
                        # vNext ended this child itself, so the provider's own
                        # terminal frame is late news about a child already
                        # settled.  Keep the one terminal state its waiters saw,
                        # record what the provider said, and stay healthy: the
                        # frame contradicts nothing the provider told us before.
                        task["late_provider_status"] = terminal_status
                        return False
                    self._set_fatal("Claude native child terminal state conflicts with prior evidence")
                    return False
                return True
            task["status"] = terminal_status
            summary = event.get("summary")
            if isinstance(summary, str):
                task["summary"] = summary
            usage = event.get("usage")
            if isinstance(usage, Mapping):
                task["usage"] = dict(usage)
            self._condition.notify_all()
            return True

    @staticmethod
    def _native_child_event_matches_task(
        event: Mapping[str, Any], task: Mapping[str, Any] | None, task_id: str
    ) -> bool:
        if not isinstance(task, Mapping):
            return False
        correlation = event.get("provider_correlation")
        return (
            task.get("task_id") == task_id
            and event.get("turn_reference") == task_id
            and event.get("reservation_id") == task.get("reservation_id")
            and event.get("generation") == task.get("generation")
            and event.get("parent_turn_reference") == task.get("parent_turn_reference")
            and (
                "parent_native_agent_id" not in event
                or event.get("parent_native_agent_id") == task.get("parent_native_agent_id")
            )
            and (
                task.get("parent_native_agent_id") is None
                or (
                    event.get("parent_native_agent_id") == task.get("parent_native_agent_id")
                    and event.get("parent_runtime_thread_id") == task.get("parent_runtime_thread_id")
                    and event.get("parent_native_task_id") == task.get("parent_native_task_id")
                    and event.get("parent_native_turn_id") == task.get("parent_native_task_id")
                )
            )
            and event.get("correlation_attested") is True
            and isinstance(correlation, Mapping)
            and correlation.get("session") == task.get("parent_session_id")
            and correlation.get("turn") == task.get("parent_turn_reference")
        )

    def read_thread(self, thread_id: str, *, after: int = 0, limit: int = 100) -> Mapping[str, Any]:
        result = self._request("read_thread", {"thread_id": thread_id, "after": after, "limit": limit})
        if result.get("reservation_echo") not in {None, thread_id}:
            raise ClaudeRuntimeError("Claude read lacks exact provider thread correlation")
        if not isinstance(result.get("items"), list) or not all(isinstance(item, Mapping) for item in result["items"]):
            raise ClaudeRuntimeError("Claude read returned an invalid transcript page")
        return result

    def _endpoint_identity(self) -> dict[str, Any]:
        """Attach endpoint truth only for a configured non-Anthropic session."""

        if not self._endpoint_attestation:
            return {}
        return {"endpoint_attestation": dict(self._endpoint_attestation)}

    def thread_identity_attestation(self, runtime_thread: str) -> Mapping[str, Any]:
        """Expose only an SDK-echoed session binding for a managed thread."""

        with self._condition:
            native_child = self._native_child_tasks.get(runtime_thread)
            if native_child is not None:
                observation = self._native_child_observations.get(runtime_thread)
                return {
                    "runtime_thread": runtime_thread,
                    "provider": self.provider,
                    # Claude task lifecycle events carry the attested parent
                    # session, not an invented child session UUID.
                    "bound": isinstance(native_child.get("parent_session_id"), str),
                    "provider_session": native_child.get("parent_session_id"),
                    "binding_phase": "native-task-attested",
                    "synthetic": False,
                    "origin": "native",
                    "native_agent_id": observation.native_child_id if observation is not None else None,
                    "native_task_id": native_child.get("task_id"),
                    "parent_runtime_thread_id": observation.parent_thread_id if observation is not None else None,
                    # The bridge parent reference is local vNext correlation,
                    # never a Claude-issued parent turn identity.
                    "parent_local_turn_reference": native_child.get("parent_turn_reference"),
                    "parent_turn_source": "bridge",
                    "parent_tool_use_id": native_child.get("parent_tool_use_id"),
                    "parent_native_agent_id": native_child.get("parent_native_agent_id"),
                    "parent_native_turn_id": observation.parent_native_turn_id if observation is not None else None,
                    **self._endpoint_identity(),
                }
        state = self._threads.get(runtime_thread)
        if state is None:
            return {
                "runtime_thread": runtime_thread,
                "provider": self.provider,
                "bound": False,
                "provider_session": None,
                "binding_phase": "reserved",
                "synthetic": False,
                **self._endpoint_identity(),
            }
        provider_session = state["provider_session"]
        return {
            "runtime_thread": runtime_thread,
            "provider": self.provider,
            "bound": provider_session is not None,
            "provider_session": provider_session,
            "binding_phase": state["binding_phase"],
            "synthetic": False,
            **self._endpoint_identity(),
        }

    def resume_thread(
        self,
        *,
        thread_id: str,
        model: str = "",
        tool_handler: DynamicToolHandler | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        developer_instructions: str | None = None,
        approvals_reviewer: str = "auto_review",
        workspace: str | None = None,
        effort: str = "high",
        permission_mode: str | None = None,
    ) -> Mapping[str, Any]:
        return self.resume_attested_thread(
            runtime_thread=thread_id,
            provider_session=thread_id,
            model=model,
            tool_handler=tool_handler,
            tools=tools,
            developer_instructions=developer_instructions,
            approvals_reviewer=approvals_reviewer,
            workspace=workspace,
            effort=effort,
            permission_mode=permission_mode,
        )

    def resume_attested_thread(
        self,
        *,
        runtime_thread: str,
        provider_session: str,
        model: str = "",
        tool_handler: DynamicToolHandler | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        developer_instructions: str | None = None,
        approvals_reviewer: str = "auto_review",
        workspace: str | None = None,
        effort: str = "high",
        permission_mode: str | None = None,
    ) -> Mapping[str, Any]:
        """Resume a native session while retaining its opaque local reservation."""

        if not isinstance(runtime_thread, str) or not runtime_thread:
            raise ClaudeRuntimeError("Claude resume lacks a local runtime reservation")
        if not _native_session_id(provider_session):
            raise ClaudeRuntimeError("Claude resume requires an attested native session")
        if effort not in CLAUDE_ACCEPTED_EFFORTS:
            raise ClaudeRuntimeError("Claude resume effort is not supported by the SDK")
        existing = self._threads.get(runtime_thread)
        permission_mode = (
            existing.get("permission_mode", self._permission_mode)
            if permission_mode is None and isinstance(existing, Mapping)
            else self._permission_mode if permission_mode is None else permission_mode
        )
        if permission_mode not in {"default", "acceptEdits"}:
            raise ClaudeRuntimeError("Claude resume permission mode is not supported by vNext")
        # A normal in-process reconnect can reuse the registration retained by
        # this adapter.  A fresh controller has no such cache, so its caller
        # must replay the deterministic manager definitions explicitly.  Do
        # not quietly substitute an empty registration for that root scope.
        if tools is None:
            definitions = [dict(value) for value in self._tool_definitions.get(runtime_thread, [])]
        else:
            definitions = _validated_tool_definitions(tools)
            if not isinstance(developer_instructions, str) or not developer_instructions.strip():
                raise ClaudeRuntimeError(
                    "Claude fresh tool resume requires developer instructions"
                )
        if developer_instructions is not None and not isinstance(developer_instructions, str):
            raise ClaudeRuntimeError("Claude resume developer instructions must be text")
        if (tool_handler is not None) is not bool(definitions):
            raise ClaudeRuntimeError("Claude tool registration requires exactly one caller handler")
        resume_workspace = self._workspace_inside(workspace)
        if resume_workspace is None:
            raise ClaudeRuntimeError("Claude resume workspace differs from owned workspace")
        requested_posture = RuntimePosture(
            workspace_writes=True,
            network="approval_gated",
            approvals_requested=True,
            reviewer=approvals_reviewer,
            environment_ready=True,
        )
        result = self._request(
            "resume",
            {
                "session_id": provider_session,
                "reservation_id": runtime_thread,
                "workspace": str(resume_workspace),
                "tools": definitions,
                **(
                    {"developer_instructions": developer_instructions}
                    if developer_instructions is not None
                    else {}
                ),
                "model": model,
                "effort": effort,
                "permission_mode": permission_mode,
                "requested_posture": requested_posture.as_dict(),
            },
        )
        registration = result.get("tool_registration")
        if (
            result.get("provider_echo") is not True
            or result.get("thread_id") != provider_session
            or result.get("reservation_echo") not in {None, runtime_thread}
            or not isinstance(result.get("policy"), Mapping)
            or result.get("connection_evidence")
            != {"connected": True, "server_info_received": True}
            or not isinstance(registration, Mapping)
            or registration.get("acknowledged") is not True
            or registration.get("model_id") != model
            or registration.get("tool_count") != len(definitions)
            or registration.get("tool_names") != [str(value["name"]) for value in definitions]
            or registration.get("definition_sha256") != _definition_digest(definitions)
            or registration.get("handler_registered") is not (tool_handler is not None)
        ):
            raise ClaudeRuntimeError("Claude resume did not attest the requested tool registration")
        try:
            require_workspace_policy(result["policy"], approvals_reviewer=approvals_reviewer)
        except RuntimePolicyError as exc:
            raise ClaudeRuntimeError("Claude resume returned an invalid resolved posture") from exc
        if result["policy"].get("posture", {}).get("network") != "approval_gated":
            raise ClaudeRuntimeError("Claude resume did not resolve approval-gated network access")
        if tool_handler is not None:
            self._tool_handlers[runtime_thread] = tool_handler
            self._tool_definitions[runtime_thread] = definitions
        else:
            self._tool_handlers.pop(runtime_thread, None)
            self._tool_definitions.pop(runtime_thread, None)
        prior_identity = existing.get("model_identity") if isinstance(existing, Mapping) else None
        self._threads[runtime_thread] = {
            **({"model_identity": dict(prior_identity)} if isinstance(prior_identity, Mapping) else {}),
            "policy": dict(result["policy"]),
            "provider_session": provider_session,
            "binding_phase": "resumed",
            "generation": 0,
            "model": model,
            "effort": effort,
            "permission_mode": permission_mode,
            "developer_instructions": developer_instructions,
            "workspace": str(resume_workspace),
            "released": False,
        }
        self._tool_registrations[runtime_thread] = dict(registration)
        self._note_model_identity(runtime_thread, model, result.get("model_identity"))
        return result

    def events_since(self, cursor: int | None = None) -> tuple[dict[str, Any], ...]:
        """Return raw correlated bridge records; never fabricate shared effects."""
        return tuple(
            dict(event)
            for event in self._events
            if cursor is None
            or (isinstance(event.get("cursor"), int) and event["cursor"] >= cursor)
        )

    def turn_process_ended(self, handle: object | None = None) -> bool | None:
        """Whether the bridge process behind a turn has exited.

        Every Claude turn runs inside the owned bridge subprocess, so its exit
        is evidence that no turn of this adapter is still running.  A bridge
        that was never started tells us nothing, and says so.
        """

        del handle
        owned = self._owned
        if owned is None:
            return None
        return owned.process.poll() is not None

    def diagnostics(self, *, timeout_seconds: float = 2.0) -> Mapping[str, Any]:
        """Return safe process and bridge-progress facts for an opt-in probe.

        The bridge intentionally keeps its stderr private because it can carry
        provider or local configuration text.  This snapshot exposes only
        process liveness, fixed counters, and bridge aggregate progress.
        """

        if timeout_seconds <= 0:
            raise ValueError("Claude diagnostics timeout must be positive")
        with self._condition:
            owned = self._owned
            pending_request_count = len(self._pending_requests)
            event_count = len(self._events)
            fatal = self._fatal is not None
            reader_thread_count = len(self._reader_threads)
            reader_threads_alive = sum(reader.is_alive() for reader in self._reader_threads)
            stderr_line_count = len(self._stderr)
            thread_count = len(self._threads)
            native_event_counts = {
                name: sum(event.get("name") == name for event in self._events)
                for name in ("native_child_identity", "native_child", "native_child_rejected",
                             "native_child_completed", "native_child_completed_rejected",
                             "native_child_stop", "native_child_stop_rejected")
            }
        process_running = owned is not None and owned.process.poll() is None
        result: dict[str, Any] = {
            "adapter": {
                "bridge_started": owned is not None,
                "bridge_process_running": process_running,
                "initialized": self._initialized,
                "closing": self._closing,
                "fatal": fatal,
                "fatal_stage": _fatal_stage(self._fatal),
                "fatal_code": _fatal_code(self._fatal),
                "pending_request_count": pending_request_count,
                "late_read_only_response_count": self._late_read_only_response_count,
                "late_response_count": self._late_response_count,
                "event_count": event_count,
                "thread_count": thread_count,
                "native_event_counts": native_event_counts,
                "native_child_task_count": len(self._native_child_tasks),
                "native_child_binding_count": len(self._native_child_bindings),
                "reader_thread_count": reader_thread_count,
                "reader_threads_alive": reader_threads_alive,
                "stderr_line_count": stderr_line_count,
            }
        }
        if owned is None or not process_running:
            return result
        try:
            result["bridge"] = dict(
                self._request("diagnostics", {}, timeout_seconds=timeout_seconds, read_only=True)
            )
        except ClaudeRuntimeError as exc:
            # The exception text is deliberately omitted: bridge stderr and
            # provider errors are not safe diagnostic content.
            result["bridge"] = {"available": False, "error_type": type(exc).__name__}
        return result

    def tool_registration_attestation(self, thread_id: str) -> Mapping[str, Any]:
        try:
            registration = self._tool_registrations[thread_id]
        except KeyError as exc:
            raise ClaudeRuntimeError("unknown provider thread")
        return dict(registration)

    def close(self) -> RuntimeCleanup:
        if self._runtime_cleanup is not None:
            return self._runtime_cleanup
        self._settle_native_children_on_close()
        for runtime_thread, (relay, manager, lease) in tuple(self._terminal_leases.items()):
            try:
                manager.release(lease.lease_id, terminal_stopped=True)
            except ClaudeTerminalError:
                pass
            relay.close()
            self._terminal_leases.pop(runtime_thread, None)
        if self._owned is None:
            self._runtime_cleanup = RuntimeCleanup(ProcessCleanup("not-started", 0, None, ()), True, True, ())
            return self._runtime_cleanup
        errors: list[str] = []
        self._closing = True
        try:
            self._request("close", {}, timeout_seconds=min(self._timeout, 2.0))
        except ClaudeRuntimeError:
            # A cooperative bridge acknowledgement is best-effort.  The owned
            # process supervisor below is the authoritative cleanup boundary.
            pass
        cleanup = self._owned.close(grace_seconds=5.0)
        for reader in self._reader_threads:
            reader.join(timeout=2.0)
        for stream in (self._owned.process.stdin, self._owned.process.stdout, self._owned.process.stderr):
            if stream is not None and not stream.closed:
                try:
                    stream.close()
                except OSError as exc:
                    errors.append(f"bridge stream close failed: {exc}")
        streams_drained = not any(reader.is_alive() for reader in self._reader_threads)
        handler_threads_drained = streams_drained
        if cleanup.residual_count:
            errors.append(f"owned bridge residual descendants: {cleanup.residual_count}")
        if not streams_drained:
            errors.append("bridge stream reader did not drain")
        self._runtime_cleanup = RuntimeCleanup(cleanup, streams_drained, handler_threads_drained, tuple(errors))
        return self._runtime_cleanup

    def _start_owned_bridge(self) -> None:
        if self._owned is not None:
            return
        from .vnext_runtimes import CLI_PATH_ENV, SDK_VERSION_ENV

        # These two name a side runtime's SDK version and Claude CLI.  Only the
        # side-runtime bridge command may set them, inside its own process, so
        # a value inherited from the server's environment never reaches a
        # bridge and never swaps in an unchecked CLI.
        environment = {
            key: value for key, value in os.environ.items()
            if key not in (CLI_PATH_ENV, SDK_VERSION_ENV)
        }
        try:
            owned = OwnedProcess.start(
                self._command,
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(self._workspace),
                bufsize=1,
            )
        except (OSError, ValueError) as exc:
            # An OSError message on Windows routinely carries an absolute
            # path; keep the class name only, as the bridge already does.
            self._record_failure(
                classify_exception(exc),
                step="spawn_bridge",
                exception_type=type(exc).__name__,
            )
            raise ClaudeRuntimeError("unable to start owned Claude bridge") from exc
        self._owned = owned
        if owned.process.stdout is None or owned.process.stderr is None:
            self._record_failure(FailureCategory.BRIDGE_STREAMS_UNAVAILABLE, step="capture_streams")
            raise ClaudeRuntimeError("owned Claude bridge has no captured streams")
        self._reader_threads = [
            threading.Thread(target=self._read_stdout, name="claude-bridge-stdout", daemon=True),
            threading.Thread(target=self._read_stderr, name="claude-bridge-stderr", daemon=True),
        ]
        for reader in self._reader_threads:
            reader.start()

    def _request(
        self,
        op: str,
        payload: Mapping[str, Any],
        *,
        timeout_seconds: float | None = None,
        read_only: bool = False,
    ) -> Mapping[str, Any]:
        if self._owned is None:
            self._record_failure(FailureCategory.BRIDGE_NOT_RUNNING, step=op)
            raise ClaudeRuntimeError("owned Claude bridge is not running")
        with self._condition:
            request_id = self._next_request
            self._next_request += 1
            self._pending_requests.add(request_id)
            try:
                self._send({"v": self._VERSION, "kind": "request", "id": request_id, "op": op, "payload": dict(payload)})
            except BaseException:
                self._pending_requests.discard(request_id)
                raise
            deadline = time.monotonic() + (self._timeout if timeout_seconds is None else timeout_seconds)
            while request_id not in self._responses:
                if self._fatal:
                    self._pending_requests.discard(request_id)
                    raise ClaudeRuntimeError(self._fatal)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._pending_requests.discard(request_id)
                    if read_only:
                        self._expired_read_only_requests.append(request_id)
                    else:
                        self._expired_requests.append(request_id)
                    self._record_failure(FailureCategory.BRIDGE_REQUEST_TIMEOUT, step=op)
                    raise ClaudeRuntimeError(f"Claude bridge timed out waiting for {op}")
                self._condition.wait(remaining)
            response = self._responses.pop(request_id)
            self._pending_requests.discard(request_id)
        if response.get("ok") is not True:
            # The single funnel every bridge-reported rejection passes
            # through.  The error literal is classified, not retained.
            error = response.get("error")
            self._record_failure(classify_message(error), step=op)
            raise ClaudeRuntimeError(str(error or f"Claude bridge rejected {op}"))
        result = response.get("result")
        if not isinstance(result, Mapping):
            self._record_failure(FailureCategory.BRIDGE_PROTOCOL_VIOLATION, step=op)
            raise ClaudeRuntimeError("Claude bridge returned a non-object result")
        return result

    @staticmethod
    def _send_step(record: Mapping[str, Any]) -> str:
        """Name the in-flight operation for a send-path record.

        Every `op` this adapter writes is a fixed literal chosen here, so it
        is a legal diagnostic label; a missing or non-literal one degrades to
        the generic step rather than travelling.
        """

        op = record.get("op")
        return f"send.{op}" if isinstance(op, str) and op else "send_request"

    def _send(self, record: Mapping[str, Any]) -> None:
        step = self._send_step(record)
        if self._owned is None or self._owned.process.stdin is None:
            self._record_failure(FailureCategory.BRIDGE_STDIN_UNAVAILABLE, step=step)
            raise ClaudeRuntimeError("owned Claude bridge stdin is unavailable")
        try:
            line = json.dumps(record, separators=(",", ":"), ensure_ascii=True)
            if len(line) > 65_536:
                raise ClaudeRuntimeError("Claude bridge record exceeds JSONL bound")
            self._owned.process.stdin.write(line + "\n")
            self._owned.process.stdin.flush()
        except (OSError, ValueError) as exc:
            self._record_failure(
                FailureCategory.BRIDGE_WRITE_FAILED,
                step=step,
                exception_type=type(exc).__name__,
            )
            raise ClaudeRuntimeError("cannot write owned Claude bridge") from exc

    def _read_stdout(self) -> None:
        assert self._owned is not None and self._owned.process.stdout is not None
        try:
            for line in self._owned.process.stdout:
                if len(line) > 65_536:
                    self._set_fatal("Claude bridge emitted oversized JSONL")
                    return
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    self._set_fatal("Claude bridge emitted malformed JSONL")
                    return
                if not isinstance(record, Mapping) or record.get("v") != self._VERSION:
                    self._set_fatal("Claude bridge emitted an invalid protocol record")
                    return
                if record.get("kind") == "response" and isinstance(record.get("id"), int):
                    with self._condition:
                        if record["id"] not in self._pending_requests:
                            if record["id"] in self._expired_read_only_requests:
                                self._expired_read_only_requests.remove(record["id"])
                                self._late_read_only_response_count += 1
                                continue
                            if record["id"] in self._expired_requests:
                                self._expired_requests.remove(record["id"])
                                self._late_response_count += 1
                                continue
                            self._set_fatal("Claude bridge response has no pending request")
                            return
                        self._responses[record["id"]] = record
                        self._condition.notify_all()
                elif record.get("kind") == "event":
                    self._record_event(record)
                else:
                    self._set_fatal("Claude bridge emitted an unknown protocol record")
                    return
        finally:
            if not self._closing:
                self._set_fatal("Claude bridge stdout closed unexpectedly")

    def _read_stderr(self) -> None:
        assert self._owned is not None and self._owned.process.stderr is not None
        for line in self._owned.process.stderr:
            self._stderr.append(line.rstrip())

    def captured_stderr(self) -> tuple[str, ...]:
        """The bridge's last stderr lines, for a local failure record.

        The diagnostics channel consumes this for a category label and never
        reports the text, because a receipt is a thing that can travel.  A
        failure record written beside the user's own workspace on the user's own
        machine is not, and the text is the only place a bridge that died before
        answering says why.  Today that cost a whole investigation: the adapter
        reported a missing SDK that pip and a plain import both found.
        """

        with self._condition:
            if self.provider == "zai":
                return ()
            return tuple(self._stderr)

    def _set_fatal(self, message: str) -> None:
        with self._condition:
            first = self._fatal is None
            if first:
                self._fatal = message
            self._condition.notify_all()
        # Diagnostics must not change observable behaviour when disarmed.
        # `drain_stderr` empties the buffer, so running it unconditionally
        # discarded the captured stderr even with the channel switched off.
        if first and diagnostics_enabled():
            category = classify_message(message)
            # The captured stderr is the only evidence of a bridge that died
            # before answering.  Consume it for a label; never report it.
            stderr_category = drain_stderr(self._stderr)
            if category in {FailureCategory.UNCLASSIFIED, FailureCategory.BRIDGE_EXITED_AT_STARTUP} and (
                stderr_category is not FailureCategory.UNCLASSIFIED
            ):
                category = stderr_category
            self._record_failure(category, step="bridge_stream")

    def _record_failure(
        self,
        category: FailureCategory,
        *,
        step: str,
        exception_type: str | None = None,
    ) -> None:
        record_failure(
            category,
            phase="claude_adapter",
            step=step,
            exception_type=exception_type,
            duration_ms=(time.monotonic() - self._started_monotonic) * 1000.0,
            counts={
                "pending_requests": len(self._pending_requests),
                "threads": len(self._threads),
                "events": len(self._events),
            },
            flags={
                "bridge_started": self._owned is not None,
                "initialized": self._initialized,
                "closing": self._closing,
            },
        )

    @staticmethod
    def _credential_refusal_detail(projection: Any) -> str | None:
        """The provider's own words, when a provider error is a rejected login.

        The SDK answers a 401 by retrying it, silently, for minutes.  A worker
        given a wrong key sat `running` with no progress for three minutes
        while this adapter was already holding the event that said why.  A
        credential is not a transient fault, so the first such event is the
        end of the turn.
        """

        if not isinstance(projection, Mapping):
            return None
        status = projection.get("api_error_status")
        http = (
            status
            if isinstance(status, int) and not isinstance(status, bool)
            else None
        )
        # Compared one by one: a set would hash each value, and a list or an
        # object sent as a code raises there.
        named = any(
            projection.get(key) == "authentication_failed"
            for key in ("code", "terminal_reason", "subtype")
        )
        if not named and http not in (401, 403):
            return None
        if named and http in (401, 403):
            return f"authentication_failed, HTTP {http}"
        return "authentication_failed" if named else f"HTTP {http}"

    def _record_event(self, record: Mapping[str, Any]) -> None:
        event = record.get("event")
        if not isinstance(event, Mapping):
            self._set_fatal("Claude bridge event is not an object")
            return
        name = event.get("name")
        if name not in {
            "system", "result", "tool_use", "tool_result", "permission", "tool_call",
            "message", "usage", "stream", "native_child", "native_child_identity",
            "native_child_usage", "native_child_completed", "native_child_stop", "native_child_control", "provider_error",
            "native_child_counters",
            "stream_deltas_dropped",
        }:
            self._set_fatal("Claude bridge emitted an unsupported event")
            return
        self._bind_emitted_identity(event)
        if name in {"tool_use", "tool_result"}:
            reservation_id = event.get("reservation_id")
            state = self._threads.get(reservation_id) if isinstance(reservation_id, str) else None
            if not isinstance(state, Mapping) or not isinstance(state.get("provider_session"), str):
                self._set_fatal("Claude bridge emitted an effect before native identity binding")
                return
        if name == "tool_call":
            # A manager tool call is control plane, not an observed effect: it
            # must never reach the effect stream that receipts are built from.
            # It carries the model the worker's replies named so far, which a
            # completion written mid-turn needs.
            self._note_model_ran(event.get("reservation_id"), event.get("model_ran"), event.get("model_ran_first"))
            self._handle_tool_call(event)
            return
        if name == "permission":
            self._handle_permission(event)
        cursor = len(self._events)
        if name == "native_child" and event.get("tracking") == "attested" and event.get("status") == "running":
            accepted = self._observe_native_child_event(event, cursor)
            if not accepted:
                rejected = dict(event)
                rejected.update({"name": "native_child_rejected", "tracking": "unavailable"})
                refusal, self._native_child_refusal = self._native_child_refusal, None
                # Every refusal names a cause.  The checks without a sentence
                # of their own test identity and lifecycle: a turn or thread
                # this child does not belong to, a missing objective, or a
                # task that has already ended.
                rejected["reason"] = refusal or (
                    "the child failed the adapter's identity or lifecycle checks for this worker"
                )
                # A refused child has no agent, so an event still addressed to
                # its runtime thread is held by the collector and never
                # reaches the run record.  The refusal is the evidence an
                # operator needs, so address it to the parent reservation.
                rejected.pop("native_runtime_thread_id", None)
                self._events.append(rejected)
                return
        elif name == "native_child_usage" and not self._record_native_child_usage(event):
            rejected = dict(event)
            rejected["name"] = "native_child_usage_rejected"
            self._events.append(rejected)
            return
        elif name == "native_child_completed" and not self._record_native_child_completion(event):
            rejected = dict(event)
            rejected["name"] = "native_child_completed_rejected"
            self._events.append(rejected)
            return
        elif name == "native_child_stop":
            # A stop hook is an observation, not a provider task outcome.
            with self._condition:
                thread_id = event.get("native_runtime_thread_id")
                task_id = event.get("task_id")
                task = self._native_child_tasks.get(thread_id) if isinstance(thread_id, str) else None
                matched = (
                    isinstance(task_id, str) and event.get("status") == "observed-stop"
                    and event.get("stop_observed") is True
                    and self._native_child_event_matches_task(event, task, task_id)
                )
            if not matched:
                rejected = dict(event)
                rejected["name"] = "native_child_stop_rejected"
                self._events.append(rejected)
                return
        elif name == "native_child_control":
            reservation_id = event.get("reservation_id")
            task_id = event.get("task_id")
            with self._condition:
                matching_tasks = [
                    task for task in self._native_child_tasks.values()
                    if task.get("reservation_id") == reservation_id and task.get("task_id") == task_id
                ]
            if (
                event.get("action") != "stop-requested"
                or not isinstance(reservation_id, str)
                or not isinstance(task_id, str)
                or len(matching_tasks) != 1
            ):
                rejected = dict(event)
                rejected["name"] = "native_child_control_rejected"
                self._events.append(rejected)
                return
        elif name == "provider_error":
            said = self._credential_refusal_detail(event.get("provider_error"))
            if said is not None:
                # Keep the event -- it is the evidence -- then end every wait
                # on this adapter.  The scheduler fails the worker with this
                # reason and releases the runtime, which stops the provider
                # process.  Closing here would be the reader thread joining
                # itself.
                # The sentence is written here rather than returned by the
                # helper so the source scan in test_vnext_diagnostics can read
                # its fixed head and confirm the failure classifies.
                self._events.append(dict(event))
                # One call per provider, each with its sentence inline, so the
                # scan reads both heads.
                if self.provider == "zai":
                    location = f"~/{PROVIDER_CONFIG_PATH.parent.name}/providers.json"
                    self._set_fatal(
                        f"Z.ai provider refused the credential ({said}) -- "
                        f"retrying will not help; correct the key in {location} "
                        "and delegate again"
                    )
                else:
                    self._set_fatal(
                        f"Claude provider refused the credential ({said}) -- "
                        "retrying will not help; run claude auth login and delegate again"
                    )
                return
        self._events.append(dict(event))

    def _bind_emitted_identity(self, event: Mapping[str, Any]) -> None:
        """Bind exactly once from a bridge-validated SDK message event."""

        reservation_id = event.get("reservation_id")
        generation = event.get("generation")
        identity = event.get("native_identity")
        if identity is None:
            return
        if not isinstance(reservation_id, str) or not isinstance(generation, int) or not isinstance(identity, Mapping):
            self._set_fatal("Claude native identity event lacks local generation correlation")
            return
        session_id = identity.get("session_id")
        if (
            not isinstance(session_id, str)
            or not session_id
            or session_id == "default"
            or identity.get("source") not in {"AssistantMessage", "ResultMessage"}
        ):
            self._set_fatal("Claude bridge emitted an invalid native session identity")
            return
        state = self._threads.get(reservation_id)
        if state is None or state["generation"] != generation:
            self._set_fatal("Claude native identity event is stale or unknown")
            return
        current = state["provider_session"]
        if current is not None and current != session_id:
            self._set_fatal("Claude bridge emitted a conflicting native session identity")
            return
        state["provider_session"] = session_id
        state["binding_phase"] = "attested"

    def _handle_tool_call(self, event: Mapping[str, Any]) -> None:
        """Answer one bridge-hosted manager tool call on its exact call id.

        Deliberately symmetric with :meth:`_handle_permission`: the bridge
        blocks its in-process SDK tool handler on a future, this adapter runs
        the caller's handler, and the answer returns as one correlated
        ``tool_call_response`` control record.  No second style of rendezvous
        is introduced.
        """

        reservation_id = event.get("reservation_id")
        turn_reference = event.get("turn_reference")
        call_id = event.get("call_id")
        tool = event.get("tool")
        arguments = event.get("arguments")
        if not all(
            isinstance(value, str) and value
            for value in (reservation_id, turn_reference, call_id, tool)
        ) or not isinstance(arguments, Mapping):
            self._set_fatal("Claude tool call lacks local correlation")
            return
        child_runtime = event.get("native_runtime_thread_id")
        child_task_id = event.get("native_child_task_id")
        thread_id = reservation_id
        if child_runtime is not None or child_task_id is not None:
            if not isinstance(child_runtime, str) or not isinstance(child_task_id, str):
                self._set_fatal("Claude native child tool call lacks exact task correlation")
                return
            with self._condition:
                child = self._native_child_tasks.get(child_runtime)
            if child is not None and child.get("status") != "running":
                # A late provider callback cannot reopen a terminal native
                # task or call its scheduler-bound dynamic tool handler.
                return
            if (
                child is None
                or child.get("reservation_id") != reservation_id
                or child.get("task_id") != child_task_id
                or turn_reference != child_task_id
                or child.get("status") != "running"
            ):
                self._set_fatal("Claude native child tool call is not attested")
                return
            thread_id = child_runtime
        if (thread_id, turn_reference) not in self._turns:
            self._set_fatal("Claude tool call has no active local turn")
            return
        handler = self._tool_handlers.get(thread_id)
        if handler is None:
            self._set_fatal("Claude tool call has no registered caller tool handler")
            return
        context = ToolCallContext(
            thread_id=thread_id, turn_id=turn_reference, call_id=call_id
        )
        try:
            result = handler(str(tool), dict(arguments), context)
            if not isinstance(result, ToolCallResult):
                raise ClaudeRuntimeError("caller tool handler returned an unsupported result")
            payload = result.as_dict()
        except Exception as exc:
            # A tool handler hosted inside the provider SDK is a blind spot:
            # the SDK catches the exception, shows the model an error result,
            # and Python never sees it.  Recording it here is the only way a
            # control-plane defect leaves any trace at all.
            self._record_failure(
                FailureCategory.MANAGER_TOOL_HANDLER_FAILED,
                step="tool_call",
                exception_type=type(exc).__name__,
            )
            payload = {"success": False, "value": {"error_code": "tool-handler-failed"}}
        self._send(
            {
                "v": self._VERSION,
                "kind": "control",
                "op": "tool_call_response",
                "reservation_id": reservation_id,
                "turn_reference": turn_reference,
                "call_id": call_id,
                "result": payload,
            }
        )

    def _handle_permission(self, event: Mapping[str, Any]) -> None:
        tool_use_id = event.get("tool_use_id")
        reservation_id, turn_id = event.get("reservation_id"), event.get("turn_reference")
        # Transitional deterministic fixtures emitted their old opaque turn
        # value only.  Translate it to the one active local reservation, never
        # to a provider session identity.
        if not isinstance(reservation_id, str) or not isinstance(turn_id, str):
            legacy_turn_id = event.get("turn_id")
            candidates = [key for key in self._turns if key[1] == legacy_turn_id]
            if len(candidates) == 1:
                reservation_id, turn_id = candidates[0]
        if not all(isinstance(value, str) and value for value in (tool_use_id, reservation_id, turn_id)):
            self._set_fatal("Claude permission event lacks local correlation")
            return
        if (reservation_id, turn_id) not in self._turns:
            self._set_fatal("Claude permission event does not match an active local turn")
            return
        state = self._threads.get(reservation_id)
        if state is None:
            self._set_fatal("Claude permission event lacks a local reservation")
            return
        provider_session = state.get("provider_session")
        envelope: dict[str, Any] = {
            "approval_reference": f"claude:{tool_use_id}",
            "provider": self.provider,
            "event": dict(event),
        }
        # The neutral approval envelope carries the requested effect at its top
        # level; the reviewer reads `params["effect"]`, never a provider-shaped
        # nested payload.  The Codex boundary already publishes it that way.
        # Emitting it only for the neutral vocabulary keeps the seam fail-closed:
        # an effect this boundary cannot name stays absent and the reviewer
        # declines, rather than being forwarded as an unrecognised value.
        effect = event.get("effect")
        if effect in _NEUTRAL_APPROVAL_EFFECTS:
            envelope["effect"] = effect
        if isinstance(provider_session, str) and provider_session:
            correlation = event.get("provider_correlation")
            if (
                event.get("correlation_attested") is not True
                or not isinstance(correlation, Mapping)
                or dict(correlation) != {
                    "session": provider_session,
                    "turn": turn_id,
                    "request": tool_use_id,
                }
                or "routing_handle" in event
            ):
                self._set_fatal("Claude bound permission event lacks attested provider correlation")
                return
            envelope["provider_correlation"] = dict(correlation)
            envelope["correlation_attested"] = True
        else:
            routing_handle = event.get("routing_handle")
            expected_handle = {"reservation_id": reservation_id, "turn_reference": turn_id}
            if routing_handle is not None and (
                not isinstance(routing_handle, Mapping)
                or dict(routing_handle) != expected_handle
                or event.get("correlation_attested") is not False
                or "provider_correlation" in event
            ):
                self._set_fatal("Claude unbound permission event lacks a local routing handle")
                return
            envelope["routing_handle"] = expected_handle
            envelope["correlation_attested"] = False
        # A decline travels with the reason the reviewer gave for it.  The
        # worker used to read one fixed sentence blaming its manager, even on
        # the paths where no manager was ever asked.
        decision: Mapping[str, Any] = {
            "decision": "decline",
            "reason": _NO_REVIEWER_DECLINE_REASON,
        }
        if self._native_approval_handler is not None:
            # The bridge owns the provider identifiers.  Present one neutral
            # envelope to the scheduler instead of masquerading a Claude tool
            # request as a Codex app-server request.
            candidate = self._native_approval_handler("approval/request", envelope)
            if isinstance(candidate, Mapping) and candidate.get("decision") == "accept":
                decision = {"decision": "accept"}
            elif isinstance(candidate, Mapping):
                reason = candidate.get("reason")
                decision = {
                    "decision": "decline",
                    "reason": (
                        reason
                        if isinstance(reason, str) and reason.strip()
                        else _UNNAMED_DECLINE_REASON
                    ),
                }
        self._send({"v": self._VERSION, "kind": "control", "op": "permission_response", "reservation_id": reservation_id, "turn_reference": turn_id, "tool_use_id": tool_use_id, "decision": dict(decision)})
