"""Standalone asynchronous Codex app-server transport for vNext canaries.

This module is intentionally separate from the v0.5 client.  It owns JSON-RPC
mechanics, thread-scoped dynamic tool routing, event waits, and process cleanup;
manager and Worker prompts remain the orchestration policy surface.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Protocol, Sequence

from vnext.child_environment import child_process_environment
from vnext.vnext_clock_hook_config import (
    hook_cli_args,
    hook_command_string,
    hook_thread_config,
    merge_config,
    trusted_hash_from_hooks_list,
)
from vnext.process_supervisor import OwnedProcess, ProcessCleanup
from vnext.vnext_model_identity import remember
from vnext.vnext_debug_transcript import ScopedDebugEvent
from vnext.vnext_runtime_types import (
    CREDENTIAL_ADJACENT_STDERR_PROVIDERS,
    DynamicToolHandler,
    NativeChildBinding,
    NativeChildObservation,
    NativeChildObserver,
    NativeParentAgentResolver,
    NativeApprovalHandler,
    RuntimeCleanup,
    RuntimePosture,
    ToolCallContext,
    ToolCallResult,
    TurnHandle,
)

if TYPE_CHECKING:
    from vnext.vnext_debug_transcript import VNextDebugTranscript


class VNextAppServerError(RuntimeError):
    pass


class AppServerTransport(Protocol):
    """A JSON-RPC line transport used by the app-server lifecycle methods."""

    @property
    def closed(self) -> bool: ...

    def recv(self) -> str | None: ...

    def send(self, payload: str) -> None: ...

    def close(self) -> None: ...


LOCAL_ENVIRONMENT_ID = "local"
WINDOWS_SANDBOX_MODE = "elevated"
WINDOWS_SANDBOX_FEATURE = "elevated_windows_sandbox"
_MAX_NATIVE_CHILD_DISCOVERY_PARENTS = 4
_NATIVE_CHILD_DISCOVERY_TOTAL_SECONDS = 15.0
# What the clock hook's command string advertises as the budget.  The real
# number for a turn is written into the state file when that turn starts and
# wins over this one; this is the fallback for a turn vNext never recorded.
DEFAULT_CLOCK_LIMIT = 1800


# A linked worktree keeps its index, objects, branch ref and reflog in the
# parent repository, outside its cwd, so a child whose only root is its
# checkout cannot commit.  A no-model probe on Codex 0.144.4
# (macOS, 2026-09-22) settled which roots are enough: granting ``.git`` itself
# still fails, because the runtime keeps any ``.git`` directory read-only;
# the worktree gitdir alone fails on the object store; the gitdir plus the
# object store plus the session's branch-ref and reflog directories commits.
# Git then prints a harmless error about ``packed-refs.lock`` and the commit
# lands.  The grant is limited to vNext's own ``vnext/`` branches, so a user's
# worktree on any other branch keeps exactly the root it had.
def _runtime_workspace_roots(cwd: Path) -> list[str]:
    """Return the workspace and, for vNext worktrees, writable Git roots."""

    roots = [cwd.resolve()]
    dot_git = cwd / ".git"
    try:
        if not dot_git.is_file():
            return [str(roots[0])]
        gitdir_line = dot_git.read_text(encoding="utf-8").strip()
        if not gitdir_line.startswith("gitdir:"):
            return [str(roots[0])]
        gitdir_value = gitdir_line.removeprefix("gitdir:").strip()
        if not gitdir_value:
            return [str(roots[0])]
        gitdir = Path(gitdir_value)
        if not gitdir.is_absolute():
            gitdir = cwd / gitdir
        gitdir = gitdir.resolve()
        head = (gitdir / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref: refs/heads/vnext/"):
            return [str(roots[0])]
        branch = head.removeprefix("ref: ")
        if not branch or branch.endswith("/"):
            return [str(roots[0])]
        commondir_value = (gitdir / "commondir").read_text(encoding="utf-8").strip()
        if not commondir_value:
            return [str(roots[0])]
        commondir = Path(commondir_value)
        if not commondir.is_absolute():
            commondir = gitdir / commondir
        commondir = commondir.resolve()
        branch_ref = Path(branch)
        extra = (
            gitdir,
            commondir / "objects",
            commondir / branch_ref.parent,
            commondir / "logs" / branch_ref.parent,
        )
        roots.extend(path.resolve() for path in extra if path.is_dir())
    except (OSError, UnicodeError, ValueError):
        return [str(roots[0])]
    return [str(path) for path in roots]


# Compatibility name for the existing Codex app-server surface.  The value is
# deliberately provider-neutral so leaf runtimes do not import this module.
AppServerCleanup = RuntimeCleanup


def text_input(text: str) -> list[dict[str, str]]:
    return [{"type": "text", "text": text}]


def project_tool_result(result: Any) -> dict[str, Any]:
    """Project a neutral tool result into the Codex app-server wire shape.

    The control plane answers a tool call with a :class:`ToolCallResult`; the
    Codex boundary is the only place that knows those answers travel as
    ``contentItems``.  A handler that already produced a raw app-server payload
    (deterministic fixtures do) is passed through unchanged.
    """

    if isinstance(result, ToolCallResult):
        return {
            "success": bool(result.success),
            "contentItems": [{"type": "inputText", "text": result.as_json_text()}],
        }
    if isinstance(result, Mapping):
        return dict(result)
    raise VNextAppServerError("dynamic tool handler returned an unsupported result")


def function_tool(
    name: str,
    description: str,
    *,
    properties: Mapping[str, Any],
    required: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": dict(properties),
            "required": list(required),
            "additionalProperties": False,
        },
    }


class VNextAppServerAdapter:
    """Concurrent stdio JSON-RPC adapter with per-thread tool handlers."""

    provider = "codex"

    def __init__(
        self,
        *,
        codex_executable: str | Path,
        codex_home: str | Path,
        workspace: str | Path,
        provider: str = "codex",
        client_name: str = "vnext_adapter",
        mcp_config_overrides: Mapping[str, Any] | None = None,
        mcp_startup_config_overrides: Mapping[str, str] | None = None,
        native_approval_handler: NativeApprovalHandler | None = None,
        debug_transcript: VNextDebugTranscript | None = None,
        transport: AppServerTransport | None = None,
    ) -> None:
        self.codex_executable = Path(codex_executable).resolve()
        self.codex_home = Path(codex_home).resolve()
        self.workspace = Path(workspace).resolve()
        self.provider = provider
        if not self.codex_executable.is_file():
            raise ValueError("Codex executable does not exist")
        if not self.codex_home.is_dir() or not self.workspace.is_dir():
            raise ValueError("Codex home and workspace must exist")
        self.client_name = client_name
        self.mcp_config_overrides = dict(mcp_config_overrides or {})
        self.mcp_startup_config_overrides = self._validated_mcp_startup_overrides(
            mcp_startup_config_overrides or {}
        )
        self.native_approval_handler = native_approval_handler
        self.debug_transcript = debug_transcript
        # Where this adapter writes each turn's start time and budget for the
        # Codex clock hook to read.  The hook gets neither from Codex: its
        # stdin carries session_id, turn_id, cwd, transcript_path, model,
        # tool_name, tool_input and tool_response, and the app-server
        # environment is allowlisted.  The directory is this adapter's own and
        # is removed on close.
        self._clock_state_dir = tempfile.mkdtemp(prefix="vnext-clock-")
        self._clock_state_path = Path(self._clock_state_dir) / "turns.json"
        self._clock_state_lock = threading.Lock()
        self._clock_command = hook_command_string(
            python=sys.executable,
            script=Path(__file__).with_name("clock_hook.py"),
            state_path=self._clock_state_path,
            limit=DEFAULT_CLOCK_LIMIT,
        )
        self._clock_trusted_hash: str | None = None
        self._clock_hash_attempted = False
        self._clock_disabled_reason: str | None = None
        self._condition = threading.Condition(threading.RLock())
        self._send_lock = threading.Lock()
        self._next_id = 1
        self._responses: dict[int, dict[str, Any]] = {}
        self._events: list[dict[str, Any]] = []
        self._tool_handlers: dict[str, DynamicToolHandler] = {}
        self._tool_registrations: dict[str, dict[str, Any]] = {}
        self._manager_relay_name: str | None = None
        self._managed_thread_mcp: dict[str, dict[str, Any]] = {}
        self._managed_mcp_origins: dict[tuple[str, str, str], None] = {}
        self._native_thread_attestations: dict[str, dict[str, str]] = {}
        self._native_child_observer: NativeChildObserver | None = None
        self._native_parent_agent_resolver: NativeParentAgentResolver | None = None
        self._native_child_observations: dict[tuple[str, str, str], NativeChildObservation] = {}
        self._native_child_bindings: dict[str, NativeChildBinding] = {}
        self._native_child_hints: dict[str, dict[str, Any]] = {}
        self._native_child_start_events: dict[str, tuple[Mapping[str, Any], int]] = {}
        self._native_child_failures: dict[str, str] = {}
        self._native_child_discovery_in_flight: set[str] = set()
        self._native_child_discovery_cursor = 0
        self._tool_calls: list[dict[str, Any]] = []
        self._handler_threads: set[threading.Thread] = set()
        self._stderr_lines: deque[str] = deque(maxlen=64)
        self._fatal: str | None = None
        self._closing = False
        self._cleanup: AppServerCleanup | None = None
        self._platform_name = os.name
        self._windows_sandbox_lock = threading.Lock()
        self._windows_sandbox_ready = False
        self._windows_sandbox_attestation: dict[str, Any] = {
            "platform_windows": self._platform_name == "nt",
            "feature": WINDOWS_SANDBOX_FEATURE,
            "feature_enabled": None,
            "setup_attempted": False,
            "setup_completed": False,
            "skip_reason": None,
        }

        self._transport = transport
        self.supervisor: OwnedProcess | None = None
        self.process: subprocess.Popen[str] | None = None
        if transport is not None:
            self._dispatcher = threading.Thread(
                target=self._dispatch_transport,
                name="vnext-app-server-remote-dispatch",
                daemon=True,
            )
            self._stderr_reader = threading.Thread(target=lambda: None, daemon=True)
            self._dispatcher.start()
            self._stderr_reader.start()
            return

        environment = self._environment()
        self.supervisor = OwnedProcess.start(
            [
                str(self.codex_executable),
                "--strict-config",
                "--disable",
                "multi_agent",
                *hook_cli_args(self._clock_command),
                *self._mcp_startup_cli_args(),
                "app-server",
                "--listen",
                "stdio://",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=environment,
            cwd=str(self.workspace),
        )
        self.process = self.supervisor.process
        self._dispatcher = threading.Thread(
            target=self._dispatch_stdout,
            name="vnext-app-server-dispatch",
            daemon=True,
        )
        self._stderr_reader = threading.Thread(
            target=self._read_stderr,
            name="vnext-app-server-stderr",
            daemon=True,
        )
        self._dispatcher.start()
        self._stderr_reader.start()

    def initialize(self, *, timeout: float = 30) -> dict[str, Any]:
        result = self.request(
            "initialize",
            {
                "clientInfo": {
                    "name": self.client_name,
                    "title": "vNext Adapter",
                    "version": "0.1",
                },
                "capabilities": {"experimentalApi": True},
            },
            timeout=timeout,
        )
        self.notify("initialized", {})
        self._load_model_list(timeout=min(timeout, 20))
        return result

    def _load_model_list(self, *, timeout: float) -> None:
        """Ask this app-server once which exact model each catalog id means.

        It rides the connection vNext already owns.  A failure is recorded as
        a sentence and the session goes on: the worker's own thread/start
        answer still names the model that ran.
        """

        catalog: dict[str, str] = {}
        error: str | None = None
        cursor: str | None = None
        try:
            for _page in range(50):
                params: dict[str, Any] = {"includeHidden": True}
                if cursor:
                    params["cursor"] = cursor
                answer = self.request("model/list", params, timeout=timeout)
                for item in answer.get("data") or ():
                    if isinstance(item, Mapping) and isinstance(item.get("id"), str) and isinstance(item.get("model"), str):
                        catalog[item["id"]] = item["model"]
                cursor = answer.get("nextCursor")
                if not isinstance(cursor, str) or not cursor:
                    break
        except Exception as exc:  # noqa: BLE001 - the run continues without it
            text = str(exc) or type(exc).__name__
            error = text if text.startswith("model/list failed") else f"model/list failed: {text}"
        self._model_catalog = catalog
        self._model_list_error = error
        provider = str(getattr(self, "provider", "codex"))
        for alias, exact in catalog.items():
            remember(provider, alias, exact, "model_list")

    def _identities(self) -> dict[str, dict[str, Any]]:
        identities = getattr(self, "_model_identities", None)
        if identities is None:
            identities = self._model_identities = {}
        return identities

    def _note_thread_model(self, thread_id: str, model: str, result: Mapping[str, Any], *, resumed: bool) -> None:
        catalog = getattr(self, "_model_catalog", None) or {}
        error = getattr(self, "_model_list_error", None)
        with self._condition:
            identity = self._identities().setdefault(thread_id, {})
            if not resumed or not identity.get("model_exact"):
                exact = catalog.get(model)
                identity["model_exact"] = exact
                identity["model_exact_source"] = "model_list" if exact else None
                if exact is None and error:
                    identity["model_exact_error"] = error
            ran = result.get("model")
            if isinstance(ran, str) and ran:
                identity["model_ran"] = ran
                identity.setdefault("model_ran_first", ran)

    def _observe_model_reroute(self, params: Mapping[str, Any]) -> None:
        thread_id = params.get("threadId")
        to_model = params.get("toModel")
        if not isinstance(thread_id, str) or not isinstance(to_model, str) or not to_model:
            return
        with self._condition:
            identity = self._identities().setdefault(thread_id, {})
            identity["model_ran"] = to_model
            identity.setdefault("model_reroutes", []).append({
                "from": params.get("fromModel"),
                "to": to_model,
                "reason": params.get("reason"),
                "turn_id": params.get("turnId"),
            })

    def model_identity(self, thread_id: str) -> Mapping[str, Any]:
        """The exact model this thread asked for and the one that answered."""

        with self._condition:
            identity = self._identities().get(thread_id)
            if not identity:
                return {}
            copy = dict(identity)
            if isinstance(copy.get("model_reroutes"), list):
                copy["model_reroutes"] = [dict(item) for item in copy["model_reroutes"]]
            return copy

    def request(self, method: str, params: Mapping[str, Any], *, timeout: float = 30) -> dict[str, Any]:
        with self._condition:
            self._raise_if_unavailable()
            request_id = self._next_id
            self._next_id += 1
        self._send({"method": method, "id": request_id, "params": dict(params)})
        deadline = time.monotonic() + timeout
        with self._condition:
            while request_id not in self._responses:
                self._raise_if_unavailable()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise VNextAppServerError(f"app-server request timed out: {method}")
                self._condition.wait(timeout=min(remaining, 0.5))
            response = self._responses.pop(request_id)
        if response.get("error"):
            raise VNextAppServerError(f"{method} failed: {self._safe_error(response['error'])}")
        result = response.get("result")
        return result if isinstance(result, dict) else {}

    def notify(self, method: str, params: Mapping[str, Any]) -> None:
        self._send({"method": method, "params": dict(params)})

    def _clock_hook_config(self, cwd: Path, *, timeout: float) -> dict[str, Any]:
        """Return the hooks table for thread/start, or an empty map.

        The trust hash is read from ``hooks/list`` at runtime.  It covers the
        event, the matcher and the normalized command string, so it changes
        whenever the command does and cannot be pinned in source.  When the
        fetch fails the thread starts with no hook and the reason is recorded
        once: a clock must never stop a worker.
        """

        command = getattr(self, "_clock_command", None)
        if not command:
            return {}
        if self._clock_trusted_hash is None and not self._clock_hash_attempted:
            self._clock_hash_attempted = True
            try:
                listed = self.request("hooks/list", {"cwds": [str(cwd)]}, timeout=timeout)
                self._clock_trusted_hash = trusted_hash_from_hooks_list(listed, command)
                if self._clock_trusted_hash is None:
                    self._clock_disabled_reason = "hooks/list reported no clock hook"
            except Exception as exc:
                self._clock_disabled_reason = f"hooks/list failed: {exc}"
        if self._clock_trusted_hash is None:
            return {}
        return hook_thread_config(command, self._clock_trusted_hash)

    def clock_hook_attestation(self) -> dict[str, Any]:
        """Content-safe facts about this adapter's clock hook."""

        return {
            "state_path": str(getattr(self, "_clock_state_path", "")),
            "trusted_hash": getattr(self, "_clock_trusted_hash", None),
            "disabled_reason": getattr(self, "_clock_disabled_reason", None),
        }

    def record_turn_clock(self, thread_id: str, *, started: float, limit: float | None, started_monotonic: float | None = None) -> None:
        """Write one turn's start time and budget where the hook will find it.

        Written atomically, because the hook reads this file from another
        process while turns on sibling threads are rewriting it.
        """

        path = getattr(self, "_clock_state_path", None)
        if path is None:
            return
        with self._clock_state_lock:
            try:
                existing = json.loads(Path(path).read_text(encoding="utf-8"))
                if not isinstance(existing, dict):
                    existing = {}
            except Exception:
                existing = {}
            existing[str(thread_id)] = {
                "started": float(started),
                "started_monotonic": time.monotonic() if started_monotonic is None else float(started_monotonic),
                "limit": limit,
            }
            try:
                temporary = Path(str(path) + f".{os.getpid()}.tmp")
                temporary.write_text(json.dumps(existing), encoding="utf-8")
                os.replace(temporary, path)
            except Exception:
                # A clock is never worth a turn.
                pass

    def start_thread(
        self,
        *,
        model: str,
        effort: str = "high",
        developer_instructions: str,
        tools: Sequence[Mapping[str, Any]],
        tool_handler: DynamicToolHandler | None = None,
        requested_posture: RuntimePosture,
        workspace: str | Path | None = None,
        timeout: float = 60,
    ) -> tuple[str, dict[str, Any]]:
        # Codex applies reasoning effort when a turn starts.  Accept it here
        # because agent binding carries the selected effort, but do not invent
        # a thread-level setting that the provider does not attest.
        _ = effort
        cwd = Path(workspace or self.workspace).resolve()
        tool_payload = [dict(value) for value in tools]
        native_posture = self._native_start_posture(requested_posture)
        # Some narrow fixture adapters are deliberately constructed without
        # ``__init__``.  Keep the diagnostic-only setting absent in that case.
        mcp_config_overrides = getattr(self, "mcp_config_overrides", {})
        thread_config = merge_config(mcp_config_overrides, self._clock_hook_config(cwd, timeout=timeout))
        result = self.request(
            "thread/start",
            {
                "model": model,
                "cwd": str(cwd),
                **native_posture,
                "permissions": ":workspace",
                "runtimeWorkspaceRoots": _runtime_workspace_roots(cwd),
                "serviceName": self.client_name,
                "developerInstructions": developer_instructions,
                "dynamicTools": tool_payload,
                # Command Code rides this same map and carries the reseller's
                # bearer in it, so the hooks table is merged into whatever is
                # already there.  ``config`` now has to be emitted whenever
                # there are overrides *or* a hook table.
                **({"config": thread_config} if thread_config else {}),
                **({"experimentalRawEvents": True}
                   if getattr(self, "_manager_relay_name", None) and tool_handler is not None else {}),
                "ephemeral": False,
            },
            timeout=timeout,
        )
        result = self._attest_default_environment(result, timeout=timeout)
        result = self._attach_neutral_posture(result)
        try:
            thread_id = str(result["thread"]["id"])
        except (KeyError, TypeError) as exc:
            raise VNextAppServerError("thread/start returned no thread id") from exc
        self._note_thread_model(thread_id, model, result, resumed=False)
        registration = self._tool_registration(model, tool_payload, tool_handler)
        relay_name = getattr(self, "_manager_relay_name", None)
        if relay_name is not None and tool_handler is not None:
            # Fresh threads also inherit the relay. Register their exact owned
            # identity before any turn can use the advertised MCP tools.
            self._attest_manager_relay_inventory(
                thread_id=thread_id, relay_name=relay_name, tools=tool_payload,
                timeout=min(timeout, 30),
            )
        with self._condition:
            if tool_handler is not None:
                self._tool_handlers[thread_id] = tool_handler
            self._native_thread_attestations[thread_id] = {
                "provider": self.provider,
                "provider_session": thread_id,
                "binding_phase": "started",
            }
            self._tool_registrations[thread_id] = registration
            if relay_name is not None and tool_handler is not None:
                self._managed_thread_mcp[thread_id] = {
                    "server_name": relay_name,
                    "tool_handler": tool_handler,
                    "registration": registration,
                }
        return thread_id, result

    @staticmethod
    def _tool_registration(
        model: str,
        tools: Sequence[Mapping[str, Any]],
        tool_handler: DynamicToolHandler | None,
    ) -> dict[str, Any]:
        payload = [dict(value) for value in tools]
        names = [str(value.get("name") or "") for value in payload]
        if any(not name for name in names) or len(set(names)) != len(names):
            raise VNextAppServerError("Codex tool definitions require unique names")
        return {
            "acknowledged": True,
            "model_id": model,
            "tool_count": len(payload),
            "tool_names": names,
            "definition_sha256": hashlib.sha256(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "handler_registered": tool_handler is not None,
        }

    @staticmethod
    def _validated_mcp_startup_overrides(overrides: Mapping[str, str]) -> dict[str, str]:
        """Accept only the relay URL and bearer fields passed to the Codex CLI.

        Startup configuration is process-local.  It deliberately does not write
        ``config.toml`` and it is separate from the legacy per-thread config
        override used by ordinary dynamic-tool starts.
        """

        values: dict[str, str] = {}
        for key, value in overrides.items():
            if (
                not isinstance(key, str)
                or not isinstance(value, str)
                or not value
                or not (
                    key.startswith("mcp_servers.")
                    and (
                        key.endswith(".url")
                        or key.endswith(".http_headers.Authorization")
                    )
                )
            ):
                raise ValueError("Codex startup MCP overrides require relay URL or bearer values")
            values[key] = value
        return values

    def _mcp_startup_cli_args(self) -> list[str]:
        """Render trusted string settings as separate CLI arguments, never a shell command."""

        return [
            value
            for key, setting in sorted(self.mcp_startup_config_overrides.items())
            for value in ("-c", f"{key}={json.dumps(setting, ensure_ascii=False)}")
        ]

    @staticmethod
    def _native_start_posture(requested: RuntimePosture) -> dict[str, str]:
        """Translate a neutral requested posture at the Codex transport edge."""

        if (
            requested.workspace_writes is not True
            or requested.network not in {"restricted", "local_only", False}
            or requested.approvals_requested is not True
            or not isinstance(requested.reviewer, str)
            or not requested.reviewer.strip()
            or requested.reviewer.strip() != requested.reviewer
            or requested.environment_ready is not True
        ):
            raise VNextAppServerError("requested runtime posture is not reviewable")
        return {
            "approvalPolicy": "on-request",
            "approvalsReviewer": requested.reviewer,
        }

    def tool_registration_attestation(self, thread_id: str) -> dict[str, Any]:
        """Return content-safe facts for an acknowledged dynamic-tool registration."""

        with self._condition:
            value = self._tool_registrations.get(thread_id)
            if value is None:
                raise VNextAppServerError("no dynamic tool registration is attested for the thread")
            return dict(value)

    def tool_call_attestations(self, thread_id: str) -> tuple[dict[str, Any], ...]:
        """Return content-safe app-server call facts without arguments or private ids."""

        with self._condition:
            return tuple(
                {
                    key: value
                    for key, value in item.items()
                    if key != "thread_id"
                }
                for item in self._tool_calls
                if item["thread_id"] == thread_id
            )

    def start_turn(
        self,
        *,
        thread_id: str,
        prompt: str,
        model: str,
        effort: str,
        approvals_reviewer: str = "auto_review",
        workspace: str | Path | None = None,
        timeout: float = 60,
        turn_timeout: float | None = None,
    ) -> TurnHandle:
        cwd = Path(workspace or self.workspace).resolve()
        self.ensure_windows_sandbox_ready(workspace=cwd)
        self.record_turn_clock(thread_id, started=time.time(), limit=turn_timeout)
        with self._condition:
            cursor = len(self._events)
        result = self.request(
            "turn/start",
            {
                "threadId": thread_id,
                "input": text_input(prompt),
                "model": model,
                "effort": effort,
                "cwd": str(cwd),
                "approvalPolicy": "on-request",
                "approvalsReviewer": approvals_reviewer,
                "runtimeWorkspaceRoots": _runtime_workspace_roots(cwd),
            },
            timeout=timeout,
        )
        try:
            turn_id = str(result["turn"]["id"])
        except (KeyError, TypeError) as exc:
            raise VNextAppServerError("turn/start returned no turn id") from exc
        return TurnHandle(thread_id=thread_id, turn_id=turn_id, cursor=cursor)

    def wait_turn(self, handle: TurnHandle, *, timeout: float = 300) -> dict[str, Any]:
        event = self.wait_event(
            lambda value: value.get("method") == "turn/completed"
            and str(((value.get("params") or {}).get("turn") or {}).get("id")) == handle.turn_id,
            cursor=handle.cursor,
            timeout=timeout,
            progress=lambda value: self._belongs_to_turn(value, handle),
        )
        turn = dict((event.get("params") or {}).get("turn") or {})
        turn_events = self.events_since(handle.cursor)
        final_messages: list[str] = []
        for candidate in turn_events:
            if candidate is event or candidate.get("method") != "item/completed":
                continue
            params = candidate.get("params") or {}
            candidate_thread = str(params.get("threadId") or "")
            candidate_turn = str(params.get("turnId") or "")
            if candidate_thread and candidate_thread != handle.thread_id:
                continue
            if candidate_turn and candidate_turn != handle.turn_id:
                continue
            item = params.get("item") or {}
            if item.get("type") == "agentMessage" and isinstance(item.get("text"), str):
                final_messages.append(item["text"])
        turn["final_response"] = final_messages[-1] if final_messages else ""
        debug_transcript = getattr(self, "debug_transcript", None)
        if debug_transcript is not None:
            debug_transcript.record_turn(
                handle,
                tuple(
                    ScopedDebugEvent(candidate, is_current_turn=True)
                    for candidate in turn_events
                    if self._is_turn_scoped_item_event(candidate, handle)
                ),
            )
        return turn

    @staticmethod
    def _belongs_to_turn(event: Mapping[str, Any], handle: TurnHandle) -> bool:
        """True when this event is the handle's own turn still working.

        Used as the liveness signal for the turn wait, so anything the turn
        emits counts -- items, deltas, token usage, diffs.  Scoping it to the
        turn matters: a shared app-server carries every agent's traffic, and a
        busy sibling must not keep a genuinely stuck turn alive.
        """

        params = event.get("params")
        params = params if isinstance(params, Mapping) else {}
        event_turn = params.get("turnId")
        if event_turn is not None:
            return str(event_turn) == handle.turn_id
        event_thread = params.get("threadId")
        if event_thread is not None:
            return str(event_thread) == handle.thread_id
        return False

    @staticmethod
    def _is_turn_scoped_item_event(
        event: Mapping[str, Any], handle: TurnHandle
    ) -> bool:
        """Keep raw Codex correlation decoding at the app-server boundary."""

        if not str(event.get("method") or "").startswith("item/"):
            return False
        params = event.get("params")
        params = params if isinstance(params, Mapping) else {}
        event_thread = params.get("threadId")
        event_turn = params.get("turnId")
        if event_thread is not None and str(event_thread) != handle.thread_id:
            return False
        if event_turn is not None and str(event_turn) != handle.turn_id:
            return False
        return event_thread is not None or event_turn is not None

    def steer(self, handle: TurnHandle, text: str, *, timeout: float = 30) -> dict[str, Any]:
        return self.request(
            "turn/steer",
            {
                "threadId": handle.thread_id,
                "expectedTurnId": handle.turn_id,
                "input": text_input(text),
            },
            timeout=timeout,
        )

    def interrupt(self, handle: TurnHandle, *, timeout: float = 30) -> dict[str, Any]:
        return self.request(
            "turn/interrupt",
            {"threadId": handle.thread_id, "turnId": handle.turn_id},
            timeout=timeout,
        )

    def active_native_child_turn(
        self, native_child_thread_id: str, *, timeout: float = 30
    ) -> TurnHandle | None:
        """Read the one exact active provider turn for an observed child.

        A provider can create a child and start its first turn between adapter
        event polls.  The child is constrained to a provider-attested
        observation/binding on this adapter; its active turn ID comes only
        from a fresh ``thread/read``.  A completion race returns ``None``
        rather than guessing an old turn ID.
        """

        if not native_child_thread_id:
            raise ValueError("native_child_thread_id is required")
        with self._condition:
            known_child = any(
                observation.native_child_thread_id == native_child_thread_id
                for observation in self._native_child_observations.values()
            )
        if not known_child:
            raise VNextAppServerError("native child is not attested on this adapter")
        payload = self.read_thread(native_child_thread_id, include_turns=True, timeout=timeout)
        thread = payload.get("thread")
        if not isinstance(thread, Mapping) or thread.get("id") != native_child_thread_id:
            raise VNextAppServerError("native child read did not attest the requested thread")
        turns = thread.get("turns")
        active = [
            turn for turn in turns if isinstance(turn, Mapping)
            and turn.get("status") == "inProgress"
            and isinstance(turn.get("id"), str) and turn.get("id")
        ] if isinstance(turns, list) else []
        if not active:
            return None
        if len(active) != 1:
            raise VNextAppServerError("native child read exposed multiple active turns")
        turn_id = str(active[0]["id"])
        return TurnHandle(native_child_thread_id, turn_id)

    def dispatch_native_mcp_call(
        self,
        *,
        tool: str,
        arguments: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> ToolCallResult | Mapping[str, Any]:
        """Route one private-MCP call only to its attested native child.

        Codex writes ``_meta.threadId`` for normal model-originated MCP calls.
        The relay supplies that metadata separately from model-controlled tool
        arguments, and this method joins it to this adapter's observed child
        edge before reading the exact active provider turn.
        """

        thread_id = metadata.get("threadId")
        if not isinstance(thread_id, str) or not thread_id:
            raise VNextAppServerError("native MCP call lacks host threadId metadata")
        if not isinstance(tool, str) or not tool:
            raise VNextAppServerError("native MCP tool is invalid")
        if not isinstance(arguments, Mapping):
            raise VNextAppServerError("native MCP arguments are invalid")
        with self._condition:
            binding = self._native_child_bindings.get(thread_id)
            observed = any(
                observation.native_child_thread_id == thread_id
                for observation in self._native_child_observations.values()
            )
        if not observed or binding is None:
            # A native child can invoke configured MCP before this observer
            # connection receives its ``thread/started`` broadcast. The MCP
            # metadata gives the exact child thread but is not itself an
            # attestation, so recover only through the provider's direct-child
            # inventory for already attested parents.
            self._recover_native_child_mcp_binding(thread_id)
        with self._condition:
            binding = self._native_child_bindings.get(thread_id)
            observed = any(
                observation.native_child_thread_id == thread_id
                for observation in self._native_child_observations.values()
            )
            handler = self._tool_handlers.get(thread_id)
            cursor = len(getattr(self, "_events", ()))
        if not observed or binding is None:
            raise VNextAppServerError("native MCP thread is not an attested child")
        if binding.native_thread_id != thread_id or binding.tool_handler is None or handler is not binding.tool_handler:
            raise VNextAppServerError("native MCP child has no registered handler")
        handle = self.active_native_child_turn(thread_id)
        if handle is None:
            raise VNextAppServerError("native MCP child has no active exact turn")
        call_id = metadata.get("itemId")
        context = ToolCallContext(
            thread_id=thread_id,
            turn_id=handle.turn_id,
            call_id=call_id if isinstance(call_id, str) else "",
            cursor=cursor,
        )
        return handler(tool, dict(arguments), context)

    def _recover_native_child_mcp_binding(self, native_child_thread_id: str) -> None:
        """Refresh direct descendants before rejecting an unobserved child call.

        This is intentionally a narrow recovery path for a configured private
        MCP relay. It never promotes the caller's metadata to an identity: a
        child becomes routable only if ``thread/list(parentThreadId=...)``
        supplies the same edge and the installed observer accepts the binding.
        """

        deadline = time.monotonic() + _NATIVE_CHILD_DISCOVERY_TOTAL_SECONDS
        for parent_thread_id in self._native_managed_parent_candidates(exclude={native_child_thread_id}):
            if time.monotonic() >= deadline:
                return
            with self._condition:
                in_flight = getattr(self, "_native_child_discovery_in_flight", set())
                # The background poll holds this mark for a whole thread/list
                # round trip, longest when the app-server is slow.  Skipping
                # the parent refused a child that the running discovery, or a
                # fresh one, would have found.  So wait for it, then look again.
                while parent_thread_id in in_flight:
                    if native_child_thread_id in self._native_child_bindings:
                        return
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return
                    self._condition.wait(timeout=min(0.05, remaining))
                if native_child_thread_id in self._native_child_bindings:
                    return
                in_flight.add(parent_thread_id)
                self._native_child_discovery_in_flight = in_flight
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                self.discover_native_children(parent_thread_id, timeout=min(15, remaining))
            except VNextAppServerError:
                # One stale parent must not stop a bounded attempt to find the
                # caller through a different attested parent edge.
                continue
            finally:
                with self._condition:
                    self._native_child_discovery_in_flight.discard(parent_thread_id)
                    self._condition.notify_all()
            with self._condition:
                if native_child_thread_id in self._native_child_bindings:
                    return

    def _native_managed_parent_candidates(self, *, exclude: set[str] | None = None) -> tuple[str, ...]:
        """Return a fair, bounded rotation of resolver-recognized parents."""

        with self._condition:
            observer = getattr(self, "_native_child_observer", None)
            resolver = getattr(self, "_native_parent_agent_resolver", None)
            candidates = tuple(
                thread_id
                for thread_id, attestation in getattr(self, "_native_thread_attestations", {}).items()
                if (
                    thread_id not in (exclude or set())
                    and isinstance(attestation, Mapping)
                    and attestation.get("provider") == self.provider
                )
            )
        if observer is None or resolver is None:
            return ()
        recognized = []
        for thread_id in candidates:
            parent_agent_id = resolver(thread_id)
            if isinstance(parent_agent_id, str) and parent_agent_id:
                recognized.append(thread_id)
        recognized = tuple(recognized)
        if not recognized:
            return ()
        with self._condition:
            start = getattr(self, "_native_child_discovery_cursor", 0) % len(recognized)
            ordered = recognized[start:] + recognized[:start]
            self._native_child_discovery_cursor = (
                start + _MAX_NATIVE_CHILD_DISCOVERY_PARENTS
            ) % len(recognized)
        return ordered[:_MAX_NATIVE_CHILD_DISCOVERY_PARENTS]

    def cancel_native_child(self, native_child_thread_id: str, *, timeout: float = 30) -> dict[str, Any]:
        """Interrupt one observed native child even before its turn is adopted."""

        handle = self.active_native_child_turn(native_child_thread_id, timeout=timeout)
        if handle is None:
            return {"status": "not-running", "thread_id": native_child_thread_id}
        self.interrupt(handle, timeout=timeout)
        return {
            "status": "interrupt-requested",
            "thread_id": native_child_thread_id,
            "turn_id": handle.turn_id,
        }

    def wait_event(
        self,
        predicate: Callable[[dict[str, Any]], bool],
        *,
        cursor: int = 0,
        timeout: float = 60,
        progress: Callable[[dict[str, Any]], bool] | None = None,
    ) -> dict[str, Any]:
        """Wait for one event, optionally counting activity as being alive.

        Without ``progress`` the timeout is wall-clock: the wait gives up that
        many seconds after it started, however busy the other side was.  With
        ``progress`` it is an idle budget instead, and every event the caller
        recognises as its own work pushes the deadline out.  A turn that is
        still running commands and still emitting tokens is not stuck, and
        timing it out only abandons it -- the model keeps going, nobody is
        listening, and the tokens are spent for nothing.
        """

        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                for event in self._events[cursor:]:
                    cursor += 1
                    if predicate(event):
                        return event
                    if progress is not None and progress(event):
                        deadline = time.monotonic() + timeout
                self._raise_if_unavailable()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise VNextAppServerError(
                        f"app-server event wait timed out after {timeout:g}s "
                        + ("without activity" if progress is not None else "of waiting")
                    )
                self._condition.wait(timeout=min(remaining, 0.5))

    def event_cursor(self) -> int:
        with self._condition:
            return len(self._events)

    def events_since(self, cursor: int | None = None) -> tuple[dict[str, Any], ...]:
        with self._condition:
            events = tuple(self._events[cursor or 0:])
        # A native TUI owns its notifications.  A second app-server client is
        # not guaranteed to receive its child ``thread/started`` broadcast, so
        # discover provider-recorded descendants through the supported list
        # endpoint as well.  The lookup runs off the reader thread: request()
        # waits for that reader to process its response.
        self._schedule_native_child_discovery()
        return events

    def discover_native_children(self, parent_thread_id: str, *, timeout: float = 30) -> int:
        """Read provider-recorded direct descendants for one attested parent.

        This is an observation path, not a child launch or an inferred graph.
        ``thread/list(parentThreadId=...)`` returns only server-attested child
        threads and includes the child's initial-message preview, which is its
        provider-supplied objective when the parent-client collab item was not
        broadcast to this adapter connection.
        """

        if not parent_thread_id:
            raise ValueError("parent_thread_id is required")
        result = self.request(
            "thread/list",
            {"parentThreadId": parent_thread_id, "limit": 100, "useStateDbOnly": False},
            timeout=timeout,
        )
        data = result.get("data")
        if not isinstance(data, list):
            raise VNextAppServerError("thread/list returned no child thread data")
        cursor = self.event_cursor()
        observed = 0
        for raw_thread in data:
            if not isinstance(raw_thread, Mapping):
                continue
            if raw_thread.get("parentThreadId") != parent_thread_id:
                continue
            child_thread_id = raw_thread.get("id")
            if not isinstance(child_thread_id, str) or not child_thread_id:
                continue
            self._observe_native_child_event(
                {"method": "thread/started", "params": {"thread": dict(raw_thread)}}, cursor
            )
            observed += 1
        return observed

    def _schedule_native_child_discovery(self) -> None:
        deadline = time.monotonic() + _NATIVE_CHILD_DISCOVERY_TOTAL_SECONDS
        parents = self._native_managed_parent_candidates()
        with self._condition:
            if not hasattr(self, "_native_child_discovery_in_flight"):
                self._native_child_discovery_in_flight = set()
            parents = tuple(
                thread_id for thread_id in parents
                if thread_id not in self._native_child_discovery_in_flight
            )
            self._native_child_discovery_in_flight.update(parents)
        for parent_thread_id in parents:
            def target(thread_id: str = parent_thread_id) -> None:
                try:
                    remaining = deadline - time.monotonic()
                    if remaining > 0:
                        self.discover_native_children(thread_id, timeout=min(15, remaining))
                except Exception:
                    # Provider listing remains a best-effort supplemental
                    # observation channel. A later event/poll can retry.
                    pass
                finally:
                    with self._condition:
                        self._native_child_discovery_in_flight.discard(thread_id)
                        self._condition.notify_all()

            threading.Thread(
                target=target,
                name=f"vnext-child-discovery-{parent_thread_id[-8:]}",
                daemon=True,
            ).start()

    def register_tool_handler(self, thread_id: str, handler: DynamicToolHandler) -> None:
        """Restore the client-owned half of a persistent dynamic-tool thread."""
        if not thread_id:
            raise ValueError("thread_id is required")
        with self._condition:
            self._tool_handlers[thread_id] = handler

    def read_thread(
        self,
        thread_id: str,
        *,
        include_turns: bool = True,
        timeout: float = 45,
    ) -> dict[str, Any]:
        return self.request(
            "thread/read",
            {"threadId": thread_id, "includeTurns": include_turns},
            timeout=timeout,
        )

    def thread_identity_attestation(self, runtime_thread: str) -> Mapping[str, Any]:
        """Attest an exact persisted native thread without inventing an identity."""

        # `thread/start` has already acknowledged this exact provider identity.
        # Recent app-server builds may not have written a readable rollout
        # until the first turn begins, so immediately calling `thread/read`
        # turns a valid new binding into a false failure. A successor adapter
        # has no cache and still performs the provider read below before it can
        # claim a resumed identity.
        with self._condition:
            cached = self._native_thread_attestations.get(runtime_thread)
        if (
            isinstance(cached, Mapping)
            and cached.get("provider") == self.provider
            and cached.get("provider_session") == runtime_thread
        ):
            phase = str(cached.get("binding_phase") or "started")
            # This adapter started or resumed the thread itself and took its
            # id from the provider's own reply, which is the same fact the
            # `thread/read` below attests.  Reporting it as "started" left
            # every vNext-started Codex worker unroutable: the scheduler routes
            # approvals only to attested identities, so each escalation waited
            # out the grace window and was declined as unrouted.  Phases this
            # adapter did not create itself, such as an observed native child,
            # pass through unchanged.
            if phase in {"started", "resumed"}:
                phase = "attested"
            result = {
                "runtime_thread": runtime_thread,
                "provider": self.provider,
                "bound": True,
                "provider_session": runtime_thread,
                "binding_phase": phase,
                "synthetic": False,
            }
            # A provider-created child is a real persisted Codex thread.  Its
            # child relationship is separately attested by the observed
            # `thread/started` edge; expose only those provider facts for the
            # runtime identity projection.  Codex does not issue a distinct
            # native agent or task identifier on this surface, so neither is
            # invented here.
            if cached.get("origin") == "native":
                result["origin"] = "native"
                parent_runtime = cached.get("parent_runtime_thread_id")
                if isinstance(parent_runtime, str) and parent_runtime:
                    result["parent_runtime_thread_id"] = parent_runtime
                parent_turn = cached.get("parent_native_turn_id")
                if isinstance(parent_turn, str) and parent_turn:
                    result["parent_native_turn_id"] = parent_turn
            return result
        value = self.read_thread(runtime_thread, include_turns=False)
        try:
            provider_session = value["thread"]["id"]
        except (KeyError, TypeError) as exc:
            raise VNextAppServerError("thread/read returned no native thread identity") from exc
        if not isinstance(provider_session, str) or provider_session != runtime_thread:
            raise VNextAppServerError("thread/read identity differs from the runtime thread")
        attestation = {
            "runtime_thread": runtime_thread,
            "provider": self.provider,
            "bound": True,
            "provider_session": provider_session,
            "binding_phase": "attested",
            "synthetic": False,
        }
        # A successor controller has no start/resume cache.  Its provider
        # read above establishes the same attestation as those paths, and
        # native-child discovery must be able to use it immediately.  Without
        # retaining it, ``thread/list(parentThreadId=...)`` could find a real
        # child but adoption silently stopped at an apparently unknown parent.
        with self._condition:
            self._native_thread_attestations[runtime_thread] = {
                "provider": self.provider,
                "provider_session": provider_session,
                "binding_phase": "attested",
            }
        return attestation

    def resume_thread(
        self,
        *,
        thread_id: str,
        model: str,
        effort: str = "high",
        tool_handler: DynamicToolHandler | None = None,
        approvals_reviewer: str = "auto_review",
        workspace: str | Path | None = None,
        timeout: float = 60,
    ) -> dict[str, Any]:
        # As with start_thread, effort remains a per-turn provider option.
        _ = effort
        cwd = Path(workspace or self.workspace).resolve()
        # A resumed thread keeps nothing of the config its start carried, so
        # without this the clock hook is lost the moment a worker is resumed.
        # ``ThreadResumeParams.config`` accepts the same map as thread/start.
        # Only the hook table travels here: the Command Code overrides belong
        # to thread/start and adding them would change its resume behaviour.
        clock_config = self._clock_hook_config(cwd, timeout=timeout)
        result = self.request(
            "thread/resume",
            {
                "threadId": thread_id,
                "model": model,
                "cwd": str(cwd),
                "approvalPolicy": "on-request",
                "approvalsReviewer": approvals_reviewer,
                "permissions": ":workspace",
                "runtimeWorkspaceRoots": _runtime_workspace_roots(cwd),
                **({"config": clock_config} if clock_config else {}),
            },
            timeout=timeout,
        )
        self._note_thread_model(thread_id, model, result, resumed=True)
        result = self._attest_default_environment(result, timeout=timeout)
        result = self._attach_neutral_posture(result)
        with self._condition:
            self._native_thread_attestations[thread_id] = {
                "provider": self.provider,
                "provider_session": thread_id,
                "binding_phase": "resumed",
            }
        if tool_handler is not None:
            self.register_tool_handler(thread_id, tool_handler)
        return result

    def configure_resumed_root_relay(self, server_name: str) -> None:
        """Compatibility entry point for the resume runtime."""
        self.configure_manager_relay(server_name)

    def configure_manager_relay(self, server_name: str) -> None:
        """Reserve the relay used by managed threads and their native descendants.

        The caller must configure the relay before starting or resuming threads.  This
        does not create a server or alter a saved Codex configuration; runtime
        owns the already-running loopback relay and passes its name here.
        """

        if not isinstance(server_name, str) or not server_name.strip() or server_name != server_name.strip():
            raise ValueError("Manager tool relay name is required")
        with self._condition:
            existing = self._manager_relay_name
            if existing is not None and existing != server_name:
                raise VNextAppServerError("Manager tool relay is already configured")
            if self._managed_thread_mcp:
                raise VNextAppServerError("Manager tool relay cannot change after thread registration")
            self._manager_relay_name = server_name

    def resume_attested_thread(
        self,
        *,
        runtime_thread: str,
        provider_session: str,
        model: str,
        effort: str = "high",
        tools: Sequence[Mapping[str, Any]],
        developer_instructions: str,
        tool_handler: DynamicToolHandler | None,
        approvals_reviewer: str = "auto_review",
        workspace: str | Path | None = None,
        timeout: float = 60,
    ) -> dict[str, Any]:
        """Resume one quiescent Codex root with a fresh scoped MCP registration.

        Codex does not accept dynamic tools in ``thread/resume``.  The runtime
        therefore starts its one loopback relay before app-server launch, reloads
        that startup configuration, and calls this method only after registering
        its exact relay name.  The post-resume inventory is provider-reported;
        no root handler is installed unless that inventory matches the supplied
        manager definitions.
        """

        if (
            not isinstance(runtime_thread, str)
            or not runtime_thread
            or not isinstance(provider_session, str)
            or not provider_session
            or runtime_thread != provider_session
        ):
            raise VNextAppServerError("Codex fresh resume requires one attested provider thread")
        if not isinstance(model, str) or not model:
            raise VNextAppServerError("Codex fresh resume requires a model")
        if not isinstance(developer_instructions, str) or not developer_instructions.strip():
            raise VNextAppServerError("Codex fresh tool resume requires developer instructions")
        tool_payload = [dict(value) for value in tools]
        registration = self._tool_registration(model, tool_payload, tool_handler)
        if not tool_payload or tool_handler is None:
            raise VNextAppServerError("Codex fresh tool resume requires manager tools and one handler")
        if not isinstance(approvals_reviewer, str) or not approvals_reviewer.strip():
            raise VNextAppServerError("Codex fresh resume requires an approvals reviewer")
        with self._condition:
            relay_name = self._manager_relay_name
            already_resumed = runtime_thread in self._managed_thread_mcp
        if relay_name is None:
            raise VNextAppServerError("Codex fresh resume requires a configured root relay")
        if already_resumed:
            raise VNextAppServerError("Codex root thread is already resumed on this adapter")
        # Effort is still per-turn at this provider boundary.  Keep the
        # explicit argument so its selected value remains part of the caller's
        # attested session contract.
        _ = effort
        cwd = Path(workspace or self.workspace).resolve()
        # The resumed root gets the clock hook back for the same reason as
        # ``resume_thread``: the config of the original start does not survive.
        clock_config = self._clock_hook_config(cwd, timeout=timeout)
        result = self.request(
            "thread/resume",
            {
                "threadId": provider_session,
                "model": model,
                "cwd": str(cwd),
                "developerInstructions": developer_instructions,
                "approvalPolicy": "on-request",
                "approvalsReviewer": approvals_reviewer,
                **({"config": clock_config} if clock_config else {}),
                # This fresh controller never replays an old turn.  It needs
                # only the saved thread metadata and freshly attested tool
                # inventory before accepting a later prompt.
                "excludeTurns": True,
            },
            timeout=timeout,
        )
        try:
            returned_thread = result["thread"]["id"]
        except (KeyError, TypeError) as exc:
            raise VNextAppServerError("thread/resume returned no provider thread identity") from exc
        if not isinstance(returned_thread, str) or returned_thread != provider_session:
            raise VNextAppServerError("thread/resume returned a different provider thread")
        self._note_thread_model(provider_session, model, result, resumed=True)
        result = self._attest_default_environment(result, timeout=timeout)
        result = self._attach_neutral_posture(result)
        # Do not turn the resume echo into an identity claim.  Read the exact
        # persisted thread before retaining the binding or installing a handler.
        identity = self.read_thread(provider_session, include_turns=False, timeout=min(timeout, 30))
        thread = identity.get("thread") if isinstance(identity, Mapping) else None
        if not isinstance(thread, Mapping) or thread.get("id") != provider_session:
            raise VNextAppServerError("Codex managed thread read did not attest the provider thread")
        self._attest_manager_relay_inventory(
            thread_id=provider_session,
            relay_name=relay_name,
            tools=tool_payload,
            timeout=min(timeout, 30),
        )
        policy = dict(result)
        with self._condition:
            # A concurrent caller can only arrive after the provider resume;
            # it must not replace this process's handler or receipt.
            if runtime_thread in self._managed_thread_mcp:
                raise VNextAppServerError("Codex root thread is already resumed on this adapter")
            self._native_thread_attestations[runtime_thread] = {
                "provider": self.provider,
                "provider_session": provider_session,
                "binding_phase": "resumed",
            }
            self._tool_registrations[runtime_thread] = registration
            self._managed_thread_mcp[runtime_thread] = {
                "server_name": relay_name,
                "tool_handler": tool_handler,
                "registration": registration,
            }
        return {
            "provider_echo": True,
            "thread_id": provider_session,
            "policy": policy,
            "tool_registration": dict(registration),
        }

    def _attest_manager_relay_inventory(
        self,
        *,
        thread_id: str,
        relay_name: str,
        tools: Sequence[Mapping[str, Any]],
        timeout: float,
    ) -> None:
        """Require the configured relay's exact names and exposed schemas."""

        status = self.request(
            "mcpServerStatus/list",
            {"detail": "toolsAndAuthOnly", "threadId": thread_id},
            timeout=timeout,
        )
        data = status.get("data") if isinstance(status, Mapping) else None
        candidates = [
            value for value in data
            if isinstance(value, Mapping) and value.get("name") == relay_name
        ] if isinstance(data, list) else []
        if len(candidates) != 1:
            raise VNextAppServerError("Manager tool relay inventory is not uniquely attested")
        observed_tools = candidates[0].get("tools")
        if not isinstance(observed_tools, Mapping):
            raise VNextAppServerError("Manager tool relay inventory has no tools")
        expected = {str(value.get("name") or ""): dict(value) for value in tools}
        if set(observed_tools) != set(expected):
            raise VNextAppServerError("Manager tool relay inventory differs from manager tools")
        schema_exposed = False
        for name, definition in expected.items():
            observed = observed_tools.get(name)
            if not isinstance(observed, Mapping):
                continue
            keys = ("type", "description", "inputSchema")
            exposed = [key for key in keys if key in observed]
            if not exposed:
                continue
            schema_exposed = True
            for key in exposed:
                if observed.get(key) != definition.get(key):
                    raise VNextAppServerError("Manager tool relay tool schema differs from manager definition")
        # ``toolsAndAuthOnly`` may expose names only.  If it exposes any
        # schema, every manager tool must expose and match its corresponding
        # definition; a partial inventory is not an attestation.
        if schema_exposed:
            for name, definition in expected.items():
                observed = observed_tools.get(name)
                if not isinstance(observed, Mapping) or not any(
                    key in observed for key in ("type", "description", "inputSchema")
                ):
                    raise VNextAppServerError("Manager tool relay tool schemas are incomplete")

    def active_managed_thread_turn(
        self, runtime_thread: str, *, expected_turn_id: str | None = None, timeout: float = 15
    ) -> TurnHandle | None:
        """Read one current managed turn without holding the dispatcher lock."""

        if not isinstance(runtime_thread, str) or not runtime_thread:
            raise ValueError("runtime_thread is required")
        if timeout <= 0:
            raise ValueError("Codex managed thread active-turn timeout must be positive")
        with self._condition:
            configured = runtime_thread in self._managed_thread_mcp
        if not configured:
            raise VNextAppServerError("Codex root is not attested for managed MCP routing")
        payload = self.read_thread(runtime_thread, include_turns=True, timeout=min(timeout, 15))
        thread = payload.get("thread") if isinstance(payload, Mapping) else None
        if not isinstance(thread, Mapping) or thread.get("id") != runtime_thread:
            raise VNextAppServerError("Codex managed thread read did not attest the requested thread")
        turns = thread.get("turns")
        active = [
            turn for turn in turns if isinstance(turn, Mapping)
            and turn.get("status") == "inProgress"
            and isinstance(turn.get("id"), str) and turn.get("id")
        ] if isinstance(turns, list) else []
        if not active:
            return None
        if len(active) != 1:
            raise VNextAppServerError("Codex managed thread read exposed multiple active turns")
        turn_id = str(active[0]["id"])
        if expected_turn_id is not None and expected_turn_id != turn_id:
            raise VNextAppServerError("Codex managed thread MCP metadata names a different active turn")
        return TurnHandle(runtime_thread, turn_id)

    def _attest_managed_thread_mcp_item(
        self, *, thread_id: str, turn_id: str, item_id: str, relay_name: str, tool: str, timeout: float
    ) -> None:
        """Bind host metadata to an executable Responses item in this turn.

        MCP itemId is the originating Responses ID (including a code-mode
        cell), not the nested mcpToolCall ID exposed by thread/items/list.
        Only provider notifications on this connection establish that origin.
        """
        deadline = time.monotonic() + min(timeout, 2)
        with self._condition:
            while True:
                if (thread_id, turn_id, item_id) in self._managed_mcp_origins:
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
        raise VNextAppServerError("Codex managed thread MCP item is not in the current active turn")

    def _consume_raw_response_event(self, message: Mapping[str, Any]) -> bool:
        """Keep only origin IDs; raw model content must not enter the journal."""
        method = message.get("method")
        if method not in {"rawResponseItem/completed", "rawResponse/completed"}:
            return False
        params = message.get("params")
        if method == "rawResponseItem/completed" and isinstance(params, Mapping):
            thread_id, turn_id, item = params.get("threadId"), params.get("turnId"), params.get("item")
            if (isinstance(thread_id, str) and thread_id and isinstance(turn_id, str) and turn_id
                    and isinstance(item, Mapping) and item.get("type") in {"function_call", "custom_tool_call"}
                    and isinstance(item.get("id"), str) and item["id"]):
                with self._condition:
                    if thread_id in self._managed_thread_mcp:
                        self._managed_mcp_origins[(thread_id, turn_id, item["id"])] = None
                        # Bound metadata retention even if the provider never closes a turn.
                        while len(self._managed_mcp_origins) > 4096:
                            self._managed_mcp_origins.pop(next(iter(self._managed_mcp_origins)))
                        self._condition.notify_all()
        return True

    def dispatch_mcp_call(
        self,
        *,
        tool: str,
        arguments: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> ToolCallResult | Mapping[str, Any]:
        """Route one relay callback to its owned managed thread or native child.

        A configured relay is shared by managed threads and their native descendants.
        The managed-thread branch has no recovery or handler fallback; child calls retain
        the established direct-child attestation path unchanged.
        """

        thread_id = metadata.get("threadId")
        with self._condition:
            root = self._managed_thread_mcp.get(thread_id) if isinstance(thread_id, str) else None
        if root is not None:
            return self._dispatch_managed_thread_mcp_call(
                tool=tool,
                arguments=arguments,
                metadata=metadata,
                registration=root,
            )
        return self.dispatch_native_mcp_call(tool=tool, arguments=arguments, metadata=metadata)

    def _dispatch_managed_thread_mcp_call(
        self,
        *,
        tool: str,
        arguments: Mapping[str, Any],
        metadata: Mapping[str, Any],
        registration: Mapping[str, Any],
    ) -> ToolCallResult | Mapping[str, Any]:
        thread_id = metadata.get("threadId")
        server_name = metadata.get("serverName")
        if not isinstance(thread_id, str) or not thread_id:
            raise VNextAppServerError("Codex managed thread MCP call lacks host threadId metadata")
        if server_name != registration.get("server_name"):
            raise VNextAppServerError("Codex managed thread MCP call used a different relay")
        if not isinstance(tool, str) or tool not in registration["registration"]["tool_names"]:
            raise VNextAppServerError("Codex managed thread MCP tool is not registered")
        if not isinstance(arguments, Mapping):
            raise VNextAppServerError("Codex managed thread MCP arguments are invalid")
        handler = registration.get("tool_handler")
        if not callable(handler):
            raise VNextAppServerError("Codex managed thread MCP handler is not registered")
        # This provider read happens outside ``_condition``.  The app-server
        # reader must remain free to resolve the callback's request response.
        expected_turn = metadata.get("turnId")
        if expected_turn is not None and (not isinstance(expected_turn, str) or not expected_turn):
            raise VNextAppServerError("Codex managed thread MCP metadata has an invalid turnId")
        handle = self.active_managed_thread_turn(thread_id, expected_turn_id=expected_turn, timeout=15)
        if handle is None:
            raise VNextAppServerError("Codex managed thread MCP has no active exact turn")
        with self._condition:
            cursor = len(getattr(self, "_events", ()))
            current = self._managed_thread_mcp.get(thread_id)
        if current is None or current.get("tool_handler") is not handler:
            raise VNextAppServerError("Codex managed thread MCP handler changed during dispatch")
        call_id = metadata.get("itemId")
        if not isinstance(call_id, str) or not call_id:
            raise VNextAppServerError("Codex managed thread MCP requires a host itemId")
        self._attest_managed_thread_mcp_item(
            thread_id=thread_id, turn_id=handle.turn_id, item_id=call_id,
            relay_name=str(registration["server_name"]), tool=tool, timeout=15,
        )
        return handler(
            tool,
            dict(arguments),
            ToolCallContext(
                thread_id=thread_id,
                turn_id=handle.turn_id,
                call_id=call_id,
                cursor=cursor,
            ),
        )

    def ensure_windows_sandbox_ready(
        self,
        *,
        workspace: str | Path | None = None,
        timeout: float = 90,
    ) -> None:
        """Complete the configured Windows sandbox setup once before the first turn.

        The elevated Windows sandbox is an experimental runtime feature. Driving
        its setup while the runtime reports the feature disabled achieves nothing
        and writes sandbox metadata into the Codex home that a non-administrator
        owner cannot subsequently delete, so the setup is skipped in that case and
        the reason is attested.
        """

        if self._platform_name != "nt":
            return
        cwd = Path(workspace or self.workspace).resolve()
        with self._windows_sandbox_lock:
            if self._windows_sandbox_ready:
                return
            if not self._windows_sandbox_feature_enabled(timeout=timeout):
                self._windows_sandbox_attestation.update(
                    {
                        "feature_enabled": False,
                        "skip_reason": "runtime reports the feature disabled",
                    }
                )
                self._windows_sandbox_ready = True
                return
            self._windows_sandbox_attestation["feature_enabled"] = True
            self._windows_sandbox_attestation["setup_attempted"] = True
            cursor = self.event_cursor()
            response = self.request(
                "windowsSandbox/setupStart",
                {"mode": WINDOWS_SANDBOX_MODE, "cwd": str(cwd)},
                timeout=min(timeout, 15),
            )
            if response.get("started") is not True:
                raise VNextAppServerError("elevated Windows sandbox setup did not start")
            try:
                completion = self.wait_event(
                    lambda value: value.get("method") == "windowsSandbox/setupCompleted",
                    cursor=cursor,
                    timeout=timeout,
                )
            except VNextAppServerError as exc:
                raise VNextAppServerError(
                    "elevated Windows sandbox setup did not complete before the deadline"
                ) from exc
            details = completion.get("params")
            details = details if isinstance(details, dict) else {}
            if (
                details.get("success") is not True
                or details.get("mode") != WINDOWS_SANDBOX_MODE
            ):
                error = self._safe_error(details.get("error") or "unknown failure")
                raise VNextAppServerError(
                    f"elevated Windows sandbox setup failed: {error}"
                )
            self._windows_sandbox_attestation["setup_completed"] = True
            self._windows_sandbox_ready = True

    def windows_sandbox_attestation(self) -> dict[str, Any]:
        """Return content-safe facts about the Windows sandbox setup decision."""

        with self._windows_sandbox_lock:
            return dict(self._windows_sandbox_attestation)

    def _windows_sandbox_feature_enabled(self, *, timeout: float) -> bool:
        """Report whether the runtime has the elevated Windows sandbox enabled."""

        try:
            listing = self.request(
                "experimentalFeature/list",
                {},
                timeout=min(timeout, 30),
            )
        except VNextAppServerError:
            return False
        data = listing.get("data")
        if not isinstance(data, list):
            return False
        for item in data:
            if not isinstance(item, dict):
                continue
            if str(item.get("name") or "") == WINDOWS_SANDBOX_FEATURE:
                return item.get("enabled") is True
        return False

    @staticmethod
    def _method_unavailable(method: str, exc: VNextAppServerError) -> bool:
        """Report whether the runtime simply does not implement one method.

        The optional environment probe is allowed to be absent, and the
        tolerance used to be spelled as the JSON-RPC "method not found" code
        alone.  A probe with Codex 0.144.4 found that it rejects an unknown
        method as
        ``-32600 Invalid request: unknown variant `environment/status``` — the
        same fact reported under the "invalid request" code — so the narrow
        check turned a supported runtime into a hard start failure.

        Both spellings are accepted, and only for the exact method named by the
        caller: an ``-32600`` about anything else is still a real protocol
        error and is re-raised.
        """

        text = str(exc)
        if "-32601" in text:
            return True
        return "-32600" in text and f"unknown variant `{method}`" in text

    def _attest_default_environment(
        self,
        result: Mapping[str, Any],
        *,
        timeout: float,
    ) -> dict[str, Any]:
        """Attach content-safe proof that the omitted selection resolved locally."""

        request_timeout = min(timeout, 30)
        try:
            environment = self.request(
                "environment/status",
                {"environmentId": LOCAL_ENVIRONMENT_ID},
                timeout=request_timeout,
            )
        except VNextAppServerError as exc:
            if not self._method_unavailable("environment/status", exc):
                raise
            environment = {"status": "ready"}
        info = self.request(
            "environment/info",
            {"environmentId": LOCAL_ENVIRONMENT_ID},
            timeout=request_timeout,
        )
        shell = info.get("shell") if isinstance(info.get("shell"), dict) else {}
        shell_available = bool(shell.get("name") and shell.get("path"))
        enriched = dict(result)
        enriched["environmentSelection"] = {
            "environmentId": LOCAL_ENVIRONMENT_ID,
            "selected": True,
            "status": environment.get("status"),
            "shellAvailable": shell_available,
        }
        return enriched

    @staticmethod
    def _attach_neutral_posture(result: Mapping[str, Any]) -> dict[str, Any]:
        """Translate native receipt fields at the Codex adapter boundary."""

        native = effective_thread_policy(result)
        network_access = native["network_access"]
        network = "restricted" if network_access is False else "unknown"
        enriched = dict(result)
        enriched["posture"] = {
            "workspace_writes": native["sandbox_type"] == "workspaceWrite",
            "network": network,
            "approvals_requested": native["approval_policy"] == "on-request",
            "reviewer": native["approvals_reviewer"],
            "environment_ready": (
                bool(native["environment_id"])
                and native["environment_selected"] is True
                and native["environment_status"] == "ready"
                and native["environment_shell_available"] is True
            ),
        }
        return enriched

    def close(self, *, grace_seconds: float = 1.0) -> AppServerCleanup:
        if self._cleanup is not None:
            return self._cleanup
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        errors: list[str] = []
        if getattr(self, "_transport", None) is not None:
            try:
                self._transport.close()
            except Exception as exc:
                errors.append(f"remote app-server transport close failed: {exc}")
            process_cleanup = ProcessCleanup(
                outcome="detached" if not errors else "residual",
                residual_count=0,
                root_exit_code=None,
                errors=tuple(errors),
            )
        else:
            if self.supervisor is None:
                raise VNextAppServerError("app-server supervisor is unavailable")
            process_cleanup = self.supervisor.close(grace_seconds=grace_seconds)
        self._dispatcher.join(timeout=3)
        self._stderr_reader.join(timeout=3)
        with self._condition:
            handlers = tuple(self._handler_threads)
        for thread in handlers:
            thread.join(timeout=3)
        streams_drained = not self._dispatcher.is_alive() and not self._stderr_reader.is_alive()
        handlers_drained = all(not thread.is_alive() for thread in handlers)
        errors = list(process_cleanup.errors)
        if not streams_drained:
            errors.append("app-server stream readers did not drain")
        if not handlers_drained:
            errors.append("dynamic tool handler threads did not drain")
        state_dir = getattr(self, "_clock_state_dir", None)
        if state_dir:
            shutil.rmtree(state_dir, ignore_errors=True)
        self._cleanup = AppServerCleanup(
            process=process_cleanup,
            streams_drained=streams_drained,
            handler_threads_drained=handlers_drained,
            errors=tuple(errors),
        )
        return self._cleanup

    def set_native_child_observer(
        self,
        observer: NativeChildObserver | None,
        *,
        parent_agent_resolver: NativeParentAgentResolver | None = None,
    ) -> None:
        """Attach the control-plane adoption callback for attested child threads.

        A child can be reported before the runtime has installed its callback,
        so replay retained app-server events once on installation. The callback
        runs outside the adapter lock and may bind the child in the control
        plane. Passing ``None`` disables future adoption without discarding
        observations already retained for diagnostics.
        """

        with self._condition:
            self._native_child_observer = observer
            self._native_parent_agent_resolver = parent_agent_resolver
            events = tuple(enumerate(self._events))
        if observer is not None and parent_agent_resolver is not None:
            for cursor, event in events:
                self._observe_native_child_event(event, cursor)

    def native_child_attestations(self) -> tuple[dict[str, Any], ...]:
        """Return content-safe child binding facts observed on this adapter."""

        with self._condition:
            values: list[dict[str, Any]] = []
            for observation in self._native_child_observations.values():
                binding = self._native_child_bindings.get(observation.native_child_thread_id)
                values.append(
                    {
                        "provider": observation.provider,
                        "parent_agent_id": observation.parent_agent_id,
                        "parent_thread_id": observation.parent_thread_id,
                        "native_child_thread_id": observation.native_child_thread_id,
                        "agent_id": binding.agent_id if binding is not None else None,
                        "origin": "native",
                        "parent_runtime_thread_id": observation.parent_thread_id,
                        "parent_native_turn_id": observation.parent_native_turn_id,
                        "capabilities": dict(observation.capabilities or {}),
                        "delivery_contract": dict(binding.delivery_contract) if binding is not None else dict(observation.delivery_contract),
                        "unsupported_reason": (
                            binding.unsupported_reason if binding is not None else
                            self._native_child_failures.get(observation.native_child_thread_id) or
                            observation.unsupported_reason
                        ),
                    }
                )
            return tuple(values)

    def _environment(self) -> dict[str, str]:
        environment = child_process_environment(self.codex_home)
        windows_runtime_names = {
            "ALLUSERSPROFILE",
            "APPDATA",
            "COMMONPROGRAMFILES",
            "COMMONPROGRAMFILES(X86)",
            "COMMONPROGRAMW6432",
            "COMPUTERNAME",
            "HOMEDRIVE",
            "HOMEPATH",
            "LOCALAPPDATA",
            "LOGONSERVER",
            "NUMBER_OF_PROCESSORS",
            "OS",
            "PROCESSOR_ARCHITECTURE",
            "PROCESSOR_IDENTIFIER",
            "PROCESSOR_LEVEL",
            "PROCESSOR_REVISION",
            "PROGRAMDATA",
            "PROGRAMFILES",
            "PROGRAMFILES(X86)",
            "PROGRAMW6432",
            "PUBLIC",
            "SYSTEMDRIVE",
            "USERNAME",
            "USERDOMAIN",
            "USERDOMAIN_ROAMINGPROFILE",
            "USERPROFILE",
        }
        for name, value in os.environ.items():
            if name.upper() in windows_runtime_names:
                environment[name.upper()] = value
        bundled_path = self.codex_executable.parent.parent / "codex-path"
        environment["PATH"] = os.pathsep.join(
            value
            for value in (
                str(self.codex_executable.parent),
                str(bundled_path),
                environment.get("PATH", ""),
            )
            if value
        )
        return environment

    def _dispatch_stdout(self) -> None:
        if self.process is None:
            self._set_fatal("app-server process is unavailable")
            return
        stream = self.process.stdout
        if stream is None:
            self._set_fatal("app-server stdout is unavailable")
            return
        try:
            for line in stream:
                if not self._dispatch_line(line):
                    return
        except Exception as exc:
            # The reader itself failed while the process may still be running;
            # the "stdout closed" sentence below would name the wrong cause.
            if not self._closing:
                self._set_fatal(f"app-server reader failed: {self._safe_error(exc)}")
        finally:
            if not self._closing:
                exit_code = self.process.poll()
                self._set_fatal(
                    f"app-server stdout closed unexpectedly; exit_code={exit_code}; "
                    f"stderr={self._reportable_stderr()}"
                )

    def _dispatch_transport(self) -> None:
        """Read websocket messages while reusing the normal JSON-RPC router."""

        transport = getattr(self, "_transport", None)
        if transport is None:
            self._set_fatal("remote app-server transport is unavailable")
            return
        try:
            while not self._closing:
                line = transport.recv()
                if line is None:
                    break
                if not self._dispatch_line(line):
                    return
        except Exception as exc:
            if not self._closing:
                self._set_fatal(f"remote app-server transport failed: {self._safe_error(exc)}")
        finally:
            if not self._closing:
                self._set_fatal("remote app-server transport closed unexpectedly")

    def _dispatch_line(self, line: str) -> bool:
        try:
            message = json.loads(line)
            if not isinstance(message, dict):
                raise ValueError("message is not an object")
        except (json.JSONDecodeError, ValueError) as exc:
            self._set_fatal(f"app-server emitted malformed JSON: {exc}")
            return False
        if "id" not in message and self._consume_raw_response_event(message):
            return True
        if "method" in message and "id" in message:
            self._spawn_server_request(message)
            return True
        reply_id = message.get("id")
        if "id" in message and (not isinstance(reply_id, int) or isinstance(reply_id, bool)):
            # This adapter numbers every request, so a reply with a null or
            # string id answers nothing it is waiting on.  JSON-RPC sends id null
            # for a parse error.  Keep it for inspection and go on reading.
            with self._condition:
                unroutable = getattr(self, "_unroutable_responses", None)
                if unroutable is None:
                    unroutable = self._unroutable_responses = []
                unroutable.append(message)
            return True
        with self._condition:
            if "id" in message:
                self._responses[reply_id] = message
            else:
                if message.get("method") == "turn/completed":
                    params = message.get("params", {})
                    turn = params.get("turn", {}) if isinstance(params, Mapping) else {}
                    if isinstance(turn, Mapping):
                        origins = getattr(self, "_managed_mcp_origins", {})
                        for key in list(origins):
                            if key[:2] == (params.get("threadId"), turn.get("id")):
                                origins.pop(key)
                cursor = len(self._events)
                self._events.append(message)
            self._condition.notify_all()
        if "id" not in message:
            if message.get("method") == "model/rerouted" and isinstance(message.get("params"), Mapping):
                self._observe_model_reroute(message["params"])
            self._observe_native_child_event(message, cursor)
        return True

    def _observe_native_child_event(self, event: Mapping[str, Any], cursor: int) -> None:
        """Collect and adopt one provider-attested native child without guessing.

        Codex emits a child as a ``thread/started`` notification with a
        ``parentThreadId``. Parent-side collaboration items add optional prompt,
        model, effort and parent-turn metadata. Neither source is trusted alone:
        the parent must already be an adapter-attested thread and the child
        thread must name that exact parent.
        """

        params = event.get("params")
        if not isinstance(params, Mapping):
            return
        method = str(event.get("method") or "")
        if method in {"item/started", "item/completed"}:
            self._record_native_child_hint(params)
            return
        if method != "thread/started":
            return
        raw_thread = params.get("thread")
        if not isinstance(raw_thread, Mapping):
            return
        child_thread_id = raw_thread.get("id")
        parent_thread_id = raw_thread.get("parentThreadId")
        if not isinstance(child_thread_id, str) or not child_thread_id:
            return
        if not isinstance(parent_thread_id, str) or not parent_thread_id:
            return
        with self._condition:
            self._native_child_start_events[child_thread_id] = (dict(event), cursor)
            parent_attestation = self._native_thread_attestations.get(parent_thread_id)
            observer = self._native_child_observer
            resolver = self._native_parent_agent_resolver
            hint = dict(self._native_child_hints.get(child_thread_id, {}))
            already_bound = child_thread_id in self._native_child_bindings
        if parent_attestation is None or observer is None or resolver is None or already_bound:
            return
        parent_agent_id = resolver(parent_thread_id)
        if not isinstance(parent_agent_id, str) or not parent_agent_id:
            return
        source = raw_thread.get("source")
        source_parent = self._source_parent_thread_id(source)
        if source_parent is not None and source_parent != parent_thread_id:
            self._record_native_child_failure(child_thread_id, "provider child source conflicts with parent thread")
            return
        model = raw_thread.get("model") or hint.get("model_id")
        native_role = raw_thread.get("agentRole") or self._source_value(source, "agent_role")
        effort = raw_thread.get("reasoningEffort")
        # Whitespace names no work, so a blank hint falls through to the
        # preview and a blank preview leaves the child without an objective.
        objective = next(
            (
                text for text in (hint.get("objective"), raw_thread.get("preview"))
                if isinstance(text, str) and text.strip()
            ),
            None,
        )
        if not isinstance(model, str) or not model:
            self._record_native_child_failure(child_thread_id, "native child model is unavailable")
            return
        if objective is None:
            self._record_native_child_failure(child_thread_id, "native child objective is unavailable")
            return
        # Provider-native role labels are descriptive and do not grant a
        # position in vNext's organization. Native children adopt as workers;
        # retain the raw label for audit instead of treating it as authority.
        role = native_role if native_role in {"root-manager", "branch-manager", "worker"} else "worker"
        observation = NativeChildObservation(
            provider=self.provider,
            parent_agent_id=parent_agent_id,
            parent_thread_id=parent_thread_id,
            native_child_thread_id=child_thread_id,
            attested=True,
            delivery_contract={
                "history": "available",
                "usage": "available",
                "interrupt": "available",
                "context_messages": "available",
                "dynamic_tools": "unknown",
            },
            native_child_id=self._source_value(source, "agent_path") or raw_thread.get("agentNickname"),
            model_id=model,
            role=role,
            effort=effort if isinstance(effort, str) and effort else hint.get("effort"),
            objective=objective,
            task_contract=None,
            source_cursor=cursor,
            parent_native_turn_id=hint.get("parent_native_turn_id"),
            capabilities={
                "native_child_thread": "attested",
                "native_role": native_role if isinstance(native_role, str) and native_role else "unavailable",
            },
        )
        key = (self.provider, parent_thread_id, child_thread_id)
        with self._condition:
            self._native_child_observations[key] = observation
            self._native_thread_attestations[child_thread_id] = {
                "provider": self.provider,
                "provider_session": child_thread_id,
                "binding_phase": "native-child-observed",
                "origin": "native",
                "parent_runtime_thread_id": parent_thread_id,
                "parent_native_turn_id": observation.parent_native_turn_id,
            }
        try:
            binding = observer(observation)
            self._validate_native_child_binding(observation, binding)
        except Exception as exc:
            self._record_native_child_failure(child_thread_id, self._safe_error(exc))
            return
        delivery = dict(binding.delivery_contract)
        if binding.tool_handler is not None and (
            delivery.get("context_messages") != "available"
            or delivery.get("interrupt") != "available"
        ):
            self._record_native_child_failure(
                child_thread_id,
                "native child tool handler requires attested context_messages and interrupt delivery",
            )
            return
        with self._condition:
            self._native_child_bindings[child_thread_id] = binding
            if binding.tool_handler is not None:
                self._tool_handlers[child_thread_id] = binding.tool_handler

    def _record_native_child_hint(self, params: Mapping[str, Any]) -> None:
        item = params.get("item")
        if not isinstance(item, Mapping) or item.get("type") != "collabAgentToolCall":
            return
        if item.get("tool") != "spawnAgent":
            return
        parent_thread_id = params.get("threadId") or item.get("senderThreadId")
        if not isinstance(parent_thread_id, str) or not parent_thread_id:
            return
        receivers = item.get("receiverThreadIds")
        if not isinstance(receivers, list):
            return
        hint = {
            "parent_thread_id": parent_thread_id,
            "parent_native_turn_id": params.get("turnId") if isinstance(params.get("turnId"), str) else None,
            "model_id": item.get("model") if isinstance(item.get("model"), str) else None,
            "effort": item.get("reasoningEffort") if isinstance(item.get("reasoningEffort"), str) else None,
            "objective": item.get("prompt") if isinstance(item.get("prompt"), str) else None,
        }
        with self._condition:
            for receiver in receivers:
                if isinstance(receiver, str) and receiver:
                    self._native_child_hints[receiver] = hint
            retry = tuple(
                self._native_child_start_events[receiver]
                for receiver in receivers
                if isinstance(receiver, str) and receiver in self._native_child_start_events
            )
        for event, cursor in retry:
            self._observe_native_child_event(event, cursor)

    @staticmethod
    def _source_parent_thread_id(source: object) -> str | None:
        if not isinstance(source, Mapping):
            return None
        spawn = source.get("thread_spawn")
        if not isinstance(spawn, Mapping):
            return None
        value = spawn.get("parent_thread_id")
        return value if isinstance(value, str) and value else None

    @staticmethod
    def _source_value(source: object, key: str) -> str | None:
        if not isinstance(source, Mapping):
            return None
        spawn = source.get("thread_spawn")
        if not isinstance(spawn, Mapping):
            return None
        value = spawn.get(key)
        return value if isinstance(value, str) and value else None

    def _record_native_child_failure(self, child_thread_id: str, reason: str) -> None:
        with self._condition:
            self._native_child_failures[child_thread_id] = reason

    def _validate_native_child_binding(
        self, observation: NativeChildObservation, binding: NativeChildBinding
    ) -> None:
        if not isinstance(binding, NativeChildBinding):
            raise VNextAppServerError("native child observer returned no NativeChildBinding")
        if binding.provider != observation.provider:
            raise VNextAppServerError("native child binding provider mismatches observation")
        if binding.native_thread_id != observation.native_child_thread_id:
            raise VNextAppServerError("native child binding thread mismatches observation")
        if not binding.agent_id:
            raise VNextAppServerError("native child binding has no vNext agent id")

    def _read_stderr(self) -> None:
        stream = self.process.stderr
        if stream is None:
            return
        for line in stream:
            self._stderr_lines.append(line.rstrip()[-2048:])

    def _spawn_server_request(self, message: dict[str, Any]) -> None:
        def target() -> None:
            try:
                self._handle_server_request(message)
            finally:
                with self._condition:
                    self._handler_threads.discard(threading.current_thread())
                    self._condition.notify_all()

        thread = threading.Thread(target=target, name="vnext-app-server-tool", daemon=True)
        with self._condition:
            self._handler_threads.add(thread)
        thread.start()

    def _handle_server_request(self, message: Mapping[str, Any]) -> None:
        request_id = message.get("id")
        method = str(message.get("method") or "")
        params = message.get("params") if isinstance(message.get("params"), dict) else {}
        try:
            if method == "item/tool/call":
                thread_id = str(params.get("threadId") or "")
                with self._condition:
                    # Dynamic-tool registration is thread-scoped.  Native
                    # children only receive a handler after their own provider
                    # thread is attested and adopted; never route an unbound
                    # child request to the parent merely because it is the
                    # sole handler in this app-server connection.
                    handler = self._tool_handlers.get(thread_id)
                    # An external native turn can call a dynamic tool before
                    # the controller owns a TurnHandle. This local checkpoint
                    # starts the adopted turn's wait after already-observed
                    # events, without claiming a cross-provider cursor shape.
                    source_cursor = len(getattr(self, "_events", ()))
                    if not hasattr(self, "_tool_calls"):
                        self._tool_calls = []
                    self._tool_calls.append(
                        {
                            "thread_id": thread_id,
                            "tool": str(params.get("tool") or ""),
                            "handler_registered": handler is not None,
                            "turn_correlated": bool(params.get("turnId")),
                            "call_correlated": bool(params.get("callId") or params.get("itemId")),
                        }
                    )
                if handler is None:
                    raise VNextAppServerError("no dynamic tool handler is registered for the thread")
                tool = str(params.get("tool") or "")
                arguments = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
                context = ToolCallContext(
                    thread_id=thread_id,
                    turn_id=str(params.get("turnId") or ""),
                    call_id=str(params.get("callId") or params.get("itemId") or ""),
                    cursor=source_cursor,
                )
                result = project_tool_result(handler(tool, arguments, context))
            elif method in {
                "item/commandExecution/requestApproval",
                "item/fileChange/requestApproval",
                "item/permissions/requestApproval",
            }:
                if self.native_approval_handler is None:
                    raise VNextAppServerError("native approval request has no configured handler")
                result = self._handle_native_approval(method, params)
            else:
                self._send(
                    {
                        "id": request_id,
                        "error": {"code": -32601, "message": "unsupported app-server request"},
                    }
                )
                return
            try:
                self._send({"id": request_id, "result": result})
            except VNextAppServerError:
                if self._closing or self._connection_closed():
                    return
                raise
        except BaseException as exc:
            try:
                self._send(
                    {
                        "id": request_id,
                        "error": {"code": -32000, "message": self._safe_error(exc)},
                    }
                )
            except VNextAppServerError:
                if self._closing or self._connection_closed():
                    return
                raise

    def _handle_native_approval(
        self,
        method: str,
        params: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Translate a Codex request into the shared opaque approval envelope."""

        effect_by_method = {
            "item/commandExecution/requestApproval": "execute",
            "item/fileChange/requestApproval": "modify",
            "item/permissions/requestApproval": "permission",
        }
        request_id = params.get("itemId")
        thread_id = params.get("threadId")
        turn_id = params.get("turnId")
        if not all(
            isinstance(value, str) and value.strip() == value and value
            for value in (request_id, thread_id, turn_id)
        ):
            return self._native_approval_decline(method)
        with self._condition:
            identity = self._native_thread_attestations.get(thread_id)
            attested = (
                isinstance(identity, Mapping)
                and identity.get("provider") == self.provider
                and identity.get("provider_session") == thread_id
            )
        if not attested:
            return self._native_approval_decline(method)
        envelope = {
            "approval_reference": f"approval:{request_id}",
            "provider": self.provider,
            "provider_correlation": {
                "session": thread_id,
                "turn": turn_id,
                "request": request_id,
            },
            "correlation_attested": True,
            "effect": effect_by_method[method],
        }
        decision = self.native_approval_handler("approval/request", envelope)
        if method == "item/permissions/requestApproval":
            return {"scope": "turn", "permissions": {}}
        if decision.get("decision") == "accept":
            return {"decision": "accept"}
        return {"decision": "decline"}

    @staticmethod
    def _native_approval_decline(method: str) -> Mapping[str, Any]:
        if method == "item/permissions/requestApproval":
            return {"scope": "turn", "permissions": {}}
        return {"decision": "decline"}

    def _send(self, message: Mapping[str, Any]) -> None:
        encoded = json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._send_lock:
            transport = getattr(self, "_transport", None)
            if transport is not None:
                if transport.closed:
                    raise VNextAppServerError("remote app-server transport is unavailable")
                try:
                    transport.send(encoded)
                    return
                except Exception as exc:
                    raise VNextAppServerError("remote app-server transport is unavailable") from exc
            if self.process is None or self.process.stdin is None or self.process.poll() is not None:
                raise VNextAppServerError("app-server stdin is unavailable")
            try:
                self.process.stdin.write(encoded)
                self.process.stdin.flush()
            except (OSError, ValueError) as exc:
                raise VNextAppServerError("app-server stdin is unavailable") from exc

    def turn_process_ended(self, handle: object | None = None) -> bool | None:
        """Whether the app-server process behind a turn has exited.

        The answer is about the process rather than the turn, which is the
        honest thing this adapter owns: a provider that is gone cannot still
        be writing the workspace.  Over a WebSocket the app-server lives
        behind a socket, so a closed transport proves nothing; the shared
        server's own process is read when this adapter started it, and
        otherwise the answer is that nobody here knows.
        """

        del handle
        process = self.process
        if getattr(self, "_transport", None) is not None:
            server = getattr(self, "server", None)
            process = getattr(getattr(server, "_process", None), "process", None)
        if process is None:
            return None
        return process.poll() is not None

    def _connection_closed(self) -> bool:
        transport = getattr(self, "_transport", None)
        if transport is not None:
            return transport.closed
        return self.process is None or self.process.poll() is not None

    def _raise_if_unavailable(self) -> None:
        if self._fatal is not None:
            raise VNextAppServerError(self._fatal)
        if self._closing:
            raise VNextAppServerError("app-server adapter is closing")

    def _set_fatal(self, message: str) -> None:
        with self._condition:
            if self._fatal is None:
                self._fatal = message
            self._condition.notify_all()

    def _reportable_stderr(self) -> str:
        """The child's stderr, unless this child was handed a credential.

        A fatal message becomes an exception string, and the scheduler puts
        that string in the session run log -- the file a user attaches to a
        bug report.  The Command Code child carries a loopback bearer on its
        own command line, so its stderr is withheld here for the same reason
        ``provider_stderr`` is: an authorization value echoed while failing
        must not be persisted.  The reason is said in its place, so a failure
        is still legible.
        """

        if self.provider in CREDENTIAL_ADJACENT_STDERR_PROVIDERS:
            return (
                f"withheld: a {self.provider} child is handed a credential, so its "
                "stderr is not persisted"
            )
        return self._safe_error(" | ".join(self._stderr_lines))

    def _safe_error(self, value: object) -> str:
        text = str(value)
        text = text.replace(str(self.codex_home), "<codex-home>")
        text = text.replace(str(self.workspace), "<workspace>")
        text = text.replace(str(Path.home()), "<user-home>")
        return text[-3000:]


def effective_thread_policy(result: Mapping[str, Any]) -> dict[str, Any]:
    sandbox = result.get("sandbox") if isinstance(result.get("sandbox"), dict) else {}
    raw_profile = result.get("activePermissionProfile")
    profile_id = raw_profile.get("id") if isinstance(raw_profile, dict) else raw_profile
    environment = result.get("environmentSelection")
    environment = environment if isinstance(environment, dict) else {}
    return {
        "approval_policy": result.get("approvalPolicy"),
        "approvals_reviewer": result.get("approvalsReviewer"),
        "sandbox_type": sandbox.get("type"),
        "network_access": sandbox.get("networkAccess"),
        "permission_profile_id": profile_id,
        "environment_id": environment.get("environmentId"),
        "environment_selected": environment.get("selected") is True,
        "environment_status": environment.get("status"),
        "environment_shell_available": environment.get("shellAvailable") is True,
    }


def require_workspace_policy(result: Mapping[str, Any]) -> None:
    policy = effective_thread_policy(result)
    if (
        policy["approval_policy"] != "on-request"
        or policy["sandbox_type"] != "workspaceWrite"
        or policy["permission_profile_id"] != ":workspace"
        or not policy["environment_id"]
        or policy["environment_selected"] is not True
        or policy["environment_status"] != "ready"
        or policy["environment_shell_available"] is not True
    ):
        raise VNextAppServerError(f"unexpected effective thread policy: {policy}")
