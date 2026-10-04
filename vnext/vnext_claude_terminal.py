"""Lease and relay primitives for an idle Claude SDK-to-terminal handoff.

This module deliberately owns neither a Claude process nor a scheduler.  It
does provide the two pieces a terminal process cannot inherit from the SDK:
an external JSON-RPC MCP relay and an external hook relay.  The runtime must
inject the host handler that turns these relayed records into vNext commands
and receipts.

It is a transport boundary.  In particular, it contains no approval policy
and it does not make native Claude ``Agent`` children managed workers.
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ClaudeTerminalError(RuntimeError):
    """A terminal handoff cannot safely proceed."""


@dataclass(frozen=True)
class ClaudeTerminalRelayEvent:
    """One opaque provider protocol record delivered to the vNext host."""

    lease_id: str
    channel: str  # ``mcp`` or ``hook``
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class ClaudeTerminalLaunchSpec:
    """Process details.  The per-lease bearer token is intentionally absent."""

    executable: str
    arguments: tuple[str, ...]
    environment: Mapping[str, str]
    native_session_id: str
    lease_id: str
    mcp_config_path: str
    settings_path: str
    native_children: str = "full-control-map-unavailable"


@dataclass(frozen=True)
class ClaudeTerminalLease:
    """Attested identity and files for exactly one terminal owner."""

    runtime_thread_id: str
    native_session_id: str
    lease_id: str
    directory: Path
    launch: ClaudeTerminalLaunchSpec


RelayHandler = Callable[[ClaudeTerminalRelayEvent], Mapping[str, Any]]


@dataclass
class _RelayLeaseRecord:
    token: str
    accepting: bool = True
    active_callbacks: int = 0


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ClaudeTerminalError(f"{name} must be a non-empty string")
    return value


class ClaudeTerminalRelayServer:
    """Loopback HTTP receiver used by the stdio MCP and hook relay processes.

    The relay token is checked here and lives only in the private lease config.
    It is never returned in :class:`ClaudeTerminalLaunchSpec` or emitted to a
    browser/event consumer.
    """

    def __init__(self, handler: RelayHandler, *, host: str = "127.0.0.1", port: int = 0) -> None:
        # ThreadingHTTPServer is IPv4 by default; accept only addresses it can
        # actually bind without widening the listener.
        if host not in {"127.0.0.1", "localhost"}:
            raise ValueError("Claude terminal relay must bind a loopback host")
        self._handler = handler
        self._leases: dict[str, _RelayLeaseRecord] = {}
        self._condition = threading.Condition(threading.RLock())
        parent = self

        class Receiver(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - stdlib entry point
                parent._receive(self)

            def log_message(self, _format: str, *_args: object) -> None:
                return

        self._server = ThreadingHTTPServer((host, port), Receiver)
        self._thread: threading.Thread | None = None

    @property
    def endpoint(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._server.serve_forever, name="vnext-claude-relay", daemon=True)
            self._thread.start()

    def close(self) -> None:
        if self._thread is not None:
            self._server.shutdown()
            self._thread.join(timeout=2)
            self._thread = None
        self._server.server_close()

    def register(self, lease_id: str) -> str:
        token = secrets.token_urlsafe(32)
        with self._condition:
            if lease_id in self._leases:
                raise ClaudeTerminalError("terminal relay lease is already registered")
            self._leases[lease_id] = _RelayLeaseRecord(token)
        return token

    def unregister(self, lease_id: str) -> None:
        """Reject new requests and drain callbacks admitted before release."""

        with self._condition:
            record = self._leases.get(lease_id)
            if record is None:
                return
            record.accepting = False
            while record.active_callbacks:
                self._condition.wait()
            self._leases.pop(lease_id, None)

    def _receive(self, request: BaseHTTPRequestHandler) -> None:
        pieces = request.path.strip("/").split("/")
        if len(pieces) != 3 or pieces[0] != "leases" or pieces[2] not in {"mcp", "hook"}:
            self._respond(request, 404, {"error": "unknown relay endpoint"})
            return
        lease_id, channel = pieces[1], pieces[2]
        authorization = request.headers.get("Authorization", "")
        with self._condition:
            record = self._leases.get(lease_id)
            if record is None or not record.accepting or not secrets.compare_digest(authorization.encode(), f"Bearer {record.token}".encode()):
                self._respond(request, 403, {"error": "invalid terminal relay credential"})
                return
            record.active_callbacks += 1
        try:
            length = int(request.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1_048_576:
                raise ValueError("request body size is invalid")
            body = json.loads(request.rfile.read(length).decode("utf-8"))
            if not isinstance(body, Mapping):
                raise ValueError("request body must be an object")
            result = self._handler(ClaudeTerminalRelayEvent(lease_id, channel, dict(body)))
            if not isinstance(result, Mapping):
                raise ValueError("host relay handler must return an object")
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
            self._respond(request, 400, {"error": str(exc)})
            return
        except Exception as exc:  # relay failure must reach the terminal as structured evidence
            self._respond(request, 502, {"error": f"vNext relay handler failed: {exc}"})
        else:
            self._respond(request, 200, dict(result))
        finally:
            with self._condition:
                record.active_callbacks -= 1
                self._condition.notify_all()

    @staticmethod
    def _respond(request: BaseHTTPRequestHandler, status: int, body: Mapping[str, Any]) -> None:
        payload = json.dumps(dict(body), separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        request.send_response(status)
        request.send_header("Content-Type", "application/json")
        request.send_header("Content-Length", str(len(payload)))
        request.end_headers()
        request.wfile.write(payload)


class ClaudeTerminalLeaseManager:
    """Create one terminal lease only after the SDK has become idle.

    A service integration supplies a durable vNext-owned ``CLAUDE_CONFIG_DIR``
    that contains the resumable native session.  This manager does not copy
    user settings or the SDK's temporary session materialization.
    """

    def __init__(self, directory: str | Path, relay: ClaudeTerminalRelayServer) -> None:
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._relay = relay
        self._leases: dict[str, ClaudeTerminalLease] = {}
        self._by_thread: dict[str, str] = {}
        self._releasing: set[str] = set()
        self._lock = threading.RLock()

    def acquire(
        self,
        *,
        runtime_thread_id: str,
        native_session_id: str,
        model: str,
        effort: str,
        permission_mode: str = "default",
        claude_config_dir: str | Path | None = None,
        sdk_turn_active: bool,
        executable: str = "claude",
        setting_sources: Sequence[str] = ("user", "project", "local"),
    ) -> ClaudeTerminalLease:
        """Write private relay config and return an idle-only terminal launch.

        ``sdk_turn_active`` must be derived from the runtime lock/turn state,
        not from a UI status.  ``claude_config_dir`` is optional: without it
        the CLI retains its ordinary native configuration/home, as the default
        SDK client does.  A service may provide a durable managed config only
        after proving it retains the needed native session and authentication.
        The caller starts the relay before launching the command and keeps its
        host handler alive for the whole lease.
        """

        if sdk_turn_active:
            raise ClaudeTerminalError("cannot hand a Claude session to terminal while an SDK turn is active")
        runtime_thread_id = _require_string(runtime_thread_id, "runtime thread id")
        native_session_id = _require_string(native_session_id, "native session id")
        try:
            uuid.UUID(native_session_id)
        except (ValueError, AttributeError) as exc:
            raise ClaudeTerminalError("terminal handoff requires an attested UUID native session id") from exc
        model, effort, executable = (
            _require_string(model, "model"),
            _require_string(effort, "effort"),
            _require_string(executable, "Claude executable"),
        )
        environment: dict[str, str] = {}
        if claude_config_dir is not None:
            config_dir = Path(claude_config_dir).resolve()
            if not config_dir.is_dir():
                raise ClaudeTerminalError("configured Claude config directory does not exist")
            environment["CLAUDE_CONFIG_DIR"] = str(config_dir)
        sources = tuple(_require_string(source, "setting source") for source in setting_sources)
        if not sources:
            raise ClaudeTerminalError("at least one Claude setting source is required")
        # The terminal is the user's own interactive CLI and holds no vNext
        # approval policy, so bypassPermissions is the user's call here.
        if permission_mode not in {"default", "acceptEdits", "plan", "auto", "bypassPermissions"}:
            raise ClaudeTerminalError("Claude terminal permission mode is unsupported")
        with self._lock:
            if runtime_thread_id in self._by_thread:
                raise ClaudeTerminalError("runtime thread already has a terminal lease")
            lease_id = str(uuid.uuid4())
            lease_dir = (self.directory / lease_id).resolve()
            if self.directory not in lease_dir.parents:
                raise ClaudeTerminalError("terminal lease directory escaped its configured root")
            lease_dir.mkdir(mode=0o700)
            token = self._relay.register(lease_id)
            relay_config = lease_dir / "relay.json"
            mcp_config = lease_dir / "mcp.json"
            settings = lease_dir / "settings.json"
            # The terminal runs in the user's workspace; -m could resolve a
            # different editable checkout or a workspace package with this name.
            relay_entrypoint = str(Path(__file__).resolve())
            try:
                self._write_private(
                    relay_config,
                    {"endpoint": self._relay.endpoint, "lease_id": lease_id, "token": token},
                )
                self._write_private(
                    mcp_config,
                    {
                        "mcpServers": {
                            "vnext": {
                                "command": sys.executable,
                                "args": [relay_entrypoint, "mcp", "--lease-config", str(relay_config)],
                            }
                        }
                    },
                )
                hook_args = [relay_entrypoint, "hook", "--lease-config", str(relay_config)]
                self._write_private(
                    settings,
                    {
                        "hooks": {
                            "PreToolUse": [{"matcher": "Agent", "hooks": [{"type": "command", "command": sys.executable, "args": hook_args}]}],
                            "UserPromptSubmit": [{"hooks": [{"type": "command", "command": sys.executable, "args": hook_args}]}],
                            "Stop": [{"hooks": [{"type": "command", "command": sys.executable, "args": hook_args}]}],
                            "SubagentStart": [{"hooks": [{"type": "command", "command": sys.executable, "args": hook_args}]}],
                            "SubagentStop": [{"hooks": [{"type": "command", "command": sys.executable, "args": hook_args}]}],
                        }
                    },
                )
            except Exception:
                self._relay.unregister(lease_id)
                for private_file in (relay_config, mcp_config, settings):
                    private_file.unlink(missing_ok=True)
                lease_dir.rmdir()
                raise
            launch = ClaudeTerminalLaunchSpec(
                executable=executable,
                arguments=(
                    f"--resume={native_session_id}",
                    f"--model={model}",
                    f"--effort={effort}",
                    f"--permission-mode={permission_mode}",
                    f"--setting-sources={','.join(sources)}",
                    "--mcp-config", str(mcp_config), "--settings", str(settings),
                ),
                environment=environment,
                native_session_id=native_session_id,
                lease_id=lease_id,
                mcp_config_path=str(mcp_config),
                settings_path=str(settings),
            )
            lease = ClaudeTerminalLease(runtime_thread_id, native_session_id, lease_id, lease_dir, launch)
            self._leases[lease_id] = lease
            self._by_thread[runtime_thread_id] = lease_id
            return lease

    def release(self, lease_id: str, *, terminal_stopped: bool, resumed_native_session_id: str | None = None) -> ClaudeTerminalLease:
        """Release after process exit and, if SDK resumes, an identity attestation."""

        if not terminal_stopped:
            raise ClaudeTerminalError("cannot release a terminal lease before terminal stop is confirmed")
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None:
                raise ClaudeTerminalError("terminal lease is not active")
            if lease_id in self._releasing:
                raise ClaudeTerminalError("terminal lease release is already in progress")
            if resumed_native_session_id is not None and resumed_native_session_id != lease.native_session_id:
                raise ClaudeTerminalError("SDK resumed a different Claude native session")
            self._releasing.add(lease_id)
        try:
            # Do not hold the manager lock while the host callback drains: the
            # callback is allowed to persist/schedule a final receipt.
            self._relay.unregister(lease_id)
            (lease.directory / "relay.json").unlink(missing_ok=True)
            with self._lock:
                self._leases.pop(lease_id, None)
                self._by_thread.pop(lease.runtime_thread_id, None)
                return lease
        finally:
            with self._lock:
                self._releasing.discard(lease_id)

    @staticmethod
    def _write_private(path: Path, value: Mapping[str, Any]) -> None:
        path.write_text(json.dumps(dict(value), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        try:
            path.chmod(0o600)
        except OSError:  # Windows ACLs govern this; callers must keep the root private.
            pass


def _load_relay_config(path: str) -> tuple[str, str, str]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        endpoint, lease_id, token = data["endpoint"], data["lease_id"], data["token"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ClaudeTerminalError("cannot read terminal relay configuration") from exc
    return _require_string(endpoint, "relay endpoint"), _require_string(lease_id, "lease id"), _require_string(token, "relay token")


def _forward_channel(config_path: str, channel: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
    endpoint, lease_id, token = _load_relay_config(config_path)
    request = Request(
        f"{endpoint}/leases/{lease_id}/{channel}",
        data=json.dumps(dict(payload), separators=(",", ":")).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"}, method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:  # nosec B310 - endpoint is local vNext-provided config
            result = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, OSError, json.JSONDecodeError) as exc:
        raise ClaudeTerminalError(f"terminal relay {channel} request failed: {exc}") from exc
    if not isinstance(result, Mapping):
        raise ClaudeTerminalError("terminal relay returned a non-object response")
    return dict(result)


def _mcp_main(config_path: str) -> int:
    for line in sys.stdin:
        request_id: object = None
        is_notification = False
        try:
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError("JSON-RPC payload must be an object")
            is_notification = "id" not in payload
            request_id = payload.get("id")
            result = _forward_channel(config_path, "mcp", payload)
        except (ValueError, json.JSONDecodeError, ClaudeTerminalError) as exc:
            result = {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32000, "message": str(exc)}}
        if not is_notification:
            sys.stdout.write(json.dumps(result, separators=(",", ":")) + "\n")
            sys.stdout.flush()
    return 0


def _hook_main(config_path: str) -> int:
    try:
        payload = json.load(sys.stdin)
        if not isinstance(payload, Mapping):
            raise ValueError("hook payload must be an object")
        result = _forward_channel(config_path, "hook", payload)
    except (ValueError, json.JSONDecodeError, ClaudeTerminalError) as exc:
        result = {"continue": True, "systemMessage": f"vNext hook relay failed: {exc}"}
    sys.stdout.write(json.dumps(result, separators=(",", ":")))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="vNext Claude terminal relay")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("mcp", "hook"):
        command = commands.add_parser(name)
        command.add_argument("--lease-config", required=True)
    parsed = parser.parse_args(argv)
    return _mcp_main(parsed.lease_config) if parsed.command == "mcp" else _hook_main(parsed.lease_config)


if __name__ == "__main__":  # pragma: no cover - subprocess entry point
    raise SystemExit(main())
