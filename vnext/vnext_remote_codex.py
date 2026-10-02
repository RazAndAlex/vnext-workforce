"""Shared-WebSocket Codex app-server transport for native terminal attachment.

The normal adapter owns a private stdio app-server.  This module instead owns a
verified pinned app-server WebSocket endpoint and lets both vNext and a Codex
TUI connect to the same provider threads.  It never treats a terminal command
line as evidence: callers must initialize the adapter and attest the thread.
"""

from __future__ import annotations

import os
import json
import secrets
import socket
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .child_environment import child_process_environment
from .live_runtime import resolve_live_runtime
from .process_supervisor import OwnedProcess, ProcessCleanup
from .vnext_app_server import AppServerTransport, VNextAppServerAdapter, VNextAppServerError


class RemoteCodexError(RuntimeError):
    pass


@dataclass(frozen=True)
class NativeTerminalLaunchSpec:
    """Opaque-to-the-host native TUI launch details for one attested thread."""

    executable: str
    arguments: tuple[str, ...]
    environment: Mapping[str, str]
    endpoint: str
    native_thread_id: str
    authenticated: bool


class WebSocketJsonRpcTransport(AppServerTransport):
    """Thread-safe line framing over the established websocket-client package."""

    def __init__(self, endpoint: str, token: str | None = None, *, timeout: float = 10) -> None:
        try:
            import websocket
        except ImportError as exc:  # pragma: no cover - packaging failure is clear to caller
            raise RemoteCodexError("websocket-client==1.9.2 is required for remote Codex transport") from exc
        headers = [f"Authorization: Bearer {token}"] if token else []
        try:
            # Codex's native TUI does not send a browser Origin.  Suppressing
            # websocket-client's synthetic Origin also avoids treating this
            # machine-local JSON-RPC client as a cross-origin browser request.
            self._socket = websocket.create_connection(
                endpoint, header=headers, timeout=timeout, suppress_origin=True
            )
            # `timeout` bounds the connection handshake.  A persistent
            # app-server session is intentionally idle between turns, so a
            # socket read timeout must not turn ordinary idleness into a fatal
            # transport disconnect.  Adapter request timeouts are enforced by
            # its JSON-RPC condition wait instead.
            self._socket.settimeout(None)
        except Exception as exc:
            raise RemoteCodexError(f"could not connect to Codex app-server endpoint: {exc}") from exc
        self._closed = False
        self._lock = threading.Lock()

    @property
    def closed(self) -> bool:
        return self._closed

    def recv(self) -> str | None:
        if self._closed:
            return None
        try:
            value = self._socket.recv()
        except Exception as exc:
            if self._closed:
                return None
            raise RemoteCodexError(f"Codex app-server websocket receive failed: {exc}") from exc
        if value in {None, ""}:
            return None
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    def send(self, payload: str) -> None:
        if self._closed:
            raise RemoteCodexError("Codex app-server websocket is closed")
        try:
            self._socket.send(payload)
        except Exception as exc:
            raise RemoteCodexError(f"Codex app-server websocket send failed: {exc}") from exc

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            # close() waits for a peer frame, whose reader lock can already
            # belong to our indefinitely idle dispatcher. Its timeout does
            # not bound that lock wait. Wake recv before closing the socket.
            self._socket.abort()
            self._socket.close()


class RemoteCodexServer:
    """Own one verified Codex app-server that has TUI-safe shared threads."""

    def __init__(
        self,
        *,
        workspace: str | Path,
        codex_home: str | Path,
        bind_host: str = "127.0.0.1",
        port: int = 0,
        require_auth: bool = False,
        native_child_participation: bool = False,
        native_max_depth: int | None = None,
        mcp_startup_config_overrides: Mapping[str, str] | None = None,
        runtime_resolver: Callable[[], tuple[Path, Mapping[str, object]]] = resolve_live_runtime,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.codex_home = Path(codex_home).resolve()
        if not self.workspace.is_dir() or not self.codex_home.is_dir():
            raise ValueError("workspace and Codex home must exist")
        if require_auth and bind_host in {"127.0.0.1", "::1", "localhost"}:
            raise ValueError("authenticated app-server must bind a non-loopback listener")
        self.bind_host = bind_host
        self.port = port or self._reserve_port(bind_host)
        self.require_auth = require_auth
        # This starts false until the runtime has installed an attested child
        # observer. Enabling Codex's multi-agent mode without that observer
        # would create unowned provider children, which is not participation.
        self.native_child_participation = native_child_participation
        if native_max_depth is not None:
            if type(native_max_depth) is not int or not 1 <= native_max_depth <= 2**31 - 1:
                raise ValueError("codex_native_max_depth must be a positive 32-bit integer")
            if not native_child_participation:
                raise ValueError("codex_native_max_depth requires native child participation")
        self.native_max_depth = native_max_depth
        self.mcp_startup_config_overrides = VNextAppServerAdapter._validated_mcp_startup_overrides(
            mcp_startup_config_overrides or {})
        self._runtime_resolver = runtime_resolver
        self._token = secrets.token_urlsafe(32) if require_auth else None
        self._token_file: Path | None = None
        self._process: OwnedProcess | None = None
        self._executable: Path | None = None
        self._metadata: Mapping[str, object] | None = None

    @property
    def endpoint(self) -> str:
        return f"ws://127.0.0.1:{self.port}"

    @property
    def authenticated(self) -> bool:
        return self.require_auth and self._token is not None

    @property
    def metadata(self) -> Mapping[str, object]:
        if self._metadata is None:
            raise RemoteCodexError("remote Codex server has not started")
        return self._metadata

    def start(self, *, timeout: float = 10) -> Mapping[str, object]:
        if self._process is not None:
            return self.metadata
        executable, metadata = self._runtime_resolver()
        command = self._start_command(executable)
        if self.require_auth:
            handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, prefix="vnext-codex-ws-")
            handle.write(self._token or "")
            handle.close()
            self._token_file = Path(handle.name)
            command.extend(["--ws-auth", "capability-token", "--ws-token-file", str(self._token_file)])
        environment = child_process_environment(self.codex_home)
        environment["PATH"] = os.pathsep.join((str(executable.parent), environment.get("PATH", "")))
        self._process = OwnedProcess.start(command, env=environment, cwd=str(self.workspace))
        self._executable = executable
        self._metadata = dict(metadata)
        deadline = time.monotonic() + timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                transport = WebSocketJsonRpcTransport(self.endpoint, self._token, timeout=1)
                transport.close()
                return self.metadata
            except Exception as exc:
                last_error = exc
                time.sleep(0.05)
        self.close()
        raise RemoteCodexError(f"pinned Codex app-server did not accept websocket clients: {last_error}")

    def _start_command(self, executable: Path) -> list[str]:
        feature = "--enable" if self.native_child_participation else "--disable"
        overrides = [argument for key, value in self.mcp_startup_config_overrides.items()
                     for argument in ("-c", f"{key}={json.dumps(value, ensure_ascii=False)}")]
        if self.native_max_depth is not None:
            overrides.extend(["-c", f"agents.max_depth={self.native_max_depth}"])
        return [
            str(executable), "--strict-config", feature, "multi_agent", *overrides, "app-server", "--listen",
            f"ws://{self.bind_host}:{self.port}",
        ]

    def connect(
        self,
        *,
        client_name: str = "vnext_remote",
        mcp_config_overrides: Mapping[str, object] | None = None,
    ) -> "VNextRemoteCodexAdapter":
        self.start()
        return VNextRemoteCodexAdapter(
            server=self,
            client_name=client_name,
            mcp_config_overrides=mcp_config_overrides,
        )

    def terminal_launch(self, native_thread_id: str) -> NativeTerminalLaunchSpec:
        if self._executable is None:
            raise RemoteCodexError("remote Codex server has not started")
        if not native_thread_id:
            raise ValueError("native thread identity is required")
        env: dict[str, str] = {}
        arguments: list[str] = ["--remote", self.endpoint]
        if self._token:
            env_name = "VNEXT_CODEX_REMOTE_TOKEN"
            env[env_name] = self._token
            arguments.extend(["--remote-auth-token-env", env_name])
        arguments.extend(["resume", native_thread_id])
        return NativeTerminalLaunchSpec(
            executable=str(self._executable), arguments=tuple(arguments), environment=env,
            endpoint=self.endpoint, native_thread_id=native_thread_id, authenticated=self.authenticated,
        )

    def close(self) -> ProcessCleanup | None:
        cleanup = self._process.close() if self._process is not None else None
        self._process = None
        if self._token_file is not None:
            try:
                self._token_file.unlink(missing_ok=True)
            finally:
                self._token_file = None
        return cleanup

    @staticmethod
    def _reserve_port(host: str) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind((host, 0))
            return int(probe.getsockname()[1])


class VNextRemoteCodexAdapter(VNextAppServerAdapter):
    """The ordinary vNext adapter lifecycle over one shared WebSocket server."""

    def __init__(
        self,
        *,
        server: RemoteCodexServer,
        client_name: str = "vnext_remote",
        mcp_config_overrides: Mapping[str, object] | None = None,
    ) -> None:
        if server._executable is None:
            raise RemoteCodexError("remote Codex server must be started before connecting")
        self.server = server
        super().__init__(
            codex_executable=server._executable,
            codex_home=server.codex_home,
            workspace=server.workspace,
            client_name=client_name,
            mcp_config_overrides=mcp_config_overrides,
            transport=WebSocketJsonRpcTransport(server.endpoint, server._token),
        )

    def terminal_launch(self, native_thread_id: str) -> NativeTerminalLaunchSpec:
        """Return a native TUI launch only for a thread this adapter attested."""

        identity = self.thread_identity_attestation(native_thread_id)
        if identity.get("bound") is not True or identity.get("provider_session") != native_thread_id:
            raise VNextAppServerError("native terminal requires an attested shared Codex thread")
        return self.server.terminal_launch(native_thread_id)
