from __future__ import annotations

import json
import socket
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from vnext.vnext_app_server import VNextAppServerAdapter
from vnext.vnext_remote_codex import RemoteCodexServer, WebSocketJsonRpcTransport


class FakeTransport:
    def __init__(self) -> None:
        self.closed = False
        self.sent: list[dict] = []
        self._items: list[str] = []
        self._condition = threading.Condition()

    def recv(self) -> str | None:
        with self._condition:
            while not self.closed and not self._items:
                self._condition.wait(timeout=0.2)
            return self._items.pop(0) if self._items else None

    def send(self, payload: str) -> None:
        message = json.loads(payload)
        self.sent.append(message)
        if "id" in message:
            with self._condition:
                self._items.append(json.dumps({"id": message["id"], "result": {}}))
                self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self.closed = True
            self._condition.notify_all()


class RemoteCodexTests(unittest.TestCase):
    def test_close_wakes_real_websocket_reader_waiting_on_an_idle_peer(self) -> None:
        import websocket

        local, peer = socket.socketpair()
        connection = websocket.WebSocket(enable_multithread=True)
        connection.sock = local
        connection.connected = True
        with patch.object(websocket, "create_connection", return_value=connection):
            transport = WebSocketJsonRpcTransport("ws://unused")
        reader = threading.Thread(target=transport.recv, daemon=True)
        closer = threading.Thread(target=transport.close, daemon=True)
        reader.start()
        try:
            deadline = time.monotonic() + 1
            while not connection.frame_buffer.lock.locked() and time.monotonic() < deadline:
                time.sleep(.005)
            self.assertTrue(connection.frame_buffer.lock.locked())
            closer.start()
            closer.join(timeout=1)
            self.assertFalse(closer.is_alive(), "close waited on the blocked receiver's frame lock")
            reader.join(timeout=1)
            self.assertFalse(reader.is_alive())
        finally:
            # Bound the red case too: the idle peer's closure releases recv.
            peer.close()
            reader.join(timeout=4)
            if closer.ident is not None:
                closer.join(timeout=4)
            local.close()

    def test_websocket_connect_timeout_does_not_become_an_idle_read_timeout(self) -> None:
        class Socket:
            def __init__(self) -> None:
                self.timeout: float | None = 0.05
                self.closed = False

            def settimeout(self, value: float | None) -> None:
                self.timeout = value

            def recv(self) -> str:
                return '{"method":"idle"}'

            def send(self, payload: str) -> None:
                return None

            def close(self) -> None:
                self.closed = True

            def abort(self) -> None:
                pass

        socket = Socket()
        websocket = types.SimpleNamespace(create_connection=lambda *args, **kwargs: socket)
        with patch.dict(sys.modules, {"websocket": websocket}):
            transport = WebSocketJsonRpcTransport("ws://127.0.0.1:1", timeout=0.05)
            self.assertIsNone(socket.timeout)
            self.assertEqual('{"method":"idle"}', transport.recv())
            transport.close()

    def test_adapter_reuses_app_server_request_lifecycle_over_transport(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            transport = FakeTransport()
            adapter = VNextAppServerAdapter(
                codex_executable=Path(sys.executable), codex_home=root, workspace=root, transport=transport
            )
            try:
                self.assertEqual({}, adapter.initialize(timeout=1))
                self.assertEqual("initialize", transport.sent[0]["method"])
                adapter.notify("initialized", {})
                self.assertEqual("initialized", transport.sent[1]["method"])
            finally:
                cleanup = adapter.close()
            self.assertEqual("detached", cleanup.process.outcome)

    def test_terminal_spec_is_remote_resume_with_capability_token(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            server = RemoteCodexServer(
                workspace=root, codex_home=root, bind_host="0.0.0.0", require_auth=True
            )
            server._executable = Path(sys.executable)
            server._metadata = {"cli_version": "fixture"}
            spec = server.terminal_launch("thread-123")
            self.assertEqual(str(Path(sys.executable)), spec.executable)
            self.assertEqual(("--remote", spec.endpoint, "--remote-auth-token-env", "VNEXT_CODEX_REMOTE_TOKEN", "resume", "thread-123"), spec.arguments)
            self.assertTrue(spec.authenticated)
            self.assertIn("VNEXT_CODEX_REMOTE_TOKEN", spec.environment)

    def test_loopback_listener_cannot_claim_capability_token_auth(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(ValueError):
                RemoteCodexServer(workspace=root, codex_home=root, require_auth=True)

    def test_native_child_participation_is_an_explicit_server_capability(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            disabled = RemoteCodexServer(workspace=root, codex_home=root)
            enabled = RemoteCodexServer(workspace=root, codex_home=root, native_child_participation=True)

            self.assertEqual("--disable", disabled._start_command(Path(sys.executable))[2])
            self.assertEqual("--enable", enabled._start_command(Path(sys.executable))[2])
            self.assertEqual("multi_agent", enabled._start_command(Path(sys.executable))[3])

    def test_native_depth_override_is_explicit_and_process_local(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            default = RemoteCodexServer(workspace=root, codex_home=root, native_child_participation=True)
            self.assertFalse(any("max_depth" in arg for arg in default._start_command(Path(sys.executable))))
            configured = RemoteCodexServer(workspace=root, codex_home=root,
                native_child_participation=True, native_max_depth=2)
            command = configured._start_command(Path(sys.executable))
            at = command.index("agents.max_depth=2")
            self.assertEqual("-c", command[at - 1])
            self.assertFalse((root / "config.toml").exists())
            for value in (True, 0, -1, 2**31, "2", 1.5):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    RemoteCodexServer(workspace=root, codex_home=root,
                        native_child_participation=True, native_max_depth=value)
            with self.assertRaisesRegex(ValueError, "requires native"):
                RemoteCodexServer(workspace=root, codex_home=root, native_max_depth=2)

    def test_startup_mcp_configuration_is_process_local_and_toml_quoted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            server = RemoteCodexServer(workspace=root, codex_home=root,
                mcp_startup_config_overrides={"mcp_servers.vnext_relay.url": "http://127.0.0.1:123/mcp"})
            command = server._start_command(Path(sys.executable))
            self.assertEqual('-c', command[4])
            self.assertEqual('mcp_servers.vnext_relay.url="http://127.0.0.1:123/mcp"', command[5])
            self.assertEqual('app-server', command[6])
            self.assertFalse((root / "config.toml").exists())
            with self.assertRaises(ValueError):
                RemoteCodexServer(workspace=root, codex_home=root,
                    mcp_startup_config_overrides={"model": "foreign"})


if __name__ == "__main__":
    unittest.main()
