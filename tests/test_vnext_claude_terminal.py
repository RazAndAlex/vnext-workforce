from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.request import Request, urlopen

from vnext.vnext_claude_terminal import (
    ClaudeTerminalError,
    ClaudeTerminalLeaseManager,
    ClaudeTerminalRelayEvent,
    ClaudeTerminalRelayServer,
)


class BearerComparisonTests(unittest.TestCase):
    def test_bearer_comparison_uses_compare_digest(self) -> None:
        import vnext.vnext_claude_terminal as terminal
        from types import SimpleNamespace

        relay = object.__new__(ClaudeTerminalRelayServer)
        relay._condition = threading.Condition()
        relay._leases = {"lease-1": SimpleNamespace(token="expected-token", accepting=True)}
        request = SimpleNamespace(path="/leases/lease-1/mcp", headers={"Authorization": "Bearer guessed"})
        with patch.object(terminal.secrets, "compare_digest", wraps=terminal.secrets.compare_digest) as compare, \
             patch.object(relay, "_respond") as respond:
            relay._receive(request)
            request.headers = {}
            relay._receive(request)
        compare.assert_any_call(b"Bearer guessed", b"Bearer expected-token")
        compare.assert_any_call(b"", b"Bearer expected-token")
        self.assertEqual(2, compare.call_count)
        self.assertEqual(2, respond.call_count)


class ClaudeTerminalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.events: list[ClaudeTerminalRelayEvent] = []

        def handler(event: ClaudeTerminalRelayEvent) -> dict[str, object]:
            self.events.append(event)
            if event.channel == "mcp":
                return {"jsonrpc": "2.0", "id": event.payload.get("id"), "result": {"accepted": event.payload["method"]}}
            return {"continue": True, "hookSpecificOutput": {"hookEventName": event.payload.get("hook_event_name")}}

        self.relay = ClaudeTerminalRelayServer(handler)
        self.relay.start()
        self.addCleanup(self.relay.close)

    def _acquire(self, root: Path):
        durable = root / "durable-claude-config"
        durable.mkdir()
        manager = ClaudeTerminalLeaseManager(root / "leases", self.relay)
        return manager, manager.acquire(
            runtime_thread_id="runtime-1",
            native_session_id="6044886c-211e-4e8e-a660-10f3bf0ed1b5",
            model="claude-opus-4-7",
            effort="high",
            claude_config_dir=durable,
            sdk_turn_active=False,
        )

    def test_idle_lease_writes_external_mcp_and_hook_relays_without_exposing_token(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager, lease = self._acquire(Path(temporary))
            mcp = json.loads(Path(lease.launch.mcp_config_path).read_text(encoding="utf-8"))
            settings = json.loads(Path(lease.launch.settings_path).read_text(encoding="utf-8"))
            private_relay = json.loads((lease.directory / "relay.json").read_text(encoding="utf-8"))

            self.assertEqual(sys.executable, mcp["mcpServers"]["vnext"]["command"])
            self.assertEqual(str(Path(__file__).parents[1] / "vnext/vnext_claude_terminal.py"), mcp["mcpServers"]["vnext"]["args"][0])
            self.assertIn("PreToolUse", settings["hooks"])
            self.assertIn("SubagentStart", settings["hooks"])
            self.assertIn("SubagentStop", settings["hooks"])
            hook = settings["hooks"]["PreToolUse"][0]["hooks"][0]
            self.assertEqual(sys.executable, hook["command"])
            self.assertEqual([str(Path(__file__).parents[1] / "vnext/vnext_claude_terminal.py"), "hook"], hook["args"][:2])
            self.assertNotIn(private_relay["token"], repr(lease.launch))
            self.assertEqual("full-control-map-unavailable", lease.launch.native_children)
            self.assertIn("--resume=6044886c-211e-4e8e-a660-10f3bf0ed1b5", lease.launch.arguments)
            self.assertIn("--setting-sources=user,project,local", lease.launch.arguments)
            self.assertIn("--permission-mode=default", lease.launch.arguments)
            self.assertEqual("runtime-1", manager.release(lease.lease_id, terminal_stopped=True).runtime_thread_id)

    def test_generated_relays_use_owned_source_from_a_shadowing_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager, lease = self._acquire(root)
            workspace = root / "workspace"
            shadow = workspace / "vnext"
            shadow.mkdir(parents=True)
            (shadow / "__init__.py").write_text("")
            (shadow / "vnext_claude_terminal.py").write_text("raise SystemExit(23)\n")
            mcp = json.loads(Path(lease.launch.mcp_config_path).read_text())["mcpServers"]["vnext"]
            hook = json.loads(Path(lease.launch.settings_path).read_text())["hooks"]["UserPromptSubmit"][0]["hooks"][0]
            for command, payload in ((mcp, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}),
                                     (hook, {"hook_event_name": "UserPromptSubmit"})):
                with self.subTest(channel="mcp" if command is mcp else "hook"):
                    result = subprocess.run([command["command"], *command["args"]], cwd=workspace,
                        input=json.dumps(payload)+"\n", text=True, capture_output=True, timeout=10)
                    self.assertEqual(0, result.returncode, result.stderr)
                    self.assertTrue(json.loads(result.stdout))
            self.assertEqual(["mcp", "hook"], [e.channel for e in self.events])
            manager.release(lease.lease_id, terminal_stopped=True)

    def test_handoff_refuses_active_sdk_unattested_identity_and_double_owner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            durable = root / "durable"
            durable.mkdir()
            manager = ClaudeTerminalLeaseManager(root / "leases", self.relay)
            common = {
                "runtime_thread_id": "runtime-1", "native_session_id": "not-attested", "model": "claude-opus-4-7",
                "effort": "high", "claude_config_dir": durable, "sdk_turn_active": False,
            }
            with self.assertRaisesRegex(ClaudeTerminalError, "attested UUID"):
                manager.acquire(**common)
            common["native_session_id"] = "6044886c-211e-4e8e-a660-10f3bf0ed1b5"
            common["permission_mode"] = "bypassPermissions"
            with self.assertRaisesRegex(ClaudeTerminalError, "permission mode"):
                manager.acquire(**common)
            common["permission_mode"] = "default"
            common["sdk_turn_active"] = True
            with self.assertRaisesRegex(ClaudeTerminalError, "SDK turn is active"):
                manager.acquire(**common)
            common["sdk_turn_active"] = False
            lease = manager.acquire(**common)
            with self.assertRaisesRegex(ClaudeTerminalError, "already has a terminal lease"):
                manager.acquire(**common)
            with self.assertRaisesRegex(ClaudeTerminalError, "before terminal stop"):
                manager.release(lease.lease_id, terminal_stopped=False)
            with self.assertRaisesRegex(ClaudeTerminalError, "different Claude native session"):
                manager.release(lease.lease_id, terminal_stopped=True, resumed_native_session_id="b7bb4f30-6d7e-4c8a-8f9c-aa227548f634")
            manager.release(lease.lease_id, terminal_stopped=True, resumed_native_session_id=lease.native_session_id)

    def test_stdio_mcp_and_hook_relay_reach_injected_host_handler(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _manager, lease = self._acquire(Path(temporary))
            relay_config = lease.directory / "relay.json"
            mcp = subprocess.run(
                [sys.executable, "-m", "vnext.vnext_claude_terminal", "mcp", "--lease-config", str(relay_config)],
                input="\n".join(
                    json.dumps(payload)
                    for payload in (
                        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
                        {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
                        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
                        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "delegate"}},
                    )
                ) + "\n",
                text=True, capture_output=True, check=True, timeout=10,
            )
            hook = subprocess.run(
                [sys.executable, "-m", "vnext.vnext_claude_terminal", "hook", "--lease-config", str(relay_config)],
                input=json.dumps({"hook_event_name": "PreToolUse", "tool_name": "Agent"}),
                text=True, capture_output=True, check=True, timeout=10,
            )
            responses = [json.loads(line) for line in mcp.stdout.splitlines()]
            self.assertEqual([1, 2, 3], [response["id"] for response in responses])
            self.assertEqual({"accepted": "tools/call"}, responses[-1]["result"])
            self.assertTrue(json.loads(hook.stdout)["continue"])
            self.assertEqual(["mcp", "mcp", "mcp", "mcp", "hook"], [event.channel for event in self.events])
            self.assertEqual("Agent", self.events[-1].payload["tool_name"])

    def test_release_drains_admitted_callback_and_removes_private_token_file(self) -> None:
        started, finish, released = threading.Event(), threading.Event(), threading.Event()

        def handler(_event: ClaudeTerminalRelayEvent) -> dict[str, object]:
            started.set()
            self.assertTrue(finish.wait(3))
            return {"jsonrpc": "2.0", "id": 1, "result": {}}

        relay = ClaudeTerminalRelayServer(handler)
        relay.start()
        self.addCleanup(relay.close)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager = ClaudeTerminalLeaseManager(root / "leases", relay)
            lease = manager.acquire(
                runtime_thread_id="runtime-1", native_session_id="6044886c-211e-4e8e-a660-10f3bf0ed1b5",
                model="claude-opus-4-7", effort="high", sdk_turn_active=False,
            )
            config = json.loads((lease.directory / "relay.json").read_text(encoding="utf-8"))
            request = Request(
                f"{config['endpoint']}/leases/{lease.lease_id}/mcp", data=b'{"jsonrpc":"2.0","id":1,"method":"tools/list"}',
                headers={"Authorization": f"Bearer {config['token']}", "Content-Type": "application/json"}, method="POST",
            )
            client = threading.Thread(target=lambda: urlopen(request, timeout=5).read())  # nosec B310 - loopback fixture
            client.start()
            self.assertTrue(started.wait(2))
            release = threading.Thread(target=lambda: (manager.release(lease.lease_id, terminal_stopped=True), released.set()))
            release.start()
            time.sleep(0.1)
            self.assertFalse(released.is_set())
            finish.set()
            client.join(3)
            release.join(3)
            self.assertFalse(client.is_alive())
            self.assertTrue(released.is_set())
            self.assertFalse((lease.directory / "relay.json").exists())

    def test_relay_lifecycle_is_loopback_only(self) -> None:
        relay = ClaudeTerminalRelayServer(lambda _event: {})
        relay.close()  # Closing before start must not wait for serve_forever.
        with self.assertRaisesRegex(ValueError, "loopback"):
            ClaudeTerminalRelayServer(lambda _event: {}, host="0.0.0.0")

    def test_mcp_relay_error_keeps_the_original_request_id(self) -> None:
        relay = ClaudeTerminalRelayServer(lambda _event: (_ for _ in ()).throw(RuntimeError("host unavailable")))
        relay.start()
        self.addCleanup(relay.close)
        with tempfile.TemporaryDirectory() as temporary:
            manager = ClaudeTerminalLeaseManager(Path(temporary) / "leases", relay)
            lease = manager.acquire(
                runtime_thread_id="runtime-1", native_session_id="6044886c-211e-4e8e-a660-10f3bf0ed1b5",
                model="claude-opus-4-7", effort="high", sdk_turn_active=False,
            )
            result = subprocess.run(
                [sys.executable, "-m", "vnext.vnext_claude_terminal", "mcp", "--lease-config", str(lease.directory / "relay.json")],
                input='{"jsonrpc":"2.0","id":"keep-me","method":"tools/list"}\n', text=True, capture_output=True, check=True, timeout=10,
            )
            response = json.loads(result.stdout)
            self.assertEqual("keep-me", response["id"])
            self.assertIn("error", response)

    def test_acquire_write_failure_unregisters_relay_and_removes_partial_lease(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager = ClaudeTerminalLeaseManager(root / "leases", self.relay)
            with patch.object(manager, "_write_private", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    manager.acquire(
                        runtime_thread_id="runtime-1", native_session_id="6044886c-211e-4e8e-a660-10f3bf0ed1b5",
                        model="claude-opus-4-7", effort="high", sdk_turn_active=False,
                    )
            self.assertEqual([], list((root / "leases").iterdir()))


if __name__ == "__main__":
    unittest.main()
