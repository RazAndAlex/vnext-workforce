from __future__ import annotations

import tempfile
import time
import unittest
from types import SimpleNamespace

from vnext.host_contract import SessionStartRequest
from vnext.process_supervisor import ProcessCleanup
from vnext.vnext_session_runtime import VNextRuntimeSession
from tests.reply_adapter_fixture import ReplyAdapter


class TerminalAdapter(ReplyAdapter):
    def terminal_launch(self, thread_id):
        return SimpleNamespace(executable="test-harness", arguments=("resume", thread_id), environment={})


class FakePTY:
    created = []
    def __init__(self, command, **kwargs):
        self.command = command
        self.on_event = kwargs["on_event"]
        self.closed = False
        self.input = []
        self.created.append(self)
    def close(self):
        self.closed = True
        return ProcessCleanup("clean", 0, 0)
    def write(self, data):
        self.input.append(data)
    def resize(self, columns, rows):
        self.size = (columns, rows)


class NativeLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.events = []
        self.adapter = TerminalAdapter()
        FakePTY.created = []
        self.runtime = VNextRuntimeSession(SessionStartRequest(
            "native-session", self.directory.name, "primary", {"provider": "codex", "model": "model"},
            config={"codex_transport": "websocket"}), self.events.append,
            adapter_factories={"codex": lambda: self.adapter}, terminal_process_factory=FakePTY)
        self.runtime.start()
    def tearDown(self):
        self.runtime.close()
        self.directory.cleanup()
    def prepare(self):
        self.runtime.prompt("primary", "A real first request", "first")
        deadline = time.monotonic() + 3
        while not any(event.type == "objective.completed" for event in self.events) and time.monotonic() < deadline:
            time.sleep(.01)
        # Provider finalization follows the explicit completion tool.
        deadline = time.monotonic() + 3
        while self.runtime.root.active_turn_id is not None and time.monotonic() < deadline:
            time.sleep(.01)
    def test_native_first_attach_does_not_spend_a_dummy_turn(self):
        with self.assertRaisesRegex(ValueError, "first chat prompt"):
            self.runtime.attach_terminal("primary", "term", "attach", mode="native")
        self.assertIsNone(self.runtime._thread)
        self.assertFalse(FakePTY.created)
    def test_detach_and_reattach_keep_process_identity_and_lease(self):
        self.prepare()
        result = self.runtime.attach_terminal("primary", "term", "attach", mode="native")
        self.assertEqual(result.mode, "native")
        process = FakePTY.created[0]
        self.runtime.detach_terminal("term", "detach")
        self.assertFalse(process.closed)
        self.assertIn("primary", self.runtime._scheduler._native_control_leases)
        self.runtime.attach_terminal("primary", "term", "reattach", mode="native")
        self.assertEqual(len(FakePTY.created), 1)
        self.runtime.terminal_command("term", "terminal_input", {"data": "hello\r"}, "input")
        self.assertEqual(process.input, ["hello\r"])
        self.runtime.terminal_command("term", "terminal_stop", {}, "stop")
        self.assertTrue(process.closed)
        self.assertNotIn("primary", self.runtime._scheduler._native_control_leases)
    def test_failed_process_start_releases_lease(self):
        self.prepare()
        def fail(*args, **kwargs):
            raise RuntimeError("spawn failed")
        self.runtime._terminal_process_factory = fail
        with self.assertRaisesRegex(RuntimeError, "spawn failed"):
            self.runtime.attach_terminal("primary", "term", "attach", mode="native")
        self.assertNotIn("primary", self.runtime._scheduler._native_control_leases)

    def test_terminal_lease_wakeups_keep_session_scheduler_running(self):
        self.prepare()
        self.runtime.attach_terminal("primary", "term", "attach", mode="native")
        self.runtime.terminal_command("term", "terminal_stop", {}, "stop")
        # The real scheduler consumes both lease notifications asynchronously.
        deadline = time.monotonic() + .5
        while time.monotonic() < deadline and not self.runtime._shutdown.is_set():
            time.sleep(.01)
        self.assertFalse(self.runtime._shutdown.is_set(),
                         [event.payload for event in self.events if event.type == "session.error"])
        self.assertTrue(self.runtime._thread.is_alive())
    def test_native_output_is_separate_from_provider_semantic_history(self):
        self.prepare()
        self.runtime.attach_terminal("primary", "term", "attach", mode="native")
        FakePTY.created[0].on_event({"type": "output", "data": "screen redraw"})
        output = [event for event in self.events if event.type == "terminal.output"]
        self.assertEqual(output[-1].payload, {"terminal_id": "term", "data": "screen redraw"})
        self.assertFalse(any("screen redraw" in str(item) for item in self.runtime.history("primary")))

    def test_failed_cleanup_keeps_lease_and_reports_stop_failure(self):
        self.prepare()
        self.runtime.attach_terminal("primary", "term", "attach", mode="native")
        process = FakePTY.created[0]
        real_close = process.close
        process.close = lambda: ProcessCleanup("residual", 1, None, ("still running",))
        try:
            with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
                self.runtime.terminal_command("term", "terminal_stop", {}, "stop")
            self.assertIn("primary", self.runtime._scheduler._native_control_leases)
            with self.assertRaisesRegex(ValueError, "already has a native terminal"):
                self.runtime.attach_terminal("primary", "other", "attach", mode="native")
        finally:
            process.close = real_close

    def test_session_close_reports_terminal_cleanup_failure(self):
        self.prepare()
        self.runtime.attach_terminal("primary", "term", "attach", mode="native")
        FakePTY.created[0].close = lambda: ProcessCleanup("residual", 1, None, ("still running",))
        with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
            self.runtime.close()
        # Failure was asserted here; keep the generic teardown from rethrowing it.
        self.runtime._close_error = None


if __name__ == "__main__":
    unittest.main()
