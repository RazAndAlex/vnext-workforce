from __future__ import annotations

import io
import json
import os
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class VNextWaitTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name).resolve()
        self.log = self.workspace / ".vnext" / "runs" / "session.jsonl"
        self.log.parent.mkdir(parents=True)
        self.clock = Clock()

    def append(self, agent, status, timestamp="2026-10-05T17:00:00+00:00", **payload):
        with self.log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "type": "agent.upsert", "session_id": "session", "agent_id": agent,
                "timestamp": timestamp, "payload": {"status": status, **payload},
            }) + "\n")

    def run_wait(self, *options, real_clock=False):
        from vnext.vnext_wait import main

        out, err = io.StringIO(), io.StringIO()
        kwargs = {} if real_clock else {"clock": self.clock, "sleep": self.clock.sleep}
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["--workspace", str(self.workspace), *options], **kwargs)
        return code, out.getvalue(), err.getvalue()

    def test_completed_exits_at_once_and_prints_the_stop_time(self):
        self.append("worker", "completed")
        code, out, err = self.run_wait("--agent", "worker")
        self.assertEqual(0, code)
        self.assertIn("worker completed 2026-10-05T17:00:00+00:00", out)
        self.assertEqual("", err)
        self.assertEqual(0, self.clock.now)

    def test_report_contains_latest_outcome_verification_evidence_and_usage(self):
        self.append("worker", "completed", result={"outcome": "Old report"})
        self.append("worker", "completed", "2026-10-05T17:01:00+00:00",
                    result={"outcome": "Fixed the issue.\nTests passed.",
                            "verified": True, "evidence": ["test run", "diff"]},
                    usage={"cost_tokens": {"total_tokens": 1234, "input_tokens": 1000,
                                           "output_tokens": 234}, "cost_usd": 0.25})
        code, out, err = self.run_wait("--agent", "worker")
        self.assertEqual(0, code)
        self.assertEqual("", err)
        self.assertEqual(
            "worker completed 2026-10-05T17:00:00+00:00\n"
            "Fixed the issue.\nTests passed.\n"
            "verified: yes, evidence: 2 item(s)\n"
            "tokens: 1234, cost_usd: 0.25\n", out)

    def test_blocker_is_used_when_outcome_is_blank(self):
        self.append("worker", "blocked", result={"outcome": "  "}, blocker="Need a decision.")
        code, out, _ = self.run_wait("--agent", "worker")
        self.assertEqual(0, code)
        self.assertIn("\nNeed a decision.\nverified: no, evidence: 0 item(s)\n", out)

    def test_missing_report_uses_each_status_fallback(self):
        from vnext.vnext_orchestration import _STOPPED_WITHOUT_A_WORD

        for status, fallback in _STOPPED_WITHOUT_A_WORD.items():
            with self.subTest(status=status):
                self.append("worker", status.value)
                code, out, _ = self.run_wait("--agent", "worker")
                self.assertEqual(0, code)
                self.assertIn("\n" + fallback + "\nverified: no, evidence: 0 item(s)\n", out)

    def test_long_outcome_is_bounded_and_names_its_session_outcome_log(self):
        from vnext.vnext_orchestration import OUTCOME_TEXT_LIMIT

        self.append("worker", "completed", result={"outcome": "x" * (OUTCOME_TEXT_LIMIT + 1)})
        code, out, _ = self.run_wait("--agent", "worker")
        self.assertEqual(0, code)
        report = out.splitlines()[1]
        self.assertEqual(OUTCOME_TEXT_LIMIT, len(report))
        self.assertTrue(report.endswith("… (cut; the full text is in .vnext/outcomes/session.jsonl)"))

    def test_reports_are_separated_and_only_available_usage_is_printed(self):
        self.append("one", "completed", result={"outcome": "First"}, usage={"cost_tokens": 0})
        self.append("two", "failed", blocker="Second", usage={"cost_usd": 0})
        code, out, _ = self.run_wait("--agent", "one", "--agent", "two")
        self.assertEqual(0, code)
        self.assertIn("tokens: 0\n\ntwo failed", out)
        self.assertTrue(out.endswith("cost_usd: 0\n"))

    def test_running_agent_completes_after_a_thread_appends_its_row(self):
        self.append("worker", "running")
        thread = threading.Thread(target=lambda: (
            time.sleep(0.03), self.append("worker", "completed")))
        thread.start()
        try:
            started = time.monotonic()
            code, out, _ = self.run_wait(
                "--agent", "worker", "--poll", "0.01", "--deadline", "0.5s",
                real_clock=True)
            self.assertEqual(0, code)
            self.assertIn("worker completed", out)
            self.assertLess(time.monotonic() - started, 0.5)
        finally:
            thread.join()

    def test_later_polls_parse_only_appended_rows(self):
        self.append("worker", "running")

        def advance(seconds):
            self.clock.now += seconds
            if self.clock.now == 2:
                self.append("worker", "completed")

        with patch("vnext.vnext_report.json.loads", wraps=json.loads) as parse:
            with patch.object(self.clock, "sleep", side_effect=advance):
                code, out, err = self.run_wait("--agent", "worker", "--poll", "1")
        self.assertEqual(0, code)
        self.assertIn("worker completed", out)
        self.assertEqual("", err)
        self.assertEqual(2, parse.call_count, "old rows must not be parsed again")

    def test_partial_trailing_line_is_completed_without_a_warning(self):
        self.append("worker", "running")
        completion = json.dumps({"type": "agent.upsert", "agent_id": "worker",
                                 "payload": {"status": "completed"}}).encode()
        cut = len(completion) // 2
        with self.log.open("ab") as handle:
            handle.write(completion[:cut])

        def finish_line(seconds):
            self.clock.now += seconds
            with self.log.open("ab") as handle:
                handle.write(completion[cut:] + b"\n")

        with patch.object(self.clock, "sleep", side_effect=finish_line):
            code, out, err = self.run_wait("--agent", "worker")
        self.assertEqual(0, code)
        self.assertIn("worker completed", out)
        self.assertEqual("", err)

    def test_session_reads_only_its_file_and_ignores_conflicting_status(self):
        self.append("worker", "completed")
        other = self.log.with_name("z-other-session.jsonl")
        other.write_text(json.dumps({"type": "agent.upsert", "agent_id": "worker",
                                     "payload": {"status": "running"}}) + "\n")
        real_open = Path.open

        def only_session(path, *args, **kwargs):
            if path != self.log.resolve():
                self.fail(f"session-scoped wait opened another file: {path}")
            return real_open(path, *args, **kwargs)

        with patch.object(Path, "open", only_session), patch(
            "vnext.vnext_wait._run_logs", side_effect=AssertionError("scanned all logs")
        ):
            code, out, err = self.run_wait("--agent", "worker", "--session", "session")
        self.assertEqual(0, code)
        self.assertIn("worker completed", out)
        self.assertEqual("", err)

    def test_incremental_reader_reads_only_appended_bytes_and_zero_when_unchanged(self):
        from vnext.vnext_wait import _LogCursor

        self.append("worker", "running")
        initial_size = self.log.stat().st_size
        cursor = _LogCursor()
        real_open = Path.open
        reads = []

        class CountingReader:
            def __enter__(self):
                self.handle = real_open(self.log, "rb")
                return self

            def __exit__(self, *args):
                self.handle.close()

            def seek(self, offset):
                return self.handle.seek(offset)

            def read(self, count):
                data = self.handle.read(count)
                reads.append(len(data))
                return data

        reader = CountingReader()
        reader.log = self.log
        with patch.object(Path, "open", return_value=reader) as opened:
            self.assertEqual(1, len(list(cursor.rows(self.log))))
            self.assertEqual(initial_size, sum(reads))
            reads.clear()
            self.assertEqual([], list(cursor.rows(self.log)))
            self.assertEqual(0, sum(reads))
            self.assertEqual(1, opened.call_count)
        self.append("worker", "completed")
        with patch.object(Path, "open", return_value=reader):
            rows = list(cursor.rows(self.log))
        self.assertEqual(["completed"], [row["payload"]["status"] for row in rows])
        self.assertEqual(self.log.stat().st_size - initial_size, sum(reads))

    def test_shrinking_file_resets_offset_partial_line_and_line_number(self):
        from vnext.vnext_wait import _LogCursor

        self.append("worker", "running")
        with self.log.open("ab") as handle:
            handle.write(b'{"unfinished":')
        cursor = _LogCursor()
        list(cursor.rows(self.log))
        self.log.write_bytes(b'{bad\n' + json.dumps({"type": "agent.upsert", "agent_id": "worker",
                                                 "payload": {"status": "completed"}}).encode() + b"\n")
        self.assertLess(self.log.stat().st_size, cursor.offset)
        rows = list(cursor.rows(self.log))
        self.assertEqual("line 1", rows[0]["where"])
        self.assertEqual("completed", rows[1]["payload"]["status"])
        self.assertEqual(b"", cursor.pending)

    def test_bad_complete_line_warns_only_once_across_polls(self):
        self.log.write_bytes(b'{bad\n')
        self.append("worker", "running")
        code, _, err = self.run_wait("--agent", "worker", "--poll", "1", "--deadline", "3s")
        self.assertEqual(3, code)
        self.assertEqual(1, err.count("skipped"))
        self.assertIn("line 1", err)

    def test_blocked_exits_at_once(self):
        self.append("worker", "blocked")
        code, out, _ = self.run_wait("--agent", "worker")
        self.assertEqual(0, code)
        self.assertIn("worker blocked", out)
        self.assertEqual(0, self.clock.now)

    def test_every_scheduler_terminal_status_exits_at_once(self):
        for status in ("failed", "cancelled", "replaced"):
            with self.subTest(status=status):
                self.append("worker", status)
                code, out, _ = self.run_wait("--agent", "worker")
                self.assertEqual(0, code)
                self.assertIn(f"worker {status}", out)
        self.assertEqual(0, self.clock.now)

    def test_deadline_exits_three(self):
        self.append("worker", "running")
        code, out, _ = self.run_wait("--agent", "worker", "--deadline", "0.1s")
        self.assertEqual(3, code)
        self.assertEqual("worker running still running after 0.1s\n", out)
        self.assertAlmostEqual(0.1, self.clock.now)

    def test_unknown_agent_exits_two_after_sixty_seconds_and_names_the_search(self):
        code, out, err = self.run_wait("--agent", "missing", "--deadline", "2m")
        self.assertEqual(2, code)
        self.assertEqual("", out)
        self.assertIn("missing", err)
        self.assertIn(str(self.log.parent), err)
        self.assertEqual(60, self.clock.now)

    def test_unknown_agent_at_a_short_deadline_is_an_error(self):
        code, _, err = self.run_wait("--agent", "missing", "--deadline", "0.1s")
        self.assertEqual(2, code)
        self.assertIn("missing", err)

    def test_deadline_names_only_the_agent_still_running(self):
        self.append("done", "completed")
        self.append("busy", "awaiting-workers")
        code, out, _ = self.run_wait(
            "--agent", "done", "--agent", "busy", "--deadline", "0.1s")
        self.assertEqual(3, code)
        self.assertNotIn("done", out)
        self.assertIn("busy awaiting-workers still running after 0.1s", out)

    def test_last_status_wins_over_an_earlier_completion(self):
        self.append("worker", "completed")
        self.append("worker", "running")
        code, _, _ = self.run_wait("--agent", "worker", "--deadline", "0.1s")
        self.assertEqual(3, code)

    def test_stop_time_is_the_status_transition_not_a_later_snapshot(self):
        self.append("worker", "completed")
        self.append("worker", "completed", "2026-10-05T17:01:00+00:00")
        code, out, _ = self.run_wait("--agent", "worker")
        self.assertEqual(0, code)
        self.assertIn("17:00:00", out)
        self.assertNotIn("17:01:00", out)

    def test_unreadable_line_is_skipped_and_reported(self):
        self.log.write_bytes(b'{torn\n\xff\n')
        self.append("worker", "completed")
        code, out, err = self.run_wait("--agent", "worker")
        self.assertEqual(0, code)
        self.assertIn("worker completed", out)
        self.assertIn("skipped", err)
        self.assertIn("line 1", err)
        self.assertIn("line 2", err)

    def test_unreadable_log_cannot_produce_success(self):
        self.append("worker", "completed")
        with patch.object(Path, "open", side_effect=PermissionError("denied")):
            code, _, err = self.run_wait("--agent", "worker", "--deadline", "0.1s")
        self.assertEqual(2, code)
        self.assertIn("denied", err)
        self.assertIn(str(self.log), err)

    def test_duration_units_are_seconds_minutes_and_hours(self):
        self.append("worker", "running")
        for duration, seconds in (("30s", 30), ("20m", 1200), ("1h", 3600)):
            with self.subTest(duration=duration):
                self.clock.now = 0
                code, out, _ = self.run_wait(
                    "--agent", "worker", "--deadline", duration, "--poll", str(seconds))
                self.assertEqual(3, code)
                self.assertEqual(seconds, self.clock.now)
                self.assertIn(f"still running after {duration}", out)

    def test_invalid_arguments_exit_two(self):
        for options in ((), ("--agent", ""), ("--agent", "worker", "--deadline", "bad"),
                        ("--agent", "worker", "--deadline", "0s"),
                        ("--agent", "worker", "--poll", "0"),
                        ("--agent", "worker", "--poll", "nan"),
                        ("--agent", "worker", "--poll", "inf"),
                        ("--agent", "worker", "--session", "../another"),
                        ("--agent", "worker", "--session", "C:another")):
            with self.subTest(options=options):
                code, _, err = self.run_wait(*options)
                self.assertEqual(2, code)
                self.assertIn("error", err)

    def test_workspace_defaults_to_cwd(self):
        from vnext.vnext_wait import main

        self.append("worker", "completed")
        with patch("pathlib.Path.cwd", return_value=self.workspace), redirect_stdout(io.StringIO()):
            self.assertEqual(0, main(["--agent", "worker"]))

    def test_module_and_console_entry_points(self):
        self.append("worker", "completed")
        result = subprocess.run([
            sys.executable, "-m", "vnext.vnext_wait", "--workspace",
            str(self.workspace), "--agent", "worker", "--deadline", "30s"],
            capture_output=True, text=True, timeout=2)
        self.assertEqual(0, result.returncode, result.stderr)
        config = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
        self.assertEqual("vnext.vnext_wait:main", config["project"]["scripts"]["vnext-wait"])

    def test_external_delegate_wake_command_quotes_the_session_workspace(self):
        from vnext.vnext_mcp_server import VNextMcpService, _external_tools
        from vnext.vnext_runtime_types import ToolCallResult
        from vnext.vnext_scheduler import VNextScheduler

        from vnext.vnext_mcp_server import NO_UPDATE_CHECK_ENV

        workspace = self.workspace / "vNext"
        workspace.mkdir()
        runtime = Mock()
        runtime.external_tools.return_value = []
        # Exercise real file selection and _record without starting a runtime,
        # checking installed providers, or fetching update metadata.
        with patch.dict(os.environ, {NO_UPDATE_CHECK_ENV: "1"}), patch(
            "vnext.vnext_mcp_server._validate_catalog", return_value=[]
        ), patch("vnext.vnext_mcp_server.read_active_runtime", return_value=None), patch(
            "vnext.vnext_mcp_server.check_for_updates", return_value=None
        ), patch("vnext.vnext_mcp_server.runtime_notice_lines", return_value=[]), patch(
            "vnext.vnext_mcp_server.session_runtime_config", return_value={}
        ), patch("vnext.vnext_mcp_server.CodeFreshness"), patch(
            "vnext.vnext_mcp_server.VNextRuntimeSession", return_value=runtime
        ), patch.object(VNextMcpService, "_publish_status"):
            service = VNextMcpService(workspace=workspace, session_id="serving-session", catalog=[])
        service._remember = Mock()
        service._publish_status = Mock()
        service._write_succeeded = Mock()
        service._closing = False
        original = ToolCallResult(True, {"agent_id": "child-123", "workspace_path": "worker tree"})
        service.session = SimpleNamespace(external_tool_call=Mock(return_value=original))
        with patch("vnext.vnext_mcp_server.sys.executable", "/Python Install/bin/python"):
            reply = service._dispatch("delegate", {"workspace": "worktree"}, {})
        command = [
            "/Python Install/bin/python", "-m", "vnext.vnext_wait",
            "--workspace", str(service.workspace.resolve()), "--agent", "child-123",
            "--session", service._session_id, "--deadline", "30m"]
        # shlex.split drops Windows backslashes, so compare the quoted string.
        quote = subprocess.list2cmdline if os.name == "nt" else shlex.join
        self.assertEqual(quote(command), reply.value["wake_command"])
        service._record(SimpleNamespace(type="agent.upsert", agent_id="child-123",
                                        payload={"agent_id": "child-123", "status": "completed"}))
        named_log = Path(command[command.index("--workspace") + 1]) / ".vnext" / "runs" / (
            command[command.index("--session") + 1] + ".jsonl")
        self.assertEqual(service._event_log.resolve(), named_log)
        recorded = json.loads(named_log.read_text())
        self.assertEqual("child-123", recorded["agent_id"])
        self.assertEqual(service._session_id, recorded["session_id"])
        code, _, err = self.run_wait("--workspace", str(service.workspace), "--agent", "child-123",
                                    "--session", service._session_id)
        self.assertEqual(0, code, err)
        self.assertNotIn("wake_command", original.value)
        internal = next(t for t in VNextScheduler.manager_tools() if t["name"] == "delegate")
        external = next(t for t in _external_tools([internal]) if t["name"] == "delegate")
        self.assertIn("background", external["description"])
        self.assertNotIn("wake_command", internal["description"])

    def test_rejected_external_delegate_has_no_wake_command(self):
        from vnext.vnext_mcp_server import VNextMcpService
        from vnext.vnext_runtime_types import ToolCallResult

        service = VNextMcpService.__new__(VNextMcpService)
        service._closing = False
        service.workspace = self.workspace
        service._session_id = "session"
        service.session = SimpleNamespace(external_tool_call=Mock(
            return_value=ToolCallResult(False, {"error": "unknown model"})))
        self.assertNotIn("wake_command", service._dispatch("delegate", {}, {}).value)
        service.session.external_tool_call.return_value = ToolCallResult(True, {"agent_id": "worker"})
        self.assertIn("wake_command", service._dispatch("delegate", {}, {}).value)


if __name__ == "__main__":
    unittest.main()
