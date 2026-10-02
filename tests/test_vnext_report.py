from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from vnext.vnext_report import main, read_workspace


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(row) + "\n" for row in rows)
    path.write_text(body, encoding="utf-8", newline="\n")


class VNextReportTests(unittest.TestCase):
    def test_a_workspace_that_never_ran_says_so_instead_of_failing(self):
        with tempfile.TemporaryDirectory() as empty:
            report = read_workspace(Path(empty))
            self.assertFalse(report["exists"])
            self.assertEqual([], report["sessions"])
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(0, main([empty]))
            self.assertIn("never run a workforce", out.getvalue())

    def test_a_path_that_is_not_there_is_an_error_and_not_an_empty_workspace(self):
        """A watcher kept passing after the project was renamed."""

        with tempfile.TemporaryDirectory() as parent:
            missing = str(Path(parent) / "nope")
            for argv in ([missing], [missing, "--failures"]):
                out, err = io.StringIO(), io.StringIO()
                with redirect_stdout(out), redirect_stderr(err):
                    self.assertEqual(2, main(argv))
                self.assertIn("does not exist", err.getvalue())
                self.assertEqual("", out.getvalue())

    def test_a_file_given_where_a_folder_belongs_is_the_same_error(self):
        with tempfile.TemporaryDirectory() as parent:
            given = Path(parent) / "notes.txt"
            given.write_text("not a project\n", encoding="utf-8")
            err = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(err):
                self.assertEqual(2, main([str(given)]))
            self.assertIn("is not a folder", err.getvalue())

    def test_a_nested_server_is_named_under_the_session_that_started_it(self):
        """A Claude worker runs its own vnext server in the same project."""

        with tempfile.TemporaryDirectory() as workspace:
            home = Path(workspace) / ".vnext"
            for session in ("parent01", "nested02"):
                _write(home / "runs" / f"{session}.jsonl", [
                    {"type": "session.upsert", "session_id": session,
                     "payload": {"status": "completed"}},
                ])
            status = home / "status"
            status.mkdir(parents=True)
            (status / "nested02.json").write_text(
                json.dumps({"session_id": "nested02", "parent_session": "parent01"}),
                encoding="utf-8")
            (status / "parent01.json").write_text(
                json.dumps({"session_id": "parent01"}), encoding="utf-8")

            report = read_workspace(Path(workspace))
            parents = {s["session_id"]: s["parent"] for s in report["sessions"]}
            self.assertEqual({"parent01": None, "nested02": "parent01"}, parents)

            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(0, main([workspace]))
            printed = out.getvalue()
            self.assertIn("(started inside session parent01)", printed)
            self.assertEqual(1, printed.count("started inside session"))

    def test_a_failure_is_reported_with_the_cause_that_names_it(self):
        with tempfile.TemporaryDirectory() as workspace:
            home = Path(workspace) / ".vnext"
            _write(home / "runs" / "abcdef1234.jsonl", [
                {"type": "session.upsert", "session_id": "abcdef1234",
                 "payload": {"status": "ready"}},
                {"type": "agent.spawned", "session_id": "abcdef1234", "agent_id": "w-1"},
                {"type": "provider.error", "session_id": "abcdef1234", "agent_id": "w-1",
                 "payload": {"provider": "claude", "model_id": "sonnet-5",
                             "error": "the Claude SDK could not be imported",
                             "cause": "ImportError: No module named 'anyio'",
                             "provider_stderr": ["ImportError: anyio"],
                             "traceback": "Traceback (most recent call last):\n  ..."}},
            ])
            out = io.StringIO()
            with redirect_stdout(out):
                # A failure exits non-zero so a watcher can ask this in a script.
                self.assertEqual(1, main([workspace, "--failures"]))
            printed = out.getvalue()
            self.assertIn("FAILED", printed)
            self.assertIn("the Claude SDK could not be imported", printed)
            self.assertIn("ImportError: No module named 'anyio'", printed)
            self.assertIn("stderr: ImportError: anyio", printed)
            self.assertNotIn("Traceback (most recent call last)", printed)

            full = io.StringIO()
            with redirect_stdout(full):
                main([workspace, "--failures", "--full"])
            self.assertIn("Traceback (most recent call last)", full.getvalue())

    def test_a_torn_record_is_counted_rather_than_ending_the_read(self):
        """Two servers appending to one log used to make the log unreadable."""

        with tempfile.TemporaryDirectory() as workspace:
            log = Path(workspace) / ".vnext" / "runs" / "torn.jsonl"
            log.parent.mkdir(parents=True)
            log.write_text(
                '{"type": "agent.spawned", "session_id": "torn", "agent_id": "w-1"}\n'
                '{"type": "session.up\n'
                '{"type": "session.upsert", "session_id": "torn", '
                '"payload": {"status": "completed"}}\n',
                encoding="utf-8", newline="\n")
            report = read_workspace(Path(workspace))
            session = report["sessions"][0]
            self.assertEqual("completed", session["status"])
            self.assertEqual(1, session["unreadable"])
            self.assertEqual(1, len(session["agents"]))

    def test_unreadable_record_names_line_and_json_cause(self):
        with tempfile.TemporaryDirectory() as workspace:
            log = Path(workspace) / ".vnext" / "runs" / "broken.jsonl"
            log.parent.mkdir(parents=True)
            log.write_text('{"type":"session.upsert"}\n{broken\n', encoding="utf-8")
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(1, main([workspace]))
            printed = out.getvalue()
            self.assertIn("unreadable record at line 2:", printed)
            self.assertIn("Expecting property name", printed)
            self.assertNotIn("overlapping appends", printed)

    def test_failures_mode_names_a_record_it_could_not_read(self):
        """A torn last line can hold the 401 that ended the session."""

        with tempfile.TemporaryDirectory() as workspace:
            log = Path(workspace) / ".vnext" / "runs" / "torn.jsonl"
            log.parent.mkdir(parents=True)
            log.write_text(
                '{"type": "session.upsert", "session_id": "torn", '
                '"payload": {"status": "running"}}\n'
                '{"type": "provider.error", "error": {"api_error_st\x00\x00\n',
                encoding="utf-8", newline="\n")
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(1, main([workspace, "--failures"]))
            printed = out.getvalue()
            self.assertIn("PART UNREADABLE", printed)
            self.assertIn("unreadable record at line 2:", printed)
            self.assertNotIn("no failures recorded", printed)

    def test_a_log_that_cannot_be_opened_counts_as_unreadable(self):
        """R21: a log the reader could not open gave "no failures recorded"."""

        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as workspace:
            log = Path(workspace) / ".vnext" / "runs" / "locked.jsonl"
            _write(log, [{"type": "session.upsert", "session_id": "locked",
                          "payload": {"status": "completed"}}])
            real = Path.read_bytes

            def refuse(path, *args, **kwargs):
                if path.name == log.name:
                    raise PermissionError(13, "Permission denied")
                return real(path, *args, **kwargs)

            for argv in ([workspace], [workspace, "--failures"]):
                with self.subTest(argv=argv):
                    out = io.StringIO()
                    with patch.object(Path, "read_bytes", refuse), redirect_stdout(out):
                        self.assertEqual(1, main(argv))
                    printed = out.getvalue()
                    self.assertIn("PART UNREADABLE", printed)
                    self.assertIn("Permission denied", printed)
                    self.assertNotIn("no failures recorded", printed)

    def test_a_line_that_is_json_but_no_record_counts_as_unreadable(self):
        """R21: a ``null`` line was dropped and the session read clean."""

        with tempfile.TemporaryDirectory() as workspace:
            log = Path(workspace) / ".vnext" / "runs" / "odd.jsonl"
            log.parent.mkdir(parents=True)
            log.write_text(
                '{"type": "session.upsert", "session_id": "odd", '
                '"payload": {"status": "completed"}}\nnull\n[1, 2]\n',
                encoding="utf-8", newline="\n")
            for argv in ([workspace], [workspace, "--failures"]):
                with self.subTest(argv=argv):
                    out = io.StringIO()
                    with redirect_stdout(out):
                        self.assertEqual(1, main(argv))
                    printed = out.getvalue()
                    self.assertIn("PART UNREADABLE", printed)
                    self.assertIn("unreadable record at line 2:", printed)
                    self.assertIn("unreadable record at line 3:", printed)

    def test_an_unreadable_outcome_row_counts_as_unreadable(self):
        """R22: a torn or ``null`` outcome row vanished and the run read clean."""

        with tempfile.TemporaryDirectory() as workspace:
            home = Path(workspace) / ".vnext"
            _write(home / "runs" / "probe.jsonl", [
                {"type": "session.upsert", "session_id": "probe", "payload": {"status": "completed"}},
            ])
            (home / "outcomes").mkdir(parents=True)
            (home / "outcomes" / "probe.jsonl").write_text(
                'null\n{"agent_id": "w-1", "status": "compl\n', encoding="utf-8", newline="\n")
            for argv in ([workspace], [workspace, "--failures"]):
                with self.subTest(argv=argv):
                    out = io.StringIO()
                    with redirect_stdout(out):
                        self.assertEqual(1, main(argv))
                    printed = out.getvalue()
                    self.assertIn("outcomes/probe.jsonl: unreadable record at line 1:", printed)
                    self.assertIn("outcomes/probe.jsonl: unreadable record at line 2:", printed)
                    self.assertNotIn("no failures recorded", printed)

    def test_an_outcome_row_with_bad_fields_counts_as_unreadable(self):
        """R23: a row with no agent was skipped, and a list or NaN price crashed or printed $nan."""

        for row, cause in (
            ('{"status": "completed", "cost_usd": 4.5}', "names no agent"),
            ('{"agent_id": "w", "cost_usd": 1, "cost_source": []}', "cost_source"),
            ('{"agent_id": "w", "cost_usd": NaN, "cost_source": "provider"}', "cost_usd"),
            ('{"agent_id": "w", "cost_usd": "1.5"}', "cost_usd"),
            # R24: a huge integer crashed the reader, and damaged claim fields counted.
            ('{"agent_id": "w", "cost_usd": 1' + "0" * 400 + '}', "cost_usd"),
            ('{"agent_id": "w", "status": ["completed"]}', "status"),
            ('{"agent_id": "w", "status": "completed", "claimed_verified": "false"}', "claimed_verified"),
        ):
            with self.subTest(row=row), tempfile.TemporaryDirectory() as workspace:
                home = Path(workspace) / ".vnext"
                _write(home / "runs" / "probe.jsonl", [
                    {"type": "session.upsert", "session_id": "probe", "payload": {"status": "completed"}},
                ])
                (home / "outcomes").mkdir(parents=True)
                (home / "outcomes" / "probe.jsonl").write_text(row + "\n", encoding="utf-8", newline="\n")
                for argv in ([workspace], [workspace, "--failures"]):
                    out = io.StringIO()
                    with redirect_stdout(out):
                        self.assertEqual(1, main(argv))
                    printed = out.getvalue()
                    self.assertIn("outcomes/probe.jsonl: unreadable record at line 1:", printed)
                    self.assertIn(cause, printed)
                    self.assertNotIn("$nan", printed)

    def test_costs_that_add_up_past_a_float_count_as_unreadable(self):
        """R25: two finite rows of 1e308 printed $inf and exited 0."""

        with tempfile.TemporaryDirectory() as workspace:
            home = Path(workspace) / ".vnext"
            _write(home / "runs" / "probe.jsonl", [
                {"type": "session.upsert", "session_id": "probe", "payload": {"status": "completed"}},
            ])
            (home / "outcomes").mkdir(parents=True)
            (home / "outcomes" / "probe.jsonl").write_text(
                '{"agent_id": "a", "status": "completed", "cost_usd": 1e308}\n'
                '{"agent_id": "b", "status": "completed", "cost_usd": 1e308}\n',
                encoding="utf-8", newline="\n")
            for argv in ([workspace], [workspace, "--failures"]):
                with self.subTest(argv=argv):
                    out = io.StringIO()
                    with redirect_stdout(out):
                        self.assertEqual(1, main(argv))
                    printed = out.getvalue()
                    self.assertNotIn("$inf", printed)
                    self.assertIn("add up to more than", printed)

    def test_a_line_that_is_not_utf8_counts_as_unreadable(self):
        """R22: a damaged byte turned provider.error into another word, silently."""

        with tempfile.TemporaryDirectory() as workspace:
            log = Path(workspace) / ".vnext" / "runs" / "bytes.jsonl"
            log.parent.mkdir(parents=True)
            log.write_bytes(
                b'{"type": "session.upsert", "session_id": "bytes", "payload": {"status": "completed"}}\n'
                b'{"type": "provider.\xfferror", "session_id": "bytes", "error": "quota"}\n')
            for argv in ([workspace], [workspace, "--failures"]):
                with self.subTest(argv=argv):
                    out = io.StringIO()
                    with redirect_stdout(out):
                        self.assertEqual(1, main(argv))
                    self.assertIn("unreadable record at line 2:", out.getvalue())

    def test_a_lone_surrogate_in_a_provider_error_prints_escaped(self):
        """R22: valid JSON holding \\ud800 crashed the printer on a UTF-8 stdout."""

        with tempfile.TemporaryDirectory() as workspace:
            _write(Path(workspace) / ".vnext" / "runs" / "sur.jsonl", [
                {"type": "session.upsert", "session_id": "sur", "payload": {"status": "failed"}},
                {"type": "provider.error", "session_id": "sur",
                 "payload": {"provider": "claude", "error": "bad\ud800value"}},
            ])
            raw = io.BytesIO()
            stream = io.TextIOWrapper(raw, encoding="utf-8")
            with redirect_stdout(stream):
                code = main([workspace])
            stream.flush()
            self.assertEqual(1, code)
            self.assertIn(b"bad\\ud800value", raw.getvalue())

    def test_finished_agents_are_summed_from_the_outcome_rows(self):
        with tempfile.TemporaryDirectory() as workspace:
            home = Path(workspace) / ".vnext"
            _write(home / "runs" / "s1.jsonl", [
                {"type": "session.upsert", "session_id": "s1", "payload": {"status": "completed"}},
            ])
            _write(home / "outcomes" / "s1.jsonl", [
                {"agent_id": "w-1", "status": "completed", "cost_usd": 0.25,
                 "claimed_verified": True},
                {"agent_id": "w-2", "status": "failed", "cost_usd": 0.5,
                 "claimed_verified": False},
            ])
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(0, main([workspace]))
            self.assertIn("2 finished agents, 1 completed, 1 claiming verification, $0.75",
                          out.getvalue())

    def test_a_run_with_no_price_is_called_unknown_rather_than_zero(self):
        """$0.00 read as free for a worker whose bill never arrived."""

        with tempfile.TemporaryDirectory() as workspace:
            home = Path(workspace) / ".vnext"
            _write(home / "runs" / "s1.jsonl", [
                {"type": "session.upsert", "session_id": "s1", "payload": {"status": "completed"}},
            ])
            _write(home / "outcomes" / "s1.jsonl", [
                {"agent_id": "w-1", "status": "completed", "cost_usd": None,
                 "claimed_verified": True,
                 "usage_unavailable": "the close grace expired first"},
            ])
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(0, main([workspace]))
            printed = out.getvalue()
            self.assertNotIn("$0.00", printed)
            self.assertIn("cost unknown for 1 run", printed)

    def test_native_child_in_parent_is_not_an_unknown_cost(self):
        with tempfile.TemporaryDirectory() as workspace:
            home = Path(workspace) / ".vnext"
            _write(home / "runs" / "s1.jsonl", [
                {"type": "session.upsert", "session_id": "s1", "payload": {"status": "completed"}},
            ])
            _write(home / "outcomes" / "s1.jsonl", [
                {"agent_id": "parent", "status": "completed", "cost_usd": 0.25,
                 "cost_source": "provider"},
                {"agent_id": "native-child", "status": "completed", "cost_usd": None,
                 "cost_source": "in_parent", "tokens": {"basis": "in_parent"}},
            ])
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(0, main([workspace]))
            printed = out.getvalue()
            self.assertIn("$0.25 as the provider reported", printed)
            self.assertNotIn("cost unknown", printed)

    def test_a_priced_run_beside_an_unpriced_one_says_both(self):
        with tempfile.TemporaryDirectory() as workspace:
            home = Path(workspace) / ".vnext"
            _write(home / "runs" / "s1.jsonl", [
                {"type": "session.upsert", "session_id": "s1", "payload": {"status": "completed"}},
            ])
            _write(home / "outcomes" / "s1.jsonl", [
                {"agent_id": "w-1", "status": "completed", "cost_usd": 0.25,
                 "cost_source": "api_rates", "claimed_verified": True},
                {"agent_id": "w-2", "status": "completed", "cost_usd": None,
                 "claimed_verified": False},
            ])
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(0, main([workspace]))
            self.assertIn("$0.25 at API rates; cost unknown for 1 run", out.getvalue())

    def test_each_part_of_a_total_is_labelled_by_where_its_price_came_from(self):
        """A provider's own figure was printed as an API-rate estimate."""

        with tempfile.TemporaryDirectory() as workspace:
            home = Path(workspace) / ".vnext"
            _write(home / "runs" / "s1.jsonl", [
                {"type": "session.upsert", "session_id": "s1", "payload": {"status": "completed"}},
            ])
            _write(home / "outcomes" / "s1.jsonl", [
                {"agent_id": "w-1", "status": "completed", "cost_usd": 0.81,
                 "cost_source": "provider"},
                {"agent_id": "w-2", "status": "completed", "cost_usd": 0.17,
                 "cost_source": "api_rates"},
                {"agent_id": "w-3", "status": "completed", "cost_usd": 0.02},
            ])
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(0, main([workspace]))
            self.assertIn(
                "$1.00 ($0.81 as the provider reported, $0.17 at API rates, "
                "$0.02 from an older record that does not say)",
                out.getvalue(),
            )

    def test_a_legacy_single_event_log_is_still_read(self):
        """The logs written before per-session run records still hold the answers."""

        with tempfile.TemporaryDirectory() as workspace:
            _write(Path(workspace) / ".vnext" / "events.jsonl", [
                {"type": "session.error", "payload": {"error": "bridge timed out"}},
            ])
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(1, main([workspace, "--failures"]))
            self.assertIn("bridge timed out", out.getvalue())


if __name__ == "__main__":
    unittest.main()


class ACleanServerQuitIsNotCalledCancelledTests(unittest.TestCase):
    """The display reads the workers before it repeats the session's status.

    Quitting the server writes ``cancelled`` on the session even when every
    worker had finished, so the report said ``cancelled`` for a run that went
    right.  The status file keeps its value, which another program reads; only
    the printed word changes.
    """

    def _printed(self, rows: list[dict]) -> str:
        with tempfile.TemporaryDirectory() as workspace:
            _write(Path(workspace) / ".vnext" / "runs" / "s.jsonl", rows)
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(0, main([workspace]))
            return out.getvalue()

    def test_a_quit_with_every_worker_finished_reads_as_a_quit(self):
        printed = self._printed([
            {"type": "session.upsert", "session_id": "s", "payload": {"status": "running"}},
            {"type": "agent.spawned", "session_id": "s", "agent_id": "w-1"},
            {"type": "agent.upsert", "session_id": "s", "agent_id": "w-1",
             "payload": {"status": "completed"}},
            {"type": "session.upsert", "session_id": "s", "payload": {"status": "cancelled"}},
        ])
        self.assertIn("ended (server quit)", printed)
        self.assertNotIn("cancelled", printed)

    def test_a_quit_that_really_cancelled_a_worker_still_says_cancelled(self):
        printed = self._printed([
            {"type": "session.upsert", "session_id": "s", "payload": {"status": "running"}},
            {"type": "agent.spawned", "session_id": "s", "agent_id": "w-1"},
            {"type": "agent.upsert", "session_id": "s", "agent_id": "w-1",
             "payload": {"status": "cancelled"}},
            {"type": "session.upsert", "session_id": "s", "payload": {"status": "cancelled"}},
        ])
        self.assertIn("cancelled", printed)
        self.assertNotIn("ended (server quit)", printed)

    def test_a_worker_that_was_blocked_then_retried_and_finished_reads_as_a_quit(self):
        printed = self._printed([
            {"type": "session.upsert", "session_id": "s", "payload": {"status": "running"}},
            {"type": "agent.spawned", "session_id": "s", "agent_id": "w-1"},
            {"type": "agent.upsert", "session_id": "s", "agent_id": "w-1",
             "payload": {"status": "blocked"}},
            {"type": "agent.upsert", "session_id": "s", "agent_id": "w-1",
             "payload": {"status": "retry-ready"}},
            {"type": "agent.upsert", "session_id": "s", "agent_id": "w-1",
             "payload": {"status": "completed"}},
            {"type": "session.upsert", "session_id": "s", "payload": {"status": "cancelled"}},
        ])
        self.assertIn("ended (server quit)", printed)
        self.assertNotIn("cancelled", printed)

    def test_a_worker_left_blocked_at_the_end_still_says_cancelled(self):
        printed = self._printed([
            {"type": "session.upsert", "session_id": "s", "payload": {"status": "running"}},
            {"type": "agent.spawned", "session_id": "s", "agent_id": "w-1"},
            {"type": "agent.upsert", "session_id": "s", "agent_id": "w-1",
             "payload": {"status": "completed"}},
            {"type": "agent.upsert", "session_id": "s", "agent_id": "w-1",
             "payload": {"status": "blocked"}},
            {"type": "session.upsert", "session_id": "s", "payload": {"status": "cancelled"}},
        ])
        self.assertIn("cancelled", printed)
        self.assertNotIn("ended (server quit)", printed)

    def test_one_failure_recorded_at_two_phases_is_printed_once(self):
        """A Z.ai worker's refused key was recorded by its turn and again by its release."""

        said = ("Z.ai provider refused the credential (authentication_failed) -- retrying "
                "will not help; correct the key in ~/.vnext/providers.json and delegate again")
        with tempfile.TemporaryDirectory() as workspace:
            home = Path(workspace) / ".vnext"
            _write(home / "runs" / "s1.jsonl", [
                {"type": "session.upsert", "session_id": "s1", "payload": {"status": "ready"}},
                {"type": "agent.spawned", "session_id": "s1", "agent_id": "w-1"},
                {"type": "provider.error", "session_id": "s1", "agent_id": "w-1",
                 "payload": {"provider": "zai", "model_id": "glm-5.3-flash",
                             "phase": "run_turn", "error": said}},
                {"type": "provider.error", "session_id": "s1", "agent_id": "w-1",
                 "payload": {"provider": "zai", "model_id": "glm-5.3-flash",
                             "phase": "release_agent", "error": said}},
            ])
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(1, main([workspace, "--failures"]))
            self.assertEqual(1, out.getvalue().count("Z.ai provider refused the credential"))

    def test_one_worker_finishing_does_not_cover_for_another_that_failed(self):
        printed = self._printed([
            {"type": "session.upsert", "session_id": "s", "payload": {"status": "running"}},
            {"type": "agent.spawned", "session_id": "s", "agent_id": "w-1"},
            {"type": "agent.spawned", "session_id": "s", "agent_id": "w-2"},
            {"type": "agent.upsert", "session_id": "s", "agent_id": "w-1",
             "payload": {"status": "completed"}},
            {"type": "agent.upsert", "session_id": "s", "agent_id": "w-2",
             "payload": {"status": "failed"}},
            {"type": "session.upsert", "session_id": "s", "payload": {"status": "cancelled"}},
        ])
        self.assertIn("cancelled", printed)
        self.assertNotIn("ended (server quit)", printed)

    def test_the_client_that_held_the_root_is_not_read_as_a_cancelled_worker(self):
        """The root of a session Claude Code holds is written cancelled too.

        It is a record about the client rather than about work this workforce
        ran, and reading it back made every clean run with an external primary
        call itself a cancellation -- which is what three reviewers saw.  Logs
        written by the older server still read right because of this.
        """

        printed = self._printed([
            {"type": "session.upsert", "session_id": "s",
             "payload": {"status": "ready", "primary_agent_id": "primary"}},
            {"type": "agent.upsert", "session_id": "s", "agent_id": "primary",
             "payload": {"status": "ready", "provider": "external"}},
            {"type": "agent.spawned", "session_id": "s", "agent_id": "w-1"},
            {"type": "agent.upsert", "session_id": "s", "agent_id": "w-1",
             "payload": {"status": "completed", "provider": "codex"}},
            {"type": "agent.upsert", "session_id": "s", "agent_id": "primary",
             "payload": {"status": "cancelled", "provider": "external"}},
            {"type": "session.upsert", "session_id": "s", "payload": {"status": "cancelled"}},
        ])
        self.assertIn("ended (server quit)", printed)
        self.assertNotIn("cancelled", printed)

    def test_a_worker_cancelled_beside_the_external_root_still_says_cancelled(self):
        """Only the root is skipped; a real cancellation keeps its word."""

        printed = self._printed([
            {"type": "session.upsert", "session_id": "s",
             "payload": {"status": "ready", "primary_agent_id": "primary"}},
            {"type": "agent.upsert", "session_id": "s", "agent_id": "primary",
             "payload": {"status": "ready", "provider": "external"}},
            {"type": "agent.spawned", "session_id": "s", "agent_id": "w-1"},
            {"type": "agent.upsert", "session_id": "s", "agent_id": "w-1",
             "payload": {"status": "cancelled", "provider": "codex"}},
            {"type": "agent.upsert", "session_id": "s", "agent_id": "primary",
             "payload": {"status": "cancelled", "provider": "external"}},
            {"type": "session.upsert", "session_id": "s", "payload": {"status": "cancelled"}},
        ])
        self.assertIn("cancelled", printed)
        self.assertNotIn("ended (server quit)", printed)

    def test_the_status_read_back_is_still_the_one_the_log_holds(self):
        with tempfile.TemporaryDirectory() as workspace:
            _write(Path(workspace) / ".vnext" / "runs" / "s.jsonl", [
                {"type": "agent.spawned", "session_id": "s", "agent_id": "w-1"},
                {"type": "agent.upsert", "session_id": "s", "agent_id": "w-1",
                 "payload": {"status": "completed"}},
                {"type": "session.upsert", "session_id": "s",
                 "payload": {"status": "cancelled"}},
            ])
            session = read_workspace(Path(workspace))["sessions"][0]
            self.assertEqual("cancelled", session["status"])


class ANestedSessionIsLabelledWithWhatTheCodeKnowsTests(unittest.TestCase):
    """The marker proves where a server started, never whose worker it is.

    ``VNEXT_PARENT_SESSION`` is written into ``os.environ``, so every descendant
    of a server process inherits it: a shell, an editor, any unrelated program
    launched from that session.  Calling each of them "a worker's own server"
    claimed a parentage the code cannot check.  The label now says the one thing
    the marker attests, and prints enough of the session id to recognise it.
    """

    def _printed(self, workspace: str, parent: str) -> str:
        home = Path(workspace) / ".vnext"
        for session in (parent, "nested02"):
            _write(home / "runs" / f"{session}.jsonl", [
                {"type": "session.upsert", "session_id": session,
                 "payload": {"status": "completed"}},
            ])
        status = home / "status"
        status.mkdir(parents=True, exist_ok=True)
        (status / "nested02.json").write_text(
            json.dumps({"session_id": "nested02", "parent_session": parent}),
            encoding="utf-8")
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(0, main([workspace]))
        return out.getvalue()

    def test_the_label_claims_only_where_the_server_started(self):
        with tempfile.TemporaryDirectory() as workspace:
            printed = self._printed(workspace, "parent01")

            self.assertIn("(started inside session parent01)", printed)
            self.assertNotIn("worker's own server", printed)

    def test_a_short_session_id_is_printed_whole(self):
        """'session-a' cut to eight characters came out as 'session-'."""

        with tempfile.TemporaryDirectory() as workspace:
            printed = self._printed(workspace, "session-a")

            self.assertIn("(started inside session session-a)", printed)

    def test_a_long_session_id_is_cut_to_its_first_eight(self):
        with tempfile.TemporaryDirectory() as workspace:
            printed = self._printed(workspace, "abcdef0123456789abcd")

            self.assertIn("(started inside session abcdef01)", printed)


class AnErrorAfterTheWorkFinishedIsNotAFailureTests(unittest.TestCase):
    """The shape of the log a quit leaves behind, and what it used to cost.

    A Claude worker completed at 04:42:39.251.  Six hundred milliseconds later
    the server was stopped, which closed that worker's still-open SDK stream,
    and the SDK answered a result message with is_error true and
    terminal_reason aborted_streaming.  vnext-report counted that one record as
    a failure, printed FAILED for a session where the work was done, and exited
    1, which is the opposite of what the README promises a watcher.
    """

    _ABORTED = {
        "provider": "claude",
        "error": {"source": "ResultMessage", "is_error": True,
                  "subtype": "error_during_execution",
                  "terminal_reason": "aborted_streaming"},
    }

    def _log(self, home: Path, rows: list[dict]) -> None:
        _write(home / "runs" / "3e8474e2.jsonl", rows)

    def _finished_worker_log(self) -> list[dict]:
        return [
            {"type": "session.upsert", "session_id": "3e8474e2",
             "timestamp": "2026-10-01T04:42:30.555610+00:00",
             "payload": {"status": "running"}},
            {"type": "agent.spawned", "session_id": "3e8474e2", "agent_id": "c227d677",
             "timestamp": "2026-10-01T04:42:33.177386+00:00"},
            {"type": "agent.upsert", "session_id": "3e8474e2", "agent_id": "c227d677",
             "timestamp": "2026-10-01T04:42:33.180109+00:00",
             "payload": {"status": "running"}},
            {"type": "agent.upsert", "session_id": "3e8474e2", "agent_id": "c227d677",
             "timestamp": "2026-10-01T04:42:39.251174+00:00",
             "payload": {"status": "completed"}},
            {"type": "provider.error", "session_id": "3e8474e2", "agent_id": "c227d677",
             "timestamp": "2026-10-01T04:42:39.889632+00:00",
             "payload": dict(self._ABORTED)},
            {"type": "session.upsert", "session_id": "3e8474e2",
             "timestamp": "2026-10-01T04:42:40.041107+00:00",
             "payload": {"status": "cancelled"}},
        ]

    def test_a_quit_that_interrupted_a_finished_worker_is_not_a_failure(self):
        with tempfile.TemporaryDirectory() as workspace:
            self._log(Path(workspace) / ".vnext", self._finished_worker_log())

            report = read_workspace(Path(workspace))
            session = report["sessions"][0]
            self.assertEqual([], session["failures"])
            self.assertEqual(1, len(session["notes"]))

            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(0, main([workspace]))
            printed = out.getvalue()
            self.assertNotIn("FAILED", printed)
            self.assertIn("the stream closed after this agent had finished", printed)
            self.assertIn("aborted_streaming", printed)

            quiet = io.StringIO()
            with redirect_stdout(quiet):
                self.assertEqual(0, main([workspace, "--failures"]))
            self.assertIn("no failures recorded", quiet.getvalue())

    def test_the_same_error_mid_turn_is_still_a_failure(self):
        with tempfile.TemporaryDirectory() as workspace:
            rows = [row for row in self._finished_worker_log()
                    if row.get("payload", {}).get("status") != "completed"]
            self._log(Path(workspace) / ".vnext", rows)

            report = read_workspace(Path(workspace))
            self.assertEqual(1, len(report["sessions"][0]["failures"]))

            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(1, main([workspace, "--failures"]))
            self.assertIn("FAILED", out.getvalue())

    def test_an_error_before_the_agent_finished_is_still_a_failure(self):
        """A worker that failed, was retried and then completed keeps its failure."""

        with tempfile.TemporaryDirectory() as workspace:
            rows = self._finished_worker_log()
            early = dict(rows[4])
            early["timestamp"] = "2026-10-01T04:42:35.000000+00:00"
            rows[4] = early
            self._log(Path(workspace) / ".vnext", rows)

            report = read_workspace(Path(workspace))
            session = report["sessions"][0]
            self.assertEqual([], session["notes"])
            self.assertEqual(1, len(session["failures"]))

    def test_a_provider_refusal_after_the_agent_finished_is_still_a_failure(self):
        """R17: only a stream our own quit closed is a note.

        A Codex-route turn that a completed worker still had open can end
        failed on a 401.  That is the provider's verdict, and --failures has
        to say so.
        """

        with tempfile.TemporaryDirectory() as workspace:
            rows = self._finished_worker_log()
            rows[4] = {**rows[4], "payload": {
                "provider": "codex", "error": "unexpected status 401 Unauthorized",
                "native": {"message": "unexpected status 401 Unauthorized"}}}
            self._log(Path(workspace) / ".vnext", rows)

            report = read_workspace(Path(workspace))
            session = report["sessions"][0]
            self.assertEqual([], session["notes"])
            self.assertEqual(1, len(session["failures"]))

            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(1, main([workspace, "--failures"]))
            self.assertIn("401", out.getvalue())

    def test_an_error_against_an_agent_that_never_completed_is_a_failure(self):
        with tempfile.TemporaryDirectory() as workspace:
            rows = self._finished_worker_log()
            rows[3] = {**rows[3], "agent_id": "another-worker"}
            self._log(Path(workspace) / ".vnext", rows)

            report = read_workspace(Path(workspace))
            self.assertEqual(1, len(report["sessions"][0]["failures"]))

    def test_a_structured_provider_error_prints_as_words(self):
        """R18 stranger: the failure line was a Python dict repr."""

        with tempfile.TemporaryDirectory() as workspace:
            rows = self._finished_worker_log()
            rows[3] = {**rows[3], "agent_id": "another-worker"}
            rows[4] = {**rows[4], "payload": {"provider": "claude", "error": {
                "source": "AssistantMessage", "is_error": True, "code": "authentication_failed",
                "api_error_status": 401, "retry_after_seconds": None}}}
            self._log(Path(workspace) / ".vnext", rows)

            out = io.StringIO()
            with redirect_stdout(out):
                main([workspace, "--failures"])
            printed = out.getvalue()
            self.assertIn("authentication_failed, HTTP 401", printed)
            self.assertNotIn("{'", printed)


class ABareCallNamesTheFolderItReportsTests(unittest.TestCase):
    """With no path this read the current folder and never said which it was."""

    def test_the_defaulted_folder_is_the_first_line(self):
        with tempfile.TemporaryDirectory() as workspace:
            here = Path(workspace).resolve()
            _write(here / ".vnext" / "runs" / "quiet01.jsonl", [
                {"type": "session.upsert", "session_id": "quiet01",
                 "payload": {"status": "completed"}},
            ])
            start = os.getcwd()
            os.chdir(here)
            try:
                out = io.StringIO()
                with redirect_stdout(out):
                    self.assertEqual(0, main([]))
            finally:
                os.chdir(start)
            first = out.getvalue().splitlines()[0]
            self.assertEqual(f"reporting {here} (the current folder)", first)

    def test_a_folder_given_by_name_is_not_announced_as_the_current_one(self):
        with tempfile.TemporaryDirectory() as workspace:
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(0, main([workspace]))
            self.assertNotIn("the current folder", out.getvalue())


class ServerLivenessReportTests(unittest.TestCase):
    def _line(self, workspace: str, *, pid: int | None, status: str = "ready") -> str:
        home = Path(workspace) / ".vnext"
        _write(home / "runs" / "session1.jsonl", [
            {"type": "session.upsert", "session_id": "session1",
             "payload": {"status": status}},
            {"type": "agent.spawned", "session_id": "session1", "agent_id": "w-1"},
        ])
        if pid is not None:
            status_file = home / "status" / "session1.json"
            status_file.parent.mkdir(parents=True, exist_ok=True)
            status_file.write_text(json.dumps({"session_id": "session1", "pid": pid}),
                                   encoding="utf-8")
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(0, main([workspace]))
        return next(line for line in out.getvalue().splitlines() if "session1" in line)

    def test_dead_server_is_shown_as_gone_without_changing_status_file(self):
        from unittest.mock import patch
        import vnext.vnext_report as report_module

        with tempfile.TemporaryDirectory() as workspace:
            with patch.object(report_module.os, "kill", side_effect=ProcessLookupError):
                line = self._line(workspace, pid=987654321)
            self.assertIn("ended (server gone)", line)
            self.assertEqual(987654321, json.loads((Path(workspace) / ".vnext" /
                "status" / "session1.json").read_text())["pid"])

    def test_live_server_keeps_ready(self):
        with tempfile.TemporaryDirectory() as workspace:
            self.assertIn("ready", self._line(workspace, pid=os.getpid()))

    def test_no_status_file_keeps_stored_word(self):
        with tempfile.TemporaryDirectory() as workspace:
            self.assertIn("ready", self._line(workspace, pid=None))

    def test_windows_liveness_uses_openprocess_and_closes_handle(self):
        from unittest.mock import Mock, patch
        import vnext.vnext_report as report_module

        class Kernel32:
            def __init__(self):
                self.closed = []
                self.OpenProcess = Mock(return_value=123)
                self.GetExitCodeProcess = Mock(side_effect=lambda handle, result: self._exit_code(result))
                self.CloseHandle = Mock(side_effect=lambda handle: self.closed.append(handle) or 1)

            @staticmethod
            def _exit_code(result):
                result._obj.value = 0
                return 1

        kernel32 = Kernel32()
        with tempfile.TemporaryDirectory() as workspace:
            with patch.object(report_module.sys, "platform", "win32"), \
                 patch.object(report_module.ctypes, "WinDLL", return_value=kernel32,
                              create=True), \
                 patch.object(report_module.os, "kill", side_effect=AssertionError("unsafe")):
                line = self._line(workspace, pid=987654321)
            self.assertIn("ended (server gone)", line)
            self.assertEqual([123], kernel32.closed)

    def test_single_counts_use_singular_nouns(self):
        with tempfile.TemporaryDirectory() as workspace:
            line = self._line(workspace, pid=None)
            self.assertIn("1 agent", line)
            self.assertNotIn("1 agents", line)
            self.assertIn("2 records", line)

    def test_status_file_without_pid_keeps_ready(self):
        with tempfile.TemporaryDirectory() as workspace:
            home = Path(workspace) / ".vnext"
            _write(home / "runs" / "session1.jsonl", [
                {"type": "session.upsert", "session_id": "session1",
                 "payload": {"status": "ready"}},
            ])
            status_file = home / "status" / "session1.json"
            status_file.parent.mkdir(parents=True)
            status_file.write_text(json.dumps({"session_id": "session1"}), encoding="utf-8")
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(0, main([workspace]))
            self.assertIn("ready", out.getvalue())

    def test_terminal_status_does_not_probe_pid(self):
        from unittest.mock import patch
        import vnext.vnext_report as report_module

        with tempfile.TemporaryDirectory() as workspace:
            with patch.object(report_module.os, "kill", side_effect=AssertionError("unsafe")):
                line = self._line(workspace, pid=987654321, status="completed")
            self.assertIn("completed", line)

    def test_windows_access_denied_is_treated_as_alive(self):
        from unittest.mock import Mock, patch
        import vnext.vnext_report as report_module

        kernel32 = Mock()
        kernel32.OpenProcess.return_value = 0
        with tempfile.TemporaryDirectory() as workspace:
            with patch.object(report_module.sys, "platform", "win32"), \
                 patch.object(report_module.ctypes, "WinDLL", return_value=kernel32,
                              create=True), \
                 patch.object(report_module.ctypes, "get_last_error", return_value=5,
                              create=True):
                line = self._line(workspace, pid=987654321)
            self.assertIn("ready", line)
            kernel32.GetExitCodeProcess.assert_not_called()

    def test_one_finished_agent_is_singular(self):
        with tempfile.TemporaryDirectory() as workspace:
            home = Path(workspace) / ".vnext"
            _write(home / "runs" / "one.jsonl", [
                {"type": "session.upsert", "session_id": "one",
                 "payload": {"status": "completed"}},
            ])
            _write(home / "outcomes" / "one.jsonl", [
                {"agent_id": "w-1", "status": "completed", "cost_usd": 0.25},
            ])
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(0, main([workspace]))
            self.assertIn("1 finished agent,", out.getvalue())
            self.assertNotIn("1 finished agents,", out.getvalue())
