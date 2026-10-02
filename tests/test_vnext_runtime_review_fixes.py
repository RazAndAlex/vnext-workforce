"""Paths a cross-vendor review of the side-runtime work found open.

Each test names the review item it covers.  No test reads PyPI or starts a
provider: PyPI answers come from fakes and the side runtime is a folder of
small scripts.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from vnext import vnext_mcp_server as server
from vnext import vnext_runtimes as runtimes
from vnext.vnext_model_identity import UNKNOWN, identity_view

from tests import test_vnext_runtimes as base

FAKE_PYPI = base.FAKE_PYPI


def _catalog_codex_models() -> list[dict]:
    return [
        {"id": entry["model"], "model": entry["model"], "hidden": False}
        for entry in server.DEFAULT_CATALOG
        if entry.get("provider") == "codex"
    ]


class InheritedBridgeEnvironmentTests(unittest.TestCase):
    """Item 1: only the side-runtime bridge command may name the SDK or CLI."""

    def _launch_env(self, **adapter_args) -> dict:
        from vnext import vnext_claude

        with tempfile.TemporaryDirectory() as workspace, patch.dict(os.environ, {
            "VNEXT_CLAUDE_CLI_PATH": "/elsewhere/claude",
            "VNEXT_CLAUDE_SDK_VERSION": "0.0.1",
        }):
            adapter = vnext_claude.ClaudeCodeAdapter(workspace=workspace, **adapter_args)
            with patch.object(vnext_claude.OwnedProcess, "start", side_effect=OSError("stop here")) as start:
                with self.assertRaises(vnext_claude.ClaudeRuntimeError):
                    adapter._start_owned_bridge()
        return start.call_args.kwargs.get("env")

    def test_the_pinned_bridge_does_not_inherit_the_side_values(self) -> None:
        env = self._launch_env()
        self.assertIsNotNone(env, "the bridge inherited the server's whole environment")
        self.assertNotIn("VNEXT_CLAUDE_CLI_PATH", env)
        self.assertNotIn("VNEXT_CLAUDE_SDK_VERSION", env)
        self.assertIn("PATH", env)

    def test_a_custom_bridge_command_keeps_the_rest_of_the_environment(self) -> None:
        with patch.dict(os.environ, {"VNEXT_TEST_MARKER": "kept"}):
            env = self._launch_env(bridge_command=[sys.executable, "-c", "pass"])
        self.assertEqual("kept", env.get("VNEXT_TEST_MARKER"))
        self.assertNotIn("VNEXT_CLAUDE_CLI_PATH", env)

    def test_the_side_command_still_sets_both_values_inside_its_own_process(self) -> None:
        from vnext.vnext_claude import side_runtime_bridge_command

        command = side_runtime_bridge_command({
            "python": sys.executable, "sdk_version": "0.2.162", "cli_path": "/side/claude",
        })
        setup = command[2].split("import runpy")[0]
        probe = setup + "import os, json; print(json.dumps([os.environ.get('VNEXT_CLAUDE_SDK_VERSION'), os.environ.get('VNEXT_CLAUDE_CLI_PATH')]))"
        env = {k: v for k, v in os.environ.items() if not k.startswith("VNEXT_CLAUDE_")}
        result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, env=env, check=True)
        self.assertEqual(["0.2.162", "/side/claude"], json.loads(result.stdout))


class UpdateRefusalTests(base._RuntimesDir):
    """Items 2 and 4: --update-runtimes refuses a pair vNext could not run as recorded."""

    def _update_with(self, *, bundled_cli: bool = True, codex_models=None):
        run, _calls = base.UpdateCommandTests._runner(self)
        if not bundled_cli:
            inner = run

            def run(command, timeout=None):  # noqa: F811 - wraps the fake runner
                result = inner(command, timeout=timeout)
                if command[1:3] == ["-c", runtimes._LOCATE_SCRIPT]:
                    payload = json.loads(result.stdout)
                    payload["claude"] = None
                    result = subprocess.CompletedProcess(command, 0, json.dumps(payload) + "\n", "")
                return result
        out, err = io.StringIO(), io.StringIO()
        code = runtimes.update_runtimes(
            "0.159.3", "0.2.162", out=out, err=err, runner=run,
            codex_probe=lambda exe, ws: (codex_models if codex_models is not None else _catalog_codex_models(), None),
            claude_probe=lambda py, cli, ws: ([{"value": "opus", "resolvedModel": "claude-opus-5-5"}], None),
        )
        return code, out.getvalue(), err.getvalue()

    def test_a_side_sdk_with_no_bundled_claude_cli_is_refused(self) -> None:
        code, _out, err = self._update_with(bundled_cli=False)
        self.assertEqual(1, code)
        self.assertIn("no Claude CLI of its own", err)
        self.assertIn("nothing changed", err)
        self.assertFalse((self.root / "runtimes" / "active.json").exists())

    def test_a_codex_that_drops_a_catalog_model_is_refused_and_the_models_are_named(self) -> None:
        models = [row for row in _catalog_codex_models() if row["id"] != "gpt-6-sol"]
        code, _out, err = self._update_with(codex_models=models)
        self.assertEqual(1, code)
        self.assertIn("gpt-6-sol", err)
        self.assertIn("nothing changed", err)
        self.assertFalse((self.root / "runtimes" / "active.json").exists())

    def test_a_codex_listing_every_catalog_model_is_accepted(self) -> None:
        code, _out, err = self._update_with()
        self.assertEqual(0, code, err)
        self.assertTrue((self.root / "runtimes" / "active.json").exists())


class HandEditedRecordTests(base._RuntimesDir):
    """Item 5: a record without the fields the server reads is not a runtime record."""

    def _write(self, value) -> None:
        folder = self.root / "runtimes"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "active.json").write_text(json.dumps(value), encoding="utf-8")

    def _read(self) -> tuple[object, str]:
        err = io.StringIO()
        with redirect_stderr(err):
            value = runtimes.read_active_runtime()
        return value, err.getvalue()

    def test_empty_sections_mean_the_pinned_runtime(self) -> None:
        self._write({"codex": {}, "claude": {}})
        value, err = self._read()
        self.assertIsNone(value)
        self.assertIn("is not a runtime record; the pinned runtime runs", err)

    def test_a_field_of_the_wrong_type_means_the_pinned_runtime(self) -> None:
        record = self._active()
        record["codex"]["sha256"] = 7
        self._write(record)
        value, err = self._read()
        self.assertIsNone(value)
        self.assertIn("is not a runtime record", err)

    def test_a_record_without_its_claude_cli_is_refused(self) -> None:
        record = self._active()
        record["claude"]["cli_path"] = None
        record["claude"]["cli_sha256"] = None
        self._write(record)
        value, err = self._read()
        self.assertIsNone(value)
        self.assertIn("is not a runtime record", err)

    def test_a_complete_record_is_read(self) -> None:
        self._active()
        value, err = self._read()
        self.assertIsNotNone(value)
        self.assertEqual("", err)
        self.assertIsNotNone(runtimes.codex_runtime_selection(value))
        self.assertIsNotNone(runtimes.claude_runtime_selection(value))


class BrokenSideRuntimeCheckTests(base._RuntimesDir):
    """Item 6: --check fails, and says what to do, when the side runtime is broken."""

    def _check(self) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with tempfile.TemporaryDirectory() as workspace, patch.object(
            server, "_load_catalog", return_value=[{"provider": "codex", "model": "gpt-6.1-sol"}]
        ), patch.object(server, "startup_check_notes", return_value=[]):
            code = server.run_startup_check(["--workspace", workspace], out=out, err=err)
        return code, out.getvalue(), err.getvalue()

    def _assert_broken(self, code: int, err: str) -> None:
        self.assertNotEqual(0, code)
        self.assertIn("side runtime", err)
        self.assertIn("--update-runtimes", err)
        self.assertIn("vnext-mcp --update-runtimes --rollback", err)

    def test_a_working_side_runtime_passes(self) -> None:
        self._active()
        code, _out, err = self._check()
        self.assertEqual(0, code, err)

    def test_a_deleted_side_codex_fails_the_check(self) -> None:
        record = self._active()
        Path(record["codex"]["executable"]).unlink()
        self._assert_broken(*self._check()[::2])

    def test_a_changed_side_codex_fails_the_check(self) -> None:
        record = self._active()
        Path(record["codex"]["executable"]).write_text("#!/bin/sh\necho changed\n", encoding="utf-8")
        self._assert_broken(*self._check()[::2])

    def test_a_missing_side_python_fails_the_check(self) -> None:
        record = self._active()
        record["claude"]["python"] = str(self.root / "gone" / "python")
        (self.root / "runtimes" / "active.json").write_text(json.dumps(record), encoding="utf-8")
        self._assert_broken(*self._check()[::2])

    @unittest.skipIf(os.name == "nt", "the execute bit is a POSIX permission")
    def test_a_side_python_that_cannot_run_fails_the_check(self) -> None:
        record = self._active()
        python = self.root / "side" / "python"
        python.write_text("not a program", encoding="utf-8")
        python.chmod(0o644)
        record["claude"]["python"] = str(python)
        (self.root / "runtimes" / "active.json").write_text(json.dumps(record), encoding="utf-8")
        self._assert_broken(*self._check()[::2])

    def test_a_changed_side_claude_cli_fails_the_check(self) -> None:
        record = self._active()
        cli = self.root / "side" / "claude"
        cli.write_text("one", encoding="utf-8")
        record["claude"]["cli_path"] = str(cli)
        record["claude"]["cli_sha256"] = hashlib.sha256(b"one").hexdigest()
        (self.root / "runtimes" / "active.json").write_text(json.dumps(record), encoding="utf-8")
        self.assertEqual(0, self._check()[0])
        cli.write_text("two", encoding="utf-8")
        self._assert_broken(*self._check()[::2])


def _bare_service(temp: str) -> server.VNextMcpService:
    service = object.__new__(server.VNextMcpService)
    service._session_id = "review"
    service._failing_writes = set()
    service.workspace = Path(temp)
    service._client = "claude-code"
    service._outcome_log = Path(temp) / "outcomes.jsonl"
    service._log_lock = threading.Lock()
    service._roster_lock = threading.Lock()
    service._roster = {}
    service._outcome_rows = {}
    service._latest_usage = {}
    service._usage_unavailable = {}
    return service


class RosterIdentityTests(unittest.TestCase):
    """Item 7: the status roster keeps the exact model beside the alias."""

    def test_the_roster_row_carries_the_identity_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            service = _bare_service(temp)
            service._remember({
                "agent_id": "worker", "role": "worker", "provider": "claude", "model": "opus",
                "status": "running", "model_exact": "claude-opus-5-5", "model_exact_source": "server_info",
                "model_ran": "claude-opus-5-1", "model_mismatch": True,
                "model_note": "asked for claude-opus-5-5, the provider ran claude-opus-5-1",
            })
            row = service._roster["worker"]
        self.assertEqual("opus", row["model"])
        self.assertEqual("claude-opus-5-5", row["model_exact"])
        self.assertEqual("server_info", row["model_exact_source"])
        self.assertEqual("claude-opus-5-1", row["model_ran"])
        self.assertTrue(row["model_mismatch"])
        self.assertIn("the provider ran claude-opus-5-1", row["model_note"])

    def test_a_row_without_identity_keeps_its_old_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            service = _bare_service(temp)
            service._remember({"agent_id": "w", "role": "worker", "provider": "codex",
                               "model": "gpt-6-sol", "status": "running"})
            row = service._roster["w"]
        self.assertNotIn("model_exact", row)


class UnknownCarriesAReasonTests(unittest.TestCase):
    """Item 8: no record says "exact model unknown" without saying why."""

    def _outcome_row(self, status: str) -> dict:
        with tempfile.TemporaryDirectory() as temp:
            service = _bare_service(temp)
            service._record_outcome({
                "agent_id": "worker", "role": "worker", "provider": "claude", "model": "opus",
                "status": status, "usage": {},
            }, time.time())
            return json.loads(service._outcome_log.read_text(encoding="utf-8").splitlines()[-1])

    def test_a_worker_cancelled_before_its_provider_connected_says_so(self) -> None:
        row = self._outcome_row("cancelled")
        self.assertTrue(row["model_exact"].startswith(f"{UNKNOWN}: "), row["model_exact"])
        self.assertIn("before its provider named a model", row["model_exact"])

    def test_a_completed_worker_with_no_identity_still_says_why(self) -> None:
        row = self._outcome_row("completed")
        self.assertTrue(row["model_exact"].startswith(f"{UNKNOWN}: "), row["model_exact"])

    def test_an_unresolved_table_with_no_reply_says_why(self) -> None:
        view = identity_view({"model_exact": None, "model_table": []})
        self.assertTrue(view["model_exact"].startswith(f"{UNKNOWN}: "), view["model_exact"])

    def test_the_check_says_why_when_it_did_not_probe(self) -> None:
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as workspace, patch.object(
            server, "_load_catalog", return_value=[{"model": "opus", "provider": "claude"}],
        ), patch.object(server, "startup_check_notes", return_value=[]):
            self.assertEqual(0, server.run_startup_check(["--workspace", workspace], out=out))
        line = next(l for l in out.getvalue().splitlines() if l.startswith("opus (claude) -> "))
        self.assertTrue(line.startswith(f"opus (claude) -> {UNKNOWN}: "), line)


class StartDoesNotWaitOnPyPITests(base._RuntimesDir):
    """Item 10: a server start never waits on the network."""

    def test_a_failed_check_is_cached_for_the_day(self) -> None:
        calls = []

        def offline(name):
            calls.append(name)
            raise OSError("offline")

        with patch.dict(os.environ, {"VNEXT_NO_UPDATE_CHECK": "0"}):
            self.assertIsNone(runtimes.check_for_updates(fetch=offline, today="2026-10-02"))
            self.assertIsNone(runtimes.check_for_updates(fetch=offline, today="2026-10-02"))
        self.assertEqual(1, len(calls))

    def test_the_check_command_retries_a_cached_failure(self) -> None:
        with patch.dict(os.environ, {"VNEXT_NO_UPDATE_CHECK": "0"}):
            runtimes.check_for_updates(fetch=lambda name: (_ for _ in ()).throw(OSError("offline")), today="2026-10-02")
            answer = runtimes.check_for_updates(
                fetch=lambda name: FAKE_PYPI[name], today="2026-10-02", retry_failed=True,
            )
        self.assertEqual("0.161.0", answer["openai-codex"])

    def _service(self, workspace: Path) -> server.VNextMcpService:
        return server.VNextMcpService(
            workspace=workspace,
            catalog=[{"provider": "codex", "model": "gpt-6-sol"}],
            event_log=workspace / "run.jsonl",
            status_file=workspace / "status.json",
            outcome_log=workspace / "outcomes.jsonl",
            adapter_factories={"codex": lambda: None},
        )

    def test_construction_returns_while_pypi_hangs_and_inspect_learns_the_answer_later(self) -> None:
        release = threading.Event()

        def slow(name):
            release.wait(20)
            return FAKE_PYPI[name]

        with tempfile.TemporaryDirectory() as temp, patch.dict(
            os.environ, {"VNEXT_NO_UPDATE_CHECK": "0"}
        ), patch.object(runtimes, "_fetch_pypi", slow):
            started = time.monotonic()
            service = self._service(Path(temp))
            try:
                elapsed = time.monotonic() - started
                self.assertLess(elapsed, 5.0, "the start waited on PyPI")
                self.assertEqual([], list(service._runtime_notice))
                release.set()
                service._update_check_thread.join(10)
                notice = list(service._runtime_notice)
            finally:
                release.set()
                service.close()
        self.assertTrue(any(line.startswith("openai-codex 0.161.0 is out") for line in notice), notice)


if __name__ == "__main__":
    unittest.main()
