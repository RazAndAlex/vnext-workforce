"""Side runtimes: the update check, the install record, rollback, and the server's use of them.

No test reads PyPI or starts a provider: PyPI answers come from a fake fetch,
the side venv is a folder of small scripts, and the model probes are fakes.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import suite_environment  # noqa: F401  # the suite settings; unittest never reads conftest.py
from vnext import vnext_mcp_server as server
from vnext import vnext_runtimes as runtimes
from vnext.live_runtime import resolve_session_runtime
from vnext.release_check import RELEASE_CODEX_VERSION
from vnext.vnext_claude import side_runtime_bridge_command
from vnext.vnext_claude_bridge import CLAUDE_SDK_VERSION


def _pypi(*versions: str, yanked: tuple[str, ...] = ()) -> dict:
    return {"releases": {v: [{"yanked": v in yanked}] for v in versions}}


FAKE_PYPI = {
    "openai-codex": _pypi("0.159.3", "0.160.0", "0.161.0", "0.162.0a1", "0.163.0", yanked=("0.163.0",)),
    "claude-agent-sdk": _pypi("0.2.162", "0.2.163", "0.2.164rc1", "0.2.170"),
}


def _write_program(path: Path, output: str) -> Path:
    """A small program that prints *output* and exits 0.

    Windows cannot start a shell script, so there it is a ``.cmd`` batch file
    next to *path*, which ``CreateProcess`` runs through ``cmd.exe``.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        program = path.with_name(path.name + ".cmd")
        program.write_bytes(f"@echo off\r\necho {output}\r\n".encode("utf-8"))
        return program
    path.write_text(f"#!/bin/sh\necho '{output}'\n", encoding="utf-8")
    path.chmod(0o755)
    return path


class _RuntimesDir(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.root = Path(self._temp.name)
        env = patch.dict(os.environ, {"VNEXT_RUNTIMES_DIR": str(self.root / "runtimes")})
        env.start()
        self.addCleanup(env.stop)

    def _fake_codex(self, version: str = "codex-cli 0.159.3") -> Path:
        return _write_program(self.root / "side" / "codex", version)

    def _active(self, **codex_extra) -> dict:
        exe = self._fake_codex()
        cli = self.root / "side" / "claude"
        cli.parent.mkdir(parents=True, exist_ok=True)
        cli.write_text("fake claude", encoding="utf-8")
        record = {
            "venv": str(self.root / "runtimes" / "0.159.3-0.2.162"),
            "codex": {
                "executable": str(exe),
                "sha256": hashlib.sha256(exe.read_bytes()).hexdigest(),
                "version": "codex-cli 0.159.3",
                "package_version": "0.159.3",
                "models": [
                    {"id": "gpt-6.1-sol", "model": "gpt-6.1-sol", "hidden": False},
                    {"id": "gpt-7-nova", "model": "gpt-7-nova", "hidden": False},
                    {"id": "codex-auto-review", "model": "codex-auto-review", "hidden": True},
                ],
                **codex_extra,
            },
            "claude": {"python": sys.executable, "sdk_version": "0.2.162", "cli_path": str(cli),
                       "cli_sha256": hashlib.sha256(b"fake claude").hexdigest(), "models": []},
        }
        folder = self.root / "runtimes"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "active.json").write_text(json.dumps(record), encoding="utf-8")
        return record


class UpdateCheckTests(_RuntimesDir):
    def test_newest_skips_prereleases_and_yanked_releases(self) -> None:
        self.assertEqual("0.161.0", runtimes.stable_newest(FAKE_PYPI["openai-codex"]))
        self.assertEqual("0.2.170", runtimes.stable_newest(FAKE_PYPI["claude-agent-sdk"]))
        self.assertIsNone(runtimes.stable_newest(_pypi("0.162.0a1")))

    def test_the_off_switch_reads_nothing(self) -> None:
        fetch = MagicMock()
        with patch.dict(os.environ, {"VNEXT_NO_UPDATE_CHECK": "1"}):
            self.assertIsNone(runtimes.check_for_updates(fetch=fetch))
        fetch.assert_not_called()

    def test_the_answer_is_cached_for_the_day(self) -> None:
        fetch = MagicMock(side_effect=lambda name: FAKE_PYPI[name])
        with patch.dict(os.environ, {"VNEXT_NO_UPDATE_CHECK": "0"}):
            first = runtimes.check_for_updates(fetch=fetch, today="2026-10-02")
            second = runtimes.check_for_updates(fetch=fetch, today="2026-10-02")
            runtimes.check_for_updates(fetch=fetch, today="2026-10-03")
        self.assertEqual({"openai-codex": "0.161.0", "claude-agent-sdk": "0.2.170"}, first)
        self.assertEqual(first, second)
        self.assertEqual(4, fetch.call_count)

    def test_a_network_failure_gives_no_answer_and_is_cached_as_a_failure(self) -> None:
        with patch.dict(os.environ, {"VNEXT_NO_UPDATE_CHECK": "0"}):
            self.assertIsNone(runtimes.check_for_updates(fetch=MagicMock(side_effect=OSError("offline"))))
        cached = json.loads((self.root / "runtimes" / "update-check.json").read_text(encoding="utf-8"))
        self.assertTrue(cached["failed"])
        self.assertNotIn("newest", cached)

    def test_update_lines_compare_against_the_pinned_runtime(self) -> None:
        view = runtimes.runtime_view(None)
        self.assertEqual("pinned", view["source"])
        self.assertEqual(f"codex-cli {RELEASE_CODEX_VERSION}", view["codex"])
        lines = runtimes.update_lines(view, {"openai-codex": "9.0.0", "claude-agent-sdk": CLAUDE_SDK_VERSION})
        self.assertEqual(1, len(lines))
        self.assertIn(f"openai-codex 9.0.0 is out (vNext runs {RELEASE_CODEX_VERSION})", lines[0])
        self.assertIn("Tell your agent: update vNext runtimes", lines[0])

    def test_update_lines_compare_against_the_active_side_runtime(self) -> None:
        view = runtimes.runtime_view(self._active())
        self.assertTrue(view["source"].startswith("side: "))
        lines = runtimes.update_lines(view, {"openai-codex": "0.160.0", "claude-agent-sdk": "0.2.163"})
        self.assertEqual(2, len(lines))
        self.assertIn("openai-codex 0.160.0 is out (vNext runs 0.159.3)", lines[0])
        self.assertIn("claude-agent-sdk 0.2.163 is out (vNext runs 0.2.162)", lines[1])
        self.assertEqual([], runtimes.update_lines(view, {"openai-codex": "0.159.3", "claude-agent-sdk": "0.2.162"}))


class ActiveRuntimeTests(_RuntimesDir):
    def test_no_active_file_means_the_pinned_runtime(self) -> None:
        self.assertIsNone(runtimes.read_active_runtime())
        self.assertEqual({}, runtimes.session_runtime_config(None))
        self.assertEqual([], runtimes.runtime_notice_lines(server.DEFAULT_CATALOG, always=False))

    def test_the_recorded_side_codex_is_accepted_by_the_session_resolver(self) -> None:
        self._active()
        active = runtimes.read_active_runtime()
        executable, identity = resolve_session_runtime(runtimes.codex_runtime_selection(active))
        self.assertEqual("registered-native-installation", identity["source"])
        self.assertEqual("codex-cli 0.159.3", identity["version"])
        self.assertEqual(Path(active["codex"]["executable"]).resolve(), executable)

    def test_a_changed_side_codex_is_refused(self) -> None:
        self._active()
        active = runtimes.read_active_runtime()
        Path(active["codex"]["executable"]).write_text("#!/bin/sh\necho changed\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "identity changed"):
            resolve_session_runtime(runtimes.codex_runtime_selection(active))

    def test_untested_models_are_the_visible_ones_the_catalog_lacks(self) -> None:
        active = self._active()
        self.assertEqual(["gpt-7-nova"], runtimes.untested_codex_models(active, server.DEFAULT_CATALOG))
        entries = runtimes.untested_catalog_entries(active, server.DEFAULT_CATALOG)
        self.assertEqual("gpt-7-nova", entries[0]["model"])
        self.assertIn("not tested by vNext", entries[0]["claims"][0]["statement"])

    def test_an_untested_model_stays_named_after_it_joins_the_catalog(self) -> None:
        active = self._active()
        catalog = list(server.DEFAULT_CATALOG) + runtimes.untested_catalog_entries(active, server.DEFAULT_CATALOG)
        self.assertEqual(["gpt-7-nova"], runtimes.untested_codex_models(active, catalog))

    def test_the_default_catalog_offers_the_untested_model_and_gates_on_side_slugs(self) -> None:
        self._active()
        probe = MagicMock()
        probe.codex_available.return_value = True
        probe.claude_available.return_value = False
        probe.claude_state.return_value = server.CLAUDE_LOGIN_AVAILABLE
        with patch.object(server, "load_zai_provider", return_value=None), patch.object(
            server, "load_commandcode_provider", return_value=None
        ):
            with self.assertRaisesRegex(server.VNextMcpServiceError, "side Codex runtime"):
                # gpt-6-sol is in the catalog but the side runtime does not list it.
                server._load_catalog(None, login=probe)
        models = [{"provider": "codex", "model": "gpt-6.1-sol"}, {"provider": "codex", "model": "gpt-7-nova"}]
        entries = server._validate_catalog(models, login=probe)
        self.assertEqual(["gpt-6.1-sol", "gpt-7-nova"], [entry["model"] for entry in entries])

    def test_the_session_config_points_both_providers_at_the_side_runtime(self) -> None:
        config = runtimes.session_runtime_config(self._active())
        self.assertEqual("codex-cli 0.159.3", config["codex_runtime"]["version"])
        self.assertEqual("0.2.162", config["claude_runtime"]["sdk_version"])
        self.assertTrue(config["claude_runtime"]["source"].startswith("side: "))

    def test_the_side_bridge_command_runs_on_the_side_python_and_names_the_sdk(self) -> None:
        command = side_runtime_bridge_command({
            "python": "/side/bin/python", "sdk_version": "0.2.162", "cli_path": "/side/claude",
        })
        self.assertEqual("/side/bin/python", command[0])
        self.assertEqual("-c", command[1])
        self.assertIn("VNEXT_CLAUDE_SDK_VERSION", command[2])
        self.assertIn("'0.2.162'", command[2])
        self.assertIn("VNEXT_CLAUDE_CLI_PATH", command[2])
        self.assertIn("vnext_claude_bridge", command[2])

    def test_the_bridge_accepts_the_side_sdk_version_it_is_told(self) -> None:
        from vnext import vnext_claude_bridge as bridge

        with patch.dict(os.environ, {"VNEXT_CLAUDE_SDK_VERSION": "0.2.162"}):
            self.assertEqual("0.2.162", bridge._expected_sdk_version())
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VNEXT_CLAUDE_SDK_VERSION", None)
            self.assertEqual(CLAUDE_SDK_VERSION, bridge._expected_sdk_version())

    def test_a_moved_side_claude_cli_is_refused(self) -> None:
        cli = self.root / "claude"
        cli.write_text("one", encoding="utf-8")
        selection = {"python": sys.executable, "sdk_version": "0.2.162", "cli_path": str(cli),
                     "cli_sha256": hashlib.sha256(b"one").hexdigest()}
        runtimes.verify_claude_runtime(selection)
        cli.write_text("two", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "changed"):
            runtimes.verify_claude_runtime(selection)


class CheckOutputTests(_RuntimesDir):
    def _check(self, entries: list[dict]) -> list[str]:
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as workspace, patch.object(
            server, "_load_catalog", return_value=entries
        ), patch.object(server, "startup_check_notes", return_value=[]):
            self.assertEqual(0, server.run_startup_check(["--workspace", workspace], out=out))
        return out.getvalue().splitlines()

    def test_the_check_names_the_pinned_runtime(self) -> None:
        printed = self._check([{"provider": "codex", "model": "gpt-6.1-sol"}])
        self.assertIn(
            f"runtime: Codex codex-cli {RELEASE_CODEX_VERSION}, Claude SDK {CLAUDE_SDK_VERSION} (source: pinned)",
            printed,
        )

    def test_the_check_names_the_side_runtime_its_updates_and_untested_models(self) -> None:
        active = self._active()
        with patch.object(runtimes, "check_for_updates", return_value=None), patch.object(
            server, "check_for_updates", return_value={"openai-codex": "0.160.0", "claude-agent-sdk": "0.2.163"}
        ):
            printed = self._check([{"provider": "codex", "model": "gpt-6.1-sol"}])
        self.assertIn(
            f"runtime: Codex codex-cli 0.159.3, Claude SDK 0.2.162 (source: side: {active['venv']})", printed
        )
        self.assertTrue(any(line.startswith("openai-codex 0.160.0 is out (vNext runs 0.159.3)") for line in printed))
        self.assertTrue(any(line.startswith("claude-agent-sdk 0.2.163 is out (vNext runs 0.2.162)") for line in printed))
        self.assertIn("models not tested by vNext: gpt-7-nova (codex)", printed)

    def test_zai_and_commandcode_say_why_the_exact_model_waits_for_the_first_reply(self) -> None:
        printed = self._check([
            {"provider": "zai", "model": "glm-5.3"},
            {"provider": "commandcode", "model": "deepseek/deepseek-v4.1-flash"},
        ])
        reason = "exact model unknown until the first reply: this provider publishes no model table"
        self.assertIn(f"glm-5.3 (zai) -> {reason}", printed)
        self.assertIn(f"deepseek/deepseek-v4.1-flash (commandcode) -> {reason}", printed)


class FirstReplyRowTests(unittest.TestCase):
    """A provider with no model table gets its exact model from the first reply."""

    def test_a_commandcode_or_zai_row_takes_the_first_reply_model(self) -> None:
        import threading
        import time

        from vnext.vnext_model_identity import identity_view

        for provider, alias, ran in (
            ("commandcode", "deepseek/deepseek-v4.1-flash", "deepseek-v4.1-flash-0925"),
            ("zai", "glm-5.3", "glm-5.3"),
        ):
            with tempfile.TemporaryDirectory() as temp:
                service = object.__new__(server.VNextMcpService)
                service._session_id = "first-reply"
                service._failing_writes = set()
                service.workspace = Path(temp)
                service._client = "claude-code"
                service._outcome_log = Path(temp) / "outcomes.jsonl"
                service._log_lock = threading.Lock()
                service._outcome_rows = {}
                service._latest_usage = {}
                service._usage_unavailable = {}
                view = {"agent_id": "worker", "role": "worker", "provider": provider, "model": alias,
                        "status": "completed", "usage": {},
                        **identity_view({"model_ran": ran, "model_ran_first": ran})}
                service._record_outcome(view, time.time())
                row = json.loads(service._outcome_log.read_text(encoding="utf-8").splitlines()[-1])
            self.assertEqual(ran, row["model_exact"], provider)
            self.assertEqual("first_reply", row["model_exact_source"], provider)


class UpdateCommandTests(_RuntimesDir):
    """The install path with a fake runner: no venv is made and nothing is downloaded."""

    def _runner(self, sdk_version: str = "0.2.162"):
        codex = self._fake_codex()
        cli = _write_program(self.root / "side" / "claude", "2.1.285 (Claude Code)")
        calls: list[list[str]] = []

        def run(command, timeout=None):
            calls.append(list(command))
            if command[:3] == [sys.executable, "-m", "venv"]:
                python = Path(command[3]) / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
                python.parent.mkdir(parents=True, exist_ok=True)
                python.write_text("", encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, "", "")
            if command[1:3] == ["-c", runtimes._LOCATE_SCRIPT]:
                payload = {"codex": str(codex), "sdk_version": sdk_version, "claude": str(cli)}
                return subprocess.CompletedProcess(command, 0, json.dumps(payload) + "\n", "")
            if command[-1] == "--version" and command[0] in (str(codex), str(cli)):
                return subprocess.run(command, capture_output=True, text=True, check=False)
            return subprocess.CompletedProcess(command, 0, "", "")

        return run, calls

    def _update(self, *versions: str | None, sdk_version: str = "0.2.162", codex_models=None):
        run, calls = self._runner(sdk_version)
        out, err = io.StringIO(), io.StringIO()
        code = runtimes.update_runtimes(
            *versions, out=out, err=err, runner=run,
            fetch=lambda name: FAKE_PYPI[name],
            codex_probe=lambda exe, ws: (codex_models or [
                {"id": entry["model"], "model": entry["model"], "hidden": False}
                for entry in server.DEFAULT_CATALOG if entry.get("provider") == "codex"
            ], None),
            claude_probe=lambda py, cli, ws: ([{"value": "opus", "resolvedModel": "claude-opus-5-5"}], None),
        )
        return code, out.getvalue(), err.getvalue(), calls

    def test_an_explicit_pair_is_installed_recorded_and_announced_for_the_next_restart(self) -> None:
        code, out, err, calls = self._update("0.159.3", "0.2.162")
        self.assertEqual(0, code, err)
        record = json.loads((self.root / "runtimes" / "active.json").read_text(encoding="utf-8"))
        self.assertEqual("codex-cli 0.159.3", record["codex"]["version"])
        self.assertEqual("0.159.3", record["codex"]["package_version"])
        self.assertEqual("0.2.162", record["claude"]["sdk_version"])
        self.assertEqual(64, len(record["codex"]["sha256"]))
        self.assertEqual(64, len(record["claude"]["cli_sha256"]))
        self.assertEqual("2.1.285 (Claude Code)", record["claude"]["cli_version"])
        self.assertTrue(record["venv"].endswith("0.159.3-0.2.162"))
        installs = [c for c in calls if "install" in c]
        self.assertTrue(any("openai-codex==0.159.3" in c and "claude-agent-sdk==0.2.162" in c for c in installs))
        self.assertIn("takes effect at the next restart", out)
        self.assertIn("a restart stops running workers", out)

    def test_no_flags_installs_the_newest_stable_pair(self) -> None:
        code, out, err, _calls = self._update(sdk_version="0.2.170")
        self.assertEqual(0, code, err)
        self.assertIn("openai-codex 0.161.0 and claude-agent-sdk 0.2.170", out)

    def test_a_prerelease_is_refused(self) -> None:
        code, _out, err, calls = self._update("0.162.0a1", "0.2.162")
        self.assertEqual(2, code)
        self.assertIn("stable X.Y.Z", err)
        self.assertEqual([], calls)

    def test_a_failed_probe_changes_nothing(self) -> None:
        run, _calls = self._runner()
        err = io.StringIO()
        code = runtimes.update_runtimes(
            "0.159.3", "0.2.162", out=io.StringIO(), err=err, runner=run,
            codex_probe=lambda exe, ws: ([], "the side Codex app-server did not answer model/list: boom"),
            claude_probe=lambda py, cli, ws: ([], None),
        )
        self.assertEqual(1, code)
        self.assertIn("nothing changed", err.getvalue())
        self.assertFalse((self.root / "runtimes" / "active.json").exists())

    def test_rollback_removes_the_record_and_keeps_the_side_folders(self) -> None:
        self.assertEqual(0, self._update("0.159.3", "0.2.162")[0])
        venv = self.root / "runtimes" / "0.159.3-0.2.162"
        out = io.StringIO()
        self.assertEqual(0, runtimes.main(["--update-runtimes", "--rollback"], out=out))
        self.assertIsNone(runtimes.read_active_runtime())
        self.assertTrue(venv.is_dir())
        self.assertIn("(source: pinned)", out.getvalue())
        self.assertIn(f"codex-cli {RELEASE_CODEX_VERSION}", out.getvalue())

    def test_the_server_entry_points_route_the_flag_to_the_update_command(self) -> None:
        from vnext import vnext_mcp_reload as reload

        with patch.object(runtimes, "main", return_value=7) as update:
            self.assertEqual(7, server.main(["--update-runtimes", "--rollback"]))
            self.assertEqual(7, reload.main(["--update-runtimes", "--rollback"]))
            self.assertEqual(7, server.main(["--rollback"]))
            self.assertEqual(7, reload.main(["--rollback"]))
        self.assertEqual(4, update.call_count)


if __name__ == "__main__":
    unittest.main()
