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


# The real CLI can omit sonnet and expose two fable rows without an alias row.
REALISTIC_CLAUDE_MODELS = [
    {"value": "default", "resolvedModel": "claude-opus-5-5"},
    {"value": "opus", "resolvedModel": "claude-opus-5-5"},
    {"value": "claude-fable-5-1", "resolvedModel": "claude-fable-5-1"},
    {"value": "claude-fable-5", "resolvedModel": "claude-fable-5"},
    {"value": "claude-haiku-4-5", "resolvedModel": "claude-haiku-4-5"},
    {"value": "claude-haiku-4-6", "resolvedModel": "claude-haiku-4-6"},
]


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


class AutoUpdateTests(_RuntimesDir):
    """Automatic updates use only fake installers, fake probes and temporary folders."""

    def setUp(self) -> None:
        super().setUp()
        env = patch.dict(os.environ, {"VNEXT_NO_UPDATE_CHECK": "0", "VNEXT_AUTO_UPDATE": "1"})
        env.start()
        self.addCleanup(env.stop)
        self.newest = {runtimes.CODEX_PACKAGE: "0.161.0", runtimes.SDK_PACKAGE: "0.2.170"}
        self.calls = []

    def _read(self, name):
        return json.loads((runtimes.runtimes_dir() / name).read_text(encoding="utf-8"))

    def _fake_runner(self, command, timeout=None):
        self.calls.append(list(command))
        if command[:3] == [sys.executable, "-m", "venv"]:
            venv = Path(command[3])
            python = runtimes._venv_python(venv)
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_bytes(b"fake python; the runner never executes this file")
            python.chmod(0o755)
            _write_program(venv / "codex", "codex-cli " + self.newest[runtimes.CODEX_PACKAGE])
            _write_program(venv / "claude", "fake claude")
        if command[1:3] == ["-c", runtimes._LOCATE_SCRIPT]:
            venv = Path(command[0]).parent.parent
            suffix = ".cmd" if sys.platform == "win32" else ""
            output = json.dumps({"codex": str(venv / ("codex" + suffix)),
                                 "claude": str(venv / ("claude" + suffix)),
                                 "sdk_version": self.newest[runtimes.SDK_PACKAGE]})
        elif command[-1] == "--version" and Path(command[0]).name.startswith("codex"):
            output = "codex-cli " + self.newest[runtimes.CODEX_PACKAGE]
        else:
            output = "fake 1.0.0"
        return subprocess.CompletedProcess(command, 0, output, "")

    def _auto(self, **overrides):
        options = {"runner": self._fake_runner,
                   "codex_probe": lambda exe, ws: ([{"id": entry["model"]}
                       for entry in server.DEFAULT_CATALOG if entry.get("provider") == "codex"], None),
                   "claude_probe": lambda py, cli, ws: ([{"value": entry["model"]}
                       for entry in server.DEFAULT_CATALOG if entry.get("provider") == "claude"], None)}
        options.update(overrides)
        runtimes.auto_update(self.newest, **options)

    def _stage(self):
        self._auto()
        return self._read("staged.json")

    def _service(self):
        return server.VNextMcpService(
            workspace=self.root, catalog=[{"provider": "codex", "model": "gpt-6-sol"}],
            event_log=None, status_file=None, outcome_log=None,
            adapter_factories={"codex": lambda: None},
        )

    def _main(self, catalog=None):
        # Resolve the real default catalog and construct the real service, but
        # never open its transport or start a provider.
        login = MagicMock()
        login.codex_available.return_value = True
        login.claude_available.return_value = False
        login.claude_state.return_value = server.CLAUDE_LOGIN_AVAILABLE
        with patch.dict(os.environ, {"VNEXT_NO_UPDATE_CHECK": "1"}), patch.object(
            server, "_LoginProbe", return_value=login
        ), patch.object(server, "load_zai_provider", return_value=None), patch.object(
            server, "load_commandcode_provider", return_value=None
        ), patch.object(server.VNextMcpService, "serve_stdio", return_value=0), patch.object(
            server, "VNextRuntimeSession", wraps=server.VNextRuntimeSession
        ) as sessions, patch.object(server, "read_active_runtime", wraps=runtimes.read_active_runtime) as reads:
            self.assertEqual(0, server.main(["--workspace", str(self.root), "--stdio",
                "--event-log", "none", "--status-file", "none", "--outcome-log", "none",
                *(["--catalog", str(catalog)] if catalog else [])]))
        return sessions.call_args.args[0].config, reads

    def test_staging_leaves_the_active_bytes_and_running_executables_untouched(self):
        old = self._active()
        pending = runtimes.runtime_notice_lines(active=old, newest=self.newest)
        self.assertTrue(any("automatically" in line for line in pending))
        self.assertFalse(any("Tell your agent" in line for line in pending))
        before = (runtimes.runtimes_dir() / "active.json").read_bytes()
        executable = Path(old["codex"]["executable"]).read_bytes()
        staged = self._stage()
        self.assertEqual(before, (runtimes.runtimes_dir() / "active.json").read_bytes())
        self.assertEqual(executable, Path(old["codex"]["executable"]).read_bytes())
        self.assertEqual("0.161.0", staged["codex"]["package_version"])
        self.assertEqual("staged", self._read("last_update.json")["state"])
        self.assertFalse((runtimes.runtimes_dir() / "update.lock").exists())

    def test_existing_unstamped_folder_is_never_reused(self):
        old = runtimes.runtimes_dir() / "0.161.0-0.2.170"
        old.mkdir(parents=True)
        sentinel = old / "codex"
        sentinel.write_bytes(b"keep this live executable")
        staged = self._stage()
        self.assertRegex(Path(staged["venv"]).name, r"^0\.161\.0-0\.2\.170-\d{14}$")
        self.assertEqual(b"keep this live executable", sentinel.read_bytes())
        self.assertNotEqual(old, Path(staged["venv"]))

    def test_interrupted_timestamped_folder_is_not_reused_even_in_the_same_second(self):
        first = self._stage()
        stamp = runtimes._dt.datetime.strptime(Path(first["venv"]).name[-14:], "%Y%m%d%H%M%S")
        (runtimes.runtimes_dir() / "staged.json").unlink()
        (runtimes.runtimes_dir() / "last_update.json").unlink()
        with patch.object(runtimes._dt, "datetime", wraps=runtimes._dt.datetime) as clock:
            clock.now.return_value = stamp.replace(tzinfo=runtimes._dt.timezone.utc)
            second = self._stage()
        self.assertNotEqual(first["venv"], second["venv"])
        self.assertTrue(Path(first["venv"]).is_dir())

    def test_live_pid_lock_blocks_a_second_installer(self):
        runtimes.runtimes_dir().mkdir()
        lock = runtimes.runtimes_dir() / "update.lock"
        lock.write_text(str(os.getpid()))
        self._auto()
        self.assertEqual([], self.calls)
        self.assertEqual(str(os.getpid()), lock.read_text())
        self.assertFalse((runtimes.runtimes_dir() / "last_update.json").exists())

    def test_dead_pid_lock_is_taken_over(self):
        runtimes.runtimes_dir().mkdir()
        (runtimes.runtimes_dir() / "update.lock").write_text("123456789")
        with patch.object(runtimes.os, "kill", side_effect=ProcessLookupError):
            self._auto()
        self.assertEqual("staged", self._read("last_update.json")["state"])
        self.assertFalse((runtimes.runtimes_dir() / "update.lock").exists())

    def test_notice_only_switch_leaves_no_install_or_attempt(self):
        with patch.dict(os.environ, {"VNEXT_AUTO_UPDATE": "0"}):
            self._auto()
            lines = runtimes.runtime_notice_lines(newest=self.newest)
        self.assertEqual([], self.calls)
        self.assertFalse(runtimes.runtimes_dir().exists())
        self.assertTrue(any("is out" in line and "vnext-mcp --update-runtimes" in line for line in lines))

    def test_full_off_switch_does_no_check_or_install(self):
        fetch = MagicMock()
        with patch.dict(os.environ, {"VNEXT_NO_UPDATE_CHECK": "1"}):
            self._auto()
            self.assertIsNone(runtimes.check_for_updates(fetch=fetch))
        fetch.assert_not_called()
        self.assertEqual([], self.calls)
        self.assertFalse(runtimes.runtimes_dir().exists())

    def test_active_pair_and_pinned_pair_are_skipped(self):
        self._active()
        self.newest = {runtimes.CODEX_PACKAGE: "0.159.3", runtimes.SDK_PACKAGE: "0.2.162"}
        self._auto()
        (runtimes.runtimes_dir() / "active.json").unlink()
        self.newest = {runtimes.CODEX_PACKAGE: RELEASE_CODEX_VERSION, runtimes.SDK_PACKAGE: CLAUDE_SDK_VERSION}
        self._auto()
        self.assertEqual([], self.calls)

    def test_staged_pair_and_todays_attempt_are_skipped(self):
        self._stage()
        self.calls.clear()
        self._auto()
        self.assertEqual([], self.calls)
        (runtimes.runtimes_dir() / "staged.json").unlink()
        self._auto()
        self.assertEqual([], self.calls)

    def test_failed_verification_keeps_active_and_records_one_line_reason(self):
        self._active()
        before = (runtimes.runtimes_dir() / "active.json").read_bytes()
        record, reason = runtimes._build_runtime(
            runtimes.runtimes_dir() / "verification-only", "0.161.0", "0.2.170",
            runner=self._fake_runner,
            codex_probe=lambda exe, ws: ([], "model/list failed\nboom"),
            claude_probe=lambda py, cli, ws: ([], None),
        )
        self.assertIsNone(record)
        self.assertNotIn("\n", reason)
        self._auto(codex_probe=lambda exe, ws: ([], "model/list failed\nboom"))
        state = self._read("last_update.json")
        self.assertEqual("failed", state["state"])
        self.assertIn("model/list failed boom", state["reason"])
        self.assertEqual(self.newest, state["to"])
        self.assertEqual(before, (runtimes.runtimes_dir() / "active.json").read_bytes())
        self.assertFalse((runtimes.runtimes_dir() / "staged.json").exists())
        self.assertFalse((runtimes.runtimes_dir() / "update.lock").exists())
        self.assertTrue(any("could not update to" in line and "still on Codex 0.159.3" in line
                            for line in runtimes.runtime_notice_lines(active=runtimes.read_active_runtime())))

    def test_thrown_install_exception_records_failure_and_releases_lock(self):
        self._auto(runner=MagicMock(side_effect=OSError("disk full")))
        self.assertIn("disk full", self._read("last_update.json")["reason"])
        self.assertFalse((runtimes.runtimes_dir() / "update.lock").exists())

    def test_main_promotes_before_catalog_validation_and_session_config(self):
        self._active(models=[{"id": "old-only"}])  # This catalog cannot start before promotion.
        staged = self._stage()
        config, reads = self._main()
        self.assertGreaterEqual(reads.call_count, 3)
        self.assertEqual(staged["codex"]["executable"], config["codex_runtime"]["executable"])
        self.assertEqual("0.2.170", config["claude_runtime"]["sdk_version"])
        self.assertEqual("switched", self._read("last_update.json")["state"])
        self.assertEqual("0.159.3", self._read("previous.json")["codex"]["package_version"])
        self.assertFalse((runtimes.runtimes_dir() / "staged.json").exists())
        self.assertTrue(any("vNext switched itself to Codex 0.161.0 and Claude SDK 0.2.170" in line
                            and "(was Codex 0.159.3" in line and "--rollback" in line
                            for line in runtimes.runtime_notice_lines(active=runtimes.read_active_runtime())))

    def test_check_reports_staging_and_never_promotes(self):
        old = self._active()
        self._stage()
        out = io.StringIO()
        with patch.object(server, "_load_catalog", return_value=[{"provider": "codex", "model": "gpt-6.1-sol"}]), patch.object(
            server, "startup_check_notes", return_value=[]
        ), patch.object(server, "check_for_updates", return_value=self.newest):
            self.assertEqual(0, server.run_startup_check(["--workspace", str(self.root)], out=out))
        self.assertEqual(old, runtimes.read_active_runtime())
        self.assertIn("update to Codex 0.161.0 and Claude SDK 0.2.170 is ready and applies at the next start", out.getvalue())
        self.assertTrue((runtimes.runtimes_dir() / "staged.json").exists())

    def test_promotion_from_pinned_records_a_pinned_previous(self):
        self._stage()
        runtimes.promote_staged()
        self.assertEqual({"pinned": True}, self._read("previous.json"))
        runtimes.rollback(out=io.StringIO())
        self.assertIsNone(runtimes.read_active_runtime())
        self.assertIn(self.newest, self._read("declined.json")["pairs"])

    def test_rollback_restores_previous_declines_pair_and_deletes_staging(self):
        old = self._active()
        staged = self._stage()
        runtimes.promote_staged()
        (runtimes.runtimes_dir() / "staged.json").write_text(json.dumps(staged))
        out = io.StringIO()
        self.assertEqual(0, runtimes.rollback(out=out))
        self.assertEqual(old, runtimes.read_active_runtime())
        self.assertFalse((runtimes.runtimes_dir() / "staged.json").exists())
        self.assertIn(self.newest, self._read("declined.json")["pairs"])
        self.assertIn("active runtime: Codex codex-cli 0.159.3", out.getvalue())
        self.calls.clear()
        self._auto()
        self.assertEqual([], self.calls)
        self.assertFalse(runtimes.auto_update_needed(self.newest))

    def test_changed_staged_codex_is_dropped_and_main_still_starts(self):
        old = self._active(models=[{"id": entry["model"]} for entry in server.DEFAULT_CATALOG if entry.get("provider") == "codex"])
        staged = self._stage()
        Path(staged["codex"]["executable"]).write_text("changed")
        config, _ = self._main()
        self.assertEqual(old, runtimes.read_active_runtime())
        self.assertEqual(old["codex"]["executable"], config["codex_runtime"]["executable"])
        self.assertEqual("failed", self._read("last_update.json")["state"])
        self.assertIn("changed", self._read("last_update.json")["reason"])
        self.assertFalse((runtimes.runtimes_dir() / "staged.json").exists())

    def test_changed_staged_claude_is_dropped(self):
        old = self._active()
        staged = self._stage()
        Path(staged["claude"]["cli_path"]).write_text("changed")
        runtimes.promote_staged()
        self.assertEqual(old, runtimes.read_active_runtime())
        self.assertIn("Claude CLI changed", self._read("last_update.json")["reason"])
        self.assertFalse((runtimes.runtimes_dir() / "staged.json").exists())

    def test_move_failure_restores_active_and_never_stops_main(self):
        old = self._active(models=[{"id": entry["model"]} for entry in server.DEFAULT_CATALOG if entry.get("provider") == "codex"])
        self._stage()
        replace = os.replace
        def fail_staged(source, target):
            if Path(source).name == "staged.json":
                raise OSError("promotion denied")
            return replace(source, target)
        with patch.object(runtimes.os, "replace", side_effect=fail_staged):
            config, _ = self._main()
        self.assertEqual(old, runtimes.read_active_runtime())
        self.assertEqual(old["codex"]["executable"], config["codex_runtime"]["executable"])
        self.assertIn("promotion denied", self._read("last_update.json")["reason"])

    def test_unwritable_failure_record_cannot_stop_main(self):
        self._stage()
        with patch.object(runtimes, "_write_record", side_effect=OSError("read-only")):
            config, _ = self._main()
        self.assertNotIn("codex_runtime", config)
        self.assertIsNone(runtimes.read_active_runtime())

    def test_recording_switch_failure_restores_the_previous_pair(self):
        old = self._active()
        self._stage()
        writer = runtimes._write_record
        def refuse_switched(name, record):
            if name == "last_update.json" and record.get("state") == "switched":
                raise OSError("cannot record switch")
            return writer(name, record)
        with patch.object(runtimes, "_write_record", side_effect=refuse_switched):
            runtimes.promote_staged()
        self.assertEqual(old, runtimes.read_active_runtime())
        self.assertEqual("failed", self._read("last_update.json")["state"])
        self.assertIn("cannot record switch", self._read("last_update.json")["reason"])

    def test_recording_switch_failure_restores_pinned_selection(self):
        self._stage()
        writer = runtimes._write_record
        def refuse_switched(name, record):
            if name == "last_update.json" and record.get("state") == "switched":
                raise OSError("cannot record switch")
            return writer(name, record)
        with patch.object(runtimes, "_write_record", side_effect=refuse_switched):
            runtimes.promote_staged()
        self.assertIsNone(runtimes.read_active_runtime())
        self.assertEqual("failed", self._read("last_update.json")["state"])

    def test_existing_staging_blocks_a_cached_newer_pair_daemon(self):
        self._stage()
        newer = {runtimes.CODEX_PACKAGE: "0.162.0", runtimes.SDK_PACKAGE: "0.2.171"}
        with patch.object(server, "check_for_updates", return_value=newer), patch.object(server, "auto_update") as update:
            service = self._service()
            try:
                self.assertIsNone(service._update_check_thread)
                update.assert_not_called()
            finally:
                service.close()

    def test_cached_newer_pair_launches_daemon_without_another_pypi_check(self):
        with patch.object(server, "check_for_updates", return_value=self.newest) as check, patch.object(
            server, "auto_update", side_effect=lambda newest: self._auto()
        ) as update:
            service = self._service()
            try:
                self.assertIsNotNone(service._update_check_thread)
                self.assertTrue(service._update_check_thread.daemon)
                service._update_check_thread.join(5)
                self.assertFalse(service._update_check_thread.is_alive())
                check.assert_called_once_with(cached_only=True)
                update.assert_called_once_with(self.newest)
                self.assertTrue(any("ready and applies at the next start" in line for line in service._runtime_notice))
            finally:
                service.close()
        self.assertEqual("staged", self._read("last_update.json")["state"])

    def test_cached_newer_pair_is_retried_after_interrupted_install(self):
        # An interrupted build has a dead lock and a partial folder, but no
        # completed attempt. The cached answer still starts the daemon today.
        runtimes.runtimes_dir().mkdir()
        partial = runtimes.runtimes_dir() / "0.161.0-0.2.170-20000101000000"
        partial.mkdir()
        (runtimes.runtimes_dir() / "update.lock").write_text("123456789")
        with patch.object(runtimes.os, "kill", side_effect=ProcessLookupError), patch.object(
            server, "check_for_updates", return_value=self.newest
        ), patch.object(server, "auto_update", side_effect=lambda newest: self._auto()):
            service = self._service()
            try:
                self.assertIsNotNone(service._update_check_thread)
                service._update_check_thread.join(5)
                self.assertEqual("staged", self._read("last_update.json")["state"])
                self.assertNotEqual(str(partial), self._read("staged.json")["venv"])
                self.assertTrue(partial.is_dir())
            finally:
                service.close()

    def test_completed_attempt_does_not_launch_another_daemon_today(self):
        self._auto(codex_probe=lambda exe, ws: ([], "failed"))
        with patch.object(server, "check_for_updates", return_value=self.newest), patch.object(server, "auto_update") as update:
            service = self._service()
            try:
                self.assertIsNone(service._update_check_thread)
                update.assert_not_called()
            finally:
                service.close()

    def test_old_failed_attempt_can_be_retried(self):
        self._auto(codex_probe=lambda exe, ws: ([], "failed"))
        state = self._read("last_update.json")
        state["day"] = "2000-01-01"
        (runtimes.runtimes_dir() / "last_update.json").write_text(json.dumps(state))
        self.calls.clear()
        self._auto()
        self.assertTrue(self.calls)
        self.assertEqual("staged", self._read("last_update.json")["state"])


    def test_review_custom_catalog_keeps_incompatible_update_staged(self):
        models = [{"id": entry["model"]} for entry in server.DEFAULT_CATALOG if entry.get("provider") == "codex"]
        old = self._active(models=[*models, {"id": "gpt-legacy"}])
        staged = self._stage()
        # Startup validation accepts Codex IDs, never a different row's model label.
        staged["codex"]["models"].append({"id": "different-id", "model": "gpt-legacy"})
        (runtimes.runtimes_dir() / "staged.json").write_text(json.dumps(staged))
        catalog = self.root / "catalog.json"
        catalog.write_text(json.dumps([{"provider": "codex", "model": "gpt-legacy"}]))
        config, _ = self._main(catalog=catalog)
        self.assertEqual(old, runtimes.read_active_runtime())
        self.assertEqual(old["codex"]["executable"], config["codex_runtime"]["executable"])
        self.assertEqual(staged, self._read("staged.json"))
        self.assertEqual("failed", self._read("last_update.json")["state"])
        self.assertIn("gpt-legacy", self._read("last_update.json")["reason"])
        self.assertTrue(any("gpt-legacy" in line and "could not update" in line
                            for line in runtimes.runtime_notice_lines(active=old)))

    def test_review_codex_catalog_model_is_checked_before_promotion(self):
        old = self._active()
        self._stage()
        runtimes.promote_staged([{"provider": "codex", "model": "unlisted-codex"}])
        self.assertEqual(old, runtimes.read_active_runtime())
        self.assertTrue((runtimes.runtimes_dir() / "staged.json").exists())
        self.assertIn("unlisted-codex", self._read("last_update.json")["reason"])

    def test_review_sdk_family_alias_is_accepted_by_catalog_validation(self):
        self._active()
        staged = self._stage()
        staged["claude"]["models"] = list(REALISTIC_CLAUDE_MODELS)
        (runtimes.runtimes_dir() / "staged.json").write_text(json.dumps(staged))
        runtimes.promote_staged([{"provider": "claude", "model": "opus"}])
        self.assertEqual(staged, runtimes.read_active_runtime())
        self.assertEqual("switched", self._read("last_update.json")["state"])

    def test_default_catalog_promotes_with_realistic_claude_models(self):
        self._active()
        self._auto(claude_probe=lambda py, cli, ws: (list(REALISTIC_CLAUDE_MODELS), None))
        staged = self._read("staged.json")
        runtimes.promote_staged(server.DEFAULT_CATALOG)
        self.assertEqual(staged, runtimes.read_active_runtime())
        self.assertFalse((runtimes.runtimes_dir() / "staged.json").exists())
        self.assertEqual("switched", self._read("last_update.json")["state"])

    def test_two_promotions_keep_an_old_runtime_used_by_a_live_pid(self):
        old = self._active()
        folder = runtimes.runtimes_dir() / "0.159.3-0.2.162-20000101000000"
        folder.mkdir()
        old["venv"] = str(folder)
        (runtimes.runtimes_dir() / "active.json").write_text(json.dumps(old))
        aged = runtimes._dt.datetime.now(runtimes._dt.timezone.utc).timestamp() - 20 * 86400
        os.utime(folder, (aged, aged))
        marker = runtimes.runtimes_dir() / "in-use" / f"{os.getpid()}.json"
        marker.parent.mkdir()
        marker.write_text(json.dumps({"venv": str(folder)}))
        self._stage()
        runtimes.promote_staged()
        self.assertTrue(folder.is_dir())
        self.newest = {runtimes.CODEX_PACKAGE: "0.162.0", runtimes.SDK_PACKAGE: "0.2.171"}
        self._stage()
        runtimes.promote_staged()
        self.assertEqual("0.162.0", runtimes.read_active_runtime()["codex"]["package_version"])
        self.assertNotEqual(str(folder), self._read("previous.json")["venv"])
        self.assertTrue(folder.is_dir())
        self.assertTrue(marker.is_file())

    def test_cleanup_removes_dead_pid_markers_and_their_old_runtime(self):
        self._active()
        folder = runtimes.runtimes_dir() / "0.1.0-0.1.0-20000101000000"
        folder.mkdir()
        aged = runtimes._dt.datetime.now(runtimes._dt.timezone.utc).timestamp() - 20 * 86400
        os.utime(folder, (aged, aged))
        marker = runtimes.runtimes_dir() / "in-use" / "123456789.json"
        marker.parent.mkdir()
        marker.write_text(json.dumps({"venv": str(folder)}))
        with patch("vnext.vnext_report._pid_alive", return_value=False) as alive:
            removed = runtimes._cleanup_runtime_folders()
        self.assertFalse(marker.exists())
        alive.assert_called_once_with(123456789)
        self.assertEqual([folder.name], removed)
        self.assertFalse(folder.exists())

    def test_main_registers_promoted_runtime_and_exit_removes_its_marker(self):
        self._active()
        staged = self._stage()
        marker = runtimes.runtimes_dir() / "in-use" / f"{os.getpid()}.json"
        resolve = server.resolve_startup
        def startup_with_marker(**kwargs):
            self.assertEqual({"venv": str(Path(staged["venv"]).resolve())},
                             json.loads(marker.read_text()) if marker.exists() else None)
            return resolve(**kwargs)
        with patch("atexit.register") as register, patch.object(
            server, "resolve_startup", side_effect=startup_with_marker
        ):
            config, _ = self._main()
        self.assertEqual(staged["codex"]["executable"], config["codex_runtime"]["executable"])
        register.assert_called_once()
        cleanup = register.call_args.args[0]
        with patch.object(Path, "unlink", side_effect=PermissionError("file is busy")):
            cleanup()  # Best effort on exit: errors must never escape.
        self.assertTrue(marker.exists())
        cleanup()
        self.assertFalse(marker.exists())

    def test_fresh_empty_lock_cannot_be_claimed_by_a_second_process(self):
        root = runtimes.runtimes_dir()
        root.mkdir()
        lock = root / "update.lock"
        lock.write_text("")
        recent = runtimes._dt.datetime.now(runtimes._dt.timezone.utc).timestamp() - 2
        os.utime(lock, (recent, recent))
        handle = runtimes._acquire_update_lock()
        try:
            self.assertIsNone(handle)
            self.assertEqual("", lock.read_text())
        finally:
            if handle is not None:
                runtimes._release_update_lock(handle)

    def test_review_killed_install_is_recorded_before_the_first_runner_call(self):
        seen = []
        def killed(command, timeout=None):
            path = runtimes.runtimes_dir() / "last_update.json"
            seen.append(json.loads(path.read_text()) if path.exists() else None)
            raise SystemExit("simulated process kill")
        with self.assertRaises(SystemExit):
            self._auto(runner=killed)
        self.assertEqual("installing", seen[0]["state"])
        self.assertEqual(self.newest, seen[0]["to"])
        self.assertEqual(runtimes._dt.date.today().isoformat(), seen[0]["day"])
        self.assertFalse(runtimes.auto_update_needed(self.newest))
        self.calls.clear()
        self._auto()
        self.assertEqual([], self.calls)
        state = self._read("last_update.json")
        state["day"] = "2000-01-01"
        (runtimes.runtimes_dir() / "last_update.json").write_text(json.dumps(state))
        self.assertTrue(runtimes.auto_update_needed(self.newest))

    def test_review_failed_build_removes_only_its_own_new_folder(self):
        old = self._active()
        retained = runtimes.runtimes_dir() / "0.160.1-0.2.160-20000101000000"
        retained.mkdir()
        (retained / "keep").write_text("another install")
        self._auto(codex_probe=lambda exe, ws: ([], "verification refused"))
        own = Path(next(call[3] for call in self.calls if call[:3] == [sys.executable, "-m", "venv"]))
        self.assertFalse(own.exists())
        self.assertEqual("another install", (retained / "keep").read_text())
        self.assertEqual(old, runtimes.read_active_runtime())
        self.assertIn("verification refused", self._read("last_update.json")["reason"])

    def test_review_install_exception_also_removes_its_new_folder(self):
        def broken(command, timeout=None):
            self._fake_runner(command, timeout=timeout)
            raise OSError("installer died")
        try:
            raise ValueError("caller is handling an unrelated error")
        except ValueError:
            self._auto(runner=broken)
        own = Path(self.calls[0][3])
        self.assertFalse(own.exists())
        self.assertIn("installer died", self._read("last_update.json")["reason"])

    def test_review_promotion_cleans_only_old_unreferenced_runtime_folders(self):
        old = self._active()
        previous = Path(old["venv"])
        previous.mkdir()
        staged = self._stage()
        active = Path(staged["venv"])
        root = runtimes.runtimes_dir()
        stale = root / "0.1.1-0.1.1-20000101000000"
        stale.mkdir()
        (stale / "binary").write_bytes(b"obsolete")
        fresh = root / "0.1.2-0.1.2-20000101000000"
        fresh.mkdir()
        pinned = root / "0.1.3-0.1.3"
        pinned.mkdir()
        (pinned / "binary").write_bytes(b"pinned bundle")
        outside = self.root / "outside"
        outside.mkdir()
        link = root / "0.1.4-0.1.4-20000101000000"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except OSError:
            link = None
        aged = runtimes._dt.datetime.now(runtimes._dt.timezone.utc).timestamp() - 15 * 86400
        for folder in (previous, active, stale, pinned):
            os.utime(folder, (aged, aged))
        with patch.object(runtimes.sys, "prefix", str(pinned)):
            runtimes.promote_staged()
        self.assertFalse(stale.exists())
        self.assertTrue(previous.is_dir())
        self.assertTrue(active.is_dir())
        self.assertTrue(fresh.is_dir())
        self.assertEqual(b"pinned bundle", (pinned / "binary").read_bytes())
        self.assertTrue(outside.is_dir())
        if link is not None:
            self.assertTrue(link.is_symlink())
        self.assertEqual([stale.name], self._read("last_update.json")["removed"])

    def test_review_cleanup_retains_an_old_staged_folder(self):
        self._active()
        staged = self._stage()
        aged = runtimes._dt.datetime.now(runtimes._dt.timezone.utc).timestamp() - 15 * 86400
        os.utime(staged["venv"], (aged, aged))
        self.assertEqual([], runtimes._cleanup_runtime_folders())
        self.assertTrue(Path(staged["venv"]).is_dir())

    def test_review_manual_update_supersedes_staging_and_restores_exactly_one_step(self):
        old = self._active()
        self._stage()
        (runtimes.runtimes_dir() / "previous.json").write_text(json.dumps({"pinned": True}))
        self.newest = {runtimes.CODEX_PACKAGE: "0.170.0", runtimes.SDK_PACKAGE: "0.2.180"}
        self.assertEqual(0, runtimes.update_runtimes(
            "0.170.0", "0.2.180", out=io.StringIO(), err=io.StringIO(), runner=self._fake_runner,
            codex_probe=lambda exe, ws: ([{"id": entry["model"]} for entry in server.DEFAULT_CATALOG
                                          if entry.get("provider") == "codex"], None),
            claude_probe=lambda py, cli, ws: ([], None)))
        selected = runtimes.read_active_runtime()
        self.assertFalse((runtimes.runtimes_dir() / "staged.json").exists())
        self.assertFalse((runtimes.runtimes_dir() / "last_update.json").exists())
        self.assertEqual(old, self._read("previous.json"))
        runtimes.promote_staged()
        self.assertEqual(selected, runtimes.read_active_runtime())
        runtimes.rollback(out=io.StringIO())
        self.assertEqual(old, runtimes.read_active_runtime())

    def test_review_manual_update_from_pinned_preserves_the_pinned_previous(self):
        self.assertEqual(0, runtimes.update_runtimes(
            "0.161.0", "0.2.170", out=io.StringIO(), err=io.StringIO(), runner=self._fake_runner,
            codex_probe=lambda exe, ws: ([{"id": entry["model"]} for entry in server.DEFAULT_CATALOG
                                          if entry.get("provider") == "codex"], None),
            claude_probe=lambda py, cli, ws: ([], None)))
        self.assertEqual({"pinned": True}, self._read("previous.json"))

    def test_review_empty_and_unparseable_locks_are_taken_over(self):
        root = runtimes.runtimes_dir()
        root.mkdir()
        for content in ("", "not-a-pid"):
            lock = root / "update.lock"
            lock.write_text(content)
            aged = runtimes._dt.datetime.now(runtimes._dt.timezone.utc).timestamp() - 11
            os.utime(lock, (aged, aged))
            self._auto()
            self.assertEqual("staged", self._read("last_update.json")["state"])
            (root / "staged.json").unlink()
            (root / "last_update.json").unlink()

    def test_review_six_hour_old_live_lock_is_taken_over(self):
        root = runtimes.runtimes_dir()
        root.mkdir()
        lock = root / "update.lock"
        lock.write_text(str(os.getpid()))
        old = runtimes._dt.datetime.now(runtimes._dt.timezone.utc).timestamp() - 6 * 3600 - 1
        os.utime(lock, (old, old))
        read = Path.read_text
        def unreadable_pid(path, *args, **kwargs):
            if path.name == "update.lock":
                raise PermissionError("expired PID is unreadable")
            return read(path, *args, **kwargs)
        with patch.object(Path, "read_text", new=unreadable_pid):
            self._auto()
        self.assertEqual("staged", self._read("last_update.json")["state"])
        self.assertFalse(lock.exists())

    def test_review_fresh_live_lock_blocks_and_explains_the_notice(self):
        root = runtimes.runtimes_dir()
        root.mkdir()
        (root / "update.lock").write_text(str(os.getpid()))
        self._auto()
        self.assertEqual([], self.calls)
        self.assertTrue(any("waiting" in line and "update.lock" in line and str(os.getpid()) in line
                            for line in runtimes.runtime_notice_lines(newest=self.newest)))

    def test_review_live_lock_defers_promotion_with_a_notice(self):
        old = self._active(models=[{"id": entry["model"]} for entry in server.DEFAULT_CATALOG
                                   if entry.get("provider") == "codex"])
        staged = self._stage()
        (runtimes.runtimes_dir() / "update.lock").write_text(str(os.getpid()))
        config, _ = self._main()
        self.assertEqual(old, runtimes.read_active_runtime())
        self.assertEqual(staged, self._read("staged.json"))
        self.assertEqual(old["codex"]["executable"], config["codex_runtime"]["executable"])
        self.assertTrue(any("waiting" in line and "update.lock" in line
                            for line in runtimes.runtime_notice_lines(active=old)))

    def test_review_previous_is_written_atomically_while_active_stays_present(self):
        old = self._active()
        self._stage()
        replace = os.replace
        events = []
        def observed(source, target):
            if Path(target).name == "previous.json":
                self.assertNotEqual("active.json", Path(source).name)
                self.assertEqual(old, runtimes.read_active_runtime())
                events.append("previous")
            if Path(source).name == "staged.json":
                self.assertEqual(old, runtimes.read_active_runtime())
                self.assertEqual(old, self._read("previous.json"))
                events.append("active")
            return replace(source, target)
        with patch.object(runtimes.os, "replace", side_effect=observed):
            runtimes.promote_staged()
        self.assertEqual(["previous", "active"], events)
        self.assertEqual("switched", self._read("last_update.json")["state"])

    def test_review_promotion_crash_checkpoints_preserve_previous(self):
        for checkpoint in ("before_previous", "after_previous", "after_active"):
            old = self._active()
            for name in ("staged.json", "previous.json", "last_update.json"):
                (runtimes.runtimes_dir() / name).unlink(missing_ok=True)
            staged = self._stage()
            replace = os.replace
            def killed(source, target):
                is_previous = Path(target).name == "previous.json"
                is_active = Path(source).name == "staged.json"
                if checkpoint == "before_previous" and is_previous:
                    raise SystemExit("killed before backup")
                result = replace(source, target)
                if (checkpoint == "after_previous" and is_previous) or (checkpoint == "after_active" and is_active):
                    raise SystemExit("killed after replace")
                return result
            with patch.object(runtimes.os, "replace", side_effect=killed):
                with self.assertRaises(SystemExit):
                    runtimes.promote_staged()
            if checkpoint != "after_active":
                self.assertEqual(old, runtimes.read_active_runtime())
            runtimes.promote_staged()
            self.assertEqual(old, self._read("previous.json"))
            self.assertEqual(staged, runtimes.read_active_runtime())
            runtimes.rollback(out=io.StringIO())
            self.assertEqual(old, runtimes.read_active_runtime())
            (runtimes.runtimes_dir() / "declined.json").unlink(missing_ok=True)

    def test_review_legacy_missing_active_crash_layout_retains_previous(self):
        old = self._active()
        self._stage()
        (runtimes.runtimes_dir() / "previous.json").write_text(json.dumps(old))
        (runtimes.runtimes_dir() / "active.json").unlink()
        runtimes.promote_staged()
        self.assertEqual(old, self._read("previous.json"))
        runtimes.rollback(out=io.StringIO())
        self.assertEqual(old, runtimes.read_active_runtime())

    def test_review_duplicate_selected_staging_does_not_overwrite_previous(self):
        old = self._active()
        staged = self._stage()
        runtimes.promote_staged()
        (runtimes.runtimes_dir() / "staged.json").write_text(json.dumps(staged))
        runtimes.promote_staged()
        self.assertEqual(old, self._read("previous.json"))

    def test_review_corrupt_auxiliary_records_are_ignored_in_service_and_main(self):
        old = self._active(models=[{"id": entry["model"]} for entry in server.DEFAULT_CATALOG
                                   if entry.get("provider") == "codex"])
        malformed = {
            "declined.json": {"pairs": None},
            "staged.json": {"codex": {}, "claude": {}},
            "previous.json": {"pinned": "wrong-type"},
            "last_update.json": {"state": "failed", "to": []},
        }
        for name, value in malformed.items():
            (runtimes.runtimes_dir() / name).write_text(json.dumps(value))
            with patch.object(server, "check_for_updates", return_value=self.newest), patch.object(server, "auto_update"):
                service = self._service()
                try:
                    if service._update_check_thread:
                        service._update_check_thread.join(5)
                    self.assertTrue(any("ignored " + name in line for line in service._runtime_notice))
                finally:
                    service.close()
            config, _ = self._main()
            self.assertEqual(old["codex"]["executable"], config["codex_runtime"]["executable"])
            self.assertEqual(old, runtimes.read_active_runtime())
            (runtimes.runtimes_dir() / name).unlink(missing_ok=True)

    def test_review_unreadable_metadata_is_noted_without_stopping_start(self):
        old = self._active(models=[{"id": entry["model"]} for entry in server.DEFAULT_CATALOG
                                   if entry.get("provider") == "codex"])
        read = Path.read_text
        def unreadable(path, *args, **kwargs):
            if path.name in ("declined.json", "staged.json", "previous.json", "last_update.json"):
                raise PermissionError("unreadable metadata")
            return read(path, *args, **kwargs)
        with patch.object(Path, "read_text", new=unreadable):
            config, _ = self._main()
            lines = runtimes.runtime_notice_lines(active=old)
        self.assertEqual(old["codex"]["executable"], config["codex_runtime"]["executable"])
        self.assertTrue(any("ignored declined.json" in line for line in lines))

    def test_review_corrupt_declined_record_cannot_break_rollback(self):
        old = self._active()
        self._stage()
        runtimes.promote_staged()
        (runtimes.runtimes_dir() / "declined.json").write_text(json.dumps({"pairs": None}))
        self.assertEqual(0, runtimes.rollback(out=io.StringIO()))
        self.assertEqual(old, runtimes.read_active_runtime())
        self.assertIn(self.newest, self._read("declined.json")["pairs"])

    def test_review_corrupt_previous_is_absent_for_rollback(self):
        self._active()
        (runtimes.runtimes_dir() / "previous.json").write_text("{}")
        runtimes.rollback(out=io.StringIO())
        self.assertFalse((runtimes.runtimes_dir() / "active.json").exists())

    def test_review_rollback_notice_never_promises_the_declined_update(self):
        old = self._active()
        self._stage()
        runtimes.promote_staged()
        runtimes.rollback(out=io.StringIO())
        lines = runtimes.runtime_notice_lines(active=old, newest=self.newest)
        self.assertFalse(any("will prepare" in line or "automatically" in line for line in lines))
        self.assertTrue(any("declined" in line for line in lines))

    def test_review_lock_descriptor_is_closed_before_unlink(self):
        handle = runtimes._acquire_update_lock()
        self.assertIsNotNone(handle)
        unlink = Path.unlink
        def windows_unlink(path, *args, **kwargs):
            if path.name == "update.lock":
                with self.assertRaises(OSError, msg="Windows cannot unlink an open update.lock"):
                    os.fstat(handle)
            return unlink(path, *args, **kwargs)
        with patch.object(Path, "unlink", new=windows_unlink):
            runtimes._release_update_lock(handle)
        self.assertFalse((runtimes.runtimes_dir() / "update.lock").exists())

    def test_review_real_daemon_records_attempt_and_stages_with_only_fake_dependencies(self):
        def observing_runner(command, timeout=None):
            if command[:3] == [sys.executable, "-m", "venv"]:
                state = self._read("last_update.json")
                self.assertEqual("installing", state["state"])
                self.assertEqual(self.newest, state["to"])
            return self._fake_runner(command, timeout=timeout)
        fake_codex = MagicMock(return_value=([
            {"id": entry["model"]} for entry in server.DEFAULT_CATALOG if entry.get("provider") == "codex"], None))
        fake_claude = MagicMock(return_value=([
            {"value": entry["model"]} for entry in server.DEFAULT_CATALOG if entry.get("provider") == "claude"], None))
        fake_runner = MagicMock(side_effect=observing_runner)
        # Only replace the fetch, runner and probe dependencies. The real
        # daemon invokes the real auto_update function on both compared commits.
        with patch.object(runtimes, "_fetch_pypi", side_effect=lambda name: FAKE_PYPI[name]) as fetch, patch.dict(
            runtimes.auto_update.__kwdefaults__, {
                "runner": fake_runner, "codex_probe": fake_codex, "claude_probe": fake_claude,
            }
        ):
            service = self._service()
            try:
                self.assertIsNotNone(service._update_check_thread)
                self.assertTrue(service._update_check_thread.daemon)
                service._update_check_thread.join(5)
                self.assertFalse(service._update_check_thread.is_alive())
                self.assertEqual("staged", self._read("last_update.json")["state"])
                self.assertEqual("0.161.0", self._read("staged.json")["codex"]["package_version"])
                self.assertTrue(any("ready and applies" in line for line in service._runtime_notice))
            finally:
                service.close()
        self.assertEqual(2, fetch.call_count)
        fake_runner.assert_called()
        fake_codex.assert_called_once()
        fake_claude.assert_called_once()
        self.assertIsNone(runtimes.read_active_runtime())


if __name__ == "__main__":
    unittest.main()
