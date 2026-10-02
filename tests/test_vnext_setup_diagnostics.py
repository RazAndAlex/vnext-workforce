"""What a first-time setup is told when the start refuses to happen.

A blind first-time-setup review of the public export found that every mistake a
person can make in the provider file they hand-write reads as "no key is
configured", that an install without the Claude SDK still offers the models that
need it, and that a refused start prints a traceback where ``--check`` prints one
line.  These are the cases from that review, C1 to C7 included.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vnext import vnext_mcp_server as server
from vnext import vnext_provider_config as provider_config
from vnext.vnext_mcp_reload import (
    _cache_paths,
    _parse_args,
    _server_child,
    plan_command,
)
from vnext.vnext_mcp_server import VNextMcpServiceError


SECRET = "the-real-provider-key-nobody-should-print"


class ProviderConfigFaultTests(unittest.TestCase):
    """The six mistakes the reviewer made, and the file that is correct."""

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.home = Path(temporary.name)
        self.path = self.home / ".vnext" / "providers.json"
        self.path.parent.mkdir()

    def _write(self, text: str) -> None:
        self.path.write_text(text, encoding="utf-8")

    def _problem(self, provider: str = "zai") -> str | None:
        return provider_config.describe_provider_config_problem(provider, self.path)

    def test_c1_a_trailing_comma_names_the_line_and_the_fault(self) -> None:
        self._write(
            "{\n"
            '  "providers": {\n'
            f'    "zai": {{"api_key": "{SECRET}"}},\n'
            "  }\n"
            "}\n"
        )
        problem = self._problem()

        self.assertIsNotNone(problem)
        assert problem is not None
        self.assertIn("could not be read", problem)
        self.assertIn("line 3", problem)
        self.assertIn("trailing comma", problem)
        self.assertNotIn(SECRET, problem)

    def test_c2_a_truncated_file_says_it_could_not_be_read(self) -> None:
        self._write('{\n  "providers": {\n    "zai": {"api_key": "' + SECRET)
        problem = self._problem()

        assert problem is not None
        self.assertIn("could not be read", problem)
        self.assertNotIn(SECRET, problem)

    def test_c3_the_wrong_key_name_is_named_with_what_was_found(self) -> None:
        self._write(json.dumps({"providers": {"zai": {"apiKey": SECRET}}}))
        problem = self._problem()

        assert problem is not None
        self.assertIn("providers.zai has no 'api_key' string", problem)
        self.assertIn("found: apiKey", problem)
        self.assertNotIn(SECRET, problem)

    def test_c4_a_missing_providers_wrapper_is_said_outright(self) -> None:
        self._write(json.dumps({"zai": {"api_key": SECRET}}))

        self.assertIn("has no 'providers' object", self._problem() or "")

    def test_c5_a_provider_name_nobody_knows_is_named_with_the_known_ones(self) -> None:
        self._write(json.dumps({"providers": {"z.ai": {"api_key": SECRET}}}))
        problem = self._problem()

        assert problem is not None
        self.assertIn("'z.ai' is not a known provider", problem)
        self.assertIn("zai", problem)
        self.assertIn("commandcode", problem)

    def test_c6_an_empty_key_is_called_empty(self) -> None:
        self._write(json.dumps({"providers": {"zai": {"api_key": ""}}}))

        self.assertIn("providers.zai.api_key is empty", self._problem() or "")

    def test_c7_a_correct_file_has_nothing_wrong_with_it(self) -> None:
        self._write(json.dumps({"providers": {"zai": {"api_key": SECRET}}}))

        self.assertIsNone(self._problem())
        self.assertEqual(SECRET, provider_config._read_api_key("zai", self.path))

    def test_a_file_that_is_not_there_is_not_a_fault(self) -> None:
        self.path.unlink(missing_ok=True)

        self.assertIsNone(self._problem())

    def test_every_fault_leaves_the_key_out_of_the_message(self) -> None:
        cases = [
            '{"providers": {"zai": {"api_key": "' + SECRET + '"},}}',
            json.dumps({"providers": {"zai": {"apiKey": SECRET}}}),
            json.dumps({"zai": {"api_key": SECRET}}),
            json.dumps({"providers": {"z.ai": {"api_key": SECRET}}}),
            json.dumps({"providers": {"zai": {"api_key": f" {SECRET}"}}}),
        ]
        for text in cases:
            with self.subTest(text=text[:40]):
                self._write(text)
                self.assertNotIn(SECRET, self._problem() or "")

    def test_the_path_is_shown_the_way_the_readme_writes_it(self) -> None:
        inside_home = Path.home() / ".vnext" / "providers.json"

        self.assertEqual("~/.vnext/providers.json", provider_config._display(inside_home))

    def test_every_distinct_fault_is_collected_once(self) -> None:
        self._write(json.dumps({"providers": {"z.ai": {"api_key": SECRET}}}))
        with patch.object(provider_config, "PROVIDER_CONFIG_PATH", self.path):
            faults = provider_config.provider_config_problems()

        self.assertEqual(1, len(faults), faults)
        self.assertIn("'z.ai' is not a known provider", faults[0])


class StartupNoteTests(unittest.TestCase):
    """``--check`` is where a person is sent, so it carries the file's fault."""

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "providers.json"
        self.path.write_text(
            '{"providers": {"zai": {"api_key": "' + SECRET + '"},}}', encoding="utf-8"
        )

    def _notes(self) -> list[str]:
        with patch.object(provider_config, "PROVIDER_CONFIG_PATH", self.path), patch.object(
            server, "_codex_login_problem", return_value="no codex login"
        ), patch.object(
            server, "_claude_login_state", return_value=server.CLAUDE_LOGIN_ABSENT
        ), patch.object(
            server, "_claude_sdk_installed", return_value=True
        ):
            return server.startup_check_notes([])

    def test_the_zai_note_names_the_file_and_the_fault(self) -> None:
        joined = "\n".join(self._notes())

        self.assertIn("zai models left out:", joined)
        self.assertIn("could not be read", joined)
        self.assertNotIn("no Z.ai provider key is configured", joined)
        self.assertNotIn(SECRET, joined)

    def test_the_commandcode_note_names_the_same_file(self) -> None:
        joined = "\n".join(self._notes())

        self.assertIn("commandcode models left out:", joined)
        self.assertIn("could not be read", joined)

    def test_a_missing_file_keeps_the_wording_it_had(self) -> None:
        self.path.unlink()
        joined = "\n".join(self._notes())

        self.assertIn("zai models left out: no Z.ai provider key is configured", joined)
        self.assertIn(
            "commandcode models left out: no Command Code account key is configured",
            joined,
        )

    def test_the_refusal_to_start_names_the_file_and_the_fault(self) -> None:
        with patch.object(provider_config, "PROVIDER_CONFIG_PATH", self.path), patch.object(
            server, "_codex_login_available", return_value=False
        ), patch.object(
            server, "_claude_login_state", return_value=server.CLAUDE_LOGIN_ABSENT
        ), patch.object(
            server, "_claude_sdk_installed", return_value=True
        ), patch(
            "vnext.vnext_mcp_server.load_zai_provider", return_value=None
        ), patch(
            "vnext.vnext_mcp_server.load_commandcode_provider", return_value=None
        ):
            with self.assertRaises(VNextMcpServiceError) as raised:
                server._validate_catalog(server.DEFAULT_CATALOG)

        message = str(raised.exception)
        self.assertIn("a workforce needs at least one available worker model", message)
        self.assertIn("could not be read", message)
        self.assertNotIn(SECRET, message)


class CheckLocationLabelTests(unittest.TestCase):
    def test_check_names_the_current_folder_without_calling_it_the_plugin_workspace(self) -> None:
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as workspace, patch.object(
            server, "resolve_startup"
        ) as resolve, patch.object(server, "startup_check_notes", return_value=[]):
            resolve.return_value.workspace = Path(workspace)
            resolve.return_value.entries = [{"provider": "zai", "model": "glm-5.2"}]
            resolve.return_value.login = object()
            self.assertEqual(0, server.run_startup_check(["--workspace", workspace], out=output))
        self.assertIn("checked from:", output.getvalue().splitlines()[0])
        self.assertNotIn("workspace:", output.getvalue().splitlines()[0])


class ClaudeSdkAvailabilityTests(unittest.TestCase):
    """A model whose harness is missing is not a model this session can run."""

    def test_the_providers_that_need_the_sdk_come_from_the_harness_table(self) -> None:
        self.assertEqual({"claude", "zai"}, set(server.CLAUDE_SDK_PROVIDERS))

    def test_the_probe_asks_for_the_module_without_importing_it(self) -> None:
        with patch("importlib.util.find_spec", return_value=None) as find_spec:
            self.assertFalse(server._claude_sdk_installed())
        find_spec.assert_called_once_with("claude_agent_sdk")

    def test_a_missing_sdk_removes_every_model_that_runs_on_it(self) -> None:
        with patch("importlib.util.find_spec", return_value=None):
            with patch.object(server, "_codex_login_available", return_value=True), patch.object(
                server, "_claude_login_state", return_value=server.CLAUDE_LOGIN_AVAILABLE
            ), patch(
                "vnext.vnext_mcp_server.load_zai_provider", return_value=object()
            ), patch(
                "vnext.vnext_mcp_server.load_commandcode_provider",
                return_value=object(),
            ):
                entries = server._validate_catalog(server.DEFAULT_CATALOG)

        providers = {entry["provider"] for entry in entries}
        self.assertNotIn("claude", providers)
        self.assertNotIn("zai", providers)
        self.assertIn("codex", providers)
        self.assertIn("commandcode", providers)

    def test_the_note_says_the_sdk_is_missing_and_how_to_get_it(self) -> None:
        with patch("importlib.util.find_spec", return_value=None), patch.object(
            server, "_codex_login_problem", return_value=None
        ):
            notes = server.startup_check_notes([{"provider": "codex", "model": "gpt-6-sol"}])

        joined = "\n".join(notes)
        self.assertIn("claude models left out: the Claude SDK is not installed", joined)
        self.assertIn("zai models left out: the Claude SDK is not installed", joined)
        self.assertIn('uv tool install ".[claude]"', joined)

    def test_a_registered_adapter_keeps_its_own_cards(self) -> None:
        with patch("importlib.util.find_spec", return_value=None), patch.object(
            server, "_codex_login_available", return_value=True
        ), patch(
            "vnext.vnext_mcp_server.load_zai_provider", return_value=object()
        ):
            entries = server._validate_catalog(
                [{"provider": "zai", "model": "glm-5.3"}], {"zai": lambda: None}
            )

        self.assertEqual([{"provider": "zai", "model": "glm-5.3"}], entries)

    def test_a_start_with_no_sdk_and_no_other_login_says_why(self) -> None:
        with patch("importlib.util.find_spec", return_value=None), patch.object(
            server, "_codex_login_available", return_value=False
        ), patch(
            "vnext.vnext_mcp_server.load_commandcode_provider", return_value=None
        ):
            with self.assertRaises(VNextMcpServiceError) as raised:
                server._validate_catalog(server.DEFAULT_CATALOG)

        self.assertIn("the Claude SDK is not installed", str(raised.exception))


class CatalogFlagTests(unittest.TestCase):
    """``--catalog`` takes a path, and says so when the path is not there."""

    def test_a_path_that_is_not_there_names_the_flag(self) -> None:
        with self.assertRaises(VNextMcpServiceError) as raised:
            server._load_catalog("opus")

        self.assertEqual("--catalog: no such file: opus", str(raised.exception))

    def test_the_check_prints_that_one_line_and_exits_one(self) -> None:
        errors = io.StringIO()
        with tempfile.TemporaryDirectory() as workspace:
            code = server.run_startup_check(
                ["--workspace", workspace, "--catalog", "opus"], err=errors
            )

        self.assertEqual(1, code)
        self.assertIn("--catalog: no such file: opus", errors.getvalue())
        self.assertNotIn("Errno", errors.getvalue())

    def test_the_launcher_help_names_the_flag_and_forwards_it(self) -> None:
        with tempfile.TemporaryDirectory() as plugin_root:
            root, server_args = _parse_args(
                ["--plugin-root", plugin_root, "--catalog", "models.json"]
            )

        self.assertEqual(["--catalog", "models.json"], server_args)
        self.assertEqual(Path(plugin_root), root)

    def test_a_flag_after_the_separator_is_forwarded_once(self) -> None:
        with tempfile.TemporaryDirectory() as plugin_root:
            _, server_args = _parse_args(
                ["--plugin-root", plugin_root, "--", "--catalog", "models.json"]
            )

        self.assertEqual(["--catalog", "models.json"], server_args)


class RefusedStartTests(unittest.TestCase):
    """A refusal prints the line ``--check`` prints, and no traceback."""

    def test_a_refused_start_is_one_line_and_exit_one(self) -> None:
        errors = io.StringIO()
        with patch.object(
            server,
            "resolve_startup",
            side_effect=VNextMcpServiceError(
                "a workforce needs at least one available worker model"
            ),
        ), patch.object(sys, "stderr", errors):
            code = server.main(["--workspace", ".", "--stdio"])

        self.assertEqual(1, code)
        printed = errors.getvalue().strip().splitlines()
        self.assertEqual(1, len(printed), printed)
        self.assertEqual(
            "vnext cannot start: a workforce needs at least one available worker model",
            printed[0],
        )

    def test_an_unexpected_exception_keeps_its_traceback(self) -> None:
        with patch.object(server, "resolve_startup", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                server.main(["--workspace", ".", "--stdio"])


class ProxyChildNameTests(unittest.TestCase):
    """The proxy names the server after the filename of what it launches."""

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.plugin_root = Path(temporary.name)
        reload_root = self.plugin_root / "reload"
        reload_root.mkdir()
        (reload_root / "package-lock.json").write_bytes(b"lock")

    def test_the_installed_console_script_is_preferred(self) -> None:
        script = Path(sys.executable).with_name("vnext-mcp-server")

        self.assertEqual(
            [str(script), "--stdio"], _server_child(lambda path: path == script)
        )

    def test_windows_looks_for_the_executable_beside_the_interpreter(self) -> None:
        script = Path(sys.executable).with_name("vnext-mcp-server.exe")
        with patch("vnext.vnext_mcp_reload._is_windows", return_value=True):
            self.assertEqual(
                [str(script), "--stdio"], _server_child(lambda path: path == script)
            )

    def test_a_checkout_with_no_script_keeps_the_interpreter(self) -> None:
        self.assertEqual(
            [sys.executable, "-m", "vnext.vnext_mcp_server", "--stdio"],
            _server_child(lambda _path: False),
        )

    def test_the_proxy_launches_the_named_script_with_the_server_arguments(self) -> None:
        script = Path(sys.executable).with_name("vnext-mcp-server")
        _, _, entry = _cache_paths(self.plugin_root)
        argv, reason = plan_command(
            self.plugin_root,
            ["--workspace", "a workspace"],
            which=lambda name: "/tools/node" if name == "node" else None,
            env={},
            exists=lambda path: path in {entry, script},
        )

        self.assertIsNone(reason)
        separator = argv.index("--")
        self.assertEqual(
            [str(script), "--stdio", "--workspace", "a workspace"],
            argv[separator + 1 :],
        )

    def test_the_direct_path_is_unchanged_by_the_name(self) -> None:
        script = Path(sys.executable).with_name("vnext-mcp-server")
        argv, reason = plan_command(
            self.plugin_root,
            ["--workspace", "here"],
            which=lambda _name: None,
            env={},
            exists=lambda path: path == script,
        )

        self.assertEqual(
            [sys.executable, "-m", "vnext.vnext_mcp_server", "--stdio", "--workspace", "here"],
            argv,
        )
        self.assertIn("node", reason or "")

    def test_the_package_ships_the_server_console_script(self) -> None:
        import tomllib

        root = Path(__file__).resolve().parents[1]
        project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

        self.assertEqual(
            "vnext.vnext_mcp_server:main",
            project["project"]["scripts"]["vnext-mcp-server"],
        )


if __name__ == "__main__":
    unittest.main()
