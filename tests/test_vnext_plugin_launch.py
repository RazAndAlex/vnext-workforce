"""The plugin's own launcher, ``plugins/vnext/launch.py``.

Release 0.1 named ``vnext-mcp`` in ``.mcp.json``.  That command exists only
after ``uv tool install``, so on a machine where it never ran Claude Code could
not start the server at all and said ``Executable not found in $PATH: "stdio"``.
The plugin now starts ``launch.py`` through uv, and the launcher looks for vNext
in a fixed order: the clone's ``.venv``, PATH, the clone through uv, and
otherwise a message that names the fix.
"""

from __future__ import annotations

import importlib.util
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import suite_environment  # noqa: F401  # the suite settings; unittest never reads conftest.py

REPO = Path(__file__).resolve().parents[1]
PLUGIN = REPO / "plugins" / "vnext"

_spec = importlib.util.spec_from_file_location("vnext_plugin_launch", PLUGIN / "launch.py")
launch = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(launch)


VNEXT_SCRIPTS = '[project.scripts]\nvnext-mcp = "vnext.vnext_mcp_reload:main"\n'


def _clone(folder: Path, name: str = "vnext", scripts: str = VNEXT_SCRIPTS) -> Path:
    """A folder shaped like a vNext clone, with the plugin two levels down."""

    (folder / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["setuptools"]\n\n'
        f'[project]\nname = "{name}"\nversion = "0.1.0"\n\n{scripts}',
        encoding="utf-8",
    )
    root = folder / "plugins" / "vnext"
    root.mkdir(parents=True)
    return root


ARGS = ["--plugin-root", "<root>", "--", "--workspace", "/a project"]


class LaunchOrderTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.root = _clone(self.folder)
        self.args = [str(self.root) if a == "<root>" else a for a in ARGS]
        self.venv = self.folder / ".venv" / "bin" / "vnext-mcp"

    def plan(self, *, exists=lambda _path: False, which=lambda _name: None, env=None):
        with patch.object(launch, "_is_windows", return_value=False):
            return launch.plan(self.args, env=env or {}, which=which, exists=exists)

    def test_a_the_clone_venv_comes_first(self) -> None:
        command, note = self.plan(
            exists=lambda path: path == self.venv,
            which=lambda name: "/elsewhere/" + name,
        )

        self.assertEqual([str(self.venv), *self.args], command)
        self.assertEqual("clone .venv", note)

    def test_b_path_comes_when_the_clone_has_no_venv(self) -> None:
        command, note = self.plan(which=lambda name: "/tools/" + name)

        self.assertEqual(["/tools/vnext-mcp", *self.args], command)
        self.assertEqual("PATH", note)

    def test_c_the_clone_through_uv_comes_third(self) -> None:
        command, note = self.plan(which=lambda name: "/tools/uv" if name == "uv" else None)

        self.assertEqual(
            ["/tools/uv", "run", "--project", str(self.folder), "--extra", "claude",
             "--quiet", "vnext-mcp", *self.args],
            command,
        )
        self.assertEqual("clone via uv", note)

    def test_c_prefers_the_uv_that_started_the_launcher(self) -> None:
        command, _ = self.plan(env={"UV": "/uv/that/ran/me"}, which=lambda name: "/tools/" + name if name == "uv" else None)

        self.assertEqual("/uv/that/ran/me", command[0])

    def test_d_no_clone_and_no_path_names_the_readme(self) -> None:
        other = Path(self.folder) / "copy"
        root = other / "plugins" / "vnext"
        root.mkdir(parents=True)
        command, message = launch.plan(
            ["--plugin-root", str(root)], env={}, which=lambda _name: None, exists=lambda _path: False
        )

        self.assertIsNone(command)
        self.assertIn("looked for vnext-mcp on PATH", message)
        self.assertIn(str(root), message)
        self.assertIn("plugins/vnext/README.md", message)
        self.assertIn('uv tool install ".[claude]"', message)

    def test_d_a_known_clone_names_the_install_command_for_it(self) -> None:
        with patch.object(launch, "_is_windows", return_value=False):
            message = launch.fix_message(self.root, self.folder)

        self.assertIn(str(self.venv), message)
        self.assertIn(f'uv tool install "{self.folder}[claude]"', message)

    def _other_clone(self, **shape) -> Path:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return _clone(Path(temporary.name), **shape)

    def test_a_clone_under_another_project_name_is_still_a_clone(self) -> None:
        root = self._other_clone(
            name="vnext-workforce",
            scripts='[project.scripts]\nvnext-mcp = "vnext.vnext_mcp_reload:main"\n',
        )

        self.assertEqual(root.parent.parent, launch.find_clone(root))

    def test_a_project_without_the_vnext_mcp_script_is_no_clone(self) -> None:
        root = self._other_clone(scripts='[project.scripts]\nsomething = "pkg.cli:main"\n')

        self.assertIsNone(launch.find_clone(root))

    def test_vnext_mcp_outside_project_scripts_does_not_count(self) -> None:
        root = self._other_clone(scripts=(
            '[project.scripts]\nother = "pkg.cli:main"\n\n'
            '[tool.notes]\nvnext-mcp = "mentioned elsewhere"\n'
            '# vnext-mcp = "vnext.vnext_mcp_reload:main"\n'
        ))

        self.assertIsNone(launch.find_clone(root))

    def test_main_prints_the_message_and_exits_1(self) -> None:
        with patch.object(launch, "plan", return_value=(None, "the fix")), \
                patch.object(sys, "stderr", new_callable=lambda: __import__("io").StringIO()) as err:
            self.assertEqual(1, launch.main([]))
        self.assertEqual("the fix\n", err.getvalue())

    def test_posix_replaces_the_process_with_every_argument(self) -> None:
        with patch.object(launch, "_is_windows", return_value=False), \
                patch.object(launch, "plan", return_value=(["/x/vnext-mcp", *self.args], "PATH")), \
                patch.object(launch.os, "execv") as execv:
            launch.main(self.args)

        execv.assert_called_once_with("/x/vnext-mcp", ["/x/vnext-mcp", *self.args])


class PluginRootTests(unittest.TestCase):
    def test_the_first_plugin_root_argument_wins(self) -> None:
        self.assertEqual(
            Path("/one"),
            launch.plugin_root(["--plugin-root", "/one", "--plugin-root", "/two"], {"CLAUDE_PLUGIN_ROOT": "/env"}),
        )

    def test_the_environment_comes_next(self) -> None:
        self.assertEqual(Path("/env"), launch.plugin_root(["--", "--workspace", "w"], {"CLAUDE_PLUGIN_ROOT": "/env"}))

    def test_the_script_folder_comes_last(self) -> None:
        self.assertEqual(PLUGIN.resolve(), launch.plugin_root([], {}))


class WindowsTests(unittest.TestCase):
    def test_windows_looks_for_the_exe_under_scripts(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        folder = Path(temporary.name)
        root = _clone(folder)
        exe = folder / ".venv" / "Scripts" / "vnext-mcp.exe"
        with patch.object(launch, "_is_windows", return_value=True):
            command, note = launch.plan(
                ["--plugin-root", str(root)], env={}, which=lambda _name: None, exists=lambda path: path == exe
            )

        self.assertEqual([str(exe), "--plugin-root", str(root)], command)
        self.assertEqual("clone .venv", note)

    def test_windows_runs_a_child_and_returns_its_exit_code(self) -> None:
        with patch.object(launch, "_is_windows", return_value=True), \
                patch.object(launch, "plan", return_value=(["C:/x/vnext-mcp.exe", "--a"], "PATH")), \
                patch.object(launch, "_run_on_windows", return_value=7) as run, \
                patch.object(launch.os, "execv") as execv:
            self.assertEqual(7, launch.main(["--a"]))

        run.assert_called_once_with(["C:/x/vnext-mcp.exe", "--a"])
        execv.assert_not_called()


def expanded_mcp_command(plugin_root: Path, project_dir: Path) -> list[str]:
    """``.mcp.json``'s command, with the variables Claude Code substitutes."""

    config = json.loads((plugin_root / ".mcp.json").read_text(encoding="utf-8"))
    server = config["mcpServers"]["vnext"]
    values = {"CLAUDE_PLUGIN_ROOT": str(plugin_root), "CLAUDE_PROJECT_DIR": str(project_dir)}

    def expand(text: str) -> str:
        for name, value in values.items():
            text = text.replace("${" + name + "}", value)
        return text

    return [expand(server["command"]), *(expand(argument) for argument in server["args"])]


def list_tools(command: list[str], *, env: dict, cwd: Path, timeout: float = 120.0) -> tuple[list[dict], float, str]:
    """Run an MCP stdio server, initialize it, and return tools/list, seconds, stderr."""

    started = time.monotonic()
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env, cwd=str(cwd),
    )
    lines: "queue.Queue[bytes]" = queue.Queue()
    errors: list[bytes] = []
    threading.Thread(target=lambda: [lines.put(line) for line in process.stdout], daemon=True).start()
    threading.Thread(target=lambda: errors.extend(process.stderr), daemon=True).start()

    def send(message: dict) -> None:
        process.stdin.write((json.dumps(message) + "\n").encode())
        process.stdin.flush()

    def answer(request_id: int) -> dict:
        deadline = started + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("no answer to request %d; stderr: %s" % (request_id, b"".join(errors).decode(errors="replace")))
            try:
                line = lines.get(timeout=remaining)
            except queue.Empty:
                continue
            reply = json.loads(line)
            if reply.get("id") == request_id:
                return reply

    try:
        send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "launch-test", "version": "0"}}})
        answer(1)
        seconds = time.monotonic() - started
        send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        tools = answer(2)["result"]["tools"]
    finally:
        try:
            process.stdin.close()
        except OSError:
            pass
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    return tools, seconds, b"".join(errors).decode(errors="replace")


@unittest.skipUnless(shutil.which("uv"), "uv is required to start the plugin")
# The server refuses to start without a provider login, and CI has none: the
# proxy would never answer initialize and the test would wait out its 120 s.
@unittest.skipIf(os.environ.get("CI"), "starting the server needs a provider login")
class PluginStartsThroughUvTests(unittest.TestCase):
    def test_the_mcp_json_command_lists_tools_without_vnext_mcp_on_path(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        workspace = Path(temporary.name)
        uv_folder = str(Path(shutil.which("uv")).parent)
        system = ["C:/Windows/System32", "C:/Windows"] if os.name == "nt" else ["/usr/bin", "/bin"]
        env = {key: value for key, value in os.environ.items() if key.startswith("VNEXT_") or key in {
            "HOME", "USERPROFILE", "SYSTEMROOT", "APPDATA", "LOCALAPPDATA", "TEMP", "TMP",
            "CODEX_HOME", "UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR"}}
        env["PATH"] = os.pathsep.join([uv_folder, *system])
        env["CLAUDE_PLUGIN_ROOT"] = str(PLUGIN)
        env["CLAUDE_PROJECT_DIR"] = str(workspace)
        # The suite's home folder holds no provider login, and the server will
        # not start without one.  It reads the Codex login by key name alone and
        # listing tools calls no provider, so a placeholder value is enough.
        codex_home = workspace / "codex-home"
        codex_home.mkdir()
        (codex_home / "auth.json").write_text(json.dumps({"OPENAI_API_KEY": "placeholder"}), encoding="utf-8")
        env["CODEX_HOME"] = str(codex_home)
        self.assertFalse(
            any((Path(folder) / ("vnext-mcp.exe" if os.name == "nt" else "vnext-mcp")).exists()
                for folder in env["PATH"].split(os.pathsep)),
            "this test needs a PATH without vnext-mcp",
        )

        command = expanded_mcp_command(PLUGIN, workspace)
        # The same route with --check answers in about a second.  Without a
        # provider login the server refuses to start and the proxy never
        # answers initialize, so the start below would wait out its 120 s.
        check = subprocess.run(
            [*command, "--check"], env=env, cwd=str(workspace),
            capture_output=True, text=True, timeout=60,
        )
        if check.returncode != 0 and "needs at least one available worker model" in check.stderr:
            self.skipTest(f"starting the server needs a provider login: {check.stderr.strip()}")
        self.assertEqual(0, check.returncode, check.stderr)

        tools, _, stderr = list_tools(command, env=env, cwd=workspace)

        self.assertTrue(tools, stderr)


if __name__ == "__main__":
    unittest.main()
