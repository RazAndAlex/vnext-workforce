from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import tomllib
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import suite_environment  # noqa: F401  # the suite settings; unittest never reads conftest.py
from vnext import vnext_mcp_reload
from vnext.vnext_mcp_reload import (
    _cache_paths,
    _install_proxy,
    check_args,
    main,
    plan_command,
)


class VNextMcpReloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.plugin_root = Path(self._temp.name)
        reload_root = self.plugin_root / "reload"
        reload_root.mkdir()
        self.lock = reload_root / "package-lock.json"
        self.lock.write_bytes(b"first lock")

    def test_proxy_command_wraps_the_server_and_preserves_arguments(self) -> None:
        _, _, entry = _cache_paths(self.plugin_root)
        argv, reason = plan_command(
            self.plugin_root,
            ["--workspace", "a workspace", "--catalog", "models.json"],
            which=lambda name: "/tools/node" if name == "node" else None,
            env={},
            exists=lambda path: path == entry,
        )

        self.assertIsNone(reason)
        self.assertEqual(["/tools/node", str(entry)], argv[:2])
        separator = argv.index("--")
        self.assertEqual(
            [
                sys.executable,
                "-m",
                "vnext.vnext_mcp_server",
                "--stdio",
                "--workspace",
                "a workspace",
                "--catalog",
                "models.json",
            ],
            argv[separator + 1 :],
        )

    def test_missing_node_falls_back_to_the_plain_server(self) -> None:
        argv, reason = plan_command(
            self.plugin_root,
            ["--workspace", "here"],
            which=lambda _name: None,
            env={},
        )

        self.assertEqual(
            [
                sys.executable,
                "-m",
                "vnext.vnext_mcp_server",
                "--stdio",
                "--workspace",
                "here",
            ],
            argv,
        )
        self.assertIn("node", reason or "")

    def test_environment_switch_forces_the_plain_server(self) -> None:
        argv, reason = plan_command(
            self.plugin_root,
            [],
            which=lambda _name: "/tools/node",
            env={"VNEXT_RELOAD": "0"},
            exists=lambda _path: True,
        )

        self.assertEqual([sys.executable, *('-m', 'vnext.vnext_mcp_server', '--stdio')], argv)
        self.assertIn("VNEXT_RELOAD=0", reason or "")

    def test_cache_key_changes_with_the_lock_bytes(self) -> None:
        first_target = _cache_paths(self.plugin_root)[1]
        self.lock.write_bytes(b"second lock")
        second_target = _cache_paths(self.plugin_root)[1]

        self.assertNotEqual(first_target.name, second_target.name)

    def test_npm_output_is_sent_to_stderr(self) -> None:
        reload_root = self.plugin_root / "reload"
        (reload_root / "package.json").write_text("{}")
        target = self.plugin_root / "cache" / "key"
        entry = target / "node_modules" / "reloaderoo" / "reloaderoo.js"

        def install_entry(*_args: object, **kwargs: object) -> None:
            temporary = kwargs["cwd"]
            self.assertIsInstance(temporary, Path)
            temporary_entry = temporary / entry.relative_to(target)
            temporary_entry.parent.mkdir(parents=True)
            temporary_entry.touch()

        with patch(
            "vnext.vnext_mcp_reload.subprocess.run",
            side_effect=install_entry,
        ) as run:
            _install_proxy(self.plugin_root, target, entry, "/tools/npm")

        self.assertIs(sys.stderr, run.call_args.kwargs["stdout"])
        self.assertIs(sys.stderr, run.call_args.kwargs["stderr"])
        self.assertEqual(180, run.call_args.kwargs["timeout"])

    def test_plugin_configuration_uses_the_reload_launcher(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        configuration = json.loads((repository_root / "plugins" / "vnext" / ".mcp.json").read_text())
        server = configuration["mcpServers"]["vnext"]

        self.assertEqual("vnext-mcp", server["command"])
        self.assertEqual("--plugin-root", server["args"][0])
        self.assertIn("${CLAUDE_PLUGIN_ROOT}", server["args"])

    def test_plugin_manifest_uses_installed_cross_platform_entrypoint(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        manifest = json.loads((repository_root / "plugins" / "vnext" / ".mcp.json").read_text())
        package = tomllib.loads((repository_root / "pyproject.toml").read_text())
        server = manifest["mcpServers"]["vnext"]

        self.assertEqual("vnext-mcp", server["command"])
        self.assertEqual(
            "vnext.vnext_mcp_reload:main",
            package["project"]["scripts"][server["command"]],
        )
        self.assertEqual("--plugin-root", server["args"][0])


class PluginReadmeTests(unittest.TestCase):
    """The README is the install contract, and round 1 found it untrue twice.

    Finding 2: a GUI-launched Claude Code reads PATH at startup out of the
    login shell, so "open a new terminal" fixed nothing for the case it was
    written for. Finding 7: the `.gitignore` sentence promised something a
    workspace with its own `.vnext/.gitignore` never got.
    """

    def setUp(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        self.readme = (repository_root / "plugins" / "vnext" / "README.md").read_text(
            encoding="utf-8"
        )

    def test_the_install_step_names_the_launcher_folder_on_both_platforms(self) -> None:
        self.assertIn("~/.local/bin", self.readme)
        self.assertIn(r"%USERPROFILE%\.local\bin", self.readme)

    def test_the_restart_advice_is_quitting_the_application(self) -> None:
        self.assertNotIn("open a new terminal", self.readme)
        self.assertIn("quit the application", self.readme)

    def test_the_gitignore_sentence_covers_a_file_the_user_already_has(self) -> None:
        self.assertIn("left exactly as you wrote it", self.readme)
        self.assertIn("add the line", self.readme)


class StartupCheckLauncherTests(unittest.TestCase):
    """Round-3 finding 4: a failed startup showed as CONNECTION_CLOSED with no cause."""

    def test_the_check_keeps_the_workspace_and_drops_the_launcher_arguments(self) -> None:
        self.assertEqual(
            ["--workspace", "/a/project"],
            check_args(
                ["--plugin-root", "/plugins/vnext", "--", "--workspace", "/a/project", "--check"]
            ),
        )
        self.assertEqual(
            ["--workspace", "/a/project"],
            check_args(["--check", "--plugin-root=/plugins/vnext", "--workspace", "/a/project"]),
        )

    def test_the_check_keeps_the_provider_registrations(self) -> None:
        """The check asks whether the server would start with these providers.

        A catalog card for a registered provider is only backed when the same
        --provider reaches the check, so the launcher hands it through.
        """

        self.assertEqual(
            ["--workspace", "/a/project", "--provider", "acme=pkg.mod:Factory"],
            check_args([
                "--plugin-root", "/plugins/vnext", "--",
                "--workspace", "/a/project",
                "--provider", "acme=pkg.mod:Factory",
                "--check",
            ]),
        )

    def test_the_check_runs_the_startup_resolution_and_starts_nothing(self) -> None:
        with patch(
            "vnext.vnext_mcp_server.run_startup_check", return_value=0
        ) as check, patch(
            "vnext.vnext_mcp_reload._prepared_command",
            side_effect=AssertionError("the check must not plan the proxy command"),
        ), patch(
            "os.execvp", side_effect=AssertionError("the check must not exec the server")
        ), patch(
            "os.execv", side_effect=AssertionError("the check must not exec the server")
        ):
            self.assertEqual(0, main(["--check", "--workspace", "/a/project"]))

        self.assertEqual(["--workspace", "/a/project"], check.call_args.args[0])

    def test_check_reports_proxy_state_without_installing(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            plugin_root = Path(root)
            (plugin_root / "reload").mkdir()
            (plugin_root / "reload" / "package-lock.json").write_bytes(b"lock")
            output = io.StringIO()
            with patch("vnext.vnext_mcp_server.run_startup_check", return_value=0), patch(
                "vnext.vnext_mcp_reload.shutil.which",
                side_effect=lambda name: {"node": "/tools/node", "npm": "/tools/npm"}.get(name),
            ), patch("vnext.vnext_mcp_reload._install_proxy", side_effect=AssertionError("must not install")), contextlib.redirect_stdout(output):
                self.assertEqual(0, main(["--check", "--plugin-root", str(plugin_root)]))
            self.assertIn("restart proxy: not installed yet", output.getvalue())

    def test_check_says_why_the_last_proxy_install_failed(self) -> None:
        """R19: npm failed at start and --check still said "not installed yet"."""

        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as home, patch.dict(
            os.environ, {"HOME": home, "USERPROFILE": home}
        ):
            plugin_root = Path(root)
            (plugin_root / "reload").mkdir()
            (plugin_root / "reload" / "package-lock.json").write_bytes(b"lock")
            tools = {"node": "/tools/node", "npm": "/tools/npm"}

            def check() -> str:
                output = io.StringIO()
                with patch("vnext.vnext_mcp_server.run_startup_check", return_value=0), patch(
                    "vnext.vnext_mcp_reload.shutil.which", side_effect=tools.get
                ), patch("vnext.vnext_mcp_reload._install_proxy", side_effect=AssertionError("must not install")), contextlib.redirect_stdout(output):
                    self.assertEqual(0, main(["--check", "--plugin-root", str(plugin_root)]))
                return output.getvalue()

            with patch("vnext.vnext_mcp_reload.shutil.which", side_effect=tools.get), patch(
                "vnext.vnext_mcp_reload._install_proxy",
                side_effect=subprocess.CalledProcessError(1, ["npm", "ci"]),
            ):
                _, reason = vnext_mcp_reload._prepared_command(plugin_root, [])
            self.assertIn("install failed", reason)
            said = check()
            self.assertIn("returned non-zero exit status 1", said)
            self.assertNotIn("not installed yet", said)

            with patch("vnext.vnext_mcp_reload.shutil.which", side_effect=tools.get), patch(
                "vnext.vnext_mcp_reload._install_proxy"
            ):
                vnext_mcp_reload._prepared_command(plugin_root, [])
            self.assertIn("not installed yet", check())

    def test_check_uses_launch_plan_for_ready_and_missing_node(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            plugin_root = Path(root)
            (plugin_root / "reload").mkdir()
            (plugin_root / "reload" / "package-lock.json").write_bytes(b"lock")
            _, _, entry = _cache_paths(plugin_root)
            for available, expected in ((False, "node was not found"), (True, "restart proxy: ready")):
                with self.subTest(available=available), patch(
                    "vnext.vnext_mcp_server.run_startup_check", return_value=0
                ), patch("vnext.vnext_mcp_reload.shutil.which", side_effect=lambda name: "/tools/node" if available and name == "node" else None), patch(
                    "vnext.vnext_mcp_reload.Path.exists", side_effect=lambda path: path == entry
                ), patch("vnext.vnext_mcp_reload._install_proxy", side_effect=AssertionError("must not install")):
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        self.assertEqual(0, main(["--check", "--plugin-root", str(plugin_root)]))
                    self.assertIn(expected, output.getvalue())

    def test_check_explains_missing_npm_without_installing(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            plugin_root = Path(root)
            (plugin_root / "reload").mkdir()
            (plugin_root / "reload" / "package-lock.json").write_bytes(b"lock")
            output = io.StringIO()
            with patch("vnext.vnext_mcp_server.run_startup_check", return_value=0), patch(
                "vnext.vnext_mcp_reload.shutil.which",
                side_effect=lambda name: "/tools/node" if name == "node" else None,
            ), patch("vnext.vnext_mcp_reload._install_proxy", side_effect=AssertionError("must not install")), contextlib.redirect_stdout(output):
                self.assertEqual(0, main(["--check", "--plugin-root", str(plugin_root)]))
            self.assertIn("npm was not found, so the restart proxy could not be installed", output.getvalue())

    def test_a_failed_check_prints_no_proxy_line(self) -> None:
        output = io.StringIO()
        with patch("vnext.vnext_mcp_server.run_startup_check", return_value=1), patch(
            "vnext.vnext_mcp_reload._check_proxy_line", side_effect=AssertionError("must not run")
        ), contextlib.redirect_stdout(output):
            self.assertEqual(1, main(["--check"]))
        self.assertEqual("", output.getvalue())

    def test_the_help_text_names_the_check_flag(self) -> None:
        """--help is where a stuck user looks for the flag the README names.

        `--check` is answered before argparse runs, so argparse never learned
        the flag existed and `vnext-mcp --help` listed two options with no
        `--check` among them.  A user who hits CONNECTION_CLOSED, reaches for
        --help and finds nothing reads the README as out of date.
        """

        text = io.StringIO()
        with contextlib.redirect_stdout(text):
            with self.assertRaises(SystemExit) as exit_code:
                main(["--help"])

        self.assertEqual(0, exit_code.exception.code)
        self.assertIn("--check", text.getvalue())

    def test_the_declared_check_flag_still_answers_before_argparse(self) -> None:
        """Declaring it must not move the answer behind --plugin-root."""

        with patch("vnext.vnext_mcp_server.run_startup_check", return_value=0) as check:
            self.assertEqual(0, main(["--check", "--workspace", "/a/project"]))

        self.assertEqual(["--workspace", "/a/project"], check.call_args.args[0])

    def test_the_check_needs_no_plugin_root(self) -> None:
        with patch("vnext.vnext_mcp_server.run_startup_check", return_value=1) as check:
            self.assertEqual(1, main(["--check"]))

        self.assertEqual([], check.call_args.args[0])


class _PlatformOs:
    """``os`` as the launcher module sees it, reporting another platform's name.

    Patching ``os.name`` itself reaches pathlib as well, and before Python 3.13
    ``Path()`` then builds a WindowsPath on a POSIX host and raises
    NotImplementedError, so argparse's ``type=Path`` failed on 3.11 and 3.12.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def __getattr__(self, attribute: str):
        return getattr(os, attribute)


def _launcher_sees(name: str):
    return patch.object(vnext_mcp_reload, "os", _PlatformOs(name))


class OwnedProxyEofTests(unittest.TestCase):
    def test_early_eof_preserves_failed_proxy_status_and_stderr(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            proxy = Path(temporary) / "fake_proxy.py"
            proxy.write_text(
                "import sys\nimport time\ntime.sleep(0.05)\n"
                "print('fake server cannot start', file=sys.stderr, flush=True)\n"
                "raise SystemExit(1)\n",
                encoding="utf-8",
            )
            code = (
                "from vnext.vnext_mcp_reload import _run_owned_proxy; "
                f"raise SystemExit(_run_owned_proxy({[sys.executable, str(proxy)]!r}))"
            )
            result = subprocess.run(
                [sys.executable, "-c", code],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=12,
                check=False,
            )

        self.assertEqual(1, result.returncode)
        self.assertIn("fake server cannot start", result.stderr)

    def test_early_eof_stops_a_live_proxy_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            proxy = Path(temporary) / "fake_proxy.py"
            proxy.write_text(
                "import signal\nimport time\n"
                "signal.signal(signal.SIGTERM, lambda *_: exit(0))\n"
                "while True: time.sleep(0.05)\n",
                encoding="utf-8",
            )
            code = (
                "import vnext.vnext_mcp_reload as reload; "
                "reload.PROXY_CLOSE_GRACE_SECONDS = 0.2; "
                f"raise SystemExit(reload._run_owned_proxy({[sys.executable, str(proxy)]!r}))"
            )
            result = subprocess.run(
                [sys.executable, "-c", code],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )

        self.assertEqual(0, result.returncode)

    @unittest.skipIf(os.name == "nt", "the stop request is a POSIX group signal")
    def test_a_client_that_spoke_and_quit_stops_the_proxy_at_once(self) -> None:
        # Only a client that never sent a byte waits for the child to fail on
        # its own; a real session ending must not idle through the grace.
        with tempfile.TemporaryDirectory() as temporary:
            proxy = Path(temporary) / "fake_proxy.py"
            proxy.write_text(
                "import signal\nimport sys\nimport time\n"
                "signal.signal(signal.SIGTERM, lambda *_: exit(0))\n"
                "sys.stdin.read()\n"
                "while True: time.sleep(0.05)\n",
                encoding="utf-8",
            )
            code = (
                "from vnext.vnext_mcp_reload import _run_owned_proxy; "
                f"raise SystemExit(_run_owned_proxy({[sys.executable, str(proxy)]!r}))"
            )
            started = time.monotonic()
            result = subprocess.run(
                [sys.executable, "-c", code],
                input='{"jsonrpc": "2.0"}\n',
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            elapsed = time.monotonic() - started

        self.assertEqual(0, result.returncode)
        self.assertLess(elapsed, 4.0)

    @unittest.skipIf(os.name == "nt", "the stop request is a POSIX group signal")
    def test_a_proxy_that_had_to_be_ended_is_not_reported_as_a_clean_stop(self) -> None:
        # The tree is ended when the grace runs out, and the server's close may
        # not have run; exit 0 would read as an orderly stop.
        # The client's EOF is held back until the fake proxy has installed its
        # handler: on a loaded machine the stop request could otherwise reach
        # a proxy that had not started ignoring it yet, which then really did
        # stop, and the test read that as the launcher's mistake.
        with tempfile.TemporaryDirectory() as temporary:
            ready = Path(temporary) / "ready"
            proxy = Path(temporary) / "fake_proxy.py"
            proxy.write_text(
                "import pathlib\nimport signal\nimport sys\nimport time\n"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                f"pathlib.Path({str(ready)!r}).write_text('ready')\n"
                "sys.stdin.read()\n"
                "while True: time.sleep(0.05)\n",
                encoding="utf-8",
            )
            code = (
                "import vnext.vnext_mcp_reload as reload; "
                "reload.PROXY_CLOSE_GRACE_SECONDS = 0.3; "
                f"raise SystemExit(reload._run_owned_proxy({[sys.executable, str(proxy)]!r}))"
            )
            launcher = subprocess.Popen(
                [sys.executable, "-c", code],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                assert launcher.stdin is not None
                launcher.stdin.write('{"jsonrpc": "2.0"}\n')
                launcher.stdin.flush()
                deadline = time.monotonic() + 20
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(ready.exists(), "the fake proxy never started")
                _stdout, stderr = launcher.communicate(timeout=20)
            finally:
                if launcher.poll() is None:
                    launcher.kill()
                    launcher.communicate()

        self.assertEqual(1, launcher.returncode)
        self.assertIn("did not stop", stderr)

    def test_on_windows_the_stop_request_is_a_ctrl_break_to_the_proxy_group(self) -> None:
        # Windows has no group signal, so a closed stdin alone left the proxy
        # running until the job was ended, before the server could close.
        owned = MagicMock()
        with _launcher_sees("nt"), patch.object(
            vnext_mcp_reload.signal, "CTRL_BREAK_EVENT", 1, create=True
        ):
            vnext_mcp_reload._ask_proxy_to_stop(owned)

        owned.process.send_signal.assert_called_once_with(1)


@unittest.skipIf(os.name == "nt", "this simulates Windows for a run that is not on it")
class TheWindowsLauncherTakesTheServerWithItTests(unittest.TestCase):
    """Round-5 finding 6: on Windows the server outlived the launcher.

    POSIX replaces the launcher with the server, so the client holds the server
    itself and closing stdin stops it.  Windows has no such call, so the
    launcher stays alive as the parent and the server is a process the client
    has no handle on.  ``subprocess.call`` left it reading a stdin pipe whose
    handle it owns a copy of, so killing the launcher stopped nothing: the
    server and every worker under it kept running.  The server now starts inside
    the same Windows job object the workers use, which carries
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE and whose only handle the launcher holds,
    so the server dies with the launcher however the launcher dies.  Real
    Windows is release Gate 1; here the platform is simulated.
    """

    def setUp(self) -> None:
        self.command = ["C:\\python\\python.exe", "-m", "vnext.vnext_mcp_server", "--stdio"]
        patcher = patch.object(
            vnext_mcp_reload,
            "_prepared_command",
            lambda plugin_root, server_args: (self.command, "direct server"),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _owned(self, exit_code: int = 0) -> MagicMock:
        owned = MagicMock()
        owned.process.wait.return_value = exit_code
        return owned

    def test_the_server_starts_inside_a_job_the_launcher_owns(self) -> None:
        owned = self._owned(exit_code=3)
        with _launcher_sees("nt"), patch.object(
            vnext_mcp_reload.subprocess,
            "call",
            side_effect=AssertionError("the Windows launcher must own the server's job"),
        ), patch(
            "vnext.process_supervisor.OwnedProcess.start", return_value=owned
        ) as start:
            exit_code = main(["--plugin-root", ".", "--", "--workspace", "."])

        # No stdin, stdout or stderr keyword: the server inherits the
        # launcher's own three, so the client closing stdin still arrives as
        # end of file and the server's output goes where it always went.
        self.assertEqual([call(self.command)], start.call_args_list)
        self.assertEqual(3, exit_code, "the server's exit code is the launcher's")
        self.assertEqual(1, owned.close.call_count, "the job handle has to be released")

    def test_the_job_is_closed_even_when_the_wait_is_interrupted(self) -> None:
        owned = self._owned()
        owned.process.wait.side_effect = KeyboardInterrupt
        with _launcher_sees("nt"), patch(
            "vnext.process_supervisor.OwnedProcess.start", return_value=owned
        ):
            with self.assertRaises(KeyboardInterrupt):
                main(["--plugin-root", ".", "--", "--workspace", "."])

        self.assertEqual(1, owned.close.call_count)

    def test_a_job_that_cannot_be_created_stops_the_launch(self) -> None:
        from vnext.process_supervisor import ProcessSupervisionError

        with _launcher_sees("nt"), patch(
            "vnext.process_supervisor.OwnedProcess.start",
            side_effect=ProcessSupervisionError("Windows process jobs are unavailable"),
        ):
            exit_code = main(["--plugin-root", ".", "--", "--workspace", "."])

        self.assertEqual(1, exit_code, "a server nobody can kill must not start")

    def test_the_posix_branch_still_replaces_the_launcher(self) -> None:
        reached: list[str] = []
        self.command = ["/usr/bin/python3", "-m", "vnext.vnext_mcp_server", "--stdio"]
        with patch.object(vnext_mcp_reload.os, "name", "posix"), patch.object(
            vnext_mcp_reload.os, "execv", lambda *args: reached.append("execv")
        ), patch.object(
            vnext_mcp_reload.os, "execvp", lambda *args: reached.append("execvp")
        ), patch(
            "vnext.process_supervisor.OwnedProcess.start",
            side_effect=AssertionError("POSIX execs instead of owning a job"),
        ):
            main(["--plugin-root", ".", "--", "--workspace", "."])

        # Both names appear because a fake exec returns where the real one
        # replaces the process.  The point is that an exec is reached at all and
        # no job is created.
        self.assertEqual(["execv", "execvp"], reached)


if __name__ == "__main__":
    unittest.main()


class LauncherStartFailureTests(unittest.TestCase):
    """R20 claude-code F2: an ``OSError`` from ``Popen`` ended in a traceback.

    Both owned starts caught ``ProcessSupervisionError`` alone, so a program
    the system would not run -- a ``node.cmd`` shim on Windows, an interpreter
    a Python upgrade removed -- crashed the launcher.  The proxy is optional,
    so its failure hands over to the direct server; the direct server failing
    is one line and exit 1.
    """

    def _not_a_program(self, folder: str) -> str:
        path = Path(folder) / "node"
        path.write_text("not a program\n", encoding="utf-8")
        path.chmod(0o644)
        return str(path)

    def test_a_proxy_that_cannot_run_hands_over_to_the_direct_server(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            proxy = self._not_a_program(temporary)
            direct = [sys.executable, "-c", "raise SystemExit(7)"]
            code = (
                "from vnext.vnext_mcp_reload import _run_owned_proxy; "
                f"raise SystemExit(_run_owned_proxy({[proxy]!r}, fallback={direct!r}))"
            )
            result = subprocess.run(
                [sys.executable, "-c", code],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )

        self.assertEqual(7, result.returncode, result.stderr)
        self.assertIn("restart proxy not started", result.stderr)
        self.assertIn("starting the server directly", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_a_server_that_cannot_run_is_one_line_and_exit_one(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            server = self._not_a_program(temporary)
            for start in (vnext_mcp_reload._run_owned_on_windows, vnext_mcp_reload._run_direct):
                with self.subTest(start=start.__name__):
                    errors = io.StringIO()
                    with contextlib.redirect_stderr(errors):
                        self.assertEqual(1, start([server]))
                    self.assertIn("vNext server not started", errors.getvalue())
