"""Launch the vNext stdio server through a restartable proxy when available."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path


_SERVER_COMMAND = ("-m", "vnext.vnext_mcp_server", "--stdio")

# The console script that runs the server and nothing else.  reloaderoo names
# the server it fronts after the filename of the command it launches, so the
# name a client reads comes from this.
_SERVER_SCRIPT = "vnext-mcp-server"


def _is_windows() -> bool:
    """The platform decision in one place a test can stand in for.

    ``os.name`` itself cannot stay patched while this runs: pathlib dispatches
    ``Path`` on it and refuses to build a ``WindowsPath`` on a POSIX host, so a
    test that patched it could not construct the path it was checking.
    """

    return os.name == "nt"


def _server_child(exists: Callable[[Path], bool]) -> list[str]:
    """The command the proxy launches, named after the product where it can be.

    Through the proxy an MCP client reads the server's name from reloaderoo,
    which builds it from the filename of this command: "python -m
    vnext.vnext_mcp_server" made every client, log and bug report call
    this server "python-dev".  The installed console script sits beside the
    interpreter that runs us, so where it is there the name reads
    "vnext-mcp-server-dev" instead.  A clone run straight from a checkout has no
    such script, and keeps the interpreter command.
    """

    suffix = ".exe" if _is_windows() else ""
    script = Path(sys.executable).with_name(f"{_SERVER_SCRIPT}{suffix}")
    if exists(script):
        return [str(script), "--stdio"]
    return [sys.executable, *_SERVER_COMMAND]


def _cache_paths(plugin_root: Path) -> tuple[Path, Path, Path]:
    lock = plugin_root / "reload" / "package-lock.json"
    key = hashlib.sha256(lock.read_bytes()).hexdigest()[:12]
    target = Path.home() / ".cache" / "vnext" / "reload" / key
    entry = target / "node_modules" / "reloaderoo" / "dist" / "bin" / "reloaderoo.js"
    return lock, target, entry


def plan_command(
    plugin_root: str | os.PathLike[str],
    server_args: Sequence[str],
    *,
    which: Callable[[str], str | None] = shutil.which,
    env: Mapping[str, str] = os.environ,
    exists: Callable[[Path], bool] = Path.exists,
) -> tuple[list[str], str | None]:
    """Return the command to run without installing or starting anything."""

    fallback = [sys.executable, *_SERVER_COMMAND, *server_args]
    if env.get("VNEXT_RELOAD") == "0":
        return fallback, "VNEXT_RELOAD=0 disables the restart proxy"

    node = which("node")
    if node is None:
        return fallback, "node was not found"

    try:
        _, _, entry = _cache_paths(Path(plugin_root))
    except (OSError, RuntimeError) as exc:
        return fallback, f"the restart proxy lock could not be read: {exc}"
    if not exists(entry):
        return fallback, "the restart proxy is not installed"

    return (
        [
            node,
            str(entry),
            "proxy",
            "--log-level",
            "warning",
            "--",
            *_server_child(exists),
            *server_args,
        ],
        None,
    )


def _install_proxy(plugin_root: Path, target: Path, entry: Path, npm: str) -> None:
    source = plugin_root / "reload"
    temporary = target.parent / f"{target.name}.tmp-{os.getpid()}"
    target.parent.mkdir(parents=True, exist_ok=True)
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    try:
        shutil.copy2(source / "package.json", temporary / "package.json")
        shutil.copy2(source / "package-lock.json", temporary / "package-lock.json")
        subprocess.run(
            [npm, "ci", "--no-fund", "--no-audit", "--omit=dev"],
            cwd=temporary,
            stdout=sys.stderr,
            stderr=sys.stderr,
            timeout=180,
            check=True,
        )
        temporary_entry = temporary / entry.relative_to(target)
        if not temporary_entry.exists():
            raise FileNotFoundError(f"npm did not install the restart proxy entry at {temporary_entry}")
        try:
            os.replace(temporary, target)
        except OSError:
            if not entry.exists():
                raise
            shutil.rmtree(temporary)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _prepared_command(
    plugin_root: Path,
    server_args: Sequence[str],
) -> tuple[list[str], str | None]:
    argv, reason = plan_command(plugin_root, server_args)
    if reason != "the restart proxy is not installed":
        return argv, reason

    npm = shutil.which("npm")
    if npm is None:
        return argv, "npm was not found, so the restart proxy could not be installed"
    target: Path | None = None
    try:
        _, target, entry = _cache_paths(plugin_root)
        _install_proxy(plugin_root, target, entry, npm)
    except (OSError, subprocess.SubprocessError) as exc:
        # npm's own words went to stderr, which a client hides.  The note
        # lets --check say why the last start ran without the proxy.
        if target is not None:
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                _install_failed_note(target).write_text(str(exc), encoding="utf-8")
            except OSError:
                pass
        return argv, f"the restart proxy install failed: {exc}"
    try:
        _install_failed_note(target).unlink(missing_ok=True)
    except OSError:
        pass
    return plan_command(plugin_root, server_args)


def _install_failed_note(target: Path) -> Path:
    return target.with_name(f"{target.name}.install-failed")


def _parse_args(argv: Sequence[str] | None = None) -> tuple[Path, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin-root", type=Path, help="plugin directory (required to start the server; --check needs none)")
    # Declared for the help text alone.  main() answers --check before it
    # reaches this parser, because a person diagnosing a server that will not
    # start has no plugin root to give, so the flag was missing from --help --
    # which is where the README sends them to recall its name.
    parser.add_argument(
        CHECK_FLAG,
        action="store_true",
        help="say whether the server will start and which worker models it "
             "offers, then exit; this starts nothing and needs no --plugin-root",
    )
    # Declared so --help names it, and forwarded again below so declaring it
    # changes nothing about the launch.  The README sends a person here to
    # recall the flag, and the flag was not in the help at all.
    parser.add_argument(
        "--catalog",
        default=None,
        metavar="<path to a catalog JSON file>",
        help="read the worker model catalog from this JSON file instead of the "
             "built-in one; the same flag --check takes",
    )
    parser.add_argument("server_args", nargs=argparse.REMAINDER)
    parsed = parser.parse_args(argv)
    server_args = list(parsed.server_args)
    if server_args[:1] == ["--"]:
        server_args.pop(0)
    if parsed.catalog is not None and "--catalog" not in server_args:
        server_args += ["--catalog", parsed.catalog]
    if parsed.plugin_root is None:
        parser.error("--plugin-root is required to start the server")
    return parsed.plugin_root, server_args


CHECK_FLAG = "--check"


def check_args(argv: Sequence[str]) -> list[str]:
    """The server arguments left once the launcher's own ones are removed.

    ``--check`` is answered before ``--plugin-root`` is required, because a
    person diagnosing a server that will not start types the command by hand
    and has no plugin root to give.  ``--workspace`` survives, so the check
    reads the same workspace a normal launch would.
    """

    remaining: list[str] = []
    skip = False
    for argument in argv:
        if skip:
            skip = False
            continue
        if argument == CHECK_FLAG or argument == "--":
            continue
        if argument == "--plugin-root":
            skip = True
            continue
        if argument.startswith("--plugin-root="):
            continue
        remaining.append(argument)
    return remaining


def _check_proxy_line(raw: Sequence[str]) -> str:
    """Explain the launcher's proxy decision without installing or starting it."""

    plugin_root: Path | None = None
    for index, argument in enumerate(raw):
        if argument == "--plugin-root" and index + 1 < len(raw):
            plugin_root = Path(raw[index + 1])
            break
        if argument.startswith("--plugin-root="):
            plugin_root = Path(argument.partition("=")[2])
            break
    if plugin_root is None:
        candidate = Path.cwd() / "plugins" / "vnext"
        if candidate.is_dir():
            plugin_root = candidate
    if plugin_root is None:
        # Node and the opt-out are independent of the plugin's lock file.
        if os.environ.get("VNEXT_RELOAD") == "0":
            reason = "VNEXT_RELOAD=0 disables the restart proxy"
        elif shutil.which("node") is None:
            reason = "node was not found"
        else:
            reason = "plugin root is unknown; run --check from the clone or pass --plugin-root"
        return f"restart proxy: unavailable: {reason}; the server starts without it (13 tools)"

    _, reason = plan_command(plugin_root, [], which=shutil.which, env=os.environ, exists=Path.exists)
    if reason is None:
        return "restart proxy: ready (restart_server offered, 14 tools)"
    if reason == "the restart proxy is not installed":
        if shutil.which("npm") is None:
            reason = "npm was not found, so the restart proxy could not be installed"
        else:
            try:
                failed = _install_failed_note(_cache_paths(plugin_root)[1]).read_text(encoding="utf-8").strip()
            except OSError:
                failed = ""
            if not failed:
                return "restart proxy: not installed yet; the first start installs it with npm and needs the network"
            reason = (
                f"the install on the last start failed ({failed}); "
                "the next start tries again with npm and needs the network"
            )
    return f"restart proxy: unavailable: {reason}; the server starts without it (13 tools)"


def _run_owned_on_windows(command: Sequence[str]) -> int:
    """Run the server inside a job the launcher owns, and hand back its exit code.

    Windows has no ``execv`` that replaces the launcher with the server, so the
    launcher stays alive as the parent and the client holds no handle on the
    server.  A plain ``subprocess.call`` therefore left an orphan: the server
    kept its own copy of the stdin pipe handle, so killing the launcher produced
    no end of file and the server, with every worker under it, carried on.

    ``OwnedProcess`` is the same Windows job object the workers already use.  It
    carries JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE and the launcher holds the only
    handle, so the server dies with the launcher however the launcher dies,
    including on a crash.  The server inherits the launcher's own stdin, stdout
    and stderr, so a client that closes stdin still reaches it as end of file,
    and the exit code returned here is the server's.
    """

    from .process_supervisor import OwnedProcess, ProcessSupervisionError

    try:
        owned = OwnedProcess.start(list(command))
    except (ProcessSupervisionError, OSError) as exc:
        # Fail closed, for the reason the job exists: a server nobody can kill
        # is worse than a server that did not start.  OSError is Popen's own
        # refusal, such as an interpreter a Python upgrade removed.
        print(f"vNext server not started: {exc}", file=sys.stderr)
        return 1
    try:
        return owned.process.wait()
    finally:
        owned.close()


# How long the proxy and the server behind it get to stop before the launcher
# ends the tree.  The server's own close may spend CLOSE_GRACE_SECONDS (5 s, in
# vnext_mcp_server) cancelling its workers and writing the run record; a kill
# inside that window cuts the record short.  Two seconds of margin on top.
PROXY_CLOSE_GRACE_SECONDS = 7.0


def _ask_proxy_to_stop(owned: object) -> None:
    """Send SIGTERM to the proxy's process group: the server unwinds on it.

    The pinned proxy does not end when its stdin closes, so the server behind
    it never sees end of file; SIGTERM is the request both of them honour.
    Windows has no group signal, so the request there is a CTRL_BREAK to the
    proxy's process group, which the server unwinds on as it does on SIGTERM.
    The job ends the tree if the grace runs out.
    """

    if os.name == "nt":
        try:
            owned.process.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
        except (OSError, ValueError):
            pass
        return
    try:
        os.killpg(owned.process.pid, signal.SIGTERM)  # type: ignore[attr-defined]
    except OSError:
        pass


def _wait_for_proxy(owned: object, seconds: float) -> None:
    try:
        owned.process.wait(timeout=seconds)  # type: ignore[attr-defined]
    except subprocess.TimeoutExpired:
        pass


def _run_owned_proxy(command: Sequence[str], *, fallback: Sequence[str] | None = None) -> int:
    """Forward stdio to the proxy and close its whole tree on client EOF.

    The direct server reads client stdin itself. The proxy does not reliably
    stop when stdin closes, so the launcher must remain its owner.  The proxy
    is optional: when it cannot be started, ``fallback`` starts the server
    directly.  On Windows ``shutil.which("node")`` can find a ``node.cmd``
    shim, which ``Popen`` refuses with an OSError.
    """
    from .process_supervisor import OwnedProcess, ProcessSupervisionError

    try:
        owned = OwnedProcess.start(list(command), stdin=subprocess.PIPE)
    except (ProcessSupervisionError, OSError) as exc:
        if fallback is None:
            print(f"vNext restart proxy not started: {exc}", file=sys.stderr)
            return 1
        print(
            f"vNext restart proxy not started: {exc}; starting the server directly.",
            file=sys.stderr,
        )
        return _run_direct(fallback)

    eof = threading.Event()
    spoke = threading.Event()
    stop_requested = False

    def forward_stdin() -> None:
        assert owned.process.stdin is not None
        try:
            while True:
                data = os.read(sys.stdin.fileno(), 65536)
                if not data:
                    break
                spoke.set()
                owned.process.stdin.write(data)
                owned.process.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        finally:
            try:
                owned.process.stdin.close()
            except OSError:
                pass
            eof.set()

    threading.Thread(target=forward_stdin, name="vnext-proxy-stdin", daemon=True).start()
    old_handlers: dict[signal.Signals, object] = {}

    def forward_signal(number: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True
        # The proxy lives in its own process group/job, so a signal aimed at
        # only this launcher needs an explicit route to the proxy.
        try:
            if os.name == "nt":
                if number == signal.SIGINT:
                    owned.process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    owned.process.terminate()
            else:
                os.killpg(owned.process.pid, number)
        except (OSError, ProcessLookupError):
            pass
        if number == signal.SIGINT:
            raise KeyboardInterrupt
        raise SystemExit(128 + number)

    try:
        for number in (signal.SIGTERM, signal.SIGINT):
            old_handlers[number] = signal.signal(number, forward_signal)
        while owned.process.poll() is None and not eof.is_set():
            time.sleep(0.05)
        if eof.is_set():
            if not spoke.is_set():
                # A client that closed stdin before sending a byte never had a
                # session, so a server failing to start is the likely story:
                # let it finish and keep its status and its printed reason.
                _wait_for_proxy(owned, PROXY_CLOSE_GRACE_SECONDS)
                status = owned.process.poll()
                if status is not None:
                    return status
            # The client is gone.  Ask the proxy and the server to stop, give
            # the server its whole close, then end whatever is left.
            _ask_proxy_to_stop(owned)
            _wait_for_proxy(owned, PROXY_CLOSE_GRACE_SECONDS)
            stopped = owned.process.poll() is not None
            owned.close(grace_seconds=0.0)
            if stopped:
                return 0
            # Ended from outside, so the server's close may not have run and
            # its run record can read "live"; a quiet 0 would hide that.
            print(
                f"vNext restart proxy did not stop within {PROXY_CLOSE_GRACE_SECONDS:g} s "
                "of being asked; its process tree was ended",
                file=sys.stderr,
            )
            return 1
        return owned.process.poll() or 0
    finally:
        for number, handler in old_handlers.items():
            signal.signal(number, handler)
        if stop_requested:
            # The signal was already forwarded; the same grace applies.
            _wait_for_proxy(owned, PROXY_CLOSE_GRACE_SECONDS)
        owned.close(grace_seconds=0.0)


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if "--update-runtimes" in raw or "--rollback" in raw:
        # Installs a side runtime or rolls back to the pinned one, then exits.
        # Nothing restarts: the server reads the choice when it next starts.
        from vnext.vnext_runtimes import main as update_main

        return update_main(raw)
    if CHECK_FLAG in raw:
        # Imported here: the check never starts the protocol or the Node proxy,
        # and a normal launch still pays no import cost for this branch.
        from vnext.vnext_mcp_server import run_startup_check

        status = run_startup_check(check_args(raw))
        # A failed check stays one line on stderr; the proxy state belongs to a
        # check that passed.
        if status == 0:
            print(_check_proxy_line(raw))
        return status

    plugin_root, server_args = _parse_args(argv)
    command, reason = _prepared_command(plugin_root, server_args)
    if reason is not None:
        print(f"vNext restart proxy unavailable: {reason}; starting the server directly.", file=sys.stderr)

    if reason is None:
        return _run_owned_proxy(command, fallback=[sys.executable, *_SERVER_COMMAND, *server_args])
    return _run_direct(command)


def _run_direct(command: Sequence[str]) -> int:
    """Start the server with no proxy: owned on Windows, exec'd elsewhere."""

    if os.name == "nt":
        return _run_owned_on_windows(command)
    try:
        if os.path.dirname(command[0]):
            os.execv(command[0], list(command))
        os.execvp(command[0], list(command))
    except OSError as exc:
        print(f"vNext server not started: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
