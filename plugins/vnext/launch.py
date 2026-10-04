"""Find the vNext launcher and hand it this plugin's arguments.

Claude Code starts this file through ``uv run --no-project``, so the plugin
needs only ``uv`` on PATH.  Release 0.1 named ``vnext-mcp`` in ``.mcp.json``
directly, and on a machine where ``uv tool install`` had never run Claude Code
could not find it and said only ``Executable not found in $PATH: "stdio"``.

The search, in order:

1. the ``vnext-mcp`` in the ``.venv`` of the clone that holds this plugin;
2. ``vnext-mcp`` on PATH, which ``uv tool install`` puts there;
3. the clone itself, through ``uv run --project <clone> --extra claude``;
4. otherwise one message on stderr naming the fix, and exit code 1.

Standard library only, and Python 3.8 or newer: uv may run this file with any
interpreter it finds, and the macOS system one is 3.9.
"""

import os
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Mapping, Optional, Sequence, Tuple

LAUNCHER = "vnext-mcp"
README_URL = "plugins/vnext/README.md"


def _is_windows() -> bool:
    """The platform decision in one place a test can stand in for."""

    return os.name == "nt"


def plugin_root(argv: Sequence[str], env: Mapping[str, str]) -> Path:
    """The first ``--plugin-root``, else CLAUDE_PLUGIN_ROOT, else this folder."""

    for index, argument in enumerate(argv):
        if argument == "--":
            break
        if argument == "--plugin-root" and index + 1 < len(argv):
            return Path(argv[index + 1])
        if argument.startswith("--plugin-root="):
            return Path(argument.partition("=")[2])
    if env.get("CLAUDE_PLUGIN_ROOT"):
        return Path(env["CLAUDE_PLUGIN_ROOT"])
    return Path(__file__).resolve().parent


def _declares_launcher(pyproject: Path) -> bool:
    """Whether ``[project.scripts]`` names ``vnext-mcp``, read without tomllib.

    The clone is known by what it provides.  Its project name differs between
    the private repository and the public export, which renames it.
    """

    try:
        text = pyproject.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    section = None
    for line in text.splitlines():
        stripped = line.strip()
        header = re.match(r"^\[([^\]]+)\]", stripped)
        if header:
            section = header.group(1).strip()
            continue
        if section == "project.scripts":
            match = re.match(r"""^["']?([A-Za-z0-9_.-]+)["']?\s*=""", stripped)
            if match and match.group(1) == LAUNCHER:
                return True
    return False


def find_clone(root: Path) -> Optional[Path]:
    """The vNext clone two folders above the plugin, when there is one."""

    clone = root.parent.parent
    if _declares_launcher(clone / "pyproject.toml"):
        return clone
    return None


def _venv_launcher(clone: Path) -> Path:
    if _is_windows():
        return clone / ".venv" / "Scripts" / (LAUNCHER + ".exe")
    return clone / ".venv" / "bin" / LAUNCHER


def fix_message(root: Path, clone: Optional[Path]) -> str:
    if clone is not None:
        return (
            "vNext could not start. It looked for {launcher} in {venv}, then on PATH, "
            "then tried uv on the clone at {clone}, and none of them worked. "
            'Run this once, then restart Claude Code: uv tool install "{clone}[claude]"'
        ).format(launcher=LAUNCHER, venv=_venv_launcher(clone), clone=clone)
    return (
        "vNext could not start. It looked for {launcher} on PATH and found no vNext "
        "clone around the plugin at {root}. Install vNext with the steps in the "
        "Install section of {readme} (uv tool install \".[claude]\" from a clone), "
        "then restart Claude Code."
    ).format(launcher=LAUNCHER, root=root, readme=README_URL)


def plan(
    argv: Sequence[str],
    *,
    env: Mapping[str, str] = os.environ,
    which: Callable[[str], Optional[str]] = shutil.which,
    exists: Callable[[Path], bool] = Path.exists,
) -> Tuple[Optional[List[str]], str]:
    """Return the command to run, or None and the message that explains why not."""

    args = list(argv)
    root = plugin_root(args, env)
    clone = find_clone(root)

    if clone is not None:
        venv = _venv_launcher(clone)
        if exists(venv):
            return [str(venv), *args], "clone .venv"

    on_path = which(LAUNCHER)
    if on_path:
        return [on_path, *args], "PATH"

    if clone is not None:
        uv = env.get("UV") or which("uv")
        if uv:
            return (
                [uv, "run", "--project", str(clone), "--extra", "claude", "--quiet", LAUNCHER, *args],
                "clone via uv",
            )

    return None, fix_message(root, clone)


def _run_on_windows(command: List[str]) -> int:
    """Run the launcher inside a job this process owns, and return its exit code.

    Windows has no exec that keeps this process, so this file stays as the
    parent.  The child starts suspended, joins a job object that carries
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, then resumes; this process holds the
    only handle, so the child and everything under it end when this process
    ends, however it ends.  The same pattern as ``OwnedProcess`` in
    ``vnext.process_supervisor``, which this file cannot import.
    """

    import ctypes
    from ctypes import wintypes

    class BASIC(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class IO(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
        )]

    class EXTENDED(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BASIC),
            ("IoInfo", IO),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    ntdll = ctypes.WinDLL("ntdll")
    kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    ntdll.NtResumeProcess.argtypes = (wintypes.HANDLE,)
    ntdll.NtResumeProcess.restype = ctypes.c_long

    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        print("vNext not started: " + str(ctypes.WinError(ctypes.get_last_error())), file=sys.stderr)
        return 1
    limits = EXTENDED()
    limits.BasicLimitInformation.LimitFlags = 0x00002000  # KILL_ON_JOB_CLOSE
    if not kernel32.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
        print("vNext not started: " + str(ctypes.WinError(ctypes.get_last_error())), file=sys.stderr)
        kernel32.CloseHandle(job)
        return 1
    try:
        process = subprocess.Popen(command, creationflags=0x00000004)  # CREATE_SUSPENDED
    except OSError as exc:
        print("vNext not started: {}".format(exc), file=sys.stderr)
        kernel32.CloseHandle(job)
        return 1
    handle = wintypes.HANDLE(int(process._handle))  # type: ignore[attr-defined]
    if not kernel32.AssignProcessToJobObject(job, handle) or ntdll.NtResumeProcess(handle) != 0:
        # Fail closed: a server nobody can stop is worse than no server.
        print("vNext not started: the launcher could not be placed in an owned job", file=sys.stderr)
        process.kill()
        process.wait()
        kernel32.CloseHandle(job)
        return 1
    # Ctrl-C reaches every process on the console, the child included; this
    # process waits for the child to finish its own shutdown and reports its code.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        return process.wait()
    finally:
        kernel32.CloseHandle(job)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    command, note = plan(args)
    if command is None:
        print(note, file=sys.stderr)
        return 1
    if _is_windows():
        return _run_on_windows(command)
    os.execv(command[0], command)
    return 1  # not reached


if __name__ == "__main__":
    sys.exit(main())
