"""A service-owned PTY. Presentation clients can leave without killing it.

The small sidecar is placed in an OwnedProcess tree before it creates the
PTY, so forced shutdown also contains the native harness and its descendants.
Only terminal bytes travel here; provider adapters own semantic agent events.
"""
from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .process_supervisor import OwnedProcess, ProcessCleanup


class NativeTerminalProcess:
    def __init__(self, command: Sequence[str], *, cwd: str,
                 environment: Mapping[str, str], on_event: Callable[[Mapping[str, Any]], None],
                 columns: int = 120, rows: int = 35) -> None:
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._failure: str | None = None
        self._on_event = on_event
        self._closed = False
        # The sidecar runs from the terminal's workspace, so ``-m`` could
        # resolve an editable package from another checkout. This prelude
        # fixes the imported source without passing a Python path through to
        # the terminal child environment.
        package_root = str(Path(__file__).resolve().parent.parent)
        sidecar_entrypoint = (
            "import runpy,sys;sys.path.insert(0," + repr(package_root) + ");"
            "runpy.run_module('vnext.native_terminal_process',run_name='__main__')"
        )
        self._owner = OwnedProcess.start(
            [sys.executable, "-u", "-c", sidecar_entrypoint],
            cwd=cwd, env=dict(os.environ), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        self._reader = threading.Thread(target=self._read, name="native-terminal-output", daemon=True)
        self._reader.start()
        try:
            self._send({"op": "start", "command": list(command), "cwd": cwd,
                        "environment": dict(environment), "columns": columns, "rows": rows})
            if not self._ready.wait(15):
                raise RuntimeError("native terminal did not start within 15 seconds")
            if self._failure:
                raise RuntimeError(self._failure)
        except BaseException:
            self.close()
            raise

    @property
    def alive(self) -> bool:
        return not self._closed and self._owner.process.poll() is None

    def _send(self, value: Mapping[str, Any]) -> None:
        with self._lock:
            stream = self._owner.process.stdin
            if self._closed or stream is None:
                raise RuntimeError("native terminal is closed")
            stream.write(json.dumps(value, ensure_ascii=True) + "\n")
            stream.flush()

    def write(self, data: str) -> None:
        if not isinstance(data, str) or len(data) > 65536:
            raise ValueError("terminal input must be a string of at most 65536 characters")
        self._send({"op": "input", "data": data})

    def resize(self, columns: int, rows: int) -> None:
        if type(columns) is not int or type(rows) is not int or not (2 <= columns <= 1000 and 2 <= rows <= 1000):
            raise ValueError("terminal dimensions must be integers between 2 and 1000")
        self._send({"op": "resize", "columns": columns, "rows": rows})

    def _read(self) -> None:
        stream = self._owner.process.stdout
        assert stream is not None
        exit_seen = False
        try:
            for line in stream:
                value = json.loads(line)
                if value.get("type") == "ready":
                    self._ready.set()
                elif value.get("type") == "error":
                    self._failure = str(value.get("error"))
                    self._ready.set()
                elif value.get("type") == "exit":
                    exit_seen = True
                self._on_event(value)
        finally:
            if not self._ready.is_set():
                self._failure = "native terminal process exited before startup"
            self._ready.set()
            if not exit_seen:
                self._on_event({"type": "exit", "code": self._owner.process.poll(), "unexpected": True})
            stream.close()

    def close(self) -> ProcessCleanup:
        with self._lock:
            self._closed = True
        cleanup = self._owner.close(grace_seconds=0.2)
        if threading.current_thread() is not self._reader:
            self._reader.join(timeout=5)
        return cleanup


# How long one terminal input may spend waiting for a PTY that will not take
# more.  A non-blocking master accepts about 1 KB at a time on macOS, so a
# paste of a few kilobytes needs several passes while the child reads; a child
# that has stopped reading altogether must not hold the sidecar's control loop
# for ever, which is what the deadline bounds.
_INPUT_DEADLINE_SECONDS = 5.0


def _write_all(payload: bytes, write_chunk: Callable[[bytes], int],
               wait_writable: Callable[[float], None]) -> int:
    """Hand every byte to the terminal, and report how many arrived.

    One ``os.write`` on a non-blocking PTY master takes what room the kernel
    buffer has and returns that count.  Discarding the count dropped the
    remainder of anything above about 1 KB without a word, so the child ran a
    truncated command.  This loops on the count and waits for the master to
    become writable again, and the caller is told the delivered count when the
    deadline passes.
    """

    delivered = 0
    deadline = time.monotonic() + _INPUT_DEADLINE_SECONDS
    while delivered < len(payload):
        try:
            count = write_chunk(payload[delivered:])
        except BlockingIOError:
            count = 0
        if count:
            delivered += count
            continue
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        wait_writable(remaining)
    return delivered


def _frame(value: Mapping[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=True), flush=True)


def _sidecar() -> None:
    config = json.loads(sys.stdin.readline())
    command = config["command"]
    env = dict(os.environ)
    env.update(config["environment"])
    # A background service often inherits TERM=dumb. This child has a real
    # VT-capable PTY, so advertise that instead of the service's pipe context.
    if env.get("TERM", "dumb") == "dumb":
        env["TERM"] = "xterm-256color"
    executable = shutil.which(command[0], path=env.get("PATH")) or command[0]
    columns, rows = int(config["columns"]), int(config["rows"])
    if not (2 <= columns <= 1000 and 2 <= rows <= 1000):
        raise ValueError("invalid terminal size")
    controls: queue.Queue[dict[str, Any]] = queue.Queue()

    def receive() -> None:
        for line in sys.stdin:
            controls.put(json.loads(line))
        controls.put({"op": "close"})

    if os.name == "nt":
        from winpty import PTY
        terminal = PTY(columns, rows)
        block = "\0".join(f"{key}={value}" for key, value in sorted(env.items())) + "\0"
        terminal.spawn(executable, " " + subprocess.list2cmdline(command[1:]), config["cwd"], block)
        read = lambda: terminal.read(False)
        def write_chunk(payload: bytes) -> int:
            # winpty takes text and reports no count, so one call is
            # the whole payload or an exception.
            terminal.write(payload.decode("utf-8", "ignore"))
            return len(payload)
        wait_writable = lambda timeout: time.sleep(min(0.015, timeout))
        resize = terminal.set_size
        alive = terminal.isalive
        status = terminal.get_exitstatus
    else:
        import errno
        import fcntl
        import pty
        import select
        import struct
        import termios
        master, slave = pty.openpty()
        os.set_blocking(master, False)
        def resize(c: int, r: int) -> None:
            fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", r, c, 0, 0))
        resize(columns, rows)
        process = subprocess.Popen([executable, *command[1:]], cwd=config["cwd"], env=env,
                                   stdin=slave, stdout=slave, stderr=slave)
        # Keep this end of the PTY open. macOS throws away whatever is still
        # buffered the moment the last slave descriptor closes, so releasing it
        # here would lose the child's final line whenever the child exits
        # between two polls. Holding it keeps that tail readable after exit.
        import codecs
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        def read() -> str:
            try:
                return decoder.decode(os.read(master, 65536))
            except OSError as exc:
                if exc.errno in {errno.EAGAIN, errno.EIO}:
                    return ""
                raise
        def write_chunk(payload: bytes) -> int:
            return os.write(master, payload)
        def wait_writable(timeout: float) -> None:
            select.select([], [master], [], timeout)
        alive = lambda: process.poll() is None
        status = process.poll
    threading.Thread(target=receive, daemon=True).start()
    _frame({"type": "ready"})
    while True:
        while not controls.empty():
            item = controls.get_nowait()
            if item["op"] == "close":
                return  # Owner terminates the remaining process tree.
            if item["op"] == "input":
                payload = item["data"].encode("utf-8")
                delivered = _write_all(payload, write_chunk, wait_writable)
                if delivered < len(payload):
                    _frame({"type": "input-error", "delivered": delivered,
                            "requested": len(payload),
                            "error": "the terminal took only "
                                     f"{delivered} of {len(payload)} bytes within "
                                     f"{_INPUT_DEADLINE_SECONDS:g} seconds"})
            elif item["op"] == "resize":
                resize(item["columns"], item["rows"])
        data = read()
        if data:
            _frame({"type": "output", "data": data})
        elif not alive():
            # The child can write its last line and exit between the empty
            # read above and this check, so drain the terminal before the exit
            # frame or that output is dropped. The sidecar still holds the
            # slave, so a Unix PTY keeps that tail buffered and one empty read
            # ends the drain; winpty can hand the tail over shortly after the
            # process goes, so keep polling there until the deadline.
            deadline = time.monotonic() + (0.2 if os.name == "nt" else 0.0)
            while True:
                data = read()
                if data:
                    _frame({"type": "output", "data": data})
                    continue
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.015)
            _frame({"type": "exit", "code": status()})
            return
        else:
            time.sleep(0.015)


if __name__ == "__main__":
    try:
        _sidecar()
    except Exception as exc:
        _frame({"type": "error", "error": str(exc)})
        sys.exit(1)
