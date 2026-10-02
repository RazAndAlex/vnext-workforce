from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from pathlib import Path

from vnext.native_terminal_process import NativeTerminalProcess


def _cleanup_temporary_directory(directory: tempfile.TemporaryDirectory[str]) -> None:
    """Wait only for Windows ConPTY's final cwd-handle release."""

    deadline = time.monotonic() + 2.0
    while True:
        try:
            directory.cleanup()
            return
        except PermissionError as exc:
            if getattr(exc, "winerror", None) != 32 or time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


class NativeTerminalProcessTests(unittest.TestCase):
    def test_sidecar_uses_the_current_package_when_workspace_shadows_it(self):
        """The temporary PTY cwd must not select an editable-package shadow."""

        directory = tempfile.TemporaryDirectory()
        try:
            cwd = Path(directory.name)
            shadow = cwd / "vnext"
            shadow.mkdir()
            (shadow / "__init__.py").write_text("", encoding="utf-8")
            (shadow / "native_terminal_process.py").write_text(
                "raise SystemExit(23)\n", encoding="utf-8"
            )
            events = []
            exited = threading.Event()

            def receive(value):
                events.append(value)
                if value["type"] == "exit":
                    exited.set()

            terminal = NativeTerminalProcess(
                [sys.executable, "-c", "print('CURRENT-SIDECAR', flush=True)"],
                cwd=str(cwd), environment={}, on_event=receive,
            )
            try:
                self.assertTrue(exited.wait(10), events)
            finally:
                cleanup = terminal.close()
            self.assertEqual(0, cleanup.residual_count)
            self.assertFalse(cleanup.errors)
            self.assertIn("CURRENT-SIDECAR", "".join(item.get("data", "") for item in events))
        finally:
            _cleanup_temporary_directory(directory)

    def test_last_output_of_a_short_child_survives_the_exit_frame(self):
        """A child that writes and exits between two polls keeps its tail."""

        directory = tempfile.TemporaryDirectory()
        try:
            for attempt in range(8):
                events = []
                exited = threading.Event()

                def receive(value):
                    events.append(value)
                    if value["type"] == "exit":
                        exited.set()

                terminal = NativeTerminalProcess(
                    [sys.executable, "-c", "print('TAIL-MARKER', flush=True)"],
                    cwd=directory.name, environment={}, on_event=receive)
                try:
                    self.assertTrue(exited.wait(10), events)
                finally:
                    terminal.close()
                text = "".join(item.get("data", "") for item in events)
                self.assertIn("TAIL-MARKER", text, f"attempt {attempt}: {events}")
        finally:
            _cleanup_temporary_directory(directory)

    def test_pty_advertises_terminal_support_when_service_parent_is_headless(self):
        events = []
        exited = threading.Event()
        def receive(value):
            events.append(value)
            if value["type"] == "exit":
                exited.set()
        directory = tempfile.TemporaryDirectory()
        try:
            with patch.dict("os.environ", {"TERM": "dumb"}):
                terminal = NativeTerminalProcess(
                    [sys.executable, "-c", "import os; print('PTY_TERM='+os.environ['TERM'])"],
                    cwd=directory.name, environment={}, on_event=receive)
                try:
                    self.assertTrue(exited.wait(10))
                finally:
                    terminal.close()
        finally:
            _cleanup_temporary_directory(directory)
        self.assertIn("PTY_TERM=xterm-256color", "".join(item.get("data", "") for item in events))

    def test_real_pty_input_resize_output_and_complete_tree_cleanup(self):
        events = []
        exited = threading.Event()
        def receive(value):
            events.append(value)
            if value["type"] == "exit":
                exited.set()
        directory = tempfile.TemporaryDirectory()
        try:
            terminal = NativeTerminalProcess(
                [sys.executable, "-u", "-c", "print('READY',flush=True); s=input(); print('RECEIVED:'+s,flush=True)"],
                cwd=directory.name, environment={}, on_event=receive)
            try:
                terminal.resize(90, 28)
                terminal.write("terminal-proof\r")
                self.assertTrue(exited.wait(10), events)
            finally:
                cleanup = terminal.close()
            output = "".join(item.get("data", "") for item in events)
            self.assertIn("RECEIVED:terminal-proof", output)
            self.assertEqual(cleanup.residual_count, 0)
            self.assertFalse(cleanup.errors)
        finally:
            _cleanup_temporary_directory(directory)

    def test_explicit_close_kills_native_descendant(self):
        directory = tempfile.TemporaryDirectory()
        try:
            cwd = directory.name
            terminal = NativeTerminalProcess([sys.executable, "-c", "import time; time.sleep(120)"],
                                             cwd=cwd, environment={}, on_event=lambda value: None)
            try:
                self.assertTrue(terminal.alive)
                with self.assertRaises(ValueError):
                    terminal.resize(0, 25)
            finally:
                cleanup = terminal.close()
            self.assertEqual(cleanup.residual_count, 0)
            self.assertFalse(cleanup.errors)
            self.assertFalse(terminal.alive)
        finally:
            _cleanup_temporary_directory(directory)


if __name__ == "__main__":
    unittest.main()


@unittest.skipIf(os.name == "nt", "the non-blocking master is the POSIX PTY path")
class LargeTerminalInputTests(unittest.TestCase):
    """A paste of several kilobytes has to reach the child whole."""

    CHILD = "import os, sys, tty\ntty.setraw(sys.stdin.fileno())\nsys.stdout.write('RAW\\r\\n')\nsys.stdout.flush()\ntotal = 0\nwhile total < 65536:\n    chunk = os.read(sys.stdin.fileno(), 65536)\n    if not chunk:\n        break\n    total += len(chunk)\nsys.stdout.write('GOT %d\\r\\n' % total)\nsys.stdout.flush()\n"

    def test_every_byte_of_a_64_kb_paste_reaches_a_raw_mode_child(self):
        directory = tempfile.TemporaryDirectory()
        try:
            events = []
            seen = threading.Event()

            def receive(value):
                events.append(value)
                if "GOT " in "".join(item.get("data", "") for item in events):
                    seen.set()

            terminal = NativeTerminalProcess(
                [sys.executable, "-u", "-c", self.CHILD],
                cwd=directory.name, environment={}, on_event=receive)
            try:
                deadline = time.monotonic() + 10
                while "RAW" not in "".join(item.get("data", "") for item in events):
                    self.assertLess(time.monotonic(), deadline, events)
                    time.sleep(0.02)
                terminal.write("a" * 65536)
                self.assertTrue(seen.wait(20), events)
            finally:
                terminal.close()
            output = "".join(item.get("data", "") for item in events)
            self.assertIn("GOT 65536", output)
            self.assertEqual(
                [], [item for item in events if item.get("type") == "input-error"])
        finally:
            _cleanup_temporary_directory(directory)

    def test_a_terminal_that_stops_reading_reports_the_delivered_count(self):
        """A deadline that passes says how much arrived; it never drops in silence."""

        from vnext import native_terminal_process as module

        taken = []

        def write_chunk(payload):
            if sum(taken) >= 1022:
                raise BlockingIOError(11, "EAGAIN")
            taken.append(1022)
            return 1022

        with patch.object(module, "_INPUT_DEADLINE_SECONDS", 0.05):
            delivered = module._write_all(b"b" * 4096, write_chunk, time.sleep)
        self.assertEqual(1022, delivered)
