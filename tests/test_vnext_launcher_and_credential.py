"""First-time setup regressions from round 8."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock

import suite_environment  # noqa: F401  # the suite settings; unittest never reads conftest.py
from vnext.vnext_claude import ClaudeCodeAdapter


class SetupCopyAndCredentialTests(unittest.TestCase):
    def test_credential_error_names_adapter_provider_on_first_event(self) -> None:
        for provider, name, path in (
            ("zai", "Z.ai provider", "providers.json"),
            ("claude", "Claude provider", "claude auth login"),
        ):
            with self.subTest(provider=provider):
                adapter = ClaudeCodeAdapter.__new__(ClaudeCodeAdapter)
                adapter.provider = provider
                adapter._events = []
                adapter._set_fatal = Mock()
                adapter._bind_emitted_identity = Mock()
                adapter._record_event({"event": {
                    "name": "provider_error",
                    "provider_error": {"code": "authentication_failed", "api_error_status": 401},
                }})
                self.assertEqual(1, adapter._set_fatal.call_count)
                said = adapter._set_fatal.call_args.args[0]
                self.assertIn(name, said)
                self.assertIn(path, said)
                self.assertEqual(1, len(adapter._events))


@unittest.skipIf(os.name == "nt", "POSIX pid lifecycle probe")
class ProxyEofTests(unittest.TestCase):
    def test_fake_proxy_and_child_end_on_client_eof(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            root = Path(temp)
            marker = root / "pids"
            fake = root / "fake_proxy.py"
            fake.write_text(
                "import os, subprocess, sys, time\n"
                "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
                "open(sys.argv[1], 'w').write(f'{os.getpid()} {child.pid}')\n"
                "sys.stdout.buffer.write(b'proxy-bytes\\n'); sys.stdout.buffer.flush()\n"
                "time.sleep(60)\n"
            )
            entry = (
                "import sys; from vnext import vnext_mcp_reload as r; "
                f"r._prepared_command=lambda *_: ([sys.executable, {str(fake)!r}, {str(marker)!r}], None); "
                "raise SystemExit(r.main(['--plugin-root', '.']))"
            )
            proc = subprocess.Popen(
                [sys.executable, "-c", entry], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            pids = []
            try:
                deadline = time.monotonic() + 3
                while not marker.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(marker.exists(), "proxy did not start")
                pids = [int(value) for value in marker.read_text().split()]
                assert proc.stdin is not None
                # A real client speaks before it quits.  One that never sent a
                # byte is given the close grace to fail on its own instead.
                proc.stdin.write(b'{"jsonrpc": "2.0"}\n')
                proc.stdin.flush()
                proc.stdin.close()
                self.assertEqual(0, proc.wait(timeout=5))
                assert proc.stdout is not None
                self.assertEqual(b"proxy-bytes\n", proc.stdout.read())
                for pid in pids:
                    with self.assertRaises(ProcessLookupError):
                        os.kill(pid, 0)
            finally:
                for pid in pids:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                if proc.poll() is None:
                    proc.kill()
                proc.wait(timeout=3)
                if proc.stdout:
                    proc.stdout.close()
                if proc.stderr:
                    proc.stderr.close()

    def test_a_slow_server_close_finishes_before_the_tree_is_ended(self) -> None:
        """The server may need seconds to close its workers; EOF must not cut it short."""
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            root = Path(temp)
            marker = root / "pids"
            closed = root / "closed"
            server = root / "slow_server.py"
            server.write_text(
                "import signal, sys, time\n"
                "def stop(*_):\n"
                "    time.sleep(3)\n"
                f"    open({str(closed)!r}, 'w').write('done')\n"
                "    sys.exit(0)\n"
                "signal.signal(signal.SIGTERM, stop)\n"
                "time.sleep(60)\n"
            )
            fake = root / "fake_proxy.py"
            fake.write_text(
                "import os, signal, subprocess, sys, time\n"
                f"child = subprocess.Popen([sys.executable, {str(server)!r}])\n"
                "signal.signal(signal.SIGTERM, lambda *_: (child.wait(), sys.exit(0)))\n"
                "open(sys.argv[1], 'w').write(f'{os.getpid()} {child.pid}')\n"
                "time.sleep(60)\n"
            )
            entry = (
                "import sys; from vnext import vnext_mcp_reload as r; "
                f"r._prepared_command=lambda *_: ([sys.executable, {str(fake)!r}, {str(marker)!r}], None); "
                "raise SystemExit(r.main(['--plugin-root', '.']))"
            )
            proc = subprocess.Popen(
                [sys.executable, "-c", entry], stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            pids: list[int] = []
            try:
                deadline = time.monotonic() + 3
                while not marker.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(marker.exists(), "proxy did not start")
                time.sleep(0.3)
                pids = [int(value) for value in marker.read_text().split()]
                assert proc.stdin is not None
                proc.stdin.close()
                self.assertEqual(0, proc.wait(timeout=15))
                self.assertTrue(closed.exists(), "the server was killed inside its own close")
                for pid in pids:
                    with self.assertRaises(ProcessLookupError):
                        os.kill(pid, 0)
            finally:
                for pid in pids:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                if proc.poll() is None:
                    proc.kill()
                proc.wait(timeout=3)


class ProxyGraceTests(unittest.TestCase):
    def test_the_launcher_waits_longer_than_the_server_close(self) -> None:
        from vnext import vnext_mcp_reload, vnext_mcp_server

        self.assertGreater(
            vnext_mcp_reload.PROXY_CLOSE_GRACE_SECONDS, vnext_mcp_server.CLOSE_GRACE_SECONDS
        )
