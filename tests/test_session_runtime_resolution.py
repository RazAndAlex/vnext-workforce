import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vnext.live_runtime import resolve_session_runtime


class SessionRuntimeResolutionTests(unittest.TestCase):
    def test_default_keeps_verified_pinned_runtime(self):
        with patch("vnext.live_runtime.resolve_live_runtime", return_value=(Path("pinned"), {})) as resolve:
            self.assertEqual(resolve_session_runtime(), (Path("pinned"), {}))
            resolve.assert_called_once_with()

    def test_changed_binary_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "codex.exe"
            executable.write_bytes(b"changed")
            with patch("vnext.live_runtime.subprocess.run") as run:
                with self.assertRaisesRegex(RuntimeError, "identity changed"):
                    resolve_session_runtime({"executable": str(executable), "sha256": "0" * 64, "version": "expected"})
                run.assert_not_called()

    def test_registered_version_must_match_and_file_is_released(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "codex.exe"
            executable.write_bytes(b"fixture")
            registration = {"executable": str(executable), "sha256": hashlib.sha256(b"fixture").hexdigest(), "version": "codex-cli 1"}
            with patch("vnext.live_runtime.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "codex-cli 1\n", "")):
                resolved, identity = resolve_session_runtime(registration)
            self.assertEqual(resolved, executable.resolve())
            self.assertEqual(identity["version"], "codex-cli 1")
            with patch("vnext.live_runtime.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "codex-cli 2\n", "")):
                with self.assertRaisesRegex(RuntimeError, "version changed"):
                    resolve_session_runtime(registration)
            executable.unlink()


if __name__ == "__main__":
    unittest.main()
