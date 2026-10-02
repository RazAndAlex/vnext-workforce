from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vnext.release_check import (
    PINNED_BUNDLE_SHA256,
    PINNED_BUNDLE_SHA256_DARWIN_ARM64,
    RELEASE_CODEX_MODEL_COMPATIBILITY,
    RELEASE_CODEX_VERSION,
    RELEASE_PACKAGES,
    RELEASE_PYTHON,
    ReleaseCheckError,
    current_platform_key,
    main,
    verify_pinned_live_runtime,
    verify_release_environment,
)


class ReleaseCheckTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.bin_dir = self.root / "bin"
        self.path_dir = self.root / "codex-path"
        self.bin_dir.mkdir()
        self.path_dir.mkdir()
        self.codex = self.bin_dir / "codex.exe"
        for path in (
            self.codex,
            self.bin_dir / "codex-code-mode-host.exe",
            self.path_dir / "rg.exe",
            self.path_dir.parent / "codex-resources" / "codex-windows-sandbox-setup.exe",
            self.path_dir.parent / "codex-resources" / "codex-command-runner.exe",
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def run_version(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=[], returncode=0, stdout=f"codex-cli {RELEASE_CODEX_VERSION}\n", stderr=""
        )

    def verify(self, **overrides):
        arguments = {
            "version_lookup": RELEASE_PACKAGES.__getitem__,
            "python_version": RELEASE_PYTHON,
            "runtime_path_lookup": lambda: self.codex,
            "runtime_path_dir_lookup": lambda: self.path_dir,
            "command_runner": self.run_version,
            "platform_name": "Windows",
            "hash_lookup": lambda path: PINNED_BUNDLE_SHA256[path.name],
        }
        arguments.update(overrides)
        return verify_release_environment(**arguments)

    def test_exact_complete_runtime_passes_without_absolute_paths(self):
        receipt = self.verify()

        self.assertEqual("passed", receipt["status"])
        self.assertEqual(f"codex-cli {RELEASE_CODEX_VERSION}", receipt["runtime"])
        self.assertEqual(
            [
                "codex-code-mode-host.exe",
                "codex-command-runner.exe",
                "codex-windows-sandbox-setup.exe",
                "codex.exe",
                "rg.exe",
            ],
            receipt["bundle_files"],
        )
        self.assertNotIn(str(self.root), str(receipt))

    def test_gpt_6_models_are_compatible_and_in_the_default_catalog(self):
        from vnext.vnext_mcp_server import DEFAULT_CATALOG

        expected = {"gpt-6-sol", "gpt-6-luna", "gpt-6-astra"}
        compatible = RELEASE_CODEX_MODEL_COMPATIBILITY[RELEASE_CODEX_VERSION]
        catalog = {item["model"] for item in DEFAULT_CATALOG}

        self.assertTrue(expected <= compatible)
        self.assertTrue(expected <= catalog)

    def test_dependency_drift_fails_closed(self):
        def drifted(name: str) -> str:
            return "9.9.9" if name == "pydantic" else RELEASE_PACKAGES[name]

        with self.assertRaisesRegex(ReleaseCheckError, "pydantic is 9.9.9"):
            self.verify(version_lookup=drifted)

    def test_missing_adjacent_command_host_fails_closed(self):
        (self.bin_dir / "codex-code-mode-host.exe").unlink()

        with self.assertRaisesRegex(ReleaseCheckError, "codex-code-mode-host.exe"):
            self.verify()

    def test_wrong_runtime_version_fails_closed(self):
        def wrong_version(*_args, **_kwargs):
            return subprocess.CompletedProcess(
                args=[], returncode=0, stdout="codex-cli 0.999.0\n", stderr=""
            )

        with self.assertRaisesRegex(ReleaseCheckError, "0.999.0"):
            self.verify(command_runner=wrong_version)


class PinnedBundlePlatformTests(unittest.TestCase):
    """verify_pinned_live_runtime() picks the bundle map for the platform."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.bin_dir = self.root / "bin"
        self.path_dir = self.root / "codex-path"
        self.resources = self.root / "codex-resources"
        for directory in (self.bin_dir, self.path_dir, self.resources):
            directory.mkdir(parents=True, exist_ok=True)
        for name in (
            "codex",
            "codex.exe",
            "codex-code-mode-host",
            "codex-code-mode-host.exe",
        ):
            (self.bin_dir / name).write_bytes(b"fixture")
        (self.path_dir / "rg").write_bytes(b"fixture")
        (self.path_dir / "rg.exe").write_bytes(b"fixture")
        (self.resources / "codex-windows-sandbox-setup.exe").write_bytes(b"fixture")
        (self.resources / "codex-command-runner.exe").write_bytes(b"fixture")
        (self.resources / "zsh" / "bin").mkdir(parents=True)
        (self.resources / "zsh" / "bin" / "zsh").write_bytes(b"fixture")

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def run_version(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=[], returncode=0, stdout=f"codex-cli {RELEASE_CODEX_VERSION}\n", stderr=""
        )

    def verify(self, platform_key, pinned, runtime_name, **overrides):
        arguments = {
            "version_lookup": lambda _name: RELEASE_CODEX_VERSION,
            "runtime_path_lookup": lambda: self.bin_dir / runtime_name,
            "runtime_path_dir_lookup": lambda: self.path_dir,
            "command_runner": self.run_version,
            "hash_lookup": lambda path: pinned[path.name],
            "platform_key": platform_key,
        }
        arguments.update(overrides)
        return verify_pinned_live_runtime(**arguments)

    def test_darwin_arm64_bundle_verifies(self):
        _path, identity = self.verify(
            "darwin-arm64", PINNED_BUNDLE_SHA256_DARWIN_ARM64, "codex"
        )

        self.assertEqual("darwin-arm64", identity["platform"])
        self.assertEqual(
            ["codex", "codex-code-mode-host", "rg", "zsh"],
            sorted(identity["bundle_sha256"]),
        )
        self.assertEqual(
            PINNED_BUNDLE_SHA256_DARWIN_ARM64["codex"], identity["codex_sha256"]
        )

    def test_windows_bundle_is_unchanged(self):
        _path, identity = self.verify(
            "win32-amd64", PINNED_BUNDLE_SHA256, "codex.exe"
        )

        self.assertEqual(PINNED_BUNDLE_SHA256, identity["bundle_sha256"])
        self.assertEqual(
            [
                "codex-code-mode-host.exe",
                "codex-command-runner.exe",
                "codex-windows-sandbox-setup.exe",
                "codex.exe",
                "rg.exe",
            ],
            sorted(identity["bundle_sha256"]),
        )

    def test_unpinned_platform_fails_closed(self):
        with self.assertRaisesRegex(ReleaseCheckError, "linux-riscv64"):
            self.verify("linux-riscv64", PINNED_BUNDLE_SHA256_DARWIN_ARM64, "codex")

    def test_unsupported_platform_names_both_supported_platforms_before_package_lookup(self):
        with patch.object(sys, "platform", "linux"), patch(
            "vnext.release_check.platform.machine", return_value="x86_64"
        ):
            with self.assertRaises(ReleaseCheckError) as raised:
                verify_pinned_live_runtime(
                    version_lookup=lambda _name: (_ for _ in ()).throw(
                        AssertionError("package lookup must not run")
                    )
                )
        self.assertIn("darwin-arm64", str(raised.exception))
        self.assertIn("win32-amd64", str(raised.exception))

    def test_tampered_darwin_hash_fails_closed(self):
        tampered = dict(PINNED_BUNDLE_SHA256_DARWIN_ARM64, rg="00" * 32)

        with self.assertRaisesRegex(ReleaseCheckError, "rg SHA-256 is 0{64}"):
            self.verify("darwin-arm64", tampered, "codex")

    def test_platform_key_normalises_architecture(self):
        self.assertEqual("darwin-arm64", current_platform_key("darwin", "aarch64"))
        self.assertEqual("win32-amd64", current_platform_key("win32", "AMD64"))


class TheModuleRunCoversBothPromisedPlatformsTests(unittest.TestCase):
    """What a user can type has to be true on the platforms the README promises.

    ``verify_release_environment`` describes one machine: Windows, the exact
    release Python, exact transitive package pins and a four-file Windows
    bundle.  It used to be what ``python -m vnext.release_check`` ran,
    so an Apple-silicon Mac - the other promised platform - collected eleven
    failures about a computer the person does not own.  The default run now does
    the per-platform bundle check, which is the one that can pass on both, and
    the release-machine check keeps its own flag and says so when it refuses.
    """

    def test_the_default_run_checks_this_platform(self):
        receipt = {"platform": "darwin-arm64", "cli_version": RELEASE_CODEX_VERSION}
        with patch(
            "vnext.release_check.verify_pinned_live_runtime",
            return_value=(Path("/bundle/codex"), receipt),
        ) as per_platform, patch(
            "vnext.release_check.verify_release_environment"
        ) as release_machine:
            exit_code = main([])

        self.assertEqual(0, exit_code)
        self.assertEqual(1, per_platform.call_count)
        self.assertEqual(0, release_machine.call_count)

    def test_a_failed_default_run_prints_the_reason_and_exits_one(self):
        with patch(
            "vnext.release_check.verify_pinned_live_runtime",
            side_effect=ReleaseCheckError("pinned Codex CLI bundle is not installed"),
        ):
            with patch("builtins.print") as printed:
                exit_code = main([])

        self.assertEqual(1, exit_code)
        self.assertIn(
            "pinned Codex CLI bundle is not installed",
            " ".join(str(call.args[0]) for call in printed.call_args_list),
        )

    def test_the_flag_runs_the_release_machine_check(self):
        with patch(
            "vnext.release_check.verify_release_environment",
            return_value={"status": "passed"},
        ) as release_machine, patch(
            "vnext.release_check.verify_pinned_live_runtime"
        ) as per_platform:
            exit_code = main(["--release-machine"])

        self.assertEqual(0, exit_code)
        self.assertEqual(1, release_machine.call_count)
        self.assertEqual(0, per_platform.call_count)

    def test_the_release_machine_refusal_names_itself(self):
        with self.assertRaisesRegex(ReleaseCheckError, "release-machine"):
            verify_release_environment(
                platform_name="Darwin",
                python_version=RELEASE_PYTHON,
                version_lookup=lambda name: RELEASE_PACKAGES[name],
                runtime_path_lookup=lambda: Path("/nowhere/codex.exe"),
                runtime_path_dir_lookup=lambda: Path("/nowhere"),
            )

    def test_the_release_machine_check_says_what_to_run_instead(self):
        with self.assertRaisesRegex(ReleaseCheckError, "without --release-machine"):
            verify_release_environment(
                platform_name="Darwin",
                python_version=RELEASE_PYTHON,
                version_lookup=lambda name: RELEASE_PACKAGES[name],
                runtime_path_lookup=lambda: Path("/nowhere/codex.exe"),
                runtime_path_dir_lookup=lambda: Path("/nowhere"),
            )


if __name__ == "__main__":
    unittest.main()
