"""Check the pinned Codex CLI bundle, and the release machine that builds it.

Two checks live here and they answer different questions.

``verify_pinned_live_runtime`` is the per-platform one.  It runs on either
supported platform, Apple-silicon macOS and Windows x64, and it is what
``python -m vnext.release_check`` does by default: the pinned
``openai-codex-cli-bin`` version, the SHA-256 of every file in that platform's
bundle, and the version the executable reports.

``verify_release_environment`` is the Windows release-machine check.  It also
demands the exact release Python and the exact transitive package pins, so it
passes on the machine that builds the release and nowhere else.  It sits behind
``--release-machine``.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from . import __version__


RELEASE_PYTHON = "3.14.6"
RELEASE_CODEX_VERSION = "0.160.0"
# Hand-maintained compatibility table for the pinned Codex CLI.  Update this
# table whenever RELEASE_CODEX_VERSION moves; it is release evidence, not a
# model-name heuristic that can be derived safely at runtime.
RELEASE_CODEX_MODEL_COMPATIBILITY = {
    RELEASE_CODEX_VERSION: frozenset({
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-6-astra",
        "gpt-6-sol",
        "gpt-6-luna",
        "gpt-6.1-sol",
    }),
}
VNEXT_VERSION = "0.3.0"
# Hand-maintained release evidence, same as the model compatibility table above.
# Each wheel of openai-codex-cli-bin ships a different set of executables, so the
# pinned digests are per platform.  Recompute every map whenever
# RELEASE_CODEX_VERSION moves.
#
# Windows: openai-codex-cli-bin 0.160.0 win_amd64.
PINNED_BUNDLE_SHA256 = {
    "codex.exe": "fdda5fa3cf3fb3d000b876720742857676293e4315e4b045fae6f8bd7e866d1d",
    "codex-code-mode-host.exe": "1d448bfde19e7a280d600d8d0bcddf77afbe9feaec1e804905becc5f39bc9db6",
    "rg.exe": "14231169855ec5205cf5a1b6f1db358ff4aed4247c86b69ce8aae647c77f6680",
    "codex-windows-sandbox-setup.exe": "8f91d62aca2aca87b0ebf446848b817720415dd3080905ffc3085c17b70bd57a",
    "codex-command-runner.exe": "dd13f8e95faba5cf539d870dd3b64c226b00d062a5b523e7e08c44fdc8235224",
}
# macOS arm64: openai-codex-cli-bin 0.160.0 macosx_11_0_arm64.  That wheel has no
# counterpart to the two Windows helpers; it ships zsh for the shell tool instead.
# Recompute these digests when RELEASE_CODEX_VERSION moves.
PINNED_BUNDLE_SHA256_DARWIN_ARM64 = {
    "codex": "112fae7a5a1223e673c8a1791d32338f37df8b527ff1159bb8adac6c4dbf1b4b",
    "codex-code-mode-host": "679eedaea70529aa1cffc9bc0a0788c186412663544fa76c09d63b57f383a65a",
    "rg": "7c7d5b09c3a57de864500c65dc2e9c4d2a5e96a3fc7b7d5318480c19887a739b",
    "zsh": "d715e06edcf1661edb2e7767d65fbd1b44af505edfb217281b701f0a6ae93b15",
}
PINNED_BUNDLE_SHA256_BY_PLATFORM = {
    "win32-amd64": PINNED_BUNDLE_SHA256,
    "darwin-arm64": PINNED_BUNDLE_SHA256_DARWIN_ARM64,
}
RELEASE_PACKAGES = {
    "annotated-types": "0.7.0",
    "openai-codex": RELEASE_CODEX_VERSION,
    "openai-codex-cli-bin": RELEASE_CODEX_VERSION,
    "packaging": "26.3",
    "pydantic": "2.13.4",
    "pydantic-core": "2.46.4",
    "setuptools": "83.0.0",
    "typing-extensions": "4.16.0",
    "typing-inspection": "0.4.2",
}


class ReleaseCheckError(RuntimeError):
    pass


def current_platform_key(
    sys_platform: str | None = None, machine: str | None = None
) -> str:
    """Name the running platform the way the pinned bundle table keys it."""

    sys_platform = sys_platform or sys.platform
    machine = (machine or platform.machine()).lower()
    architecture = {
        "amd64": "amd64",
        "x86_64": "amd64",
        "arm64": "arm64",
        "aarch64": "arm64",
    }.get(machine, machine)
    return f"{sys_platform}-{architecture}"


def _bundle_layout(
    platform_key: str, runtime_path: Path, path_dir: Path
) -> tuple[dict[str, Path], dict[str, str], str]:
    """Return (label -> path, label -> pinned SHA-256, runtime label)."""

    expected = PINNED_BUNDLE_SHA256_BY_PLATFORM.get(platform_key)
    if expected is None:
        raise ReleaseCheckError(
            f"no pinned Codex CLI bundle for platform {platform_key!r}; "
            f"pinned platforms are "
            f"{', '.join(sorted(PINNED_BUNDLE_SHA256_BY_PLATFORM))}"
        )
    if platform_key == "win32-amd64":
        paths = {
            "codex.exe": runtime_path,
            "codex-code-mode-host.exe": runtime_path.with_name(
                "codex-code-mode-host.exe"
            ),
            "rg.exe": path_dir / "rg.exe",
            "codex-windows-sandbox-setup.exe": path_dir.parent
            / "codex-resources"
            / "codex-windows-sandbox-setup.exe",
            "codex-command-runner.exe": path_dir.parent
            / "codex-resources"
            / "codex-command-runner.exe",
        }
        return paths, expected, "codex.exe"
    paths = {
        "codex": runtime_path,
        "codex-code-mode-host": runtime_path.with_name("codex-code-mode-host"),
        "rg": path_dir / "rg",
        "zsh": path_dir.parent / "codex-resources" / "zsh" / "bin" / "zsh",
    }
    return paths, expected, "codex"


def verify_pinned_live_runtime(
    *,
    version_lookup: Callable[[str], str] = importlib.metadata.version,
    runtime_path_lookup: Callable[[], Path] | None = None,
    runtime_path_dir_lookup: Callable[[], Path | None] | None = None,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    hash_lookup: Callable[[Path], str] | None = None,
    platform_key: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Verify the exact executable bundle before it can receive copied auth."""

    platform_key = platform_key or current_platform_key()
    if platform_key not in PINNED_BUNDLE_SHA256_BY_PLATFORM:
        raise ReleaseCheckError(
            f"unsupported Codex worker platform {platform_key!r}; supported platforms are "
            f"{', '.join(sorted(PINNED_BUNDLE_SHA256_BY_PLATFORM))}"
        )

    try:
        package_version = version_lookup("openai-codex-cli-bin")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ReleaseCheckError("pinned Codex CLI bundle is not installed") from exc
    if package_version != RELEASE_CODEX_VERSION:
        raise ReleaseCheckError(
            f"openai-codex-cli-bin is {package_version}, expected {RELEASE_CODEX_VERSION}"
        )
    if runtime_path_lookup is None or runtime_path_dir_lookup is None:
        try:
            from codex_cli_bin import bundled_codex_path, bundled_path_dir
        except ImportError as exc:
            raise ReleaseCheckError("openai-codex-cli-bin could not be imported") from exc
        runtime_path_lookup = runtime_path_lookup or bundled_codex_path
        runtime_path_dir_lookup = runtime_path_dir_lookup or bundled_path_dir
    runtime_path = runtime_path_lookup().resolve()
    path_dir = runtime_path_dir_lookup()
    if path_dir is None:
        raise ReleaseCheckError("pinned Codex CLI PATH directory is unavailable")
    path_dir = path_dir.resolve()
    paths, expected_hashes, runtime_label = _bundle_layout(
        platform_key, runtime_path, path_dir
    )
    hash_lookup = hash_lookup or _sha256_file
    verified_hashes: dict[str, str] = {}
    for label, path in paths.items():
        if not path.is_file():
            raise ReleaseCheckError(f"complete runtime bundle is missing {label}")
        actual_hash = hash_lookup(path).lower()
        expected_hash = expected_hashes[label]
        if actual_hash != expected_hash:
            raise ReleaseCheckError(
                f"{label} SHA-256 is {actual_hash}, expected {expected_hash}"
            )
        verified_hashes[label] = actual_hash
    try:
        completed = command_runner(
            [str(runtime_path), "--version"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ReleaseCheckError(
            f"runtime version probe failed: {type(exc).__name__}: {exc}"
        ) from exc
    runtime_version = completed.stdout.strip()
    expected_output = f"codex-cli {RELEASE_CODEX_VERSION}"
    if runtime_version != expected_output:
        raise ReleaseCheckError(
            f"runtime reports {runtime_version!r}, expected {expected_output!r}"
        )
    identity = {
        "distribution": "openai-codex-cli-bin",
        "distribution_version": package_version,
        "cli_version": RELEASE_CODEX_VERSION,
        "platform": platform_key,
        "codex_sha256": verified_hashes[runtime_label],
        "bundle_sha256": verified_hashes,
    }
    return runtime_path, identity


def verify_release_environment(
    *,
    version_lookup: Callable[[str], str] = importlib.metadata.version,
    python_version: str | None = None,
    runtime_path_lookup: Callable[[], Path] | None = None,
    runtime_path_dir_lookup: Callable[[], Path | None] | None = None,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    hash_lookup: Callable[[Path], str] | None = None,
    platform_name: str | None = None,
) -> dict[str, Any]:
    """Check the Windows machine that builds the release, and only that machine.

    This is release evidence rather than a user diagnostic: it demands Windows,
    the exact release Python, every locked transitive package version, and the
    four-file Windows bundle.  A supported computer that is not the release
    machine fails it by design, so run the module without ``--release-machine``
    to check a normal install.
    """

    actual_python = python_version or platform.python_version()
    actual_platform = platform_name or platform.system()
    errors: list[str] = []

    if __version__ != VNEXT_VERSION:
        errors.append(
            f"package version is {__version__}, expected {VNEXT_VERSION}"
        )
    if actual_platform != "Windows":
        errors.append(
            f"the release machine is Windows and this is {actual_platform}; "
            f"run the module without --release-machine to check this computer"
        )
    if actual_python != RELEASE_PYTHON:
        errors.append(
            f"Python version is {actual_python}, expected the release pin {RELEASE_PYTHON}"
        )

    resolved_versions: dict[str, str] = {}
    for name, expected in RELEASE_PACKAGES.items():
        try:
            actual = version_lookup(name)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"locked package is missing: {name}=={expected}")
            continue
        resolved_versions[name] = actual
        if actual != expected:
            errors.append(f"{name} is {actual}, expected {expected}")

    if runtime_path_lookup is None or runtime_path_dir_lookup is None:
        try:
            from codex_cli_bin import bundled_codex_path, bundled_path_dir
        except ImportError:
            errors.append("openai-codex-cli-bin could not be imported")
            bundled_codex_path = None
            bundled_path_dir = None
        runtime_path_lookup = runtime_path_lookup or bundled_codex_path
        runtime_path_dir_lookup = runtime_path_dir_lookup or bundled_path_dir

    runtime_path = runtime_path_lookup() if runtime_path_lookup else None
    path_dir = runtime_path_dir_lookup() if runtime_path_dir_lookup else None
    required_files: list[tuple[str, Path | None]] = [
        ("codex.exe", runtime_path),
        (
            "codex-code-mode-host.exe",
            runtime_path.with_name("codex-code-mode-host.exe")
            if runtime_path is not None
            else None,
        ),
        ("rg.exe", path_dir / "rg.exe" if path_dir is not None else None),
        (
            "codex-windows-sandbox-setup.exe",
            path_dir.parent / "codex-resources" / "codex-windows-sandbox-setup.exe"
            if path_dir is not None
            else None,
        ),        (
            "codex-command-runner.exe",
            path_dir.parent / "codex-resources" / "codex-command-runner.exe"
            if path_dir is not None
            else None,
        ),
    ]
    present_files: list[str] = []
    verified_hashes: dict[str, str] = {}
    hash_lookup = hash_lookup or _sha256_file
    for label, path in required_files:
        if path is None or not path.is_file():
            errors.append(f"complete runtime bundle is missing {label}")
        else:
            present_files.append(label)
            actual_hash = hash_lookup(path).lower()
            expected_hash = PINNED_BUNDLE_SHA256[label]
            if actual_hash != expected_hash:
                errors.append(
                    f"{label} SHA-256 is {actual_hash}, expected {expected_hash}"
                )
            else:
                verified_hashes[label] = actual_hash

    runtime_version = None
    if runtime_path is not None and runtime_path.is_file():
        try:
            completed = command_runner(
                [str(runtime_path), "--version"],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=15,
            )
            runtime_version = completed.stdout.strip()
            expected_output = f"codex-cli {RELEASE_CODEX_VERSION}"
            if runtime_version != expected_output:
                errors.append(
                    f"runtime reports {runtime_version!r}, expected {expected_output!r}"
                )
        except (OSError, subprocess.SubprocessError) as exc:
            errors.append(f"runtime version probe failed: {type(exc).__name__}: {exc}")

    if errors:
        raise ReleaseCheckError(
            "Windows release-machine check failed: " + " | ".join(errors)
        )

    return {
        "status": "passed",
        "vnext": __version__,
        "python": actual_python,
        "platform": actual_platform,
        "packages": resolved_versions,
        "runtime": runtime_version,
        "bundle_files": sorted(present_files),
        "bundle_sha256": verified_hashes,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m vnext.release_check",
        description=(
            "Check the pinned Codex CLI bundle on this computer. The default "
            "check runs on either supported platform, Apple-silicon macOS and "
            "Windows x64."
        ),
    )
    parser.add_argument(
        "--release-machine",
        action="store_true",
        help=(
            "run the Windows release-machine check instead. It adds the exact "
            "release Python, the locked transitive package versions and the "
            "four-file Windows bundle, so it passes on the machine that builds "
            "the release and nowhere else."
        ),
    )
    parsed = parser.parse_args(argv)

    try:
        if parsed.release_machine:
            receipt = verify_release_environment()
        else:
            runtime_path, identity = verify_pinned_live_runtime()
            receipt = {
                "status": "passed",
                "check": "pinned runtime bundle for this platform",
                "runtime_path": str(runtime_path),
                **identity,
            }
    except ReleaseCheckError as exc:
        print(str(exc))
        return 1
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
