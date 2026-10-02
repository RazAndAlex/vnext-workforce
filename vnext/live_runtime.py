"""Resolution of the pinned, hash-verified live Codex runtime."""

from __future__ import annotations

from pathlib import Path
import hashlib
import subprocess
from typing import Any, Mapping

from .release_check import ReleaseCheckError, verify_pinned_live_runtime


def resolve_live_runtime() -> tuple[Path, dict[str, object]]:
    """Return only the identity- and hash-verified pinned Live runtime."""

    try:
        return verify_pinned_live_runtime()
    except ReleaseCheckError as exc:
        raise RuntimeError(
            "Live mode requires the exact pinned and hash-verified Codex runtime bundle"
        ) from exc


def resolve_session_runtime(selection: Mapping[str, Any] | None = None) -> tuple[Path, dict[str, object]]:
    """Resolve a registered native installation for a host session.

    The host may select a newer installed harness than the historical diagnostic
    bundle. Its registration includes the expected executable identity; a request
    cannot substitute another binary after that registration silently.
    """
    if not selection:
        return resolve_live_runtime()
    executable = Path(str(selection.get("executable", "")))
    expected_hash = selection.get("sha256")
    expected_version = selection.get("version")
    if not executable.is_absolute() or not executable.is_file():
        raise ValueError("registered Codex executable must be an existing absolute path")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError("registered Codex executable requires its SHA-256 identity")
    if not isinstance(expected_version, str) or not expected_version:
        raise ValueError("registered Codex executable requires its version")
    with executable.open("rb") as stream:
        actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual_hash.lower() != expected_hash.lower():
        raise RuntimeError("registered Codex executable identity changed")
    probe = subprocess.run([str(executable), "--version"], capture_output=True, text=True, timeout=15, check=True)
    actual_version = probe.stdout.strip()
    if actual_version != expected_version:
        raise RuntimeError("registered Codex executable version changed")
    return executable.resolve(), {"source": "registered-native-installation",
        "version": actual_version, "sha256": actual_hash}
