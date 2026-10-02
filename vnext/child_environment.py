"""The minimal environment handed to a controlled Codex child process.

These names live apart from ``app_server`` on purpose.  The runtime needs only
the environment builder, while ``app_server`` also holds
``prepare_private_codex_home``, which copies ``auth.json`` into a private Codex
home.  Codex refresh tokens rotate when they are used, so a second copy of that
file signs the owner out of the first one.  Keeping the builder here means the
import closure of the shipped package never reaches the copy, and the public
export never carries it.
"""

from __future__ import annotations

import os
from pathlib import Path


# The app-server is launched directly, so it only needs executable lookup,
# temporary-directory, locale, and Windows loader/shell settings. Authentication
# is supplied through the isolated CODEX_HOME rather than a credential variable.
_CHILD_ENVIRONMENT_ALLOWLIST = frozenset(
    {
        "COMSPEC",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "WINDIR",
    }
)


def child_process_environment(
    codex_home: str | Path,
    source: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build the minimal environment used by the controlled app-server process."""

    inherited = os.environ if source is None else source
    environment = {
        name.upper(): value
        for name, value in inherited.items()
        if name.upper() in _CHILD_ENVIRONMENT_ALLOWLIST
    }
    environment["CODEX_HOME"] = str(Path(codex_home).resolve())
    return environment


def child_environment_diagnostic(environment: dict[str, str]) -> dict[str, object]:
    """Describe a child environment without exposing any of its values."""

    names = sorted(environment, key=str.casefold)
    return {
        "included_count": len(names),
        "included_names": names,
        "values_redacted": True,
    }
