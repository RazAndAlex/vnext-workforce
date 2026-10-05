"""Suite-wide settings, for pytest and for ``python -m unittest`` alike.

``vnext-mcp --check`` names the exact model behind each alias by starting the
Claude CLI and a Codex app-server.  No test may start a provider, including a
check run in a subprocess, so the probe is switched off for the whole suite.
A test that covers the probe patches it directly.

pytest loads ``conftest.py`` by itself, but the Windows release gate runs
``python -m unittest discover -s tests``, which never reads it.  So the
settings live here, ``conftest.py`` imports this module, and so does every
test module that reaches the check, the catalog or the side runtimes.
Importing it twice changes nothing.
"""

import os
import tempfile
from pathlib import Path

# No test writes into the developer's real home: ~/.vnext, ~/.vnext,
# ~/.cache and the rest resolve inside a folder only this run uses.  HOME is
# set before any vnext import, because some modules read it once.
_SUITE_HOME_ENV = "VNEXT_TEST_SUITE_HOME"


def _pin_uv_dirs():
    # uv's package cache and managed Pythons are shared tool state, not vNext
    # state.  Pin them to the real locations before HOME moves, or a test that
    # starts the plugin through `uv run` downloads everything again (minutes).
    import shutil
    import subprocess

    uv = shutil.which("uv")
    if not uv:
        return
    for name, args in (("UV_CACHE_DIR", ["cache", "dir"]), ("UV_PYTHON_INSTALL_DIR", ["python", "dir"])):
        if os.environ.get(name):
            continue
        try:
            out = subprocess.run([uv, *args], capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            continue
        if out:
            os.environ[name] = out


if not os.environ.get(_SUITE_HOME_ENV):
    _pin_uv_dirs()
if not os.environ.get(_SUITE_HOME_ENV):
    os.environ[_SUITE_HOME_ENV] = tempfile.mkdtemp(prefix="vnext-test-home-")
os.environ["HOME"] = os.environ[_SUITE_HOME_ENV]
os.environ["USERPROFILE"] = os.environ["HOME"]  # Path.home() on Windows
for _name, _relative in (
    ("XDG_CONFIG_HOME", ".config"),
    ("XDG_CACHE_HOME", ".cache"),
    ("XDG_DATA_HOME", ".local/share"),
    ("XDG_STATE_HOME", ".local/state"),
    ("CODEX_HOME", ".codex"),
    ("CLAUDE_CONFIG_DIR", ".claude"),
):
    os.environ[_name] = os.path.join(os.environ["HOME"], _relative)
TEST_HOME = os.environ["HOME"]

import vnext.vnext_runtimes as _runtimes  # noqa: E402

os.environ["VNEXT_CHECK_SKIP_MODEL_PROBE"] = "1"

# No test reads PyPI, and none finds a developer's own side runtime: the update
# check is off, and side runtimes live in a folder only this run uses.
os.environ["VNEXT_NO_UPDATE_CHECK"] = "1"
# Tests of the daily notice may enable checking, but only auto-update tests
# with fake installers may enable automatic installation.
os.environ["VNEXT_AUTO_UPDATE"] = "0"
# A shell's own VNEXT_RUNTIMES_DIR is replaced, never trusted.  The folder made
# here is remembered, so a second import keeps the same one.
_SUITE_FOLDER_ENV = "VNEXT_TEST_SUITE_RUNTIMES_DIR"
if not os.environ.get(_SUITE_FOLDER_ENV):
    os.environ[_SUITE_FOLDER_ENV] = tempfile.mkdtemp(prefix="vnext-test-runtimes-")
os.environ["VNEXT_RUNTIMES_DIR"] = os.environ[_SUITE_FOLDER_ENV]

# A test that clears the environment loses the settings above, and the server
# it builds would then read PyPI and the real ~/.vnext/runtimes.  The module is
# patched as well, so such a test still reads neither.
TEST_RUNTIMES_DIR = os.environ["VNEXT_RUNTIMES_DIR"]


def _no_pypi(name):
    raise OSError(f"the test suite never reads PyPI ({name})")


def _test_runtimes_dir():
    return Path(os.environ.get(_runtimes.RUNTIMES_ENV) or TEST_RUNTIMES_DIR)


_runtimes._fetch_pypi = _no_pypi
_runtimes.runtimes_dir = _test_runtimes_dir
