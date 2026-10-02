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

import vnext.vnext_runtimes as _runtimes

os.environ["VNEXT_CHECK_SKIP_MODEL_PROBE"] = "1"

# No test reads PyPI, and none finds a developer's own side runtime: the update
# check is off, and side runtimes live in a folder only this run uses.
os.environ["VNEXT_NO_UPDATE_CHECK"] = "1"
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
