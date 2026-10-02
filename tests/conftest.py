"""Suite-wide settings.

``vnext-mcp --check`` names the exact model behind each alias by starting the
Claude CLI and a Codex app-server.  No test may start a provider, including a
check run in a subprocess, so the probe is switched off for the whole suite.
A test that covers the probe patches it directly.
"""

import os

os.environ["VNEXT_CHECK_SKIP_MODEL_PROBE"] = "1"

# No test reads PyPI, and none finds a developer's own side runtime: the update
# check is off, and side runtimes live in a folder only this run uses.
import tempfile

os.environ["VNEXT_NO_UPDATE_CHECK"] = "1"
os.environ["VNEXT_RUNTIMES_DIR"] = tempfile.mkdtemp(prefix="vnext-test-runtimes-")

# A test that clears the environment loses both settings above, and the server
# it builds would then read PyPI and the real ~/.vnext/runtimes.  The module is
# patched as well, so such a test still reads neither.
from pathlib import Path

import vnext.vnext_runtimes as _runtimes

_TEST_RUNTIMES_DIR = os.environ["VNEXT_RUNTIMES_DIR"]


def _no_pypi(name):
    raise OSError(f"the test suite never reads PyPI ({name})")


def _test_runtimes_dir():
    return Path(os.environ.get(_runtimes.RUNTIMES_ENV) or _TEST_RUNTIMES_DIR)


_runtimes._fetch_pypi = _no_pypi
_runtimes.runtimes_dir = _test_runtimes_dir
