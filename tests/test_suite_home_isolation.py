"""The suite runs with a home folder of its own, never the developer's."""

import os
import tempfile
import unittest
from pathlib import Path

import suite_environment  # noqa: F401


def _real_home() -> Path:
    """The account's home folder, read from where the suite does not rewrite it."""

    if os.name == "nt":
        # The suite rewrites HOME and USERPROFILE; HOMEDRIVE and HOMEPATH stay.
        return Path(os.environ["HOMEDRIVE"] + os.environ["HOMEPATH"])
    import pwd

    return Path(pwd.getpwuid(os.getuid()).pw_dir)


REAL_HOME = _real_home().resolve()
TEMP_ROOT = Path(tempfile.gettempdir()).resolve()


class SuiteHomeIsolationTest(unittest.TestCase):
    def test_home_is_not_the_real_home(self) -> None:
        self.assertNotEqual(Path.home().resolve(), REAL_HOME)
        self.assertNotEqual(Path(os.environ["HOME"]).resolve(), REAL_HOME)

    def test_tilde_expands_inside_the_temp_folder(self) -> None:
        expanded = Path(os.path.expanduser("~/.vnext")).resolve()
        self.assertTrue(expanded.is_relative_to(TEMP_ROOT), expanded)
        # On Windows the temp folder lies inside the real home, so the check is
        # against the real ~/.vnext itself.
        self.assertNotEqual(expanded, REAL_HOME / ".vnext")


if __name__ == "__main__":
    unittest.main()
