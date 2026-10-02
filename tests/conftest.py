"""Suite-wide settings: see ``suite_environment.py``.

pytest loads this file by itself; ``python -m unittest`` does not, so the
settings live in a module both runners import.
"""

import suite_environment  # noqa: F401
