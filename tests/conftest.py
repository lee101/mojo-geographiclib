import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import pytest  # noqa: E402

from mojogeographiclib import _lib  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _build():
    """Compile src/capi.mojo once, so the tests can never run against a
    stale or missing shared library."""
    _lib.build()
