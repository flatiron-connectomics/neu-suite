"""Stop a whole-repository pytest run before it collects anything.

The nine suites are run one package at a time (``make test``, or ``pytest`` inside a
package directory), because each has its own pytest settings and ``tests/conftest.py``,
and test files share names across packages. Run from here, pytest would report dozens
of import errors that say nothing about the code. This file is only loaded when pytest's
rootdir is the repository root; a run inside a package never sees it.
"""

import pytest


def pytest_configure(config):
    raise pytest.UsageError(
        "run the suites one package at a time: `make test` from the repository root, "
        "`make test-neu-vol` for one package, or `pytest` inside a package directory")
