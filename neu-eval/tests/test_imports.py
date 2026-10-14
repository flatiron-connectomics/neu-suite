"""What importing the package costs, and that its lazy exports resolve.

Run in subprocesses: this session's `sys.modules` already holds whatever other tests
imported, so only a fresh interpreter shows what `import neu_eval` pulls in.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from neu_eval import _EXPORTS, _SUBMODULES

#: Nothing here computes a metric, and all of it is expensive.
HEAVY = ["neu_vol", "tensorstore", "dask", "distributed", "blockrun", "zarr", "pandas"]


def _run(code: str) -> str:
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


@pytest.mark.parametrize("imports, banned", [
    ("neu_eval", HEAVY),
    # the pure metric modules: what the conda-free CI rests on
    ("neu_eval.overlap, neu_eval.voxel, neu_eval.assign", HEAVY),
    # scipy is a dependency, but only best_match(method="hungarian") needs it
    ("neu_eval", ["scipy"]),
    # the contingency kernel imports neu_proc when called, not at import
    ("neu_eval.overlap", ["neu_proc"]),
])
def test_importing_pulls_in_nothing_heavy(imports, banned):
    leaked = _run(f"import sys, {imports};"
                  f"print(','.join(m for m in {banned!r} if m in sys.modules))")
    assert leaked == ""


def test_the_two_export_mappings_stay_disjoint():
    """A name in both would silently resolve from whichever is checked first."""
    assert not set(_EXPORTS) & set(_SUBMODULES)


@pytest.mark.parametrize("name", sorted({*_EXPORTS, *_SUBMODULES}))
def test_every_declared_export_resolves(name):
    import neu_eval

    assert getattr(neu_eval, name) is not None


def test_an_undeclared_name_raises_attribute_error():
    import neu_eval

    with pytest.raises(AttributeError, match="has no attribute"):
        neu_eval.definitely_not_a_thing


def test_dir_lists_what_getattr_can_resolve():
    import neu_eval

    assert set(dir(neu_eval)) == set(neu_eval.__all__)
