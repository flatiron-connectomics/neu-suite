"""What this package costs to import, and what its lazy exports promise.

Subprocess-isolated, because `sys.modules` in a test session has already been polluted by
every other test file — a check on the in-process module table proves nothing about what a
fresh `import neu_eval` does. The same reason `neu-proc`'s version of this file runs out of
process.

The claim being defended: the metric modules compute and do not read, so importing them
must not pull in a store stack. It is what keeps the core testable in a conda-free CI, and
it is the kind of property that decays silently the first time somebody needs a path helper
and reaches for the nearest one.
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


def test_importing_the_package_pulls_in_nothing_heavy():
    leaked = _run(
        "import sys, neu_eval;"
        f"print(','.join(m for m in {HEAVY!r} if m in sys.modules))")
    assert leaked == ""


def test_importing_the_metric_modules_pulls_in_nothing_heavy():
    """The whole pure half, at once — this is the property the conda-free CI rests on."""
    leaked = _run(
        "import sys;"
        "import neu_eval.overlap, neu_eval.voxel, neu_eval.assign;"
        f"print(','.join(m for m in {HEAVY!r} if m in sys.modules))")
    assert leaked == ""


def test_importing_the_package_does_not_pay_for_scipy_either():
    """scipy is a real dependency, but only `best_match(method='hungarian')` needs it."""
    leaked = _run("import sys, neu_eval; print('scipy' in sys.modules)")
    assert leaked == "False"


def test_the_contingency_kernel_resolves_its_backend_at_call_time():
    """`neu_proc` is imported by the function, not by the module.

    So `neu_eval.overlap` stays importable and readable without the label-tool stack, which
    is what lets a metric be reviewed without one.
    """
    leaked = _run("import sys, neu_eval.overlap; print('neu_proc' in sys.modules)")
    assert leaked == "False"


def test_the_two_export_mappings_stay_disjoint():
    """A name in both would be resolved by whichever is checked first, silently."""
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
