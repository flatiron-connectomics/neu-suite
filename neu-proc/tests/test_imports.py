"""What importing this package is allowed to cost.

`ops/` is the pure half, and "pure" here has a precise, checkable meaning: no store, no
dask, no blockrun. The guard matters because the layer's whole value is being usable and
testable on its own — the moment `import neu_proc.ops` drags `neu_vol` in, the array
functions can no longer be read, run or reasoned about without an I/O stack, and the
distributed machinery stops being a thing built *on top of* them.

Same shape as the guards neu-lib, neu-mark and `neu_morpho.measure` already carry, and for
the same reason: an import graph regresses silently, since everything still works.
"""

import subprocess
import sys


def _modules_after(code: str) -> set[str]:
    """The top-level modules in `sys.modules` after running `code` in a fresh interpreter.

    A subprocess because this test's own imports would otherwise pollute the answer — the
    suite has certainly imported numpy and probably scipy by now.
    """
    out = subprocess.run(
        [sys.executable, "-c",
         code + "\nimport sys; print(' '.join(sorted({m.split('.')[0] "
                "for m in sys.modules})))"],
        capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return set(out.stdout.split())


def test_importing_ops_pulls_in_no_store_and_no_dask():
    loaded = _modules_after("import neu_proc.ops")
    for forbidden in ("neu_vol", "tensorstore", "dask", "distributed", "blockrun", "zarr"):
        assert forbidden not in loaded, (
            f"`import neu_proc.ops` pulled in {forbidden}, which makes the pure layer "
            f"untestable on its own")


def test_importing_the_package_pays_for_no_algorithm_library():
    """The lazy `__getattr__` is what buys this, and a future `--help` is what spends it."""
    loaded = _modules_after("import neu_proc")
    for heavy in ("scipy", "skimage", "cc3d", "fastmorph", "edt", "fill_voids", "dask"):
        assert heavy not in loaded, f"`import neu_proc` pulled in {heavy}"


def test_kernels_needs_no_scipy():
    """Converting a length is arithmetic and a frame. Keeping it scipy-free is what lets the
    conversion be tested without the filters, and reasoned about beside the halo it
    produces."""
    loaded = _modules_after("import neu_proc.ops.kernels")
    assert "scipy" not in loaded and "skimage" not in loaded


def test_every_export_resolves():
    """A name in `_EXPORTS` pointing at a module that does not define it fails only when
    someone reaches for it, which may be much later. Same for a `_SUBMODULES` entry naming
    a module that is not there."""
    import neu_proc

    for name in (*neu_proc._EXPORTS, *neu_proc._SUBMODULES):
        assert getattr(neu_proc, name) is not None
    assert set(neu_proc.__all__) == {"__version__", *neu_proc._EXPORTS,
                                     *neu_proc._SUBMODULES}


def test_the_two_export_mappings_stay_disjoint():
    """They resolve differently — a name in `_EXPORTS` is looked up *inside* its module,
    one in `_SUBMODULES` *is* the module — so a key in both means whichever `__getattr__`
    checks first silently wins, and the other entry is dead with nothing to say so."""
    import neu_proc

    assert not set(neu_proc._EXPORTS) & set(neu_proc._SUBMODULES)


def test_a_submodule_export_is_the_module_itself():
    """`from neu_proc import measure` cannot fall back to the import system's own submodule
    lookup, which would go looking for `neu_proc/measure.py` — the module lives under
    `ops/`. So this is entirely `__getattr__`'s doing, and worth pinning."""
    import types

    from neu_proc import measure

    import neu_proc.ops.measure

    assert isinstance(measure, types.ModuleType)
    assert measure is neu_proc.ops.measure, "the same object, not a copy"


def test_an_unknown_top_level_name_still_raises_AttributeError():
    """`__getattr__` grew a branch; the miss must stay a miss. An ImportError leaking out of
    here instead would break `hasattr` and every `from neu_proc import *`."""
    import neu_proc
    import pytest

    with pytest.raises(AttributeError, match="has no attribute 'nope'"):
        neu_proc.nope
