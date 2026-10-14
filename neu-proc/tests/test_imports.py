"""What importing this package is allowed to cost, and what its top level resolves to.

`ops/` is the pure half (no store, no dask, no blockrun), and `import neu_proc` loads no
algorithm library until a name is used. An import graph regresses silently, so both are
pinned.
"""

import importlib
import subprocess
import sys

import pytest


def _modules_after(code: str) -> set[str]:
    """Top-level modules in `sys.modules` after running `code` in a fresh interpreter."""
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
        assert forbidden not in loaded, f"`import neu_proc.ops` pulled in {forbidden}"


def test_importing_the_package_pays_for_no_algorithm_library():
    loaded = _modules_after("import neu_proc")
    for heavy in ("scipy", "skimage", "cc3d", "fastmorph", "edt", "fill_voids", "dask"):
        assert heavy not in loaded, f"`import neu_proc` pulled in {heavy}"


def test_kernels_needs_no_scipy():
    loaded = _modules_after("import neu_proc.ops.kernels")
    assert "scipy" not in loaded and "skimage" not in loaded


def test_every_export_resolves():
    """Both mappings resolve, are disjoint (a shared key would shadow one silently), and
    match `__all__`."""
    import neu_proc

    for name in (*neu_proc._EXPORTS, *neu_proc._SUBMODULES):
        assert getattr(neu_proc, name) is not None
    assert not set(neu_proc._EXPORTS) & set(neu_proc._SUBMODULES)
    assert set(neu_proc.__all__) == {"__version__", *neu_proc._EXPORTS,
                                     *neu_proc._SUBMODULES}


@pytest.mark.parametrize("name", ["dilate", "dust", "offset", "opening"])
def test_a_top_level_export_is_the_function_itself(name):
    import neu_proc

    module = importlib.import_module(f"neu_proc.{neu_proc._EXPORTS[name]}")
    assert getattr(neu_proc, name) is getattr(module, name)


def test_a_submodule_export_is_the_module_itself():
    """`measure` lives under `ops/`, so `from neu_proc import measure` is `__getattr__`'s
    doing."""
    import types

    from neu_proc import measure

    import neu_proc.ops.measure

    assert isinstance(measure, types.ModuleType)
    assert measure is neu_proc.ops.measure


def test_an_unknown_top_level_name_still_raises_AttributeError():
    """An ImportError leaking out instead would break `hasattr`."""
    import neu_proc

    with pytest.raises(AttributeError, match="'nope'"):
        neu_proc.nope


def _type_checking_names(module) -> set[str]:
    """Names imported in the module's `if TYPE_CHECKING:` block: what editors and type
    checkers see, since they cannot run `__getattr__`."""
    import ast
    import inspect

    names: set[str] = set()
    for node in ast.parse(inspect.getsource(module)).body:
        if isinstance(node, ast.If) and getattr(node.test, "id", None) == "TYPE_CHECKING":
            for stmt in ast.walk(node):
                if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                    names |= {(a.asname or a.name).split(".")[0] for a in stmt.names}
    return names


def test_editors_can_see_every_lazy_export():
    """The `TYPE_CHECKING` block lists the same names as the lazy lookup tables, so a
    name added to one and not the other cannot become invisible to editors."""
    import neu_proc

    assert _type_checking_names(neu_proc) == {*neu_proc._EXPORTS, *neu_proc._SUBMODULES}
