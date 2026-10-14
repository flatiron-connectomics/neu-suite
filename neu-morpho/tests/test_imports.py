"""The package imports, its lazy top level resolves, and `measure` stays lazy about pandas."""


def test_package_modules_import():
    import neu_morpho
    from neu_morpho import config, mesh, precomputed, occupancy, fragments, allowlist
    from neu_morpho.ops import meshify
    assert neu_morpho.__version__
    assert callable(meshify)


def test_no_module_loads_a_gpl_package():
    """The package is Apache-2.0: importing any of it must not load the GPL tracers'
    dependencies, which live in neu-skel. Checked in a fresh interpreter, with both
    blocked so that even a lazy import inside a module body would fail."""
    import subprocess
    import sys

    out = subprocess.run([sys.executable, "-c", """
import importlib, pkgutil, sys
sys.modules['kimimaro'] = None
sys.modules['dijkstra3d'] = None
import neu_morpho
bad = []
for m in pkgutil.walk_packages(neu_morpho.__path__, 'neu_morpho.'):
    if m.name.endswith('__main__'):
        continue
    try:
        importlib.import_module(m.name)
    except ImportError as e:
        if 'kimimaro' in str(e) or 'dijkstra3d' in str(e):
            bad.append(m.name)
print(bad)
"""], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"


def test_measure_does_not_cost_pandas_at_import():
    """pandas and pyarrow are the optional `measure` extra, so importing `measure` must
    not load them. Checked in a fresh interpreter."""
    import subprocess
    import sys

    out = subprocess.run([sys.executable, "-c", """
import sys
before = set(sys.modules)
import neu_morpho, neu_morpho.measure
added = {m.split('.')[0] for m in set(sys.modules) - before}
assert 'pandas' not in added, 'importing neu_morpho.measure pulled in pandas'
assert 'pyarrow' not in added, 'importing neu_morpho.measure pulled in pyarrow'
print('clean')
"""], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "clean"


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


def test_every_top_level_export_is_the_defining_modules_own_object():
    import importlib

    import neu_morpho

    assert not set(neu_morpho._EXPORTS) & set(neu_morpho._SUBMODULES)
    assert set(neu_morpho.__all__) == {"__version__", *neu_morpho._EXPORTS,
                                       *neu_morpho._SUBMODULES}
    for name, module in neu_morpho._EXPORTS.items():
        assert getattr(neu_morpho, name) is getattr(
            importlib.import_module(f"neu_morpho.{module}"), name)
    for name in neu_morpho._SUBMODULES:
        assert getattr(neu_morpho, name).__name__ == f"neu_morpho.{name}"


def test_editors_can_see_every_lazy_export():
    """The `TYPE_CHECKING` block lists the same names as the lazy lookup tables, so a
    name added to one and not the other cannot become invisible to editors."""
    import neu_morpho

    assert _type_checking_names(neu_morpho) == {*neu_morpho._EXPORTS, *neu_morpho._SUBMODULES}


def test_importing_the_package_loads_none_of_its_exports():
    """The CLI imports the package for `__version__`; the exports load on first use."""
    import subprocess
    import sys

    out = subprocess.run([sys.executable, "-c",
                          "import sys, neu_morpho; print(','.join(m for m in "
                          "('neu_morpho.readback', 'neu_morpho.measure', 'neu_morpho.metrics_db')"
                          " if m in sys.modules))"], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ""

