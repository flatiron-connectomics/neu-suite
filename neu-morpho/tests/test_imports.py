"""The package imports, and `measure` stays lazy about pandas."""


def test_package_modules_import():
    import neu_morpho
    from neu_morpho import config, mesh, precomputed, skeleton, occupancy, fragments, allowlist
    from neu_morpho.ops import meshify
    assert neu_morpho.__version__
    assert callable(meshify)


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
