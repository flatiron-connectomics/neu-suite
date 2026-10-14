"""Importing neu-lib may pull in numpy and the standard library, nothing else. Checked in a
fresh interpreter, since the shared environment has the whole stack installed."""

import subprocess
import sys


def _import_in_a_fresh_interpreter(code: str) -> str:
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def test_importing_neu_lib_pulls_in_nothing_but_numpy():
    """Measured as the delta, since a .pth can inject packages at startup."""
    out = _import_in_a_fresh_interpreter("""
import sys
before = set(sys.modules)
import neu_lib
added = {m.split('.')[0] for m in set(sys.modules) - before}
allowed = set(sys.stdlib_module_names) | {'numpy', 'neu_lib'}
extra = sorted(added - allowed)
assert not extra, f'neu_lib pulled in {extra}'
print('clean')
""")
    assert out == "clean"


def test_it_does_not_import_anything_from_the_suite():
    out = _import_in_a_fresh_interpreter("""
import sys
import neu_lib
banned = [m for m in ('neu_vol', 'neu_draw', 'neu_morpho', 'blockrun',
                      'tensorstore', 'dask', 'pandas', 'pygfx')
          if m in sys.modules]
assert not banned, f'neu_lib pulled in {banned}'
print('clean')
""")
    assert out == "clean"


def test_every_exported_name_is_reachable():
    import neu_lib

    for name in neu_lib.__all__:
        assert hasattr(neu_lib, name), name
