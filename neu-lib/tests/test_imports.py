"""What importing this package is allowed to cost.

**numpy and nothing else.** That is the property the package exists for: every tier of
the suite names these types, so anything imported here lands under all of them. A stray
`import tensorstore` would be invisible in the shared conda environment, which has the
whole stack installed — it would only fail for someone who installed `neu-lib` alone,
which is exactly the case this package is meant to serve.

Asserted in a fresh interpreter rather than by reading the imports, because the
regression is one convenience import away and nothing else would notice.
"""

import subprocess
import sys


def _import_in_a_fresh_interpreter(code: str) -> str:
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def test_importing_neu_lib_pulls_in_nothing_but_numpy():
    """Measured as the DELTA, not the absolute set: an environment can inject namespace
    packages at startup (`sphinxcontrib` does, via a .pth), and blaming those on this
    import would make the test fail for reasons that have nothing to do with it."""
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
    """The layering, from below. Nothing here may know about a store or a renderer."""
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
    """`__all__` and the module agree — a name listed but not imported is an
    AttributeError for the caller and nothing catches it here otherwise."""
    import neu_lib

    for name in neu_lib.__all__:
        assert hasattr(neu_lib, name), name
