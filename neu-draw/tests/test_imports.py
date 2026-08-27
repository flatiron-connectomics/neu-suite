"""What importing this package is allowed to cost.

The renderer seam only works if it is real. `import neu_draw` happens in terminals, in CI
and on cluster workers, none of which have a display — and importing a renderer there
does not degrade, it raises: on a headless box it fails outright for want of a glfw or Qt
backend, and that is the failure this split exists to prevent.

Asserting it here rather than trusting the layout, because the regression is one stray
top-level import away and nothing else would notice.
"""

import subprocess
import sys


def _import_in_a_fresh_interpreter(code: str) -> str:
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_importing_neu_draw_pulls_in_no_renderer():
    out = _import_in_a_fresh_interpreter("""
import sys
import neu_draw
banned = [m for m in ("pygfx", "wgpu", "rendercanvas")
          if m in sys.modules]
assert not banned, f"importing neu_draw pulled in {banned}"
print("clean")
""")
    assert out == "clean"


def test_importing_neu_draw_pulls_in_no_source_backend():
    """`neu-morpho` and pandas belong to the `sources` extra. Geometry needs neither,
    and dragging them in would make a scene-building import pay for vol2mesh."""
    out = _import_in_a_fresh_interpreter("""
import sys
import neu_draw
banned = [m for m in ("neu_morpho", "pandas", "neuclease") if m in sys.modules]
assert not banned, f"importing neu_draw pulled in {banned}"
print("clean")
""")
    assert out == "clean"


def test_the_geometry_types_are_reachable_from_the_top_level():
    out = _import_in_a_fresh_interpreter("""
import neu_draw
for name in ("BBox", "Frame", "Mesh", "Skeleton", "to_xyz"):
    assert hasattr(neu_draw, name), name
print(neu_draw.__version__)
""")
    assert out == "0.1.0"


def test_the_shared_types_are_neu_libs_own_not_local_copies():
    """One notion of each across the suite. A second would diverge on the half-open
    rule, or on zyx, which is exactly the kind of disagreement nothing surfaces.

    What this package re-exports at its top level is a convenience for notebooks, and
    it must stay an alias rather than becoming a fork.
    """
    import neu_lib

    import neu_draw

    for name in ("BBox", "Frame", "Mesh", "Skeleton", "Vec3", "to_xyz"):
        assert getattr(neu_draw, name) is getattr(neu_lib, name), name
