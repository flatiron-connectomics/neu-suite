"""What importing this package is allowed to cost.

The renderer seam only works if it is real. `import em_viz` happens in terminals, in CI
and on cluster workers, none of which have a display — and importing a renderer there
does not degrade, it raises: fastplotlib on a headless box fails outright for want of a
glfw or Qt backend, and that is the failure this split exists to prevent.

Asserting it here rather than trusting the layout, because the regression is one stray
top-level import away and nothing else would notice.
"""

import subprocess
import sys


def _import_in_a_fresh_interpreter(code: str) -> str:
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_importing_em_viz_pulls_in_no_renderer():
    out = _import_in_a_fresh_interpreter("""
import sys
import em_viz
banned = [m for m in ("pygfx", "wgpu", "rendercanvas", "fastplotlib")
          if m in sys.modules]
assert not banned, f"importing em_viz pulled in {banned}"
print("clean")
""")
    assert out == "clean"


def test_importing_em_viz_pulls_in_no_source_backend():
    """`em-seg-morpho` and pandas belong to the `sources` extra. Geometry needs neither,
    and dragging them in would make a scene-building import pay for vol2mesh."""
    out = _import_in_a_fresh_interpreter("""
import sys
import em_viz
banned = [m for m in ("em_seg_morpho", "pandas", "neuclease") if m in sys.modules]
assert not banned, f"importing em_viz pulled in {banned}"
print("clean")
""")
    assert out == "clean"


def test_the_geometry_types_are_reachable_from_the_top_level():
    out = _import_in_a_fresh_interpreter("""
import em_viz
for name in ("BBox", "Frame", "Mesh", "Skeleton", "to_xyz"):
    assert hasattr(em_viz, name), name
print(em_viz.__version__)
""")
    assert out == "0.1.0"


def test_bbox_is_the_shared_one_not_a_local_copy():
    """One notion of a box across the suite. A second would diverge on the half-open
    rule, which is exactly the kind of disagreement nothing surfaces."""
    from em_volume_tools.grid import BBox as shared

    from em_viz.geometry import BBox

    assert BBox is shared
