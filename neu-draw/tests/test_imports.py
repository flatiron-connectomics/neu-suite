"""What importing neu_draw may pull in: no renderer, no source backend, no notebook UI.

A renderer import on a headless machine fails outright (no glfw or Qt), so a stray
top-level import would break `import neu_draw` in CI and on cluster workers.
"""

import subprocess
import sys


def _import_in_a_fresh_interpreter(code: str) -> str:
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_importing_neu_draw_pulls_in_no_optional_dependency():
    """The render extra (pygfx), the sources and dvid extras (pandas, neuclease) and the
    notebook UI (ipywidgets, the toolbar) load only when used; neu_morpho never."""
    out = _import_in_a_fresh_interpreter("""
import sys
import neu_draw
banned = [m for m in ("pygfx", "wgpu", "rendercanvas", "neu_morpho", "pandas", "neuclease",
                      "ipywidgets", "imageio", "neu_draw.toolbar") if m in sys.modules]
assert not banned, f"importing neu_draw pulled in {banned}"
assert neu_draw.views == {}
print("clean")
""")
    assert out == "clean"


def test_the_shared_types_are_neu_libs_own_not_local_copies():
    """The top-level re-exports are aliases of neu_lib's types, never copies that could
    drift (on the half-open rule, or zyx)."""
    import neu_lib

    import neu_draw

    for name in ("BBox", "Frame", "Mesh", "Skeleton", "Vec3", "to_xyz"):
        assert getattr(neu_draw, name) is getattr(neu_lib, name), name
