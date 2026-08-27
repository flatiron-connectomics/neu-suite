"""neu-draw: interactive 3D rendering of EM segment morphology in Jupyter.

Meshes, skeletons and synapse points — plus ROI meshes — drawn locally with pygfx. The
sibling neu-glance builds neuroglancer states for a *remote* viewer; the axis that separates
the two is where the rendering happens, not what is rendered.

Layout, and the reason for it:

* ``neu_lib`` — ``BBox``, ``Frame``, ``Mesh``, ``Skeleton``: nm, zyx, arrays, no I/O
  and no renderer. Shared with the rest of the suite, and re-exported here because a
  notebook wants ``neu_draw.Skeleton``. Aliases, never a fork — a test pins that.
* ``scene`` — drawables, colours and camera intent. Pure.
* ``layout`` — where a set of objects is placed, as boxes in and offsets out. Pure, and
  independent of ``scene`` on purpose: the arithmetic of a row or a grid is testable
  without building one.
* ``sources`` — the only module that reads anything.
* ``viewstate`` — saved camera viewpoints, kept outside any one view so they outlive it.
  Pure.
* ``backends`` — pygfx. The renderer seam, plus the clickable legend, which is drawn
  **in** the canvas so that it is part of every snapshot.
* ``toolbar`` — ipywidgets buttons above the canvas. Reached only when a view is shown,
  so nothing here pays for a notebook front end.

The geometry types import eagerly: they cost a numpy import and nothing else. The
**backend does not**, and must not — importing a renderer needs a canvas backend, and on
a headless machine that fails outright rather than degrading. So `import neu_draw` stays
safe in a terminal, in CI, and on a worker.
"""

__version__ = "0.1.0"

from typing import Any

from . import cache, layout, logs, sources, viewstate
from .colors import assign_colors, to_rgba
from .logs import install_quiet_stores, quiet_stores, remove_quiet_stores
from .viewstate import ViewState, views
from neu_lib import (BBox, Frame, Mesh, Skeleton, Vec3, box_predicate,
                     mask_predicate, skeleton_tube, to_xyz, union)
from .scene import (Camera, Legend, LinesDrawable, MeshDrawable, PointsDrawable,
                    Scene, build_scene)
from .sources import (body_mesh, body_meshes, body_skeleton, body_skeletons,
                      synapse_points, volume_frame)


def show(scene: Scene, *, backend: str = "pygfx", **kwargs) -> Any:
    """Render a :class:`~neu_draw.scene.Scene` and return the backend's view.

    A **function**, not a re-export, so the renderer is imported only when something is
    actually drawn — see the module docstring. In a notebook the returned view displays
    itself; elsewhere it renders offscreen and ``.snapshot()`` gives you the pixels.
    """
    from .backends import get_backend

    return get_backend(backend).show(scene, **kwargs)


__all__ = [
    "__version__",
    "BBox", "Frame", "Mesh", "Skeleton", "Vec3", "to_xyz",
    "box_predicate", "mask_predicate", "skeleton_tube", "union",
    "Scene", "build_scene", "MeshDrawable", "LinesDrawable", "PointsDrawable",
    "Camera", "Legend",
    "assign_colors", "to_rgba",
    "ViewState", "views", "viewstate",
    "cache", "layout", "logs", "sources",
    "quiet_stores", "install_quiet_stores", "remove_quiet_stores",
    "body_mesh", "body_meshes", "body_skeleton", "body_skeletons",
    "volume_frame", "skeleton_tube", "synapse_points",
    "box_predicate", "mask_predicate",
    "show",
]
