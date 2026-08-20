"""em-viz: interactive 3D rendering of EM segment morphology in Jupyter.

Meshes, skeletons and synapse points — plus ROI meshes — drawn locally with pygfx. The
sibling em-ngl builds neuroglancer states for a *remote* viewer; the axis that separates
the two is where the rendering happens, not what is rendered.

Layout, and the reason for it:

* ``geometry`` — nm, zyx, arrays. No I/O, no renderer.
* ``scene`` — drawables, colours and camera intent. Pure.
* ``sources`` — the only module that reads anything.
* ``backends`` — pygfx. The renderer seam.

``geometry`` imports eagerly: it costs a numpy import and nothing else. The **backend
does not**, and must not — importing a renderer needs a canvas backend, and on a
headless machine that fails outright rather than degrading. So `import em_viz` stays
safe in a terminal, in CI, and on a worker.
"""

__version__ = "0.1.0"

from typing import Any

from . import cache, sources
from .colors import assign_colors, to_rgba
from .geometry import BBox, Frame, Mesh, Skeleton, to_xyz
from .scene import (LinesDrawable, MeshDrawable, PointsDrawable, Scene,
                    build_scene)
from .sources import (body_mesh, body_meshes, body_skeleton, body_skeletons,
                      box_predicate, mask_predicate, skeleton_tube,
                      synapse_points, volume_frame)


def show(scene: Scene, *, backend: str = "pygfx", **kwargs) -> Any:
    """Render a :class:`~em_viz.scene.Scene` and return the backend's view.

    A **function**, not a re-export, so the renderer is imported only when something is
    actually drawn — see the module docstring. In a notebook the returned view displays
    itself; elsewhere it renders offscreen and ``.snapshot()`` gives you the pixels.
    """
    from .backends import get_backend

    return get_backend(backend).show(scene, **kwargs)


__all__ = [
    "__version__",
    "BBox", "Frame", "Mesh", "Skeleton", "to_xyz",
    "Scene", "build_scene", "MeshDrawable", "LinesDrawable", "PointsDrawable",
    "assign_colors", "to_rgba",
    "cache", "sources",
    "body_mesh", "body_meshes", "body_skeleton", "body_skeletons",
    "volume_frame", "skeleton_tube", "synapse_points",
    "box_predicate", "mask_predicate",
    "show",
]
