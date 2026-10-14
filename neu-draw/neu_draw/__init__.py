"""neu-draw: interactive 3D rendering of EM segment morphology in Jupyter.

Meshes, skeletons, synapse points and ROI meshes, drawn locally with pygfx. neu-glance,
by contrast, builds states for a remote neuroglancer viewer.

* ``neu_lib`` types (``BBox``, ``Frame``, ``Mesh``, ``Skeleton``): re-exported here on
  purpose, as the same objects, so a notebook can write ``neu_draw.Skeleton``.
* ``scene``: drawables, colours and camera intent. Pure.
* ``layout``: placing several objects side by side, as boxes in and offsets out. Pure.
* ``sources``: the only module that reads data.
* ``viewstate``: saved camera viewpoints, which outlive any one view. Pure.
* ``backends``: the pygfx renderer and the legend drawn inside the canvas.
* ``toolbar``: ipywidgets buttons above the canvas, loaded only when a view is shown.

The renderer is imported only by :func:`show`, never at package import: it needs a canvas
backend, which fails outright on a headless machine. ``import neu_draw`` is safe anywhere.
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
                    Scene, build_scene, label_of)
from .sources import (body_mesh, body_meshes, body_skeleton, body_skeletons,
                      synapse_points, volume_frame)


def show(scene: Scene, *, backend: str = "pygfx", **kwargs) -> Any:
    """Render a :class:`~neu_draw.scene.Scene` and return the backend's view.

    The renderer is imported here, on first use. In a notebook the returned view displays
    itself; elsewhere it renders offscreen and ``.snapshot()`` returns the pixels.
    """
    from .backends import get_backend

    return get_backend(backend).show(scene, **kwargs)


__all__ = [
    "__version__",
    "BBox", "Frame", "Mesh", "Skeleton", "Vec3", "to_xyz",
    "box_predicate", "mask_predicate", "skeleton_tube", "union",
    "Scene", "build_scene", "MeshDrawable", "LinesDrawable", "PointsDrawable",
    "Camera", "Legend", "label_of",
    "assign_colors", "to_rgba",
    "ViewState", "views", "viewstate",
    "cache", "layout", "logs", "sources",
    "quiet_stores", "install_quiet_stores", "remove_quiet_stores",
    "body_mesh", "body_meshes", "body_skeleton", "body_skeletons",
    "volume_frame", "synapse_points",
    "show",
]
