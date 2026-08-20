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

from .geometry import BBox, Frame, Mesh, Skeleton, to_xyz

__all__ = ["__version__", "BBox", "Frame", "Mesh", "Skeleton", "to_xyz"]
