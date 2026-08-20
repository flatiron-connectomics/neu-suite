"""Pure geometry: nm, zyx, no I/O and no renderer.

Nothing in this subpackage reads a store, opens a canvas, or imports pygfx. That is what
makes it testable on a headless box with nothing installed but numpy, and it is the line
the predecessor crossed — its mask, mesh and skeleton classes each carried a ``.render``
that imported fastplotlib, which is how viewer knowledge ended up inside the data model.
"""

from em_volume_tools import BBox

from .frame import Frame, to_xyz
from .mesh import Mesh
from .skeleton import Skeleton

__all__ = ["BBox", "Frame", "Mesh", "Skeleton", "to_xyz"]
