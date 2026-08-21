"""Pure geometry: nm, zyx, no I/O and no renderer.

Nothing in this subpackage reads a store, opens a canvas, or imports pygfx. That is what
makes it testable on a headless box with nothing installed but numpy. Keep it that way:
a ``.render`` method on a geometry class is how viewer knowledge ends up inside the data
model.
"""

from neu_vol import BBox

from .frame import Frame, to_xyz
from .mesh import Mesh
from .skeleton import Skeleton

__all__ = ["BBox", "Frame", "Mesh", "Skeleton", "to_xyz"]
