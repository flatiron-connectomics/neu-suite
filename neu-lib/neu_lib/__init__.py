"""neu-lib: the vocabulary the neu-* suite shares.

Boxes and grid arithmetic, the voxel-to-nanometre frame, and the two geometry types
meshes and skeletons are carried in. **Nothing here reads a store, opens a canvas or
imports anything but numpy** — that is what lets every tier of the suite name these
types without inheriting an I/O stack, and it is the line to hold: if something needs
to open a store it belongs in neu-vol.

Two conventions run through all of it, and both fail silently when broken:

* **zyx in memory, and physical nanometres as the one model space.** A geometry
  object never carries an integer scale, and never derives a factor from
  ``2 ** level`` — real pyramids are anisotropic. :func:`~neu_lib.frame.to_xyz` is
  the single conversion to the renderer's order.
* **Boxes are half-open**, ``[lo, hi)``. A ``hi`` already on a block boundary must
  not move, which is the off-by-one behind ``align_box(..., "outer")`` and behind
  :meth:`BBox.from_points` adding one to its maximum.
"""

from .frame import Frame, to_xyz
from .grid import MODES, BBox, align_box, clamp_box, lcm_grid, misaligned_axes
from .mesh import Mesh
from .predicates import box_predicate, mask_predicate, union
from .piece import KINDS, Piece
from .scale import ScaleInfo
from .skeleton import InsideFn, Skeleton
from .tube import frustum_mesh, skeleton_tube
from .vec import Vec3

__version__ = "0.1.0"

__all__ = [
    "BBox",
    "Frame",
    "InsideFn",
    "MODES",
    "Mesh",
    "KINDS",
    "Piece",
    "ScaleInfo",
    "Skeleton",
    "Vec3",
    "align_box",
    "box_predicate",
    "clamp_box",
    "frustum_mesh",
    "lcm_grid",
    "mask_predicate",
    "misaligned_axes",
    "skeleton_tube",
    "to_xyz",
    "union",
]
