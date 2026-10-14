"""neu-lib: the vocabulary the neu-* suite shares.

Boxes and grid arithmetic, the voxel-to-nanometre frame, arrays placed in that frame,
pyramid levels, meshes and skeletons. **Nothing here reads a store, opens a canvas or
imports anything but numpy**, so every package in the suite can use these types without
pulling in an I/O stack. Anything that opens a store belongs in neu-vol.

Two conventions run through all of it, and both fail silently when broken:

* **zyx in memory, physical nanometres as the one model space.** Nothing derives a voxel
  size from ``2 ** level``, because real pyramids are anisotropic.
  :func:`~neu_lib.frame.to_xyz` is the single conversion to a renderer's xyz order.
* **Boxes are half-open**, ``[lo, hi)``.
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
