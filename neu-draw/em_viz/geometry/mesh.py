"""A triangle mesh in nm, zyx.

Deliberately plain arrays rather than a ``vol2mesh.Mesh`` wrapper. The old
``VolumeMesh`` wrapped one, which meant every geometric operation went through a
conda-only package that exists to *build* meshes from label volumes — work a viewer
never does. Holding the two arrays directly is what keeps this module importable, and
testable, with nothing but numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
from em_volume_tools import BBox


@dataclass
class Mesh:
    """Vertices in nm (zyx) and the triangles indexing them.

    ``normals_zyx`` is optional: pygfx computes flat shading without it, and a mesh read
    back from precomputed carries none.
    """

    vertices_zyx_nm: np.ndarray          # (N, 3) float32
    faces: np.ndarray                    # (M, 3) int32
    normals_zyx: Optional[np.ndarray] = None
    name: Optional[str] = None

    def __post_init__(self) -> None:
        self.vertices_zyx_nm = _as_points(self.vertices_zyx_nm, "vertices_zyx_nm")
        faces = np.ascontiguousarray(self.faces, dtype=np.int32).reshape(-1, 3)
        if len(faces) and (faces.min() < 0 or faces.max() >= len(self.vertices_zyx_nm)):
            raise ValueError(
                f"face indices out of range for {len(self.vertices_zyx_nm)} vertices "
                f"(saw {faces.min()}..{faces.max()})")
        self.faces = faces
        if self.normals_zyx is not None:
            normals = _as_points(self.normals_zyx, "normals_zyx")
            if len(normals) != len(self.vertices_zyx_nm):
                raise ValueError(
                    f"{len(normals)} normals for {len(self.vertices_zyx_nm)} vertices")
            self.normals_zyx = normals

    @classmethod
    def from_precomputed(cls, vertices_xyz_nm: Any, faces: Any,
                         name: Optional[str] = None) -> "Mesh":
        """From what ``em_seg_morpho.readback.read_body_mesh`` returns.

        That reader hands back **xyz** — the order the precomputed format stores — so
        this is where the one flip into the package's zyx convention happens.
        """
        verts = _as_points(vertices_xyz_nm, "vertices_xyz_nm")
        return cls(np.ascontiguousarray(verts[:, ::-1]), faces, name=name)

    # -- geometry --------------------------------------------------------------

    @property
    def bbox(self) -> BBox:
        """Integer nm bounds. Empty mesh gives an empty box rather than raising."""
        if not len(self.vertices_zyx_nm):
            return BBox.empty(3)
        return BBox.from_points(self.vertices_zyx_nm)

    def translate(self, offset_zyx_nm: Sequence[float]) -> "Mesh":
        """Move the mesh. **Only the vertices move.**

        Faces are indices and normals are directions; the old implementation added the
        offset to all three, which silently scrambled the topology and the shading.
        """
        offset = np.asarray(offset_zyx_nm, dtype=np.float64)
        if offset.shape != (3,):
            raise ValueError(f"offset must be zyx, got shape {offset.shape}")
        return Mesh(self.vertices_zyx_nm + offset, self.faces,
                    None if self.normals_zyx is None else self.normals_zyx.copy(),
                    self.name)

    def __repr__(self) -> str:
        return (f"Mesh(name={self.name!r}, vertices={len(self.vertices_zyx_nm)}, "
                f"faces={len(self.faces)}, normals={self.normals_zyx is not None})")


def _as_points(value: Any, label: str) -> np.ndarray:
    arr = np.ascontiguousarray(value, dtype=np.float32)
    if arr.size == 0:
        return arr.reshape(0, 3)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{label} must be (N, 3), got shape {arr.shape}")
    return arr
