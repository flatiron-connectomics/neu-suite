"""The coordinate contract that keeps meshes and skeletons aligned.

Both live in one model space, physical nanometres (full-resolution world coords, zyx),
with **identity** neuroglancer transforms. Every scale is expressed through its own voxel
size (nm), never an assumed ``2**scale``, so anisotropic pyramids stay aligned.

- Mesh: ``Mesh.from_label_volume(block, physical_box(region, voxel_size))`` yields nm.
- Skeleton: a tracer run with ``anisotropy = voxel_size`` returns nm local to the crop;
  :func:`skeleton_to_physical` adds the crop origin, the step whose omission offsets
  skeletons from their meshes.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def physical_box(region: Sequence[slice], voxel_size_zyx: Sequence[float],
                 origin_nm_zyx: Sequence[float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    """Physical (nm) bounding box (2x3, zyx) for a block region at a scale.

    ``region`` is in that scale's array indices; ``voxel_size_zyx`` is its voxel size
    (nm) and ``origin_nm_zyx`` where its index 0 lies (the level's ``Frame``), so the box
    lands in the same nm world whatever the scale or the volume's offset.
    """
    vz, vy, vx = (float(v) for v in voxel_size_zyx)
    (z0, z1), (y0, y1), (x0, x1) = [(s.start, s.stop) for s in region]
    box = np.array([[z0 * vz, y0 * vy, x0 * vx], [z1 * vz, y1 * vy, x1 * vx]])
    return box + np.asarray(origin_nm_zyx, dtype=np.float64)


def crop_origin_nm(origin_vox_zyx: Sequence[int], voxel_size_zyx: Sequence[float]) -> np.ndarray:
    """Physical (nm) origin of a crop given its origin in that scale's voxels."""
    return np.asarray(origin_vox_zyx, float) * np.asarray(voxel_size_zyx, float)


def skeleton_to_physical(vertices_nm_zyx: np.ndarray, origin_nm_zyx: Sequence[float]) -> np.ndarray:
    """Shift crop-local skeleton vertices (nm) to global physical nm (zyx), by adding the
    crop origin; the tracer must already have returned nm."""
    return np.asarray(vertices_nm_zyx, float) + np.asarray(origin_nm_zyx, float)


def block_chunk_shape_xyz(block_shape_zyx: Sequence[int], voxel_size_zyx: Sequence[float]) -> list[float]:
    """Octree LOD-0 cell size in nm (xyz) for the multires writer = block extent in nm."""
    vz, vy, vx = voxel_size_zyx
    bz, by, bx = block_shape_zyx
    return [bx * vx, by * vy, bz * vz]     # xyz
