"""Coarse-scale occupancy prefilter — skip empty blocks up front.

A small coarse-scale level is reduced to a boolean over the meshing block grid, so empty
blocks are never read. With an allowlist, "occupied" means ``isin(coarse, allowlist)``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def occupied_blocks(
    occ_array_zyx: np.ndarray,
    *,
    occ_voxel_size: Sequence[float],
    mesh_voxel_size: Sequence[float],
    block_shape: Sequence[int],
    grid_shape: Sequence[int],
    allowlist: set[int] | None = None,
    dilate: int = 0,
) -> set[tuple[int, int, int]]:
    """Return the set of block indices (in the meshing grid) that are non-empty.

    ``occ_array_zyx`` is the whole coarse-scale segmentation (small). Physical
    voxel sizes map the coarse grid onto the meshing block grid; ``grid_shape`` is
    the number of blocks per axis at the meshing scale.

    ``dilate`` grows the occupied set by N blocks in every direction. **Use it**: a
    coarse scale drops sparse tissue, and the miss does not shrink with finer scales, so
    an un-dilated filter silently skips blocks holding data. An extra block costs one
    read; a missed one costs a body. On a sparse volume of small objects, turn occupancy
    off altogether.

    ``allowlist`` is discouraged here: small allowlisted bodies vanish at coarse scales.
    Prefer the default ``!= 0`` test and filter by allowlist inside the block.
    """
    if allowlist is None:
        occ_mask = occ_array_zyx != 0
    else:
        vals = np.array(sorted(allowlist), dtype=occ_array_zyx.dtype)
        occ_mask = np.isin(occ_array_zyx, vals)          # one pass over the small array

    # coarse-scale voxels spanned by one meshing block, per axis
    occ_per_block = [block_shape[a] * mesh_voxel_size[a] / occ_voxel_size[a] for a in range(3)]

    grid = np.zeros(tuple(int(g) for g in grid_shape), dtype=bool)
    for index in np.ndindex(grid.shape):
        sl = tuple(
            slice(int(round(i * opb)), int(round((i + 1) * opb)))
            for i, opb in zip(index, occ_per_block)
        )
        sub = occ_mask[sl]
        if sub.size and sub.any():
            grid[index] = True

    if dilate > 0:
        from scipy.ndimage import binary_dilation
        grid = binary_dilation(grid, iterations=int(dilate))

    return {tuple(int(v) for v in idx) for idx in np.argwhere(grid)}
