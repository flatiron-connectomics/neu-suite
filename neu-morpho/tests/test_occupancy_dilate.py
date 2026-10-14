"""Dilation of the occupancy prefilter.

A coarse scale misses blocks the meshing scale would find, so an un-dilated filter
silently skips data. Dilation trades a wasted read for not losing a body.
"""

import numpy as np

from neu_morpho.occupancy import occupied_blocks


def _occ(arr, dilate=0, allowlist=None, grid=(4, 4, 4)):
    return occupied_blocks(arr, occ_voxel_size=(4.0, 4.0, 4.0),
                           mesh_voxel_size=(1.0, 1.0, 1.0), block_shape=(16, 16, 16),
                           grid_shape=grid, allowlist=allowlist, dilate=dilate)


def test_dilation_grows_by_whole_blocks():
    arr = np.zeros((16, 16, 16), np.uint64)
    arr[5, 5, 5] = 7
    d1 = _occ(arr, dilate=1)
    assert (1, 1, 1) in d1
    # 6-connected neighbours of (1,1,1) come in; corners do not (binary_dilation
    # default structuring element)
    for n in [(0, 1, 1), (2, 1, 1), (1, 0, 1), (1, 2, 1), (1, 1, 0), (1, 1, 2)]:
        assert n in d1, n
    assert len(d1) == 7


def test_dilation_is_a_superset_and_clipped_to_the_grid():
    arr = np.zeros((16, 16, 16), np.uint64)
    arr[1, 1, 1] = 7                      # block (0,0,0) — at the grid corner
    base, grown = _occ(arr), _occ(arr, dilate=1)
    assert base <= grown
    assert all(0 <= i < 4 for idx in grown for i in idx)


def test_dilation_recovers_a_block_a_coarse_scale_missed():
    """Downsampling erased the sparser of two adjacent blocks; dilation recovers it."""
    fine = np.zeros((16, 16, 16), np.uint64)
    fine[4:8, 4:8, 4:8] = 7               # block (1,1,1): dense
    fine[8, 5, 5] = 7                     # block (2,1,1): a single sparse voxel

    coarse = fine.copy()
    coarse[8, 5, 5] = 0                   # ...which downsampling erased
    assert _occ(coarse) == {(1, 1, 1)}                       # the miss
    assert _occ(fine) == {(1, 1, 1), (2, 1, 1)}              # ground truth
    assert _occ(fine) <= _occ(coarse, dilate=1)              # dilation covers it


def test_allowlist_at_a_coarse_scale_can_hide_blocks():
    """Why the ops pass allowlist=None: an allowlisted body erased by downsampling
    makes its block look empty."""
    arr = np.zeros((16, 16, 16), np.uint64)
    arr[5, 5, 5] = 7                      # body 7 survived downsampling
    arr[9, 9, 9] = 99                     # body 99 did too, but 42 did not

    assert _occ(arr, allowlist={42}) == set()          # body 42 vanished -> no blocks
    assert _occ(arr) == {(1, 1, 1), (2, 2, 2)}         # plain !=0 keeps both
