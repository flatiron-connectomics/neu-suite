"""Meshes and skeletons land in the same physical-nm space."""

import numpy as np

from neu_morpho.coords import crop_origin_nm, physical_box, skeleton_to_physical


def test_physical_box_scales_region_by_voxel_size():
    region = (slice(10, 20), slice(0, 6), slice(3, 9))
    box = physical_box(region, (8, 8, 8))
    np.testing.assert_array_equal(box, [[80, 0, 24], [160, 48, 72]])


def test_skeleton_to_physical_adds_crop_origin():
    verts = np.array([[0.0, 0.0, 0.0], [16.0, 24.0, 32.0]])   # local nm
    origin = crop_origin_nm((10, 10, 10), (8, 8, 8))          # (80, 80, 80) nm
    out = skeleton_to_physical(verts, origin)
    np.testing.assert_array_equal(out, [[80, 80, 80], [96, 104, 112]])


def test_mesh_vertices_land_in_nm():
    # cube at mesh-voxels [4:12] of a 16^3 block, voxel size 8 nm -> nm [32:96]
    block = np.zeros((16, 16, 16), np.uint64)
    block[4:12, 4:12, 4:12] = 1
    from vol2mesh import Mesh
    box = physical_box((slice(0, 16),) * 3, (8, 8, 8))        # [[0,0,0],[128,128,128]]
    m = Mesh.from_label_volume(block, box, labels=[1], ensure_halo=True, progress=False)[1]
    lo, hi = m.vertices_zyx.min(0), m.vertices_zyx.max(0)
    assert np.allclose(lo, 32, atol=1.0) and np.allclose(hi, 96, atol=1.0)


