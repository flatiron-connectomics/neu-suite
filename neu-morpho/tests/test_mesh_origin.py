"""Meshes land on their voxels whatever the volume's origin, and edge-touching bodies mesh."""

import numpy as np
import pytest

pytest.importorskip("vol2mesh")

from neu_morpho import pipeline  # noqa: E402


def _volume(tmp_path, origin_nm):
    """A 32^3 precomputed segmentation at 8 nm whose index 0 lies at ``origin_nm``."""
    from neu_lib import Frame, Piece
    from neu_vol import convert, write_piece

    vol = np.zeros((32, 32, 32), np.uint64)
    vol[4:12, 4:12, 4:12] = 7                      # inside
    vol[0:6, 20:28, 20:28] = 9                     # touching the z = 0 face
    write_piece(Piece(vol, Frame((8.0,) * 3, origin_nm), kind="segmentation", name="seg"),
                str(tmp_path / "seg.h5"))
    src = str(tmp_path / "seg")
    convert(str(tmp_path / "seg.h5"), src, kind="segmentation", profile="local-neuroglancer",
            chunk=(16, 16, 16), multiscale=False, progress_path=str(tmp_path / "p.jsonl"))
    return src


@pytest.mark.parametrize("origin_nm", [(0.0, 0.0, 0.0), (800.0, 1600.0, 2400.0)])
def test_meshes_land_on_their_voxels(tmp_path, origin_nm):
    from neu_vol.readback import read_body_mesh

    src = _volume(tmp_path, origin_nm)
    summary = pipeline.run(pipeline.plan(src, src, work_dir=str(tmp_path / "work"),
                                         stages=("mesh",), mesh_scale=0, block=(16, 16, 16),
                                         occupancy_scale=-1, metrics_db=False))
    assert summary["n_failed_bodies"] == 0

    origin = np.asarray(origin_nm)[::-1]                       # xyz, as meshes store them
    verts, _faces = read_body_mesh(src, 7, lod=0)[:2]          # xyz nm
    np.testing.assert_allclose(verts.min(axis=0), origin + 32, atol=8)
    np.testing.assert_allclose(verts.max(axis=0), origin + 96, atol=8)

    # the edge-touching body meshes, its surface on the volume's face
    verts, _faces = read_body_mesh(src, 9, lod=0)[:2]
    assert abs(verts[:, 2].min() - origin[2]) <= 4


def test_a_surface_past_the_volumes_corner_is_written(tmp_path):
    """A decimated edge-touching surface can sit just outside the volume (the head ROI:
    -42 nm). The octree starts a cell below the corner, so its cells stay non-negative."""
    from vol2mesh import Mesh, multires

    from neu_morpho.config import MeshConfig
    from neu_morpho.precomputed import write_body_multires

    lo, hi = -42.0, 300.0                                       # nm, zyx
    corners = [[z, y, x] for z in (lo, hi) for y in (lo, hi) for x in (lo, hi)]
    faces = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
             [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]]
    chunk = [512.0] * 3

    with pytest.raises(ValueError, match="non-negative"):
        write_body_multires(str(tmp_path), 1, Mesh(np.array(corners), np.array(faces)),
                            MeshConfig(num_lods=1), chunk_shape_xyz=chunk,
                            grid_origin_xyz=[0.0] * 3)
    assert write_body_multires(str(tmp_path), 2, Mesh(np.array(corners), np.array(faces)),
                               MeshConfig(num_lods=1), chunk_shape_xyz=chunk,
                               grid_origin_xyz=[-512.0] * 3)
    decoded = multires.read_object_mesh(str(tmp_path), 2, 16)
    verts = np.concatenate([f["vertices_xyz"] for f in decoded["fragments"]])
    np.testing.assert_allclose(verts.min(axis=0), [lo] * 3, atol=0.1)
