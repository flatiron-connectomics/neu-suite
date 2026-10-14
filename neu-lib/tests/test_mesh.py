"""Mesh: the xyz to zyx flip on the way in, translate, and construction checks."""

import numpy as np
import pytest

from neu_lib import BBox, Mesh


def _tetra():
    verts = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0],
                      [0.0, 20.0, 0.0], [0.0, 0.0, 30.0]])
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    return verts, faces


def test_from_precomputed_flips_xyz_to_zyx():
    mesh = Mesh.from_precomputed(np.array([[1.0, 2.0, 3.0]]), np.zeros((0, 3)))
    assert mesh.vertices_zyx_nm.tolist() == [[3.0, 2.0, 1.0]]


def test_translate_moves_only_the_vertices_of_a_copy():
    """Faces are indices and normals are directions; neither may move."""
    verts, faces = _tetra()
    normals = np.tile([1.0, 0.0, 0.0], (4, 1))
    mesh = Mesh(verts, faces, normals)
    moved = mesh.translate([5.0, 0.0, -2.0])
    assert np.array_equal(moved.faces, faces)
    assert np.array_equal(moved.normals_zyx, normals)
    assert moved.vertices_zyx_nm[0].tolist() == [5.0, 0.0, -2.0]
    assert mesh.vertices_zyx_nm[0].tolist() == [0.0, 0.0, 0.0], "original unchanged"


def test_bbox_contains_every_vertex():
    verts, faces = _tetra()
    box = Mesh(verts, faces).bbox
    assert box == BBox((0, 0, 0), (11, 21, 31))


def test_an_empty_mesh_has_an_empty_box_rather_than_raising():
    assert Mesh(np.zeros((0, 3)), np.zeros((0, 3))).bbox.is_empty()


def test_out_of_range_faces_are_rejected_at_construction():
    verts, _ = _tetra()
    with pytest.raises(ValueError, match="out of range"):
        Mesh(verts, np.array([[0, 1, 99]]))


def test_mismatched_normals_are_rejected():
    verts, faces = _tetra()
    with pytest.raises(ValueError, match="normals"):
        Mesh(verts, faces, np.zeros((2, 3)))
