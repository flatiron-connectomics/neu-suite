"""Meshes: the xyz→zyx flip on the way in, and the translate bug that preceded this."""

import numpy as np
import pytest

from em_viz.geometry import BBox, Mesh


def _tetra():
    verts = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0],
                      [0.0, 20.0, 0.0], [0.0, 0.0, 30.0]])
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    return verts, faces


def test_from_precomputed_flips_xyz_to_zyx():
    """`read_body_mesh` returns the order the format stores, which is xyz. Getting this
    wrong mirrors the mesh through the z=x diagonal — a valid mesh in the wrong place."""
    mesh = Mesh.from_precomputed(np.array([[1.0, 2.0, 3.0]]), np.zeros((0, 3)))
    assert mesh.vertices_zyx_nm.tolist() == [[3.0, 2.0, 1.0]]


def test_translate_moves_only_the_vertices():
    """The predecessor added the offset to `faces` and `normals` as well. Faces are
    indices and normals are directions; doing that scrambles topology and shading."""
    verts, faces = _tetra()
    normals = np.tile([1.0, 0.0, 0.0], (4, 1))
    moved = Mesh(verts, faces, normals).translate([5.0, 0.0, -2.0])
    assert np.array_equal(moved.faces, faces)
    assert np.array_equal(moved.normals_zyx, normals)
    assert moved.vertices_zyx_nm[0].tolist() == [5.0, 0.0, -2.0]


def test_translate_does_not_mutate_the_original():
    verts, faces = _tetra()
    mesh = Mesh(verts, faces)
    mesh.translate([100.0, 100.0, 100.0])
    assert mesh.vertices_zyx_nm[0].tolist() == [0.0, 0.0, 0.0]


def test_bbox_contains_every_vertex():
    verts, faces = _tetra()
    box = Mesh(verts, faces).bbox
    assert box == BBox((0, 0, 0), (11, 21, 31))


def test_an_empty_mesh_has_an_empty_box_rather_than_raising():
    """A body whose LOD came back with no fragments must not take a scene down."""
    assert Mesh(np.zeros((0, 3)), np.zeros((0, 3))).bbox.is_empty()


def test_out_of_range_faces_are_rejected_at_construction():
    """Silent when unchecked: the renderer reads whatever is at that buffer offset."""
    verts, _ = _tetra()
    with pytest.raises(ValueError, match="out of range"):
        Mesh(verts, np.array([[0, 1, 99]]))


def test_mismatched_normals_are_rejected():
    verts, faces = _tetra()
    with pytest.raises(ValueError, match="normals"):
        Mesh(verts, faces, np.zeros((2, 3)))
