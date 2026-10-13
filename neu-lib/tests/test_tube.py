"""The tube builder: one truncated cone per skeleton edge, from its radii.

Pure geometry — numpy in, numpy out. The property worth asserting is that a ring sits at
the node's own radius and perpendicular to its edge, because that is what makes a radius
which does not fit inside the body show up as the tube breaking the surface.
"""

import numpy as np
import pytest

from neu_lib import Mesh, Skeleton, frustum_mesh, skeleton_tube

# --------------------------------------------------------------------------- #
# frustum geometry
# --------------------------------------------------------------------------- #
def test_frustum_rings_have_the_node_radii_at_each_end():
    v = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]])
    fv, ff = frustum_mesh(v, np.array([[0, 1]]), np.array([3.0, 7.0]), sides=8)
    assert fv.shape == (16, 3) and ff.shape == (16, 3)
    assert ff.max() < len(fv)
    ring0, ring1 = fv[:8], fv[8:]
    np.testing.assert_allclose(ring0[:, 2], 0.0, atol=1e-9)
    np.testing.assert_allclose(ring1[:, 2], 10.0, atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(ring0[:, :2], axis=1), 3.0, atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(ring1[:, :2], axis=1), 7.0, atol=1e-9)


@pytest.mark.parametrize("direction", [
    (1.0, 0.0, 0.0),          # along the axis the reference vector is swapped for
    (0.0, 1.0, 0.0),
    (1.0, 1.0, 1.0),
    (-3.0, 0.5, 2.0),
])
def test_frustum_rings_are_perpendicular_to_the_edge(direction):
    """The per-edge basis must stay orthonormal in every orientation.

    The reference direction is swapped near x to keep the cross product from
    degenerating; if that guard were wrong, the ring would collapse or tilt only for
    edges pointing that way — a defect no single-orientation test would find.
    """
    d = np.asarray(direction, float)
    v = np.array([[0.0, 0.0, 0.0], d * 9.0])
    fv, _ = frustum_mesh(v, np.array([[0, 1]]), np.array([2.0, 2.0]), sides=6)
    axis = d / np.linalg.norm(d)
    ring = fv[:6]
    assert np.abs(ring @ axis).max() < 1e-9, "ring is not perpendicular to the edge"
    np.testing.assert_allclose(np.linalg.norm(ring, axis=1), 2.0, atol=1e-9)


def test_frustum_drops_zero_length_edges_without_nans():
    v = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 4.0]])
    fv, ff = frustum_mesh(v, np.array([[0, 1], [1, 2]]), np.ones(3), sides=4)
    assert len(fv) == 8, "the degenerate edge should contribute nothing"
    assert not np.isnan(fv).any() and ff.max() < len(fv)


def test_frustum_handles_an_empty_skeleton():
    fv, ff = frustum_mesh(np.zeros((0, 3)), np.zeros((0, 2), int), np.zeros(0))
    assert fv.shape == (0, 3) and ff.shape == (0, 3)


def test_frustum_face_indices_stay_within_each_edges_own_block():
    """Faces must never bridge two edges: the base offset is per-edge, not global."""
    v = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0], [0.0, 5.0, 5.0]])
    sides = 4
    fv, ff = frustum_mesh(v, np.array([[0, 1], [1, 2]]), np.ones(3), sides=sides)
    block = ff // (2 * sides)
    assert (block.min(axis=1) == block.max(axis=1)).all(), \
        "a triangle spans two frusta; the per-edge offset is wrong"


# --------------------------------------------------------------------------- #
# the zyx-facing wrapper
# --------------------------------------------------------------------------- #
def test_a_skeleton_without_radii_has_no_tube():
    bare = Skeleton(np.zeros((2, 3)), np.array([[0, 1]]), name="bare")
    with pytest.raises(ValueError, match="carries no radii"):
        skeleton_tube(bare)


def test_the_tube_surrounds_the_centreline():
    """A tube of radius r around a segment must contain the segment's own box."""
    skel = Skeleton(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1000.0]]),
                    np.array([[0, 1]]), radii_nm=np.array([50.0, 50.0]), name="s")
    tube = skeleton_tube(skel, sides=8)
    assert isinstance(tube, Mesh) and len(tube.faces) > 0
    assert tube.bbox.union(skel.bbox) == tube.bbox


def test_the_tube_is_built_in_xyz_so_winding_survives_the_round_trip():
    """Feeding zyx straight in reverses handedness and the normals face inward, which
    renders dark rather than raising — so the flip is asserted, not trusted."""
    skel = Skeleton(np.array([[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0]]),
                    np.array([[0, 1]]), radii_nm=np.array([100.0, 100.0]), name="z")
    tube = skeleton_tube(skel, sides=6)
    # The skeleton runs along z, so the tube's z extent is the edge and its y/x extents
    # are the radius. Reversed, the extents would swap.
    shape = tube.bbox.shape
    assert shape[0] > 900 and shape[1] < 300 and shape[2] < 300
