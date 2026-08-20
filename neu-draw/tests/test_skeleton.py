"""Skeletons as vertices and edges.

The thing worth pinning is that **topology is never interpreted**. The predecessor's
graph walker had to handle cycles, self-loops and disconnected components explicitly,
and got each wrong at least once. An edge list handed straight to
`LineSegmentMaterial` cannot: the tests below run those shapes through and assert the
segment buffer is exactly the edges, in order.
"""

import numpy as np
import pytest

from neu_draw.geometry import BBox, Frame, Skeleton, to_xyz


def _chain(n=4, step=10.0):
    """A path 0-1-2-3 along z."""
    verts = np.zeros((n, 3), dtype=np.float32)
    verts[:, 0] = np.arange(n) * step
    edges = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1)
    return Skeleton(verts, edges, name="chain")


# --------------------------------------------------------------------------- #
# the rendering path
# --------------------------------------------------------------------------- #

def test_segments_are_consecutive_pairs_one_per_edge():
    """The entire rendering path. `LineSegmentMaterial` draws positions[0:2], [2:4], …
    so the buffer must be exactly the edge list, flattened and in order."""
    skel = _chain(4)
    seg = skel.segments()
    assert seg.shape == (6, 3)                      # 3 edges, 2 points each
    assert seg[0].tolist() == [0.0, 0.0, 0.0]
    assert seg[1].tolist() == [10.0, 0.0, 0.0]
    assert seg[2].tolist() == [10.0, 0.0, 0.0]      # the shared vertex, repeated
    assert seg[3].tolist() == [20.0, 0.0, 0.0]


def test_the_renderer_buffer_is_contiguous_float32_xyz():
    seg = to_xyz(_chain().segments())
    assert seg.dtype == np.float32 and seg.flags["C_CONTIGUOUS"]
    assert seg[1].tolist() == [0.0, 0.0, 10.0]      # z became the last axis


@pytest.mark.parametrize("edges, expected_segments", [
    ([[0, 1], [1, 2], [2, 0]], 6),                  # a cycle
    ([[0, 1], [2, 3]], 4),                          # two disconnected components
    ([[0, 0]], 2),                                  # a self-loop
    ([], 0),                                        # no edges at all
])
def test_any_topology_renders_without_being_interpreted(edges, expected_segments):
    """Cycles, disconnected components and self-loops each broke the branch walker at
    some point. Here nothing looks at the topology, so there is nothing to break."""
    verts = np.arange(12, dtype=np.float32).reshape(4, 3)
    skel = Skeleton(verts, np.asarray(edges, dtype=np.int64).reshape(-1, 2))
    assert len(skel.segments()) == expected_segments


def test_a_vertex_with_no_edges_draws_nothing_but_is_kept():
    """It still counts toward the bounding box, which is why it is not dropped."""
    skel = Skeleton(np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]]), np.zeros((0, 2)))
    assert len(skel.segments()) == 0
    assert skel.bbox == BBox((0, 0, 0), (101, 1, 1))


# --------------------------------------------------------------------------- #
# construction
# --------------------------------------------------------------------------- #

def test_from_precomputed_flips_xyz_to_zyx():
    skel = Skeleton.from_precomputed(np.array([[1.0, 2.0, 3.0]]), np.zeros((0, 2)))
    assert skel.vertices_zyx_nm.tolist() == [[3.0, 2.0, 1.0]]


def test_from_swc_reads_columns_in_the_named_order_and_scales_to_nm():
    """SWC coordinates are voxel indices, so nm requires the level's real voxel size —
    which is why `frame` is required rather than defaulted."""
    table = {"node": [1, 2], "parent": [-1, 1],
             "x": [0, 1], "y": [0, 2], "z": [0, 3]}
    skel = Skeleton.from_swc(table, frame=Frame((40.0, 8.0, 8.0)))
    assert skel.vertices_zyx_nm.tolist() == [[0.0, 0.0, 0.0], [120.0, 16.0, 8.0]]
    assert skel.edges.tolist() == [[1, 0]]          # node 2's parent is node 1


def test_from_swc_treats_an_absent_parent_as_a_root():
    """`-1` is SWC's root marker, and a dangling id means the same thing here: no edge.
    Nothing is required to be a tree."""
    table = {"node": [1, 2, 3], "parent": [-1, 999, 1], "x": [0, 1, 2],
             "y": [0, 0, 0], "z": [0, 0, 0]}
    skel = Skeleton.from_swc(table, frame=Frame.identity())
    assert skel.edges.tolist() == [[2, 0]]


def test_from_swc_rejects_duplicate_node_ids():
    """Ambiguous parent lookup. Resolving to the last silently wires edges elsewhere."""
    table = {"node": [1, 1], "parent": [-1, 1], "x": [0, 1], "y": [0, 0], "z": [0, 0]}
    with pytest.raises(ValueError, match="duplicate node ids"):
        Skeleton.from_swc(table, frame=Frame.identity())


def test_out_of_range_edges_are_rejected_at_construction():
    with pytest.raises(ValueError, match="out of range"):
        Skeleton(np.zeros((3, 3)), np.array([[0, 7]]))


def test_mismatched_radii_are_rejected():
    with pytest.raises(ValueError, match="radii"):
        Skeleton(np.zeros((3, 3)), np.zeros((0, 2)), radii_nm=np.zeros(2))


# --------------------------------------------------------------------------- #
# cropping against a region predicate
# --------------------------------------------------------------------------- #

def _z_below(limit):
    """Region predicate: inside means z < limit. Takes zyx nm, returns one bool each."""
    return lambda points: np.asarray(points)[:, 0] < limit


def test_crop_keeps_whole_edges_and_bisects_the_crossing_one():
    """The point of bisecting: a branch terminates at the region surface, not at the
    last vertex inside it."""
    skel = _chain(4)                                 # vertices at z = 0, 10, 20, 30
    out = skel.crop(_z_below(25.0), tolerance_nm=0.01)
    assert len(out.vertices_zyx_nm) == 4             # 3 kept, plus the boundary vertex
    assert len(out.edges) == 3
    boundary = out.vertices_zyx_nm[-1]
    assert boundary[0] == pytest.approx(25.0, abs=0.05)


def test_the_boundary_vertex_lands_on_the_kept_side():
    """Strictly inside, never past. A vertex the predicate rejects would put a visible
    stub outside the region."""
    out = _chain(4).crop(_z_below(25.0), tolerance_nm=0.01)
    assert _z_below(25.0)(out.vertices_zyx_nm).all()


def test_tolerance_sets_how_close_the_boundary_lands():
    coarse = _chain(4).crop(_z_below(25.0), tolerance_nm=8.0)
    fine = _chain(4).crop(_z_below(25.0), tolerance_nm=0.001)
    assert abs(fine.vertices_zyx_nm[-1][0] - 25.0) < abs(
        coarse.vertices_zyx_nm[-1][0] - 25.0)


def test_exclude_is_the_complement_of_crop():
    skel = _chain(4)
    kept = skel.crop(_z_below(25.0), tolerance_nm=0.01)
    dropped = skel.exclude(_z_below(25.0), tolerance_nm=0.01)
    inside = set(map(tuple, kept.vertices_zyx_nm[:3].tolist()))
    outside = set(map(tuple, dropped.vertices_zyx_nm[:1].tolist()))
    assert not (inside & outside)
    assert (30.0, 0.0, 0.0) in outside


def test_crop_that_keeps_everything_changes_nothing():
    skel = _chain(4)
    out = skel.crop(lambda p: np.ones(len(p), bool))
    assert np.array_equal(out.vertices_zyx_nm, skel.vertices_zyx_nm)
    assert np.array_equal(out.edges, skel.edges)


def test_crop_that_keeps_nothing_gives_an_empty_skeleton():
    out = _chain(4).crop(lambda p: np.zeros(len(p), bool))
    assert len(out.vertices_zyx_nm) == 0 and len(out.edges) == 0


def test_crop_interpolates_the_radius_at_the_boundary():
    verts = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    skel = Skeleton(verts, np.array([[0, 1]]), radii_nm=np.array([10.0, 20.0]))
    out = skel.crop(_z_below(50.0), tolerance_nm=0.01)
    assert out.radii_nm[-1] == pytest.approx(15.0, abs=0.1)
    assert len(out.radii_nm) == len(out.vertices_zyx_nm)


def test_crop_bisects_every_crossing_edge_in_one_batch():
    """All crossings advance together — the round count is set by the longest segment,
    so `inside` is called a fixed number of times, not once per point."""
    calls = []

    def counting(points):
        calls.append(len(points))
        return np.asarray(points)[:, 0] < 5.0

    verts = np.array([[0.0, 0, 0], [10.0, 0, 0], [0.0, 1, 0], [10.0, 1, 0],
                      [0.0, 2, 0], [10.0, 2, 0]])
    skel = Skeleton(verts, np.array([[0, 1], [2, 3], [4, 5]]))
    skel.crop(counting, tolerance_nm=0.5)
    assert calls[0] == 6                             # the initial per-vertex test
    assert set(calls[1:]) == {3}                     # then all 3 crossings per round


def test_crop_rejects_a_predicate_returning_the_wrong_shape():
    with pytest.raises(ValueError, match="one bool per point"):
        _chain(4).crop(lambda p: np.array([True]))


def test_crop_handles_a_cycle_crossing_the_boundary_twice():
    """A closed loop half in and half out. The walker had to special-case this; an edge
    list does not, and both crossings get their own boundary vertex."""
    verts = np.array([[0.0, 0, 0], [10.0, 0, 0], [10.0, 10, 0], [0.0, 10, 0]])
    skel = Skeleton(verts, np.array([[0, 1], [1, 2], [2, 3], [3, 0]]))
    out = skel.crop(_z_below(5.0), tolerance_nm=0.01)
    assert len(out.vertices_zyx_nm) == 4             # 2 kept + 2 boundary vertices
    assert len(out.edges) == 3                       # 1 whole edge + 2 stubs
