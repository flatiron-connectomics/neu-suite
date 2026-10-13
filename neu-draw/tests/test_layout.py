"""Arranging a set: the offsets, and the Scene methods that apply them.

The layout functions take boxes and return offsets, so most of this needs no drawable and
no renderer — which is the point of putting the arithmetic in its own module.

Boxes are given deliberately **unequal sizes and unrelated origins**, because that is what
two datasets look like, and because a layout tested on identical boxes at the origin passes
whatever the packing does.
"""

import numpy as np
import pytest

from neu_lib import BBox, Mesh, Vec3
from neu_draw import layout
from neu_draw.scene import MeshDrawable, Scene

# Extents 101, 301, 51 — a mean of 151, so the default 10% gap is 15.1 nm.
BOXES = [
    BBox((0, 0, 0), (101, 101, 101)),
    BBox((5000, -3000, 900), (5301, -2699, 1201)),
    BBox((-200, 7000, 40), (-149, 7051, 91)),
]


def _moved(boxes, offsets):
    return [b.translate(o) for b, o in zip(boxes, offsets)]


def _cube(origin, side, name):
    o = np.asarray(origin, dtype=np.float32)
    verts = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]],
                     dtype=np.float32) * side + o
    return Mesh(verts, np.array([[0, 1, 2], [0, 1, 3]]), name=name)


def _scene():
    return Scene(drawables=[
        MeshDrawable(_cube((0, 0, 0), 100, "a"), name="a"),
        MeshDrawable(_cube((5000, -3000, 900), 300, "b"), name="b"),
        MeshDrawable(_cube((-200, 7000, 40), 50, "c"), name="c"),
    ])


# --------------------------------------------------------------------------- #
# axes and anchors
# --------------------------------------------------------------------------- #
def test_axes_are_named_in_the_packages_zyx_order():
    """A letter is what a caller means. Getting the order wrong lays a row out along the
    wrong axis, which looks like a layout bug rather than a convention one."""
    assert (layout.axis_index("z"), layout.axis_index("y"),
            layout.axis_index("x")) == (0, 1, 2)
    assert layout.axis_index(2) == 2
    with pytest.raises(ValueError, match="unknown axis"):
        layout.axis_index("w")
    with pytest.raises(ValueError, match="out of range"):
        layout.axis_index(3)


def test_the_anchor_vocabulary_matches_the_one_used_for_a_single_object():
    """Same word must mean the same thing whether placing one drawable or laying out a set."""
    from neu_draw.scene import ANCHORS, anchor_of

    assert layout.ANCHORS == ANCHORS
    box = BBox((0, 0, 0), (10, 20, 30))
    for anchor in ANCHORS:
        assert layout.anchor_point(box, anchor) == anchor_of(box, anchor)


# --------------------------------------------------------------------------- #
# superimpose
# --------------------------------------------------------------------------- #
def test_superimpose_puts_every_anchor_on_the_same_point():
    moved = _moved(BOXES, layout.superimpose_offsets(BOXES))
    centres = {tuple(b.center.tolist()) for b in moved}
    assert len(centres) == 1


def test_superimpose_defaults_to_the_first_box_not_the_origin():
    """The usual reason to superimpose is "show the others against this one", and moving
    the reference too makes the result harder to relate to anything."""
    offsets = layout.superimpose_offsets(BOXES)
    assert offsets[0] == Vec3.zero()
    assert _moved(BOXES, offsets)[0] == BOXES[0]


def test_superimpose_can_be_told_where_instead():
    """Within a nanometre, because a box of odd extent has a fractional centre and a
    half-open integer box cannot sit exactly on it. The OFFSET is exact — only asking for
    the drawn box quantises, and the renderer uses the float."""
    offsets = layout.superimpose_offsets(BOXES, at=(0, 0, 0))
    assert offsets[0] == Vec3(-50.5, -50.5, -50.5)
    for box in _moved(BOXES, offsets):
        assert box.center.tolist() == pytest.approx([0, 0, 0], abs=1.0)


def test_superimpose_on_a_corner_rather_than_the_centre():
    moved = _moved(BOXES, layout.superimpose_offsets(BOXES, anchor="min"))
    assert {b.lo for b in moved} == {BOXES[0].lo}


def test_an_axis_subset_leaves_the_other_axes_at_their_true_coordinates():
    """The parameter that keeps a meaningful axis meaningful. Aligning depth should not
    quietly also collapse the two axes that were telling you where the cells came from."""
    moved = _moved(BOXES, layout.superimpose_offsets(BOXES, axes="z"))
    assert len({b.center[0] for b in moved}) == 1          # z aligned
    assert [b.lo[1] for b in moved] == [b.lo[1] for b in BOXES]   # y untouched
    assert [b.lo[2] for b in moved] == [b.lo[2] for b in BOXES]   # x untouched


def test_an_axis_subset_accepts_a_string_a_sequence_or_indices():
    a = layout.superimpose_offsets(BOXES, axes="zx")
    assert a == layout.superimpose_offsets(BOXES, axes=("z", "x"))
    assert a == layout.superimpose_offsets(BOXES, axes=(0, 2))


# --------------------------------------------------------------------------- #
# arrange: packing
# --------------------------------------------------------------------------- #
def test_packed_boxes_do_not_overlap_however_their_sizes_differ():
    """The reason packing uses each box's own extent rather than a single pitch."""
    moved = _moved(BOXES, layout.arrange_offsets(BOXES, along="x"))
    spans = sorted((b.lo[2], b.hi[2]) for b in moved)
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))


def test_the_gap_is_a_fraction_of_the_MEAN_extent_so_spacing_is_uniform():
    """A gap that grew beside the big cells would read as grouping."""
    moved = _moved(BOXES, layout.arrange_offsets(BOXES, along="x", gap=0.1))
    spans = sorted((b.lo[2], b.hi[2]) for b in moved)
    gaps = [b[0] - a[1] for a, b in zip(spans, spans[1:])]
    assert len(set(gaps)) == 1
    assert gaps[0] == pytest.approx(0.1 * (101 + 301 + 51) / 3, abs=1.0)


def test_a_fixed_pitch_puts_the_anchors_on_a_regular_lattice():
    """What a figure with labels wants, where uneven spacing would read as meaningful."""
    moved = _moved(BOXES, layout.arrange_offsets(BOXES, along="x", spacing=1000))
    centres = sorted(b.center[2] for b in moved)
    assert [b - a for a, b in zip(centres, centres[1:])] == pytest.approx([1000, 1000])


def test_zero_gap_puts_them_edge_to_edge():
    moved = _moved(BOXES, layout.arrange_offsets(BOXES, along="x", gap=0.0))
    spans = sorted((b.lo[2], b.hi[2]) for b in moved)
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))


def test_the_first_box_does_not_move_by_default():
    """An arrangement starts where the objects already are rather than jumping to nm zero."""
    assert layout.arrange_offsets(BOXES, along="x")[0] == Vec3.zero()


# --------------------------------------------------------------------------- #
# arrange: the cross axes, and grids
# --------------------------------------------------------------------------- #
def test_the_cross_axes_are_aligned_so_a_row_reads_as_a_row():
    """Objects from different datasets have unrelated coordinates on the axes not being
    laid out; leaving those alone scatters them and the row stops looking like one."""
    moved = _moved(BOXES, layout.arrange_offsets(BOXES, along="x"))
    assert len({b.center[0] for b in moved}) == 1
    assert len({b.center[1] for b in moved}) == 1


def test_align_cross_false_leaves_the_untouched_axes_alone():
    moved = _moved(BOXES, layout.arrange_offsets(BOXES, along="x", align_cross=False))
    assert [b.lo[0] for b in moved] == [b.lo[0] for b in BOXES]
    assert [b.lo[1] for b in moved] == [b.lo[1] for b in BOXES]
    assert len({b.lo[2] for b in moved}) == 3


def test_wrap_makes_a_grid_on_the_second_axis():
    moved = _moved(BOXES, layout.arrange_offsets(BOXES, along="x", wrap=2, down="z"))
    rows = {round(b.center[0]) for b in moved}
    assert len(rows) == 2
    assert moved[0].center[0] == moved[1].center[0]        # first row
    assert moved[2].center[0] != moved[0].center[0]        # wrapped


def test_a_grid_needs_two_different_axes():
    with pytest.raises(ValueError, match="two different axes"):
        layout.arrange_offsets(BOXES, along="x", wrap=2, down="x")


def test_the_arguments_that_make_no_sense_are_refused():
    with pytest.raises(ValueError, match="must not be negative"):
        layout.arrange_offsets(BOXES, gap=-1)
    with pytest.raises(ValueError, match="must be positive"):
        layout.arrange_offsets(BOXES, spacing=0)
    with pytest.raises(ValueError, match="at least 1"):
        layout.arrange_offsets(BOXES, wrap=0)


def test_laying_out_nothing_is_not_an_error():
    assert layout.arrange_offsets([]) == []
    assert layout.superimpose_offsets([]) == []


def test_one_box_is_left_where_it_is():
    assert layout.arrange_offsets(BOXES[:1]) == [Vec3.zero()]


# --------------------------------------------------------------------------- #
# through a Scene
# --------------------------------------------------------------------------- #
def test_a_scene_lays_itself_out_and_keeps_the_geometry_untouched():
    """The invariant that survives every placement operation: the data still says where the
    tissue is, whatever the arrangement on screen."""
    scene = _scene()
    laid = scene.arrange(along="x")
    assert [tuple(d.mesh.bbox.lo) for d in laid.drawables] == [
        tuple(d.mesh.bbox.lo) for d in scene.drawables]
    assert len({d.bbox.center[2] for d in laid.drawables}) == 3


def test_laying_out_returns_a_copy():
    scene = _scene()
    scene.arrange(along="x")
    assert all(d.offset_zyx_nm == Vec3.zero() for d in scene.drawables)


def test_the_composition_is_the_point():
    """Align depth, spread horizontally, and keep the axis that carried information."""
    scene = _scene()
    composed = scene.superimpose(axes="z").arrange(along="x", align_cross=False)

    assert len({d.bbox.center[0] for d in composed.drawables}) == 1     # z aligned
    assert len({d.bbox.center[2] for d in composed.drawables}) == 3     # x spread
    assert ([d.bbox.lo[1] for d in composed.drawables]
            == [d.bbox.lo[1] for d in scene.drawables])                 # y untouched


def test_a_hidden_drawable_reserves_no_slot():
    """Matching `bbox`, which also ignores them — a gap in a row for something nobody can
    see reads as a missing object."""
    scene = Scene(drawables=[
        MeshDrawable(_cube((0, 0, 0), 100, "a"), name="a"),
        MeshDrawable(_cube((0, 0, 0), 100, "h"), name="h", visible=False),
        MeshDrawable(_cube((0, 0, 0), 100, "b"), name="b"),
    ])
    laid = scene.arrange(along="x")
    assert laid.get("h").offset_zyx_nm == Vec3.zero()
    assert laid.get("b").bbox.lo[2] > laid.get("a").bbox.hi[2] - 1


def test_key_orders_the_layout_without_reordering_the_scene():
    """So a legend built from `scene.names` still matches, however the figure is arranged."""
    scene = _scene()
    laid = scene.arrange(along="x", key=lambda d: -d.bbox.size)
    assert laid.names == scene.names
    by_position = sorted(laid.drawables, key=lambda d: d.bbox.lo[2])
    assert [d.name for d in by_position] == ["b", "a", "c"]


def test_an_arrangement_can_still_be_baked_for_export():
    laid = _scene().arrange(along="x").bake()
    assert all(d.offset_zyx_nm == Vec3.zero() for d in laid.drawables)
    assert len({d.mesh.bbox.lo[2] for d in laid.drawables}) == 3
