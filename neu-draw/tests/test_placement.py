"""Placing drawables relative to one another without touching their coordinates.

An offset is a display transform on the drawable; the geometry keeps its true nm position
throughout, which is what every test here checks.
"""

import numpy as np
import pytest

from neu_lib import BBox, Mesh, Skeleton, Vec3
from neu_draw.scene import (LinesDrawable, MeshDrawable, PointsDrawable, Scene,
                            anchor_of)


def _mesh(origin=0.0, side=10.0, name="body-1"):
    verts = np.array([[0.0, 0, 0], [side, 0, 0], [0.0, side, 0], [0.0, 0, side]]) + origin
    return Mesh(verts, np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]), name=name)


def _skeleton(origin=0.0):
    return Skeleton(np.array([[0.0, 0, 0], [0.0, 0, 50.0]]) + origin,
                    np.array([[0, 1]]), name="skel-1")


# --------------------------------------------------------------------------- #
# the offset is a display transform, not an edit
# --------------------------------------------------------------------------- #
def test_moving_a_drawable_leaves_its_geometry_alone():
    mesh = _mesh(origin=1000.0)
    moved = MeshDrawable(mesh).offset_by((-1000, -1000, -1000))

    assert moved.mesh is mesh                       # not even copied
    assert moved.mesh.bbox.lo == (1000, 1000, 1000)  # the truth, unchanged
    assert moved.bbox.lo == (0, 0, 0)                # where it is drawn


def test_the_drawn_box_is_the_data_box_moved_by_the_offset():
    d = MeshDrawable(_mesh())
    assert d.data_bbox == d.bbox
    assert d.offset_by((5, 0, 0)).bbox == d.data_bbox.translate((5, 0, 0))


def test_offsets_accumulate_and_placed_at_replaces():
    d = MeshDrawable(_mesh())
    assert d.offset_by((5, 0, 0)).offset_by((5, 0, 0)).offset_zyx_nm == Vec3(10, 0, 0)
    assert d.offset_by((5, 0, 0)).placed_at((1, 1, 1)).offset_zyx_nm == Vec3(1, 1, 1)


def test_every_drawable_kind_can_be_placed():
    for drawable in (MeshDrawable(_mesh()),
                     LinesDrawable(_skeleton()),
                     PointsDrawable(np.array([[0.0, 0, 0], [10.0, 10, 10]]))):
        moved = drawable.offset_by((100, 0, 0))
        assert moved.bbox.lo[0] == drawable.bbox.lo[0] + 100
        assert moved.data_bbox == drawable.data_bbox


def test_an_offset_given_at_construction_is_coerced():
    assert MeshDrawable(_mesh(), offset_zyx_nm=(1, 2, 3)).offset_zyx_nm == Vec3(1, 2, 3)


# --------------------------------------------------------------------------- #
# alignment
# --------------------------------------------------------------------------- #
def test_aligning_puts_one_cells_anchor_on_anothers():
    here = MeshDrawable(_mesh(origin=0.0))
    there = MeshDrawable(_mesh(origin=1000.0))

    aligned = there.aligned_to(here)
    assert aligned.bbox.center.tolist() == here.bbox.center.tolist()
    assert aligned.offset_zyx_nm == Vec3(-1000, -1000, -1000)


def test_the_three_anchors_differ_when_the_boxes_do():
    """Different sizes, so the anchors give different offsets."""
    small = MeshDrawable(_mesh(origin=0.0, side=10.0))      # box (0,0,0)..(11,11,11)
    big = MeshDrawable(_mesh(origin=100.0, side=50.0))      # box (100,..)..(151,..)

    assert big.aligned_to(small, anchor="min").bbox.lo == small.bbox.lo
    assert big.aligned_to(small, anchor="max").bbox.hi == small.bbox.hi
    centred = big.aligned_to(small, anchor="center")
    assert centred.bbox.center.tolist() == small.bbox.center.tolist()
    assert len({big.aligned_to(small, anchor=a).offset_zyx_nm
                for a in ("min", "max", "center")}) == 3


def test_centering_puts_the_anchor_at_the_origin_by_default():
    d = MeshDrawable(_mesh(origin=1000.0))
    assert d.centered().bbox.center.tolist() == pytest.approx([0, 0, 0], abs=1.0)
    assert d.centered(at=(50, 0, 0)).bbox.center.tolist() == pytest.approx(
        [50, 0, 0], abs=1.0)


def test_aligning_accepts_anything_with_a_bbox():
    target = MeshDrawable(_mesh(origin=500.0))
    scene = Scene(drawables=[target])
    mover = MeshDrawable(_mesh(origin=0.0))

    by_drawable = mover.aligned_to(target).offset_zyx_nm
    assert mover.aligned_to(scene).offset_zyx_nm == by_drawable
    assert mover.aligned_to(target.bbox).offset_zyx_nm == by_drawable


def test_an_unknown_anchor_is_refused():
    with pytest.raises(ValueError, match="unknown anchor"):
        anchor_of(BBox((0, 0, 0), (1, 1, 1)), "middle")


# --------------------------------------------------------------------------- #
# scenes move as a whole
# --------------------------------------------------------------------------- #
def test_centering_a_scene_preserves_the_arrangement():
    """One shift for the whole scene, not one per drawable."""
    a = MeshDrawable(_mesh(origin=0.0), name="a")
    b = MeshDrawable(_mesh(origin=1000.0), name="b")
    scene = Scene(drawables=[a, b])

    centred = scene.centered()
    assert centred.bbox.center.tolist() == pytest.approx([0, 0, 0], abs=1.0)
    gap_before = (b.bbox.center - a.bbox.center).tolist()
    gap_after = (centred.drawables[1].bbox.center
                 - centred.drawables[0].bbox.center).tolist()
    assert gap_after == gap_before


def test_placing_a_scene_returns_a_moved_copy():
    """The original is untouched, and the scene box (what the camera frames) follows."""
    scene = Scene(drawables=[MeshDrawable(_mesh())])
    moved = scene.offset_by((100, 0, 0))
    assert scene.drawables[0].offset_zyx_nm == Vec3.zero()
    assert moved.drawables[0].offset_zyx_nm == Vec3(100, 0, 0)
    assert moved is not scene
    assert moved.bbox.lo == (100, 0, 0)


# --------------------------------------------------------------------------- #
# bake
# --------------------------------------------------------------------------- #
def test_bake_folds_the_offset_into_the_geometry():
    """For when the shifted coordinates are the output, not just the view."""
    moved = MeshDrawable(_mesh(origin=1000.0)).offset_by((-1000, -1000, -1000))
    baked = Scene(drawables=[moved]).bake().drawables[0]

    assert baked.offset_zyx_nm == Vec3.zero()
    assert baked.mesh.bbox.lo == (0, 0, 0)      # the geometry moved this time
    assert baked.bbox == moved.bbox             # and nothing about the view changed


def test_bake_moves_every_kind_of_geometry():
    scene = Scene(drawables=[
        MeshDrawable(_mesh(), name="m"),
        LinesDrawable(_skeleton(), name="s"),
        PointsDrawable(np.array([[0.0, 0, 0]]), name="p"),
    ]).offset_by((100, 0, 0))

    for before, after in zip(scene.drawables, scene.bake().drawables):
        assert after.offset_zyx_nm == Vec3.zero()
        assert after.data_bbox == before.bbox
        assert after.bbox == before.bbox


def test_baking_an_unmoved_scene_changes_nothing():
    scene = Scene(drawables=[MeshDrawable(_mesh())])
    assert scene.bake().drawables[0].data_bbox == scene.drawables[0].data_bbox
