"""Region tests, and how they compose with `Skeleton.crop`.

The interface is one callable answering "is each of these points in the region", and the
thing worth pinning is that **a point outside the region's own extent is outside the
region, not an error** — a skeleton normally runs past any one ROI or compartment mask,
and that is the question being asked of it.
"""

import numpy as np
import pytest

from neu_lib import (BBox, Frame, Skeleton, box_predicate, mask_predicate, union)


def test_a_box_predicate_feeds_straight_into_crop():
    skel = Skeleton(np.array([[0.0, 0, 0], [0.0, 0, 1000.0]]), np.array([[0, 1]]))
    inside = box_predicate(BBox((-10, -10, -10), (10, 10, 500)))
    out = skel.crop(inside, tolerance_nm=1.0)
    assert len(out.vertices_zyx_nm) == 2                 # one kept + one boundary
    assert out.vertices_zyx_nm[1][2] == pytest.approx(500.0, abs=2.0)


def test_a_mask_predicate_resolves_points_through_its_own_frame():
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[0, 0, 0] = True
    inside = mask_predicate(mask, Frame((100.0, 100.0, 100.0)))
    assert inside(np.array([[50.0, 50.0, 50.0]])).tolist() == [True]
    assert inside(np.array([[150.0, 50.0, 50.0]])).tolist() == [False]


def test_a_mask_read_from_a_crop_places_by_its_origin():
    """The frame carries an origin, so a compartment mask read in a small box around
    the thing that localises it lands where that box actually was — rather than being
    assumed to start at nm zero, which is the mistake that hid in read_scales."""
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[0, 0, 0] = True
    inside = mask_predicate(mask, Frame((100.0, 100.0, 100.0), (1000.0, 0.0, 0.0)))
    assert inside(np.array([[1050.0, 50.0, 50.0]])).tolist() == [True]
    assert inside(np.array([[50.0, 50.0, 50.0]])).tolist() == [False]


def test_points_outside_the_mask_array_are_outside_the_region():
    inside = mask_predicate(np.ones((2, 2, 2), bool), Frame((10.0, 10.0, 10.0)))
    assert inside(np.array([[-100.0, 0, 0], [1e6, 0, 0]])).tolist() == [False, False]


def test_a_mask_must_be_three_dimensional():
    with pytest.raises(ValueError, match="3-D zyx"):
        mask_predicate(np.ones((4, 4), bool), Frame.identity())


# --------------------------------------------------------------------------- #
# composition
# --------------------------------------------------------------------------- #
def test_union_lets_compartments_keep_their_own_frames():
    """Excluding soma AND nucleus is two masks, and merging them into one array would
    force them to share a resolution and an origin, which they need not."""
    soma = mask_predicate(np.ones((1, 1, 1), bool), Frame((100.0, 100.0, 100.0)))
    nucleus = mask_predicate(np.ones((1, 1, 1), bool),
                             Frame((10.0, 10.0, 10.0), (500.0, 0.0, 0.0)))
    inside = union(soma, nucleus)
    pts = np.array([[50.0, 50.0, 50.0],       # in the soma mask
                    [505.0, 5.0, 5.0],        # in the nucleus mask
                    [50_000.0, 0.0, 0.0]])    # in neither
    assert inside(pts).tolist() == [True, True, False]


def test_union_of_nothing_is_inside_nothing():
    assert union()(np.zeros((3, 3))).tolist() == [False, False, False]


def test_union_composes_with_exclude():
    """`exclude` is the complement of `crop`, so a union of compartments removes all
    of them in one pass — which is what a measurement needs."""
    skel = Skeleton(np.array([[0.0, 0, 0], [0.0, 0, 100.0], [0.0, 0, 1000.0]]),
                    np.array([[0, 1], [1, 2]]))
    a = box_predicate(BBox((-1, -1, -1), (1, 1, 1)))
    b = box_predicate(BBox((-1, -1, 900), (1, 1, 1100)))
    kept = skel.exclude(union(a, b), tolerance_nm=1.0)
    # the middle vertex survives; the two ends are inside one region or the other
    assert any(abs(v[2] - 100.0) < 1e-6 for v in kept.vertices_zyx_nm)
