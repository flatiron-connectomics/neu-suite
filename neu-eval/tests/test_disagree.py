"""Ranking, classifying and locating the disagreements."""

from __future__ import annotations

import numpy as np
import pytest

from neu_eval import disagree
from neu_eval.overlap import contingency
from neu_eval.voxel import voi


def _c(a, b, labels=("gt", "seg")):
    a = np.asarray(a, dtype=np.uint64).reshape(1, 1, -1)
    b = np.asarray(b, dtype=np.uint64).reshape(1, 1, -1)
    return contingency(a, b, ignore_a=(), labels=labels)


def _piece(arr):
    from neu_lib import Frame, Piece

    return Piece(np.asarray(arr, dtype=np.uint64), Frame(voxel_size_nm=(8.0, 8.0, 8.0)),
                 kind="segmentation")


# -- classification ---------------------------------------------------------------------

@pytest.mark.parametrize("a, b, kinds, n_rows", [
    ([1] * 10, [5] * 5 + [6] * 5, {"split"}, 2),
    ([1] * 5 + [2] * 5, [5] * 10, {"merge"}, 2),
    ([1, 1, 2, 2], [5, 6, 5, 6], {"tangle"}, 4),     # both bodies over both segments
    ([1, 1, 2, 2], [5, 5, 6, 6], set(), 0),          # a clean correspondence
    ([], [], set(), 0),
], ids=["split", "merge", "tangle", "clean", "empty"])
def test_disagreements_are_classified(a, b, kinds, n_rows):
    rows = disagree.rows(_c(a, b))
    assert {r["kind"] for r in rows} == kinds
    assert len(rows) == n_rows


def test_matches_can_be_asked_for():
    rows = disagree.rows(_c([1, 1, 2, 2], [5, 5, 6, 6]), include_matches=True)
    assert len(rows) == 2
    assert {r["kind"] for r in rows} == {"match"}


# -- the speck filter -------------------------------------------------------------------

def test_specks_do_not_make_a_body_look_split():
    """One 1000-voxel body in one segment plus 20 single-voxel strays is a clean match,
    not a 21-way split."""
    a = [1] * 1000
    b = [5] * 980 + list(range(100, 120))
    rows = disagree.rows(_c(a, b))
    assert rows == []
    assert disagree.rows(_c(a, b), min_frac=0.0) != [], "and min_frac=0 still shows them"


def test_min_voxels_filters_independently_of_the_fraction():
    a = [1] * 100 + [2] * 3
    b = [5] * 100 + [5] * 3
    assert disagree.rows(_c(a, b), min_frac=0.0, min_voxels=1) != []
    assert disagree.rows(_c(a, b), min_frac=0.0, min_voxels=10_000) == []


def test_an_out_of_range_min_frac_is_refused():
    with pytest.raises(ValueError, match="min_frac"):
        disagree.rows(_c([1], [5]), min_frac=1.5)


# -- ranking ----------------------------------------------------------------------------

def test_severity_ranking_inherits_vois_size_bias_and_fraction_ranking_fixes_it():
    """Body 1 (600 voxels) cut in half vs body 2 (100,000 voxels) losing a 6% sliver: VOI
    weights by size, so severity (the default rank) puts the sliver first; ranking by
    fraction puts the halved body first."""
    a = [1] * 600 + [2] * 100_000
    b = [5] * 300 + [6] * 300 + [7] * 94_000 + [8] * 6_000

    by_severity = disagree.rows(_c(a, b))
    severities = [r["severity"] for r in by_severity]
    assert severities == sorted(severities, reverse=True)
    assert by_severity[0]["gt_id"] == 2, "VOI is dominated by the large body"
    assert by_severity[0]["severity"] > 50 * by_severity[-1]["severity"]

    by_fraction = disagree.rows(_c(a, b), rank="fraction")
    assert by_fraction[0]["gt_id"] == 1, "relative damage puts the halved body first"


def test_one_merge_event_is_grouped_not_repeated():
    """A segment swallowing three bodies is three rows, grouped under the segment."""
    c = _c([1] * 4 + [2] * 4 + [3] * 4, [9] * 12)
    rows = disagree.rows(c)
    assert {r["kind"] for r in rows} == {"merge"}
    assert len(rows) == 3
    assert {r["group"] for r in rows} == {"9"}, "grouped by the merging SEGMENT"
    assert all(r["group_size"] == 3 for r in rows)
    assert sorted(r["group_rank"] for r in rows) == [1, 2, 3]


def test_a_split_is_grouped_by_the_body_that_came_apart():
    rows = disagree.rows(_c([1] * 9, [5, 5, 5, 6, 6, 6, 7, 7, 7]))
    assert {r["kind"] for r in rows} == {"split"}
    assert {r["group"] for r in rows} == {"1"}, "grouped by the BODY"


def test_max_per_group_keeps_the_worst_of_each():
    """One bad segment cannot crowd a top-N list; its group size stays visible."""
    c = _c([1] * 4 + [2] * 4 + [3] * 4, [9] * 12)
    capped = disagree.rows(c, max_per_group=1)
    assert len(capped) == 1
    assert capped[0]["group_size"] == 3, "the group's true size survives the cap"
    assert capped[0]["group_rank"] == 1
    # and it is the most severe of the three, not an arbitrary one
    assert capped[0]["severity"] == max(r["severity"] for r in disagree.rows(c))


def test_the_cap_is_applied_across_groups_not_globally():
    a = [1] * 4 + [2] * 4 + [3] * 4 + [4] * 6
    b = [9] * 12 + [8] * 3 + [7] * 3
    rows = disagree.rows(_c(a, b), max_per_group=1)
    groups = {r["group"] for r in rows}
    assert len(groups) == len(rows), "one row survives per group, several groups survive"


def test_an_unknown_rank_is_refused_by_name():
    with pytest.raises(ValueError, match="'severity' or 'fraction'"):
        disagree.rows(_c([1], [5]), rank="voxels")


def test_severity_sums_to_the_metric_it_came_from():
    c = _c([1, 1, 2, 2], [5, 6, 5, 6])
    rows = disagree.rows(c)
    total = voi(c)
    assert sum(r["severity"] for r in rows) == pytest.approx(total.total)


def test_top_truncates_after_ranking():
    a = [1] * 10 + [2] * 10
    b = [5] * 5 + [6] * 5 + [7] * 5 + [8] * 5
    everything = disagree.rows(_c(a, b))
    assert len(disagree.rows(_c(a, b), top=2)) == 2
    assert disagree.rows(_c(a, b), top=2) == everything[:2]


def test_columns_are_named_after_the_sides():
    rows = disagree.rows(_c([1] * 10, [5] * 5 + [6] * 5, labels=("truth", "auto")))
    assert {"truth_id", "auto_id", "frac_of_truth", "frac_of_auto"} <= set(rows[0])


def test_counts_by_kind_covers_every_kind():
    tally = disagree.counts_by_kind(disagree.rows(_c([1] * 10, [5] * 5 + [6] * 5)))
    assert set(tally) == set(disagree.KINDS)
    assert tally["split"] == 2 and tally["merge"] == 0


# -- locating ---------------------------------------------------------------------------

def test_the_located_point_is_inside_its_own_pair():
    a = np.ones((8, 8, 8), dtype=np.uint64)
    b = np.full((8, 8, 8), 5, dtype=np.uint64)
    b[4:] = 6
    pa, pb = _piece(a), _piece(b)
    rows = disagree.locate(disagree.rows(contingency(a, b, ignore_a=())), pa, pb)
    assert len(rows) == 2
    for row in rows:
        z, y, x = row["z_vox"], row["y_vox"], row["x_vox"]
        assert int(a[z, y, x]) == row["a_id"]
        assert int(b[z, y, x]) == row["b_id"]


def _slab():
    """One reference body, cut into two segments at z=5 and z=11 — a false cut."""
    a = np.ones((16, 16, 16), dtype=np.uint64)
    b = np.full((16, 16, 16), 5, dtype=np.uint64)
    b[5:11] = 6
    return a, b


def test_the_seam_point_sits_on_the_false_cut():
    """`at="seam"` (the default) points at a cut, z=5 or z=11, not the slab's middle,
    where the two labelings agree."""
    a, b = _slab()
    rows = disagree.locate(
        disagree.rows(contingency(a, b, ignore_a=())), _piece(a), _piece(b))
    six = next(r for r in rows if r["b_id"] == 6)
    assert six["point_at"] == "seam-split"
    assert six["z_vox"] in (5, 10), "on a face of the slab, i.e. on the cut"
    # And the voxel across the cut really belongs to the other segment.
    z = six["z_vox"]
    beyond = z - 1 if z == 5 else z + 1
    assert int(b[beyond, six["y_vox"], six["x_vox"]]) == 5


def test_the_overlap_point_is_still_available_and_is_interior():
    """`at="overlap"` points into the agreeing interior."""
    a, b = _slab()
    rows = disagree.locate(
        disagree.rows(contingency(a, b, ignore_a=())), _piece(a), _piece(b),
        at="overlap")
    six = next(r for r in rows if r["b_id"] == 6)
    assert six["point_at"] == "overlap"
    assert 6 <= six["z_vox"] <= 9, "inside the slab, off its faces"


def test_a_merge_seam_is_where_the_reference_bodies_meet():
    """One segment spanning two bodies: the seam is inside the segment."""
    a = np.ones((16, 16, 16), dtype=np.uint64)
    a[8:] = 2
    b = np.full((16, 16, 16), 5, dtype=np.uint64)
    rows = disagree.locate(
        disagree.rows(contingency(a, b, ignore_a=())), _piece(a), _piece(b))
    assert {r["kind"] for r in rows} == {"merge"}
    for row in rows:
        assert row["point_at"] == "seam-merge"
        assert row["z_vox"] in (7, 8), "on the false join between body 1 and body 2"


def test_an_ignored_label_is_not_a_seam_partner():
    """Membrane (ignored segment 0) is the body's ordinary surface, not a cut; the real cut
    is at z=8."""
    a = np.ones((16, 16, 16), dtype=np.uint64)
    b = np.full((16, 16, 16), 5, dtype=np.uint64)
    b[8:] = 6
    b[:, 0, :] = 0          # a membrane sheet through the body, ignored
    b[:, -1, :] = 0

    rows = disagree.locate(
        disagree.rows(contingency(a, b, ignore_a=(), ignore_b=(0,))),
        _piece(a), _piece(b), ignore_a=(), ignore_b=(0,))
    for row in rows:
        assert row["point_at"] == "seam-split"
        assert row["z_vox"] in (7, 8), "the real cut, not the membrane surface"
        assert row["y_vox"] not in (0, 15), "never on the ignored sheet"


def test_a_disconnected_fragment_falls_back_and_says_so():
    """Two pieces of one body that never touch have no seam to point at."""
    a = np.ones((16, 16, 16), dtype=np.uint64)
    b = np.full((16, 16, 16), 5, dtype=np.uint64)
    b[12:] = 6                      # segment 6 is separated from 5 by nothing...
    a[10:12] = 0                    # ...except a gap in the BODY, so no contact in `a`
    rows = disagree.locate(
        disagree.rows(contingency(a, b)), _piece(a), _piece(b), ignore_a=(0,))
    six = [r for r in rows if r["b_id"] == 6]
    assert six, "the fixture must keep segment 6's row"
    assert six[0]["point_at"] == "overlap"
    assert int(b[six[0]["z_vox"], six[0]["y_vox"], six[0]["x_vox"]]) == 6


def test_cropping_to_the_pair_changes_nothing(monkeypatch):
    """`locate` crops each row's work to its two labels; the result must match the
    uncropped path voxel for voxel."""
    rng = np.random.default_rng(0)
    a = rng.integers(1, 7, size=(24, 24, 24), dtype=np.uint64)
    b = rng.integers(1, 9, size=(24, 24, 24), dtype=np.uint64)
    # plus a structured pair, so it is not only noise
    a[4:12, 4:12, 4:12] = 100
    b[4:12, 4:12, 8:12] = 200
    b[4:12, 4:12, 4:8] = 201
    rows = disagree.rows(contingency(a, b, ignore_a=()), top=30)

    cropped = disagree.locate(rows, _piece(a), _piece(b), ignore_a=())
    whole = disagree.locate(rows, _piece(a), _piece(b), ignore_a=(), crop=False)
    assert len(cropped) == len(whole)
    for c, w in zip(cropped, whole):
        assert c["pair_key"] == w["pair_key"]
        assert (c["z_vox"], c["y_vox"], c["x_vox"]) == (w["z_vox"], w["y_vox"], w["x_vox"])
        assert c["point_at"] == w["point_at"]


def test_a_label_missing_from_the_box_map_falls_back_to_the_whole_array():
    """No box for a label means the whole array, not a guess."""
    a = np.ones((8, 8, 8), dtype=np.uint64)
    b = np.full((8, 8, 8), 5, dtype=np.uint64)
    b[4:] = 6
    rows = disagree.rows(contingency(a, b, ignore_a=()))
    assert disagree._window(None, (slice(0, 2),) * 3, (8, 8, 8)) is None
    assert disagree.locate(rows, _piece(a), _piece(b), ignore_a=())


def test_an_unknown_at_is_refused_by_name():
    a, b = _slab()
    rows = disagree.rows(contingency(a, b, ignore_a=()))
    with pytest.raises(ValueError, match="'seam' or 'overlap'"):
        disagree.locate(rows, _piece(a), _piece(b), at="middle")


def test_a_c_shaped_region_gets_a_point_inside_it():
    """The case a centroid gets wrong: the middle of a C is outside the C."""
    a = np.ones((3, 21, 21), dtype=np.uint64)
    b = np.full((3, 21, 21), 5, dtype=np.uint64)
    b[:, 2:19, 2:6] = 6            # left upright
    b[:, 2:6, 2:19] = 6            # top bar
    b[:, 15:19, 2:19] = 6          # bottom bar
    rows = disagree.locate(
        disagree.rows(contingency(a, b, ignore_a=())), _piece(a), _piece(b))
    six = next(r for r in rows if r["b_id"] == 6)
    assert int(b[six["z_vox"], six["y_vox"], six["x_vox"]]) == 6


def test_world_coordinates_come_from_the_frame():
    a = np.ones((4, 4, 4), dtype=np.uint64)
    b = np.full((4, 4, 4), 5, dtype=np.uint64)
    b[2:] = 6
    pa = _piece(a)
    rows = disagree.locate(
        disagree.rows(contingency(a, b, ignore_a=())), pa, _piece(b))
    for row in rows:
        expected = pa.to_nm([[row["z_vox"], row["y_vox"], row["x_vox"]]])[0]
        assert (row["z_nm"], row["y_nm"], row["x_nm"]) == pytest.approx(tuple(expected))


def test_the_strided_fallback_still_lands_inside_the_pair():
    """Forced by a tiny cap, so the approximate path is exercised."""
    a = np.ones((12, 12, 12), dtype=np.uint64)
    b = np.full((12, 12, 12), 5, dtype=np.uint64)
    b[3:9, 3:9, 3:9] = 6
    rows = disagree.locate(
        disagree.rows(contingency(a, b, ignore_a=())), _piece(a), _piece(b),
        max_edt_voxels=8)
    for row in rows:
        z, y, x = row["z_vox"], row["y_vox"], row["x_vox"]
        assert int(b[z, y, x]) == row["b_id"]


def test_locating_against_the_wrong_pieces_says_so():
    a = np.ones((4, 4, 4), dtype=np.uint64)
    b = np.full((4, 4, 4), 5, dtype=np.uint64)
    b[2:] = 6
    rows = disagree.rows(contingency(a, b, ignore_a=()))
    other = _piece(np.full((4, 4, 4), 99, dtype=np.uint64))
    with pytest.raises(ValueError, match="not the ones the table was built from"):
        disagree.locate(rows, other, _piece(b))


def test_mismatched_piece_shapes_are_refused():
    a = np.ones((4, 4, 4), dtype=np.uint64)
    b = np.full((4, 4, 4), 5, dtype=np.uint64)
    b[2:] = 6
    rows = disagree.rows(contingency(a, b, ignore_a=()))
    with pytest.raises(ValueError, match="same shape"):
        disagree.locate(rows, _piece(a), _piece(np.ones((4, 4, 5), np.uint64)))


# -- shapes: what the error IS, not only where ------------------------------------------

def _shaped(a, b, **kw):
    return disagree.locate(disagree.rows(contingency(a, b, ignore_a=())),
                           _piece(a), _piece(b), shapes=True, **kw)


def test_a_merge_is_a_line_from_one_reference_body_to_the_other():
    """One end in each body, both ends in the one segment."""
    a = np.ones((16, 16, 16), dtype=np.uint64)
    a[8:] = 2
    b = np.full((16, 16, 16), 5, dtype=np.uint64)
    for row in _shaped(a, b):
        (s,) = row["shapes"]
        assert s["side"] == "merge"
        p0, p1 = tuple(s["p0_vox"]), tuple(s["p1_vox"])
        assert {int(a[p0]), int(a[p1])} == {1, 2}, "one end in each reference body"
        assert int(b[p0]) == int(b[p1]) == 5, "both ends in the merging segment"
        assert int(a[p0]) == int(row["a_id"]), "p0 is this row's own body"
        assert s["other_id"] == str(3 - int(row["a_id"]))
        assert abs(s["centre_nm"][0] / 8.0 - 7.5) <= 1.0, "centred on the join at z=8"


def test_a_split_is_an_ellipsoid_flat_across_the_cut():
    """The cut is a z-plane, so the ellipsoid is thin in z and wide in y and x."""
    a, b = _slab()
    six = next(r for r in _shaped(a, b) if r["b_id"] == 6)
    (s,) = six["shapes"]
    assert s["side"] == "split" and s["other_id"] == "5"
    rz, ry, rx = (r / 8.0 for r in s["radii_nm"])
    assert rz < ry and rz < rx, "flat across the cut"
    assert int(b[tuple(s["p0_vox"])]) == 6 and int(b[tuple(s["p1_vox"])]) == 5
    assert int(a[tuple(s["p0_vox"])]) == int(a[tuple(s["p1_vox"])]) == 1


def test_the_largest_patch_wins_not_a_stray_contact():
    """Segments meeting over a plane and at one stray voxel: the shape sits on the plane."""
    a = np.ones((16, 16, 16), dtype=np.uint64)
    b = np.full((16, 16, 16), 5, dtype=np.uint64)
    b[8:] = 6
    b[2, 2, 2] = 6                                     # the stray contact, far from z=8
    six = next(r for r in _shaped(a, b) if r["b_id"] == 6)
    (s,) = six["shapes"]
    assert abs(s["centre_nm"][0] / 8.0 - 7.5) <= 1.0


def test_a_cut_drawn_as_membrane_is_still_a_cut():
    """With membrane as an ignored 0, the two pieces never touch; bridging `gap` ignored
    voxels finds the cut, and gap=0 finds nothing."""
    a = np.ones((16, 16, 16), dtype=np.uint64)
    b = np.full((16, 16, 16), 5, dtype=np.uint64)
    b[8:] = 6
    b[8] = 0                                           # a one-voxel membrane at z=8
    rows = disagree.rows(contingency(a, b, ignore_a=(), ignore_b=(0,)))
    loc = lambda gap: disagree.locate(rows, _piece(a), _piece(b), ignore_b=(0,),  # noqa: E731
                                      shapes=True, gap=gap)
    five = next(r for r in loc(2) if r["b_id"] == 5)
    (s,) = five["shapes"]
    assert s["other_id"] == "6"
    assert abs(s["centre_nm"][0] / 8.0 - 8.0) <= 1.0, "on the membrane between them"
    assert next(r for r in loc(0) if r["b_id"] == 5)["shapes"] == []


def test_a_tangle_gets_both_shapes():
    a = np.ones((16, 16, 16), dtype=np.uint64)
    a[:, 8:] = 2
    b = np.full((16, 16, 16), 5, dtype=np.uint64)
    b[8:] = 6
    b[:4, :4] = 7                                      # enough to make it a tangle
    rows = [r for r in _shaped(a, b) if r["kind"] == "tangle"]
    assert rows, "the fixture must produce a tangle"
    assert {s["side"] for s in rows[0]["shapes"]} == {"split", "merge"}


def test_sample_ids_names_what_every_layer_has_at_the_anchors():
    """Each labelling names what it has at the two ends. A third that agrees with the
    reference names two objects, which shows it did not make this merge."""
    a = np.ones((16, 16, 16), dtype=np.uint64)
    a[8:] = 2
    b = np.full((16, 16, 16), 5, dtype=np.uint64)
    third = np.where(a == 1, 70, 80).astype(np.uint64)
    third[0] = 0                                       # membrane, never named
    rows = disagree.sample_ids(_shaped(a, b), {"gt": _piece(a), "seg": _piece(b),
                                              "other": _piece(third)},
                               ignore={"other": (0,)})
    ids = rows[0]["shapes"][0]["ids"]
    assert ids["gt"] == ["1", "2"] and ids["seg"] == ["5"] and ids["other"] == ["70", "80"]


def test_sample_ids_refuses_pieces_on_different_grids():
    a = np.ones((4, 4, 4), dtype=np.uint64)
    with pytest.raises(ValueError, match="one grid"):
        disagree.sample_ids([], {"x": _piece(a), "y": _piece(np.ones((4, 4, 5)))})


def test_without_shapes_the_rows_are_as_before():
    a, b = _slab()
    rows = disagree.rows(contingency(a, b, ignore_a=()))
    assert all("shapes" not in r for r in disagree.locate(rows, _piece(a), _piece(b)))


def test_a_boundary_sliver_is_not_taken_for_the_other_half_of_a_split():
    """Boundaries drawn a voxel apart put a strip of the neighbour's segment inside the
    body, possibly with more contact than the true cut. The other half of a split must be
    a piece mostly inside the body."""
    a = np.full((16, 16, 16), 1, dtype=np.uint64)
    a[:, :, 12:] = 2                                    # body 1 is x<12, body 2 beyond
    b = np.full((16, 16, 16), 5, dtype=np.uint64)       # segment 5: most of body 1
    b[15:, :, :12] = 6                                  # segment 6: body 1's other piece,
    b[:, :, 12:] = 9                                    #   thin, so the strip outweighs it
    b[:15, :, 11] = 9                                   # segment 9: body 2, poking one voxel
                                                        #   into body 1 along its boundary
    rows = _shaped(a, b)
    five = next(r for r in rows if r["a_id"] == 1 and r["b_id"] == 5)
    (s,) = [s for s in five["shapes"] if s["side"] == "split"]
    assert s["other_id"] == "6", "the true second piece, not the boundary strip"
    assert abs(s["centre_nm"][0] / 8.0 - 14.5) <= 1.5, "on the cut at z=15"


# -- the thin-pair gate: boundary disagreements are not structure -----------------------

def _strip_fixture():
    """Body 1 (x<12) cut at z=8 into segments 5 and 6, a real split, and segment 9 (body 2)
    reaching one voxel into body 1 along their boundary."""
    a = np.full((16, 16, 16), 1, dtype=np.uint64)
    a[:, :, 12:] = 2
    b = np.full((16, 16, 16), 5, dtype=np.uint64)
    b[8:, :, :12] = 6
    b[:, :, 12:] = 9
    b[:, :, 11] = 9
    return a, b


def test_thin_pairs_finds_the_strip_and_not_the_real_piece():
    a, b = _strip_fixture()
    c = contingency(a, b, ignore_a=())
    thin = disagree.thin_pairs(c, _piece(a), _piece(b), radius=3)
    assert "1:9" in thin and thin["1:9"] <= 1.0, "a one-voxel strip has no interior"
    assert "1:6" not in thin and "1:5" not in thin, "the halves of a real split are thick"


def test_excluding_thin_pairs_removes_the_false_merge_and_keeps_the_split():
    """The strip is dropped from partner counting too, so segment 9 is no longer a merge."""
    a, b = _strip_fixture()
    c = contingency(a, b, ignore_a=())
    before = disagree.rows(c)
    assert any(r["b_id"] == 9 for r in before), "the fixture must produce the false merge"
    after = disagree.rows(c, exclude=disagree.thin_pairs(c, _piece(a), _piece(b)))
    assert not any(r["b_id"] == 9 for r in after)
    assert {(r["a_id"], r["b_id"], r["kind"]) for r in after} >= {(1, 5, "split"),
                                                                    (1, 6, "split")}


def test_boundary_band_has_the_width_asked_for():
    from neu_eval.overlap import boundary_band

    a = np.ones((4, 4, 10), dtype=np.uint64)
    a[:, :, 5:] = 2
    assert boundary_band(a, 1)[0, 0].nonzero()[0].tolist() == [4, 5]
    assert boundary_band(a, 2)[0, 0].nonzero()[0].tolist() == [3, 4, 5, 6]
