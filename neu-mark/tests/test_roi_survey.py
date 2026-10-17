"""Measuring a node's ROIs: containment, coverage, sibling overlaps, changes."""

import json

import numpy as np
import pytest

from neu_mark import roi_layers as rl
from neu_mark import roi_survey as rs


def _box(lo, hi, grid=(12, 12, 32)):
    m = np.zeros(grid, bool)
    m[tuple(slice(a, b) for a, b in zip(lo, hi))] = True
    return rl.Region(m[tuple(slice(a, b) for a, b in zip(lo, hi))].copy(), np.array(lo))


@pytest.fixture
def regions():
    """H holds everything. A (x 0..12) has parts a1 (x 0..4) and a2 (x 7..12), a gap
    between. B (x 11..24) touches A by one slab; B2 is B again. E (x 20..26) sticks out of
    B by a third."""
    return {"H": _box((0, 0, 0), (12, 12, 26)),
            "A": _box((0, 0, 0), (12, 12, 12)),
            "a1": _box((0, 0, 0), (12, 12, 4)),
            "a2": _box((0, 0, 7), (12, 12, 12)),
            "B": _box((0, 0, 11), (12, 12, 24)),
            "B2": _box((0, 0, 11), (12, 12, 24)),
            "E": _box((0, 0, 20), (12, 12, 26))}


def _record(regions, **kw):
    sizes = {n: r.size for n, r in regions.items()} | {"EMPTY": 0}
    return rs.survey(regions, sizes, **kw)


def test_each_roi_hangs_under_the_smallest_larger_roi_holding_most_of_it(regions):
    rois = _record(regions)["rois"]
    parents = {n: r.get("parent") for n, r in rois.items()}
    assert parents == {"H": None, "A": "H", "a1": "A", "a2": "A", "B": "H", "B2": "H",
                       "E": "B", "EMPTY": None}
    assert rois["E"]["inside"] == pytest.approx(4 / 6, abs=1e-4)
    assert rois["a1"]["inside"] == 1.0


def test_a_parent_reports_how_well_its_children_cover_it(regions):
    comps = _record(regions)["composites"]
    assert comps["A"] == {"children": ["a1", "a2"], "coverage": 0.75,
                          "gap_blocks": 144 * 3, "outside": 0.0, "outside_blocks": 0}
    assert comps["B"]["outside_blocks"] == 144 * 2


def test_siblings_are_overlaps_between_rois_that_do_not_nest(regions):
    rec = _record(regions, max_overlap=0.1)
    pairs = {(s["a"], s["b"]): s for s in rec["siblings"]}
    # A/a1 nest, B/B2 are identical: neither is a sibling overlap
    assert set(pairs) == {("A", "B"), ("A", "B2"), ("B", "a2"), ("B2", "a2")}
    assert pairs[("B", "a2")]["of_smaller"] == 0.2 and pairs[("B", "a2")]["too_much"]
    assert not pairs[("A", "B")]["too_much"]
    assert rec["identical"] == [["B", "B2"]]
    assert rec["empty"] == ["EMPTY"]


def test_extent_reports_how_far_each_roi_reaches_past_it(regions):
    rec = _record(regions, extent=(12, 12, 24))
    assert rec["past_extent"] == {"E": {"below_zyx": [0, 0, 0], "above_zyx": [0, 0, 2]},
                                  "H": {"below_zyx": [0, 0, 0], "above_zyx": [0, 0, 2]}}


def test_compare_counts_the_blocks_each_roi_gained_and_lost(regions):
    new = dict(regions, a1=_box((0, 0, 1), (12, 12, 5)))
    new.pop("E")
    old_sizes = {n: r.size for n, r in regions.items()}
    new_sizes = {n: r.size for n, r in new.items()} | {"NEW": 0}
    out = rs.compare(regions, new, old_sizes, new_sizes)
    assert out["added"] == ["NEW"] and out["removed"] == ["E"]
    assert out["changed"] == [{"roi": "a1", "before": 576, "after": 576, "gained": 144,
                               "lost": 144, "jaccard": 0.6}]
    assert out["unchanged"] == 5


def test_a_spec_is_checked_against_the_node(regions, tmp_path):
    spec = tmp_path / "rois.toml"
    spec.write_text("""
[[layer]]
name = "top"
tree = "top"
[[layer]]
name = "bs"
rois = ["B", "B2"]
[neuropils]
A = ["a1", "a2"]
B = ["E"]
[ignore]
H = "everything"
[ids]
A = 1
a1 = 2
a2 = 3
B = 4
E = 5
B2 = 6
""")
    sizes = {n: r.size for n, r in regions.items()}
    out = rs.check_spec(str(spec), regions, sizes, contained=0.95)
    assert out["problems"] == []
    assert out["parts_outside_top"] == [{"part": "E", "top": "B", "inside": 0.6667}]
    assert out["layers"]["top"]["overlaps"] == [["A", "B", 144]]
    assert "100% of the smaller" in out["layers"]["bs"]["refused"]


def test_the_cli_prints_the_tree_and_writes_the_record(regions, monkeypatch, capsys,
                                                       tmp_path):
    from neu_mark import cli

    sizes = {n: r.size for n, r in regions.items()}
    monkeypatch.setattr(rs, "fetch_all", lambda src, **kw: {
        "provenance": {"uuid": "abc123", "ref": "abc:main"}, "sizes": sizes,
        "regions": regions, "seconds": 1.0})
    out_json = tmp_path / "survey.json"
    assert cli.main(["roi-survey", "--src", "dvid://dvid.example.org/abc:main",
                     "--compare", "dvid://dvid.example.org/def456",
                     "--out", str(out_json)]) == 0
    out = capsys.readouterr().out
    assert "node abc123 (abc:main): 7 ROIs" in out
    assert "    A " in out and "children cover 75.0%" in out
    assert "      a1 " in out
    assert "66.7% inside" in out                                   # E in B
    assert "identical: B = B2" in out
    assert "0 changed, 7 unchanged" in out
    assert json.loads(out_json.read_text())["rois"]["a2"]["parent"] == "A"
