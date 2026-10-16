"""Each body's synapse shares by neuropil, as columns segment-property rules can read."""

import pandas as pd
import pytest

from neu_mark import io, ops, roi_layers, rule, segprops, tables

POINTS = pd.DataFrame({
    "body": pd.Series([1, 1, 1, 1, 2, 3], dtype="uint64"),
    "kind": pd.Series(["PreSyn", "PreSyn", "PreSyn", "PostSyn", "PostSyn", "PreSyn"],
                      dtype="string"),
    "roi": pd.Series(["ME(L)", "ME(L)", None, "LO(L)", None, "CBL_EB"], dtype="string"),
})
TOP = {"ME(L)": "OL(L)", "LO(L)": "OL(L)", "CBL_EB": "CX"}


def test_shares_are_out_of_all_of_a_sides_synapses():
    shares = tables.body_neuropil_fractions(POINTS).set_index("body")["synapse_neuropils"]
    assert shares[1] == {"pre": {"ME(L)": 0.6667}, "post": {"LO(L)": 1.0},
                         "all": {"ME(L)": 0.5, "LO(L)": 0.25}}
    assert shares[2] == {"pre": {}, "post": {}, "all": {}}       # in no ROI at all


def test_top_level_shares_group_the_parts():
    out = tables.body_neuropil_fractions(POINTS, top_level=TOP).set_index("body")
    assert out.loc[1, "synapse_neuropils_top"]["all"] == {"OL(L)": 0.75}
    assert out.loc[3, "synapse_neuropils_top"]["pre"] == {"CX": 1.0}


def test_shares_run_largest_first():
    pts = pd.DataFrame({"body": [5] * 3, "kind": ["PreSyn"] * 3,
                        "roi": ["LO(L)", "ME(L)", "ME(L)"]})
    shares = tables.body_neuropil_fractions(pts)["synapse_neuropils"][0]
    assert list(shares["pre"]) == ["ME(L)", "LO(L)"]


def test_points_without_rois_are_refused():
    with pytest.raises(KeyError, match="--rois"):
        tables.body_neuropil_fractions(POINTS.drop(columns="roi"))


def test_every_body_gets_the_columns_and_a_rule_can_tag_by_them(tmp_path):
    io.write_table(POINTS, str(tmp_path / "syn"), "points")
    spec = tmp_path / "rois.toml"
    spec.write_text('[[layer]]\nname = "parts"\ntree = "parts"\n'
                    '[neuropils]\n"OL(L)" = ["ME(L)", "LO(L)"]\n"CX" = ["CBL_EB"]\n'
                    '[ids]\n"OL(L)" = 1\n"ME(L)" = 2\n"LO(L)" = 3\n"CX" = 4\n"CBL_EB" = 5\n')
    bodies = pd.DataFrame({"body": pd.Series([1, 3, 9], dtype="uint64"),
                           "instance": pd.Series(["a", "b", "c"], dtype="string")})
    frame = ops._with_neuropils(bodies, str(tmp_path / "syn"), str(spec))
    assert frame["body"].tolist() == [1, 3, 9]
    assert frame.loc[2, "synapse_neuropils"] == {"pre": {}, "post": {}, "all": {}}

    @rule(needs=("synapse_neuropils_top",))
    def region(r):
        """where most of the cell's synapses are"""
        shares = r["synapse_neuropils_top"]["all"]
        return next(iter(shares), None) if shares and max(shares.values()) > 0.5 else None

    built = segprops.build(frame, rules=[region], rules_only=True)
    props = {p["id"]: p for p in built["info"]["inline"]["properties"]}
    vocab = props["tags"]["tags"]
    tags = [{vocab[i] for i in v} for v in props["tags"]["values"]]
    assert tags[:2] == [{"region:OL(L)"}, {"region:CX"}]


def test_points_can_be_labelled_with_a_roi_layers_layer(tmp_path):
    from neu_mark import cli

    spec = tmp_path / "rois.toml"
    spec.write_text('[[layer]]\nname = "top"\ntree = "top"\n'
                    '[[layer]]\nname = "parts"\ntree = "parts"\n'
                    '[[layer]]\nname = "tracts"\nrois = ["GC", "mALT(L)"]\n'
                    '[neuropils]\n"OL(L)" = ["ME(L)", "LO(L)"]\n"MB(L)" = []\n'
                    '[ids]\n"OL(L)" = 1\n"ME(L)" = 2\n"LO(L)" = 3\n"MB(L)" = 4\n'
                    '"GC" = 5\n"mALT(L)" = 6\n')
    assert cli._roi_list(f"{spec}:parts") == ["ME(L)", "LO(L)", "MB(L)"]
    assert cli._roi_list(f"{spec}:tracts") == ["GC", "mALT(L)"]
    with pytest.raises(roi_layers.SpecError, match="built from other ROIs"):
        cli._roi_list(f"{spec}:top")
    with pytest.raises(roi_layers.SpecError, match="no layer 'nope'"):
        cli._roi_list(f"{spec}:nope")
