"""Resolving sets to body ids, and giving them colours."""

import io

import pytest

from conftest import make_spec
from neu_movie import palette, sets

pytest.importorskip("neu_glance")

PROPERTIES = {
    "ids": ["10", "11", "12", "13", "20", "30"],
    "labels": ["Tm1", "Tm2", "Tm3", "Tm4", "Mi1", "Mi2"],
    "tags": [[], [], ["fragment"], [], [], []],
    "numbers": {"voxels": [50, 40, 30, 20, 10, 5]},
    "vocabulary": ["fragment"],
}


@pytest.fixture
def meshes(monkeypatch):
    """Every body has a mesh except those in `meshes.missing`."""
    import neu_glance.sources as src

    class Meshes:
        missing = set()
        checked = []

    def with_meshes(volume, ids):
        Meshes.checked.append(list(ids))
        return [b for b in ids if b not in Meshes.missing]

    monkeypatch.setattr(src, "segments_with_meshes", with_meshes)
    return Meshes


def resolve(spec, **kw):
    log = io.StringIO()
    return sets.resolve(spec, "s3://my-bucket/seg", properties=PROPERTIES, log=log, **kw), \
        log.getvalue()


def test_a_query_uses_the_spec_defaults(meshes):
    spec = make_spec([{"name": "Tm", "label": "^Tm"}],
                     neurons={"library": "lib", "query": {"without": ["fragment"]}})
    assert resolve(spec)[0] == {"Tm": ["10", "11", "13"]}


def test_exclude_applies_before_limit(meshes):
    spec = make_spec([{"name": "Tm", "label": "^Tm", "limit": 2, "exclude": [10]}])
    ids, _ = resolve(spec)
    assert ids == {"Tm": ["11", "12"]}


def test_a_global_limit_caps_every_set_and_ids_too(meshes):
    spec = make_spec([{"sets": [{"name": "Tm", "label": "^Tm", "limit": 3},
                                {"name": "L", "ids": [1, 2, 3]}]}])
    assert resolve(spec, limit=2)[0] == {"Tm": ["10", "11"], "L": ["1", "2"]}


def test_an_exclude_not_in_the_set_is_reported(meshes):
    spec = make_spec([{"name": "Mi", "label": "^Mi", "exclude": [99]}])
    ids, log = resolve(spec)
    assert ids == {"Mi": ["20", "30"]}
    assert "exclude 99 is not in the set" in log


def test_query_sets_drop_bodies_without_meshes_and_listed_ids_are_trusted(meshes):
    meshes.missing = {"20", "7"}
    spec = make_spec([{"sets": [{"name": "Mi", "label": "^Mi"}, {"name": "L", "ids": [7]}]}])
    ids, log = resolve(spec)
    assert ids == {"Mi": ["30"], "L": ["7"]}
    assert "(1 with no mesh, dropped)" in log
    assert meshes.checked == [["20", "30"]]


def test_empty_sets_are_left_out(meshes):
    spec = make_spec([{"sets": [{"name": "X", "label": "^nothing"}, {"name": "L", "ids": [7]}]}])
    ids, log = resolve(spec)
    assert ids == {"L": ["7"]}
    assert "no matches, skipped" in log


def test_palette_skips_dim_colours():
    assert "#8c564b" not in palette.SETS and "#666666" not in palette.SETS
    assert all(palette.luminance(c) >= palette.MIN_LUMINANCE for c in palette.SETS)
    assert palette.SETS[:2] == ["#1f77b4", "#ff7f0e"]


def test_colours_follow_first_appearance_and_explicit_ones_use_no_palette_slot():
    spec = make_spec([{"sets": [{"name": "A", "ids": [1]},
                                {"name": "B", "ids": [2], "color": "#123456"}]},
                      {"sets": [{"name": "C", "ids": [3]}, {"name": "A"}]}])
    assert sets.colours(spec) == {"A": palette.SETS[0], "B": "#123456", "C": palette.SETS[1]}


def test_clashes_count_sets_on_screen_together_with_their_fades(replace_spec):
    same = {"A": "#111111", "B": "#111111", "C": "#111111"}
    # A and B share beat two; C starts at 11, and B's fade-out ends at 9 + 1.
    assert sets.clashes(replace_spec, same) == [("A", "B", "#111111")]
    assert sets.clashes(replace_spec, same, ids={"A": [], "C": []}) == []


def test_clashes_in_accumulate_mode(accumulate_spec):
    assert len(sets.clashes(accumulate_spec, {"A": "#1", "B": "#2", "C": "#1"})) == 1


@pytest.mark.parametrize("colour", ["#1f77b4", "#ffffff", "#000000"])
def test_linear_hex_round_trips_through_the_srgb_curve(colour):
    lin = palette.rgb(palette.linear_hex(colour))
    back = [12.92 * v if v <= 0.0031308 else 1.055 * v ** (1 / 2.4) - 0.055 for v in lin]
    # Quantising the linear value to 8 bits loses most in the darks, a few steps at most.
    assert back == pytest.approx(palette.rgb(colour), abs=4 / 255)


def test_mute_moves_toward_grey():
    assert palette.mute("#ff0000", 0) == "#ff0000"
    assert palette.mute("#ff0000", 1) == palette.to_hex([0.6, 0.6, 0.6])


def test_delta_e_tells_the_same_colour_from_a_different_one():
    assert palette.delta_e("#1f77b4", "#1f77b4") == 0
    assert palette.delta_e("#d95f02", "#ff7f0e") < palette.MIN_DELTA_E     # two oranges
    assert palette.delta_e("#1b9e77", "#ff7f0e") > palette.MIN_DELTA_E     # teal, orange


def test_neuropils_keep_given_colours_and_prefer_ones_apart_from_the_sets():
    neurons = palette.SETS[:8]
    regions = [f"R{i}" for i in range(12)]
    # Given colours are kept.
    out = palette.neuropil_colours(regions, {"R0": "#d62728", "R1": "#17becf"}, neurons)
    assert out["R0"] == "#d62728" and out["R1"] == "#17becf"
    # The rest use each fitting colour once before any twice, those apart from every set
    # first; Set2's green and tan are too grey once muted.
    rest = [out[r] for r in regions[2:]]
    clear = ["#1b9e77", "#e7298a", "#e6ab02", "#a6761d", "#fc8d62"]
    assert rest[:5] == clear
    assert len(set(rest)) == len(rest) and "#66c2a5" not in rest


def test_a_neuropil_is_neither_grey_nor_in_its_own_neurons_hue():
    blue, orange = "#1f77b4", "#ff7f0e"
    own = {"LA": [blue], "ME": [blue], "LO": [blue], "AOTU": [orange]}
    out = palette.neuropil_colours(list(own), {"LO": "#ffa21f"}, [blue, orange], own)
    for region, colour in out.items():
        muted = palette.mute(colour, 0.5)
        assert palette.chroma(muted) >= palette.MIN_CHROMA          # not the head's grey
        assert all(palette.hue_gap(muted, c) >= palette.MIN_OWN_HUE for c in own[region])
    assert out["LO"] == "#ffa21f"             # given, so ME, chosen first, avoids its hue
    optic = [out[r] for r in ("LA", "ME", "LO")]
    assert all(palette.hue_gap(a, b) >= palette.MIN_SIBLING_HUE
               for i, a in enumerate(optic) for b in optic[i + 1:])
    assert palette.chroma(palette.mute("#8c564b", 0.5)) < palette.MIN_CHROMA   # muted brown
    assert palette.hue_gap(palette.mute("#17becf", 0.5), blue) < palette.MIN_OWN_HUE  # cyan


def test_a_given_neuropil_colour_is_kept_even_where_it_would_not_be_chosen():
    out = palette.neuropil_colours(["ME"], {"ME": "#17becf"}, ["#1f77b4"], {"ME": ["#1f77b4"]})
    assert out == {"ME": "#17becf"}
