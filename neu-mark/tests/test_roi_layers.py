"""DVID ROIs as precomputed segmentation layers: spec, overlap resolution, writing."""

import json

import numpy as np
import pytest

from neu_mark import roi_layers as rl


def _ranges(mask, lo=(0, 0, 0)):
    """A boolean array as DVID ranges: ``[z, y, x0, x1]`` with ``x1`` inclusive."""
    out = []
    for z, y in zip(*np.nonzero(mask.any(axis=2))):
        row = np.flatnonzero(mask[z, y])
        breaks = np.flatnonzero(np.diff(row) > 1)
        for s, e in zip(np.r_[0, breaks + 1], np.r_[breaks, len(row) - 1]):
            out.append([z + lo[0], y + lo[1], row[s] + lo[2], row[e] + lo[2]])
    return np.array(out, dtype=np.int64)


def _box(shape, lo, hi, *, grid=(12, 12, 24)):
    """A region holding the box ``[lo, hi)`` of a ``grid``-sized array."""
    m = np.zeros(grid, bool)
    m[tuple(slice(a, b) for a, b in zip(lo, hi))] = True
    return rl.region_from_ranges(_ranges(m))


def _spec(**raw):
    return rl.parse_spec(raw)


# --------------------------------------------------------------------------- #
# masks
# --------------------------------------------------------------------------- #
def test_ranges_decode_with_an_inclusive_end():
    m = np.zeros((3, 4, 9), bool)
    m[1, 2, 2:5] = m[2, 0, 7:9] = m[0, 3, 0] = True
    region = rl.region_from_ranges(_ranges(m, lo=(10, 20, 30)))
    assert region.lo.tolist() == [10, 20, 30]
    np.testing.assert_array_equal(region.mask, m)
    assert region.size == 6


# --------------------------------------------------------------------------- #
# spec
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw, match", [
    ({"layers": []}, "unknown section"),
    ({"layer": [{"name": "a", "rois": ["A"], "tree": "top"}]}, "exactly one of"),
    ({"layer": [{"name": "a", "rois": ["A"]}]}, r"no id for \['A'\]"),
    ({"layer": [{"name": "a", "rois": ["A", "B"]}], "ids": {"A": 1, "B": 1}}, "used twice"),
    ({"layer": [{"name": "a", "from": "b", "label": "x"}], "ids": {"x": 1}}, "not a layer"),
    ({"layer": [{"name": "a", "rois": ["A"], "fill_gaps": True}], "ids": {"A": 1}},
     "tree = 'parts' layer only"),
    ({"layer": [{"name": "a", "rois": ["A"], "overlap": "last"}], "ids": {"A": 1}},
     "overlap must be one of"),
])
def test_a_malformed_spec_is_refused(raw, match):
    with pytest.raises(rl.SpecError, match=match):
        rl.parse_spec(raw)


def test_every_roi_on_the_node_must_be_placed_or_ignored():
    spec = _spec(layer=[{"name": "n", "tree": "top"}],
                 neuropils={"T": ["P"], "U": []}, ids={"T": 1, "P": 2, "U": 3},
                 ignore={"X": "empty"})
    # T has a part, so the node need not have a T of its own
    assert rl.check_against_node(spec, ["P", "U", "X"], {"P": 4, "U": 5}) == []
    problems = rl.check_against_node(spec, ["P", "X", "NEW"], {"P": 0})
    assert problems == ["not on the node: U", "empty on the node: P",
                        "on the node but in no layer and not in [ignore]: NEW"]


# --------------------------------------------------------------------------- #
# painting one layer
# --------------------------------------------------------------------------- #
def _grid(*regions):
    whole = rl.union(list(regions))
    return rl.Grid(whole.lo, tuple(whole.mask.shape))


def test_split_divides_an_overlap_down_the_middle():
    a, b = _box(None, (0, 0, 0), (12, 12, 10)), _box(None, (0, 0, 6), (12, 12, 16))
    grid = _grid(a, b)
    labels, pairs = rl.paint(grid, [("A", 1, a), ("B", 2, b)], policy="split",
                             max_overlap=0.5)
    assert pairs == [("A", "B", 12 * 12 * 4)]
    # x = 6..9 is contested; each block goes to the ROI it is deeper inside
    assert labels[6, 6, :].tolist() == [1] * 8 + [2] * 8


def test_order_gives_an_overlap_to_the_roi_listed_first():
    a, b = _box(None, (0, 0, 0), (12, 12, 10)), _box(None, (0, 0, 6), (12, 12, 16))
    labels, _ = rl.paint(_grid(a, b), [("B", 2, b), ("A", 1, a)], policy="order",
                         max_overlap=0.5)
    assert labels[6, 6, :].tolist() == [1] * 6 + [2] * 10


def test_error_refuses_any_overlap():
    a, b = _box(None, (0, 0, 0), (12, 12, 10)), _box(None, (0, 0, 9), (12, 12, 16))
    with pytest.raises(ValueError, match="1 overlapping pair"):
        rl.paint(_grid(a, b), [("A", 1, a), ("B", 2, b)], policy="error", max_overlap=0.5)


def test_siblings_that_nest_are_refused_under_every_policy():
    a, b = _box(None, (0, 0, 0), (12, 12, 16)), _box(None, (2, 2, 2), (6, 6, 6))
    for policy in rl.POLICIES:
        with pytest.raises(rl.NestedRois, match="100% of the smaller"):
            rl.paint(_grid(a, b), [("A", 1, a), ("B", 2, b)], policy=policy,
                     max_overlap=0.2)


# --------------------------------------------------------------------------- #
# the neuropil tree, and a layer built from another
# --------------------------------------------------------------------------- #
@pytest.fixture
def tree():
    """T: its own ROI x 0..12, with parts P1 x 0..4 and P2 x 7..12 (a gap at 4..7);
    U: no parts, x 10..20, overlapping T at 10..12."""
    regions = {"T": _box(None, (0, 0, 0), (12, 12, 12)),
               "P1": _box(None, (0, 0, 0), (12, 12, 4)),
               "P2": _box(None, (0, 0, 7), (12, 12, 12)),
               "U": _box(None, (0, 0, 10), (12, 12, 20))}
    spec = _spec(build={"max_overlap": 0.5},
                 layer=[{"name": "top", "tree": "top"},
                        {"name": "parts", "tree": "parts", "fill_gaps": True},
                        {"name": "bare", "tree": "parts"},
                        {"name": "all", "from": "top", "label": "every", "close_blocks": 1}],
                 neuropils={"T": ["P1", "P2"], "U": []},
                 ids={"T": 1, "P1": 2, "P2": 3, "U": 4, "every": 9})
    return spec, regions


def test_parts_are_cut_from_the_top_level_so_the_layers_nest(tree):
    spec, regions = tree
    _grid, out = rl.resolve(spec, regions)
    top, parts = out["top"]["labels"], out["parts"]["labels"]
    assert out["top"]["overlaps"] == [("T", "U", 12 * 12 * 2)]
    row = top[6, 6, :].tolist()
    assert row == [1] * 11 + [4] * 9                       # the T/U overlap, split
    assert np.array_equal(top != 0, parts != 0)
    # P2 keeps only what T kept; the gap keeps T's id; U is its own part
    assert parts[6, 6, :].tolist() == [2] * 4 + [1] * 3 + [3] * 4 + [4] * 9
    for ident in np.unique(parts[parts != 0]):
        owner = np.unique(top[parts == ident])
        assert len(owner) == 1                              # each part inside one top


def test_without_fill_gaps_the_uncovered_blocks_are_empty(tree):
    spec, regions = tree
    _grid, out = rl.resolve(spec, regions, only=["bare"])
    assert set(out) == {"bare"}
    assert out["bare"]["labels"][6, 6, :].tolist() == [2] * 4 + [0] * 3 + [3] * 4 + [4] * 9


def test_a_layer_built_from_another_is_one_closed_label(tree):
    spec, regions = tree
    regions = {**regions, "U": _box(None, (0, 0, 13), (12, 12, 20))}   # a 1-block gap
    _grid, out = rl.resolve(spec, regions)
    every = out["all"]["labels"]
    assert np.unique(every).tolist() == [0, 9]
    assert out["top"]["labels"][6, 6, 12] == 0 and every[6, 6, 12] == 9
    assert np.all(every[out["top"]["labels"] != 0] == 9)


# --------------------------------------------------------------------------- #
# segment properties
# --------------------------------------------------------------------------- #
def test_descriptions_carry_the_side_and_any_exclusion():
    names = {"VLNP": "ventrolateral neuropils", "AOTU": "anterior optic tubercle",
             "CX": "central complex"}
    assert rl.describe("VLNP(L)", names) == "ventrolateral neuropils (left)"
    assert rl.describe("VLNP(-AOTU)(R)", names) == (
        "ventrolateral neuropils, without anterior optic tubercle (right)")
    assert rl.describe("CX", names) == "central complex"
    assert rl.describe("GL01(L)", names) == ""


def test_properties_name_tag_and_measure_each_segment(tree):
    spec, regions = tree
    _grid, out = rl.resolve(spec, regions)
    info = rl.properties(spec, "parts", out["parts"]["labels"], (256.0,) * 3)
    inline = info["inline"]
    assert info["@type"] == "neuroglancer_segment_properties"
    assert inline["ids"] == ["1", "2", "3", "4"]
    label, _desc, tags, volume = inline["properties"]
    assert label["values"] == ["T", "P1", "P2", "U"]
    vocab = tags["tags"]
    p1_tags = {vocab[i] for i in tags["values"][1]}
    assert p1_tags == {"layer:parts", "parent:T"}
    assert volume["values"][1] == round(12 * 12 * 4 * 256.0 ** 3 / 1e9, 3)


# --------------------------------------------------------------------------- #
# the node URL
# --------------------------------------------------------------------------- #
def test_the_source_is_a_node_not_an_instance():
    spec = rl.parse_node_url("dvid://dvid.example.org:8000/abc123")
    assert (spec["server"], spec["uuid"]) == ("dvid.example.org:8000", "abc123")
    with pytest.raises(ValueError, match="drop the instance and pass "
                                         "dvid://dvid.example.org/abc123"):
        rl.parse_node_url("dvid://dvid.example.org/abc123/synapses")


# --------------------------------------------------------------------------- #
# writing
# --------------------------------------------------------------------------- #
@pytest.fixture
def written(tmp_path, tree, monkeypatch):
    """A spec file, a small segmentation to sit beside, and the fetch stubbed."""
    from neu_lib import Frame, Piece
    from neu_vol import convert, write_piece

    spec, regions = tree
    spec_path = tmp_path / "rois.toml"
    spec_path.write_text("""
[build]
max_overlap = 0.5
[[layer]]
name = "top"
tree = "top"
[[layer]]
name = "parts"
tree = "parts"
fill_gaps = true
[neuropils]
T = ["P1", "P2"]
U = []
[names]
T = "the T neuropil"
[ids]
T = 1
P1 = 2
P2 = 3
U = 4
""")
    like = str(tmp_path / "seg")
    write_piece(Piece(np.zeros((64, 64, 64), np.uint64), Frame((8.0,) * 3), kind="segmentation",
                      name="seg"), str(tmp_path / "seg.h5"))
    convert(str(tmp_path / "seg.h5"), like, kind="segmentation", profile="local-neuroglancer",
            chunk=(32, 32, 32), multiscale=False, progress_path=str(tmp_path / "p.jsonl"))
    monkeypatch.setattr(rl, "fetch", lambda src, spec, **kw: {
        "sizes": {n: r.size for n, r in regions.items()}, "regions": regions,
        "provenance": {"source": "dvid", "uuid": "abc123"}})
    return {"spec": str(spec_path), "like": like, "dst": str(tmp_path / "rois")}


def test_build_writes_each_layer_beside_the_segmentation(written):
    from neu_vol import read_piece

    rec = rl.build("dvid://dvid.example.org/abc123", written["spec"], written["like"],
                   written["dst"])
    dst = written["dst"]
    piece = read_piece(f"{dst}/parts")
    assert piece.frame.voxel_size_nm == (256.0, 256.0, 256.0)
    assert piece.array[6, 6, :].tolist() == [2] * 4 + [1] * 3 + [3] * 4 + [4] * 9
    info = json.load(open(f"{dst}/top/info"))
    assert info["segment_properties"] == "segment_properties"
    props = json.load(open(f"{dst}/top/segment_properties/info"))["inline"]
    assert props["properties"][1]["values"] == ["the T neuropil", ""]
    on_disk = json.load(open(f"{dst}/rois.json"))
    assert on_disk["source"]["uuid"] == "abc123" and on_disk["ids"]["P2"] == 3
    assert on_disk["layers"]["top"]["overlaps"] == [["T", "U", 288]]
    assert rec["grid"]["past_like_extent_blocks"] == [10, 10, 18]


def test_build_refuses_to_replace_a_layer_unless_told(written):
    args = ("dvid://dvid.example.org/abc123", written["spec"], written["like"],
            written["dst"])
    rl.build(*args, layers=["top"])
    with pytest.raises(FileExistsError, match=r"\['top'\] already exist"):
        rl.build(*args)
    rl.build(*args, overwrite=True)


def test_adding_one_layer_keeps_the_record_of_the_others(written):
    """`--layers NAME` on a populated destination adds NAME and keeps rois.json whole."""
    args = ("dvid://dvid.example.org/abc123", written["spec"], written["like"],
            written["dst"])
    rl.build(*args, layers=["top"])
    rl.build(*args, layers=["parts"])
    on_disk = json.load(open(f"{written['dst']}/rois.json"))
    assert set(on_disk["layers"]) == {"top", "parts"}
    assert on_disk["rebuilt"] == ["parts"]


def test_a_dry_run_writes_nothing(written, tmp_path):
    rec = rl.build("dvid://dvid.example.org/abc123", written["spec"], written["like"],
                   written["dst"], dry_run=True)
    assert set(rec["layers"]) == {"top", "parts"}
    assert not (tmp_path / "rois").exists()


def test_the_cli_reports_each_layer(written, capsys):
    from neu_mark import cli

    assert cli.main(["roi-layers", "--src", "dvid://dvid.example.org/abc123",
                     "--spec", written["spec"], "--like", written["like"],
                     "--dst", written["dst"], "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "top                     2 segments, 1 overlapping pair(s)" in out
    assert "dry run: nothing written" in out
