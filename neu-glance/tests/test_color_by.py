"""`gen --color-by`: segmentColors from segment properties, and the short link to a state."""

import json

import pytest

from neu_glance import cli
from neu_glance.shaders import segment_color, value_color
from neu_glance.state import StateProblem, colors_by_property, parse_url

from test_gen import _volume

#: (id, label, tags) — 3 has no column and falls back to its group; 4 has neither.
ROWS = [("1", "Tm2_A2(L)", ("col:A2", "group:Tm2", "side:L")),
        ("2", "Tm2_A10(L)", ("col:A10", "col:A2", "group:Tm2")),
        ("3", "LN_C5(L)", ("group:LN",)),
        ("4", "4", ("glia",))]


def _props(rows=ROWS, numbers=None):
    vocabulary = sorted({t for *_, tags in rows for t in tags})
    return {"ids": [r[0] for r in rows], "labels": [r[1] for r in rows],
            "tags": [r[2] for r in rows], "numbers": numbers or {},
            "vocabulary": vocabulary}


def test_segment_color_is_neuroglancers():
    """Pinned values: the colours a headless neuroglancer drew for these ids (2026-10-22)."""
    assert segment_color(1) == "#5e80ff"
    assert segment_color(37) == "#ffe63e"
    # neuroglancer's seeds are random 32-bit values; a small one's bits rotate out of the
    # bytes the colour is read from
    assert segment_color(3) != segment_color(3, seed=0x9E3779B9)


def test_the_first_property_present_wins_and_each_value_has_one_colour():
    colors, report = colors_by_property(_props(), ["col", "group:"])
    assert colors["1"] == value_color("col:A2")
    assert colors["2"] == value_color("col:A10")       # several values: the first
    assert colors["3"] == value_color("group:LN")      # no column: falls back
    assert "4" not in colors                           # neither: keeps its hash colour
    assert report["by_key"] == {"col": 2, "group": 1}
    assert report["several_values"] == 1


def test_a_flag_and_the_label_are_properties_too():
    colors, _ = colors_by_property(_props(), ["glia", "label"])
    assert colors["4"] == value_color("glia:glia")
    assert colors["1"] == value_color("label:Tm2_A2(L)")
    # a segment with no label is labelled by its id, which is not a value
    colors, _ = colors_by_property(_props(), ["label"])
    assert "4" not in colors


@pytest.mark.parametrize("key, message", [("post", "number property"),
                                          ("colour", "not a property")])
def test_what_cannot_colour_is_refused(key, message):
    with pytest.raises(StateProblem, match=message):
        colors_by_property(_props(numbers={"post": [1, 2, 3, 4]}), [key])


def _with_properties(tmp_path, name="v"):
    vol = _volume(tmp_path, name, kind="segmentation", dtype="uint64")
    info_path = f"{vol}/info"
    info = json.load(open(info_path))
    info["segment_properties"] = "segment_properties"
    json.dump(info, open(info_path, "w"))
    props = _props()
    (tmp_path / name / "segment_properties").mkdir()
    (tmp_path / name / "segment_properties" / "info").write_text(json.dumps({
        "@type": "neuroglancer_segment_properties",
        "inline": {"ids": props["ids"], "properties": [
            {"id": "instance", "type": "label", "values": props["labels"]},
            {"id": "tags", "type": "tags", "tags": props["vocabulary"],
             "values": [sorted(props["vocabulary"].index(t) for t in tags)
                        for tags in props["tags"]]}]}}))
    return vol


def test_the_cli_writes_segment_colors_from_a_flag_or_the_config(tmp_path, capsys):
    vol = _with_properties(tmp_path)
    cli.cmd_gen(cli._parse_args(["gen", "--seg", vol, "--color-by", "v=col,group"]))
    out = capsys.readouterr()
    layer = parse_url(out.out.strip())["layers"][0]
    assert layer["segmentColors"]["3"] == value_color("group:LN")
    assert "coloured 3 of 4 segments by col,group" in out.err

    cfg = tmp_path / "c.toml"
    cfg.write_text('[layer.v]\ncolor_by = "group"\n')
    cli.cmd_gen(cli._parse_args(["gen", "--seg", vol, "--config", str(cfg)]))
    layer = parse_url(capsys.readouterr().out.strip())["layers"][0]
    assert layer["segmentColors"]["1"] == value_color("group:Tm2")


def test_color_by_names_a_seg_layer(tmp_path):
    vol = _with_properties(tmp_path)
    with pytest.raises(SystemExit, match="names no --seg layer"):
        cli.cmd_gen(cli._parse_args(["gen", "--seg", vol, "--color-by", "other=col"]))


# --------------------------------------------------------------------------- #
# a link to a state file on an object store
# --------------------------------------------------------------------------- #
STATE = {"dimensions": {"x": [8e-9, "m"]}, "layers": []}


@pytest.fixture
def cloud_state(monkeypatch):
    monkeypatch.setattr(cli, "_load_into", lambda source: json.loads(json.dumps(STATE)))
    return "s3://my-bucket/states/v1.json"


def test_a_cloud_state_becomes_a_link_naming_the_file(cloud_state, capsys):
    cli.cmd_gen(cli._parse_args(["gen", "--into", cloud_state, "--viewer", "https://ng.x/"]))
    assert capsys.readouterr().out.strip() == f"https://ng.x/#!{cloud_state}"


def test_expand_puts_the_state_in_the_link(cloud_state, capsys):
    cli.cmd_gen(cli._parse_args(["gen", "--into", cloud_state, "--expand"]))
    assert parse_url(capsys.readouterr().out.strip())["dimensions"] == STATE["dimensions"]


def test_with_expand_a_cloud_state_takes_layers_and_view_options(cloud_state, tmp_path,
                                                                 capsys):
    vol = _volume(tmp_path, "v", kind="segmentation", dtype="uint64")
    cli.cmd_gen(cli._parse_args(["gen", "--into", cloud_state, "--seg", vol,
                                 "--layout", "3d", "--expand"]))
    state = parse_url(capsys.readouterr().out.strip())
    assert state["layout"] == "3d"
    assert [lyr["name"] for lyr in state["layers"]] == ["v"]


def test_a_short_link_cannot_carry_changes(cloud_state):
    with pytest.raises(SystemExit, match="--expand"):
        cli.cmd_gen(cli._parse_args(["gen", "--into", cloud_state, "--layout", "3d"]))


def test_a_local_state_is_expanded(tmp_path, capsys):
    path = tmp_path / "s.json"
    path.write_text(json.dumps(STATE))
    cli.cmd_gen(cli._parse_args(["gen", "--into", str(path)]))
    assert parse_url(capsys.readouterr().out.strip())["dimensions"] == STATE["dimensions"]
