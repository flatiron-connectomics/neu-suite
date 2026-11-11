"""`gen --config`: site display defaults, checked before they reach a state."""

import json

import pytest

from neu_glance import cli
from neu_glance.config import ConfigProblem, apply, parse
from neu_glance.state import parse_url, unique_names

from test_gen import _annotation_source, _volume


def _gen(tmp_path, capsys, toml: str, *argv: str) -> dict:
    cfg = tmp_path / "ngl.toml"
    cfg.write_text(toml)
    assert cli.cmd_gen(cli._parse_args(["gen", *argv, "--config", str(cfg)])) == 0
    return parse_url(capsys.readouterr().out.strip())


def test_layer_settings_apply_by_name_and_a_flag_wins(tmp_path, capsys):
    a = _volume(tmp_path, "a", kind="segmentation", dtype="uint64")
    b = _volume(tmp_path, "b", kind="segmentation", dtype="uint64")
    state = _gen(tmp_path, capsys, """
[layer.a]
silhouette = 5
opacity_2d = 0
hide = true
[layer.b]
opacity_3d = 0.5
""", "--seg", a, "--seg", b, "--opacity-3d", "b=0.25")
    layers = {lyr["name"]: lyr for lyr in state["layers"]}
    assert layers["a"]["meshSilhouetteRendering"] == 5.0
    assert layers["a"]["selectedAlpha"] == 0.0
    assert layers["a"]["visible"] is False
    assert layers["b"]["objectAlpha"] == 0.25
    assert "visible" not in layers["b"]


def test_colour_and_pick_from_the_config_and_from_flags(tmp_path, capsys):
    a = _volume(tmp_path, "a", kind="segmentation", dtype="uint64")
    b = _volume(tmp_path, "b", kind="segmentation", dtype="uint64")
    state = _gen(tmp_path, capsys, '[layer.a]\ncolor = "white"\npick = false\n',
                 "--seg", a, "--seg", b, "--color", "b=#00FF00", "--no-pick", "b")
    layers = {lyr["name"]: lyr for lyr in state["layers"]}
    assert layers["a"]["segmentDefaultColor"] == "#ffffff"
    assert layers["a"]["pick"] is False
    assert layers["b"]["segmentDefaultColor"] == "#00ff00"
    assert layers["b"]["pick"] is False


@pytest.mark.parametrize("bounds", ["false", "true"])
def test_skeletons_off_switches_the_subsource_and_keeps_the_others(tmp_path, capsys, bounds):
    a = _volume(tmp_path, "a", kind="segmentation", dtype="uint64")
    state = _gen(tmp_path, capsys, f"[volume]\nbounds = {bounds}\n"
                                   "[layer.a]\nskeletons = false\n", "--seg", a)
    source = state["layers"][0]["source"]
    assert source["url"].startswith("precomputed://")
    assert source["subsources"] == ({"skeletons": False} if bounds == "true"
                                    else {"bounds": False, "skeletons": False})


def test_skeletons_is_a_switch_on_a_segmentation_layer():
    with pytest.raises(ConfigProblem, match="true or false"):
        parse({"layer": {"a": {"skeletons": "off"}}})
    with pytest.raises(ConfigProblem, match="only a segmentation"):
        apply(parse({"layer": {"em": {"skeletons": False}}}), [{"name": "em", "type": "image"}])


def test_a_colour_neuroglancer_would_ignore_is_refused():
    with pytest.raises(ConfigProblem, match="color"):
        parse({"layer": {"a": {"color": "not-a-colour-at-all"}}})
    with pytest.raises(ConfigProblem, match="true or false"):
        parse({"layer": {"a": {"pick": 0}}})


def test_a_layer_the_state_lacks_is_reported_not_refused(tmp_path, capsys):
    a = _volume(tmp_path, "a", kind="segmentation", dtype="uint64")
    cfg = tmp_path / "ngl.toml"
    cfg.write_text("[layer.elsewhere]\nhide = true\n")
    assert cli.cmd_gen(cli._parse_args(["gen", "--seg", a, "--config", str(cfg)])) == 0
    assert "[layer.elsewhere]: no such layer" in capsys.readouterr().err


def test_the_bounds_box_can_be_turned_back_on(tmp_path, capsys):
    a = _volume(tmp_path, "a", kind="segmentation", dtype="uint64")
    state = _gen(tmp_path, capsys, "[volume]\nbounds = true\n", "--seg", a)
    assert isinstance(state["layers"][0]["source"], str)


def test_shader_controls_must_be_ones_the_shader_declares(tmp_path, capsys):
    seg = _volume(tmp_path, "a", kind="segmentation", dtype="uint64")
    src = _annotation_source(tmp_path)
    state = _gen(tmp_path, capsys, '[annotations.shader_controls]\nline_color = "#ffffff"\n',
                 "--seg", seg, "--annotations", src)
    ann = [lyr for lyr in state["layers"] if lyr["type"] == "annotation"][0]
    assert ann["shaderControls"] == {"line_color": "#ffffff"}

    cfg = tmp_path / "bad.toml"
    cfg.write_text("[annotations.shader_controls]\nline_colour = '#ffffff'\n")
    with pytest.raises(SystemExit, match="line_colour"):
        cli.cmd_gen(cli._parse_args(["gen", "--seg", seg, "--annotations", src,
                                     "--config", str(cfg)]))


def test_all_segments_from_the_config_selects_the_listed_ids(tmp_path, capsys, monkeypatch):
    a = _volume(tmp_path, "a", kind="segmentation", dtype="uint64")
    monkeypatch.setattr(cli, "_listed_segments", lambda volume: [3, 4])
    state = _gen(tmp_path, capsys, "[layer.a]\nall_segments = true\n", "--seg", a)
    assert state["layers"][0]["segments"] == ["3", "4"]


@pytest.mark.parametrize("doc, message", [
    ({"colour": {}}, "unknown section"),
    ({"volume": {"bound": False}}, "unknown key"),
    ({"layer": {"a": {"silhouete": 4}}}, "unknown key"),
    ({"layer": {"a": {"opacity_2d": 2}}}, "between 0 and 1"),
    ({"layer": {"a": {"hide": "yes"}}}, "true or false"),
])
def test_what_neuroglancer_would_ignore_is_refused(doc, message):
    with pytest.raises(ConfigProblem, match=message):
        parse(doc)


def test_display_keys_on_a_non_segmentation_layer_are_refused():
    with pytest.raises(ConfigProblem, match="only a segmentation"):
        apply(parse({"layer": {"em": {"silhouette": 2}}}), [{"name": "em", "type": "image"}])


VIEW = """
[view]
position = [64.5, 32.0, 16.0]
projection_orientation = [-0.1, 0.7, 0.68, 0.12]
projection_scale = 1500.0
cross_section_scale = 2.5
select = "a"
[layer.a]
tab = "segments"
"""


def test_the_view_comes_from_the_config_as_neuroglancer_writes_it(tmp_path, capsys):
    a = _volume(tmp_path, "a", kind="segmentation", dtype="uint64")
    state = _gen(tmp_path, capsys, VIEW, "--seg", a)
    assert state["position"] == [64.5, 32.0, 16.0]          # xyz, unchanged
    assert state["projectionOrientation"] == [-0.1, 0.7, 0.68, 0.12]
    assert state["projectionScale"] == 1500.0 and state["crossSectionScale"] == 2.5
    assert state["selectedLayer"] == {"visible": True, "layer": "a"}
    assert state["layers"][0]["tab"] == "segments"


def test_view_flags_win_over_the_config(tmp_path, capsys):
    a = _volume(tmp_path, "a", kind="segmentation", dtype="uint64")
    state = _gen(tmp_path, capsys, VIEW, "--seg", a, "--position", "1,2,3",
                 "--projection-scale", "9", "--layout", "3d")
    assert state["position"] == [3.0, 2.0, 1.0]               # --position is zyx
    assert state["projectionScale"] == 9.0
    assert state["crossSectionScale"] == 2.5
    assert state["layout"] == "3d"


@pytest.mark.parametrize("doc, message", [
    ({"view": {"position": [1, 2]}}, "3 numbers"),
    ({"view": {"orientation": [0, 0, 0, 1]}}, "unknown key"),
    ({"view": {"projection_scale": "far"}}, "a number"),
    ({"layer": {"a": {"tab": "seg"}}}, "tab must be one of"),
])
def test_a_view_or_tab_neuroglancer_would_ignore_is_refused(doc, message):
    with pytest.raises(ConfigProblem, match=message):
        parse(doc)


def test_a_layer_is_renamed_last_from_the_config_or_a_flag(tmp_path, capsys):
    em = _volume(tmp_path, "em")
    a = _volume(tmp_path, "a", kind="segmentation", dtype="uint64")
    b = _volume(tmp_path, "b", kind="segmentation", dtype="uint64")
    state = _gen(tmp_path, capsys, """
[view]
select = "a"
[layer.em]
name = "EM"
[layer.a]
name = "neurons"
tab = "segments"
hide = true
[layer.b]
name = "from the config"
""", "--image", em, "--seg", a, "--seg", b, "--rename", "b=from the flag",
                 "--opacity-3d", "a=0.5")
    assert [lyr["name"] for lyr in state["layers"]] == ["EM", "neurons", "from the flag"]
    neurons = state["layers"][1]
    # the other settings still named the layer by its path
    assert neurons["tab"] == "segments" and neurons["objectAlpha"] == 0.5
    assert neurons["visible"] is False
    assert state["selectedLayer"] == {"visible": True, "layer": "neurons"}


def test_an_annotation_layer_follows_its_renamed_segmentation(tmp_path, capsys):
    seg = _volume(tmp_path, "a", kind="segmentation", dtype="uint64")
    src = _annotation_source(tmp_path)
    state = _gen(tmp_path, capsys, "", "--seg", seg, "--annotations", src,
                 "--presyn-postsyn", "--rename", "a=neurons")
    linked = [lyr["linkedSegmentationLayer"] for lyr in state["layers"]
              if lyr["type"] == "annotation"]
    assert len(linked) == 3
    assert all(set(links.values()) == {"neurons"} for links in linked)


def test_renaming_a_layer_of_an_into_state_keeps_it_selected(tmp_path, capsys):
    a = _volume(tmp_path, "a", kind="segmentation", dtype="uint64")
    saved = tmp_path / "state.json"
    assert cli.cmd_gen(cli._parse_args(["gen", "--seg", a, "--select", "a", "--format",
                                        "state", "--out", str(saved)])) == 0
    assert cli.cmd_gen(cli._parse_args(["gen", "--into", str(saved), "--rename", "a=A",
                                        "--format", "state"])) == 0
    state = json.loads(capsys.readouterr().out)
    assert [lyr["name"] for lyr in state["layers"]] == ["A"]
    assert state["selectedLayer"]["layer"] == "A"


def test_annotations_added_into_a_state_follow_their_suffixed_segmentation(tmp_path, capsys):
    """The state has a layer `a`, so the incoming `a` becomes `a-2`, and the synapses
    added with it link to it, not to the state's own."""
    first = _volume(tmp_path, "a", kind="segmentation", dtype="uint64")
    (tmp_path / "other").mkdir()
    second = _volume(tmp_path / "other", "a", kind="segmentation", dtype="uint64")
    src = _annotation_source(tmp_path)
    saved = tmp_path / "state.json"
    assert cli.cmd_gen(cli._parse_args(["gen", "--seg", first, "--format", "state",
                                        "--out", str(saved)])) == 0
    assert cli.cmd_gen(cli._parse_args(["gen", "--into", str(saved), "--seg", second,
                                        "--annotations", src, "--format", "state"])) == 0
    state = json.loads(capsys.readouterr().out)
    assert [lyr["name"] for lyr in state["layers"]] == ["a", "a-2", "syn"]
    assert set(state["layers"][2]["linkedSegmentationLayer"].values()) == {"a-2"}


def test_a_link_to_the_first_of_two_incoming_layers_stays_on_it():
    existing = [{"name": "x"}]
    incoming = [{"name": "x"}, {"name": "x"}, {"name": "syn",
                                               "linkedSegmentationLayer": {"pre": "x"}}]
    unique_names(existing, incoming)
    assert [lyr["name"] for lyr in incoming] == ["x-2", "x-3", "syn"]
    assert incoming[2]["linkedSegmentationLayer"] == {"pre": "x-2"}

    incoming = [{"name": "y"}, {"name": "y"}, {"name": "syn",
                                               "linkedSegmentationLayer": {"pre": "y"}}]
    unique_names(existing, incoming)
    assert [lyr["name"] for lyr in incoming] == ["y", "y-2", "syn"]
    assert incoming[2]["linkedSegmentationLayer"] == {"pre": "y"}


def test_a_rename_that_names_no_layer_or_clashes_is_refused(tmp_path):
    a = _volume(tmp_path, "a", kind="segmentation", dtype="uint64")
    b = _volume(tmp_path, "b", kind="segmentation", dtype="uint64")
    with pytest.raises(SystemExit, match="--rename names no layer"):
        cli.cmd_gen(cli._parse_args(["gen", "--seg", a, "--rename", "c=C"]))
    with pytest.raises(SystemExit, match="LAYER_NAME=NEW_NAME"):
        cli.cmd_gen(cli._parse_args(["gen", "--seg", a, "--rename", "a="]))
    with pytest.raises(SystemExit, match="more than one layer named 'b'"):
        cli.cmd_gen(cli._parse_args(["gen", "--seg", a, "--seg", b, "--rename", "a=b"]))
    with pytest.raises(ConfigProblem, match="non-empty string"):
        parse({"layer": {"a": {"name": ""}}})


def test_the_segments_tab_is_refused_on_an_image_layer():
    with pytest.raises(ConfigProblem, match="segmentation layer"):
        apply(parse({"layer": {"em": {"tab": "segments"}}}), [{"name": "em", "type": "image"}])
    apply(parse({"layer": {"em": {"tab": "rendering"}}}), [{"name": "em", "type": "image"}])
