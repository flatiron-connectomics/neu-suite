"""`gen --config`: site display defaults, checked before they reach a state."""

import pytest

from neu_glance import cli
from neu_glance.config import ConfigProblem, apply, parse
from neu_glance.state import parse_url

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
