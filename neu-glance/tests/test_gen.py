"""Assembling a neuroglancer state and encoding it into a URL.

A wrong `dimensions` block or source scheme loads fine and shows the wrong thing, so these
check values, not just that a URL came out.
"""

import json
import urllib.parse

import numpy as np
import pytest

from neu_vol import convert
from neu_vol.backends.tensorstore import TensorStoreBackend
from neu_vol.profiles import zarr3_create_spec

from neu_glance import cli
from neu_glance.sources import SourceProblem, volume_layer
from neu_glance.state import (DEFAULT_VIEWER, StateProblem, build_state, load_layer,
                          parse_url, state_url)


def _volume(tmp_path, name, *, kind="image", profile="local-neuroglancer",
            voxel=(8, 8, 8), dtype="uint8", shape=(16, 16, 16)):
    data = np.ones(shape, dtype)
    src = str(tmp_path / f"{name}.src.zarr")
    be = TensorStoreBackend.create(
        zarr3_create_spec("local", src, data.shape, dtype,
                          dimension_names=("z", "y", "x"), chunk=(8, 8, 8)),
        delete_existing=True)
    be.write_region(tuple(slice(0, s) for s in data.shape), data)
    dst = str(tmp_path / name)
    convert(src, dst, voxel_size=voxel, kind=kind, profile=profile, chunk=(8, 8, 8),
            min_dim=8, delete_existing=True)
    return dst


# --------------------------------------------------------------------------- #
# layers
# --------------------------------------------------------------------------- #
def test_the_source_scheme_comes_from_the_format(tmp_path):
    """`precomputed://` or `zarr://`, from the volume's format."""
    pre = _volume(tmp_path, "pre", profile="local-neuroglancer")
    zar = _volume(tmp_path, "zar", profile="local")
    assert volume_layer(pre)[0]["source"]["url"] == f"precomputed://{pre}"
    assert volume_layer(zar)[0]["source"]["url"] == f"zarr://{zar}"


def test_a_volume_layer_leaves_its_bounding_box_off(tmp_path):
    """Only `bounds` is named, so every other subsource keeps neuroglancer's default."""
    layer, _ = volume_layer(_volume(tmp_path, "v"))
    assert layer["source"]["subsources"] == {"bounds": False}
    assert "enableDefaultSubsources" not in layer["source"]


def test_the_recorded_kind_decides_image_versus_segmentation(tmp_path):
    """The kind recorded in the volume decides the layer type; an explicit kind overrides."""
    seg = _volume(tmp_path, "seg", kind="segmentation", dtype="uint64")
    img = _volume(tmp_path, "img", kind="image")
    assert volume_layer(seg)[0]["type"] == "segmentation"
    assert volume_layer(img)[0]["type"] == "image"
    # ...and an explicit kind overrides it
    assert volume_layer(seg, kind="image")[0]["type"] == "image"


def test_segment_ids_are_strings(tmp_path):
    """A uint64 id above 2**53 would be rounded as a JSON number."""
    seg = _volume(tmp_path, "seg", kind="segmentation", dtype="uint64")
    big = 2 ** 60 + 1
    layer, _ = volume_layer(seg, segments=[1, big])
    assert layer["segments"] == ["1", str(big)]
    assert json.loads(json.dumps(layer))["segments"][1] == str(big)


def test_a_missing_volume_is_named(tmp_path):
    with pytest.raises(SourceProblem, match="no volume found"):
        volume_layer(str(tmp_path / "nope"))


def test_the_frame_is_reported_for_the_state_to_use(tmp_path):
    vol = _volume(tmp_path, "v", voxel=(40, 8, 8))
    _layer, frame = volume_layer(vol)
    assert tuple(frame["voxel_size"]) == (40, 8, 8)      # zyx
    assert frame["units"] == "nm"


# --------------------------------------------------------------------------- #
# loading layer files
# --------------------------------------------------------------------------- #
def test_a_layer_file_is_accepted_as_a_layer_or_a_whole_state(tmp_path):
    """A layer file may hold a bare layer or a whole state."""
    layer = {"type": "annotation", "name": "a", "annotations": []}
    bare, wrapped = tmp_path / "bare.json", tmp_path / "state.json"
    bare.write_text(json.dumps(layer))
    wrapped.write_text(json.dumps({"dimensions": {}, "layers": [layer]}))
    assert load_layer(str(bare)) == [layer]
    assert load_layer(str(wrapped)) == [layer]


def test_a_file_that_is_neither_says_which_it_expected(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"hello": 1}')
    with pytest.raises(StateProblem, match="not a neuroglancer layer or state"):
        load_layer(str(bad))


# --------------------------------------------------------------------------- #
# the state and the URL
# --------------------------------------------------------------------------- #
def test_position_is_reversed_to_xyz():
    """zyx in memory, xyz in the state."""
    state, _ = build_state([{"type": "image", "name": "i"}],
                           voxel_size_zyx=(8, 8, 8), units="nm",
                           position_zyx=(100, 200, 300))
    assert state["position"] == [300.0, 200.0, 100.0]


def test_dimensions_come_from_the_voxel_size():
    state, warning = build_state([], voxel_size_zyx=(40, 8, 8), units="nm")
    assert warning is None
    assert state["dimensions"] == {"x": [8e-9, "m"], "y": [8e-9, "m"], "z": [40e-9, "m"]}


def test_the_url_round_trips():
    state, _ = build_state([{"type": "image", "name": "i"}], voxel_size_zyx=(8, 8, 8),
                           units="nm", position_zyx=(1, 2, 3), layout="xy-3d")
    url = state_url(state)
    assert url.startswith(DEFAULT_VIEWER.rstrip("/") + "/#!")
    assert parse_url(url) == state


def test_the_fragment_is_percent_encoded():
    """Raw braces and quotes in a fragment get mangled by chat clients and shells."""
    state, _ = build_state([{"type": "image", "name": "i"}], voxel_size_zyx=(8, 8, 8),
                           units="nm")
    fragment = state_url(state).split("#!", 1)[1]
    assert "{" not in fragment and '"' not in fragment
    assert json.loads(urllib.parse.unquote(fragment))["layers"][0]["name"] == "i"


# --------------------------------------------------------------------------- #
# the opening view
# --------------------------------------------------------------------------- #
def test_the_frame_comes_from_the_volume_metadata(tmp_path):
    """Read from the precomputed `info`, xyz there and zyx here."""
    from neu_glance.sources import volume_extent

    vol = _volume(tmp_path, "v")            # 16^3
    extent, offset = volume_extent(vol, "neuroglancer_precomputed")
    assert extent == (16, 16, 16) and offset == (0, 0, 0)


def test_a_voxel_offset_shifts_the_frame(tmp_path):
    """`voxel_offset` (xyz in `info`) becomes a zyx offset."""
    from neu_vol.location import read_json, write_json
    from neu_glance.sources import volume_extent

    vol = _volume(tmp_path, "v")
    info = read_json(vol, "info")
    for scale in info["scales"]:
        scale["voxel_offset"] = [100, 200, 300]          # xyz
    write_json(vol, info, "info")
    extent, offset = volume_extent(vol, "neuroglancer_precomputed")
    assert offset == (300, 200, 100), "voxel_offset is xyz and must be reversed"


def test_zarr_volumes_report_a_frame_too(tmp_path):
    from neu_glance.sources import volume_extent

    vol = _volume(tmp_path, "z", profile="local")
    assert volume_extent(vol, "zarr3")[0] == (16, 16, 16)


def test_default_view_centres_and_zooms_out():
    """Centred on the volume and zoomed to fit it (neuroglancer would open at the corner)."""
    from neu_glance.state import NOMINAL_VIEWPORT_PX, default_view

    centre, cross, projection = default_view((100, 200, 400))
    assert centre == [50.0, 100.0, 200.0]
    # the largest extent, plus margin, is what has to fit
    assert cross == pytest.approx(400 * 1.15 / NOMINAL_VIEWPORT_PX)
    assert projection == pytest.approx(400 * 1.15)


def test_an_unspecified_view_is_filled_in_from_the_volume(tmp_path, capsys):
    vol = _volume(tmp_path, "v")            # 16^3, voxel 8 nm
    cli.cmd_gen(cli._parse_args(["gen", "--image", vol]))
    state = parse_url(capsys.readouterr().out.strip())
    assert state["position"] == [8.0, 8.0, 8.0]           # centre of a 16^3 volume
    assert state["crossSectionScale"] > 0
    assert state["projectionScale"] > state["crossSectionScale"]


@pytest.mark.parametrize("flag,key,value", [
    ("--position", "position", "1,2,3"),
    ("--cross-section-scale", "crossSectionScale", "0.5"),
    ("--projection-scale", "projectionScale", "99"),
])
def test_an_explicit_view_value_wins_over_the_default(tmp_path, capsys, flag, key, value):
    vol = _volume(tmp_path, "v")
    cli.cmd_gen(cli._parse_args(["gen", "--image", vol, flag, value]))
    state = parse_url(capsys.readouterr().out.strip())
    if key == "position":
        assert state["position"] == [3.0, 2.0, 1.0]       # zyx in, xyz out
    else:
        assert state[key] == float(value)


def test_a_layer_only_link_frames_its_annotations(tmp_path, capsys):
    """With no volume, the view centres on the layer's boxes."""
    box = tmp_path / "box.json"
    box.write_text(json.dumps({
        "type": "annotation", "name": "boxes",
        "annotations": [{"type": "axis_aligned_bounding_box", "id": "a",
                         "pointA": [100, 100, 100], "pointB": [200, 300, 400]}]}))
    cli.cmd_gen(cli._parse_args(
        ["gen", "--layer", str(box), "--voxel-size", "8,8,8"]))
    state = parse_url(capsys.readouterr().out.strip())
    assert state["position"] == [150.0, 200.0, 250.0]     # centre of the box, xyz


def test_a_link_with_no_frame_at_all_says_so(tmp_path, capsys):
    """With nothing to frame on, a warning says the link opens at the origin."""
    box = tmp_path / "box.json"
    box.write_text(json.dumps({"type": "annotation", "name": "empty",
                               "annotations": []}))
    cli.cmd_gen(cli._parse_args(
        ["gen", "--layer", str(box), "--voxel-size", "8,8,8"]))
    captured = capsys.readouterr()
    assert "will open at the origin" in captured.err
    assert "position" not in parse_url(captured.out.strip())


def test_the_largest_volume_sets_the_frame(tmp_path, capsys):
    """The view fits the largest volume."""
    small = _volume(tmp_path, "small")
    big = _volume(tmp_path, "big", shape=(64, 64, 64))
    cli.cmd_gen(cli._parse_args(
        ["gen", "--image", small, "--image", big]))
    state = parse_url(capsys.readouterr().out.strip())
    assert state["position"] == [32.0, 32.0, 32.0]        # centre of the 64^3


# --------------------------------------------------------------------------- #
# the CLI
# --------------------------------------------------------------------------- #
def test_layers_are_ordered_image_then_segmentation_then_files(tmp_path, capsys):
    """Image, then segmentation, then layer files, so each draws over the last."""
    img = _volume(tmp_path, "img")
    seg = _volume(tmp_path, "seg", kind="segmentation", dtype="uint64")
    box = tmp_path / "box.json"
    box.write_text(json.dumps({"type": "annotation", "name": "boxes",
                               "annotations": []}))
    assert cli.cmd_gen(cli._parse_args(
        ["gen", "--seg", seg, "--image", img, "--layer", str(box)])) == 0
    state = parse_url(capsys.readouterr().out.strip())
    assert [lyr["name"] for lyr in state["layers"]] == ["img", "seg", "boxes"]


def test_segments_apply_to_the_seg_volumes_in_order(tmp_path, capsys):
    a = _volume(tmp_path, "a", kind="segmentation", dtype="uint64")
    b = _volume(tmp_path, "b", kind="segmentation", dtype="uint64")
    cli.cmd_gen(cli._parse_args(
        ["gen", "--seg", a, "--segments", "1,2", "--seg", b,
         "--segments", "7"]))
    state = parse_url(capsys.readouterr().out.strip())
    picked = {lyr["name"]: lyr.get("segments") for lyr in state["layers"]}
    assert picked == {"a": ["1", "2"], "b": ["7"]}


def test_position_order_xyz_lets_you_paste_from_the_viewer(tmp_path, capsys):
    """`--position-order xyz` takes positions as neuroglancer displays them."""
    vol = _volume(tmp_path, "v")
    cli.cmd_gen(cli._parse_args(
        ["gen", "--image", vol, "--position", "1,2,3",
         "--position-order", "xyz"]))
    assert parse_url(capsys.readouterr().out.strip())["position"] == [1.0, 2.0, 3.0]

    cli.cmd_gen(cli._parse_args(
        ["gen", "--image", vol, "--position", "1,2,3"]))      # zyx default
    assert parse_url(capsys.readouterr().out.strip())["position"] == [3.0, 2.0, 1.0]


def test_layer_files_alone_need_an_explicit_voxel_size(tmp_path, capsys):
    """A layer file does not establish the viewer's frame, so `--voxel-size` is required."""
    box = tmp_path / "box.json"
    box.write_text(json.dumps({"type": "annotation", "name": "boxes",
                               "annotations": []}))
    with pytest.raises(SystemExit, match="no voxel size available"):
        cli.cmd_gen(cli._parse_args(["gen", "--layer", str(box)]))

    assert cli.cmd_gen(cli._parse_args(
        ["gen", "--layer", str(box), "--voxel-size", "8,8,8"])) == 0
    assert parse_url(capsys.readouterr().out.strip())["dimensions"]["x"] == [8e-9, "m"]


def test_select_last_opens_the_boxes_panel(tmp_path, capsys):
    vol = _volume(tmp_path, "v", kind="segmentation", dtype="uint64")
    box = tmp_path / "box.json"
    box.write_text(json.dumps({"type": "annotation", "name": "boxes",
                               "annotations": []}))
    cli.cmd_gen(cli._parse_args(
        ["gen", "--seg", vol, "--layer", str(box), "--select-last"]))
    state = parse_url(capsys.readouterr().out.strip())
    assert state["selectedLayer"] == {"visible": True, "layer": "boxes"}


def test_hide_slices_sets_show_slices_false_and_is_otherwise_absent(tmp_path, capsys):
    """`showSlices` is written only when asked, so the viewer's own setting stands."""
    vol = _volume(tmp_path, "v")
    cli.cmd_gen(cli._parse_args(["gen", "--seg", vol]))
    assert "showSlices" not in parse_url(capsys.readouterr().out.strip())

    cli.cmd_gen(cli._parse_args(["gen", "--seg", vol, "--hide-slices"]))
    assert parse_url(capsys.readouterr().out.strip())["showSlices"] is False


def test_hide_adds_a_layer_switched_off(tmp_path, capsys):
    a = _volume(tmp_path, "a", kind="segmentation", dtype="uint64")
    b = _volume(tmp_path, "b", kind="segmentation", dtype="uint64")
    cli.cmd_gen(cli._parse_args(["gen", "--seg", a, "--seg", b, "--hide", "b"]))
    layers = {lyr["name"]: lyr for lyr in parse_url(capsys.readouterr().out.strip())["layers"]}
    assert layers["b"]["visible"] is False and "visible" not in layers["a"]


def test_hide_names_a_layer_that_exists(tmp_path):
    vol = _volume(tmp_path, "v", kind="segmentation", dtype="uint64")
    with pytest.raises(SystemExit, match="--hide names no layer: w. The layers are v"):
        cli.cmd_gen(cli._parse_args(["gen", "--seg", vol, "--hide", "w"]))


def _with_properties(volume, ids):
    """Give a segmentation volume a segment_properties document listing ``ids``."""
    import os

    os.makedirs(os.path.join(volume, "segment_properties"))
    with open(os.path.join(volume, "segment_properties", "info"), "w") as f:
        json.dump({"@type": "neuroglancer_segment_properties",
                   "inline": {"ids": [str(i) for i in ids], "properties": []}}, f)
    info_path = os.path.join(volume, "info")
    info = json.load(open(info_path))
    json.dump({**info, "segment_properties": "segment_properties"}, open(info_path, "w"))


def test_all_segments_selects_what_the_properties_list(tmp_path, capsys):
    a = _volume(tmp_path, "a", kind="segmentation", dtype="uint64")
    rois = _volume(tmp_path, "rois", kind="segmentation", dtype="uint64")
    _with_properties(rois, [3, 1, 2])
    cli.cmd_gen(cli._parse_args(["gen", "--seg", a, "--seg", rois, "--segments", "",
                                 "--segments", "5", "--all-segments", "rois"]))
    layers = {lyr["name"]: lyr for lyr in parse_url(capsys.readouterr().out.strip())["layers"]}
    assert sorted(layers["rois"]["segments"], key=int) == ["1", "2", "3", "5"]
    assert "segments" not in layers["a"]


@pytest.mark.parametrize("args, match", [
    (["--all-segments", "nope"], "names no --seg layer: nope"),
    (["--all-segments", "v"], "needs segment_properties"),
])
def test_all_segments_refuses_what_it_cannot_list(tmp_path, args, match):
    vol = _volume(tmp_path, "v", kind="segmentation", dtype="uint64")
    with pytest.raises(SystemExit, match=match):
        cli.cmd_gen(cli._parse_args(["gen", "--seg", vol, *args]))


def test_out_can_also_be_stdout(tmp_path, capsys):
    vol = _volume(tmp_path, "v")
    path = tmp_path / "link.txt"
    cli.cmd_gen(cli._parse_args(["gen", "--image", vol, "--out", str(path), "--out", "-"]))
    printed = capsys.readouterr().out.strip()
    assert printed.startswith("https://") and path.read_text().strip() == printed


def test_out_writes_through_the_kvstore(tmp_path, capsys):
    """`--out` creates missing parents and keeps stdout clean."""
    vol = _volume(tmp_path, "v")
    url_path = str(tmp_path / "deep" / "link.txt")
    cli.cmd_gen(cli._parse_args(["gen", "--image", vol, "--out", url_path]))
    assert capsys.readouterr().out == "", "--out must keep stdout clean"
    with open(url_path) as f:
        url = f.read().strip()
    assert url.startswith(DEFAULT_VIEWER.rstrip("/") + "/#!")


def test_the_state_and_the_url_carry_the_same_thing(tmp_path, capsys):
    """`--format state` and `--format url` serialize the same state."""
    vol = _volume(tmp_path, "v")
    cli.cmd_gen(cli._parse_args(["gen", "--image", vol]))
    url = capsys.readouterr().out.strip()
    cli.cmd_gen(cli._parse_args(["gen", "--image", vol, "--format", "state"]))
    state = json.loads(capsys.readouterr().out)
    assert state == parse_url(url)


def test_the_viewer_base_is_overridable(tmp_path, capsys):
    vol = _volume(tmp_path, "v")
    cli.cmd_gen(cli._parse_args(
        ["gen", "--image", vol, "--viewer", "https://ng.example.org/"]))
    assert capsys.readouterr().out.startswith("https://ng.example.org/#!")


def test_building_a_layer_does_not_open_every_level(tmp_path, monkeypatch):
    """A layer needs only `info`; opening every level costs a store open per level."""
    from neu_vol import source_metadata

    vol = _volume(tmp_path, "v")
    calls = []
    real = source_metadata.existing_levels
    monkeypatch.setattr(source_metadata, "existing_levels",
                        lambda *a, **k: calls.append(a) or real(*a, **k))

    layer, frame = volume_layer(vol)
    assert layer["source"]["url"].startswith("precomputed://")
    assert tuple(frame["voxel_size"]) == (8, 8, 8)
    assert not calls, "volume_layer opened every level to build one layer"


# --------------------------------------------------------------------------- #
# precomputed annotation sources
# --------------------------------------------------------------------------- #
def _annotation_source(tmp_path, name="syn", *, properties=("conf_pre", "conf_post"),
                       relationships=("body_pre", "body_post"),
                       bounds=((0, 0, 0), (600, 800, 1000)), voxel=8e-9):
    """An annotation source's `info` only, which is all these code paths read."""
    dst = tmp_path / name
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "info").write_text(json.dumps({
        "@type": "neuroglancer_annotations_v1",
        "dimensions": {a: [voxel, "m"] for a in ("x", "y", "z")},
        "lower_bound": list(bounds[0]), "upper_bound": list(bounds[1]),
        "annotation_type": "line",
        "properties": [{"id": p, "type": "float32"} for p in properties],
        "relationships": [{"id": r, "key": f"by_rel_{r}"} for r in relationships],
        "by_id": {"key": "by_id"},
        "spatial": [],
    }))
    return str(dst)


def test_an_annotation_source_becomes_an_annotation_layer(tmp_path):
    from neu_glance.sources import annotation_layer

    layer, _info = annotation_layer(_annotation_source(tmp_path))
    assert layer["type"] == "annotation"
    assert layer["source"].startswith("precomputed://")


def test_a_volume_passed_as_annotations_is_refused(tmp_path):
    """An annotation layer pointed at a volume would load and draw nothing."""
    from neu_glance.sources import annotation_layer

    with pytest.raises(SourceProblem, match="neuroglancer_annotations_v1"):
        annotation_layer(_volume(tmp_path, "v"))


def test_the_shader_is_chosen_from_the_properties_the_source_declares(tmp_path):
    from neu_glance.shaders import synapse_shader
    from neu_glance.sources import annotation_layer

    layer, info = annotation_layer(_annotation_source(tmp_path))
    assert layer["shader"] == synapse_shader({"conf_pre", "conf_post"})
    assert "synapse" in info["shader"]


def test_no_shader_is_applied_when_the_properties_do_not_match(tmp_path):
    """A shader naming an absent `prop_` fails to compile and the layer draws nothing."""
    from neu_glance.sources import annotation_layer

    src = _annotation_source(tmp_path, properties=("weight",))
    layer, info = annotation_layer(src)
    assert "shader" not in layer
    assert "no built-in shader matches" in info["shader"]


def test_asking_for_a_shader_the_source_cannot_feed_is_an_error(tmp_path):
    from neu_glance.sources import annotation_layer

    src = _annotation_source(tmp_path, properties=("weight",))
    with pytest.raises(SourceProblem, match="conf_pre"):
        annotation_layer(src, shader="synapse")


def test_a_shader_can_be_a_file_or_switched_off(tmp_path):
    from neu_glance.sources import annotation_layer

    src = _annotation_source(tmp_path)
    glsl = tmp_path / "s.glsl"
    glsl.write_text("void main() { setColor(vec4(1.0)); }")
    layer, info = annotation_layer(src, shader=str(glsl))
    assert layer["shader"].startswith("void main()") and str(glsl) in info["shader"]

    layer, _info = annotation_layer(src, shader="none")
    assert "shader" not in layer


def test_an_unknown_shader_name_lists_the_built_ins(tmp_path):
    from neu_glance.sources import annotation_layer

    with pytest.raises(SourceProblem, match="synapse"):
        annotation_layer(_annotation_source(tmp_path), shader="nope")


def test_relationships_bind_to_the_named_segmentation_layer(tmp_path):
    """Relationships take effect only once bound to a segmentation layer."""
    from neu_glance.sources import annotation_layer

    layer, _info = annotation_layer(_annotation_source(tmp_path), linked_segmentation="seg")
    assert layer["linkedSegmentationLayer"] == {"body_pre": "seg", "body_post": "seg"}
    assert layer["filterBySegmentation"] == ["body_pre", "body_post"]


def test_without_a_segmentation_there_is_no_binding(tmp_path):
    from neu_glance.sources import annotation_layer

    layer, _info = annotation_layer(_annotation_source(tmp_path))
    assert "linkedSegmentationLayer" not in layer
    assert "filterBySegmentation" not in layer


@pytest.mark.parametrize("properties, boxes", [
    (("conf_pre", "conf_post"), []),
    (("conf_pre", "conf_post", "body_pre_u32", "body_post_u32"), ["color_by_neuron_id"]),
    (("conf_pre", "conf_post", "body_pre_u32", "body_post_u32", "neuropil", "roi",
      "tbar_fanout"),
     ["color_by_neuropil", "color_by_neuropil_part", "color_by_neuron_id",
      "color_by_tbar_fanout"]),
])
def test_the_synapse_shader_offers_only_the_colourings_the_source_can_feed(
        tmp_path, properties, boxes):
    """A colouring reading an absent property would stop the shader compiling."""
    import re

    from neu_glance.sources import annotation_layer

    layer, info = annotation_layer(_annotation_source(tmp_path, properties=properties))
    found = re.findall(r"#uicontrol bool (color_by_\w+)", layer["shader"])
    assert found == boxes
    declared = set(properties)
    used = set(re.findall(r"prop_(\w+)\(\)", layer["shader"]))
    assert used <= declared, f"the shader reads undeclared {sorted(used - declared)}"
    assert ("hide_uncolored" in layer["shader"]) == bool(set(boxes) - {"color_by_tbar_fanout"})
    for box in boxes:
        assert box in info["shader"]


def test_with_a_colouring_the_border_keeps_the_pre_and_post_colours():
    from neu_glance.shaders import synapse_shader

    plain = synapse_shader({"conf_pre", "conf_post"})
    coloured = synapse_shader({"conf_pre", "conf_post", "body_pre_u32", "body_post_u32"})
    assert "setEndpointMarkerBorderColor" not in plain and "border_width" not in plain
    assert "if (color_by_neuron_id) {\n    setEndpointMarkerBorderColor(pre_color, post_color)" \
        in coloured
    assert "#uicontrol float border_width" in coloured


@pytest.mark.parametrize("listed", [False, True])
def test_the_hide_unlisted_filters_come_only_with_the_listed_flags(tmp_path, listed):
    import re

    from neu_glance.sources import annotation_layer

    props = ("conf_pre", "conf_post") + (("pre_listed", "post_listed") if listed else ())
    layer, info = annotation_layer(_annotation_source(tmp_path, properties=props))
    boxes = re.findall(r"#uicontrol bool (hide_unlisted\w*)", layer["shader"])
    assert boxes == (["hide_unlisted", "hide_unlisted_partner"] if listed else [])
    assert ("prop_pre_listed()" in layer["shader"]) == listed
    assert ("filters: hide_unlisted" in info["shader"]) == listed


def test_filtering_on_a_relationship_the_source_lacks_is_refused(tmp_path):
    from neu_glance.sources import annotation_layer

    src = _annotation_source(tmp_path, relationships=("body_pre",))
    with pytest.raises(SourceProblem, match="body_post"):
        annotation_layer(src, linked_segmentation="seg",
                         filter_relationships=["body_post"])


def test_seg_display_flags_set_one_layer_each(tmp_path, capsys):
    a = _volume(tmp_path, "a", kind="segmentation", dtype="uint64")
    b = _volume(tmp_path, "b", kind="segmentation", dtype="uint64")
    cli.cmd_gen(cli._parse_args(
        ["gen", "--seg", a, "--seg", b, "--silhouette", "a=5", "--opacity-2d", "a=0",
         "--opacity-3d", "b=0.4"]))
    layers = {lyr["name"]: lyr for lyr in parse_url(capsys.readouterr().out.strip())["layers"]}
    assert layers["a"]["meshSilhouetteRendering"] == 5.0
    assert layers["a"]["selectedAlpha"] == 0.0
    assert "objectAlpha" not in layers["a"]
    assert layers["b"]["objectAlpha"] == 0.4
    assert "meshSilhouetteRendering" not in layers["b"]


@pytest.mark.parametrize("flag, value, message", [
    ("--silhouette", "nope=5", "names no --seg layer"),
    ("--opacity-2d", "a", "LAYER_NAME=VALUE"),
    ("--opacity-3d", "a=1.5", "between 0 and 1"),
    ("--silhouette", "a=x", "not a number"),
])
def test_seg_display_flags_refuse_what_neuroglancer_would_ignore(tmp_path, flag, value,
                                                                 message):
    a = _volume(tmp_path, "a", kind="segmentation", dtype="uint64")
    with pytest.raises(SystemExit, match=message):
        cli.cmd_gen(cli._parse_args(["gen", "--seg", a, flag, value]))


def test_the_extent_and_voxel_size_come_from_the_sources_own_info(tmp_path):
    """Read from the source's `info`, so an annotations-only link needs no --voxel-size."""
    from neu_glance.sources import (annotation_source_extent, annotation_source_voxel_size,
                                read_annotation_info)

    info = read_annotation_info(_annotation_source(tmp_path))
    extent, offset = annotation_source_extent(info)
    assert extent == (1000.0, 800.0, 600.0)          # xyz bounds read back as zyx
    assert offset == (0.0, 0.0, 0.0)
    assert annotation_source_voxel_size(info) == (8.0, 8.0, 8.0)


def test_the_cli_adds_the_layer_bound_to_the_first_segmentation(tmp_path, capsys):
    vol = _volume(tmp_path, "v", kind="segmentation", dtype="uint64")
    src = _annotation_source(tmp_path)
    assert cli.cmd_gen(cli._parse_args(
        ["gen", "--seg", vol, "--annotations", src, "--segments", "7"])) == 0
    layers = parse_url(capsys.readouterr().out.strip())["layers"]
    ann = [lyr for lyr in layers if lyr["type"] == "annotation"][0]
    assert ann["linkedSegmentationLayer"] == {"body_pre": "v", "body_post": "v"}
    assert ann["filterBySegmentation"] == ["body_pre", "body_post"]


def test_filtering_is_on_by_default_even_with_nothing_selected(tmp_path, capsys):
    """The filter is written even with no --segments, so the link opens empty until a body is
    clicked; a warning says so."""
    vol = _volume(tmp_path, "v", kind="segmentation", dtype="uint64")
    assert cli.cmd_gen(cli._parse_args(
        ["gen", "--seg", vol, "--annotations", _annotation_source(tmp_path)])) == 0
    out = capsys.readouterr()
    ann = [lyr for lyr in parse_url(out.out.strip())["layers"]
           if lyr["type"] == "annotation"][0]
    assert ann["filterBySegmentation"] == ["body_pre", "body_post"]
    assert ann["linkedSegmentationLayer"] == {"body_pre": "v", "body_post": "v"}
    assert "opens EMPTY" in out.err


def test_no_filter_by_segmentation_overrides_a_selection(tmp_path, capsys):
    vol = _volume(tmp_path, "v", kind="segmentation", dtype="uint64")
    cli.cmd_gen(cli._parse_args(
        ["gen", "--seg", vol, "--annotations", _annotation_source(tmp_path),
         "--segments", "7", "--no-filter-by-segmentation"]))
    ann = [lyr for lyr in parse_url(capsys.readouterr().out.strip())["layers"]
           if lyr["type"] == "annotation"][0]
    assert "filterBySegmentation" not in ann


def test_an_annotations_only_link_needs_no_voxel_size(tmp_path, capsys):
    src = _annotation_source(tmp_path)
    assert cli.cmd_gen(cli._parse_args(["gen", "--annotations", src])) == 0
    state = parse_url(capsys.readouterr().out.strip())
    assert state["dimensions"]["x"] == [8e-9, "m"]
    # the centre of the declared bounds, xyz
    assert state["position"] == [300.0, 400.0, 500.0]


def test_nothing_to_show_is_refused_and_names_annotations(tmp_path):
    with pytest.raises(SystemExit, match="nothing to show") as excinfo:
        cli.cmd_gen(cli._parse_args(["gen"]))
    assert "--annotations" in str(excinfo.value)
