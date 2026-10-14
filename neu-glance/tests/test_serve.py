"""Hosting arrays locally: `neu-glance serve` and :mod:`neu_glance.serving`.

The first half needs no `neuroglancer` (the CI condition: `--help` must work without the
`serve` extra); the rest skips without it. The silent failures guarded here: a uint8 label
array served as an image, a dropped origin, and `encoding="raw"` (broken under NumPy 2).
"""

import numpy as np
import pytest


# --------------------------------------------------------------------------- #
# no neuroglancer needed — the CI condition
# --------------------------------------------------------------------------- #
def test_the_parser_builds_without_neuroglancer():
    """`serve` must not make `neu-glance --help` need neuroglancer."""
    import subprocess
    import sys

    code = (
        "import sys;"
        "sys.modules['neuroglancer'] = None;"
        "from neu_glance.cli import build_parser, _parse_args;"
        "p = build_parser();"
        "a = _parse_args(['serve', '--seg', 'x.h5']);"
        "assert a.func.__name__ == 'cmd_serve', a.func;"
        "print('ok')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


def test_a_missing_extra_names_the_extra(monkeypatch):
    """The failure lands at run time, so its message is the only guidance there is."""
    import builtins

    import neu_glance.serving as serving

    real = builtins.__import__

    def blocked(name, *a, **k):
        if name == "neuroglancer":
            raise ImportError("blocked")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(serving.ServeProblem, match=r"neu-glance\[serve\]"):
        serving._neuroglancer()


def test_the_cli_shader_names_agree_with_the_registry():
    """`cli.py` keeps its own copy of the names so building the parser imports nothing."""
    from neu_glance.cli import _IMAGE_SHADER_NAMES
    from neu_glance.shaders import IMAGE_SHADERS

    assert set(_IMAGE_SHADER_NAMES) == set(IMAGE_SHADERS) | {"none"}


# --------------------------------------------------------------------------- #
# image shaders
# --------------------------------------------------------------------------- #
def test_the_default_shader_follows_kind_and_channel_count():
    from neu_glance.shaders import IMAGE_SHADERS, image_shader

    src = {name: entry["source"] for name, entry in IMAGE_SHADERS.items()}
    assert image_shader(None, kind="image") == src["grayscale"]
    assert image_shader(None, kind="probability") == src["probability"]
    # three channels beats the kind default
    assert image_shader(None, kind="probability", channels=3) == src["rgb"]
    assert image_shader("colormap") == src["colormap"]
    assert image_shader("none") is None


def test_a_channel_count_mismatch_raises():
    """`rgb` on one channel fails to compile; one channel's shader on three shows a third
    of the data."""
    from neu_glance.shaders import ShaderProblem, image_shader

    with pytest.raises(ShaderProblem, match="reads 3 channel"):
        image_shader("rgb", channels=1)
    with pytest.raises(ShaderProblem, match="reads 1 channel"):
        image_shader("grayscale", channels=3)
    with pytest.raises(ShaderProblem, match="unknown image shader"):
        image_shader("nope")


def test_the_layer_color_is_set_as_a_shader_control(frame, stop_server):
    """Set in the state, so every layer shares one shader and the viewer shows the value."""
    prob = _labels(dtype="float32")
    server = serve([ServedLayer(prob, kind="probability", name="a", frame=frame,
                                color="#00ff00")])
    server.add_layer(ServedLayer(prob, kind="probability", name="b", frame=frame),
                     color=(0, 0, 255))

    by_name = {lyr["name"]: lyr for lyr in server.state()["layers"]}
    assert by_name["a"]["shaderControls"] == {"color": "#00ff00"}
    assert by_name["b"]["shaderControls"] == {"color": "#0000ff"}, "ints read as 0..255"
    assert by_name["a"]["shader"] == by_name["b"]["shader"], "one shader, two controls"


def test_add_layer_color_does_not_mutate_the_layer_it_was_given(frame, stop_server):
    """Two colours off one ServedLayer is the ordinary way to do a before/after."""
    layer = ServedLayer(_labels(dtype="float32"), kind="probability", frame=frame)
    server = serve([ServedLayer(_labels(), name="em", frame=frame)])
    server.add_layer(layer, name="red")
    server.add_layer(layer, name="green", color="#00ff00")

    by_name = {lyr["name"]: lyr for lyr in server.state()["layers"]}
    assert layer.color is None, "the caller's object is untouched"
    assert not by_name["red"].get("shaderControls")
    assert by_name["green"]["shaderControls"] == {"color": "#00ff00"}


def test_a_color_the_shader_cannot_take_is_refused_not_ignored(frame, stop_server):
    """Neuroglancer silently ignores a shaderControl it does not recognise."""
    from neu_glance.shaders import ShaderProblem

    server = serve([ServedLayer(_labels(), name="em", frame=frame)])
    with pytest.raises(ShaderProblem, match="no single colour"):
        server.add_layer(ServedLayer(_labels(), kind="image", name="grey", frame=frame),
                         color="#00ff00")
    with pytest.raises(ShaderProblem, match="it has three"):
        server.add_layer(ServedLayer(np.zeros((3, 4, 8, 8), "float32"), kind="probability",
                                     name="rgb", frame=frame), color="#00ff00")
    with pytest.raises(ShaderProblem, match="label hashing"):
        server.add_layer(ServedLayer(_labels(dtype="uint64"), kind="segmentation",
                                     name="seg", frame=frame), color="#00ff00")


def test_colors_are_accepted_in_the_forms_a_caller_has():
    from neu_glance.shaders import ShaderProblem, as_hex_color

    assert as_hex_color("#FF0000") == "#ff0000"
    assert as_hex_color("ff0000") == "#ff0000"
    assert as_hex_color("red") == "#ff0000", "a name, with nothing installed"
    assert as_hex_color("Cyan") == "#00ffff", "case-insensitive"
    assert as_hex_color((1.0, 0.5, 0.0)) == "#ff8000", "floats as 0..1"
    assert as_hex_color((255, 128, 0)) == "#ff8000", "ints as 0..255"
    assert as_hex_color((1, 1, 1)) == "#ffffff", "all <= 1 reads as floats"
    for bad in ("reddish", (1.0, 0.5), 7):
        with pytest.raises(ShaderProblem):
            as_hex_color(bad)
    with pytest.raises(ShaderProblem, match="alpha is not settable"):
        as_hex_color((1.0, 0.0, 0.0, 0.5))


def test_the_builtin_color_names_agree_with_matplotlib():
    """Installing matplotlib must not change what a built-in name means."""
    to_hex = pytest.importorskip("matplotlib.colors").to_hex
    from neu_glance.shaders import BUILTIN_COLORS

    assert {name: to_hex(name) for name in BUILTIN_COLORS} == BUILTIN_COLORS
    # single letters are left to matplotlib, whose 'c' is not CSS 'cyan'
    assert not any(len(name) == 1 for name in BUILTIN_COLORS)


def test_the_long_tail_of_names_needs_matplotlib_and_says_so(monkeypatch):
    """Without matplotlib, a name outside the built-in table is a clear error."""
    import builtins

    from neu_glance.shaders import BUILTIN_COLORS, ShaderProblem, as_hex_color

    assert as_hex_color("forestgreen") == "#228b22", "when matplotlib is installed"

    real_import = builtins.__import__

    def no_matplotlib(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError("no matplotlib")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_matplotlib)
    with pytest.raises(ShaderProblem, match="not a dependency of this package"):
        as_hex_color("forestgreen")
    # hex, the built-in names and tuples still work
    assert as_hex_color("#ff0000") == "#ff0000"
    assert as_hex_color("orange") == BUILTIN_COLORS["orange"]
    assert as_hex_color((0, 255, 0)) == "#00ff00"


# --------------------------------------------------------------------------- #
# the rest needs neuroglancer
# --------------------------------------------------------------------------- #
ng = pytest.importorskip("neuroglancer", reason="the serve extra is not installed")

from neu_lib import BBox  # noqa: E402

from neu_glance.serving import (ServedLayer, ServeProblem,  # noqa: E402
                                _coordinate_space, _volume_type, _voxel_offset, serve)


@pytest.fixture
def frame():
    from neu_lib import Frame

    return Frame(voxel_size_nm=(40.0, 8.0, 8.0), origin_nm=(400.0, 160.0, 240.0))


@pytest.fixture
def stop_server():
    yield
    ng.server.stop()


def _labels(shape=(8, 16, 16), dtype="uint8"):
    return (np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape) % 7)


# --- the layer spec --------------------------------------------------------- #
def test_a_bad_kind_or_shape_is_refused_before_anything_is_served():
    with pytest.raises(ServeProblem, match="kind must be"):
        ServedLayer(_labels(), kind="labels")
    with pytest.raises(ServeProblem, match="3-D or 4-D"):
        ServedLayer(np.zeros((4, 4)), kind="image")
    with pytest.raises(ServeProblem, match="expected an array"):
        ServedLayer([1, 2, 3], kind="image")


# --- the frame, which is the silent one ------------------------------------- #
def test_the_coordinate_space_carries_the_frames_nm_voxel_size(frame):
    space = _coordinate_space(ng, frame)
    assert list(space.names) == ["z", "y", "x"]
    # neuroglancer normalizes to base SI on construction, so 40 nm is stored as 4e-8 m
    assert space.to_json() == {"z": [4e-08, "m"], "y": [8e-09, "m"], "x": [8e-09, "m"]}


def test_a_channel_axis_becomes_a_LOCAL_dimension(frame):
    """The trailing `^` hands every channel to the shader instead of offering a slider."""
    space = _coordinate_space(ng, frame, channel_axis=True)
    assert list(space.names) == ["c^", "z", "y", "x"]
    assert space.to_json()["c^"] == [1.0, ""], "a channel is unitless, not 1 nm"


def test_the_origin_becomes_a_voxel_offset(frame):
    from neu_lib import Frame

    assert _voxel_offset(frame) == [10, 20, 30]
    assert _voxel_offset(frame, channel_axis=True) == [0, 10, 20, 30]
    assert _voxel_offset(Frame(voxel_size_nm=(8, 8, 8), origin_nm=(0, 0, 0))) is None
    assert _voxel_offset(None) is None


def test_a_non_integral_origin_raises_rather_than_rounding():
    """Rounding would shift the volume by up to half a voxel, undetectably."""
    from neu_lib import Frame

    with pytest.raises(ServeProblem, match="not a whole number"):
        _voxel_offset(Frame(voxel_size_nm=(8, 8, 8), origin_nm=(4, 0, 0)))


# --- the uint8 trap -------------------------------------------------------- #
def test_kind_decides_volume_type_never_the_dtype():
    """neuroglancer guesses segmentation only for uint16/32/64, so a uint8 label array
    would be served as an image."""
    assert _volume_type(ServedLayer(_labels(dtype="uint8"), kind="segmentation")) \
        == "segmentation"
    assert _volume_type(ServedLayer(_labels(dtype="uint64"), kind="image")) == "image"
    assert _volume_type(ServedLayer(_labels(), kind="probability")) == "image"

    # ...and it survives all the way onto the served volume
    guessed = ng.LocalVolume(_labels(dtype="uint8"),
                             ng.CoordinateSpace(names=["z", "y", "x"], units="nm",
                                                scales=[8, 8, 8]))
    assert guessed.volume_type == "image", "the guess this overrides"


# --- serving --------------------------------------------------------------- #
def test_serving_sets_the_viewers_own_dimensions(frame, stop_server):
    """`dimensions` must match the data (a mismatch misplaces every layer), and are spatial
    only: a channel axis is local to the layer."""
    server = serve([ServedLayer(np.zeros((3, 4, 4, 4)), kind="probability",
                                channel_axis=True, frame=frame, name="p")])
    state = server.state()
    assert list(state["dimensions"]) == ["z", "y", "x"]


def test_each_kind_gets_the_right_layer_type_and_shader(frame, stop_server):
    server = serve([
        ServedLayer(_labels(), kind="image", name="em", frame=frame),
        ServedLayer(_labels(), kind="segmentation", name="seg", frame=frame),
        ServedLayer(_labels().astype("float32"), kind="probability", name="p",
                    frame=frame),
    ])
    by_name = {lyr["name"]: lyr for lyr in server.state()["layers"]}
    assert by_name["em"]["type"] == "image" and by_name["em"].get("shader")
    assert by_name["p"]["type"] == "image" and "threshold" in by_name["p"]["shader"]
    # A segmentation gets NO shader: the viewer's label hashing beats anything written
    # here, and a shader would defeat the selection UI it exists for.
    assert by_name["seg"]["type"] == "segmentation"
    assert not by_name["seg"].get("shader")
    assert {n: v.volume_type for n, v in server.volumes.items()} == {
        "em": "image", "seg": "segmentation", "p": "image"}
    # never "raw": its encoder calls ndarray.tostring, removed in NumPy 2.0
    assert {v.encoding for v in server.volumes.values()} == {"npz"}


def test_a_served_source_is_a_python_url(frame, stop_server):
    """And therefore not shareable, which is why `serve` has no --format url."""
    server = serve([ServedLayer(_labels(), name="a", frame=frame)])
    source = server.state()["layers"][0]["source"]
    source = source[0] if isinstance(source, list) else source
    url = source["url"] if isinstance(source, dict) else source
    assert url.startswith("python://volume/")


def test_the_annotation_layer_is_OPT_IN(frame, stop_server):
    """No annotation layer unless asked for."""
    plain = serve([ServedLayer(_labels(), name="b", frame=frame)])
    assert not any(lyr["name"] == "annotations" for lyr in plain.state()["layers"])
    with pytest.raises(ServeProblem, match="no annotation layer"):
        plain.boxes()

    asked = serve([ServedLayer(_labels(), name="a", frame=frame)], annotations=True)
    assert any(lyr["name"] == "annotations" for lyr in asked.state()["layers"])
    assert asked.boxes() == [], "empty, but present and readable"

    named = serve([ServedLayer(_labels(), name="c", frame=frame)], annotations="picks")
    assert any(lyr["name"] == "picks" for lyr in named.state()["layers"])


def test_annotate_adds_the_layer_boxes_reads_and_arms_a_tool(frame, stop_server):
    """Adds a layer to a viewer already up, arms the box tool and selects the layer."""
    server = serve([ServedLayer(_labels(), name="b", frame=frame)])
    with pytest.raises(ServeProblem, match="no annotation layer"):
        server.boxes()

    assert server.annotate() == "annotations"
    layer = server.viewer.state.layers["annotations"].layer
    assert layer.to_json()["type"] == "annotation"
    assert layer.tool.to_json()["type"] == "annotateBoundingBox"
    assert server.boxes() == [], "present and readable with no layer= argument"

    # the tool acts on the selected layer; unselected, clicks would place nothing
    assert server.viewer.state.selected_layer.layer == "annotations"
    assert server.viewer.state.selected_layer.visible is True


def test_annotate_takes_the_viewers_own_dimensions(frame, stop_server):
    """A mismatched annotation coordinate space reads back wrong coordinates."""
    server = serve([ServedLayer(_labels(), name="b", frame=frame)])
    server.annotate()
    source = server.viewer.state.layers["annotations"].layer.to_json()["source"]
    dims = (source[0] if isinstance(source, list) else source)
    dims = dims["transform"]["outputDimensions"]
    assert {k: float(v[0]) for k, v in dims.items()} == pytest.approx(
        {"z": 40e-9, "y": 8e-9, "x": 8e-9})


def test_annotate_again_KEEPS_what_was_already_drawn(frame, stop_server):
    """Calling it twice (a re-run cell) must not empty the layer."""
    server = serve([ServedLayer(_labels(), name="b", frame=frame)], annotations=True)
    with server.viewer.txn() as s:
        s.layers["annotations"].annotations = [
            ng.PointAnnotation(id="1", point=[4, 5, 6]),
        ]

    assert server.annotate(tool="point") == "annotations", "the layer already read back"
    assert server.points() == [(4, 5, 6)], "not replaced"
    assert (server.viewer.state.layers["annotations"].layer.tool.to_json()["type"]
            == "annotatePoint"), "re-armed on the layer that was there"


def test_annotate_will_not_replace_a_volume_layer(frame, stop_server):
    """That one would take real data with it."""
    server = serve([ServedLayer(_labels(), kind="segmentation", name="seg", frame=frame)])
    with pytest.raises(ServeProblem, match="already a segmentation layer"):
        server.annotate("seg")
    assert server.selected_segments() == set(), "the volume layer is untouched"


def test_annotate_refuses_a_tool_nothing_can_read_back(frame, stop_server):
    """boxes()/points() cannot read lines or ellipsoids back."""
    server = serve([ServedLayer(_labels(), name="b", frame=frame)])
    with pytest.raises(ServeProblem, match="tool must be one of box, point"):
        server.annotate(tool="line")
    assert "annotations" not in [lyr.name for lyr in server.viewer.state.layers], \
        "refused before the layer is added, not half-way through"


def test_annotate_names_the_layer_boxes_then_defaults_to(frame, stop_server):
    server = serve([ServedLayer(_labels(), name="b", frame=frame)])
    assert server.annotate("picks") == "picks"
    assert server.boxes() == [], "picks is now the default"
    with server.viewer.txn() as s:
        s.layers["picks"].annotations = [
            ng.AxisAlignedBoundingBoxAnnotation(id="1", point_a=[9, 8, 7],
                                                point_b=[1, 2, 3]),
        ]
    assert server.boxes() == [((1, 2, 3), (9, 8, 7))]


def test_nothing_to_serve_is_an_error(stop_server):
    with pytest.raises(ServeProblem, match="at least one layer"):
        serve([])


# --- add_layer: one more array, into a viewer already open ------------------ #
def test_add_layer_hosts_another_array_live(frame, stop_server):
    """An array becomes a layer in the viewer already open, with no new URL."""
    server = serve([ServedLayer(_labels(), kind="image", name="em", frame=frame)])
    url = server.url

    assert server.add_layer(
        ServedLayer(_labels(dtype="uint64"), kind="segmentation", name="seg",
                    frame=frame)) is server, "the server comes back, so calls chain"
    assert server.url == url, "the same viewer, not a new one"
    assert [lyr["name"] for lyr in server.state()["layers"]] == ["em", "seg"]
    assert sorted(server.volumes) == ["em", "seg"], "and it is a served volume"
    assert server.selected_segments() == set(), "so the read-back methods find it"


def test_add_layer_goes_through_the_SAME_builder_as_serve(frame, stop_server):
    """Volume type, shader and voxel offset must match what `serve` builds."""
    layer = ServedLayer(_labels(dtype="uint8"), kind="segmentation", name="seg",
                        frame=frame)
    at_serve = serve([layer]).state()["layers"][0]

    added = serve([ServedLayer(_labels(), kind="image", name="em", frame=frame)])
    added.add_layer(layer)
    later = [lyr for lyr in added.state()["layers"] if lyr["name"] == "seg"][0]

    assert later["type"] == at_serve["type"] == "segmentation", \
        "a uint8 label array, which neuroglancer would otherwise guess is an image"
    assert added.volumes["seg"].volume_type == "segmentation"
    assert added.volumes["seg"].voxel_offset.tolist() == [10, 20, 30], "the frame's origin"


def test_a_name_collision_is_renamed_not_shadowed(frame, stop_server):
    """neuroglancer keys a layer by name, so a second "dup" would shadow the first."""
    at_serve = serve([ServedLayer(_labels(), name="dup", frame=frame),
                      ServedLayer(_labels(), name="dup", frame=frame)])
    assert [lyr["name"] for lyr in at_serve.state()["layers"]] == ["dup", "dup_1"]

    server = serve([ServedLayer(_labels(), name="dup", frame=frame)])
    server.add_layer(ServedLayer(_labels(), name="dup", frame=frame))
    assert list(server.volumes) == ["dup", "dup_1"], "insertion-ordered, newest last"
    assert [lyr["name"] for lyr in server.state()["layers"]] == ["dup", "dup_1"]


def test_add_layer_does_not_move_the_view(frame, stop_server):
    """Adding a layer must not change where somebody is looking."""
    from neu_lib import Frame

    server = serve([ServedLayer(_labels(), name="em", frame=frame)])
    before = (server.position, server.viewer.state.cross_section_scale)

    far = Frame(voxel_size_nm=(40.0, 8.0, 8.0), origin_nm=(40000.0, 8000.0, 8000.0))
    server.add_layer(ServedLayer(_labels(), name="elsewhere", frame=far))
    assert (server.position, server.viewer.state.cross_section_scale) == before
    assert server.bounds("elsewhere").lo == (1000, 1000, 1000), \
        "it is off screen, and bounds() is how you find it"


def test_add_layer_takes_the_dict_form(frame, stop_server):
    server = serve([ServedLayer(_labels(), name="em", frame=frame)])
    assert server.add_layer({"array": _labels(), "kind": "segmentation",
                             "name": "seg", "frame": frame}) is server
    assert server.volumes["seg"].volume_type == "segmentation", "built from the dict"


def test_a_piece_is_a_layer_input_everywhere(frame, stop_server):
    """`serve` and `add_layer` both accept a Piece, named after it and keeping its kind."""
    from neu_lib import Piece

    em = Piece(array=_labels(), frame=frame, kind="image", name="em")
    seg = Piece(array=_labels(dtype="uint64"), frame=frame, kind="segmentation", name="seg")

    server = serve([em]).add_layer(seg)

    assert [lyr["name"] for lyr in server.state()["layers"]] == ["em", "seg"], \
        "named after the pieces, which is what `read_piece` puts there"
    assert server.volumes["seg"].volume_type == "segmentation", "and the kind travelled"
    assert server.volumes["em"].volume_type == "image"


def test_a_piece_layer_still_honours_an_explicit_name(frame, stop_server):
    from neu_lib import Piece

    server = serve([ServedLayer(_labels(), name="em", frame=frame)])
    server.add_layer(Piece(array=_labels(), frame=frame, kind="image", name="ignored"),
                     name="mine")
    assert "mine" in server.volumes and "ignored" not in server.volumes


def test_a_piece_with_no_kind_says_what_to_do(frame, stop_server):
    """A Piece without a kind cannot be served: there is no kind= to pass."""
    from neu_lib import Piece

    server = serve([ServedLayer(_labels(), name="em", frame=frame)])
    with pytest.raises(ServeProblem, match="kind= is required"):
        server.add_layer(Piece(array=_labels(), frame=frame))


def test_anything_else_is_refused_by_the_one_coercion(frame, stop_server):
    server = serve([ServedLayer(_labels(), name="em", frame=frame)])
    with pytest.raises(ServeProblem, match="a ServedLayer, a neu_lib.Piece"):
        server.add_layer(_labels())
    with pytest.raises(ServeProblem, match="a ServedLayer, a neu_lib.Piece"):
        serve([_labels()])


# --- enclose: points -> a real box ------------------------------------------ #
def _drawn(server, *annotations, layer="annotations"):
    with server.viewer.txn() as s:
        s.layers[layer].annotations = list(annotations)
    return server


def _annotations(server, layer="annotations"):
    return server.viewer.state.layers[layer].layer.annotations


def test_enclose_bounds_the_points_and_draws_the_box(frame, stop_server):
    """Points clicked at the corners become one half-open box, added to the layer."""
    server = serve([ServedLayer(_labels(), name="b", frame=frame)], annotations=True)
    _drawn(server,
           ng.PointAnnotation(id="a", point=[10, 20, 30]),
           ng.PointAnnotation(id="b", point=[14, 25, 40]))

    assert server.enclose() == BBox((10, 20, 30), (15, 26, 41)), \
        "half-open, so the far point's own voxel is inside"
    assert len(_annotations(server)) == 3, "the points are left in place by default"
    assert tuple(server.enclose(replace=True)) == ((10, 20, 30), (15, 26, 41)), \
        "and it unpacks like the (lo, hi) pair it is"
    assert server.boxes() == [((10, 20, 30), (15, 26, 41))], "it is in the layer"


def test_a_POSITION_and_a_FACE_are_not_rounded_the_same(frame, stop_server):
    """A point sits inside a voxel, so its box runs to floor+1, wherever in the voxel it
    was clicked. A box corner is a boundary, so enclosing a box is a fixed point."""
    server = serve([ServedLayer(_labels(), name="b", frame=frame)], annotations=True)

    _drawn(server, ng.PointAnnotation(id="a", point=[10, 20, 30]))
    assert server.enclose(replace=True) == BBox((10, 20, 30), (11, 21, 31)), "a position"
    _drawn(server, ng.PointAnnotation(id="a", point=[10.4, 20.4, 30.4]))
    assert server.enclose(replace=True) == BBox((10, 20, 30), (11, 21, 31)), "same voxel"

    _drawn(server, ng.AxisAlignedBoundingBoxAnnotation(
        id="b", point_a=[10, 20, 30], point_b=[11, 21, 31]))
    assert server.enclose(replace=True) == BBox((10, 20, 30), (11, 21, 31)), "faces"
    assert server.enclose(replace=True) == BBox((10, 20, 30), (11, 21, 31)), \
        "and enclosing it again is a fixed point, not a box that creeps outward"


def test_enclose_rounds_OUTWARD_where_boxes_rounds_to_nearest(frame, stop_server):
    """`points()` reports the nearest voxel; `enclose` must contain the points."""
    server = serve([ServedLayer(_labels(), name="b", frame=frame)], annotations=True)
    _drawn(server,
           ng.PointAnnotation(id="a", point=[10.4, 20.9, 30.1]),
           ng.PointAnnotation(id="b", point=[14.6, 25.2, 40.8]))

    assert server.points() == [(10, 21, 30), (15, 25, 41)], "nearest, as ever"
    assert server.enclose() == BBox((10, 20, 30), (15, 26, 41)), "outward, and contains them"


def test_enclose_takes_a_margin_per_axis_or_for_all(frame, stop_server):
    """Per axis is what an anisotropic volume usually wants."""
    server = serve([ServedLayer(_labels(), name="b", frame=frame)], annotations=True)
    _drawn(server, ng.PointAnnotation(id="a", point=[10, 20, 30]),
           ng.PointAnnotation(id="b", point=[14, 25, 40]))

    assert server.enclose(margin=2, replace=True) == BBox((8, 18, 28), (17, 28, 43))

    _drawn(server, ng.PointAnnotation(id="a", point=[10, 20, 30]),
           ng.PointAnnotation(id="b", point=[14, 25, 40]))
    assert server.enclose(margin=(1, 4, 4)) == BBox((9, 16, 26), (16, 30, 45))

    with pytest.raises(ServeProblem, match="margin has 2 axes"):
        server.enclose(margin=(1, 2))


def test_enclose_replace_leaves_only_the_box(frame, stop_server):
    """`replace=True` swaps the points for the box."""
    server = serve([ServedLayer(_labels(), name="b", frame=frame)], annotations=True)
    _drawn(server, ng.PointAnnotation(id="a", point=[10, 20, 30]),
           ng.PointAnnotation(id="b", point=[14, 25, 40]))

    server.enclose(replace=True)
    assert server.points() == [], "the points that went in are gone"
    assert [a.id for a in _annotations(server)] != ["a", "b"]
    assert len(_annotations(server)) == 1
    assert server.boxes() == [((10, 20, 30), (15, 26, 41))]


def test_enclose_without_replace_encloses_ITSELF_next_time(frame, stop_server):
    """Intended: the box drawn last time is part of what is drawn now."""
    server = serve([ServedLayer(_labels(), name="b", frame=frame)], annotations=True)
    _drawn(server, ng.PointAnnotation(id="a", point=[10, 20, 30]))

    assert server.enclose(margin=2) == BBox((8, 18, 28), (13, 23, 33))
    assert server.enclose(margin=2) == BBox((6, 16, 26), (15, 25, 35)), "grown by the margin"


def test_enclose_counts_every_annotation_type(frame, stop_server):
    """Lines and ellipsoids count too; ignoring one would make the box silently too small."""
    server = serve([ServedLayer(_labels(), name="b", frame=frame)], annotations=True)
    _drawn(server,
           ng.PointAnnotation(id="p", point=[10, 10, 10]),
           ng.AxisAlignedBoundingBoxAnnotation(id="b", point_a=[8, 8, 8],
                                               point_b=[12, 12, 12]),
           ng.LineAnnotation(id="l", point_a=[5, 10, 10], point_b=[10, 10, 10]),
           ng.EllipsoidAnnotation(id="e", center=[10, 10, 20], radii=[1, 1, 3]))

    assert server.enclose() == BBox((5, 8, 8), (12, 12, 23)), \
        "the line's low x and the ellipsoid's far z are both in"


def test_enclose_refuses_an_annotation_it_cannot_place():
    """An unknown annotation type raises rather than being dropped."""
    from neu_glance.serving import _annotation_bbox

    with pytest.raises(ServeProblem, match="cannot tell where a"):
        _annotation_bbox(object())


def test_enclose_with_nothing_drawn_says_so(frame, stop_server):
    server = serve([ServedLayer(_labels(), name="b", frame=frame)], annotations=True)
    with pytest.raises(ServeProblem, match="nothing is drawn"):
        server.enclose()

    plain = serve([ServedLayer(_labels(), name="c", frame=frame)])
    with pytest.raises(ServeProblem, match="no annotation layer"):
        plain.enclose()


def test_a_margin_that_collapses_an_axis_is_an_error(frame, stop_server):
    """Raises rather than returning a box with lo >= hi."""
    server = serve([ServedLayer(_labels(), name="b", frame=frame)], annotations=True)
    _drawn(server, ng.PointAnnotation(id="a", point=[10, 20, 30]),
           ng.PointAnnotation(id="b", point=[14, 25, 40]))

    with pytest.raises(ServeProblem, match="leaves an empty box"):
        server.enclose(margin=-3)


def test_a_margin_reaching_past_the_data_warns_when_it_is_not_clipped(frame, stop_server,
                                                                      caplog):
    """Unclipped by default, but a negative coordinate is warned about."""
    import logging

    server = serve([ServedLayer(_labels(), name="b", frame=frame)], annotations=True)
    _drawn(server, ng.PointAnnotation(id="a", point=[2, 2, 2]))
    with caplog.at_level(logging.WARNING, logger="neu_glance.serving"):
        assert server.enclose(margin=5) == BBox((-3, -3, -3), (8, 8, 8))
    assert "outside the volume" in caplog.text


# --- clipping: the volume's extent, or where its data is -------------------- #
def test_bounds_is_the_served_arrays_extent_offset_included(frame, stop_server):
    """`_labels()` is (8, 16, 16) and the frame's origin is 10, 20, 30 voxels in."""
    server = serve([ServedLayer(_labels(), name="b", frame=frame)])
    assert server.bounds() == BBox((10, 20, 30), (18, 36, 46))
    assert server.bounds("b") == server.bounds(), "one layer, so the union is the same"
    with pytest.raises(ServeProblem, match="no served layer named"):
        server.bounds("nope")


def test_bounds_unions_the_layers_and_converts_a_COARSER_ones_voxels(frame, stop_server):
    """Bounds are in the first layer's voxels, so a coarser layer's are converted."""
    from neu_lib import Frame

    coarse = Frame(voxel_size_nm=(80.0, 16.0, 16.0), origin_nm=(400.0, 160.0, 240.0))
    server = serve([ServedLayer(_labels(), name="fine", frame=frame),
                    ServedLayer(_labels((4, 8, 8)), name="coarse", frame=coarse)])
    assert server.bounds("fine") == BBox((10, 20, 30), (18, 36, 46))
    assert server.bounds("coarse") == BBox((10, 20, 30), (18, 36, 46)), \
        "the same physical box, in the viewer's voxels"
    assert server.bounds() == BBox((10, 20, 30), (18, 36, 46))


def test_data_bounds_is_where_the_labels_are(frame, stop_server):
    """The extent of nonzero labels, origin included."""
    array = np.zeros((8, 16, 16), "uint64")
    array[2:5, 3:9, 4:6] = 7
    server = serve([ServedLayer(array, kind="segmentation", name="seg", frame=frame)])

    assert server.bounds() == BBox((10, 20, 30), (18, 36, 46))
    assert server.data_bounds() == BBox((12, 23, 34), (15, 29, 36)), "offset included"


def test_data_bounds_on_an_empty_layer_says_so(frame, stop_server):
    """Raises rather than returning an empty box."""
    server = serve([ServedLayer(np.zeros((8, 16, 16), "uint64"), kind="segmentation",
                                name="seg", frame=frame)])
    with pytest.raises(ServeProblem, match="entirely zero"):
        server.data_bounds()


def test_enclose_clips_to_the_volume_or_to_the_data(frame, stop_server):
    """`clip=` bounds the box to the volume, the data, or a given box."""
    array = np.zeros((8, 16, 16), "uint64")
    array[2:5, 3:9, 4:6] = 7
    server = serve([ServedLayer(array, kind="segmentation", name="seg", frame=frame)],
                   annotations=True)
    _drawn(server, ng.PointAnnotation(id="a", point=[13, 25, 35]))

    assert server.enclose(margin=50) == BBox((-37, -25, -15), (64, 76, 86)), "unclipped"
    assert server.enclose(margin=50, clip="volume") == server.bounds()
    assert server.enclose(margin=50, clip="data") == server.data_bounds()
    assert server.enclose(margin=50, clip=server.bounds("seg")) == server.bounds("seg"), \
        "an explicit box, for staying inside one particular layer"
    assert server.enclose(margin=50, clip=((0, 0, 0), (20, 30, 40))) \
        == BBox((0, 0, 0), (20, 30, 40)), "or a plain (lo, hi) pair"


def test_a_clip_that_leaves_nothing_is_an_error(frame, stop_server):
    """A clip that empties the box raises, and so does a clip of the wrong type."""
    server = serve([ServedLayer(_labels(), name="b", frame=frame)], annotations=True)
    _drawn(server, ng.PointAnnotation(id="a", point=[10, 20, 30]))
    with pytest.raises(ServeProblem, match="leaves nothing"):
        server.enclose(clip=((900, 900, 900), (999, 999, 999)))
    with pytest.raises(ServeProblem, match="clip must be a BBox"):
        server.enclose(clip="whatever")


# --- reading the browser's state back -------------------------------------- #
def test_boxes_come_back_sorted_per_axis(frame, stop_server):
    """A box drawn up-and-left has point_a > point_b; boxes() returns lo < hi."""
    server = serve([ServedLayer(_labels(), name="a", frame=frame)], annotations=True)
    with server.viewer.txn() as s:
        s.layers["annotations"].annotations = [
            ng.AxisAlignedBoundingBoxAnnotation(id="1", point_a=[9, 8, 7],
                                               point_b=[1, 2, 3]),
            ng.PointAnnotation(id="2", point=[4, 5, 6]),
        ]
    assert server.boxes() == [((1, 2, 3), (9, 8, 7))]
    assert server.points() == [(4, 5, 6)], "points are read apart from boxes"


def test_selected_segments_reads_the_segmentation_layer(frame, stop_server):
    server = serve([ServedLayer(_labels(), kind="segmentation", name="seg",
                                frame=frame)])
    assert server.selected_segments() == set()
    with server.viewer.txn() as s:
        s.layers["seg"].segments = {3, 5}
    assert server.selected_segments() == {3, 5}


def test_asking_for_segments_with_no_sole_segmentation_says_so(frame, stop_server):
    server = serve([ServedLayer(_labels(), kind="image", name="em", frame=frame)])
    with pytest.raises(ServeProblem, match="say which layer"):
        server.selected_segments()


def test_a_click_handler_is_bound_and_survives_a_bad_callback(frame, stop_server):
    """A failing callback must not raise into neuroglancer's loop, and a click off the data
    does not call it."""
    server = serve([ServedLayer(_labels(), name="a", frame=frame)])
    action = server.on_click(lambda click: None)
    bindings = server.viewer.config_state.state.input_event_bindings.data_view.to_json()
    assert bindings["control+mousedown0"] == action

    boom = server.on_click(lambda click: 1 / 0, button=2)
    handler = next(iter(server.viewer.actions._action_handlers[boom]))

    class _State:
        mouse_voxel_coordinates = np.array([1.0, 2.0, 3.0])
        selected_values = {}

    handler(_State())          # must not raise

    # no position means the cursor is not over data; the callback is simply not called
    seen = []
    named = server.on_click(seen.append, button=1)
    nohit = next(iter(server.viewer.actions._action_handlers[named]))

    class _Nowhere:
        mouse_voxel_coordinates = None
        selected_values = {}

    nohit(_Nowhere())
    assert seen == []


def test_click_position_floors_to_the_voxel_actually_clicked():
    from neu_glance.serving import Click

    assert Click(position=(1.7, 2.2, 3.9)).voxel == (1, 2, 3)


# --- the constructors: infer what a source records, require what it does not -- #


def _piece(tmp_path, name="piece.h5", **attrs):
    h5py = pytest.importorskip("h5py")
    path = str(tmp_path / name)
    with h5py.File(path, "w") as f:
        d = f.create_dataset("data", data=_labels((6, 8, 8), "uint64"))
        for key, value in attrs.items():
            d.attrs[key] = value
    return path


def test_from_array_builds_the_frame_for_you():
    layer = ServedLayer.from_array(_labels(), "segmentation", voxel_size=(40, 8, 8),
                                   origin=(80, 16, 24))
    assert layer.kind == "segmentation"
    assert tuple(layer.frame.voxel_size_nm) == (40.0, 8.0, 8.0)
    assert tuple(layer.frame.origin_nm) == (80.0, 16.0, 24.0)
    assert layer.channel_axis is False


def test_the_channel_axis_comes_from_the_RANK_and_cannot_disagree_with_it():
    """4-D means a channel axis and 3-D means none; a contradicting flag raises."""
    assert ServedLayer.from_array(np.zeros((3, 4, 4, 4)), "probability").channel_axis
    assert not ServedLayer.from_array(np.zeros((4, 4, 4)), "image").channel_axis
    assert ServedLayer(np.zeros((3, 4, 4, 4)), kind="probability").channel_axis, \
        "the plain constructor derives it too"
    with pytest.raises(ServeProblem, match="contradicts a 4-D array"):
        ServedLayer.from_array(np.zeros((3, 4, 4, 4)), "probability", channel_axis=False)
    with pytest.raises(ServeProblem, match="3-D or 4-D"):
        ServedLayer.from_array(np.zeros((4, 4)), "image")


def test_two_frames_for_one_array_is_refused():
    from neu_lib import Frame

    with pytest.raises(ServeProblem, match="either frame="):
        ServedLayer.from_array(_labels(), "image",
                               frame=Frame(voxel_size_nm=(8, 8, 8)),
                               voxel_size=(4, 4, 4))


def test_from_hdf5_takes_the_frame_the_file_records(tmp_path):
    """Voxel size and offset come from the attributes `neu-vol to-hdf5` writes."""
    path = _piece(tmp_path, voxel_size=np.asarray([40.0, 8.0, 8.0]),
                  voxel_offset=np.asarray([2, 3, 4], "int64"), axes="zyx")
    layer = ServedLayer.from_hdf5(path, kind="segmentation")
    assert layer.array.shape == (6, 8, 8)
    assert tuple(layer.frame.voxel_size_nm) == (40.0, 8.0, 8.0)
    assert tuple(layer.frame.origin_nm) == (80.0, 24.0, 32.0)
    # `stem/dataset`: either half alone collides across files or datasets
    assert layer.name == "piece/data"


def test_from_hdf5_needs_a_kind_and_says_why(tmp_path):
    """HDF5 records no kind, and the dtype must not decide it."""
    path = _piece(tmp_path, voxel_size=np.asarray([8.0, 8.0, 8.0]), axes="zyx")
    with pytest.raises(ServeProblem, match="kind= is required"):
        ServedLayer.from_hdf5(path)
    assert ServedLayer.from_hdf5(path, kind="image").kind == "image"


def test_from_hdf5_needs_a_voxel_size_when_the_file_records_none(tmp_path):
    path = _piece(tmp_path)
    with pytest.raises(ServeProblem, match="records no voxel size"):
        ServedLayer.from_hdf5(path, kind="image")
    layer = ServedLayer.from_hdf5(path, kind="image", voxel_size=(30, 6, 6))
    assert tuple(layer.frame.voxel_size_nm) == (30.0, 6.0, 6.0)


def test_a_crop_lands_at_the_sum_of_the_pieces_origin_and_the_box(tmp_path):
    """The crop's origin is the piece's offset plus the box, as in `neu-vol to-hdf5`."""
    path = _piece(tmp_path, voxel_size=np.asarray([8.0, 8.0, 8.0]),
                  voxel_offset=np.asarray([100, 100, 100], "int64"), axes="zyx")
    layer = ServedLayer.from_hdf5(path, kind="segmentation",
                                  crop=((1, 2, 3), (5, 6, 7)))
    assert layer.array.shape == (4, 4, 4)
    assert tuple(layer.frame.origin_nm) == (808.0, 816.0, 824.0)   # 100*8 + crop*8


def test_a_box_outside_the_extent_is_refused(tmp_path):
    path = _piece(tmp_path, voxel_size=np.asarray([8.0, 8.0, 8.0]), axes="zyx")
    with pytest.raises(ServeProblem, match="does not fit"):
        ServedLayer.from_hdf5(path, kind="image", crop=((0, 0, 0), (99, 8, 8)))


def test_a_multi_dataset_container_must_be_told_which_array(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = str(tmp_path / "bag.h5")
    with h5py.File(path, "w") as f:
        for key in ("a", "b"):
            d = f.create_dataset(key, data=_labels((4, 4, 4), "uint64"))
            d.attrs["voxel_size"] = np.asarray([8.0, 8.0, 8.0])
            d.attrs["axes"] = "zyx"

    with pytest.raises(KeyError, match="2 volumetric datasets"):
        ServedLayer.from_hdf5(path, kind="segmentation")
    assert ServedLayer.from_hdf5(path, "/b", "segmentation").name == "bag/b"


def test_a_level_on_a_single_array_is_an_error(tmp_path):
    path = _piece(tmp_path, voxel_size=np.asarray([8.0, 8.0, 8.0]), axes="zyx")
    with pytest.raises(ServeProblem, match="needs a multiscale volume"):
        ServedLayer.from_source(path, "image", level=2)


def test_from_source_splits_a_dataset_only_on_a_LEADING_slash(tmp_path):
    """Only `:/name` selects an array, since `s3://...` has a colon of its own."""
    path = _piece(tmp_path, voxel_size=np.asarray([8.0, 8.0, 8.0]), axes="zyx")
    assert ServedLayer.from_source(f"{path}:/data", "image").name == "piece/data"
    # no leading slash: the whole string is the path, and there is no such file
    with pytest.raises((ServeProblem, FileNotFoundError)):
        ServedLayer.from_source(f"{path}:data", "image")


def _ome_volume(tmp_path, kind="image", dtype="uint8", profile="local"):
    """A (16, 32, 32) volume at 40x8x8 nm with one (1, 2, 2) level: a multiscale zarr
    group by default, precomputed with ``profile="local-neuroglancer"``."""
    from neu_vol import convert
    from neu_vol.backends.tensorstore import TensorStoreBackend
    from neu_vol.profiles import zarr3_create_spec

    src = str(tmp_path / "src.zarr")
    be = TensorStoreBackend.create(
        zarr3_create_spec("local", src, (16, 32, 32), dtype,
                          dimension_names=("z", "y", "x"), chunk=(8, 8, 8)),
        delete_existing=True)
    be.write_region(tuple(slice(0, s) for s in (16, 32, 32)),
                    _labels((16, 32, 32), dtype))
    vol = str(tmp_path / ("vol.zarr" if profile == "local" else "vol"))
    convert(src, vol, voxel_size=(40, 8, 8), kind=kind, profile=profile,
            chunk=(8, 8, 8), factors=[(1, 2, 2)], min_dim=8, delete_existing=True)
    return vol


def test_from_volume_takes_kind_from_what_the_volume_RECORDS(tmp_path):
    """precomputed records the kind as `info["type"]`; an explicit kind still wins."""
    vol = _ome_volume(tmp_path, "segmentation", "uint32", profile="local-neuroglancer")
    layer = ServedLayer.from_volume(vol)          # no kind passed
    assert layer.kind == "segmentation"
    assert tuple(layer.frame.voxel_size_nm) == (40.0, 8.0, 8.0)
    assert ServedLayer.from_volume(vol, "image").kind == "image"


def test_a_coarser_level_reports_its_OWN_voxel_size(tmp_path):
    """Read from the volume, never 2**level: this pyramid halves x/y only."""
    vol = _ome_volume(tmp_path, "segmentation", "uint32", profile="local-neuroglancer")
    assert tuple(ServedLayer.from_volume(vol, level=1).frame.voxel_size_nm) \
        == (40.0, 16.0, 16.0)


def test_level_0_of_a_multiscale_ZARR_opens(tmp_path):
    """Regression: an OME-zarr group is not an array, so level 0 (the default) must open
    the level's subdirectory, not the group path."""
    vol = _ome_volume(tmp_path)
    assert ServedLayer.from_volume(vol, level=0).array.shape == (16, 32, 32)
    assert ServedLayer.from_volume(vol).array.shape == (16, 32, 32), "level 0 by default"
    assert ServedLayer.from_volume(vol, level=1).array.shape == (16, 16, 16)
    assert ServedLayer.from_volume(
        vol, crop=((0, 0, 0), (8, 16, 16))).array.shape == (8, 16, 16)


def test_a_bare_zarr_array_still_works(tmp_path):
    """A bare zarr array has no levels, so it must not be opened as `<path>/0`."""
    from neu_vol.backends.tensorstore import TensorStoreBackend
    from neu_vol.profiles import zarr3_create_spec

    bare = str(tmp_path / "bare.zarr")
    be = TensorStoreBackend.create(
        zarr3_create_spec("local", bare, (8, 8, 8), "uint8",
                          dimension_names=("z", "y", "x"), chunk=(8, 8, 8)),
        delete_existing=True)
    be.write_region(tuple(slice(0, 8) for _ in range(3)), _labels((8, 8, 8), "uint8"))
    layer = ServedLayer.from_volume(bare, "image", voxel_size=(8, 8, 8))
    assert layer.array.shape == (8, 8, 8)


def test_a_crop_takes_either_shape():
    """`((lo), (hi))` or the CLI's flat `(z0, y0, x0, z1, y1, x1)`. This is neu-vol's
    helper and its only test."""
    from neu_vol.piece import _crop_request

    assert _crop_request(((1, 2, 3), (4, 5, 6))) == (((1, 2, 3), (4, 5, 6)), None)
    assert _crop_request((1, 2, 3, 4, 5, 6)) == (((1, 2, 3), (4, 5, 6)), None)
    assert _crop_request(None) == (None, None)
    with pytest.raises(ValueError, match="VOXELS"):
        _crop_request((1, 2, 3, 4))
    with pytest.raises(ValueError, match="3 values each"):
        _crop_request(((1, 2), (3, 4)))


def test_a_crop_given_in_NANOMETRES_says_so(tmp_path):
    """A crop is in voxels; an out-of-extent box says which units it wanted."""
    vol = _ome_volume(tmp_path)
    with pytest.raises(ServeProblem, match="whole VOXELS"):
        ServedLayer.from_volume(vol, crop=((0, 0, 0), (640, 256, 256)))
    with pytest.raises(ServeProblem, match="NANOMETRES"):
        ServedLayer.from_volume(vol, crop=((0, 0, 0), (6400, 2560, 2560)))


def test_a_layer_reports_where_it_is_through_its_Piece(tmp_path):
    """`bbox` and `bounds_nm` come from the layer's Piece."""
    path = _piece(tmp_path, voxel_size=np.asarray([40.0, 8.0, 8.0]),
                  voxel_offset=np.asarray([2, 3, 4], "int64"), axes="zyx")
    layer = ServedLayer.from_hdf5(path, kind="segmentation")
    assert layer.bbox.lo == (2, 3, 4)
    assert layer.bbox.hi == (2 + 6, 3 + 8, 4 + 8)
    assert layer.bounds_nm[0] == (80.0, 24.0, 32.0)
    assert layer.piece.frame == layer.frame


def test_a_layer_with_no_frame_says_it_has_no_position():
    layer = ServedLayer.from_array(_labels(), "image")
    with pytest.raises(ServeProblem, match="no frame"):
        layer.bbox


def test_from_piece(tmp_path):
    from neu_lib import Frame, Piece

    piece = Piece(array=_labels((4, 8, 8), "uint64"),
                  frame=Frame(voxel_size_nm=(40, 8, 8), origin_nm=(80, 24, 32)))
    layer = ServedLayer.from_piece(piece, "segmentation", name="gt")
    assert layer.kind == "segmentation" and layer.name == "gt"
    assert layer.frame == piece.frame
    assert layer.bbox == piece.bbox


def test_crop_takes_the_same_PHYSICAL_box_as_another_layer(tmp_path):
    """`crop=<layer>` reads the same physical box, including at a coarser level."""
    vol = _ome_volume(tmp_path)                       # 40x8x8 nm at level 0
    gt = ServedLayer.from_volume(vol, "segmentation", crop=((4, 8, 8), (12, 24, 24)))
    assert gt.bbox.lo == (4, 8, 8)

    same = ServedLayer.from_volume(vol, "image", crop=gt)
    assert same.bounds_nm == gt.bounds_nm, "the same physical box"
    assert same.array.shape == gt.array.shape

    # ...and coarser, where the voxel box differs but the physical box does not
    coarse = ServedLayer.from_volume(vol, "image", level=1, crop=gt)
    assert coarse.bbox.lo != gt.bbox.lo
    assert coarse.bounds_nm[0] == gt.bounds_nm[0]
    assert coarse.frame.voxel_size_nm == (40.0, 16.0, 16.0)


def test_crop_also_takes_a_piece_or_an_explicit_nm_box(tmp_path):
    from neu_lib import Frame, Piece

    vol = _ome_volume(tmp_path)
    piece = Piece(array=np.zeros((8, 16, 16)),
                  frame=Frame(voxel_size_nm=(40, 8, 8), origin_nm=(160, 64, 64)))
    by_piece = ServedLayer.from_volume(vol, "image", crop=piece)
    by_nm = ServedLayer.from_volume(vol, "image", crop={"nm": piece.bounds_nm})
    assert by_piece.bounds_nm == by_nm.bounds_nm
    assert by_piece.array.shape == (8, 16, 16)

    with pytest.raises(ServeProblem, match="must carry 'nm'"):
        ServedLayer.from_volume(vol, "image", crop={"voxels": ((0, 0, 0), (1, 1, 1))})


def test_a_physical_box_clipped_to_almost_nothing_is_LOUD(tmp_path, caplog):
    """Mostly-clipped boxes warn, since they usually mean a crop from another dataset."""
    from neu_lib import Frame, Piece

    vol = _ome_volume(tmp_path)                       # level 0 is (16, 32, 32)
    # a box mostly past the volume's z extent
    far = Piece(array=np.zeros((8, 8, 8)),
                frame=Frame(voxel_size_nm=(40, 8, 8), origin_nm=(560, 0, 0)))
    with caplog.at_level("WARNING"):
        layer = ServedLayer.from_volume(vol, "image", crop=far)
    assert layer.array.shape[0] == 2, "clipped to what exists"
    assert "clipped to" in caplog.text and "different dataset" in caplog.text


def test_a_physical_box_entirely_outside_raises(tmp_path):
    from neu_lib import Frame, Piece

    vol = _ome_volume(tmp_path)
    away = Piece(array=np.zeros((8, 8, 8)),
                 frame=Frame(voxel_size_nm=(40, 8, 8), origin_nm=(99999, 0, 0)))
    with pytest.raises(ServeProblem, match="not the same volume"):
        ServedLayer.from_volume(vol, "image", crop=away)


def test_the_viewer_opens_CENTRED_on_the_data_not_at_the_origin(frame, stop_server):
    """Without a position, neuroglancer opens at the origin corner, often far from a crop."""
    layer = ServedLayer(_labels((8, 16, 16)), name="a", frame=frame)   # at voxel 10,20,30
    state = serve([layer]).state()
    assert "position" in state, "no position means the origin corner"
    lo, hi = layer.bbox.lo, layer.bbox.hi
    want = [(a + b) / 2 for a, b in zip(lo, hi)]
    assert [round(v, 3) for v in state["position"]] == [round(v, 3) for v in want]
    # scaled so the piece fits one panel: a 4-panel layout gives each about half the window
    from neu_glance.state import FIT_MARGIN, NOMINAL_VIEWPORT_PX

    assert state["crossSectionScale"] == pytest.approx(
        max(layer.piece.spatial_shape) * FIT_MARGIN / NOMINAL_VIEWPORT_PX / 0.5)
    # a single-panel layout needs no such correction
    single = serve([ServedLayer(_labels((8, 16, 16)), name="b", frame=frame)],
                   layout="xy").state()
    assert single["crossSectionScale"] == pytest.approx(
        max(layer.piece.spatial_shape) * FIT_MARGIN / NOMINAL_VIEWPORT_PX)


def test_the_opening_view_spans_EVERY_layer(stop_server):
    """The opening view frames the union of every layer, computed in nm."""
    from neu_lib import Frame

    small = ServedLayer(_labels((4, 4, 4)), name="gt", kind="segmentation",
                        frame=Frame(voxel_size_nm=(8, 8, 8), origin_nm=(80, 80, 80)))
    big = ServedLayer(_labels((32, 32, 32)), name="em",
                      frame=Frame(voxel_size_nm=(8, 8, 8), origin_nm=(0, 0, 0)))
    state = serve([big, small]).state()
    # union is 0..256 nm = 0..32 voxels, so the centre is 16
    assert [round(v) for v in state["position"]] == [16, 16, 16]


def test_no_frame_anywhere_leaves_neuroglancers_own_default(stop_server):
    """Nothing to centre on, so nothing is claimed."""
    state = serve([ServedLayer(_labels(), name="a")]).state()
    assert "position" not in state


def test_the_cli_annotation_flags():
    """serve's `--annotate` is opt-in; `gen --annotations SOURCE` loads a source and is not
    a serve flag."""
    from neu_glance.cli import _parse_args

    assert _parse_args(["serve", "--seg", "x.h5"]).annotate is False
    assert _parse_args(["serve", "--seg", "x.h5", "--annotate"]).annotate is True
    assert _parse_args(["serve", "--seg", "x.h5", "--annotate", "picks"]).annotate == "picks"

    assert _parse_args(["gen", "--annotations", "s3://my-bucket/syn"]).annotations \
        == ["s3://my-bucket/syn"]
    with pytest.raises(SystemExit):
        _parse_args(["serve", "--seg", "x.h5", "--annotations", "picks"])


def test_stop_serving_works_without_a_handle(frame, stop_server):
    """One server per process, so it can be stopped without the Server handle."""
    import neuroglancer as _ng

    from neu_glance import stop_serving

    assert stop_serving() is False, "nothing running yet"
    serve([ServedLayer(_labels(), name="a", frame=frame)])          # result discarded
    assert _ng.server.is_server_running()
    assert stop_serving() is True
    assert not _ng.server.is_server_running()
    assert stop_serving() is False, "idempotent"


def test_serve_does_not_block(frame, stop_server):
    """The server runs on a background thread; only the `neu-glance serve` command blocks."""
    import threading
    import time

    before = threading.active_count()
    start = time.monotonic()
    server = serve([ServedLayer(_labels(), name="a", frame=frame)])
    assert time.monotonic() - start < 20, "serve() returned promptly"
    assert server.url and threading.active_count() > before
