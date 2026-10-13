"""The legend: layout arithmetic, then a real render with real clicks.

The arithmetic half needs no GPU. The interaction half does, and it goes through the
**real** event path — ``renderer.convert_event`` is what a canvas calls, so feeding it
pointer dicts exercises picking, click synthesis and dispatch rather than a shortcut past
them. That matters here: everything interesting about this legend is in the plumbing
(which rect gets rendered, which object owns the pick id, which viewport the controller
was registered on), and a test that called ``entry.toggle()`` directly would check none
of it.
"""

import numpy as np
import pytest

pygfx = pytest.importorskip("pygfx", reason="the render extra is not installed")

from neu_lib import Mesh, Skeleton                              # noqa: E402
from neu_draw.backends import pygfx as backend                  # noqa: E402
from neu_draw.backends.legend import (MAX_WIDTH_FRACTION, MIN_WIDTH,  # noqa: E402
                                      split_rects)
from neu_draw.scene import Legend, Scene, build_scene           # noqa: E402


@pytest.fixture
def has_gpu():
    import wgpu
    try:
        wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    except Exception as exc:                                    # pragma: no cover
        pytest.skip(f"no wgpu adapter available: {exc}")
    return True


_FACES = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
_VERTS = np.array([[0.0, 0, 0], [100.0, 0, 0], [0.0, 100.0, 0], [0.0, 0, 100.0]])


def _mesh(name="body", shift=0.0):
    return Mesh(_VERTS + shift, _FACES, name=name)


def _skeleton(name="skel"):
    return Skeleton(np.array([[0.0, 0, 0], [0.0, 0, 300.0], [0.0, 150.0, 300.0]]),
                    np.array([[0, 1], [1, 2]]), name=name)


def _scene(n=3):
    return build_scene(meshes=[_mesh(f"body {i}", i * 300.0) for i in range(n)])


def _click(view, x, y, button=1):
    """A full press-and-release at a canvas position, through the renderer's own path."""
    def event(kind, stamp):
        return dict(event_type=kind, x=float(x), y=float(y), button=button,
                    buttons=(button,), modifiers=(), pointer_id=1, ntouches=0,
                    touches={}, time_stamp=stamp)

    view.renderer.convert_event(event("pointer_down", 1.0))
    view.renderer.convert_event(event("pointer_up", 1.05))


def _drag(view, x0, y0, x1, y1):
    def event(kind, x, y, stamp):
        return dict(event_type=kind, x=float(x), y=float(y), button=1, buttons=(1,),
                    modifiers=(), pointer_id=1, ntouches=0, touches={}, time_stamp=stamp)

    view.renderer.convert_event(event("pointer_down", x0, y0, 1.0))
    view.renderer.convert_event(event("pointer_move", x1, y1, 1.1))
    view.renderer.convert_event(event("pointer_up", x1, y1, 1.2))
    view.controller.tick()


# --------------------------------------------------------------------------- #
# intent, and the split arithmetic — no GPU
# --------------------------------------------------------------------------- #

def test_only_the_two_vertical_locations_are_offered():
    """A body name needs a column. 'top' would be a row of long strings, which is why it
    is refused rather than silently laid out badly."""
    with pytest.raises(ValueError, match="unknown legend location"):
        Legend(location="top")


def test_legend_colours_are_normalised_on_the_way_in():
    legend = Legend(text_color="k", row_color="#ffffff80", panel_color="w")
    assert legend.text_color == (0.0, 0.0, 0.0, 1.0)
    assert legend.row_color[3] == pytest.approx(128 / 255, abs=1e-3)
    assert legend.panel_color == (1.0, 1.0, 1.0, 1.0)


def test_an_unset_panel_colour_stays_none_so_the_scene_can_decide():
    assert Legend().panel_color is None


def test_the_split_leaves_the_scene_the_rest_of_the_canvas():
    main, strip = split_rects((400, 300), "right", 120)
    assert strip == (280.0, 0.0, 120.0, 300.0)
    assert main == (0.0, 0.0, 280.0, 300.0)


def test_a_left_legend_pushes_the_scene_right():
    main, strip = split_rects((400, 300), "left", 120)
    assert strip == (0.0, 0.0, 120.0, 300.0)
    assert main == (120.0, 0.0, 280.0, 300.0)


def test_the_width_cap_beats_the_width_floor_on_a_small_canvas():
    """These two disagree below ~180 px, and letting the floor win gave the SCENE a rect
    of width zero — a division by zero inside the camera's projection, reached from a
    50-px-wide test canvas rather than from anything a user would do."""
    main, strip = split_rects((50, 40), "right", MIN_WIDTH)
    assert strip[2] == pytest.approx(50 * MAX_WIDTH_FRACTION)
    assert main[2] > 0


def test_the_scene_keeps_the_whole_canvas_when_there_is_no_legend(has_gpu):
    view = backend.show(_scene(), size=(200, 150), canvas="offscreen", legend=False)
    try:
        assert view.legend is None
        assert view._main_size() == (200.0, 150.0)
    finally:
        view.close()


# --------------------------------------------------------------------------- #
# what gets built
# --------------------------------------------------------------------------- #

def test_one_entry_per_named_drawable_in_scene_order(has_gpu):
    scene = build_scene(meshes=[_mesh("1401")], skeletons=[_skeleton("1401")],
                        points={"presyn": np.array([[0.0, 0, 0]])})
    view = backend.show(scene, size=(400, 300), canvas="offscreen")
    try:
        assert view.legend.labels == ["1401 mesh", "1401 skeleton", "presyn"]
    finally:
        view.close()


def test_the_glyph_says_what_kind_of_thing_the_row_is(has_gpu):
    """A body's mesh and its skeleton are two entries with the SAME colour, so a
    colour-only legend cannot tell them apart."""
    scene = build_scene(meshes=[_mesh("1401")], skeletons=[_skeleton("1401")],
                        points={"presyn": np.array([[0.0, 0, 0]])})
    view = backend.show(scene, size=(400, 300), canvas="offscreen")
    try:
        kinds = {e.text: type(e.glyph).__name__ for e in view.legend}
        assert kinds == {"1401 mesh": "Mesh", "1401 skeleton": "Line",
                         "presyn": "Points"}
    finally:
        view.close()


def test_a_hidden_drawable_still_gets_an_entry(has_gpu):
    """Otherwise it is a drawable nobody can turn back on, which is the entire reason the
    legend is clickable."""
    scene = _scene(2)
    scene.get("body 1").visible = False
    view = backend.show(scene, size=(400, 300), canvas="offscreen")
    try:
        assert view.legend.labels == ["body 0", "body 1"]
        assert view.legend["body 1"].label.material.color != Legend().text_color
    finally:
        view.close()


def test_an_unnamed_drawable_gets_no_entry(has_gpu):
    scene = Scene().add_mesh(_mesh("named")).add_points(np.array([[0.0, 0, 0]]))
    view = backend.show(scene, size=(400, 300), canvas="offscreen")
    try:
        assert view.legend.labels == ["named"]
    finally:
        view.close()


def test_a_scene_with_nothing_named_gets_no_strip_at_all(has_gpu):
    """Rather than an empty panel taking 45% of the canvas."""
    scene = Scene().add_points(np.array([[0.0, 0, 0]]))
    view = backend.show(scene, size=(400, 300), canvas="offscreen")
    try:
        assert view.legend is None
    finally:
        view.close()


def test_an_unknown_entry_name_says_what_there_is(has_gpu):
    view = backend.show(_scene(2), size=(400, 300), canvas="offscreen")
    try:
        with pytest.raises(KeyError, match="body 0"):
            view.legend["body 7"]
    finally:
        view.close()


# --------------------------------------------------------------------------- #
# groups: many drawables, one row
# --------------------------------------------------------------------------- #

def test_drawables_sharing_a_label_share_one_row(has_gpu):
    """The whole point: forty bodies of a cell type are one entry, not forty."""
    scene = build_scene(meshes=[_mesh(f"body {i}", i * 300.0) for i in range(5)],
                        labels={"body 0": "Tm2", "body 1": "Tm2", "body 2": "LC6"})
    view = backend.show(scene, size=(500, 400), canvas="offscreen")
    try:
        assert view.legend.labels == ["Tm2", "LC6", "body 3", "body 4"]
        assert view.legend["Tm2"].names == ["body 0", "body 1"]
        assert scene.names == [f"body {i}" for i in range(5)]   # identities untouched
    finally:
        view.close()


def test_a_grouped_row_says_how_many_it_covers(has_gpu):
    """Without the count, "Tm2" and "presyn" look like the same kind of thing, and hiding
    one is a much larger action than hiding the other."""
    scene = build_scene(meshes=[_mesh(f"b{i}", i * 300.0) for i in range(3)],
                        labels={"b0": "Tm2", "b1": "Tm2"})
    view = backend.show(scene, size=(500, 400), canvas="offscreen")
    try:
        assert view.legend["Tm2"].row_text == "Tm2 (2)"
        assert view.legend["b2"].row_text == "b2"
    finally:
        view.close()


def test_a_body_mesh_and_its_skeleton_land_in_the_same_group(has_gpu):
    """`labels` is keyed on the item's OWN name, before build_scene's mesh/skeleton
    suffix, because "label this body's geometry as Tm2" has to cover both."""
    scene = build_scene(meshes=[_mesh("1401")], skeletons=[_skeleton("1401")],
                        labels={"1401": "Tm2"})
    view = backend.show(scene, size=(500, 400), canvas="offscreen")
    try:
        assert view.legend.labels == ["Tm2"]
        assert view.legend["Tm2"].names == ["1401 mesh", "1401 skeleton"]
    finally:
        view.close()


def test_an_int_keyed_label_map_finds_a_string_named_body(has_gpu):
    """A body id arrives as an int about as often as a string, and a label map that
    silently matched nothing would look exactly like a legend bug."""
    scene = build_scene(meshes=[_mesh("10014014")], labels={10014014: "Tm2"})
    view = backend.show(scene, size=(400, 300), canvas="offscreen")
    try:
        assert view.legend.labels == ["Tm2"]
    finally:
        view.close()


def test_a_mixed_group_gets_the_neutral_square(has_gpu):
    """A type group normally holds meshes AND skeletons, and a line glyph on it would be a
    claim about the row that is only a third true."""
    scene = build_scene(meshes=[_mesh("1401")], skeletons=[_skeleton("1401")],
                        labels={"1401": "Tm2"})
    view = backend.show(scene, size=(400, 300), canvas="offscreen")
    try:
        assert type(view.legend["Tm2"].glyph).__name__ == "Mesh"
        assert view.legend["Tm2"].kinds == {"MeshDrawable", "LinesDrawable"}
    finally:
        view.close()


def test_a_group_of_one_kind_keeps_that_kinds_glyph(has_gpu):
    scene = build_scene(skeletons=[_skeleton("a"), _skeleton("b")],
                        labels={"a": "Tm2", "b": "Tm2"})
    view = backend.show(scene, size=(400, 300), canvas="offscreen")
    try:
        assert type(view.legend["Tm2"].glyph).__name__ == "Line"
    finally:
        view.close()


def test_clicking_a_group_hides_every_member(has_gpu):
    scene = build_scene(meshes=[_mesh(f"b{i}", i * 300.0) for i in range(3)],
                        labels={"b0": "Tm2", "b1": "Tm2"})
    view = backend.show(scene, size=(500, 400), canvas="offscreen", pixel_ratio=1.0)
    try:
        view.snapshot()
        strip = view.legend.rects_for(view.logical_size())[1]
        _click(view, strip[0] + strip[2] / 2, view.legend.row_height * 0.5)

        assert [d.visible for d in scene] == [False, False, True]
        assert [o.visible for o in view.group.children] == [False, False, True]
    finally:
        view.close()


def test_a_partly_hidden_group_shows_a_third_state(has_gpu):
    """Reachable without anyone clicking a row — hide one body by name, or reset a scene
    that started with something hidden. A row showing either extreme would be lying about
    half its members."""
    scene = build_scene(meshes=[_mesh("a"), _mesh("b", 300.0)],
                        labels={"a": "Tm2", "b": "Tm2"})
    view = backend.show(scene, size=(400, 300), canvas="offscreen")
    try:
        entry = view.legend["Tm2"]
        assert entry.visibility == "all"
        plates = {"all": tuple(entry.plate.material.color)}

        scene.get("a").visible = False
        entry.refresh()
        assert entry.visibility == "some"
        plates["some"] = tuple(entry.plate.material.color)

        scene.get("b").visible = False
        entry.refresh()
        assert entry.visibility == "none"
        plates["none"] = tuple(entry.plate.material.color)

        assert len(set(plates.values())) == 3
    finally:
        view.close()


def test_toggling_a_partly_hidden_group_hides_the_rest(has_gpu):
    """Anything showing means hide. Inverting each member would turn the group inside out,
    which is not what clicking one row is asking for."""
    scene = build_scene(meshes=[_mesh("a"), _mesh("b", 300.0)],
                        labels={"a": "Tm2", "b": "Tm2"})
    view = backend.show(scene, size=(400, 300), canvas="offscreen")
    try:
        scene.get("a").visible = False
        assert view.legend.toggle("Tm2") is False
        assert [d.visible for d in scene] == [False, False]
        assert view.legend.toggle("Tm2") is True
        assert [d.visible for d in scene] == [True, True]
    finally:
        view.close()


def test_recolouring_a_group_moves_every_member(has_gpu):
    scene = build_scene(meshes=[_mesh("a"), _mesh("b", 300.0)],
                        labels={"a": "Tm2", "b": "Tm2"})
    view = backend.show(scene, size=(400, 300), canvas="offscreen")
    try:
        view.legend.recolor("Tm2", "tab:pink")
        pink = pytest.approx((0.8902, 0.4667, 0.7608), abs=1e-3)
        assert [tuple(d.color)[:3] for d in scene] == [pink, pink]
        assert tuple(view.group.children[1].material.color)[:3] == pink
    finally:
        view.close()


def test_highlighting_a_group_lights_every_member(has_gpu):
    scene = build_scene(meshes=[_mesh("a"), _mesh("b", 300.0), _mesh("c", 600.0)],
                        labels={"a": "Tm2", "b": "Tm2"})
    view = backend.show(scene, size=(400, 300), canvas="offscreen")
    try:
        view.legend.highlight("Tm2")
        white = pytest.approx(Legend().highlight_color[:3], abs=1e-3)
        lit = [tuple(o.material.color)[:3] for o in view.group.children]
        assert lit[0] == white and lit[1] == white and lit[2] != white
    finally:
        view.close()


def test_a_group_labelled_scene_gets_one_colour_per_label(has_gpu):
    """Otherwise the palette hands out one colour per body and the row's single swatch can
    only show one of them, which is most of the value of grouping thrown away."""
    scene = build_scene(meshes=[_mesh(f"b{i}", i * 300.0) for i in range(4)],
                        labels={"b0": "Tm2", "b1": "Tm2", "b2": "LC6", "b3": "LC6"})
    assert scene.get("b0").color == scene.get("b1").color
    assert scene.get("b2").color == scene.get("b3").color
    assert scene.get("b0").color != scene.get("b2").color


def test_colouring_per_drawable_can_still_be_asked_for(has_gpu):
    scene = build_scene(meshes=[_mesh("a"), _mesh("b", 300.0)],
                        labels={"a": "Tm2", "b": "Tm2"}, color_by="name")
    assert scene.get("a").color != scene.get("b").color


def test_an_unnamed_drawable_can_still_join_a_labelled_row(has_gpu):
    """Which is why the legend pairs drawables with world objects by POSITION rather than
    looking them up by name: a labelled drawable need not have one."""
    scene = Scene().add_mesh(_mesh("named"), label="Tm2")
    scene.add_points(np.array([[0.0, 0, 0]]), label="Tm2")
    view = backend.show(scene, size=(400, 300), canvas="offscreen")
    try:
        assert view.legend.labels == ["Tm2"]
        assert len(view.legend["Tm2"].drawables) == 2
        assert view.legend["Tm2"].names == ["named"]
    finally:
        view.close()


# --------------------------------------------------------------------------- #
# layout
# --------------------------------------------------------------------------- #

def test_the_strip_is_sized_from_the_longest_label(has_gpu):
    """Measured, not estimated: pygfx lays text out eagerly, so the real width is known
    before anything is drawn and the strip is right on the first frame."""
    short = backend.show(_scene(1), size=(600, 400), canvas="offscreen")
    scene = Scene().add_mesh(_mesh("a name very much longer than the other one"))
    long = backend.show(scene, size=(600, 400), canvas="offscreen")
    try:
        assert long.legend.width > short.legend.width
    finally:
        short.close()
        long.close()


def test_many_entries_wrap_into_columns_rather_than_shrinking_to_nothing(has_gpu):
    """A single column of 30 bodies is several canvases tall, and fitting it means
    shrinking by that factor — which, since the shrink must keep the strip's aspect to
    avoid distorting text, spends most of the strip's width on empty margin. Measured
    before this: 4-px text beside 200 px of nothing."""
    view = backend.show(_scene(30), size=(700, 400), canvas="offscreen")
    try:
        view.snapshot()
        assert view.legend.columns > 1
        assert view.legend.rows * view.legend.columns >= 30
        # and the whole legend fits, so nothing had to be shrunk away
        strip = view.legend.rects_for((700, 400))[1]
        assert view.legend.content_height <= strip[3]
    finally:
        view.close()


def test_the_column_count_is_capped_by_the_width_budget(has_gpu):
    """A narrow canvas cannot afford columns, so it shrinks instead — the legend stays
    whole either way, which is the property worth keeping."""
    view = backend.show(_scene(40), size=(260, 200), canvas="offscreen")
    try:
        view.snapshot()
        strip = view.legend.rects_for((260, 200))[1]
        assert strip[2] <= 260 * MAX_WIDTH_FRACTION + 1e-6
        assert view.legend.columns >= 1
    finally:
        view.close()


def test_an_explicit_width_is_honoured(has_gpu):
    scene = build_scene(meshes=[_mesh("a")], legend=Legend(width=150))
    view = backend.show(scene, size=(800, 400), canvas="offscreen")
    try:
        assert view.legend.rects_for((800, 400))[1][2] == pytest.approx(150.0)
    finally:
        view.close()


def test_the_rects_follow_a_resized_canvas_with_no_handler(has_gpu):
    """Both rects are computed from the renderer's current size, so the controller's
    `is_inside` test cannot end up disagreeing with the rect that was drawn."""
    view = backend.show(_scene(2), size=(400, 300), canvas="offscreen")
    try:
        first = view.legend.viewport.rect
        view.canvas.set_logical_size(800, 300)
        assert view.legend.viewport.rect[0] > first[0]
    finally:
        view.close()


# --------------------------------------------------------------------------- #
# it is actually drawn, and it is in the saved figure
# --------------------------------------------------------------------------- #

def test_the_strip_is_painted_opaquely(has_gpu):
    """The panel is opaque on purpose. Docked beside the scene there is nothing behind it,
    so a translucent one composites against the *page* — fine in a dark-themed notebook
    and washed out in a light one."""
    view = backend.show(_scene(3), size=(400, 300), canvas="offscreen", pixel_ratio=1.0)
    try:
        image = view.snapshot()
        x0 = int(view.legend.rects_for((400, 300))[1][0])
        assert image[:, x0 + 2:, 3].min() == 255
        # …and something was drawn in it: a panel, plates and antialiased text.
        assert len(np.unique(image[:, x0 + 2:].reshape(-1, 4), axis=0)) > 10
    finally:
        view.close()


def test_the_legend_survives_the_offscreen_resnapshot(has_gpu):
    """`snapshot(size=...)` builds a renderer of its own, and the legend has to go through
    that pass too — a saved figure without its legend is not the figure. This was the one
    place the second render pass could quietly be skipped."""
    view = backend.show(_scene(3), size=(400, 300), canvas="offscreen", pixel_ratio=1.0)
    try:
        image = view.snapshot(size=(320, 240))
        x0 = int(view.legend.rects_for((320, 240))[1][0])
        assert image[:, x0 + 2:, 3].min() == 255
    finally:
        view.close()


def test_a_hidden_drawable_is_not_drawn_but_its_row_is(has_gpu):
    view = backend.show(_scene(2), size=(400, 300), canvas="offscreen", pixel_ratio=1.0)
    try:
        view.legend.set_visible("body 0", False)
        assert view.group.children[0].visible is False
        assert view.legend["body 0"].group.visible is True
    finally:
        view.close()


# --------------------------------------------------------------------------- #
# clicking
# --------------------------------------------------------------------------- #

def test_clicking_a_row_toggles_that_drawable(has_gpu):
    view = backend.show(_scene(3), size=(400, 300), canvas="offscreen", pixel_ratio=1.0)
    try:
        view.snapshot()                       # a pick buffer only exists after a render
        strip = view.legend.rects_for(view.logical_size())[1]
        x = strip[0] + strip[2] / 2
        row_h = view.legend.row_height

        _click(view, x, row_h * 0.5)
        assert view.scene_data.get("body 0").visible is False
        assert view.group.children[0].visible is False

        _click(view, x, row_h * 0.5)
        assert view.scene_data.get("body 0").visible is True
    finally:
        view.close()


def test_the_click_lands_on_the_row_it_looks_like_it_lands_on(has_gpu):
    """The plate is the only entry object that writes a pick id. If the glyph or the label
    wrote one too they would carve their own shapes out of the row, and a click just left
    of a short name would silently do nothing."""
    view = backend.show(_scene(3), size=(400, 300), canvas="offscreen", pixel_ratio=1.0)
    try:
        view.snapshot()
        strip = view.legend.rects_for(view.logical_size())[1]
        row_h = view.legend.row_height

        # the second row, at its far right — past the end of the label text
        _click(view, strip[0] + strip[2] - 3, row_h * 1.5)
        assert [d.visible for d in view.scene_data] == [True, False, True]
    finally:
        view.close()


def test_a_click_in_the_legend_does_not_also_spin_the_camera(has_gpu):
    """The controller is registered on the main viewport, and this is what that buys. A
    pointer handler flipping `controller.enabled` could not do it reliably: pygfx keeps
    event handlers in a **set**, so there is no order to register ahead of."""
    view = backend.show(_scene(3), size=(400, 300), canvas="offscreen", pixel_ratio=1.0)
    try:
        view.snapshot()
        strip = view.legend.rects_for(view.logical_size())[1]
        before = np.array(view.camera.local.position, dtype=float)

        _drag(view, strip[0] + 20, 200, strip[0] + 20, 260)
        assert np.allclose(view.camera.local.position, before)

        _drag(view, 60, 100, 140, 160)
        assert not np.allclose(view.camera.local.position, before)
    finally:
        view.close()


def test_a_middle_click_does_nothing(has_gpu):
    """Left is hide, right is highlight, and nothing else is claimed."""
    view = backend.show(_scene(2), size=(400, 300), canvas="offscreen", pixel_ratio=1.0)
    try:
        view.snapshot()
        strip = view.legend.rects_for(view.logical_size())[1]
        _click(view, strip[0] + strip[2] / 2, view.legend.row_height * 0.5, button=3)
        assert view.scene_data.get("body 0").visible is True
        assert view.legend.highlighted == []
    finally:
        view.close()


def test_centering_after_a_toggle_frames_only_what_is_left(has_gpu):
    """`get_bounding_box` walks every child regardless of `visible`, so framing the whole
    group would pull the camera back for a body that has just been switched off. It used
    not to matter, because a hidden drawable was never built at all."""
    scene = Scene().add_mesh(_mesh("near")).add_mesh(_mesh("far", 50_000.0))
    view = backend.show(scene, size=(400, 300), canvas="offscreen")
    try:
        both = float(np.linalg.norm(view.camera.local.position))
        view.legend.set_visible("far", False)
        view.center()
        assert float(np.linalg.norm(view.camera.local.position)) < both / 2
    finally:
        view.close()


# --------------------------------------------------------------------------- #
# highlighting
# --------------------------------------------------------------------------- #

def test_right_clicking_a_row_highlights_it(has_gpu):
    view = backend.show(_scene(3), size=(400, 300), canvas="offscreen", pixel_ratio=1.0)
    try:
        view.snapshot()
        strip = view.legend.rects_for(view.logical_size())[1]
        x, row_h = strip[0] + strip[2] / 2, view.legend.row_height

        _click(view, x, row_h * 1.5, button=2)
        assert view.legend.highlighted == ["body 1"]

        _click(view, x, row_h * 1.5, button=2)
        assert view.legend.highlighted == []
    finally:
        view.close()


def test_a_highlight_never_touches_the_drawables_own_colour(has_gpu):
    """The property the whole design rests on, and where the predecessor differed: it
    swapped the graphic's real colour and stashed the original. That holds until something
    else reads the colour — `Scene.recolor`, `bake()`, a saved scene — and then a temporary
    highlight has quietly become the body's colour."""
    view = backend.show(_scene(2), size=(400, 300), canvas="offscreen")
    try:
        before = view.scene_data.get("body 0").color
        view.legend.highlight("body 0")
        assert view.scene_data.get("body 0").color == before
        assert tuple(view.group.children[0].material.color)[:3] == pytest.approx(
            Legend().highlight_color[:3], abs=1e-3)
    finally:
        view.close()


def test_the_highlight_keeps_the_drawables_alpha(has_gpu):
    """Turning a translucent surface opaque would change what you can see through, and
    being translucent is often exactly why it could not be found."""
    scene = Scene().add_mesh(_mesh("faint"), alpha=0.4)
    view = backend.show(scene, size=(400, 300), canvas="offscreen")
    try:
        view.legend.highlight("faint")
        assert tuple(view.group.children[0].material.color)[3] == pytest.approx(0.4)
    finally:
        view.close()


def test_dropping_the_highlight_restores_the_real_colour(has_gpu):
    view = backend.show(_scene(2), size=(400, 300), canvas="offscreen")
    try:
        own = view.scene_data.get("body 0").color
        view.legend.highlight("body 0").unhighlight("body 0")
        assert tuple(view.group.children[0].material.color)[:3] == pytest.approx(
            own[:3], abs=1e-3)
        assert tuple(view.legend["body 0"].glyph.material.color)[:3] == pytest.approx(
            own[:3], abs=1e-3)
    finally:
        view.close()


def test_several_rows_can_be_lit_at_once_and_cleared_together(has_gpu):
    view = backend.show(_scene(3), size=(400, 300), canvas="offscreen")
    try:
        view.legend.highlight("body 0", "body 2")
        assert view.legend.highlighted == ["body 0", "body 2"]
        view.legend.clear_highlights()
        assert view.legend.highlighted == []
    finally:
        view.close()


def test_an_exclusive_highlight_drops_the_others(has_gpu):
    """The "where is this one" case, as against right-clicking rows one after another."""
    view = backend.show(_scene(3), size=(400, 300), canvas="offscreen")
    try:
        view.legend.highlight("body 0", "body 1")
        view.legend.highlight("body 2", exclusive=True)
        assert view.legend.highlighted == ["body 2"]
    finally:
        view.close()


def test_the_row_says_it_is_highlighted_and_not_only_the_swatch(has_gpu):
    """Once two entries are lit their swatches are both the highlight colour, and nothing
    then separates "temporarily lit" from "this body really is white"."""
    view = backend.show(_scene(2), size=(400, 300), canvas="offscreen")
    try:
        plain = tuple(view.legend["body 0"].plate.material.color)
        view.legend.highlight("body 0")
        assert tuple(view.legend["body 0"].plate.material.color) != plain
        assert tuple(view.legend["body 1"].plate.material.color) == plain
    finally:
        view.close()


def test_recolouring_a_highlighted_entry_takes_effect_when_it_is_dropped(has_gpu):
    """Because the highlight is an override rather than a swap, neither operation can lose
    the other's work."""
    view = backend.show(_scene(2), size=(400, 300), canvas="offscreen")
    try:
        view.legend.highlight("body 0")
        view.legend.recolor("body 0", "tab:pink")

        assert tuple(view.group.children[0].material.color)[:3] == pytest.approx(
            Legend().highlight_color[:3], abs=1e-3)
        view.legend.unhighlight("body 0")
        assert tuple(view.group.children[0].material.color)[:3] == pytest.approx(
            (0.8902, 0.4667, 0.7608), abs=1e-3)
    finally:
        view.close()


def test_hiding_a_highlighted_entry_still_hides_it(has_gpu):
    """The two states are independent, and `refresh` recomputes both from scratch — so the
    order they were set in cannot matter."""
    view = backend.show(_scene(2), size=(400, 300), canvas="offscreen")
    try:
        view.legend.highlight("body 0").set_visible("body 0", False)
        assert view.group.children[0].visible is False
        view.legend.set_visible("body 0", True)
        assert view.legend.highlighted == ["body 0"]
    finally:
        view.close()


def test_a_highlight_survives_a_relabel_and_a_relayout(has_gpu):
    """Relabelling **rebuilds** the rows, since it can merge and split them — so the
    highlight has to be carried across by label, or pressing a button and then relabelling
    would silently drop it."""
    view = backend.show(_scene(3), size=(400, 300), canvas="offscreen")
    try:
        view.legend.highlight("body 1")
        view.legend.relabel("body 1", "a much longer label than before")
        assert view.legend.highlighted == ["a much longer label than before"]
    finally:
        view.close()


def test_an_explicit_highlight_colour_is_honoured(has_gpu):
    """The knob for a figure whose own palette collides with white."""
    scene = build_scene(meshes=[_mesh("a")], legend=Legend(highlight_color="tab:cyan"))
    view = backend.show(scene, size=(400, 300), canvas="offscreen")
    try:
        view.legend.highlight("a")
        assert tuple(view.group.children[0].material.color)[:3] == pytest.approx(
            (0.0902, 0.7451, 0.8118), abs=1e-3)
    finally:
        view.close()


# --------------------------------------------------------------------------- #
# relabelling and recolouring from a notebook
# --------------------------------------------------------------------------- #

def test_relabelling_a_row_leaves_the_drawables_names_alone(has_gpu):
    """The difference from the `rename` this replaced. A row's text is a LABEL once a row
    can hold several drawables, so changing it must not touch identity — `scene.get` keeps
    answering to the name you built the scene with."""
    view = backend.show(_scene(2), size=(400, 300), canvas="offscreen")
    try:
        view.legend.relabel("body 0", "MeCN-01 (L)")
        assert view.legend.labels == ["MeCN-01 (L)", "body 1"]
        assert view.scene_data.names == ["body 0", "body 1"]
        assert view.scene_data.get("body 0").label == "MeCN-01 (L)"
        assert view.legend["MeCN-01 (L)"].label_width > 0
    finally:
        view.close()


def test_relabelling_takes_a_mapping(has_gpu):
    view = backend.show(_scene(3), size=(600, 400), canvas="offscreen")
    try:
        view.legend.relabel({"body 0": "Tm2", "body 2": "LC6"})
        assert view.legend.labels == ["Tm2", "body 1", "LC6"]
    finally:
        view.close()


def test_relabelling_two_rows_to_one_label_merges_them(has_gpu):
    """Relabel *is* the grouping operation, so merging after the fact needs no extra API."""
    view = backend.show(_scene(3), size=(600, 400), canvas="offscreen")
    try:
        view.legend.relabel({"body 0": "Tm2", "body 1": "Tm2"})
        assert view.legend.labels == ["Tm2", "body 2"]
        assert view.legend["Tm2"].names == ["body 0", "body 1"]
    finally:
        view.close()


def test_a_longer_label_widens_the_strip(has_gpu):
    view = backend.show(_scene(2), size=(600, 400), canvas="offscreen")
    try:
        before = view.legend.width
        view.legend.relabel("body 0", "an extremely long descriptive cell name")
        assert view.legend.width > before
    finally:
        view.close()


def test_a_direct_scene_rename_is_picked_up_by_refresh(has_gpu):
    """The bug Erik hit: `legend.labels` reported the new name while the canvas still drew
    the old text, because `refresh` only re-read materials and a row's text is baked into a
    `pygfx.Text`. It rebuilds the rows now."""
    view = backend.show(_scene(2), size=(400, 300), canvas="offscreen")
    try:
        before = view.legend["body 0"].label_width
        view.scene_data.rename("body 0", "a considerably longer name")
        view.legend.refresh()
        assert view.legend.labels == ["a considerably longer name", "body 1"]
        assert view.legend["a considerably longer name"].label_width > before
    finally:
        view.close()


# --------------------------------------------------------------------------- #
# keeping up with a scene edited from a cell
# --------------------------------------------------------------------------- #

def test_a_scene_method_repaints_without_anyone_asking(has_gpu):
    """`Scene.relabel` schedules the frame and the frame re-reads the scene, so an edit in
    a notebook cell lands with no `refresh()` and no re-run of `show()`. The two halves are
    checked separately: that the edit asks for a repaint, and that the repaint catches up."""
    view = backend.show(_scene(2), size=(400, 300), canvas="offscreen")
    try:
        asked = []
        view.scene_data.on_change(lambda: asked.append(1))
        assert view.request_draw in view.scene_data._listeners

        view.scene_data.relabel({"body 0": "Tm2"})
        assert asked == [1]                      # the repaint was requested…
        assert view.legend.sync() is True        # …and the frame catches up
        assert view.legend.labels == ["Tm2", "body 1"]
    finally:
        view.close()


def test_a_field_set_directly_is_caught_by_the_next_frame(has_gpu):
    """A `Scene` is a plain mutable dataclass, so `drawable.label = ...` cannot notify
    anyone. The legend re-reads the truth each frame instead, which misses nothing —
    half-automatic notification would be worse, because you would learn to rely on it."""
    view = backend.show(_scene(2), size=(400, 300), canvas="offscreen", pixel_ratio=1.0)
    try:
        view.scene_data.drawables[0].label = "Tm2"
        view.scene_data.drawables[1].label = "Tm2"
        view.snapshot()                          # any frame at all
        assert view.legend.labels == ["Tm2"]
        assert view.legend["Tm2"].row_text == "Tm2 (2)"
    finally:
        view.close()


def test_a_visibility_change_is_caught_without_a_rebuild(has_gpu):
    """Two fingerprints, two responses: a changed *structure* needs the rows rebuilt (new
    text layouts, a re-measured strip), a changed *state* only needs materials reassigned.
    Doing the expensive one for a colour change would be a per-frame cost for nothing."""
    view = backend.show(_scene(2), size=(400, 300), canvas="offscreen")
    try:
        entry = view.legend["body 0"]
        view.scene_data.drawables[0].visible = False
        assert view.legend.sync() is True
        assert view.legend["body 0"] is entry            # same row object: no rebuild
        assert view.group.children[0].visible is False
    finally:
        view.close()


def test_an_unchanged_scene_costs_nothing_per_frame(has_gpu):
    view = backend.show(_scene(3), size=(400, 300), canvas="offscreen")
    try:
        assert view.legend.sync() is False
        assert view.legend.sync() is False
    finally:
        view.close()


def test_a_snapshot_taken_straight_after_an_edit_is_current(has_gpu):
    """`sync` runs from `draw`, so `view.save(...)` right after a relabel writes the new
    text — the saved figure cannot lag behind the scene."""
    view = backend.show(_scene(2), size=(400, 300), canvas="offscreen", pixel_ratio=1.0)
    try:
        before = view.snapshot().copy()
        view.scene_data.relabel({"body 0": "a very much longer label indeed"})
        assert not np.array_equal(view.snapshot(), before)
        assert view.legend.labels == ["a very much longer label indeed", "body 1"]
    finally:
        view.close()


def test_view_refresh_catches_up_on_everything_at_once(has_gpu):
    """Including drawables on no legend row, which is why it does not go through the
    legend alone."""
    scene = Scene().add_mesh(_mesh("named"))
    scene.add_points(np.array([[0.0, 0, 0]]))            # unnamed, unlabelled: no row
    view = backend.show(scene, size=(400, 300), canvas="offscreen")
    try:
        scene.drawables[1].visible = False
        scene.drawables[1].color = (1.0, 0.0, 0.0, 1.0)
        view.refresh()
        assert view.group.children[1].visible is False
        assert tuple(view.group.children[1].material.color)[:3] == pytest.approx(
            (1.0, 0.0, 0.0), abs=1e-3)
    finally:
        view.close()


def test_editing_a_scene_after_its_view_closed_is_harmless(has_gpu):
    """A scene outlives the views built from it, and it holds a listener that pokes a
    canvas — so this is ordinary, not a mistake."""
    scene = _scene(2)
    view = backend.show(scene, size=(400, 300), canvas="offscreen")
    view.close()
    scene.relabel({"body 0": "Tm2"})                      # must not raise
    assert scene.labels == ["Tm2", "body 1"]


def test_recolouring_moves_the_object_and_the_swatch_together(has_gpu):
    view = backend.show(_scene(2), size=(400, 300), canvas="offscreen")
    try:
        view.legend.recolor("body 0", "tab:pink")
        entry = view.legend["body 0"]
        assert entry.drawables[0].color == pytest.approx((0.8902, 0.4667, 0.7608, 1.0),
                                                     abs=1e-3)
        assert tuple(entry.glyph.material.color)[:3] == pytest.approx(
            entry.drawables[0].color[:3], abs=1e-3)
        assert tuple(entry.world_objects[0].material.color)[:3] == pytest.approx(
            entry.drawables[0].color[:3], abs=1e-3)
    finally:
        view.close()


def test_recolouring_one_entry_leaves_the_others_alone(has_gpu):
    """`Scene.recolor` chooses over the whole set at once, so handing it one explicit
    colour re-draws every other name from the top of the palette. That is right when you
    are assigning a scene and wrong when you are adjusting one entry of a figure."""
    view = backend.show(_scene(3), size=(400, 300), canvas="offscreen")
    try:
        others = [view.legend[n].drawables[0].color for n in ("body 1", "body 2")]
        view.legend.recolor("body 0", "w")
        assert [view.legend[n].drawables[0].color for n in ("body 1", "body 2")] == others
    finally:
        view.close()


def test_the_glyph_is_opaque_whatever_the_drawables_alpha(has_gpu):
    """A scene alpha is about overlapping surfaces not hiding each other. A 0.2 swatch is
    just illegible."""
    scene = Scene().add_mesh(_mesh("faint"), alpha=0.15)
    view = backend.show(scene, size=(400, 300), canvas="offscreen")
    try:
        assert tuple(view.legend["faint"].glyph.material.color)[3] == 1.0
    finally:
        view.close()


def test_refresh_picks_up_a_scene_changed_behind_the_legends_back(has_gpu):
    view = backend.show(_scene(2), size=(400, 300), canvas="offscreen")
    try:
        view.scene_data.get("body 0").visible = False
        view.legend.refresh()
        assert view.group.children[0].visible is False
    finally:
        view.close()
