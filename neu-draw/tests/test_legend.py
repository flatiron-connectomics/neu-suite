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
        assert view.legend.names == ["1401 mesh", "1401 skeleton", "presyn"]
    finally:
        view.close()


def test_the_glyph_says_what_kind_of_thing_the_row_is(has_gpu):
    """A body's mesh and its skeleton are two entries with the SAME colour, so a
    colour-only legend cannot tell them apart."""
    scene = build_scene(meshes=[_mesh("1401")], skeletons=[_skeleton("1401")],
                        points={"presyn": np.array([[0.0, 0, 0]])})
    view = backend.show(scene, size=(400, 300), canvas="offscreen")
    try:
        kinds = {e.drawable.name: type(e.glyph).__name__ for e in view.legend}
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
        assert view.legend.names == ["body 0", "body 1"]
        assert view.legend["body 1"].label.material.color != Legend().text_color
    finally:
        view.close()


def test_an_unnamed_drawable_gets_no_entry(has_gpu):
    scene = Scene().add_mesh(_mesh("named")).add_points(np.array([[0.0, 0, 0]]))
    view = backend.show(scene, size=(400, 300), canvas="offscreen")
    try:
        assert view.legend.names == ["named"]
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


def test_a_highlight_survives_a_rename_and_a_relayout(has_gpu):
    view = backend.show(_scene(3), size=(400, 300), canvas="offscreen")
    try:
        view.legend.highlight("body 1")
        view.legend.rename("body 1", "a much longer name than before")
        assert view.legend.highlighted == ["a much longer name than before"]
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
# renaming and recolouring from a notebook
# --------------------------------------------------------------------------- #

def test_renaming_an_entry_renames_the_drawable(has_gpu):
    """One name, not a name and a separate display label. A label held alongside the name
    would let the two disagree: you relabel the row and `scene.get(that_label)` raises."""
    view = backend.show(_scene(2), size=(400, 300), canvas="offscreen")
    try:
        view.legend.rename("body 0", "MeCN-01 (L)")
        assert view.legend.names == ["MeCN-01 (L)", "body 1"]
        assert view.scene_data.get("MeCN-01 (L)") is not None
        assert view.legend["MeCN-01 (L)"].label_width > 0
    finally:
        view.close()


def test_renaming_onto_an_existing_name_is_refused(has_gpu):
    view = backend.show(_scene(2), size=(400, 300), canvas="offscreen")
    try:
        with pytest.raises(ValueError, match="already here"):
            view.legend.rename("body 0", "body 1")
    finally:
        view.close()


def test_a_longer_name_widens_the_strip(has_gpu):
    view = backend.show(_scene(2), size=(600, 400), canvas="offscreen")
    try:
        before = view.legend.width
        view.legend.rename("body 0", "an extremely long descriptive cell name")
        assert view.legend.width > before
    finally:
        view.close()


def test_recolouring_moves_the_object_and_the_swatch_together(has_gpu):
    view = backend.show(_scene(2), size=(400, 300), canvas="offscreen")
    try:
        view.legend.recolor("body 0", "tab:pink")
        entry = view.legend["body 0"]
        assert entry.drawable.color == pytest.approx((0.8902, 0.4667, 0.7608, 1.0),
                                                     abs=1e-3)
        assert tuple(entry.glyph.material.color)[:3] == pytest.approx(
            entry.drawable.color[:3], abs=1e-3)
        assert tuple(entry.world_object.material.color)[:3] == pytest.approx(
            entry.drawable.color[:3], abs=1e-3)
    finally:
        view.close()


def test_recolouring_one_entry_leaves_the_others_alone(has_gpu):
    """`Scene.recolor` chooses over the whole set at once, so handing it one explicit
    colour re-draws every other name from the top of the palette. That is right when you
    are assigning a scene and wrong when you are adjusting one entry of a figure."""
    view = backend.show(_scene(3), size=(400, 300), canvas="offscreen")
    try:
        others = [view.legend[n].drawable.color for n in ("body 1", "body 2")]
        view.legend.recolor("body 0", "w")
        assert [view.legend[n].drawable.color for n in ("body 1", "body 2")] == others
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
