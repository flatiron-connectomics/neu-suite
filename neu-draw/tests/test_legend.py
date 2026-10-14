"""The legend: layout arithmetic, then a real render with real clicks.

Clicks go through ``renderer.convert_event``, the path a canvas uses, so picking, click
synthesis and viewport dispatch are all exercised rather than calling ``entry.toggle()``.
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

def test_legend_colours_are_normalised_on_the_way_in():
    legend = Legend(text_color="k", row_color="#ffffff80", panel_color="w")
    assert legend.text_color == (0.0, 0.0, 0.0, 1.0)
    assert legend.row_color[3] == pytest.approx(128 / 255, abs=1e-3)
    assert legend.panel_color == (1.0, 1.0, 1.0, 1.0)


def test_an_unset_panel_colour_stays_none_so_the_scene_can_decide():
    assert Legend().panel_color is None


@pytest.mark.parametrize("location, strip, main", [
    ("right", (280.0, 0.0, 120.0, 300.0), (0.0, 0.0, 280.0, 300.0)),
    ("left", (0.0, 0.0, 120.0, 300.0), (120.0, 0.0, 280.0, 300.0)),
])
def test_the_split_leaves_the_scene_the_rest_of_the_canvas(location, strip, main):
    assert split_rects((400, 300), location, 120) == (main, strip)


def test_the_width_cap_beats_the_width_floor_on_a_small_canvas():
    """Below ~180 px the two disagree; the floor winning would give the scene zero width."""
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

def test_one_entry_per_named_drawable_with_a_glyph_per_kind(has_gpu):
    """In scene order. A body's mesh and skeleton share a colour, so the glyph tells them
    apart."""
    scene = build_scene(meshes=[_mesh("1401")], skeletons=[_skeleton("1401")],
                        points={"presyn": np.array([[0.0, 0, 0]])})
    view = backend.show(scene, size=(400, 300), canvas="offscreen")
    try:
        assert view.legend.labels == ["1401 mesh", "1401 skeleton", "presyn"]
        kinds = {e.text: type(e.glyph).__name__ for e in view.legend}
        assert kinds == {"1401 mesh": "Mesh", "1401 skeleton": "Line",
                         "presyn": "Points"}
    finally:
        view.close()


def test_a_hidden_drawable_still_gets_an_entry(has_gpu):
    """Otherwise nothing could turn it back on."""
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
    """Forty bodies of one cell type are one entry; their names are untouched."""
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
    scene = build_scene(meshes=[_mesh(f"b{i}", i * 300.0) for i in range(3)],
                        labels={"b0": "Tm2", "b1": "Tm2"})
    view = backend.show(scene, size=(500, 400), canvas="offscreen")
    try:
        assert view.legend["Tm2"].row_text == "Tm2 (2)"
        assert view.legend["b2"].row_text == "b2"
    finally:
        view.close()


def test_a_body_mesh_and_its_skeleton_land_in_the_same_group(has_gpu):
    """`labels` is keyed on the item's own name, before the mesh/skeleton suffix."""
    scene = build_scene(meshes=[_mesh("1401")], skeletons=[_skeleton("1401")],
                        labels={"1401": "Tm2"})
    view = backend.show(scene, size=(500, 400), canvas="offscreen")
    try:
        assert view.legend.labels == ["Tm2"]
        assert view.legend["Tm2"].names == ["1401 mesh", "1401 skeleton"]
    finally:
        view.close()


def test_an_int_keyed_label_map_finds_a_string_named_body(has_gpu):
    scene = build_scene(meshes=[_mesh("10014014")], labels={10014014: "Tm2"})
    view = backend.show(scene, size=(400, 300), canvas="offscreen")
    try:
        assert view.legend.labels == ["Tm2"]
    finally:
        view.close()


@pytest.mark.parametrize("make, glyph, kinds", [
    # a mixed group gets the neutral square
    (lambda: build_scene(meshes=[_mesh("1401")], skeletons=[_skeleton("1401")],
                         labels={"1401": "Tm2"}), "Mesh", {"MeshDrawable", "LinesDrawable"}),
    # a group of one kind keeps that kind's glyph
    (lambda: build_scene(skeletons=[_skeleton("a"), _skeleton("b")],
                         labels={"a": "Tm2", "b": "Tm2"}), "Line", {"LinesDrawable"}),
])
def test_a_group_glyph_reflects_its_members(has_gpu, make, glyph, kinds):
    view = backend.show(make(), size=(400, 300), canvas="offscreen")
    try:
        assert type(view.legend["Tm2"].glyph).__name__ == glyph
        assert view.legend["Tm2"].kinds == kinds
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
    """All, some or none of the members visible, each drawn differently."""
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
    """Anything showing means hide; members are not each inverted."""
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
    """Drawables pair with world objects by position, since a labelled one may have no name."""
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
    """Measured from pygfx's text layout, so it is right on the first frame."""
    short = backend.show(_scene(1), size=(600, 400), canvas="offscreen")
    scene = Scene().add_mesh(_mesh("a name very much longer than the other one"))
    long = backend.show(scene, size=(600, 400), canvas="offscreen")
    try:
        assert long.legend.width > short.legend.width
    finally:
        short.close()
        long.close()


def test_many_entries_wrap_into_columns_rather_than_shrinking_to_nothing(has_gpu):
    """One column of 30 rows would have to shrink to unreadable text."""
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
    """Both rects come from the renderer's current size, so hit-testing matches the drawing."""
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
    """A translucent panel would composite against the notebook page."""
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
    """`snapshot(size=...)` uses its own renderer, which must draw the legend too."""
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
    """Only the plate writes a pick id, so a click anywhere on the row counts."""
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
    """The camera controller is registered on the main viewport only."""
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
    """Left hides, right highlights, nothing else is claimed."""
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
    """pygfx's `get_bounding_box` ignores `visible`, so hidden objects must be excluded."""
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
    """A highlight overrides the material only; the scene's colour is never changed."""
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


def test_several_rows_can_be_lit_and_an_exclusive_highlight_drops_the_others(has_gpu):
    view = backend.show(_scene(3), size=(400, 300), canvas="offscreen")
    try:
        view.legend.highlight("body 0", "body 2")
        assert view.legend.highlighted == ["body 0", "body 2"]
        view.legend.highlight("body 1", exclusive=True)
        assert view.legend.highlighted == ["body 1"]
        view.legend.clear_highlights()
        assert view.legend.highlighted == []
    finally:
        view.close()


def test_the_row_says_it_is_highlighted_and_not_only_the_swatch(has_gpu):
    """Otherwise a lit row and a body that really is white would look the same."""
    view = backend.show(_scene(2), size=(400, 300), canvas="offscreen")
    try:
        plain = tuple(view.legend["body 0"].plate.material.color)
        view.legend.highlight("body 0")
        assert tuple(view.legend["body 0"].plate.material.color) != plain
        assert tuple(view.legend["body 1"].plate.material.color) == plain
    finally:
        view.close()


def test_recolouring_a_highlighted_entry_takes_effect_when_it_is_dropped(has_gpu):
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
    view = backend.show(_scene(2), size=(400, 300), canvas="offscreen")
    try:
        view.legend.highlight("body 0").set_visible("body 0", False)
        assert view.group.children[0].visible is False
        view.legend.set_visible("body 0", True)
        assert view.legend.highlighted == ["body 0"]
    finally:
        view.close()


def test_a_highlight_survives_a_relabel_and_a_relayout(has_gpu):
    """Relabelling rebuilds the rows; the highlight is carried across by label."""
    view = backend.show(_scene(3), size=(400, 300), canvas="offscreen")
    try:
        view.legend.highlight("body 1")
        view.legend.relabel("body 1", "a much longer label than before")
        assert view.legend.highlighted == ["a much longer label than before"]
    finally:
        view.close()


def test_an_explicit_highlight_colour_is_honoured(has_gpu):
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
    """A row's text is a label; `scene.get` keeps answering to the original names."""
    view = backend.show(_scene(2), size=(400, 300), canvas="offscreen")
    try:
        view.legend.relabel("body 0", "MeCN-01 (L)")
        assert view.legend.labels == ["MeCN-01 (L)", "body 1"]
        assert view.scene_data.names == ["body 0", "body 1"]
        assert view.scene_data.get("body 0").label == "MeCN-01 (L)"
        assert view.legend["MeCN-01 (L)"].label_width > 0
    finally:
        view.close()


@pytest.mark.parametrize("mapping, labels", [
    ({"body 0": "Tm2", "body 2": "LC6"}, ["Tm2", "body 1", "LC6"]),
    ({"body 0": "Tm2", "body 1": "Tm2"}, ["Tm2", "body 2"]),   # merges two rows
])
def test_relabelling_takes_a_mapping(has_gpu, mapping, labels):
    view = backend.show(_scene(3), size=(600, 400), canvas="offscreen")
    try:
        view.legend.relabel(mapping)
        assert view.legend.labels == labels
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
    """`refresh` rebuilds the rows, since a row's text is baked into a `pygfx.Text`."""
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
    """`Scene.relabel` requests a repaint, and the next frame catches up."""
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
    """A plain attribute set cannot notify, so the legend re-reads the scene each frame."""
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
    """A state change reassigns materials; only a structural change rebuilds rows."""
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
    """`sync` runs from `draw`, so a saved figure cannot lag the scene."""
    view = backend.show(_scene(2), size=(400, 300), canvas="offscreen", pixel_ratio=1.0)
    try:
        before = view.snapshot().copy()
        view.scene_data.relabel({"body 0": "a very much longer label indeed"})
        assert not np.array_equal(view.snapshot(), before)
        assert view.legend.labels == ["a very much longer label indeed", "body 1"]
    finally:
        view.close()


def test_view_refresh_catches_up_on_everything_at_once(has_gpu):
    """Including drawables that have no legend row."""
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
    """A scene outlives its views, and keeps their listeners."""
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
    """Unlike `Scene.recolor`, which re-assigns the whole palette."""
    view = backend.show(_scene(3), size=(400, 300), canvas="offscreen")
    try:
        others = [view.legend[n].drawables[0].color for n in ("body 1", "body 2")]
        view.legend.recolor("body 0", "w")
        assert [view.legend[n].drawables[0].color for n in ("body 1", "body 2")] == others
    finally:
        view.close()


def test_the_glyph_is_opaque_whatever_the_drawables_alpha(has_gpu):
    scene = Scene().add_mesh(_mesh("faint"), alpha=0.15)
    view = backend.show(scene, size=(400, 300), canvas="offscreen")
    try:
        assert tuple(view.legend["faint"].glyph.material.color)[3] == 1.0
    finally:
        view.close()

