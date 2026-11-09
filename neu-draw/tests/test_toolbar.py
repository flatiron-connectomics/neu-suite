"""The notebook toolbar: its widgets, and what each button does.

Uses a real `rendercanvas.jupyter` canvas, which constructs without a browser. Button
handlers are called directly, since `Button.on_click` dispatch needs a front end.
"""

import numpy as np
import pytest

pytest.importorskip("pygfx", reason="the render extra is not installed")
widgets = pytest.importorskip("ipywidgets")

from neu_lib import Mesh                                        # noqa: E402
from neu_draw import toolbar as toolbar_mod, viewstate          # noqa: E402
from neu_draw.backends import pygfx as backend                  # noqa: E402
from neu_draw.scene import Scene                                # noqa: E402


@pytest.fixture
def has_gpu():
    import wgpu
    try:
        wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    except Exception as exc:                                    # pragma: no cover
        pytest.skip(f"no wgpu adapter available: {exc}")
    return True


@pytest.fixture(autouse=True)
def clean_globals():
    viewstate.views.clear()
    toolbar_mod.last_prefix, toolbar_mod.last_suffix = None, ".png"
    toolbar_mod.last_view_file = "view.json"
    yield
    viewstate.views.clear()
    toolbar_mod.last_prefix, toolbar_mod.last_suffix = None, ".png"
    toolbar_mod.last_view_file = "view.json"


def _mesh(name="body"):
    verts = np.array([[0.0, 0, 0], [100.0, 0, 0], [0.0, 100.0, 0], [0.0, 0, 100.0]])
    return Mesh(verts, np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]), name=name)


@pytest.fixture
def view(has_gpu):
    view = backend.show(Scene().add_mesh(_mesh()), size=(120, 90), canvas="jupyter")
    yield view
    view.close()


# --------------------------------------------------------------------------- #
# naming a capture
# --------------------------------------------------------------------------- #

def test_the_default_capture_name_carries_a_sortable_timestamp():
    path = toolbar_mod.default_path()
    assert path.startswith("snapshot_") and path.endswith(".png")


def test_the_remembered_prefix_does_not_accumulate_timestamps():
    """Otherwise each capture would add another timestamp to the next default name."""
    first = toolbar_mod.default_path("cell")
    toolbar_mod.remember_prefix(first)
    assert toolbar_mod.last_prefix == "cell"
    assert toolbar_mod.default_path().count("_") == first.count("_")


@pytest.mark.parametrize("path, prefix", [
    ("figures/left_optic_lobe_2026-08-26_11-00-00.png", "left_optic_lobe"),
    ("/tmp/final_figure.png", "final_figure"),          # not timestamped: kept whole
])
def test_the_remembered_prefix_strips_only_the_timestamp(path, prefix):
    toolbar_mod.remember_prefix(path)
    assert toolbar_mod.last_prefix == prefix


# --------------------------------------------------------------------------- #
# attachment
# --------------------------------------------------------------------------- #

def test_a_notebook_canvas_gets_a_toolbar_with_no_one_asking(view):
    """`toolbar="auto"` is the default."""
    assert isinstance(view.ui, toolbar_mod.Toolbar)
    bar, entry, panels, opacity, stage = view.ui.widget.children
    assert panels.children == (view.ui.options["panel"], view.ui.animation["panel"],
                               view.ui.view_file["panel"], view.ui.legend_row["panel"])
    assert [b.description for b in bar.children] == [
        "Center", "Reset", "Refresh", "Save", "Restore", "Last", "Capture", "Animate",
        "Close"]
    assert entry.children == (view.ui.path, view.ui.status)
    assert opacity.children == (view.ui.sliders["mesh"],)
    assert stage.children == (view.canvas,)


def test_an_offscreen_view_quietly_has_no_toolbar(has_gpu):
    """The headless path must not raise."""
    view = backend.show(Scene(), size=(32, 32), canvas="offscreen")
    try:
        assert view.ui is None
    finally:
        view.close()


def test_insisting_on_a_toolbar_off_a_notebook_says_why(has_gpu):
    with pytest.raises(TypeError, match="ipywidget"):
        backend.show(Scene(), size=(32, 32), canvas="offscreen", toolbar=True)


def test_the_toolbar_can_be_declined(has_gpu):
    view = backend.show(Scene(), size=(32, 32), canvas="jupyter", toolbar=False)
    try:
        assert view.ui is None
        assert view._repr_mimebundle_() == view.canvas._repr_mimebundle_()
    finally:
        view.close()


def test_the_mimebundle_shows_the_toolbar_when_there_is_one(view):
    bundle = view._repr_mimebundle_(include=None, exclude=None)
    assert "application/vnd.jupyter.widget-view+json" in bundle


# --------------------------------------------------------------------------- #
# the buttons
# --------------------------------------------------------------------------- #

def test_save_and_restore_report_through_the_status_line(view):
    view.ui._restore_view()
    assert "nothing saved" in view.ui.status.value

    view.ui._save_view()
    assert "saved" in view.ui.status.value and viewstate.SAVED in viewstate.views

    view.ui._restore_view()
    assert "restored" in view.ui.status.value


def test_last_and_restore_are_separate_buttons_on_separate_slots(view):
    """"The angle I saved" and "where the last figure was closed" are different slots."""
    view.ui._save_view()
    view.ui._restore_last()
    assert "no closed figure" in view.ui.status.value

    viewstate.views[viewstate.LAST] = view.save_view("scratch")
    view.ui._restore_last()
    assert "last closed figure" in view.ui.status.value


def test_reset_reports_that_it_left_colours_alone(view):
    view.ui._reset()
    assert "opening view" in view.ui.status.value
    assert "colours" in view.ui.status.value


def test_capture_writes_the_named_file_and_reseeds_the_box(view, tmp_path):
    pytest.importorskip("imageio")
    view.ui.path.value = str(tmp_path / "left_lobe.png")
    view.ui._capture()

    assert (tmp_path / "left_lobe.png").exists()
    assert "wrote" in view.ui.status.value
    assert view.ui.path.value.startswith("left_lobe_")


def test_capturing_an_svg_writes_one_and_keeps_the_next_default_an_svg(view, tmp_path):
    pytest.importorskip("imageio")
    view.ui.path.value = str(tmp_path / "fig.svg")
    view.ui._capture()
    assert "<svg" in (tmp_path / "fig.svg").read_text()[:200]
    assert view.ui.path.value.startswith("fig_") and view.ui.path.value.endswith(".svg")


def test_the_format_dropdown_and_the_paths_extension_follow_each_other(view):
    view.ui.path.value = "figs/fig.png"
    view.ui.options["format"].value = "pdf"
    assert view.ui.path.value == "figs/fig.pdf"
    view.ui.path.value = "figs/other.svg"
    assert view.ui.options["format"].value == "svg"


def test_the_layers_dropdown_is_only_live_for_a_vector_format(view):
    view.ui.path.value = "fig.png"
    assert view.ui.options["layers"].disabled
    view.ui.options["format"].value = "svg"
    assert not view.ui.options["layers"].disabled


def test_capture_nests_layers_as_the_dropdown_says(view, tmp_path):
    pytest.importorskip("imageio")
    view.scene_data.export_groups = lambda d: ("cell",)
    view.ui.options["layers"].value = "group"
    view.ui.path.value = str(tmp_path / "fig.svg")
    view.ui._capture()
    text = (tmp_path / "fig.svg").read_text()
    assert text.index('inkscape:label="cell"') < text.index('inkscape:label="meshes"')


# --------------------------------------------------------------------------- #
# view files
# --------------------------------------------------------------------------- #

def test_a_view_file_round_trips_through_the_buttons(view, tmp_path):
    path = tmp_path / "views" / "angle.json"
    view.ui.view_file["path"].value = str(path)
    view.camera.local.position = (11.0, 22.0, 33.0)
    view.ui._save_view_file()
    assert path.is_file() and "written" in view.ui.status.value

    view.camera.local.position = (900.0, 900.0, 900.0)
    viewstate.views.clear()
    view.ui._load_view_file()
    assert np.allclose(view.camera.local.position, (11.0, 22.0, 33.0))
    assert viewstate.SAVED in viewstate.views and "loaded" in view.ui.status.value


def test_saving_a_view_file_refuses_to_overwrite_unless_asked(view, tmp_path):
    path = tmp_path / "angle.json"
    path.write_text("keep me")
    view.ui.view_file["path"].value = str(path)
    view.ui._save_view_file()
    assert "overwrite" in view.ui.status.value and path.read_text() == "keep me"
    assert viewstate.SAVED not in viewstate.views         # the slot is left alone too

    view.ui.view_file["overwrite"].value = True
    view.ui._save_view_file()
    assert path.read_text() != "keep me"


def test_the_next_toolbar_offers_the_last_view_file(view, has_gpu, tmp_path):
    view.ui.view_file["path"].value = str(tmp_path / "angle.json")
    view.ui._save_view_file()
    second = backend.show(Scene().add_mesh(_mesh()), size=(120, 90), canvas="jupyter")
    try:
        assert second.ui.view_file["path"].value == str(tmp_path / "angle.json")
    finally:
        second.close()


def test_loading_a_missing_view_file_reports_it(view, tmp_path):
    view.ui.view_file["path"].value = str(tmp_path / "nothing.json")
    view.ui._guarded(view.ui._load_view_file)
    assert "FileNotFoundError" in view.ui.status.value


def test_capture_honours_the_options(view, tmp_path):
    pytest.importorskip("imageio")
    from imageio import v3 as iio

    opts = view.ui.options
    opts["width"].value, opts["height"].value, opts["resolution"].value = 60, 40, 1.0
    view.ui.path.value = str(tmp_path / "small.png")
    view.ui._capture()
    assert iio.imread(tmp_path / "small.png").shape[:2] == (40, 60)

    view.ui.path.value = str(tmp_path / "fig.pdf")
    view.ui._capture()
    assert (tmp_path / "fig.pdf").read_bytes().startswith(b"%PDF-")
    assert view.ui.path.value.endswith(".pdf")


def test_animate_renders_the_options_and_reports_the_encode_line(view, tmp_path):
    """Frames only, so the line to run is what the status shows."""
    pytest.importorskip("imageio")
    a, c = view.ui.animation, view.ui.options
    a["seconds"].value, a["fps"].value, a["degrees"].value = 0.5, 8, 90.0
    c["width"].value, c["height"].value = 60, 40
    a["folder"].value = str(tmp_path / "spin")
    a["encode"].value = False              # hermetic: never an ffmpeg this machine has
    view.ui._animate()
    assert len(list((tmp_path / "spin").glob("frame_*.png"))) == 4
    assert "ffmpeg" in view.ui.status.value and "yuv420p" in view.ui.status.value


def test_animate_encodes_with_the_ffmpeg_it_finds(view, tmp_path, monkeypatch):
    import stat
    import sys

    fake = tmp_path / "ffmpeg"
    fake.write_text(f"#!{sys.executable}\nimport sys\nopen(sys.argv[-1], 'wb').write(b'x')\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("NEU_DRAW_FFMPEG", str(fake))
    a = view.ui.animation
    a["seconds"].value, a["fps"].value = 0.25, 8
    a["folder"].value, a["video"].value = str(tmp_path / "spin"), str(tmp_path / "t4.mp4")
    view.ui._animate()
    assert (tmp_path / "t4.mp4").exists() and "wrote" in view.ui.status.value


def test_the_frames_folder_defaults_beside_the_capture(view):
    view.ui.path.value = "figs/t4.svg"
    assert view.ui._frames_folder() == "figs/t4_orbit"


def test_an_empty_path_is_refused_rather_than_guessed_at(view):
    view.ui.path.value = "   "
    view.ui._capture()
    assert "type a filename" in view.ui.status.value


def test_a_failing_button_reports_instead_of_vanishing(view):
    """A callback's traceback goes to the kernel log, so errors are shown in the status line."""
    view.ui._guarded(lambda: 1 / 0)
    assert "ZeroDivisionError" in view.ui.status.value


def test_closing_leaves_the_last_image_where_the_canvas_was(view):
    pytest.importorskip("imageio")
    view.ui._close()

    (still,) = view.ui._stage.children
    assert isinstance(still, widgets.Image) and still.value[:4] == b"\x89PNG"
    assert still.layout.width == "120px"       # scaled back from the supersampled shot
    assert all(button.disabled for button in view.ui._buttons.values())
    assert all(s.disabled for s in view.ui.sliders.values())


def test_closing_records_the_viewpoint_it_is_about_to_lose(view):
    """So the next figure can reopen there, via 'Last' or `viewpoint="last"`."""
    pytest.importorskip("imageio")
    view.ui._close()
    assert viewstate.LAST in viewstate.views
    assert "Last" in view.ui.status.value and "viewpoint=" in view.ui.status.value


# --------------------------------------------------------------------------- #
# the opacity sliders
# --------------------------------------------------------------------------- #

def test_there_is_one_slider_per_kind_the_scene_holds(has_gpu):
    """Per kind, not per row."""
    from neu_lib import Skeleton

    skel = Skeleton(np.array([[0.0, 0, 0], [50.0, 50, 50]]), np.array([[0, 1]]),
                    name="s")
    scene = Scene().add_mesh(_mesh()).add_skeleton(skel)
    view = backend.show(scene, size=(120, 90), canvas="jupyter")
    try:
        assert list(view.ui.sliders) == ["mesh", "skeleton"]
    finally:
        view.close()


def test_a_slider_starts_at_the_largest_alpha_and_moving_it_sets_them_all(has_gpu):
    scene = Scene().add_mesh(_mesh("a"), alpha=0.3).add_mesh(_mesh("b"), alpha=0.7)
    view = backend.show(scene, size=(120, 90), canvas="jupyter")
    try:
        slider = view.ui.sliders["mesh"]
        assert slider.value == pytest.approx(0.7)
        slider.value = 0.2
        assert [d.alpha for d in scene] == [pytest.approx(0.2)] * 2
        assert [tuple(o.material.color)[3] for o in view.group.children] == [
            pytest.approx(0.2)] * 2
    finally:
        view.close()


# --------------------------------------------------------------------------- #
# the Legend row panel
# --------------------------------------------------------------------------- #

@pytest.fixture
def two_rows(has_gpu):
    """Rows "a" (one mesh) and "pair": b and d green (the swatch) at opacities 0.2 and
    0.6, c blue at 0.8."""
    scene = (Scene().add_mesh(_mesh("a"), color="#ff0000", alpha=0.5)
             .add_mesh(_mesh("b"), color="#00ff00", alpha=0.2)
             .add_mesh(_mesh("c"), color="#0000ff", alpha=0.8)
             .add_mesh(_mesh("d"), color="#00ff00", alpha=0.6))
    for name in "bcd":
        scene.get(name).label = "pair"
    view = backend.show(scene, size=(200, 120), canvas="jupyter")
    yield view
    view.close()


def test_the_legend_row_controls_wait_for_a_pick(two_rows):
    c = two_rows.ui.legend_row
    assert c["color"].disabled and c["alpha"].disabled
    assert "Shift+click" in c["row"].value


def test_picking_a_row_loads_its_colour_and_opacity_and_opens_the_panel(two_rows):
    ui = two_rows.ui
    ui._pick_row("a")
    c = ui.legend_row
    assert c["color"].value == "#ff0000" and c["alpha"].value == pytest.approx(0.5)
    assert not c["color"].disabled and not c["alpha"].disabled
    assert c["panel"].selected_index == 0 and "a" in c["row"].value
    # loading the values changed nothing
    assert two_rows.scene_data.get("a").alpha == pytest.approx(0.5)
    assert "#b00" not in ui.status.value


def test_the_controls_recolour_the_picked_row_and_say_how_to_keep_it(two_rows):
    ui = two_rows.ui
    ui._pick_row("a")
    ui.legend_row["color"].value = "#336699"
    assert two_rows.scene_data.get("a").color[:3] == pytest.approx(
        (0x33 / 255, 0x66 / 255, 0x99 / 255))
    assert "view.legend.recolor(&#x27;a&#x27;, &#x27;#336699&#x27;)" in ui.status.value

    ui.legend_row["alpha"].value = 0.3
    assert two_rows.scene_data.get("a").alpha == pytest.approx(0.3)
    assert "view.legend.set_alpha(&#x27;a&#x27;, 0.30)" in ui.status.value
    assert two_rows.scene_data.get("b").alpha == pytest.approx(0.2)   # other rows alone


def test_a_single_colour_row_offers_no_matching_checkbox(two_rows):
    two_rows.ui._pick_row("a")
    m = two_rows.ui.legend_row["matching"]
    assert m.disabled and not m.value


def test_a_mixed_row_changes_only_the_swatch_coloured_members_by_default(two_rows):
    ui, get = two_rows.ui, two_rows.scene_data.get
    ui._pick_row("pair")
    assert ui.legend_row["matching"].value and not ui.legend_row["matching"].disabled
    assert "2 of 3" in ui.legend_row["row"].value
    # b and d still differ in opacity, so that is warned of; colours are not
    assert "0.2–0.6" in ui.status.value and "colours" not in ui.status.value
    assert ui.legend_row["alpha"].value == pytest.approx(0.6)

    ui.legend_row["color"].value = "#336699"
    ui.legend_row["color"].value = "#ffcc00"     # the same members, though now recoloured
    assert get("b").color == get("d").color != get("c").color
    assert get("b").color[:3] == pytest.approx((1.0, 0xcc / 255, 0.0))
    assert "only=&#x27;#00ff00&#x27;" in ui.status.value

    ui.legend_row["alpha"].value = 0.4
    assert [get(n).alpha for n in "bcd"] == [pytest.approx(0.4), pytest.approx(0.8),
                                              pytest.approx(0.4)]


def test_unticking_matching_acts_on_every_member_with_a_warning(two_rows):
    ui, get = two_rows.ui, two_rows.scene_data.get
    ui._pick_row("pair")
    ui.legend_row["matching"].value = False
    assert "2 colours" in ui.status.value and "0.2–0.8" in ui.status.value
    assert "3 drawables" in ui.legend_row["row"].value
    assert [get(n).alpha for n in "bcd"] == [pytest.approx(0.2), pytest.approx(0.8),
                                              pytest.approx(0.6)]   # nothing yet

    ui.legend_row["color"].value = "#ffffff"
    assert get("b").color == get("c").color == get("d").color
    assert "only=" not in ui.status.value


def test_the_toolbar_listens_for_the_legends_pick(two_rows):
    """The Shift+click itself is tested in test_legend; a notebook canvas is 1x1 until a
    browser lays it out, so here the legend's pick is fired directly."""
    for handler in two_rows.legend._pick_handlers:
        handler("pair")
    assert two_rows.ui._row == "pair"
    assert two_rows.ui.legend_row["panel"].selected_index == 0


def test_every_swatch_is_a_colour_neu_draw_understands():
    from neu_draw.colors import SWATCHES, to_rgba
    for values in SWATCHES.values():
        for value in values:
            to_rgba(value)


def test_swatches_wait_for_a_pick_then_recolour_by_name(two_rows):
    ui, c = two_rows.ui, two_rows.ui.legend_row
    assert c["palette"].value == "tab10"
    assert all(b.disabled for b in c["swatches"].children)

    ui._pick_row("pair")                                # matching ticked: b and d
    assert not any(b.disabled for b in c["swatches"].children)
    orange = next(b for b in c["swatches"].children if b.tooltip == "tab:orange")
    orange.click()
    get = two_rows.scene_data.get
    assert get("b").color[:3] == get("d").color[:3] == pytest.approx(
        (1.0, 0.498, 0.0549), abs=1e-3)
    assert get("c").color[:3] == (0.0, 0.0, 1.0)
    assert c["color"].value == "#ff7f0e"
    # the paste line names the swatch, not the picker's hex
    assert "&#x27;tab:orange&#x27;, only=&#x27;#00ff00&#x27;" in ui.status.value


def test_choosing_another_swatch_set_shows_its_colours(two_rows):
    from neu_draw.colors import SWATCHES
    c = two_rows.ui.legend_row
    two_rows.ui._pick_row("a")
    c["palette"].value = "Okabe-Ito"
    assert [b.tooltip for b in c["swatches"].children] == list(SWATCHES["Okabe-Ito"])
    assert not any(b.disabled for b in c["swatches"].children)


def test_closing_disables_the_legend_row_and_view_file_controls(two_rows):
    pytest.importorskip("imageio")
    two_rows.ui._pick_row("a")
    two_rows.ui._close()
    assert two_rows.ui.legend_row["color"].disabled
    assert two_rows.ui.legend_row["alpha"].disabled
    assert all(b.disabled for b in two_rows.ui.legend_row["swatches"].children)
    assert two_rows.ui.view_file["save"].disabled and two_rows.ui.view_file["load"].disabled


def test_a_view_without_a_legend_has_no_legend_row_panel(has_gpu):
    view = backend.show(Scene().add_mesh(_mesh()), size=(120, 90), canvas="jupyter",
                        legend=False)
    try:
        assert view.ui.legend_row == {}
        _, _, panels, *_ = view.ui.widget.children
        assert len(panels.children) == 3
    finally:
        view.close()


def test_closing_twice_is_harmless(view):
    pytest.importorskip("imageio")
    view.ui._close()
    view.ui._close()
    assert "already closed" in view.ui.status.value
