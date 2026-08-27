"""The notebook toolbar: what it is made of, and what each button does.

Driven against a **real** Jupyter canvas rather than a stub. `rendercanvas.jupyter`
constructs perfectly well outside a notebook — it is an ipywidget either way, and the
browser is only needed to *paint* it — so the widget tree, the button handlers and the
close-and-leave-the-image swap are all exercisable here. The handlers are called
directly; `Button.on_click` is ipywidgets' own dispatch and needs a front end.
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
    toolbar_mod.last_prefix = None
    yield
    viewstate.views.clear()
    toolbar_mod.last_prefix = None


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
    """The failure this exists to prevent: capture once, and the next default would be
    `cell_<ts>_<ts>.png`, then three, then four."""
    first = toolbar_mod.default_path("cell")
    toolbar_mod.remember_prefix(first)
    assert toolbar_mod.last_prefix == "cell"
    assert toolbar_mod.default_path().count("_") == first.count("_")


def test_a_prefix_with_underscores_of_its_own_survives():
    toolbar_mod.remember_prefix("figures/left_optic_lobe_2026-08-26_11-00-00.png")
    assert toolbar_mod.last_prefix == "left_optic_lobe"


def test_a_name_that_is_not_timestamped_is_kept_whole():
    toolbar_mod.remember_prefix("/tmp/final_figure.png")
    assert toolbar_mod.last_prefix == "final_figure"


# --------------------------------------------------------------------------- #
# attachment
# --------------------------------------------------------------------------- #

def test_a_notebook_canvas_gets_a_toolbar_with_no_one_asking(view):
    """`toolbar="auto"` is the default, for the same reason the store-log filter is
    automatic: a feature you have to remember to switch on looks identical to a broken
    one after a kernel restart."""
    assert isinstance(view.ui, toolbar_mod.Toolbar)
    bar, entry, stage = view.ui.widget.children
    assert [b.description for b in bar.children] == [
        "Center", "Reset", "Save", "Restore", "Last", "Capture", "Close"]
    assert entry.children == (view.ui.path, view.ui.status)
    assert stage.children == (view.canvas,)


def test_an_offscreen_view_quietly_has_no_toolbar(has_gpu):
    """Nothing to sit above, and nothing that wanted buttons. It must not raise: this is
    the path every test and every headless render takes."""
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
    """What Jupyter calls to display the cell. Wrapping the canvas must not change how a
    view is shown, only what it is shown inside."""
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
    """They answer different questions — "the angle I chose" against "wherever I happened
    to be" — and one button choosing between them would leave it unclear which you got."""
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


def test_an_empty_path_is_refused_rather_than_guessed_at(view):
    view.ui.path.value = "   "
    view.ui._capture()
    assert "type a filename" in view.ui.status.value


def test_a_failing_button_reports_instead_of_vanishing(view):
    """A button callback's traceback goes to the kernel log, not the cell, so an
    unguarded failure is a control that visibly does nothing."""
    view.ui._guarded(lambda: 1 / 0)
    assert "ZeroDivisionError" in view.ui.status.value


def test_closing_leaves_the_last_image_where_the_canvas_was(view):
    pytest.importorskip("imageio")
    view.ui._close()

    (still,) = view.ui._stage.children
    assert isinstance(still, widgets.Image) and still.value[:4] == b"\x89PNG"
    assert still.layout.width == "120px"       # scaled back from the supersampled shot
    assert all(button.disabled for button in view.ui._buttons.values())


def test_closing_records_the_viewpoint_it_is_about_to_lose(view):
    """The reason to press it rather than delete the cell: the angle survives, so the
    next figure can open where this one left off — via 'Last' or `viewpoint="last"`, both
    of which the status line names."""
    pytest.importorskip("imageio")
    view.ui._close()
    assert viewstate.LAST in viewstate.views
    assert "Last" in view.ui.status.value and "viewpoint=" in view.ui.status.value


def test_closing_twice_is_harmless(view):
    pytest.importorskip("imageio")
    view.ui._close()
    view.ui._close()
    assert "already closed" in view.ui.status.value
