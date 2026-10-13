"""Saved viewpoints, and the actions the toolbar's buttons are.

The store is pure and tests without a GPU. The four camera actions need a real view, so
they sit behind the same adapter skip as the rest of the render tests.
"""

import numpy as np
import pytest

pygfx = pytest.importorskip("pygfx", reason="the render extra is not installed")

from neu_lib import Mesh                                        # noqa: E402
from neu_draw import viewstate                                  # noqa: E402
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
def empty_store():
    """The store is session-wide by design, so a test must not inherit another's slots."""
    viewstate.views.clear()
    yield
    viewstate.views.clear()


def _mesh(name="body"):
    verts = np.array([[0.0, 0, 0], [100.0, 0, 0], [0.0, 100.0, 0], [0.0, 0, 100.0]])
    return Mesh(verts, np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]), name=name)


def _view(**kwargs):
    kwargs.setdefault("canvas", "offscreen")
    kwargs.setdefault("size", (80, 60))
    kwargs.setdefault("pixel_ratio", 1.0)
    return backend.show(Scene().add_mesh(_mesh()), **kwargs)


# --------------------------------------------------------------------------- #
# the store
# --------------------------------------------------------------------------- #

def test_a_saved_viewpoint_outlives_the_view_that_saved_it(has_gpu):
    """The whole reason the store is a module-level dict: you save an angle in order to
    use it in the *next* figure, by which time the first one is closed."""
    first = _view()
    saved = first.save_view()
    first.close()

    second = _view()
    try:
        assert second.restore_view() == saved
    finally:
        second.close()


def test_restoring_an_empty_slot_returns_none_rather_than_raising(has_gpu):
    """A fresh session has nothing saved, and that is the state a button is most likely
    pressed in. Raising there would put a traceback where the figure is."""
    view = _view()
    try:
        assert view.restore_view() is None
        assert view.restore_view("no-such-slot") is None
    finally:
        view.close()


def test_a_viewpoint_records_the_size_it_was_framed_for(has_gpu):
    """A perspective camera's horizontal field of view follows the rect's aspect, so the
    same camera state in a differently shaped canvas is a different picture."""
    view = _view(size=(200, 100))
    try:
        assert view.save_view().size == (200, 100)
    finally:
        view.close()


def test_slots_are_independent(has_gpu):
    view = _view()
    try:
        first = view.save_view("a")
        view.camera.local.position = (5.0, 6.0, 7.0)
        second = view.save_view("b")
        assert first.camera["position"].tolist() != second.camera["position"].tolist()
        assert set(viewstate.views) == {"a", "b"}
    finally:
        view.close()


# --------------------------------------------------------------------------- #
# the camera actions
# --------------------------------------------------------------------------- #

def test_save_then_restore_puts_the_camera_back(has_gpu):
    view = _view()
    try:
        view.save_view()
        before = np.array(view.camera.local.position, dtype=float)

        view.camera.local.position = tuple(before + 10_000.0)
        assert not np.allclose(view.camera.local.position, before)

        view.restore_view()
        assert np.allclose(view.camera.local.position, before)
    finally:
        view.close()


def test_restoring_can_leave_the_canvas_size_alone(has_gpu):
    """``size=False`` is for when you want the angle in the canvas you already have."""
    view = _view(size=(120, 90))
    try:
        view.save_view()
        view.canvas.set_logical_size(200, 200)
        view.restore_view(size=False)
        assert tuple(int(v) for v in view.canvas.get_logical_size()) == (200, 200)
        view.restore_view()
        assert tuple(int(v) for v in view.canvas.get_logical_size()) == (120, 90)
    finally:
        view.close()


def test_closing_records_where_the_camera_was(has_gpu):
    """In `View.close`, not in the toolbar, so a `view.close()` from a cell counts too —
    the slot is only useful if it is reliably populated, and "I closed the figure and want
    that angle back" should not depend on which route closed it."""
    view = _view()
    view.camera.local.position = (1234.0, 5678.0, 9012.0)
    view.close()

    assert viewstate.LAST in viewstate.views
    assert viewstate.views[viewstate.LAST].camera["position"].tolist() == [
        1234.0, 5678.0, 9012.0]


def test_the_last_slot_is_what_the_next_figure_reopens_on(has_gpu):
    """The gap this closes: `Close` wrote `views["last"]` and nothing consumed it."""
    first = _view()
    first.camera.local.position = (4321.0, 8765.0, 2109.0)
    first.close()

    second = _view(viewpoint=viewstate.LAST)
    try:
        assert np.allclose(second.camera.local.position, (4321.0, 8765.0, 2109.0))
    finally:
        second.close()


def test_an_empty_viewpoint_slot_leaves_the_figure_its_own_framing(has_gpu):
    """So `show(scene, viewpoint="last")` is safe in the first cell of a session."""
    framed = _view()
    expected = np.array(framed.camera.local.position, dtype=float)
    framed.close()
    viewstate.views.clear()

    view = _view(viewpoint="never-saved")
    try:
        assert np.allclose(view.camera.local.position, expected)
    finally:
        view.close()


def test_a_viewpoint_can_be_a_state_rather_than_a_slot_name(has_gpu):
    """So an angle can live in an ordinary variable."""
    first = _view()
    first.camera.local.position = (7.0, 8.0, 9.0)
    state = first.save_view("tmp")
    first.close()

    second = _view(viewpoint=state)
    try:
        assert np.allclose(second.camera.local.position, (7.0, 8.0, 9.0))
    finally:
        second.close()


# --------------------------------------------------------------------------- #
# reset
# --------------------------------------------------------------------------- #

def test_reset_goes_back_to_the_opening_view(has_gpu):
    view = _view()
    try:
        opening = np.array(view.camera.local.position, dtype=float)
        view.camera.local.position = tuple(opening + 5_000.0)
        view.reset()
        assert np.allclose(view.camera.local.position, opening)
    finally:
        view.close()


def test_reset_unhides_and_unhighlights(has_gpu):
    view = _view()
    try:
        view.legend.set_visible("body", False)
        view.legend.highlight("body")
        view.reset()

        assert view.scene_data.get("body").visible is True
        assert view.group.children[0].visible is True
        assert view.legend.highlighted == []
    finally:
        view.close()


def test_reset_respects_a_drawable_that_was_hidden_from_the_start(has_gpu):
    """"Everything visible" is not the same as "how this opened": a scene may deliberately
    arrive with something switched off, and reset must not turn it on."""
    scene = Scene().add_mesh(_mesh("shown")).add_mesh(_mesh("hidden"))
    scene.get("hidden").visible = False
    view = backend.show(scene, size=(80, 60), canvas="offscreen")
    try:
        view.legend.set_visible("hidden", True)
        view.reset()
        assert view.scene_data.get("hidden").visible is False
        assert view.group.children[1].visible is False
    finally:
        view.close()


def test_reset_leaves_colours_alone(has_gpu):
    """Visibility and highlights are transient exploration; a recolour is an authored
    change to the scene, and reverting it would be destroying work rather than tidying."""
    view = _view()
    try:
        view.legend.recolor("body", "tab:pink")
        view.reset()
        assert view.scene_data.get("body").color == pytest.approx(
            (0.8902, 0.4667, 0.7608, 1.0), abs=1e-3)
    finally:
        view.close()


def test_reset_returns_to_a_restored_viewpoint_not_to_a_fit(has_gpu):
    """The opening view is recorded AFTER `viewpoint=` is applied, so "reset" means what
    this figure opened showing rather than what it would have opened showing."""
    first = _view()
    first.camera.local.position = (1.0, 2.0, 3.0)
    state = first.save_view("angle")
    first.close()

    view = _view(viewpoint=state)
    try:
        view.camera.local.position = (900.0, 900.0, 900.0)
        view.reset()
        assert np.allclose(view.camera.local.position, (1.0, 2.0, 3.0))
    finally:
        view.close()


def test_center_refits_a_camera_that_has_been_dragged_away(has_gpu):
    far = _mesh()
    far.vertices_zyx_nm = far.vertices_zyx_nm + 50_000.0
    view = backend.show(Scene().add_mesh(far), size=(80, 80), canvas="offscreen")
    try:
        framed = np.array(view.camera.local.position, dtype=float)
        view.camera.local.position = (0.0, 0.0, 0.0)
        view.center()
        assert np.allclose(view.camera.local.position, framed)
    finally:
        view.close()


def test_logical_size_does_not_believe_a_browsers_placeholder(has_gpu):
    """A Jupyter canvas reports (1, 1) until the widget has been laid out, which is the
    same trap that made `snapshot()` write a 2x2 PNG. A viewpoint saved in a freshly
    executed notebook must not record that as its framing."""
    view = _view(size=(300, 200))
    try:
        view.canvas.set_logical_size(1, 1)
        assert view.logical_size() == (300, 200)
    finally:
        view.close()


def test_capture_writes_a_png(has_gpu, tmp_path):
    pytest.importorskip("imageio")
    view = _view(size=(40, 30))
    try:
        path = view.save(str(tmp_path / "shot.png"))
        assert path.endswith("shot.png")
        from imageio import v3 as iio
        assert iio.imread(path).shape[:2] == (30, 40)
    finally:
        view.close()
