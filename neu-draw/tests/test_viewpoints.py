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
