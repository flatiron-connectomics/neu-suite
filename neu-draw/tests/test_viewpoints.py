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
    """The store is module-level, so an angle carries to the next figure."""
    first = _view()
    saved = first.save_view()
    first.close()

    second = _view()
    try:
        assert second.restore_view() == saved
    finally:
        second.close()


def test_restoring_an_empty_slot_returns_none_rather_than_raising(has_gpu):
    """The likely state when a button is first pressed."""
    view = _view()
    try:
        assert view.restore_view() is None
        assert view.restore_view("no-such-slot") is None
    finally:
        view.close()


def test_a_viewpoint_records_the_size_it_was_framed_for(has_gpu):
    """The same camera state in a different aspect is a different picture."""
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
# files
# --------------------------------------------------------------------------- #

def _state(**camera):
    """A state shaped like ``Camera.get_state()``: numpy vectors beside plain values."""
    camera = {"position": np.array([1.5, -2.0, 3.25]),
              "rotation": np.array([0.0, 0, 0, 1]), "fov": 50.0, "zoom": np.float32(1.5), "maintain_aspect": True,
              "depth_range": None, **camera}
    return viewstate.ViewState(camera=camera, size=(640, 480))


def test_a_viewpoint_survives_a_file_round_trip(tmp_path):
    loaded = viewstate.ViewState.load(_state().save(tmp_path / "view.json"))

    assert loaded.size == (640, 480)
    assert loaded.camera["position"] == [1.5, -2.0, 3.25]
    assert loaded.camera["rotation"] == [0.0, 0.0, 0.0, 1.0]
    assert loaded.camera["zoom"] == 1.5
    assert loaded.camera["maintain_aspect"] is True
    assert loaded.camera["depth_range"] is None


def test_saving_creates_the_directory_and_writes_a_file(tmp_path):
    path = tmp_path / "new" / "dir" / "view.json"
    assert _state().save(path) == path
    assert path.is_file()


def test_saving_without_overwrite_refuses_and_keeps_the_file(tmp_path):
    path = tmp_path / "view.json"
    path.write_text("keep me")
    with pytest.raises(FileExistsError):
        _state().save(path, overwrite=False)
    assert path.read_text() == "keep me"


def test_a_failed_save_does_not_truncate_the_existing_file(tmp_path):
    """Serialization happens before the file is opened."""
    path = tmp_path / "view.json"
    path.write_text("keep me")
    with pytest.raises(TypeError, match="object is not JSON serializable"):
        _state(extra=object()).save(path)
    assert path.read_text() == "keep me"


def test_the_slot_functions_round_trip_through_the_store(tmp_path):
    viewstate.views["a"] = _state()
    path = viewstate.save_view_file(tmp_path / "view.json", "a")
    viewstate.views.clear()

    loaded = viewstate.load_view_file(path, "b")
    assert viewstate.views == {"b": loaded}
    assert loaded.camera["position"] == [1.5, -2.0, 3.25]


def test_saving_an_empty_slot_names_the_slot(tmp_path):
    viewstate.views["a"] = _state()
    with pytest.raises(KeyError, match="'missing'.*'a'"):
        viewstate.save_view_file(tmp_path / "view.json", "missing")


def test_a_viewpoint_loaded_from_a_file_restores_the_camera(has_gpu, tmp_path):
    """Loaded vectors are lists, which ``Camera.set_state`` takes as it does arrays."""
    first = _view(size=(120, 90))
    first.camera.local.position = (11.0, 22.0, 33.0)
    path = first.save_view().save(tmp_path / "view.json")
    first.close()

    view = _view()
    try:
        view.restore_view(viewstate.ViewState.load(path))
        assert np.allclose(view.camera.local.position, (11.0, 22.0, 33.0))
        assert view.logical_size() == (120, 90)
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
    """In `View.close`, so a `view.close()` from a cell counts as well as the button."""
    view = _view()
    view.camera.local.position = (1234.0, 5678.0, 9012.0)
    view.close()

    assert viewstate.LAST in viewstate.views
    assert viewstate.views[viewstate.LAST].camera["position"].tolist() == [
        1234.0, 5678.0, 9012.0]


def test_the_last_slot_is_what_the_next_figure_reopens_on(has_gpu):
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
    """Reset restores how the figure opened, not "everything visible"."""
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
    """A recolour is an authored change, not transient exploration."""
    view = _view()
    try:
        view.legend.recolor("body", "tab:pink")
        view.reset()
        assert view.scene_data.get("body").color == pytest.approx(
            (0.8902, 0.4667, 0.7608, 1.0), abs=1e-3)
    finally:
        view.close()


def test_reset_returns_to_a_restored_viewpoint_not_to_a_fit(has_gpu):
    """The opening view is recorded after `viewpoint=` is applied."""
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
    """A Jupyter canvas reports (1, 1) before layout; a saved viewpoint must not record that."""
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
