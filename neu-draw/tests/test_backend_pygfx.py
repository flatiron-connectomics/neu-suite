"""The pygfx backend: the translation, then an actual render.

Split deliberately. `build()` is assertable without a GPU — what got made, with which
colours and how many vertices — and those tests run wherever pygfx imports. The render
tests need an adapter and skip without one.
"""

import numpy as np
import pytest

pygfx = pytest.importorskip("pygfx", reason="the render extra is not installed")

from neu_draw.geometry import Mesh, Skeleton                      # noqa: E402
from neu_draw.scene import MARKERS, PointsDrawable, Scene, build_scene, resolve_marker  # noqa: E402
from neu_draw.backends import pygfx as backend                    # noqa: E402


@pytest.fixture
def has_gpu():
    """An adapter, or a skip. Rendering needs one; building the object graph does not."""
    import wgpu
    try:
        wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    except Exception as exc:                                    # pragma: no cover
        pytest.skip(f"no wgpu adapter available: {exc}")
    return True


def _mesh(name="body"):
    verts = np.array([[0.0, 0, 0], [100.0, 0, 0], [0.0, 100.0, 0], [0.0, 0, 100.0]])
    return Mesh(verts, np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]), name=name)


def _skeleton(name="skel"):
    verts = np.array([[0.0, 0, 0], [0.0, 0, 100.0], [0.0, 50.0, 100.0]])
    return Skeleton(verts, np.array([[0, 1], [1, 2]]), name=name)


# --------------------------------------------------------------------------- #
# the marker vocabulary, checked against the real enum
# --------------------------------------------------------------------------- #

def test_every_marker_we_advertise_exists_in_pygfx():
    """`scene.py` writes the vocabulary out so it needs no renderer, which means it can
    drift. It already did: the obvious name `triangle` does not exist — pygfx has four
    directional ones — and pygfx only rejects it at draw time."""
    import pygfx.utils.enums as enums

    real = {m for m in dir(enums.MarkerShape) if not m.startswith("_")}
    assert set(MARKERS) <= real, f"not real pygfx markers: {set(MARKERS) - real}"


def test_the_matplotlib_shorthands_the_old_code_used_resolve():
    """`presyn_marker='^'`, `postsyn_marker='s'` were the actual arguments."""
    assert resolve_marker("^") == "triangle_up"
    assert resolve_marker("s") == "square"


def test_a_bare_triangle_is_refused_with_the_alternatives():
    with pytest.raises(ValueError, match="unknown marker"):
        PointsDrawable(np.zeros((1, 3)), marker="triangle")


# --------------------------------------------------------------------------- #
# translation
# --------------------------------------------------------------------------- #

def test_each_drawable_becomes_the_expected_pygfx_object():
    scene = build_scene(meshes=[_mesh()], skeletons=[_skeleton()],
                        points={"presyn": np.array([[0.0, 0, 50.0]])})
    kinds = {obj.name: type(obj).__name__ for obj in backend.build(scene).children}
    assert kinds == {"body": "Mesh", "skel": "Line", "presyn": "Points"}


def test_a_skeleton_becomes_one_object_holding_every_edge():
    """The heart of it. Two edges is one Line of four positions — not two graphics, and
    not a polyline of three. A fragmented body stays one draw call."""
    scene = Scene().add_skeleton(_skeleton())
    (line,) = backend.build(scene).children
    assert isinstance(line.material, pygfx.LineSegmentMaterial)
    assert len(line.geometry.positions.data) == 4


def test_positions_reach_pygfx_as_xyz():
    """zyx in memory, xyz at the boundary. Getting this wrong mirrors the data through
    the z=x diagonal, which renders fine and is in the wrong place."""
    skel = Skeleton(np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 4.0]]), np.array([[0, 1]]))
    (line,) = backend.build(Scene().add_skeleton(skel)).children
    assert line.geometry.positions.data[0].tolist() == [3.0, 2.0, 1.0]


def test_alpha_multiplies_the_colours_own_alpha():
    """So a per-drawable alpha and an #rrggbbaa colour compose, instead of one winning."""
    scene = Scene().add_mesh(_mesh(), color="#ff000080", alpha=0.5)
    (mesh,) = backend.build(scene).children
    assert mesh.material.color[3] == pytest.approx(0.5 * 128 / 255, abs=1e-3)


def test_a_hidden_drawable_is_not_built():
    scene = Scene().add_mesh(_mesh("shown")).add_mesh(_mesh("hidden"))
    scene.get("hidden").visible = False
    assert [o.name for o in backend.build(scene).children] == ["shown"]


def test_mesh_normals_are_passed_through_when_present():
    mesh = _mesh()
    mesh.normals_zyx = np.tile([0.0, 0.0, 1.0], (4, 1)).astype(np.float32)
    (obj,) = backend.build(Scene().add_mesh(mesh)).children
    assert obj.geometry.normals.data[0].tolist() == [1.0, 0.0, 0.0]   # flipped to xyz


def test_an_empty_skeleton_builds_rather_than_erroring():
    """A body cropped away to nothing is ordinary; a zero-length wgpu buffer is not."""
    empty = Skeleton(np.zeros((0, 3)), np.zeros((0, 2)), name="gone")
    (line,) = backend.build(Scene().add_skeleton(empty)).children
    assert len(line.geometry.positions.data) == 2


def test_an_unmappable_drawable_says_so():
    class Odd:
        visible = True
        name = "odd"

    scene = Scene()
    scene.drawables.append(Odd())
    with pytest.raises(TypeError, match="no pygfx mapping"):
        backend.build(scene)


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #

def test_a_scene_renders_to_pixels(has_gpu):
    scene = build_scene(meshes=[_mesh()], skeletons=[_skeleton()])
    view = backend.show(scene, size=(200, 150), canvas="offscreen", pixel_ratio=1.0)
    try:
        image = view.snapshot()
        assert image.shape[:2] == (150, 200) and image.dtype == np.uint8
        # Something was actually drawn: a lit mesh cannot leave every pixel identical.
        assert len(np.unique(image.reshape(-1, image.shape[-1]), axis=0)) > 1
    finally:
        view.close()


def test_snapshots_are_supersampled_by_default(has_gpu):
    """pygfx renders to an internal texture at >=2x and downsamples, which is where the
    antialiasing comes from — so a snapshot is bigger than the size you asked for. A
    surprise worth pinning, since it silently changes what `save()` writes."""
    view = backend.show(build_scene(meshes=[_mesh()]), size=(100, 80),
                        canvas="offscreen")
    try:
        assert view.pixel_ratio >= 2
        h, w = view.snapshot().shape[:2]
        assert (w, h) == (100 * view.pixel_ratio, 80 * view.pixel_ratio)
    finally:
        view.close()


def test_the_camera_frames_the_data_not_the_origin(has_gpu):
    """Bodies sit tens of microns out in the volume. A camera left at its default looks
    at empty space, and the cell comes back blank."""
    far = _mesh()
    far.vertices_zyx_nm = far.vertices_zyx_nm + 50_000.0
    view = backend.show(Scene().add_mesh(far), size=(120, 120), canvas="offscreen")
    try:
        assert np.linalg.norm(view.camera.local.position) > 10_000
        assert len(np.unique(view.snapshot().reshape(-1, 4), axis=0)) > 1
    finally:
        view.close()


def test_an_empty_scene_still_renders(has_gpu):
    view = backend.show(Scene(), size=(64, 64), canvas="offscreen", pixel_ratio=1.0)
    try:
        assert view.snapshot().shape[:2] == (64, 64)
    finally:
        view.close()


def test_snapshot_honours_an_explicit_size_off_the_live_canvas(has_gpu):
    """The Jupyter path: the live framebuffer is sized by whatever the browser reported,
    so before the widget is laid out it is a placeholder — a view created at (900, 700)
    snapshotted as 2x2 in a freshly executed notebook, and `save()` wrote it. An explicit
    size therefore re-renders offscreen instead of reading that framebuffer."""
    view = backend.show(build_scene(meshes=[_mesh()]), size=(50, 40),
                        canvas="offscreen", pixel_ratio=1.0)
    try:
        assert view.snapshot(size=(200, 120)).shape[:2] == (120, 200)
        assert view.snapshot().shape[:2] == (40, 50)
    finally:
        view.close()


def test_an_offscreen_resnapshot_draws_the_same_scene(has_gpu):
    """The second pass must show the picture, not an empty frame — it builds its own
    renderer and camera, so a scene or camera it failed to carry over would be blank."""
    view = backend.show(build_scene(meshes=[_mesh()], skeletons=[_skeleton()]),
                        size=(80, 80), canvas="offscreen", pixel_ratio=1.0)
    try:
        image = view.snapshot(size=(80, 80))
        assert len(np.unique(image.reshape(-1, 4), axis=0)) > 1
    finally:
        view.close()


def test_a_view_offers_a_mimebundle_without_raising(has_gpu):
    """What Jupyter calls to display the cell. The offscreen canvas has no bundle, so
    this degrades to a repr — and it must never raise, because the failure would land
    exactly where the figure was meant to appear."""
    view = backend.show(Scene(), size=(32, 32), canvas="offscreen")
    try:
        bundle = view._repr_mimebundle_(include=None, exclude=None)
        assert isinstance(bundle, dict) and bundle
    finally:
        view.close()


def test_get_backend_resolves_and_rejects_unknown_names():
    from neu_draw.backends import get_backend

    assert get_backend("pygfx") is backend
    with pytest.raises(ValueError, match="unknown backend"):
        get_backend("opengl")
