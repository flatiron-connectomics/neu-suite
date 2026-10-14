"""The pygfx backend: `build()` (needs pygfx, not a GPU), then real renders (skip without
an adapter)."""

import numpy as np
import pytest

pygfx = pytest.importorskip("pygfx", reason="the render extra is not installed")

from neu_lib import Mesh, Skeleton                      # noqa: E402
from neu_draw.scene import MARKERS, Scene, build_scene, resolve_marker  # noqa: E402
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
    """`scene.py` lists the markers without importing pygfx, so the list can drift."""
    import pygfx.utils.enums as enums

    real = {m for m in dir(enums.MarkerShape) if not m.startswith("_")}
    assert set(MARKERS) <= real, f"not real pygfx markers: {set(MARKERS) - real}"


def test_the_matplotlib_shorthands_resolve():
    assert resolve_marker("^") == "triangle_up"
    assert resolve_marker("s") == "square"


# --------------------------------------------------------------------------- #
# translation
# --------------------------------------------------------------------------- #

def test_each_drawable_becomes_the_expected_pygfx_object():
    scene = build_scene(meshes=[_mesh()], skeletons=[_skeleton()],
                        points={"presyn": np.array([[0.0, 0, 50.0]])})
    kinds = {obj.name: type(obj).__name__ for obj in backend.build(scene).children}
    assert kinds == {"body": "Mesh", "skel": "Line", "presyn": "Points"}


def test_a_skeleton_becomes_one_object_holding_every_edge():
    """Two edges is one Line of four positions: one draw call per body, whatever its topology."""
    scene = Scene().add_skeleton(_skeleton())
    (line,) = backend.build(scene).children
    assert isinstance(line.material, pygfx.LineSegmentMaterial)
    assert len(line.geometry.positions.data) == 4


def test_positions_reach_pygfx_as_xyz():
    """zyx in memory, xyz for pygfx; the wrong order mirrors the data without any error."""
    skel = Skeleton(np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 4.0]]), np.array([[0, 1]]))
    (line,) = backend.build(Scene().add_skeleton(skel)).children
    assert line.geometry.positions.data[0].tolist() == [3.0, 2.0, 1.0]


def test_an_offset_becomes_a_transform_and_never_touches_the_vertices():
    """So `mesh.bbox` keeps reporting where the tissue is, not where it is drawn."""
    scene = Scene().add_mesh(_mesh())
    (unmoved,) = backend.build(scene).children
    assert tuple(unmoved.local.position) == (0.0, 0.0, 0.0)

    moved = scene.offset_by((0.0, 0.0, 500.0))          # zyx: +500 in x
    (obj,) = backend.build(moved).children
    assert tuple(obj.local.position) == (500.0, 0.0, 0.0)          # xyz at the boundary
    assert obj.geometry.positions.data[0].tolist() == [0.0, 0.0, 0.0]   # untouched


def test_every_drawable_kind_carries_its_transform():
    scene = Scene().add_mesh(_mesh()).add_skeleton(_skeleton()).add_points(
        np.array([[0.0, 0, 0]]), name="pts")
    built = backend.build(scene.offset_by((1.0, 2.0, 3.0)))
    assert all(tuple(o.local.position) == (3.0, 2.0, 1.0) for o in built.children)


def test_alpha_multiplies_the_colours_own_alpha():
    """A per-drawable alpha and an #rrggbbaa colour compose."""
    scene = Scene().add_mesh(_mesh(), color="#ff000080", alpha=0.5)
    (mesh,) = backend.build(scene).children
    assert mesh.material.color[3] == pytest.approx(0.5 * 128 / 255, abs=1e-3)


def test_a_hidden_drawable_is_built_and_switched_off():
    """Built, so a legend row can turn it back on."""
    scene = Scene().add_mesh(_mesh("shown")).add_mesh(_mesh("hidden"))
    scene.get("hidden").visible = False
    built = {o.name: o.visible for o in backend.build(scene).children}
    assert built == {"shown": True, "hidden": False}


def test_mesh_normals_are_passed_through_when_present():
    mesh = _mesh()
    mesh.normals_zyx = np.tile([0.0, 0.0, 1.0], (4, 1)).astype(np.float32)
    (obj,) = backend.build(Scene().add_mesh(mesh)).children
    assert obj.geometry.normals.data[0].tolist() == [1.0, 0.0, 0.0]   # flipped to xyz


def test_an_empty_skeleton_builds_rather_than_erroring():
    """A body cropped to nothing is ordinary; a zero-length wgpu buffer would fail."""
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
        # A lit mesh cannot leave every pixel identical.
        assert len(np.unique(image.reshape(-1, image.shape[-1]), axis=0)) > 1
    finally:
        view.close()


def test_snapshots_are_supersampled_by_default(has_gpu):
    """pygfx renders at >=2x and downsamples, so a snapshot is larger than the logical size."""
    view = backend.show(build_scene(meshes=[_mesh()]), size=(100, 80),
                        canvas="offscreen")
    try:
        assert view.pixel_ratio >= 2
        h, w = view.snapshot().shape[:2]
        assert (w, h) == (100 * view.pixel_ratio, 80 * view.pixel_ratio)
    finally:
        view.close()


def test_the_camera_frames_the_data_not_the_origin(has_gpu):
    """Bodies sit tens of microns from the origin; a default camera would show nothing."""
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
    """An explicit size re-renders offscreen: in Jupyter the live framebuffer can still be a
    2x2 placeholder before the widget is laid out."""
    view = backend.show(build_scene(meshes=[_mesh()]), size=(50, 40),
                        canvas="offscreen", pixel_ratio=1.0)
    try:
        assert view.snapshot(size=(200, 120)).shape[:2] == (120, 200)
        assert view.snapshot().shape[:2] == (40, 50)
    finally:
        view.close()


def test_an_offscreen_resnapshot_draws_the_same_scene(has_gpu):
    """The offscreen pass builds its own renderer and camera, and must not come out blank."""
    view = backend.show(build_scene(meshes=[_mesh()], skeletons=[_skeleton()]),
                        size=(80, 80), canvas="offscreen", pixel_ratio=1.0)
    try:
        image = view.snapshot(size=(80, 80))
        assert len(np.unique(image.reshape(-1, 4), axis=0)) > 1
    finally:
        view.close()


def test_a_view_offers_a_mimebundle_without_raising(has_gpu):
    """Jupyter's display hook; without a bundle it falls back to a repr, never raising."""
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
