"""SVG export: pure arithmetic first, then checks against what the GPU actually drew.

The two ways this goes wrong silently are both checked against a real render, not
against the formulas: vectors projected to somewhere other than where pygfx put them,
and mesh layers that composite to a different colour than the canvas showed.
"""

import base64
import xml.etree.ElementTree as ET

import numpy as np
import pytest

pygfx = pytest.importorskip("pygfx", reason="the render extra is not installed")

from neu_lib import Mesh, Skeleton                              # noqa: E402
from neu_draw.backends import pygfx as backend                  # noqa: E402
from neu_draw.backends import vector                            # noqa: E402
from neu_draw import vectorfile                                 # noqa: E402
from neu_draw.scene import Scene, build_scene                   # noqa: E402

SVG = "{http://www.w3.org/2000/svg}"
LABEL = "{http://www.inkscape.org/namespaces/inkscape}label"
XLINK = "{http://www.w3.org/1999/xlink}href"


@pytest.fixture
def has_gpu():
    import wgpu
    try:
        wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    except Exception as exc:                                    # pragma: no cover
        pytest.skip(f"no wgpu adapter available: {exc}")
    return True


def _mesh(name="body", shift=0.0, scale=100.0):
    verts = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]) * scale + shift
    return Mesh(verts, np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]), name=name)


def _layers(root, name):
    return next(g for g in root.iter(f"{SVG}g") if g.get(LABEL) == name)


def _svg(view, **kwargs):
    tree, width, height = vector.figure(view, **kwargs)
    return ET.fromstring(vectorfile.to_svg(tree, width, height).split("\n", 1)[1])


# --------------------------------------------------------------------------- #
# pure
# --------------------------------------------------------------------------- #

def test_projection_maps_ndc_onto_the_rect_with_y_down():
    px, _, front = vector.project(np.array([[0.0, 0, 0], [1, 1, 0], [-1, -1, 0]]),
                                  np.eye(4), (10, 20, 100, 50))
    assert px.tolist() == [[60.0, 45.0], [110.0, 20.0], [10.0, 70.0]]
    assert front.all()


def test_a_point_behind_the_camera_is_flagged_rather_than_mirrored():
    matrix = np.eye(4)
    matrix[3] = [0, 0, -1, 0]                   # w = -z, as a perspective camera has
    _, _, front = vector.project(np.array([[0.0, 0, -5], [0, 0, 5]]), matrix, (0, 0, 1, 1))
    assert front.tolist() == [True, False]


@pytest.mark.parametrize("background", [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.2, 0.3, 0.5)])
def test_a_layer_pixel_composited_in_srgb_is_the_canvas_pixel(background):
    """Including the bright-and-faint case, whose colour alone would have to exceed 1:
    over black the canvas shows the premultiplied value, 90, and at alpha 51 an sRGB
    blend can reach only 51 — so the alpha must be raised, not the colour clipped."""
    rng = np.random.default_rng(0)
    alpha = rng.integers(1, 256, (64, 1))
    linear = rng.random((64, 3)) * alpha / 255.0          # premultiplied, linear
    v = np.round(vector._srgb_encode(linear) * 255.0)
    rgba = np.c_[v, alpha].astype(np.uint8)[None]
    bg = np.array(background)
    canvas = vector._srgb_encode(vector._srgb_decode(rgba[0, :, :3] / 255.0)
                                 + vector._srgb_decode(bg) * (1 - alpha / 255.0))

    out = vector.layer_pixels(rgba, background)[0].astype(float) / 255.0
    composite = out[:, :3] * out[:, 3:] + bg * (1 - out[:, 3:])
    assert np.abs(composite - canvas).max() < 2.5 / 255
    assert (out[:, 3] >= alpha[:, 0] / 255.0 - 1e-9).all()     # only ever raised


def test_a_transparent_pixel_stays_transparent():
    out = vector.layer_pixels(np.zeros((1, 1, 4), dtype=np.uint8), (1.0, 1.0, 1.0))
    assert out.tolist() == [[[0, 0, 0, 0]]]


def test_group_ids_use_illustrators_own_escaping():
    """Illustrator writes a layer name into an id this way and decodes it on import, so
    the layer panel reads the name rather than an id mangled to underscores."""
    assert vectorfile.illustrator_id("T4a 78044864 : Mi1 (17)") == (
        "T4a_x20_78044864_x20__x3A__x20_Mi1_x20__x28_17_x29_")
    assert vectorfile.illustrator_id("78044864") == "_x37_8044864"
    assert vectorfile.illustrator_id("a_b") == "a_x5F_b"


# --------------------------------------------------------------------------- #
# against the GPU
# --------------------------------------------------------------------------- #

def test_a_projected_point_lands_where_pygfx_drew_it(has_gpu):
    scene = Scene(background=(0.0, 0.0, 0.0, 1.0))
    scene.add_points(np.array([[0.0, 0, 0], [0.0, 400.0, 900.0]]), name="p",
                     color=(1.0, 1.0, 1.0, 1.0), size=12.0)
    view = backend.show(scene, size=(200, 150), canvas="offscreen", legend=False,
                        pixel_ratio=1.0)
    try:
        img = view.snapshot()[..., 0].astype(float)
        root = _svg(view)
        circles = list(_layers(root, "p").iter(f"{SVG}circle"))
        assert len(circles) == 2
        for c in circles:
            cx, cy = float(c.get("cx")), float(c.get("cy"))
            y0, x0 = int(cy) - 8, int(cx) - 8
            patch = img[max(y0, 0):y0 + 17, max(x0, 0):x0 + 17]
            ys, xs = np.nonzero(patch > 128)
            assert len(ys), f"nothing drawn near ({cx}, {cy})"
            assert xs.mean() + max(x0, 0) + 0.5 == pytest.approx(cx, abs=1.0)
            assert ys.mean() + max(y0, 0) + 0.5 == pytest.approx(cy, abs=1.0)
    finally:
        view.close()


@pytest.mark.parametrize("background", [(0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0, 1.0),
                                        (0.2, 0.3, 0.5, 1.0)])
def test_a_mesh_layer_composited_like_an_svg_viewer_matches_the_canvas(has_gpu,
                                                                        background):
    """The measured failure: straight alpha done 'correctly' came out far too dark in an
    editor, because the GPU blends in linear light and SVG viewers blend in sRGB."""
    scene = build_scene(meshes=[_mesh()], colors={"body": (0.8, 0.7, 0.9)}, alpha=0.25,
                        background=background)
    view = backend.show(scene, size=(160, 120), canvas="offscreen", legend=False,
                        pixel_ratio=1.0)
    try:
        canvas = view.snapshot()[..., :3].astype(float)
        camera = pygfx.PerspectiveCamera(view.camera.fov)
        camera.set_state(view.camera.get_state())
        layer = vector.layer_pixels(
            vector._render_alone(view, 0, camera, (160, 120), 1.0), background)
        a = layer[..., 3:4] / 255.0
        bg = np.array(background[:3]) * 255.0
        composite = layer[..., :3] * a + bg * (1 - a)       # what an SVG viewer does
        covered = layer[..., 3] > 0
        assert covered.sum() > 200
        assert np.abs(composite - canvas)[covered].mean() < 2.0
    finally:
        view.close()


@pytest.fixture
def figure(has_gpu):
    skel = Skeleton(np.array([[0.0, 0, 0], [0, 0, 300.0], [0, 150.0, 300]]),
                    np.array([[0, 1], [1, 2]]), name="A")
    scene = build_scene(meshes=[_mesh("A"), _mesh("B", 300.0)], skeletons=[skel],
                        points={"syn": np.array([[0.0, 50, 50], [0, 60, 60]])})
    scene.get("B").visible = False
    view = backend.show(scene, size=(400, 300), canvas="offscreen")
    yield view
    view.close()


def test_the_svg_is_layered_by_kind_then_by_drawable(figure):
    root = _svg(figure)
    top = [g.get(LABEL) for g in root if g.tag == f"{SVG}g"]
    assert top == ["background", "meshes", "skeletons", "points", "legend"]
    assert [g.get(LABEL) for g in _layers(root, "meshes")] == ["A mesh"]   # B hidden
    assert len(list(_layers(root, "A mesh").iter(f"{SVG}image"))) == 1
    assert len(list(_layers(root, "syn").iter(f"{SVG}circle"))) == 2


def test_a_mesh_layer_is_a_cropped_png_with_a_transparent_surround(figure):
    root = _svg(figure)
    (image,) = _layers(root, "A mesh").iter(f"{SVG}image")
    png = base64.b64decode(image.get(XLINK).split(",", 1)[1])
    from imageio import v3 as iio

    rgba = iio.imread(png)
    assert rgba.shape[2] == 4
    assert float(image.get("width")) < 400 * 0.7          # cropped to the body
    assert rgba[..., 3].min() == 0 and rgba[..., 3].max() > 0


def test_the_legend_is_real_text_and_keeps_the_hidden_row_dimmed(figure):
    root = _svg(figure)
    texts = {t.text: t for t in _layers(root, "legend").iter(f"{SVG}text")}
    assert set(figure.legend.labels) <= set(texts)
    assert texts["B"].get("fill") != texts["A mesh"].get("fill")


def test_save_writes_svg_for_an_svg_path(figure, tmp_path):
    pytest.importorskip("imageio")
    path = figure.save(str(tmp_path / "fig.svg"))
    assert ET.parse(path).getroot().tag == f"{SVG}svg"


# --------------------------------------------------------------------------- #
# nesting, Illustrator compatibility, PDF
# --------------------------------------------------------------------------- #

@pytest.fixture
def cells(has_gpu):
    points = {f"{c} : {t}": np.array([[0.0, 50 + 10 * i, 50]])
              for c in ("A", "B") for i, t in enumerate(("Mi1", "Mi2"))}
    scene = build_scene(meshes=[_mesh("A"), _mesh("B", 300.0)], points=points)
    scene.group("all Mi1", lambda d: d.name.endswith(": Mi1"))
    scene.export_groups = lambda d: (d.name.split(" : ")[0],) if " : " in d.name else ()
    view = backend.show(scene, size=(400, 300), canvas="offscreen")
    yield view
    view.close()


def test_points_nest_by_cell_then_by_input_type(cells):
    root = _svg(cells)
    by_cell = [g.get(LABEL) for g in _layers(root, "points") if g.tag == f"{SVG}g"]
    assert by_cell == ["A", "B"]
    assert [g.get(LABEL) for g in _layers(_layers(root, "points"), "A")] == [
        "A : Mi1", "A : Mi2"]


def test_legend_rows_nest_the_way_their_drawables_do(cells):
    legend = _layers(_svg(cells), "legend")
    assert [g.get(LABEL) for g in legend] == ["panel", "group rows", "rows"]
    assert [g.get(LABEL) for g in _layers(legend, "group rows")] == ["all Mi1"]
    rows = _layers(legend, "rows")
    assert [g.get(LABEL) for g in rows] == ["meshes", "points"]
    assert [g.get(LABEL) for g in _layers(_layers(rows, "points"), "B")] == [
        "B : Mi1", "B : Mi2"]


def test_an_explicit_groups_function_overrides_the_scenes(cells):
    root = _svg(cells, groups=lambda d: ())
    assert [g.get(LABEL) for g in _layers(root, "points")][:1] == ["A : Mi1"]


def test_the_svg_is_what_illustrators_importer_reads(cells):
    """SVG 1.1, images by xlink:href (a bare href imports blank), no clip path (it would
    wrap everything in a clip group), no dominant-baseline (ignored, text lands high)."""
    text = vectorfile.to_svg(*vector.figure(cells))
    assert 'version="1.1"' in text and "xlink:href=" in text
    assert " href=" not in text.replace("xlink:href=", "")
    assert "clipPath" not in text and "dominant-baseline" not in text


def _pdf_objects(data: bytes) -> list[bytes]:
    import re
    return re.findall(rb"\d+ 0 obj\n(.*?)\nendobj", data, flags=re.S)


def test_the_pdf_has_one_nested_layer_per_group(cells, tmp_path):
    path = cells.save_pdf(str(tmp_path / "fig.pdf"))
    data = open(path, "rb").read()
    assert data.startswith(b"%PDF-1.7") and data.rstrip().endswith(b"%%EOF")
    tree, _, _ = vector.figure(cells)
    names = [g.name for g in tree.walk()][1:]                 # the root is not a layer
    ocgs = [o for o in _pdf_objects(data) if o.startswith(b"<< /Type /OCG")]
    assert len(ocgs) == len(names)
    assert b"/Name (A : Mi1)" in data and b"/Order [" in data


def test_the_pdf_xref_points_at_its_objects(cells, tmp_path):
    """A wrong offset opens in forgiving viewers and fails in strict ones."""
    import re
    data = open(cells.save_pdf(str(tmp_path / "fig.pdf")), "rb").read()
    start = int(re.search(rb"startxref\n(\d+)", data).group(1))
    entries = re.findall(rb"(\d{10}) 00000 n ", data[start:])
    for number, offset in enumerate(entries, start=1):
        assert data[int(offset):].startswith(f"{number} 0 obj".encode())


def test_the_pdf_renders_what_the_svg_does(cells, tmp_path):
    """Rasterise both with the system's poppler and librsvg, where present, and compare."""
    import shutil
    import subprocess

    if not (shutil.which("pdftoppm") and shutil.which("rsvg-convert")):
        pytest.skip("needs pdftoppm and rsvg-convert")
    from imageio import v3 as iio

    cells.save_pdf(str(tmp_path / "fig.pdf"))
    cells.save_svg(str(tmp_path / "fig.svg"))
    subprocess.run(["pdftoppm", "-png", "-r", "72", "-singlefile",
                    str(tmp_path / "fig.pdf"), str(tmp_path / "pdf")], check=True)
    subprocess.run(["rsvg-convert", "-w", "400", str(tmp_path / "fig.svg"),
                    "-o", str(tmp_path / "svg.png")], check=True)
    a = iio.imread(tmp_path / "pdf.png")[..., :3].astype(float)
    b = iio.imread(tmp_path / "svg.png")[..., :3].astype(float)
    assert a.shape == b.shape
    assert np.abs(a - b).mean() < 3.0
