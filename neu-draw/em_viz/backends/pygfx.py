"""Turn a :class:`~em_viz.scene.Scene` into pygfx objects and draw them.

The whole renderer-specific surface of this package. Everything above it — geometry,
colours, scene assembly — is arrays and dataclasses, which is what lets the interesting
parts be tested without a GPU.

Three primitives cover segment morphology:

============  =========================================================
drawable      pygfx
============  =========================================================
mesh          ``Mesh`` + ``MeshPhongMaterial``
skeleton      ``Line`` + ``LineSegmentMaterial``, fed the edge list
points        ``Points`` + ``PointsMarkerMaterial``
============  =========================================================

``LineSegmentMaterial`` is the one that shapes the design: it "renders line segments
between each two subsequent points", so a skeleton's edge list draws directly, as **one
object per body** whatever its topology. The predecessor decomposed each skeleton into
branch polylines and emitted one graphic per branch — hundreds or thousands for a
fragmented body — purely because ``fastplotlib.add_line_collection`` wanted a list.

Volume drawables are not implemented; see :class:`~em_viz.scene.VolumeDrawable`.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pygfx

from ..geometry import to_xyz
from ..scene import LinesDrawable, MeshDrawable, PointsDrawable, Scene

#: Nothing in a scene is at the origin — a body sits wherever it sits in the volume, tens
#: of microns out — so the camera must be framed from the data, never left at a default.
DEFAULT_SIZE = (900, 700)


def build(scene: Scene) -> pygfx.Group:
    """Scene → a pygfx object graph. No canvas, no renderer, no draw.

    Separate from :func:`show` so the translation can be asserted on: what got built,
    with which colours and how many vertices, is a question about this function and not
    about a GPU.
    """
    group = pygfx.Group()
    for drawable in scene:
        if not drawable.visible:
            continue
        obj = _build_one(drawable)
        if drawable.name is not None:
            obj.name = str(drawable.name)
        group.add(obj)
    return group


def _build_one(drawable: Any) -> pygfx.WorldObject:
    if isinstance(drawable, MeshDrawable):
        return _mesh(drawable)
    if isinstance(drawable, LinesDrawable):
        return _lines(drawable)
    if isinstance(drawable, PointsDrawable):
        return _points(drawable)
    raise TypeError(f"no pygfx mapping for {type(drawable).__name__}")


def _mesh(drawable: MeshDrawable) -> pygfx.Mesh:
    mesh = drawable.mesh
    fields: dict[str, Any] = {
        "positions": to_xyz(mesh.vertices_zyx_nm),
        "indices": np.ascontiguousarray(mesh.faces, dtype=np.uint32),
    }
    # Normals go in the CONSTRUCTOR. Assigning `geometry.normals = arr` afterwards
    # stores the bare ndarray instead of wrapping it in a `Buffer`, so it never reaches
    # the GPU — and the only symptom is an unhelpful "multi-dimensional sub-views are
    # not implemented" the next time anything reads it back.
    if mesh.normals_zyx is not None:
        fields["normals"] = to_xyz(mesh.normals_zyx)
    return pygfx.Mesh(pygfx.Geometry(**fields),
                      pygfx.MeshPhongMaterial(**_material_kwargs(drawable)))


def _lines(drawable: LinesDrawable) -> pygfx.Line:
    positions = to_xyz(drawable.skeleton.segments())
    if not len(positions):
        # An empty buffer is a wgpu error, not an empty picture. A body whose skeleton
        # cropped away to nothing is ordinary, so it becomes a degenerate line instead.
        positions = np.zeros((2, 3), dtype=np.float32)
    return pygfx.Line(
        pygfx.Geometry(positions=positions),
        pygfx.LineSegmentMaterial(thickness=drawable.thickness,
                                  **_material_kwargs(drawable)),
    )


def _points(drawable: PointsDrawable) -> pygfx.Points:
    positions = to_xyz(drawable.positions_zyx_nm)
    if not len(positions):
        positions = np.zeros((1, 3), dtype=np.float32)
    return pygfx.Points(
        pygfx.Geometry(positions=positions),
        pygfx.PointsMarkerMaterial(size=drawable.size, marker=drawable.marker,
                                   **_material_kwargs(drawable)),
    )


def _material_kwargs(drawable: Any) -> dict:
    """Colour and transparency, uniformly.

    ``alpha`` multiplies whatever the colour already carried rather than replacing it,
    so a per-drawable alpha and an ``#rrggbbaa`` colour compose instead of one silently
    winning. ``alpha_mode`` is left at pygfx's ``auto``: it picks depth-write behaviour
    from the alpha, which is the right default for translucent overlapping surfaces.
    """
    r, g, b, a = drawable.color
    return {"color": (r, g, b, a * float(drawable.alpha))}


class View:
    """A rendered scene: canvas, renderer, camera and controller, kept together.

    Displays itself in Jupyter. Held as an object rather than returned as a tuple
    because the canvas must outlive the call — a garbage-collected canvas is a blank
    output cell, which reads as a rendering bug.
    """

    def __init__(self, scene: Scene, size: tuple[int, int] = DEFAULT_SIZE,
                 canvas: Any = "auto", background: Optional[tuple] = None,
                 pixel_ratio: Optional[float] = None):
        self.scene_data = scene
        self.canvas = (_make_canvas(size, canvas) if isinstance(canvas, (str, type(None)))
                       else canvas)
        # `pixel_ratio=None` is pygfx's default and means "at least 2" — it renders to an
        # internal texture at twice the logical size and downsamples, which is where the
        # antialiasing comes from. Worth knowing because `snapshot()` returns that
        # internal texture, so `size=(900, 700)` yields an 1800x1400 image. Pass 1.0 for
        # pixel-exact output; see `.pixel_ratio`.
        self.renderer = pygfx.renderers.WgpuRenderer(self.canvas,
                                                     pixel_ratio=pixel_ratio)

        self.scene = pygfx.Scene()
        bg = background if background is not None else scene.background
        if bg is not None:
            self.scene.add(pygfx.Background(
                None, pygfx.BackgroundMaterial(tuple(bg))))

        # A phong surface is black without light. Ambient alone flattens a mesh to a
        # silhouette, which is the failure that reads as "my mesh did not load".
        self.scene.add(pygfx.AmbientLight(intensity=0.6))
        directional = pygfx.DirectionalLight(intensity=2.5)
        directional.local.position = (-1, -1, 1)
        self.scene.add(directional)

        self.group = build(scene)
        self.scene.add(self.group)

        self.camera = pygfx.PerspectiveCamera(50)
        self.controller = pygfx.TrackballController(self.camera,
                                                    register_events=self.renderer)
        self.frame()

        if scene.axes_visible:
            self.scene.add(pygfx.AxesHelper(size=_extent(scene) * 0.2 or 1.0))

        self.canvas.request_draw(self._draw)

    # -- camera ----------------------------------------------------------------

    def frame(self) -> "View":
        """Fit the camera to the scene, honouring its :class:`~em_viz.scene.Camera`.

        Passes the built ``Group`` when there is one, so the fit follows the objects'
        *world* transforms rather than the scene's nm bounding box — the two agree today
        because nothing applies a transform, and the group stays right if anything ever
        does. Otherwise a bounding **sphere** ``(x, y, z, radius)``, the only other thing
        ``show_object`` accepts, which is needed for an explicit ``camera.target`` (a
        BBox) and for an empty scene, which has no bounding sphere and would raise.

        Either way the fit is **conservative**: pygfx frames the bounding sphere, so a
        long thin neurite leaves slack around its short axes. That is the safe direction
        to err — nothing is ever cropped out — and ``Camera.zoom`` is the way in.
        """
        intent = self.scene_data.camera
        if intent.target is not None:
            target: Any = _bounding_sphere(intent.target)
        elif self.scene_data.bbox.is_empty():
            target = _bounding_sphere(self.scene_data.bbox)
        else:
            target = self.group
        self.camera.show_object(target, view_dir=intent.view_angle, up=intent.up)
        self.camera.zoom = intent.zoom
        return self

    # -- output ----------------------------------------------------------------

    def _draw(self) -> None:
        self.renderer.render(self.scene, self.camera)

    @property
    def pixel_ratio(self) -> float:
        """Internal pixels per logical pixel. ``>= 2`` by default — see the constructor."""
        return float(self.renderer.pixel_ratio)

    def snapshot(self) -> np.ndarray:
        """Render once and return the pixels as ``(h, w, 4)`` uint8.

        **The result is `pixel_ratio` times the requested size**, not the requested size:
        this is pygfx's supersampled internal texture, which is why the image is smooth.
        Construct the view with ``pixel_ratio=1.0`` if you need pixel-exact output.
        """
        self.renderer.render(self.scene, self.camera)
        return np.asarray(self.renderer.snapshot())

    def save(self, path: str) -> str:
        """Write a snapshot to a PNG. Alpha is dropped — a figure wants a flat image."""
        try:
            from imageio import v3 as iio
        except ImportError as exc:                              # pragma: no cover
            raise ImportError(
                "saving needs imageio: pip install 'em-viz[render]'. "
                "`snapshot()` returns the array if you would rather write it "
                "yourself.") from exc

        iio.imwrite(path, self.snapshot()[..., :3])
        return path

    def close(self) -> None:
        self.canvas.close()

    def _repr_mimebundle_(self, *args, **kwargs):
        """Let Jupyter display the canvas when the View is the cell's value."""
        return self.canvas._repr_mimebundle_(*args, **kwargs)

    def __repr__(self) -> str:
        return f"View({self.scene_data!r})"


def show(scene: Scene, size: tuple[int, int] = DEFAULT_SIZE, **kwargs) -> View:
    """Render a scene and return the :class:`View`. The one call a notebook needs."""
    return View(scene, size=size, **kwargs)


def in_notebook() -> bool:
    """True inside an IPython **kernel** — a notebook — and not a terminal REPL."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and hasattr(shell, "kernel")


def _make_canvas(size: tuple[int, int], kind: str = "auto") -> Any:
    """Pick a canvas: ``auto``, ``jupyter``, ``offscreen``, or ``gui``.

    ``auto`` means **jupyter in a notebook, offscreen everywhere else** — deliberately
    not ``rendercanvas.auto``, which prefers a desktop toolkit. On this workstation that
    picks Qt and opens real windows, which is wrong twice over: a test run should not
    spawn windows, and a desktop window renders at the display's device pixel ratio
    (2.5 here), so ``snapshot()`` comes back at a size the caller never asked for.

    Pass ``kind="gui"`` to ask for a desktop window on purpose.
    """
    if kind == "auto":
        kind = "jupyter" if in_notebook() else "offscreen"
    if kind == "jupyter":
        from rendercanvas.jupyter import RenderCanvas
        return RenderCanvas(size=size)
    if kind == "offscreen":
        from rendercanvas.offscreen import RenderCanvas
        return RenderCanvas(size=size)
    if kind == "gui":
        from rendercanvas.auto import RenderCanvas
        return RenderCanvas(size=size)
    raise ValueError(
        f"unknown canvas {kind!r}; use 'auto', 'jupyter', 'offscreen', 'gui', "
        f"or pass a canvas instance")


def _extent(scene: Scene) -> float:
    box = scene.bbox
    return 0.0 if box.is_empty() else float(max(box.shape))


def _bounding_sphere(box) -> tuple[float, float, float, float]:
    """A zyx :class:`~em_volume_tools.BBox` as the ``(x, y, z, radius)`` pygfx wants.

    An empty box becomes a unit sphere at the origin, so a scene with nothing in it
    still opens on something rather than raising.
    """
    if box.is_empty():
        return (0.0, 0.0, 0.0, 1.0)
    lo = np.asarray(box.lo, dtype=float)[::-1]          # zyx -> xyz
    hi = np.asarray(box.hi, dtype=float)[::-1]
    center = (lo + hi) / 2.0
    radius = float(np.linalg.norm(hi - lo) / 2.0) or 1.0
    return (float(center[0]), float(center[1]), float(center[2]), radius)
