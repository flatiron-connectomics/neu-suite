"""Turn a :class:`~neu_draw.scene.Scene` into pygfx objects and draw them.

The only renderer-specific module apart from the legend and the exporters. Three
primitives cover segment morphology:

============  =========================================================
drawable      pygfx
============  =========================================================
mesh          ``Mesh`` + ``MeshPhongMaterial``
skeleton      ``Line`` + ``LineSegmentMaterial``, fed the edge list
points        ``Points`` + ``PointsMarkerMaterial``
============  =========================================================

``LineSegmentMaterial`` draws a line between each consecutive pair of points, so a
skeleton's edge list is one object per body whatever its topology.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import pygfx

from neu_lib import to_xyz
from ..scene import LinesDrawable, MeshDrawable, PointsDrawable, Scene
from ..viewstate import LAST, SAVED, ViewState, views

#: Canvas size in logical pixels when none is given.
DEFAULT_SIZE = (900, 700)


def build(scene: Scene) -> pygfx.Group:
    """Scene → a pygfx object graph. No canvas, no renderer, no draw.

    Separate from :func:`show`, so what is built can be tested without a GPU.
    """
    group = pygfx.Group()
    for drawable in scene:
        obj = _build_one(drawable)
        # A hidden drawable is built and switched off, not skipped, so the legend can turn
        # it back on.
        obj.visible = bool(drawable.visible)
        if drawable.name is not None:
            obj.name = str(drawable.name)
        # The drawable's offset becomes the object's transform; vertices are not moved.
        # `.local.position` is xyz, like every coordinate handed to pygfx.
        offset = getattr(drawable, "offset_zyx_nm", None)
        if offset is not None and tuple(offset) != (0.0, 0.0, 0.0):
            obj.local.position = offset.xyz
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
    # Normals go in the constructor: assigned afterwards, `geometry.normals` stays a bare
    # ndarray rather than a `Buffer`, and never reaches the GPU.
    if mesh.normals_zyx is not None:
        fields["normals"] = to_xyz(mesh.normals_zyx)
    return pygfx.Mesh(pygfx.Geometry(**fields),
                      pygfx.MeshPhongMaterial(**_material_kwargs(drawable)))


def _lines(drawable: LinesDrawable) -> pygfx.Line:
    positions = to_xyz(drawable.skeleton.segments())
    if not len(positions):
        # An empty buffer is a wgpu error, so an empty skeleton becomes a degenerate line.
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


def display_color(drawable: Any, override: Optional[tuple] = None) -> tuple:
    """The colour a drawable's material carries: its own, times its ``alpha``.

    ``alpha`` multiplies the colour's own alpha, so the two compose. ``override``
    substitutes another colour (the legend's highlight) but keeps the drawable's alpha, so
    a highlighted translucent surface stays translucent.
    """
    r, g, b, a = drawable.color if override is None else override
    return (r, g, b, a * float(drawable.alpha))


def _material_kwargs(drawable: Any) -> dict:
    """Colour for a new material. ``alpha_mode`` is left at pygfx's ``auto``: it picks
    depth-write behaviour from the alpha, which is right for translucent surfaces."""
    return {"color": display_color(drawable)}


class View:
    """A rendered scene: canvas, renderer, camera and controller, kept together.

    Displays itself in Jupyter. Keep a reference: a garbage-collected canvas leaves a
    blank output cell.
    """

    def __init__(self, scene: Scene, size: tuple[int, int] = DEFAULT_SIZE,
                 canvas: Any = "auto", background: Optional[tuple] = None,
                 pixel_ratio: Optional[float] = None, toolbar: Any = "auto",
                 legend: Optional[bool] = None,
                 viewpoint: Optional[Union[str, ViewState]] = None):
        self.scene_data = scene
        self.ui = None
        self.legend = None
        self._closed = False
        self._size = tuple(size)
        # What `reset()` restores: the visibility the scene opened with, which may already
        # hide something.
        self._initial_visible = [bool(d.visible) for d in scene.drawables]
        self._pixel_ratio = pixel_ratio
        self.canvas = (_make_canvas(size, canvas) if isinstance(canvas, (str, type(None)))
                       else canvas)
        # None is pygfx's default and means "at least 2" (supersampling). See `snapshot`.
        self.renderer = pygfx.renderers.WgpuRenderer(self.canvas,
                                                     pixel_ratio=pixel_ratio)

        self.scene = pygfx.Scene()
        bg = background if background is not None else scene.background
        if bg is not None:
            self.scene.add(pygfx.Background(
                None, pygfx.BackgroundMaterial(tuple(bg))))

        # Phong surfaces need light; ambient alone flattens a mesh to a silhouette.
        self.scene.add(pygfx.AmbientLight(intensity=0.6))
        directional = pygfx.DirectionalLight(intensity=2.5)
        directional.local.position = (-1, -1, 1)
        self.scene.add(directional)

        self.group = build(scene)
        self.scene.add(self.group)
        self._looks_at = self._looks()

        self._install_legend(legend)

        self.camera = pygfx.PerspectiveCamera(50)
        # With a legend, the controller listens on the main viewport only, so a click on
        # the legend does not also rotate the camera (see `backends/legend.py`).
        self.controller = pygfx.TrackballController(
            self.camera,
            register_events=self.legend.main if self.legend else self.renderer)
        self.frame()
        if viewpoint is not None:
            self.restore_view(viewpoint)
        # The camera `reset()` returns to: after the fit and any `viewpoint`.
        self._opening = ViewState(camera=dict(self.camera.get_state()),
                                  size=self.logical_size())

        if scene.axes_visible:
            self.scene.add(pygfx.AxesHelper(size=_extent(scene) * 0.2 or 1.0))

        self.canvas.request_draw(self._draw)
        # Repaint when the scene's methods change it; each frame re-reads the scene.
        scene.on_change(self.request_draw)
        self._install_toolbar(toolbar)

    # -- camera ----------------------------------------------------------------

    def frame(self) -> "View":
        """Fit the camera to what is **visible**, honouring the scene's ``Camera``.

        Fits a bounding sphere over the built objects' world bounding boxes, so offsets are
        followed. Hidden objects are excluded (pygfx's ``get_bounding_box`` would include
        them). Framing a sphere leaves slack around a long thin neurite but never crops;
        use ``Camera.zoom`` to tighten it.
        """
        intent = self.scene_data.camera
        if intent.target is not None:
            target = _bounding_sphere(intent.target)
        else:
            target = _visible_sphere(self.group)
            if target is None:
                target = _bounding_sphere(self.scene_data.bbox)

        # Fit for the rect beside the legend, whose aspect the perspective fit depends on.
        width, height = self._main_size()
        if width > 0 and height > 0:
            self.camera.set_view_size(width, height)

        self.camera.show_object(target, view_dir=intent.view_angle, up=intent.up)
        self.camera.zoom = intent.zoom
        return self

    def _main_size(self) -> tuple[float, float]:
        """The size of the rect the scene itself renders into, legend strip excluded."""
        if self.legend is None:
            return tuple(float(v) for v in self.logical_size())
        main, _ = self.legend.rects_for(self.logical_size())
        return float(main[2]), float(main[3])

    def center(self) -> "View":
        """Re-fit the camera to what is visible now, and redraw. :meth:`reset` returns to
        the opening view instead."""
        self.frame()
        self.request_draw()
        return self

    def reset(self) -> "View":
        """Back to the view this figure opened with.

        Restores the opening visibility, clears highlights, and returns to the opening
        camera. Colours and opacity changes are kept, since they are edits to the scene.
        """
        for drawable, visible in zip(self.scene_data.drawables, self._initial_visible):
            drawable.visible = visible
        if self.legend is not None:
            self.legend.clear_highlights()          # refreshes every entry as it goes
        self._sync_objects()
        self.camera.set_state(self._opening.camera)
        self.request_draw()
        return self

    def _sync_objects(self) -> None:
        """Push each drawable's colour and visibility onto the object built for it.

        Paired by index (:func:`build` makes one object per drawable, in order), so
        drawables on no legend row are covered too. Highlights are re-applied afterwards
        by the legend.
        """
        for drawable, obj in zip(self.scene_data.drawables, self.group.children):
            obj.material.color = display_color(drawable)
            obj.visible = bool(drawable.visible)
        self._looks_at = self._looks()

    def _looks(self) -> tuple:
        """Each drawable's visibility, colour and opacity, as `_sync_objects` pushes them."""
        return tuple((bool(d.visible), tuple(d.color), float(d.alpha))
                     for d in self.scene_data.drawables)

    def set_alpha(self, alpha: float, *, kind: Optional[str] = None,
                  names: Optional[Any] = None) -> "View":
        """Opacity for every drawable of a ``kind`` (``"mesh"``, ``"skeleton"``,
        ``"points"``), or of ``names``; what the toolbar's sliders do.
        :meth:`Scene.set_alpha <neu_draw.scene.Scene.set_alpha>` plus the repaint."""
        self.scene_data.set_alpha(alpha, kind=kind, names=names)
        self._sync_objects()
        if self.legend is not None:
            self.legend.sync()
        self.request_draw()
        return self

    # -- viewpoints ------------------------------------------------------------

    def logical_size(self) -> tuple[int, int]:
        """The canvas size in logical pixels, falling back to the size asked for.

        A Jupyter canvas reports (1, 1) until the browser lays the widget out, so a size
        below a few pixels is treated as unknown.
        """
        try:
            width, height = self.canvas.get_logical_size()
        except Exception:                                       # pragma: no cover
            return self._size
        return (int(width), int(height)) if width > 4 and height > 4 else self._size

    def save_view(self, name: str = SAVED) -> ViewState:
        """Record this camera into :data:`neu_draw.views`, and return what was stored.

        The store outlives the view, so the next figure can restore it.
        """
        state = ViewState(camera=dict(self.camera.get_state()), size=self.logical_size())
        views[name] = state
        return state

    def restore_view(self, name: Union[str, ViewState] = SAVED, *,
                     size: bool = True) -> Optional[ViewState]:
        """Apply a saved viewpoint; returns ``None``, not an error, if the slot is empty.

        So ``show(scene, viewpoint="last")`` is safe in a fresh session. ``name`` may also
        be a :class:`~neu_draw.viewstate.ViewState`. ``size`` restores the canvas size too,
        since the aspect ratio changes the framing; ``size=False`` keeps the canvas.
        """
        state = name if isinstance(name, ViewState) else views.get(name)
        if state is None:
            return None
        if size and state.size[0] and hasattr(self.canvas, "set_logical_size"):
            self.canvas.set_logical_size(*state.size)
        self.camera.set_state(state.camera)
        self.request_draw()
        return state

    # -- output ----------------------------------------------------------------

    def request_draw(self) -> None:
        """Ask the canvas to repaint, using the draw function already installed.

        A no-op once closed, since a scene can be edited after its view is closed.
        """
        if self._closed:
            return
        self.canvas.request_draw()

    def refresh(self) -> "View":
        """Re-read the scene and repaint, for a field set directly
        (``scene.get("a").label = "Tm2"``) when nothing else will draw a frame. The
        :class:`~neu_draw.scene.Scene` methods repaint on their own."""
        self._sync_objects()
        if self.legend is not None:
            self.legend.refresh()
        self.request_draw()
        return self

    def _draw(self) -> None:
        self._paint(self.renderer, self.camera)

    def _paint(self, renderer: Any, camera: Any, legend: bool = True) -> None:
        """One frame: the scene, then the legend strip beside it.

        Two ``render`` calls into two rects, only the second flushing (pygfx clears on the
        first render after a flush). ``renderer`` may be an offscreen one, so snapshots
        include the legend.
        """
        # Pick up fields set directly (`drawable.alpha = 0.2`), legend or not.
        if self._looks() != self._looks_at:
            self._sync_objects()
        if self.legend is None or not legend:
            renderer.render(self.scene, camera)
            return
        main, strip = self.legend.rects_for(renderer.logical_size)
        renderer.render(self.scene, camera, rect=main, flush=False)
        self.legend.draw(renderer, rect=strip)

    @property
    def pixel_ratio(self) -> float:
        """Internal pixels per logical pixel; at least 2 by default."""
        return float(self.renderer.pixel_ratio)

    def snapshot(self, size: Optional[tuple[int, int]] = None, *,
                 pixel_ratio: Optional[float] = None, legend: bool = True) -> np.ndarray:
        """Render once and return the pixels as ``(h, w, 4)`` uint8.

        On a Jupyter canvas this renders a separate offscreen pass at the view's requested
        size: the live framebuffer is sized by the browser and is a 2x2 placeholder until
        the widget is laid out.

        The result is ``pixel_ratio`` times the requested size (pygfx's supersampling,
        which antialiases); use ``pixel_ratio=1.0`` for pixel-exact output.
        ``legend=False`` omits the strip.
        """
        if (size is None and pixel_ratio is None and legend
                and _is_offscreen(self.canvas)):
            self._paint(self.renderer, self.camera)
            return np.asarray(self.renderer.snapshot())
        return self._offscreen_snapshot(tuple(size) if size else self._size,
                                        pixel_ratio=pixel_ratio, legend=legend)

    def _offscreen_snapshot(self, size: tuple[int, int], *,
                            pixel_ratio: Optional[float] = None,
                            legend: bool = True) -> np.ndarray:
        """Re-render this scene and camera at an exact size, off any live canvas."""
        from rendercanvas.offscreen import RenderCanvas as Offscreen

        canvas = Offscreen(size=size)
        ratio = pixel_ratio if pixel_ratio is not None else self._pixel_ratio
        renderer = pygfx.renderers.WgpuRenderer(canvas, pixel_ratio=ratio)
        camera = pygfx.PerspectiveCamera(self.camera.fov)
        camera.set_state(self.camera.get_state())
        self._paint(renderer, camera, legend=legend)
        return np.asarray(renderer.snapshot())

    def save(self, path: str, size: Optional[tuple[int, int]] = None, *,
             pixel_ratio: Optional[float] = None, legend: bool = True,
             groups: Optional[Any] = None) -> str:
        """Write the figure to ``path``: PNG, or by extension ``.svg`` / ``.pdf``.

        ``pixel_ratio`` is the PNG's supersampling, or the mesh images' resolution in a
        vector file (default 2). ``legend=False`` drops the strip. ``groups`` applies to
        vector files only (see :meth:`save_svg`). PNGs are written without alpha.
        """
        if str(path).lower().endswith((".svg", ".pdf")):
            from .vector import save as save_vector

            return save_vector(self, path, size=size, legend=legend, groups=groups,
                               pixel_ratio=2.0 if pixel_ratio is None else pixel_ratio)
        try:
            from imageio import v3 as iio
        except ImportError as exc:                              # pragma: no cover
            raise ImportError(
                "saving needs imageio: pip install 'neu-draw[render]'. "
                "`snapshot()` returns the array if you would rather write it "
                "yourself.") from exc

        iio.imwrite(path, self.snapshot(size, pixel_ratio=pixel_ratio,
                                        legend=legend)[..., :3])
        return path

    def save_svg(self, path: str, **kwargs) -> str:
        """Write the view as SVG, laid out so Illustrator keeps the named groups. Each mesh
        is a transparent image; points, skeletons and the legend are vectors.

        ``groups`` maps a drawable to a tuple of group names to nest it under, e.g.
        ``lambda d: (d.name.split(" : ")[0],)``; defaults to ``scene.export_groups``. See
        :mod:`neu_draw.backends.vector` for what is lost (occlusion between groups).
        """
        if not str(path).lower().endswith(".svg"):
            raise ValueError(f"save_svg writes .svg, not {path!r}")
        return self.save(path, **kwargs)

    def save_pdf(self, path: str, **kwargs) -> str:
        """The same figure as :meth:`save_svg`, as PDF, with the groups as nested layers
        in Acrobat's layer panel. Illustrator opens a PDF it did not write onto a single
        layer, so for editing there, prefer the SVG."""
        if not str(path).lower().endswith(".pdf"):
            raise ValueError(f"save_pdf writes .pdf, not {path!r}")
        return self.save(path, **kwargs)

    def orbit(self, directory: str, **kwargs) -> Any:
        """Render the camera turning about an axis, as numbered PNGs in ``directory``.

        Keywords: ``degrees`` (360), ``seconds`` (12), ``fps`` (30), ``axis`` (``"up"``;
        also ``right``/``view``, data ``x``/``y``/``z``, or an xyz vector), ``light``
        (``"camera"`` turns the lights with it, ``"fixed"`` does not), ``ease``
        (``linear``/``in-out``), ``size``, ``supersample`` (2), ``legend``, ``overwrite``,
        and ``encode`` (``"auto"``: write an mp4 if an ffmpeg is found via ``ffmpeg=``,
        ``$NEU_DRAW_FFMPEG`` or ``PATH``), ``output`` (the mp4's path). Returns an
        :class:`~neu_draw.video.Orbit` with the video, or the command that would make it.
        The view's own camera is not moved.
        """
        from .animation import orbit

        return orbit(self, directory, **kwargs)

    def close(self) -> None:
        """Close the canvas, recording where the camera was into ``views["last"]``.

        Done here, not in the toolbar, so ``view.close()`` from a cell records it too. A
        failure to record is ignored rather than raised.
        """
        try:
            self.save_view(LAST)
        except Exception:                                       # pragma: no cover
            pass
        self._closed = True
        self.canvas.close()

    # -- the legend ------------------------------------------------------------

    def _install_legend(self, legend: Optional[bool]) -> None:
        """Build the legend strip, unless it is turned off or there is nothing to label.

        ``None`` (the default) follows the scene's :class:`~neu_draw.scene.Legend`, which
        is visible by default. A scene with no labels gets no strip.
        """
        spec = self.scene_data.legend
        if legend is False or spec is None or not spec.visible:
            return
        if not self.scene_data.labels:
            return
        from .legend import LegendOverlay

        self.legend = LegendOverlay(self.scene_data, self.group, self.renderer)

    # -- the notebook toolbar --------------------------------------------------

    def _install_toolbar(self, toolbar: Any) -> None:
        """Attach a :class:`~neu_draw.toolbar.Toolbar`, if one is wanted and possible.

        ``"auto"`` (the default) adds one wherever possible, i.e. in a notebook.
        ``False`` gives the bare canvas; ``True`` raises if ipywidgets is missing.
        """
        if not toolbar:
            return
        from ..toolbar import attach

        self.ui = attach(self, required=toolbar != "auto")

    def _repr_mimebundle_(self, *args, **kwargs):
        """Let Jupyter display the view when it is the cell's value.

        Shows the toolbar's widget if there is one, else the canvas, else a text repr (an
        offscreen canvas). Positional arguments are dropped, since the Jupyter canvas
        accepts keywords only.
        """
        widget = getattr(self.ui, "widget", None)
        bundle = getattr(widget if widget is not None else self.canvas,
                         "_repr_mimebundle_", None)
        if bundle is None:
            return {"text/plain": repr(self)}
        return bundle(**kwargs)

    def __repr__(self) -> str:
        return f"View({self.scene_data!r})"


def show(scene: Scene, size: tuple[int, int] = DEFAULT_SIZE, **kwargs) -> View:
    """Render a scene and return the :class:`View`. The one call a notebook needs."""
    return View(scene, size=size, **kwargs)


def _is_offscreen(canvas: Any) -> bool:
    return type(canvas).__module__.endswith("offscreen")


def in_notebook() -> bool:
    """True inside an IPython kernel (a notebook), not a terminal REPL."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and hasattr(shell, "kernel")


def _make_canvas(size: tuple[int, int], kind: str = "auto") -> Any:
    """Pick a canvas: ``auto``, ``jupyter``, ``offscreen``, or ``gui``.

    ``auto`` is jupyter in a notebook and offscreen elsewhere; not ``rendercanvas.auto``,
    which would open desktop windows and render at the display's pixel ratio. Pass
    ``kind="gui"`` for a desktop window.
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


def _visible_sphere(group: pygfx.Group) -> Optional[tuple[float, float, float, float]]:
    """A bounding sphere over the group's **visible** children, in world xyz.

    Needed because ``get_bounding_box`` ignores ``visible``. ``None`` when nothing is
    showing.
    """
    boxes = [child.get_world_bounding_box() for child in group.children if child.visible]
    boxes = [box for box in boxes if box is not None]
    if not boxes:
        return None
    lo = np.min([box[0] for box in boxes], axis=0)
    hi = np.max([box[1] for box in boxes], axis=0)
    center = (lo + hi) / 2.0
    radius = float(np.linalg.norm(hi - lo) / 2.0) or 1.0
    return (float(center[0]), float(center[1]), float(center[2]), radius)


def _bounding_sphere(box) -> tuple[float, float, float, float]:
    """A zyx :class:`~neu_lib.BBox` as the ``(x, y, z, radius)`` pygfx wants.

    An empty box becomes a unit sphere at the origin.
    """
    if box.is_empty():
        return (0.0, 0.0, 0.0, 1.0)
    lo = np.asarray(box.lo, dtype=float)[::-1]          # zyx -> xyz
    hi = np.asarray(box.hi, dtype=float)[::-1]
    center = (lo + hi) / 2.0
    radius = float(np.linalg.norm(hi - lo) / 2.0) or 1.0
    return (float(center[0]), float(center[1]), float(center[2]), radius)
