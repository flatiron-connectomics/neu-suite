"""Turn a :class:`~neu_draw.scene.Scene` into pygfx objects and draw them.

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
object per body** whatever its topology. A renderer wanting a list of polylines instead
would need one graphic per branch — hundreds or thousands for a fragmented body.

Volume drawables are not implemented; see :class:`~neu_draw.scene.VolumeDrawable`.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import pygfx

from neu_lib import to_xyz
from ..scene import LinesDrawable, MeshDrawable, PointsDrawable, Scene
from ..viewstate import LAST, SAVED, ViewState, views

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
        obj = _build_one(drawable)
        # A hidden drawable is BUILT and switched off, not skipped. It used to be skipped,
        # which is cheaper and was fine while nothing could change its mind — but a legend
        # whose rows toggle visibility needs an object to toggle, and an entry with nothing
        # behind it is one nobody can turn back on. `frame()` filters on `visible` for the
        # same reason: `get_bounding_box` does not.
        obj.visible = bool(drawable.visible)
        if drawable.name is not None:
            obj.name = str(drawable.name)
        # The drawable's offset becomes the object's own transform rather than being added
        # to its vertices: the geometry keeps saying where the tissue is, and re-placing a
        # large mesh costs nothing. `.local.position` is **xyz**, like every other
        # coordinate handed to pygfx.
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
    # In the CONSTRUCTOR: assigning `geometry.normals` afterwards stores the bare
    # ndarray rather than wrapping it in a `Buffer`, so it never reaches the GPU.
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


def display_color(drawable: Any, override: Optional[tuple] = None) -> tuple:
    """The colour a drawable's material carries: its own, times its ``alpha``.

    ``alpha`` multiplies whatever the colour already carried rather than replacing it,
    so a per-drawable alpha and an ``#rrggbbaa`` colour compose instead of one silently
    winning.

    ``override`` substitutes a different **hue** while keeping that arithmetic — which is
    what the legend's highlight needs, and the reason this is a named function rather than
    two lines inline. A highlight that also reset the alpha would turn a translucent
    surface opaque, and the surface being translucent is often why you cannot find it.
    """
    r, g, b, a = drawable.color if override is None else override
    return (r, g, b, a * float(drawable.alpha))


def _material_kwargs(drawable: Any) -> dict:
    """Colour for a new material. ``alpha_mode`` is left at pygfx's ``auto``: it picks
    depth-write behaviour from the alpha, which is right for translucent surfaces."""
    return {"color": display_color(drawable)}


class View:
    """A rendered scene: canvas, renderer, camera and controller, kept together.

    Displays itself in Jupyter. Held as an object rather than returned as a tuple
    because the canvas must outlive the call — a garbage-collected canvas is a blank
    output cell, which reads as a rendering bug.
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
        # What "reset" goes back to. Captured before anything can be toggled, because
        # `Scene` is mutable and a scene may deliberately arrive with something hidden —
        # so "everything visible" is not the same thing as "how this opened".
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

        # A phong surface is black without light. Ambient alone flattens a mesh to a
        # silhouette, which is the failure that reads as "my mesh did not load".
        self.scene.add(pygfx.AmbientLight(intensity=0.6))
        directional = pygfx.DirectionalLight(intensity=2.5)
        directional.local.position = (-1, -1, 1)
        self.scene.add(directional)

        self.group = build(scene)
        self.scene.add(self.group)

        self._install_legend(legend)

        self.camera = pygfx.PerspectiveCamera(50)
        # Registered on the **main viewport**, not the renderer, when there is a legend:
        # a controller starts a drag only for events inside the viewport it was given, so
        # this is the whole of what keeps a click on a legend row from also spinning the
        # camera. See `backends/legend.py` for why a pointer handler could not do it.
        self.controller = pygfx.TrackballController(
            self.camera,
            register_events=self.legend.main if self.legend else self.renderer)
        self.frame()
        if viewpoint is not None:
            self.restore_view(viewpoint)
        # The viewpoint `reset()` returns to — after the fit and after any `viewpoint`, so
        # it is what the figure actually opened showing rather than what it would have.
        self._opening = ViewState(camera=dict(self.camera.get_state()),
                                  size=self.logical_size())

        if scene.axes_visible:
            self.scene.add(pygfx.AxesHelper(size=_extent(scene) * 0.2 or 1.0))

        self.canvas.request_draw(self._draw)
        # Ask to be woken when the scene changes, so `scene.relabel(...)` in a cell repaints
        # without anyone calling anything. The notification carries nothing: it schedules a
        # frame, and the frame re-reads the scene (`LegendOverlay.sync`). That split is what
        # keeps `scene.py` free of any notion of a renderer.
        scene.on_change(self.request_draw)
        self._install_toolbar(toolbar)

    # -- camera ----------------------------------------------------------------

    def frame(self) -> "View":
        """Fit the camera to what is **visible**, honouring the scene's ``Camera``.

        Always a bounding **sphere** ``(x, y, z, radius)``, computed from the built
        objects' *world* bounding boxes — so the fit follows their transforms rather than
        the scene's nm boxes, which is what an offset drawable needs, and it can be
        computed for a case ``show_object`` cannot take at all (an empty scene has no
        bounding sphere and would raise).

        **Hidden objects are excluded, and this is the part that is easy to get wrong.**
        ``get_bounding_box`` walks every child regardless of ``visible``, so handing
        ``show_object`` the whole group would frame a body the legend has just switched
        off — the camera pulls back for something nobody can see. That used not to matter
        because a hidden drawable was never built.

        The fit is **conservative**: pygfx frames the bounding sphere, so a long thin
        neurite leaves slack around its short axes. That is the safe direction to err —
        nothing is ever cropped out — and ``Camera.zoom`` is the way in.
        """
        intent = self.scene_data.camera
        if intent.target is not None:
            target = _bounding_sphere(intent.target)
        else:
            target = _visible_sphere(self.group)
            if target is None:
                target = _bounding_sphere(self.scene_data.bbox)

        # The scene renders into the rect left over beside the legend, and a perspective
        # camera's fit depends on that rect's aspect. Setting it first means the very
        # first frame is framed for where it will actually be drawn, rather than for the
        # whole canvas and then corrected.
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
        """Re-fit the camera and redraw — :meth:`frame` plus the repaint.

        The one action a "centre view" button needs, and the reason it is separate is
        that :meth:`frame` runs during construction, before there is a draw to request.

        Note this fits **what is visible now**, so after hiding half the bodies it frames
        the remainder. :meth:`reset` is the one that goes back to the opening picture.
        """
        self.frame()
        self.request_draw()
        return self

    def reset(self) -> "View":
        """Back to the view this figure opened with.

        Un-hides whatever was hidden, drops every highlight, and returns to the camera the
        figure opened at — which is not necessarily a fit, since ``viewpoint=`` may have
        placed it somewhere else.

        **Colours are deliberately left alone.** Visibility and highlights are transient
        exploration, so undoing them is a convenience; ``legend.recolor`` is an authored
        change to the scene, and a button that silently reverted it would be destroying
        work rather than tidying up.
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

        Index correspondence, because :func:`build` makes exactly one object per drawable
        in order — **including the unnamed and unlabelled ones**, which are on no legend row
        and so would be missed by going through the legend. Any highlight override is
        re-applied afterwards by ``legend.refresh()``, which runs second for that reason.
        """
        for drawable, obj in zip(self.scene_data.drawables, self.group.children):
            obj.material.color = display_color(drawable)
            obj.visible = bool(drawable.visible)

    # -- viewpoints ------------------------------------------------------------

    def logical_size(self) -> tuple[int, int]:
        """The canvas size in logical pixels, falling back to the size asked for.

        **A Jupyter canvas reports (1, 1) until the browser has laid the widget out**, so
        the live value is a placeholder in a freshly executed notebook — the same trap
        :meth:`snapshot` documents. Anything below a few pixels is therefore treated as
        "not yet known" rather than believed.
        """
        try:
            width, height = self.canvas.get_logical_size()
        except Exception:                                       # pragma: no cover
            return self._size
        return (int(width), int(height)) if width > 4 and height > 4 else self._size

    def save_view(self, name: str = SAVED) -> ViewState:
        """Record this camera into :data:`neu_draw.views`, and return what was stored.

        The store outlives the view (see :mod:`neu_draw.viewstate`), so the saved angle
        is available to the next figure — which is the whole reason to save one.
        """
        state = ViewState(camera=dict(self.camera.get_state()), size=self.logical_size())
        views[name] = state
        return state

    def restore_view(self, name: Union[str, ViewState] = SAVED, *,
                     size: bool = True) -> Optional[ViewState]:
        """Apply a saved viewpoint. ``None`` — not an error — if that slot is empty.

        An empty slot is the ordinary state of a fresh session, and a button that raises
        the first time it is pressed is worse than one that says nothing was saved. It is
        also why ``show(scene, viewpoint="last")`` is safe on the first cell of a session:
        no ``last`` yet just means the figure keeps its own framing.

        ``name`` may be a :class:`~neu_draw.viewstate.ViewState` outright, so a viewpoint
        can be held in an ordinary variable rather than a named slot.

        ``size`` also restores the canvas size, because the aspect ratio is part of what
        a viewpoint means; pass ``size=False`` to keep the canvas as it is and accept a
        differently framed version of the same angle.
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

        A no-op once closed. Worth guarding rather than letting it raise: this is a
        scene-change listener, and a scene outlives the views built from it — so editing a
        scene after closing its figure is ordinary, not a mistake.
        """
        if self._closed:
            return
        self.canvas.request_draw()

    def refresh(self) -> "View":
        """Re-read the scene and repaint. The "catch up now" button.

        Rarely needed: the mutating methods on :class:`~neu_draw.scene.Scene` schedule a
        repaint themselves, and every frame re-reads the scene anyway. What it is for is the
        route neither of those catches at the moment you want it — a field set directly
        (``scene.get("a").label = "Tm2"``, ``.visible = False``) in a context where nothing
        else is going to draw a frame.
        """
        self._sync_objects()
        if self.legend is not None:
            self.legend.refresh()
        self.request_draw()
        return self

    def _draw(self) -> None:
        self._paint(self.renderer, self.camera)

    def _paint(self, renderer: Any, camera: Any) -> None:
        """One frame: the scene, then the legend strip beside it.

        Two ``render`` calls into two rects, with only the second flushing. pygfx clears
        on the **first** render since a flush and not afterwards, so the pair composes
        without either half knowing about the other.

        ``renderer`` is a parameter because ``_offscreen_snapshot`` builds one of its own
        at an exact size, and the legend has to go through that pass too — a saved figure
        without its legend is not the figure.
        """
        if self.legend is None:
            renderer.render(self.scene, camera)
            return
        main, strip = self.legend.rects_for(renderer.logical_size)
        renderer.render(self.scene, camera, rect=main, flush=False)
        self.legend.draw(renderer, rect=strip)

    @property
    def pixel_ratio(self) -> float:
        """Internal pixels per logical pixel. ``>= 2`` by default — see the constructor."""
        return float(self.renderer.pixel_ratio)

    def snapshot(self, size: Optional[tuple[int, int]] = None) -> np.ndarray:
        """Render once and return the pixels as ``(h, w, 4)`` uint8.

        **On a Jupyter canvas this renders through a separate offscreen pass**, at the
        size the view was asked for, rather than reading the live framebuffer. That
        framebuffer is sized by whatever the *browser* reported, so before the widget
        has been displayed and laid out it is a placeholder — measured, a view created
        at ``(900, 700)`` snapshots as **2x2** in a freshly executed notebook, and
        ``save()`` writes that 2x2 image without complaint. Saving a figure should not
        depend on a browser having painted it.

        **The result is `pixel_ratio` times the requested size**, not the requested size:
        that is pygfx's supersampled internal texture, and it is where the antialiasing
        comes from. Construct the view with ``pixel_ratio=1.0`` for pixel-exact output.
        """
        if size is None and _is_offscreen(self.canvas):
            self._paint(self.renderer, self.camera)
            return np.asarray(self.renderer.snapshot())
        return self._offscreen_snapshot(tuple(size) if size else self._size)

    def _offscreen_snapshot(self, size: tuple[int, int]) -> np.ndarray:
        """Re-render this scene and camera at an exact size, off any live canvas."""
        from rendercanvas.offscreen import RenderCanvas as Offscreen

        canvas = Offscreen(size=size)
        renderer = pygfx.renderers.WgpuRenderer(canvas, pixel_ratio=self._pixel_ratio)
        camera = pygfx.PerspectiveCamera(self.camera.fov)
        camera.set_state(self.camera.get_state())
        self._paint(renderer, camera)
        return np.asarray(renderer.snapshot())

    def save(self, path: str, size: Optional[tuple[int, int]] = None) -> str:
        """Write a snapshot to a PNG. Alpha is dropped — a figure wants a flat image."""
        try:
            from imageio import v3 as iio
        except ImportError as exc:                              # pragma: no cover
            raise ImportError(
                "saving needs imageio: pip install 'neu-draw[render]'. "
                "`snapshot()` returns the array if you would rather write it "
                "yourself.") from exc

        iio.imwrite(path, self.snapshot(size)[..., :3])
        return path

    def close(self) -> None:
        """Close the canvas, recording where the camera was into ``views["last"]``.

        **Here rather than in the toolbar**, so that `view.close()` from a notebook counts
        too — the slot is only useful if it is reliably populated, and "I closed the figure
        and now want that angle back" does not depend on which route closed it.

        The record happens first: after the canvas is gone there is nothing to ask. It is
        also guarded, because a `close()` that raises on the way out is worse than a lost
        viewpoint — this runs in every test's `finally`.
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

        ``None`` — the default — follows the scene's own
        :class:`~neu_draw.scene.Legend`, whose ``visible`` has defaulted to ``True`` since
        the field existed; the backend ignored it until there was a legend to draw. So a
        scene from :func:`~neu_draw.scene.build_scene` gets one without asking, which is
        the same argument as the toolbar's default: a legend you have to remember to
        request is one that is usually missing.

        **A scene with nothing to label gets no strip**, rather than an empty one taking
        45% of the canvas. "Nothing to label" is no *labels* rather than no names, since a
        drawable may carry a label and no name.
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

        ``"auto"`` — the default — means **a toolbar wherever one can exist**, so a
        notebook gets the buttons without anyone having to ask for them, and an offscreen
        render is unaffected. That follows the package's existing rule for store logging:
        a user should not have to invoke anything to get the usable behaviour, because
        forgetting the call after a kernel restart looks exactly like the feature being
        broken. Pass ``toolbar=False`` for the bare canvas, or ``True`` to insist and get
        the exception if ipywidgets is missing.

        Imported here rather than at module scope: ipywidgets is a notebook dependency,
        and the backend already loads on a worker with no front-end at all.
        """
        if not toolbar:
            return
        from ..toolbar import attach

        self.ui = attach(self, required=toolbar != "auto")

    def _repr_mimebundle_(self, *args, **kwargs):
        """Let Jupyter display the view when it is the cell's value.

        The **toolbar's** widget where there is one, and the bare canvas otherwise — so
        wrapping the canvas in a button bar changes nothing about how a view is shown.

        Positional arguments are **dropped**: IPython passes ``include``/``exclude`` as
        keywords, but the Jupyter canvas accepts ``**kwargs`` only, so forwarding
        anything positional is a ``TypeError`` in the one place it matters — the cell
        that was supposed to show the figure. An offscreen canvas has no bundle at all,
        which is why this degrades to a repr rather than raising.
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


def _visible_sphere(group: pygfx.Group) -> Optional[tuple[float, float, float, float]]:
    """A bounding sphere over the group's **visible** children, in world xyz.

    ``WorldObject.get_bounding_box`` deliberately ignores ``visible`` — it answers "how
    much space could this take up", not "what is on screen" — so this is the filter, and
    it is the only thing standing between a legend toggle and a camera that pulls back
    for a body that is switched off. ``None`` when nothing is showing.
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
    """A zyx :class:`~neu_vol.BBox` as the ``(x, y, z, radius)`` pygfx wants.

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
