"""Buttons above a rendered view in a notebook, as ipywidgets.

The buttons: **Center** the camera on what is visible, **Reset** to the opening view,
**Refresh** after editing the scene in a cell, **Save** a viewpoint and **Restore** it, go
to where the **Last** closed figure was, **Capture** a PNG, SVG or PDF, **Animate** an
orbit, and **Close** the figure, leaving its last image in place of the canvas. Capture
and Animate read collapsible options panels.

Save/Restore and Last are separate slots on purpose ("the angle I chose" against "where I
was"). ``View.close`` writes the ``last`` slot, so re-running a cell and pressing Last, or
``show(scene, viewpoint="last")``, brings the angle back.

The buttons sit outside the canvas, so they are not part of the rendered image; the
legend is drawn inside it (:mod:`neu_draw.backends.legend`). The capture path is
pre-filled as ``<prefix>_<timestamp>.png``, the prefix being the last one used.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from .viewstate import LAST, SAVED

#: ``strftime`` pattern for the timestamp in a capture filename; sorts lexicographically.
TS_FORMAT = "%Y-%m-%d_%H-%M-%S"

DEFAULT_PREFIX = "snapshot"

#: What Capture can write, keyed by extension; the path's extension decides.
FORMATS = {
    "png": "PNG image",
    "svg": "SVG — layered, for Illustrator",
    "pdf": "PDF — layered, for Acrobat",
}

#: The stem of the last path captured through any toolbar, for the next default.
last_prefix: Optional[str] = None
#: …and its extension, so capturing one ``.svg`` keeps the next default an ``.svg``.
last_suffix: str = ".png"


def default_path(prefix: Optional[str] = None) -> str:
    """The pre-filled path, ``<prefix>_<timestamp>.png`` (or the last extension used)."""
    stem = prefix or last_prefix or DEFAULT_PREFIX
    return f"{stem}_{datetime.now().strftime(TS_FORMAT)}{last_suffix}"


def remember_prefix(path: str) -> str:
    """Record ``path``'s stem as the next default, dropping a trailing timestamp so they do
    not accumulate."""
    global last_prefix, last_suffix
    base = path.rsplit("/", 1)[-1]
    stem = base.rsplit(".", 1)[0]
    if "." in base and base.rsplit(".", 1)[1].lower() in FORMATS:
        last_suffix = "." + base.rsplit(".", 1)[1].lower()
    parts = stem.split("_")
    if len(parts) > 2:
        try:
            datetime.strptime("_".join(parts[-2:]), TS_FORMAT)
        except ValueError:
            pass
        else:
            stem = "_".join(parts[:-2])
    last_prefix = stem or DEFAULT_PREFIX
    return last_prefix


class Toolbar:
    """The button bar and the canvas, as one widget.

    ``view`` is anything with ``canvas``, ``center()``, ``save_view()``,
    ``restore_view()``, ``save()``, ``snapshot()``, ``logical_size()`` and ``close()``, so
    this module needs no pygfx. Raises ``TypeError`` if the canvas is not an ipywidget
    (an offscreen or desktop canvas).
    """

    def __init__(self, view: Any, *, path: Optional[str] = None):
        import ipywidgets as widgets

        canvas = view.canvas
        if not isinstance(canvas, widgets.Widget):
            raise TypeError(
                f"a toolbar needs a canvas that is an ipywidget, and this one is a "
                f"{type(canvas).__name__}. Use canvas='jupyter' (the default in a "
                f"notebook); an offscreen or desktop canvas has no widget to sit above.")

        self.view = view
        self._widgets = widgets
        self._closed = False

        self.path = widgets.Text(
            value=path or default_path(),
            placeholder="snapshot.png",
            layout=widgets.Layout(width="260px"),
            tooltip="where 'capture' writes: .png for an image, .svg for vector points, "
                    "lines and legend with one image layer per mesh")
        self.status = widgets.HTML(value="")

        self._buttons = {
            "center": self._button("Center", "crosshairs",
                                   "fit the camera to everything visible", self._center),
            "reset": self._button("Reset", "undo-alt",
                                  "back to the view this figure opened with: everything "
                                  "shown, no highlights", self._reset),
            "refresh": self._button("Refresh", "sync",
                                    "re-read the scene — for a field you set directly, "
                                    "like scene.get(name).label", self._refresh),
            "save": self._button("Save", "bookmark",
                                 f"remember this viewpoint as views[{SAVED!r}], for this "
                                 f"figure or any later one", self._save_view),
            "restore": self._button("Restore", "undo",
                                    f"go to the viewpoint saved in views[{SAVED!r}]",
                                    self._restore_view),
            "last": self._button("Last", "history",
                                 f"go to where the last CLOSED figure was — "
                                 f"views[{LAST!r}], written on the way out",
                                 self._restore_last),
            "capture": self._button("Capture", "camera",
                                    "write the file named in the box below — a .png, or "
                                    "a layered .svg", self._capture),
            "animate": self._button("Animate", "film",
                                    "render an orbit to PNG frames, as set in Animation "
                                    "options, and encode it where ffmpeg is available",
                                    self._animate),
            "close": self._button("Close", "power-off",
                                  "close the canvas, leaving its last image behind",
                                  self._close),
        }

        self.sliders = self._opacity_sliders()
        self.options = self._capture_options()
        self.animation = self._animation_options()

        # The canvas sits in a box of its own so closing can swap it for the snapshot.
        self._stage = widgets.Box([canvas])
        # Buttons wrap rather than clip on a narrow canvas; the path box has its own row.
        self.widget = widgets.VBox([
            widgets.HBox(list(self._buttons.values()),
                         layout=widgets.Layout(flex_flow="row wrap", width="100%")),
            widgets.HBox([self.path, self.status]),
            widgets.HBox([self.options["panel"], self.animation["panel"]],
                         layout=widgets.Layout(flex_flow="row wrap", width="100%")),
            *([widgets.HBox(list(self.sliders.values()),
                            layout=widgets.Layout(flex_flow="row wrap", width="100%"))]
              if self.sliders else []),
            self._stage,
        ])

    # -- construction helpers --------------------------------------------------

    def _button(self, label: str, icon: str, tooltip: str, handler) -> Any:
        """A button with a word as well as an icon: the Font Awesome icons depend on the
        notebook front end, and may not render."""
        widgets = self._widgets
        button = widgets.Button(description=label, icon=icon, tooltip=tooltip,
                                layout=widgets.Layout(width="98px", flex="0 0 auto"))
        button.on_click(lambda _button: self._guarded(handler))
        return button

    def _capture_options(self) -> dict:
        """The collapsed "Capture options" panel: format, resolution, size, legend.

        The path's extension decides the format, and the dropdown edits it: choosing
        ``svg`` rewrites the extension, and typing ``fig.pdf`` moves the dropdown.
        ``resolution`` is pixels per logical pixel (a PNG's supersampling, or a vector
        file's mesh images). ``width`` and ``height`` start at the canvas size; changing
        them renders the same camera into a different frame.
        """
        widgets = self._widgets
        width, height = self.view.logical_size()
        style = {"description_width": "80px"}
        wide = widgets.Layout(width="330px")
        fmt = widgets.Dropdown(options=[(label, ext) for ext, label in FORMATS.items()],
                               value=self._extension(self.path.value) or "png",
                               description="format", style=style, layout=wide)
        resolution = widgets.BoundedFloatText(value=2.0, min=0.5, max=8.0, step=0.5,
                                              description="resolution ×", style=style,
                                              layout=widgets.Layout(width="180px"))
        w = widgets.BoundedIntText(value=int(width), min=16, max=16384, description="width",
                                   style=style, layout=widgets.Layout(width="180px"))
        h = widgets.BoundedIntText(value=int(height), min=16, max=16384,
                                   description="height", style=style,
                                   layout=widgets.Layout(width="180px"))
        legend = widgets.Checkbox(value=True, description="legend", indent=False)
        reset = widgets.Button(description="canvas size", icon="expand",
                               tooltip="set width and height back to the canvas's",
                               layout=widgets.Layout(width="130px"))

        syncing = {"on": False}

        def on_format(change) -> None:
            if syncing["on"]:
                return
            syncing["on"] = True
            try:
                stem = self.path.value.strip()
                if self._extension(stem):
                    stem = stem.rsplit(".", 1)[0]
                self.path.value = f"{stem}.{change['new']}"
            finally:
                syncing["on"] = False

        def on_path(change) -> None:
            ext = self._extension(change["new"])
            if syncing["on"] or ext is None or ext == fmt.value:
                return
            syncing["on"] = True
            try:
                fmt.value = ext
            finally:
                syncing["on"] = False

        def on_reset(_button) -> None:
            w.value, h.value = (int(v) for v in self.view.logical_size())

        fmt.observe(on_format, names="value")
        self.path.observe(on_path, names="value")
        reset.on_click(lambda b: self._guarded(lambda: on_reset(b)))

        body = widgets.VBox([
            fmt,
            widgets.HBox([resolution, legend]),
            widgets.HBox([w, h, reset]),
        ])
        panel = widgets.Accordion(children=[body], selected_index=None)
        panel.set_title(0, "Capture options")
        return {"panel": panel, "format": fmt, "resolution": resolution,
                "width": w, "height": h, "legend": legend}

    def _animation_options(self) -> dict:
        """The collapsed "Animation options" panel, read by the Animate button.

        Size and legend come from Capture options. The frames folder defaults to the
        capture path's stem plus ``_orbit``, so a figure and its animation sit together.
        """
        from .backends.animation import LIGHTS
        from .orbit import DATA_AXES, EASES, SCREEN_AXES

        widgets = self._widgets
        style = {"description_width": "80px"}
        short = widgets.Layout(width="180px")
        wide = widgets.Layout(width="330px")
        axes = ([(f"screen: {a}", a) for a in SCREEN_AXES]
                + [(f"data: {a}", a) for a in DATA_AXES])
        controls = {
            "degrees": widgets.FloatText(value=360.0, description="degrees", style=style,
                                         layout=short),
            "seconds": widgets.BoundedFloatText(value=12.0, min=0.1, max=3600.0,
                                                description="seconds", style=style,
                                                layout=short),
            "fps": widgets.BoundedIntText(value=30, min=1, max=240, description="fps",
                                          style=style, layout=short),
            "axis": widgets.Dropdown(options=axes, value="up", description="axis",
                                     style=style, layout=short),
            "light": widgets.Dropdown(
                options=[("turns with camera", "camera"), ("fixed in scene", "fixed")],
                value=LIGHTS[0], description="light", style=style, layout=wide),
            "ease": widgets.Dropdown(options=list(EASES), value="linear",
                                     description="ease", style=style, layout=short),
            "supersample": widgets.BoundedIntText(value=2, min=1, max=4,
                                                  description="smoothing ×", style=style,
                                                  layout=short),
            "folder": widgets.Text(value="", placeholder="<capture name>_orbit",
                                   description="frames in", style=style, layout=wide),
            "overwrite": widgets.Checkbox(value=False, description="overwrite frames",
                                          indent=False),
            "encode": widgets.Checkbox(value=True, description="encode to mp4", indent=False,
                                       tooltip="needs an ffmpeg: on PATH, or named by "
                                               "$NEU_DRAW_FFMPEG; without one, the frames "
                                               "are written and the command is shown"),
            "video": widgets.Text(value="", placeholder="<frames folder>.mp4",
                                  description="video", style=style, layout=wide),
        }
        note = widgets.HTML("<span style='color:#666'>size and legend: from Capture "
                            "options</span>")
        body = widgets.VBox([
            widgets.HBox([controls["degrees"], controls["seconds"], controls["fps"]]),
            widgets.HBox([controls["axis"], controls["ease"], controls["supersample"]]),
            controls["light"],
            widgets.HBox([controls["folder"], controls["overwrite"]]),
            widgets.HBox([controls["video"], controls["encode"]]),
            note,
        ])
        panel = widgets.Accordion(children=[body], selected_index=None)
        panel.set_title(0, "Animation options")
        return {"panel": panel, **controls}

    def _frames_folder(self) -> str:
        chosen = self.animation["folder"].value.strip()
        if chosen:
            return chosen
        stem = self.path.value.strip() or default_path()
        if self._extension(stem):
            stem = stem.rsplit(".", 1)[0]
        return f"{stem}_orbit"

    @staticmethod
    def _extension(path: str) -> Optional[str]:
        base = str(path).strip().rsplit("/", 1)[-1]
        ext = base.rsplit(".", 1)[1].lower() if "." in base else None
        return ext if ext in FORMATS else None

    def _opacity_sliders(self) -> dict:
        """One opacity slider per kind of drawable the scene holds, keyed by kind.

        A slider sets every drawable of its kind to one alpha; it starts at the largest
        current alpha. For one row, call ``view.set_alpha(a, names=[...])``. No sliders
        for a view without ``set_alpha``.
        """
        scene = getattr(self.view, "scene_data", None)
        if scene is None or not hasattr(self.view, "set_alpha"):
            return {}
        from .scene import DRAWABLE_KINDS

        widgets = self._widgets
        sliders = {}
        for kind in DRAWABLE_KINDS:
            members = scene.of_kind(kind)
            if not members:
                continue
            slider = widgets.FloatSlider(
                value=max(float(d.alpha) for d in members), min=0.0, max=1.0, step=0.02,
                description=f"{kind} α", readout_format=".2f", continuous_update=True,
                style={"description_width": "70px"},
                layout=widgets.Layout(width="260px"))
            slider.observe(
                lambda change, kind=kind: self._guarded(
                    lambda: self.view.set_alpha(change["new"], kind=kind)),
                names="value")
            sliders[kind] = slider
        return sliders

    def _guarded(self, handler) -> None:
        """Run a handler, showing any failure in the status line (a button callback's
        traceback otherwise goes to the kernel log, not the cell)."""
        try:
            handler()
        except Exception as exc:                                # noqa: BLE001
            self._say(f"{type(exc).__name__}: {exc}", error=True)

    def _say(self, text: str, error: bool = False) -> None:
        color = "#b00" if error else "#666"
        self.status.value = f"<span style='color:{color}'>{text}</span>"

    # -- the actions -----------------------------------------------------------

    def _center(self) -> None:
        self.view.center()
        self._say("centred on the visible drawables")

    def _reset(self) -> None:
        self.view.reset()
        self._say("back to the opening view — everything shown, no highlights "
                  "(colours are left as you set them)")

    def _refresh(self) -> None:
        """Re-read the scene. The scene's methods repaint on their own; this covers a field
        set directly on a drawable."""
        self.view.refresh()
        self._say(f"re-read the scene: {len(self.view.legend or ())} legend row(s)"
                  if self.view.legend is not None else "re-read the scene")

    def _save_view(self) -> None:
        state = self.view.save_view()
        self._say(f"viewpoint saved ({state.size[0]}x{state.size[1]}) — "
                  f"'Restore' brings it back, here or in any other figure")

    def _restore_view(self) -> None:
        if self.view.restore_view() is None:
            self._say("nothing saved yet — press 'Save' first", error=True)
        else:
            self._say("restored the saved viewpoint")

    def _restore_last(self) -> None:
        """Go to the viewpoint 'Close' recorded for the last closed figure."""
        if self.view.restore_view(LAST) is None:
            self._say("no closed figure to go back to yet — 'Close' records one",
                      error=True)
        else:
            self._say(f"restored views[{LAST!r}], where the last closed figure was")

    def _capture(self) -> None:
        path = self.path.value.strip()
        if not path:
            self._say("type a filename in the box first", error=True)
            return
        options = self.options
        written = self.view.save(
            path, size=(int(options["width"].value), int(options["height"].value)),
            pixel_ratio=float(options["resolution"].value),
            legend=bool(options["legend"].value))
        remember_prefix(written)
        self.path.value = default_path()
        self._say(f"wrote {written}")

    def _animate(self) -> None:
        """Render the orbit the Animation options describe, with per-frame progress in the
        status line (the kernel is busy throughout)."""
        a, c = self.animation, self.options
        folder = self._frames_folder()

        def progress(done: int, total: int) -> None:
            self._say(f"rendering {folder}: frame {done} / {total}"
                      + (" — encoding…" if done == total else ""))

        result = self.view.orbit(
            folder, degrees=float(a["degrees"].value), seconds=float(a["seconds"].value),
            fps=float(a["fps"].value), axis=a["axis"].value, light=a["light"].value,
            ease=a["ease"].value, supersample=int(a["supersample"].value),
            size=(int(c["width"].value), int(c["height"].value)),
            legend=bool(c["legend"].value), overwrite=bool(a["overwrite"].value),
            encode="auto" if a["encode"].value else False,
            output=a["video"].value.strip() or None, progress=progress)
        import html as _html

        from .video import FFMPEG_ENV

        self.last_orbit = result
        if result.video:
            self._say(f"{result.frames} frames in {folder}; wrote {result.video}")
        elif a["encode"].value:
            self._say(f"{result.frames} frames in {folder}. No ffmpeg found — put one on "
                      f"PATH or set ${FFMPEG_ENV} and it runs itself; or encode with: "
                      f"<code>{_html.escape(result.command())}</code>")
        else:
            self._say(f"{result.frames} frames in {folder}. Encode with: "
                      f"<code>{_html.escape(result.command())}</code>")

    def _close(self) -> None:
        """Close the canvas and leave the image it last showed in its place.

        ``View.close`` records the viewpoint in :data:`~neu_draw.viewstate.LAST`, so the
        next figure's 'Last' button, or ``show(scene, viewpoint="last")``, reopens on it.
        """
        if self._closed:
            self._say("already closed")
            return

        image = self.view.snapshot()
        size = self.view.logical_size()
        self.view.close()
        self._closed = True

        self._stage.children = (self._still(image, size),)
        # Every control off: there is no canvas left to act on.
        for button in self._buttons.values():
            button.disabled = True
        for slider in self.sliders.values():
            slider.disabled = True
        for panel in (self.options, self.animation):
            for key, control in panel.items():
                if key != "panel":
                    control.disabled = True
        self.path.disabled = True
        self._say(f"closed. The viewpoint is in views[{LAST!r}] — press 'Last' in the next "
                  f"figure, or open it with show(scene, viewpoint='{LAST}')")

    def _still(self, image, size: tuple[int, int]) -> Any:
        """The snapshot as an ``Image`` widget, or a note saying why not.

        Shown at the canvas's logical size, since the snapshot is ``pixel_ratio`` times
        larger.
        """
        widgets = self._widgets
        try:
            from imageio import v3 as iio
        except ImportError:
            return widgets.HTML(
                "<i>closed. Install imageio to keep the last image here:"
                " pip install 'neu-draw[render]'</i>")

        png = iio.imwrite("<bytes>", image[..., :3], extension=".png")
        layout = widgets.Layout(width=f"{int(size[0])}px") if size[0] else widgets.Layout()
        return widgets.Image(value=png, format="png", layout=layout)

    def _repr_mimebundle_(self, *args, **kwargs):
        """Display as the widget, so the toolbar can be the cell's value too."""
        return self.widget._repr_mimebundle_(**kwargs)


def attach(view: Any, *, required: bool = True) -> Optional[Toolbar]:
    """A :class:`Toolbar` for ``view``, or ``None`` when one cannot exist.

    With ``required=False`` (``toolbar="auto"``), missing ipywidgets or a canvas that is
    not a widget returns ``None``; otherwise it raises.
    """
    try:
        return Toolbar(view)
    except (ImportError, TypeError):
        if required:
            raise
        return None
