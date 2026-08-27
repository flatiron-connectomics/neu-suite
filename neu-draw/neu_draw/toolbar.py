"""Buttons above a rendered view, in a notebook. ipywidgets, not an in-canvas UI.

Seven actions: **centre** the camera on what is visible, **reset** to the view the figure
opened with, **save** a viewpoint and go back to it (**restore**) or to wherever the
**last** closed figure was, **capture** a PNG, and **close** the figure — replacing the
live canvas with the image it last showed, so the notebook keeps a picture where the widget
was.

**Two viewpoint slots, two buttons.** They answer different questions — "the angle I chose"
against "wherever I happened to be" — and a single button picking between them would leave
it unclear which you got. `Close` writes the `last` slot on the way out (in `View.close`, so
a `view.close()` from a cell counts too), which is what makes re-running a cell and pressing
`Last` reproduce the angle. `show(scene, viewpoint="last")` does it without the button.

## Why ipywidgets and not imgui

The predecessor drew its toolbar *inside* the canvas with imgui, because fastplotlib
already had an imgui render pass and a docked-subplot layout manager to hang it off.
Neither exists here, and reproducing them would be the larger part of the work. Against
that:

* **A dead widget cannot host its own replacement.** "Close the figure and leave the
  snapshot behind" needs somewhere outside the canvas to put the image; the predecessor
  had to reach for a matplotlib figure to have a container at all. An ipywidgets ``VBox``
  *is* that container, so the swap is one assignment and needs no plotting library.
* **Text entry in an in-canvas UI is a browser bug away from unusable.** The
  predecessor's save dialog needed a runtime string-patch of jupyter_rfb's frontend
  JavaScript, because its ``input`` handler named an attribute that does not exist and
  threw before dispatching typed characters — backspace worked, typing did not. A real
  ``Text`` widget has no such problem.
* It costs nothing when absent: ipywidgets is a notebook dependency, and outside a
  notebook there is no toolbar to want.

What is given up is that the buttons are **not part of the rendered image** — which is
right for buttons, and would be wrong for a legend. The legend is therefore drawn in the
canvas (see :mod:`neu_draw.backends.legend`), and the two live in different layers on
purpose.

## The one piece of remembered state

:data:`last_prefix`. Capturing a figure is something you do a dozen times in a row while
adjusting it, and typing the name each time is the friction that stops people doing it.
So the path box is pre-filled with ``<prefix>_<timestamp>.png``, and the prefix is
whatever you last typed — with a trailing timestamp stripped off first, or every capture
would grow another one.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from .viewstate import LAST, SAVED

#: ``strftime`` pattern for the generated part of a capture filename. Sorts
#: lexicographically, which is the only real requirement.
TS_FORMAT = "%Y-%m-%d_%H-%M-%S"

DEFAULT_PREFIX = "snapshot"

#: The stem of the last path captured through a toolbar, for the next one's default.
#: Module-level rather than per-toolbar: it is a preference about naming files, and it
#: should survive the view it was expressed in.
last_prefix: Optional[str] = None


def default_path(prefix: Optional[str] = None) -> str:
    """``<prefix>_<timestamp>.png`` — the pre-filled capture path."""
    stem = prefix or last_prefix or DEFAULT_PREFIX
    return f"{stem}_{datetime.now().strftime(TS_FORMAT)}.png"


def remember_prefix(path: str) -> str:
    """Record ``path``'s stem as the next default, dropping a trailing timestamp.

    Without the strip, capturing ``cell_2026-08-26_11-00-00.png`` would seed the next
    default with that whole string and produce ``cell_2026-08-26_11-00-00_2026-08-26_11-00-42.png``.
    """
    global last_prefix
    stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
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

    ``view`` is a rendered view — anything with ``canvas``, ``center()``, ``save_view()``,
    ``restore_view()``, ``save()``, ``snapshot()``, ``logical_size()`` and ``close()``.
    Duck-typed rather than importing the backend, so this module stays free of pygfx and
    a test can drive it without a GPU.

    Raises ``TypeError`` if the canvas is not an ipywidget — an offscreen or desktop
    canvas has nothing to compose with, and silently returning a toolbar that displays
    nothing would be worse than saying so.
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
            tooltip="where 'capture' writes the PNG")
        self.status = widgets.HTML(value="")

        self._buttons = {
            "center": self._button("Center", "crosshairs",
                                   "fit the camera to everything visible", self._center),
            "reset": self._button("Reset", "sync",
                                  "back to the view this figure opened with: everything "
                                  "shown, no highlights", self._reset),
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
                                    "write the PNG named in the box below", self._capture),
            "close": self._button("Close", "power-off",
                                  "close the canvas, leaving its last image behind",
                                  self._close),
        }

        # The canvas sits in a box of its own so closing can swap it for the snapshot.
        # Replacing a child of the outer VBox would work too, but this keeps the bar's
        # position fixed and the swap a single-element assignment.
        self._stage = widgets.Box([canvas])
        # Seven buttons no longer fit beside a path box, so the path moved to its own row
        # with the status line. `row wrap` because a narrow canvas should stack the buttons
        # rather than clip the last two — and it is the last two, Capture and Close, that
        # you would most notice missing.
        self.widget = widgets.VBox([
            widgets.HBox(list(self._buttons.values()),
                         layout=widgets.Layout(flex_flow="row wrap", width="100%")),
            widgets.HBox([self.path, self.status]),
            self._stage,
        ])

    # -- construction helpers --------------------------------------------------

    def _button(self, label: str, icon: str, tooltip: str, handler) -> Any:
        """A button carrying **both** a word and an icon.

        The icon names are Font Awesome, which is a font the notebook front-end supplies
        rather than something this package can guarantee — so an icon-only bar is five
        blank squares wherever it is missing, with no way to tell that from a broken
        toolbar. The word always renders.
        """
        widgets = self._widgets
        button = widgets.Button(description=label, icon=icon, tooltip=tooltip,
                                layout=widgets.Layout(width="98px", flex="0 0 auto"))
        button.on_click(lambda _button: self._guarded(handler))
        return button

    def _guarded(self, handler) -> None:
        """Run a handler, putting any failure in the status line.

        A button callback's traceback goes to the kernel log, not the cell — so an
        unguarded failure is a button that visibly does nothing, which is the hardest
        kind of bug to report.
        """
        try:
            handler()
        except Exception as exc:                                # noqa: BLE001
            self._say(f"{type(exc).__name__}: {exc}", error=True)

    def _say(self, text: str, error: bool = False) -> None:
        color = "#b00" if error else "#666"
        self.status.value = f"<span style='color:{color}'>{text}</span>"

    # -- the five actions ------------------------------------------------------

    def _center(self) -> None:
        self.view.center()
        self._say("centred on the visible drawables")

    def _reset(self) -> None:
        self.view.reset()
        self._say("back to the opening view — everything shown, no highlights "
                  "(colours are left as you set them)")

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
        """The other half of what 'Close' records, and the reason 'Close' records it.

        Kept as a **second button** rather than folded into 'Restore' with a fallback: the
        two slots answer different questions ("the angle I chose" against "wherever I was"),
        and one button silently picking between them would make it unclear which you got.
        """
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
        written = self.view.save(path)
        remember_prefix(written)
        self.path.value = default_path()
        self._say(f"wrote {written}")

    def _close(self) -> None:
        """Close the canvas and leave the image it last showed in its place.

        ``View.close`` records the viewpoint into :data:`~neu_draw.viewstate.LAST` on its
        way out, so the next figure's 'Last' button — or ``show(scene,
        viewpoint="last")`` — reopens on this angle. That is the reason to press this
        rather than deleting the cell.
        """
        if self._closed:
            self._say("already closed")
            return

        image = self.view.snapshot()
        size = self.view.logical_size()
        self.view.close()
        self._closed = True

        self._stage.children = (self._still(image, size),)
        # Every button off, including 'restore': there is no canvas left to draw the
        # restored angle on, and a control that responds by doing nothing visible is
        # worse than one that is plainly spent. The viewpoint is in the store for the
        # NEXT figure, which is what the status line says.
        for button in self._buttons.values():
            button.disabled = True
        self.path.disabled = True
        self._say(f"closed. The viewpoint is in views[{LAST!r}] — press 'Last' in the next "
                  f"figure, or open it with show(scene, viewpoint='{LAST}')")

    def _still(self, image, size: tuple[int, int]) -> Any:
        """The snapshot as an ``Image`` widget, or a note saying why not.

        Scaled back to the canvas's logical size: a snapshot is ``pixel_ratio`` times
        bigger than the view was asked for (that is where the antialiasing comes from),
        so shown at its natural size it would jump to two or three times the width the
        figure just occupied.
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

    ``required=False`` is the ``toolbar="auto"`` path: no ipywidgets, or a canvas that is
    not a widget, means no toolbar rather than an error — the view still draws, which is
    the thing that matters. An explicit request gets the exception.
    """
    try:
        return Toolbar(view)
    except (ImportError, TypeError):
        if required:
            raise
        return None
