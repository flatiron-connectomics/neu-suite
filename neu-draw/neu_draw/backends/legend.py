"""The legend: a clickable strip of entries, drawn **in the canvas** beside the scene.

One row per named drawable — a plate, a glyph matching what the thing is, and its name.
**Left-click a row to hide that drawable; right-click to highlight it.** Rows for hidden
drawables are dimmed rather than removed, because a hidden drawable with no entry is one
nobody can turn back on.

Those are the two questions a crowded scene actually raises — *what does it look like
without this* and *which one of these is this* — and the highlight answers the second
without touching the data: it is a display override, so ``drawable.color`` still says what
colour the body is. The predecessor swapped the real colour and stashed the original, which
holds until anything else reads it and a temporary highlight has become the body's colour.

## Why in the canvas, when the toolbar is ipywidgets

Because a figure without its legend is not the figure. The buttons are transient and have
no business in a saved PNG; the legend is what makes the picture readable, so it has to be
part of what ``snapshot()`` and ``save()`` produce. That single requirement decides the
implementation: pygfx objects in a second render pass, not widgets beside the canvas.

## The three mechanisms that make it work

**A second render pass into a rect.** ``renderer.render(scene, camera, rect=…,
flush=False)`` for the main scene, then the legend into the strip beside it. The first
call clears, the second does not — pygfx keys that off "first render since the last
flush" — so the two compose without either knowing about the other. Every legend material
is ``depth_test=False``, since the depth buffer still holds the main pass's geometry.

**Picking comes free with it.** The renderer resolves a pointer event by reading its pick
buffer at that pixel, and the second pass writes pick ids inside the strip, so a click
there lands on the legend rather than on whatever neuron is behind it. The plate is the
only entry object with ``pick_write=True``: the glyph and the label sit in front of it and
would otherwise steal the id from part of the row, making the click target a strange shape.

**The camera controller is registered on a VIEWPORT, not the renderer.** ``Controller.
handle_event`` starts a drag only when ``viewport.is_inside(event.x, event.y)``, so giving
it a viewport limited to the main rect is all it takes for a click on the legend not to
also spin the camera. The alternative — toggling ``controller.enabled`` from a pointer
handler — depends on which handler runs first, and pygfx keeps handlers in a **set**, so
there is no order to depend on.

Both rects are *computed* from the renderer's current logical size (:class:`_Part`
overrides ``Viewport._get_rect``, which pygfx documents as the extension point), so a
resized canvas needs no handler and cannot get out of step with what was drawn.

## Sizes are pixels, and text is measured rather than estimated

The legend is a fixed panel docked beside the scene: one world unit is one logical pixel,
the ortho camera's width and height are the strip's, and ``font_size`` therefore means
what it says. ``pygfx.Text`` lays its glyphs out eagerly, so ``get_bounding_box()`` gives
the *real* label width before anything is drawn — which is how the strip can be sized from
the longest name in one pass. (The fastplotlib predecessor estimated ~0.5 em per character
and then re-fitted for the first few frames to correct itself. Nothing here needs that, and
a snapshot rendered exactly once is right the first time.)

When there are more entries than fit, the camera zooms out instead of clipping: everything
stays visible and the text gets smaller. Preferring that over a scrollbar or a second
column is deliberate — a legend you cannot see all of is worse than a small one, and in a
figure there is nobody to scroll it.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import pygfx

from ..colors import RGBA, to_rgba
from ..scene import Legend, LinesDrawable, MeshDrawable, PointsDrawable, Scene
from .pygfx import display_color

#: Blank space inside the strip, in px.
PAD = 9.0
#: Between the glyph column and the label.
GAP = 7.0
#: Row height, and the glyph column's width, as multiples of ``font_size``.
ROW_HEIGHT = 1.95
GLYPH_WIDTH = 1.5
#: A strip narrower than this is not worth the space it takes from the scene.
MIN_WIDTH = 80.0
#: …and one wider than this share of the canvas is taking too much of it. A very long body
#: name is clipped rather than being allowed to squeeze the picture out of the frame.
#: **This cap wins over MIN_WIDTH**, which is not merely a preference: on a small canvas
#: the two disagree, and letting the floor win gave the scene a rect of width **zero** — a
#: division by zero inside the camera's projection, from a 50-px-wide test canvas.
MAX_WIDTH_FRACTION = 0.45
#: How far a hidden entry's colours are pulled toward black.
DIM = 0.35
#: Alpha of the highlight tint on a highlighted row's plate. Enough to read as lit, not
#: enough to stop the label being legible against it.
HIGHLIGHT_ROW_ALPHA = 0.38
#: The panel behind the strip when neither the legend nor the scene names a colour.
DEFAULT_PANEL: RGBA = (0.07, 0.07, 0.09, 1.0)
#: Where the legend's camera sits. Everything it draws is at z in [0, 1] in front of it;
#: see the `depth_range` note in `LegendOverlay.__init__` for why that matters.
CAMERA_Z = 100.0


def split_rects(size: Sequence[float], location: str, width: float
                ) -> tuple[tuple, tuple]:
    """``(main_rect, strip_rect)`` for a canvas of ``size``, both ``(x, y, w, h)``.

    Docked, not overlaid: the scene gets the rest of the canvas. An overlay would be
    cheaper — no controller gating, no aspect change — but it would also cover the
    neuron, and a legend that hides the data it labels is not much of a legend.
    """
    canvas_w, canvas_h = max(float(size[0]), 0.0), max(float(size[1]), 0.0)
    strip = min(max(float(width), MIN_WIDTH), canvas_w * MAX_WIDTH_FRACTION)
    strip = max(strip, 0.0)
    if location == "left":
        return (strip, 0.0, canvas_w - strip, canvas_h), (0.0, 0.0, strip, canvas_h)
    return (0.0, 0.0, canvas_w - strip, canvas_h), (canvas_w - strip, 0.0, strip, canvas_h)


def _dim(color: RGBA, factor: float = DIM) -> RGBA:
    """A colour pulled toward black, keeping its alpha. What "hidden" looks like."""
    r, g, b, a = color
    return (r * factor, g * factor, b * factor, a)


def _flat(color: RGBA, **kwargs) -> pygfx.MeshBasicMaterial:
    """An unlit overlay material. ``depth_test=False`` because the main pass's depth
    buffer is still there, and the legend is in front of all of it by definition."""
    return pygfx.MeshBasicMaterial(color=color, alpha_mode="blend",
                                   depth_test=False, depth_write=False, **kwargs)


class LegendEntry:
    """One row of the legend, and the drawable it stands for.

    Holds the drawable (whose ``visible``, ``name`` and ``color`` are the truth) and the
    ``WorldObject`` built for it, so a toggle changes both together and neither can drift.
    """

    def __init__(self, drawable: Any, world_object: Optional[pygfx.WorldObject],
                 spec: Legend):
        self.drawable = drawable
        self.world_object = world_object
        self.spec = spec
        #: Right-clicked. A **display** state and nothing more — see :meth:`refresh`.
        self.highlighted = False

        size = spec.font_size
        # The plate is sized in `layout()`, once every label has been measured; it starts
        # as a unit plane so the geometry object exists to be replaced.
        self.plate = pygfx.Mesh(pygfx.plane_geometry(1.0, size * ROW_HEIGHT),
                                _flat(spec.row_color, pick_write=True))
        self.plate.render_order = -1

        self.glyph = _glyph(drawable, size)
        self.label = pygfx.Text(
            text=str(drawable.name), font_size=size, screen_space=False,
            anchor="middle-left",
            material=pygfx.TextMaterial(color=spec.text_color, alpha_mode="blend",
                                        aa=True, pick_write=False,
                                        depth_test=False, depth_write=False))

        self.group = pygfx.Group()
        self.group.add(self.plate, self.glyph, self.label)
        self.refresh()

    # -- measurement -----------------------------------------------------------

    @property
    def label_width(self) -> float:
        """The label's real laid-out width in px. Measured, not estimated — pygfx lays
        text out at construction, so this is available before any draw."""
        box = self.label.get_bounding_box()
        return 0.0 if box is None else float(box[1][0] - box[0][0])

    # -- layout ----------------------------------------------------------------

    def layout(self, column: int, row: int, column_width: float) -> None:
        """Place this entry at ``(column, row)`` of a grid whose columns are that wide.

        Rows run **downward from y = 0**, which is what the render rect's own coordinates
        do: a rect's origin is its top-left, so ``y = -pixels from the top``. Columns run
        rightward from ``x = PAD``.
        """
        size = self.spec.font_size
        height = size * ROW_HEIGHT
        left = PAD + column * column_width
        self.group.local.position = (left, -row * height, 0.0)

        # The plate spans its whole column, so the click target is the row and not the
        # glyph — a strangely shaped hit area is the thing people report as "it ignored me".
        self.plate.geometry = pygfx.plane_geometry(column_width, height)
        self.plate.local.position = (column_width / 2.0, -height / 2.0, 0.0)
        self.glyph.local.position = (size * GLYPH_WIDTH / 2.0, -height / 2.0, 0.1)
        self.label.local.position = (size * GLYPH_WIDTH + GAP, -height / 2.0, 0.1)

    # -- state -----------------------------------------------------------------

    def refresh(self) -> None:
        """Push this entry's state onto the objects: colour, highlight, visibility.

        **The highlight is a display override and the drawable is never touched**, which is
        the same rule the placement offsets follow (see ``scene.Placed``) and it is what
        makes the feature safe to leave switched on. The predecessor swapped the graphic's
        real colour and stashed the original in the field that also held the swatch colour;
        that works until something else reads the colour — ``Scene.recolor``, a saved
        figure, ``bake()`` — and then a temporary highlight has quietly become the body's
        colour. Here ``drawable.color`` still answers "what colour is this body", so
        recolouring a highlighted entry is meaningful and takes effect when the highlight
        comes off.

        Everything is recomputed from scratch on every call, so the order in which
        highlight, visibility and colour were set never matters.
        """
        shown = bool(self.drawable.visible)
        base = to_rgba(self.drawable.color)
        drawn = self.spec.highlight_color if self.highlighted else base

        if self.world_object is not None:
            # `display_color` keeps the drawable's own alpha, so highlighting a translucent
            # surface does not also turn it opaque — being translucent is often exactly why
            # it could not be found.
            self.world_object.material.color = display_color(self.drawable, drawn)
            self.world_object.visible = shown

        # Full opacity for the glyph whatever the drawable's alpha. A scene alpha is about
        # overlapping surfaces not hiding each other; a 0.2 swatch is just illegible.
        glyph = (*drawn[:3], 1.0)
        self.glyph.material.color = glyph if shown else _dim(glyph)
        self.label.material.color = (self.spec.text_color if shown
                                     else _dim(self.spec.text_color, 0.55))
        self.plate.material.color = self._plate_color(shown)

    def _plate_color(self, shown: bool) -> RGBA:
        """The row's plate: tinted when highlighted, dimmed when hidden.

        The row has to say so itself, because the swatch cannot: once two entries are
        highlighted their swatches are both the highlight colour, and nothing then
        distinguishes "temporarily lit" from "this body really is white".
        """
        if self.highlighted:
            return (*self.spec.highlight_color[:3], HIGHLIGHT_ROW_ALPHA)
        return self.spec.row_color if shown else _dim(self.spec.row_color, 0.6)

    def set_label(self, text: str) -> None:
        # `Text.set_text`, not `Text.geometry.set_text`: in pygfx 0.17 a Text's glyph
        # layout lives on the object, and its geometry is a plain `Geometry` with no such
        # method — the older call raises `AttributeError`.
        self.label.set_text(str(text))

    def toggle(self) -> bool:
        self.drawable.visible = not self.drawable.visible
        self.refresh()
        return bool(self.drawable.visible)

    def toggle_highlight(self) -> bool:
        self.highlighted = not self.highlighted
        self.refresh()
        return self.highlighted


def _glyph(drawable: Any, size: float) -> pygfx.WorldObject:
    """A marker saying *what kind of thing* this row is, not only what colour.

    A skeleton and a mesh of the same body are two entries with the same colour (see
    ``build_scene``'s naming rule), so a colour-only legend cannot tell them apart. None
    of these write pick ids: the plate behind them is the click target, and a glyph that
    wrote its own would carve a hole in it.
    """
    color = (*to_rgba(drawable.color)[:3], 1.0)

    if isinstance(drawable, MeshDrawable):
        return pygfx.Mesh(pygfx.plane_geometry(size * 0.95, size * 0.95), _flat(color))

    if isinstance(drawable, LinesDrawable):
        half = size * GLYPH_WIDTH / 2.0
        return pygfx.Line(
            pygfx.Geometry(positions=np.array([[-half, 0.0, 0.0], [half, 0.0, 0.0]],
                                              dtype=np.float32)),
            pygfx.LineMaterial(thickness=max(3.0, size * 0.32), color=color,
                               alpha_mode="blend", pick_write=False,
                               depth_test=False, depth_write=False))

    if isinstance(drawable, PointsDrawable):
        return pygfx.Points(
            pygfx.Geometry(positions=np.zeros((1, 3), dtype=np.float32)),
            pygfx.PointsMarkerMaterial(size=size * 1.15, marker=drawable.marker,
                                       color=color, edge_width=0.0,
                                       alpha_mode="blend", pick_write=False,
                                       depth_test=False, depth_write=False))

    # A drawable kind the legend has no glyph for still gets a row: the name and the
    # click target are the useful part, and a blank swatch is better than no entry.
    return pygfx.Mesh(pygfx.plane_geometry(size * 0.95, size * 0.95), _flat(color))


class _Part(pygfx.Viewport):
    """One half of the split — the scene's rect, or the legend's.

    The rect is **computed from the renderer's current size** rather than stored, which
    ``Viewport._get_rect`` exists to allow. So a resized canvas needs no handler, and the
    controller's ``is_inside`` test cannot disagree with the rect that was drawn.
    """

    def __init__(self, overlay: "LegendOverlay", index: int):
        super().__init__(overlay.renderer)
        self._overlay = overlay
        self._index = index

    def _get_rect(self):
        return self._overlay.rects_for(self.renderer.logical_size)[self._index]


class LegendOverlay:
    """The legend for one :class:`~neu_draw.scene.Scene`, bound to one renderer.

    Entries come from the scene's **named** drawables in scene order. The
    ``WorldObject``s are looked up by name in the built group, which is sound because
    ``Scene.add`` refuses a duplicate name.
    """

    def __init__(self, scene: Scene, group: pygfx.Group, renderer: Any):
        self.scene_data = scene
        self.group = group
        self.renderer = renderer
        self.spec = scene.legend

        objects = {obj.name: obj for obj in group.children if obj.name}
        self.entries = [LegendEntry(d, objects.get(d.name), self.spec)
                        for d in scene.drawables if d.name is not None]

        # An opaque plate covering the whole strip, kept as a unit plane and **scaled**
        # rather than rebuilt: `_aim` runs every frame, and allocating a geometry per
        # frame to fill a fixed rectangle would be a per-draw cost for nothing.
        panel = self.spec.panel_color or scene.background or DEFAULT_PANEL
        self.backdrop = pygfx.Mesh(pygfx.plane_geometry(1.0, 1.0),
                                   _flat((*to_rgba(panel)[:3], 1.0)))
        self.backdrop.render_order = -2

        self.scene = pygfx.Scene()
        self.scene.add(self.backdrop)
        for entry in self.entries:
            self.scene.add(entry.group)
            self._bind(entry)

        self.camera = pygfx.OrthographicCamera(1.0, 1.0, maintain_aspect=False)
        # An EXPLICIT depth range, and it is load-bearing. Left at ``None``, pygfx derives
        # near and far from the camera's dimensions, which here are set *after*
        # construction — and the backdrop, at z = -1, fell outside the resulting range and
        # simply did not draw. Nothing raised: the strip rendered with its rows and no
        # panel, which reads as "the panel colour did not apply". Everything the legend
        # draws sits at z in [0, 1] and the camera at CAMERA_Z, so this range holds it all,
        # and depth_test is off anyway — layering is `render_order`, never z.
        self.camera.depth_range = (1.0, CAMERA_Z * 10.0)
        self.main = _Part(self, 0)
        self.viewport = _Part(self, 1)
        self.columns = 1
        self.measure()
        self.relayout(1)

    # -- geometry --------------------------------------------------------------

    def measure(self) -> None:
        """Re-read the label widths, which is everything the grid is derived from.

        Separate from :meth:`relayout` because it is the size-*independent* half: a rename
        changes this, a resized canvas does not.
        """
        size = self.spec.font_size
        self.row_height = size * ROW_HEIGHT
        labels = max((e.label_width for e in self.entries), default=0.0)
        # The trailing PAD doubles as the gap to the next column.
        self.column_width = size * GLYPH_WIDTH + GAP + labels + PAD

    def relayout(self, columns: int) -> None:
        """Place every entry into a grid ``columns`` wide, filling each column top-down."""
        self.columns = max(int(columns), 1)
        rows = int(np.ceil(len(self.entries) / self.columns)) if self.entries else 1
        self.rows = max(rows, 1)
        self.content_width = PAD + self.columns * self.column_width
        self.content_height = self.rows * self.row_height
        for index, entry in enumerate(self.entries):
            entry.layout(index // self.rows, index % self.rows, self.column_width)

    def layout(self) -> None:
        """Re-measure and re-place at the current column count. Idempotent."""
        self.measure()
        self.relayout(self.columns)

    def plan(self, size: Sequence[float]) -> tuple[int, float]:
        """``(columns, strip_width)`` for a canvas of ``size``.

        **Columns are chosen from the canvas, not from a fixed strip width**, and that
        direction is the whole trick. A single column of 30 bodies is four times the height
        of the canvas, so fitting it means shrinking by 4 — and since the shrink has to keep
        the strip's aspect ratio to avoid distorting the text, most of the strip's width
        then goes to empty margin. Measured on a 500x300 canvas: 4-px text beside 200 px of
        nothing. So: as many columns as the entries need, capped by what the width budget
        can afford, and only then shrink for whatever is still left over.

        An explicit ``Legend.width`` fixes the strip and the column count follows from it,
        which is what "I want it this wide" should mean.
        """
        canvas_w, canvas_h = max(float(size[0]), 1.0), max(float(size[1]), 1.0)
        budget = canvas_w * MAX_WIDTH_FRACTION
        if self.spec.width:
            budget = float(self.spec.width)

        rows_that_fit = max(int(canvas_h // self.row_height), 1)
        needed = max(int(np.ceil(len(self.entries) / rows_that_fit)), 1)
        affordable = max(int((budget - PAD) // self.column_width), 1)
        columns = max(min(needed, affordable), 1)

        width = float(self.spec.width) if self.spec.width else PAD + columns * self.column_width
        return columns, width

    @property
    def width(self) -> float:
        """The strip's width for the renderer it is attached to, in px."""
        return self.plan(self.renderer.logical_size)[1]

    def rects_for(self, size: Sequence[float]) -> tuple[tuple, tuple]:
        return split_rects(size, self.spec.location, self.plan(size)[1])

    def _aim(self, rect: Sequence[float]) -> None:
        """Point the ortho camera at the content: 1 world unit = 1 px, shrinking to fit.

        The camera's width and height keep the **rect's** aspect ratio, so nothing is
        distorted; when the content still does not fit, both grow by the same factor and
        the whole legend renders smaller rather than being clipped.
        """
        rect_w, rect_h = max(float(rect[2]), 1.0), max(float(rect[3]), 1.0)
        scale = min(1.0, rect_w / self.content_width, rect_h / self.content_height)
        width, height = rect_w / scale, rect_h / scale
        self.camera.width, self.camera.height = width, height
        # Content spans x in [0, content_width] and y in [-content_height, 0]: centre it
        # horizontally in the strip, and hang it from the top rather than the middle.
        center_x, center_y = self.content_width / 2.0, -height / 2.0
        self.camera.local.position = (center_x, center_y, CAMERA_Z)

        self.backdrop.local.scale = (width, height, 1.0)
        self.backdrop.local.position = (center_x, center_y, 0.0)

    def draw(self, renderer: Any = None, rect: Optional[Sequence[float]] = None,
             flush: bool = True) -> None:
        """Render the strip. ``renderer`` defaults to the one this was built against.

        The parameter exists for ``View._offscreen_snapshot``, which builds a renderer of
        its own at an exact size — the legend has to go through that pass too, or a saved
        figure would come out without it.

        Re-lays out only when the column count has actually changed, so a resize adapts
        while an ordinary frame allocates nothing.
        """
        renderer = renderer if renderer is not None else self.renderer
        size = renderer.logical_size
        columns, _ = self.plan(size)
        if columns != self.columns:
            self.relayout(columns)
        if rect is None:
            rect = self.rects_for(size)[1]
        self._aim(rect)
        renderer.render(self.scene, self.camera, rect=tuple(rect), flush=flush)

    # -- interaction -----------------------------------------------------------

    def _bind(self, entry: LegendEntry) -> None:
        """Left-click a row to hide it; **right-click to highlight it**.

        The pair the predecessor settled on, and the division is between the two questions
        you actually ask of a crowded scene: *what does it look like without this* (hide)
        and *which one of these is this* (highlight). A middle click does nothing.
        """
        def on_click(event) -> None:
            button = getattr(event, "button", 1)
            if button == 1:
                entry.toggle()
            elif button == 2:
                entry.toggle_highlight()
            else:
                return
            self._request_draw()

        entry.plate.add_event_handler(on_click, "click")

    def _request_draw(self) -> None:
        request = getattr(self.renderer, "request_draw", None)
        if request is not None:
            request()

    # -- the notebook surface --------------------------------------------------

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    @property
    def names(self) -> list[str]:
        return [e.drawable.name for e in self.entries]

    def entry(self, name: str) -> LegendEntry:
        for entry in self.entries:
            if entry.drawable.name == name:
                return entry
        raise KeyError(f"no legend entry named {name!r}; have {self.names}")

    def __getitem__(self, name: str) -> LegendEntry:
        return self.entry(name)

    def rename(self, old: str, new: str) -> "LegendOverlay":
        """Relabel an entry — which renames the drawable, because they are one name.

        Re-lays out afterwards, since a longer name means a wider strip.
        """
        self.scene_data.rename(old, new)
        self.entry(new).set_label(new)
        self.layout()
        self._request_draw()
        return self

    def recolor(self, name: str, color: Any) -> "LegendOverlay":
        """Recolour one drawable **and** its swatch, leaving the rest of the scene alone.

        On a **highlighted** entry this changes the colour underneath and leaves the
        highlight showing, so the new colour appears when the highlight comes off. That
        follows from the highlight being a display override rather than a colour swap, and
        it is the behaviour that cannot lose work either way round.
        """
        self.scene_data.set_color(name, color)
        self.entry(name).refresh()
        self._request_draw()
        return self

    def set_visible(self, name: str, visible: bool) -> "LegendOverlay":
        """What clicking a row does, by name."""
        entry = self.entry(name)
        entry.drawable.visible = bool(visible)
        entry.refresh()
        self._request_draw()
        return self

    def toggle(self, name: str) -> bool:
        state = self.entry(name).toggle()
        self._request_draw()
        return state

    # -- highlighting ----------------------------------------------------------

    @property
    def highlighted(self) -> list[str]:
        """The names currently drawn in the highlight colour."""
        return [e.drawable.name for e in self.entries if e.highlighted]

    def highlight(self, *names: str, exclusive: bool = False) -> "LegendOverlay":
        """Draw these entries in the highlight colour, leaving their real colours alone.

        ``exclusive=True`` drops every other highlight first, which is the "where is this
        one" case; the default adds to whatever is already lit, matching what right-clicking
        rows one after another does.
        """
        if exclusive:
            self.clear_highlights()
        for name in names:
            self.entry(name).highlighted = True
            self.entry(name).refresh()
        self._request_draw()
        return self

    def unhighlight(self, *names: str) -> "LegendOverlay":
        """Put these entries back to their own colours. No names means all of them."""
        chosen = [self.entry(n) for n in names] if names else list(self.entries)
        for entry in chosen:
            entry.highlighted = False
            entry.refresh()
        self._request_draw()
        return self

    def clear_highlights(self) -> "LegendOverlay":
        return self.unhighlight()

    def toggle_highlight(self, name: str) -> bool:
        """What right-clicking a row does."""
        state = self.entry(name).toggle_highlight()
        self._request_draw()
        return state

    def refresh(self) -> "LegendOverlay":
        """Re-read every drawable, for when a scene was changed behind the legend's back."""
        for entry in self.entries:
            entry.refresh()
        self.layout()
        self._request_draw()
        return self

    def __repr__(self) -> str:
        shown = sum(1 for e in self.entries if e.drawable.visible)
        lit = len(self.highlighted)
        return (f"LegendOverlay({shown}/{len(self.entries)} shown"
                f"{f', {lit} highlighted' if lit else ''}, "
                f"{self.spec.location}, {self.width:.0f}px, {self.columns} col)")
