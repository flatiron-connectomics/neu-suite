"""The legend: a clickable strip of entries, drawn **in the canvas** beside the scene.

One row per **label** — a plate, a glyph matching what the row holds, and the text. A row
is a *group*: the drawables sharing a label, which is usually one and may be forty bodies of
a cell type under a single name and colour.

**Left-click a row to hide everything on it; right-click to highlight it.** Hidden rows are
dimmed rather than removed, because a hidden drawable with no row is one nobody can turn
back on — and a *partly* hidden group gets a third appearance, since a row that showed
either extreme would be lying about half its members.

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

from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import pygfx

from ..colors import RGBA, to_rgba
from ..scene import (Legend, LinesDrawable, MeshDrawable, PointsDrawable, Scene,
                     label_of)
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
    """One row of the legend, and **every** drawable it stands for.

    A row is a *group*: the drawables sharing one :func:`~neu_draw.scene.label_of`. Usually
    that is one drawable, and everything below reads as if it were — but forty bodies of a
    cell type are one row, and clicking it acts on all forty.

    Holds the drawables (whose ``visible`` and ``color`` are the truth) alongside the
    ``WorldObject``s built for them, paired by position, so a toggle changes both together
    and neither can drift.
    """

    def __init__(self, label: str, indices: Sequence[int], drawables: Sequence[Any],
                 world_objects: Sequence[Optional[pygfx.WorldObject]], spec: Legend):
        self.text = str(label)
        #: Positions of the members in ``Scene.drawables``. Carried so a rebuild can match
        #: rows up across a relabel, which is the one thing their labels cannot do.
        self.indices = list(indices)
        self.drawables = list(drawables)
        self.world_objects = list(world_objects)
        self.spec = spec
        #: Right-clicked. A **display** state and nothing more — see :meth:`refresh`.
        self.highlighted = False

        size = spec.font_size
        # The plate is sized in `layout()`, once every label has been measured; it starts
        # as a unit plane so the geometry object exists to be replaced.
        self.plate = pygfx.Mesh(pygfx.plane_geometry(1.0, size * ROW_HEIGHT),
                                _flat(spec.row_color, pick_write=True))
        self.plate.render_order = -1

        self.glyph = _glyph(self.drawables, size)
        self.label = pygfx.Text(
            text=self.row_text, font_size=size, screen_space=False,
            anchor="middle-left",
            material=pygfx.TextMaterial(color=spec.text_color, alpha_mode="blend",
                                        aa=True, pick_write=False,
                                        depth_test=False, depth_write=False))

        self.group = pygfx.Group()
        self.group.add(self.plate, self.glyph, self.label)
        self.refresh()

    # -- what it stands for ----------------------------------------------------

    @property
    def names(self) -> list[str]:
        """The member drawables' names — their identities, as against this row's label."""
        return [d.name for d in self.drawables if d.name is not None]

    @property
    def row_text(self) -> str:
        """The label, with a member count when the row is a group of more than one.

        The count is what tells you a row *is* a group; without it "Tm2" and "presyn" look
        like the same kind of thing and hiding one is a much bigger action than the other.
        """
        n = len(self.drawables)
        return f"{self.text} ({n})" if n > 1 else self.text

    @property
    def kinds(self) -> set[str]:
        return {type(d).__name__ for d in self.drawables}

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
        state = self.visibility
        drawn = self.spec.highlight_color if self.highlighted else None

        for drawable, obj in zip(self.drawables, self.world_objects):
            if obj is None:
                continue
            # `display_color` keeps the drawable's own alpha, so highlighting a translucent
            # surface does not also turn it opaque — being translucent is often exactly why
            # it could not be found. Each member keeps its own colour when not highlighted,
            # so a group whose members disagree is drawn honestly even though one swatch
            # cannot show all of it.
            obj.material.color = display_color(drawable, drawn)
            obj.visible = bool(drawable.visible)

        # Full opacity for the glyph whatever the drawable's alpha. A scene alpha is about
        # overlapping surfaces not hiding each other; a 0.2 swatch is just illegible.
        swatch = drawn if drawn is not None else to_rgba(self.drawables[0].color)
        glyph = (*swatch[:3], 1.0)
        self.glyph.material.color = glyph if state != "none" else _dim(glyph)
        self.label.material.color = (self.spec.text_color if state != "none"
                                     else _dim(self.spec.text_color, 0.55))
        self.plate.material.color = self._plate_color(state)

    @property
    def visibility(self) -> str:
        """``"all"``, ``"some"`` or ``"none"`` of the members are showing.

        **A group has three states, not two**, and the middle one is reachable without
        anyone clicking anything: hide one body by name, or `reset()` a scene that started
        with something hidden, and the row is neither on nor off. Showing it as either
        would make the row lie about half its members.
        """
        shown = sum(1 for d in self.drawables if d.visible)
        if shown == len(self.drawables):
            return "all"
        return "none" if shown == 0 else "some"

    def _plate_color(self, state: str) -> RGBA:
        """The row's plate: tinted when highlighted, dimmed by how much of it is hidden.

        The row has to carry the highlight itself, because the swatch cannot: once two
        entries are highlighted their swatches are both the highlight colour, and nothing
        then distinguishes "temporarily lit" from "this body really is white".
        """
        if self.highlighted:
            return (*self.spec.highlight_color[:3], HIGHLIGHT_ROW_ALPHA)
        if state == "all":
            return self.spec.row_color
        return _dim(self.spec.row_color, 0.6 if state == "none" else 0.8)

    def toggle(self) -> bool:
        """Hide the whole group, or show it. Returns whether it is now visible.

        **Anything showing means hide; nothing showing means show.** The alternative —
        inverting each member — would turn a partly-hidden group inside out, which is not
        what clicking one row is asking for.
        """
        target = self.visibility == "none"
        for drawable in self.drawables:
            drawable.visible = target
        self.refresh()
        return target

    def toggle_highlight(self) -> bool:
        self.highlighted = not self.highlighted
        self.refresh()
        return self.highlighted


def _glyph(drawables: Sequence[Any], size: float) -> pygfx.WorldObject:
    """A marker saying *what kind of thing* this row is, not only what colour.

    A body's mesh and its skeleton are two entries with the same colour (see
    ``build_scene``'s naming rule), so a colour-only legend cannot tell them apart.

    **A group of mixed kinds gets the neutral square**, because a group labelled by cell
    type normally holds meshes *and* skeletons and a line glyph on it would be a claim
    about the row that is only a third true. A group of one kind gets that kind's glyph,
    which covers every ungrouped row.

    The colour is the **first member's**: one swatch cannot show forty, and where the point
    of grouping is that a type shares a colour they all agree anyway. None of these write
    pick ids — the plate behind them is the click target, and a glyph that wrote its own
    would carve a hole in it.
    """
    color = (*to_rgba(drawables[0].color)[:3], 1.0)
    kinds = {type(d) for d in drawables}
    square = pygfx.Mesh(pygfx.plane_geometry(size * 0.95, size * 0.95), _flat(color))

    if len(kinds) != 1:
        return square
    (kind,) = kinds

    if kind is MeshDrawable:
        return square

    if kind is LinesDrawable:
        half = size * GLYPH_WIDTH / 2.0
        return pygfx.Line(
            pygfx.Geometry(positions=np.array([[-half, 0.0, 0.0], [half, 0.0, 0.0]],
                                              dtype=np.float32)),
            pygfx.LineMaterial(thickness=max(3.0, size * 0.32), color=color,
                               alpha_mode="blend", pick_write=False,
                               depth_test=False, depth_write=False))

    if kind is PointsDrawable:
        return pygfx.Points(
            pygfx.Geometry(positions=np.zeros((1, 3), dtype=np.float32)),
            pygfx.PointsMarkerMaterial(size=size * 1.15, marker=drawables[0].marker,
                                       color=color, edge_width=0.0,
                                       alpha_mode="blend", pick_write=False,
                                       depth_test=False, depth_write=False))

    # A drawable kind the legend has no glyph for still gets a row: the label and the
    # click target are the useful part, and a blank swatch is better than no entry.
    return square


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

    **One row per distinct label**, in first-appearance order — see
    :func:`~neu_draw.scene.label_of`. A drawable with neither label nor name gets no row.

    Drawables are paired with their ``WorldObject``s **by position**, because
    :func:`~neu_draw.backends.pygfx.build` makes exactly one object per drawable in order.
    Looking them up by name would be the obvious alternative and is wrong here: a drawable
    may carry a label and no name at all, and that is a legitimate way to put an anonymous
    thing on a labelled row.
    """

    def __init__(self, scene: Scene, group: pygfx.Group, renderer: Any):
        self.scene_data = scene
        self.group = group
        self.renderer = renderer
        self.spec = scene.legend

        # An opaque plate covering the whole strip, kept as a unit plane and **scaled**
        # rather than rebuilt: `_aim` runs every frame, and allocating a geometry per
        # frame to fill a fixed rectangle would be a per-draw cost for nothing.
        panel = self.spec.panel_color or scene.background or DEFAULT_PANEL
        self.backdrop = pygfx.Mesh(pygfx.plane_geometry(1.0, 1.0),
                                   _flat((*to_rgba(panel)[:3], 1.0)))
        self.backdrop.render_order = -2

        self.scene = pygfx.Scene()
        self.scene.add(self.backdrop)
        self.entries: list[LegendEntry] = []
        self._structure_at: tuple = ()
        self._state_at: tuple = ()
        self.build_entries()

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

    # -- the rows ---------------------------------------------------------------

    def build_entries(self) -> None:
        """(Re)group the scene's drawables into rows, discarding any rows there were.

        **Relabelling is structural, not a text edit** — it merges rows and splits them —
        so there is no in-place version of this and every path that changes a label comes
        through here. Rebuilding a few dozen small pygfx objects is cheap, and it is not on
        the draw path.

        Highlights are carried across **by member**, not by label — a rebuild is invisible
        to whoever pressed the button, and the labels are exactly what a relabel changes, so
        matching on them would drop the highlight on the row being renamed. A rebuilt row is
        lit when *every* member of it was lit: a split keeps both halves, a plain rename
        keeps the row, and merging a lit row into an unlit one goes dark rather than lighting
        up bodies nobody asked about.
        """
        lit = {i for e in self.entries if e.highlighted for i in e.indices}
        for entry in self.entries:
            self.scene.remove(entry.group)
        self._structure_at, self._state_at = self._structure(), self._state()

        groups: dict[str, list[int]] = {}
        for index, drawable in enumerate(self.scene_data.drawables):
            label = label_of(drawable)
            if label is not None:
                groups.setdefault(str(label), []).append(index)

        objects = list(self.group.children)
        self.entries = []
        for label, indices in groups.items():
            entry = LegendEntry(
                label, indices,
                [self.scene_data.drawables[i] for i in indices],
                [objects[i] if i < len(objects) else None for i in indices],
                self.spec)
            entry.highlighted = bool(lit) and set(indices) <= lit
            entry.refresh()
            self.entries.append(entry)
            self.scene.add(entry.group)
            self._bind(entry)

    def _structure(self) -> tuple:
        """What the rows ARE: one label per drawable, in order.

        Membership and order both follow from the sequence, so a merge, a split, a rename,
        an added drawable and a reordering all show up as a different value.
        """
        return tuple(label_of(d) for d in self.scene_data.drawables)

    def _state(self) -> tuple:
        """What the rows LOOK like: each drawable's visibility and colour."""
        return tuple((bool(d.visible), tuple(d.color)) for d in self.scene_data.drawables)

    def sync(self) -> bool:
        """Catch up with the scene if it has changed since the last look. Cheap when not.

        **This is what makes an edit take effect without anyone asking.** A ``Scene`` is a
        plain mutable dataclass, so ``scene.get("a").label = "Tm2"`` cannot notify anyone —
        adding change tracking to every field would mean properties on all of them, and
        half-automatic notification is worse than none, because you learn to rely on it and
        then meet the case it misses. Re-reading the truth instead misses nothing.

        Two fingerprints, two responses, because the work they need is not the same: a
        changed *structure* needs the rows rebuilt (new ``pygfx.Text`` layouts, a re-measured
        strip), while a changed *state* only needs materials reassigned. Comparing tuples of
        a few dozen strings per frame is nothing; rebuilding text layouts per frame would not
        be.

        Returns whether anything was done, and is called from :meth:`draw` — so a snapshot
        is also correct, which is why ``view.save(...)`` right after a relabel writes the new
        text.
        """
        structure = self._structure()
        if structure != self._structure_at:
            self.build_entries()
            self.layout()
            return True

        state = self._state()
        if state != self._state_at:
            for entry in self.entries:
                entry.refresh()
            self._state_at = state
            return True
        return False

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

        Re-reads the scene first (:meth:`sync`) and re-lays out only when the column count
        has actually changed, so an edit made anywhere takes effect and an ordinary frame
        still allocates nothing.
        """
        renderer = renderer if renderer is not None else self.renderer
        self.sync()
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
    def labels(self) -> list[str]:
        """This legend's rows, in order. **Every method here is keyed on these**, not on
        drawable names — a row is what you can see and click."""
        return [e.text for e in self.entries]

    def entry(self, label: str) -> LegendEntry:
        for entry in self.entries:
            if entry.text == str(label):
                return entry
        raise KeyError(f"no legend row labelled {label!r}; have {self.labels}. "
                       f"Rows are keyed by LABEL, which for an ungrouped drawable is its "
                       f"name — see Scene.relabel.")

    def __getitem__(self, label: str) -> LegendEntry:
        return self.entry(label)

    def relabel(self, labels: Union[str, Mapping[str, str]],
                new: Optional[str] = None) -> "LegendOverlay":
        """Change rows' text, keyed on their **current** label. Mapping or ``(old, new)``.

        This is how you change what a legend says. **Giving two rows the same new label
        merges them**, and that is the intended way to group after the fact::

            view.legend.relabel({"1401 mesh": "Tm2", "1402 mesh": "Tm2"})

        It sets the members' ``label`` and leaves their **names** alone, so
        ``scene.get("1401 mesh")`` keeps working. It replaced a ``rename`` method that
        renamed the drawable instead: once a row can hold several drawables, its text is a
        label and not a name, and renaming one member would not change the row at all.
        Since a label defaults to the name, the single-row call reads the same as before.
        """
        mapping = dict(labels) if isinstance(labels, Mapping) else {labels: new}
        for old, value in mapping.items():
            for drawable in self.entry(old).drawables:
                drawable.label = value
        self.build_entries()
        self.layout()
        self._request_draw()
        return self

    def recolor(self, label: str, color: Any) -> "LegendOverlay":
        """Recolour a row — **every drawable on it** — leaving the rest of the scene alone.

        On a **highlighted** row this changes the colour underneath and leaves the highlight
        showing, so the new colour appears when the highlight comes off. That follows from
        the highlight being a display override rather than a colour swap, and it is the
        behaviour that cannot lose work either way round.
        """
        rgba = to_rgba(color)
        for drawable in self.entry(label).drawables:
            drawable.color = rgba
        self.entry(label).refresh()
        self._request_draw()
        return self

    def set_visible(self, label: str, visible: bool) -> "LegendOverlay":
        """Show or hide a whole row. What clicking it does, by label."""
        entry = self.entry(label)
        for drawable in entry.drawables:
            drawable.visible = bool(visible)
        entry.refresh()
        self._request_draw()
        return self

    def toggle(self, label: str) -> bool:
        state = self.entry(label).toggle()
        self._request_draw()
        return state

    # -- highlighting ----------------------------------------------------------

    @property
    def highlighted(self) -> list[str]:
        """The row labels currently drawn in the highlight colour."""
        return [e.text for e in self.entries if e.highlighted]

    def highlight(self, *labels: str, exclusive: bool = False) -> "LegendOverlay":
        """Draw these rows in the highlight colour, leaving their real colours alone.

        ``exclusive=True`` drops every other highlight first, which is the "where is this
        one" case; the default adds to whatever is already lit, matching what right-clicking
        rows one after another does.
        """
        if exclusive:
            self.clear_highlights()
        for label in labels:
            entry = self.entry(label)
            entry.highlighted = True
            entry.refresh()
        self._request_draw()
        return self

    def unhighlight(self, *labels: str) -> "LegendOverlay":
        """Put these rows back to their own colours. No labels means all of them."""
        chosen = [self.entry(x) for x in labels] if labels else list(self.entries)
        for entry in chosen:
            entry.highlighted = False
            entry.refresh()
        self._request_draw()
        return self

    def clear_highlights(self) -> "LegendOverlay":
        return self.unhighlight()

    def toggle_highlight(self, label: str) -> bool:
        """What right-clicking a row does."""
        state = self.entry(label).toggle_highlight()
        self._request_draw()
        return state

    def refresh(self) -> "LegendOverlay":
        """Re-read the scene, for when it was changed behind the legend's back.

        **Rebuilds the rows** rather than only re-reading colours, because a label may have
        changed through ``Scene.relabel`` — or a name, for an unlabelled drawable — and a
        row's text is baked into a ``pygfx.Text``. It used to only refresh materials, so a
        direct ``scene.rename(...)`` left the canvas drawing the old text while
        ``legend.labels`` reported the new one. Erik hit exactly that.
        """
        self.build_entries()
        self.layout()
        self._request_draw()
        return self

    def __repr__(self) -> str:
        shown = sum(1 for e in self.entries if e.visibility != "none")
        lit = len(self.highlighted)
        return (f"LegendOverlay({shown}/{len(self.entries)} shown"
                f"{f', {lit} highlighted' if lit else ''}, "
                f"{self.spec.location}, {self.width:.0f}px, {self.columns} col)")
