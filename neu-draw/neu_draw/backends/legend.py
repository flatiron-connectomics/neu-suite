"""The legend: a clickable strip drawn in the canvas beside the scene, so it is part of
every snapshot and saved figure.

One row per label (a plate, a glyph for the kind of drawable, and the text); a row may
stand for many drawables. Left-click a row to hide or show it, right-click to highlight
it. Hidden rows are dimmed, not removed, and a partly hidden group has a third look. The
highlight is a display override: ``drawable.color`` is never changed.

How it works:

- **A second render pass into the strip's rect** (``render(..., rect=, flush=False)``)
  after the main scene. Legend materials use ``depth_test=False``.
- **Picking**: the second pass writes pick ids inside the strip, so a click there hits the
  legend. Only the plate writes them, so the whole row is the click target.
- **The camera controller is bound to a viewport covering the main rect**, so a click on
  the legend does not also rotate the camera.
- Both rects are computed from the renderer's current size (:class:`_Part`), so resizing
  needs no handler.

Sizes are logical pixels: one world unit is one pixel. Label widths are measured from
``pygfx.Text``'s laid-out bounding box, so the strip fits the longest label in one pass.
When entries do not fit, the legend adds columns up to a width budget, then shrinks
rather than clipping.
"""

from __future__ import annotations

import warnings
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
#: The narrowest strip, in px.
MIN_WIDTH = 80.0
#: The widest strip, as a share of the canvas. This cap wins over MIN_WIDTH: on a tiny
#: canvas the floor would leave the scene a zero-width rect.
MAX_WIDTH_FRACTION = 0.45
#: How far a hidden entry's colours are pulled toward black.
DIM = 0.35
#: Alpha of the highlight tint on a highlighted row's plate.
HIGHLIGHT_ROW_ALPHA = 0.38
#: The panel behind the strip when neither the legend nor the scene names a colour.
DEFAULT_PANEL: RGBA = (0.07, 0.07, 0.09, 1.0)
#: Where the legend's camera sits; everything it draws is at z in [0, 1].
CAMERA_Z = 100.0


def split_rects(size: Sequence[float], location: str, width: float
                ) -> tuple[tuple, tuple]:
    """``(main_rect, strip_rect)`` for a canvas of ``size``, both ``(x, y, w, h)``.

    Docked beside the scene rather than over it, so it never hides the data.
    """
    canvas_w, canvas_h = max(float(size[0]), 0.0), max(float(size[1]), 0.0)
    strip = min(max(float(width), MIN_WIDTH), canvas_w * MAX_WIDTH_FRACTION)
    strip = max(strip, 0.0)
    if location == "left":
        return (strip, 0.0, canvas_w - strip, canvas_h), (0.0, 0.0, strip, canvas_h)
    return (0.0, 0.0, canvas_w - strip, canvas_h), (canvas_w - strip, 0.0, strip, canvas_h)


def _dim(color: RGBA, factor: float = DIM) -> RGBA:
    """A colour pulled toward black, keeping its alpha (how hidden rows look)."""
    r, g, b, a = color
    return (r * factor, g * factor, b * factor, a)


def _flat(color: RGBA, **kwargs) -> pygfx.MeshBasicMaterial:
    """An unlit overlay material, without depth testing against the main pass."""
    return pygfx.MeshBasicMaterial(color=color, alpha_mode="blend",
                                   depth_test=False, depth_write=False, **kwargs)


class LegendEntry:
    """One legend row and every drawable it stands for: those sharing a
    :func:`~neu_draw.scene.label_of`, or a declared group.

    Holds the drawables (whose ``visible`` and ``color`` are authoritative) and the
    ``WorldObject`` built for each, paired by position, so a toggle updates both.
    """

    def __init__(self, label: str, indices: Sequence[int], drawables: Sequence[Any],
                 world_objects: Sequence[Optional[pygfx.WorldObject]], spec: Legend,
                 is_group: bool = False):
        self.text = str(label)
        #: A row declared in ``Legend.groups``; its members also have rows of their own.
        self.is_group = is_group
        #: Positions of the members in ``Scene.drawables``, used to match rows across a
        #: relabel.
        self.indices = list(indices)
        self.drawables = list(drawables)
        self.world_objects = list(world_objects)
        self.spec = spec
        #: Right-clicked; a display state only (see :meth:`refresh`).
        self.highlighted = False

        size = spec.font_size
        # Sized in `layout()`, once every label is measured.
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
        """The member drawables' names."""
        return [d.name for d in self.drawables if d.name is not None]

    @property
    def row_text(self) -> str:
        """The label, with a member count when the row stands for more than one drawable."""
        n = len(self.drawables)
        return f"{self.text} ({n})" if n > 1 else self.text

    @property
    def kinds(self) -> set[str]:
        return {type(d).__name__ for d in self.drawables}

    # -- measurement -----------------------------------------------------------

    @property
    def label_width(self) -> float:
        """The label's laid-out width in px, available before any draw."""
        box = self.label.get_bounding_box()
        return 0.0 if box is None else float(box[1][0] - box[0][0])

    # -- layout ----------------------------------------------------------------

    def layout(self, column: int, row: int, column_width: float) -> None:
        """Place this entry at ``(column, row)`` of a grid whose columns are that wide.

        Rows run downward from y = 0 (``y`` is minus the pixels from the top); columns run
        rightward from ``x = PAD``.
        """
        size = self.spec.font_size
        height = size * ROW_HEIGHT
        left = PAD + column * column_width
        self.group.local.position = (left, -row * height, 0.0)

        # The plate spans its whole column, so the whole row is the click target.
        self.plate.geometry = pygfx.plane_geometry(column_width, height)
        self.plate.local.position = (column_width / 2.0, -height / 2.0, 0.0)
        self.glyph.local.position = (size * GLYPH_WIDTH / 2.0, -height / 2.0, 0.1)
        self.label.local.position = (size * GLYPH_WIDTH + GAP, -height / 2.0, 0.1)

    # -- state -----------------------------------------------------------------

    def refresh(self, lit: Optional[set] = None) -> None:
        """Push this entry's state onto the objects: colour, highlight, visibility.

        ``lit`` is the set of scene indices to highlight, computed by the overlay over every
        row, since a drawable can be on its own row and on group rows; ``None`` uses this
        row's own flag. The highlight only changes the drawn material, never
        ``drawable.color``, so recolouring a highlighted row takes effect when the highlight
        comes off. Everything is recomputed on each call.
        """
        state = self.visibility
        drawn = self.spec.highlight_color if self.highlighted else None
        if lit is None:
            lit = set(self.indices) if self.highlighted else set()

        for index, drawable, obj in zip(self.indices, self.drawables, self.world_objects):
            if obj is None:
                continue
            override = self.spec.highlight_color if index in lit else None
            # `display_color` keeps the drawable's own alpha, so a highlighted translucent
            # surface stays translucent. Unhighlighted members keep their own colours.
            obj.material.color = display_color(drawable, override)
            obj.visible = bool(drawable.visible)

        # The glyph is opaque whatever the drawable's alpha, or a faint mesh's swatch would
        # be illegible.
        swatch = drawn if drawn is not None else to_rgba(self.drawables[0].color)
        glyph = (*swatch[:3], 1.0)
        self.glyph.material.color = glyph if state != "none" else _dim(glyph)
        self.label.material.color = (self.spec.text_color if state != "none"
                                     else _dim(self.spec.text_color, 0.55))
        self.plate.material.color = self._plate_color(state)

    @property
    def visibility(self) -> str:
        """``"all"``, ``"some"`` or ``"none"`` of the members are showing.

        A group can be partly shown, for instance after hiding one member by name.
        """
        shown = sum(1 for d in self.drawables if d.visible)
        if shown == len(self.drawables):
            return "all"
        return "none" if shown == 0 else "some"

    def _plate_color(self, state: str) -> RGBA:
        """The row's plate: tinted when highlighted, so a lit row is distinguishable from a
        body that really is the highlight colour, and dimmed by how much is hidden."""
        if self.highlighted:
            return (*self.spec.highlight_color[:3], HIGHLIGHT_ROW_ALPHA)
        base = self.spec.group_row_color if self.is_group else self.spec.row_color
        if state == "all":
            return base
        return _dim(base, 0.6 if state == "none" else 0.8)

    def toggle(self) -> bool:
        """Hide the whole group, or show it. Returns whether it is now visible.

        If anything is showing, hide all; if nothing is, show all (rather than inverting
        each member).
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
    """A marker for the kind of drawable on the row: a square for meshes, a line for
    skeletons, the drawable's marker for points, and a square for mixed kinds. Coloured
    like the first member. Glyphs write no pick ids, so the plate stays the click target.
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

    # Any other kind still gets a row, with a square.
    return square


class _Part(pygfx.Viewport):
    """One half of the split, the scene's rect or the legend's, computed from the
    renderer's current size (via ``Viewport._get_rect``), so resizing needs no handler."""

    def __init__(self, overlay: "LegendOverlay", index: int):
        super().__init__(overlay.renderer)
        self._overlay = overlay
        self._index = index

    def _get_rect(self):
        return self._overlay.rects_for(self.renderer.logical_size)[self._index]


class LegendOverlay:
    """The legend for one :class:`~neu_draw.scene.Scene`, bound to one renderer.

    One row per distinct :func:`~neu_draw.scene.label_of`, in first-appearance order; a
    drawable with neither label nor name gets no row. Drawables are paired with their
    world objects by position (:func:`~neu_draw.backends.pygfx.build` makes one per
    drawable, in order), not by name, since a drawable may have a label and no name.
    """

    def __init__(self, scene: Scene, group: pygfx.Group, renderer: Any):
        self.scene_data = scene
        self.group = group
        self.renderer = renderer
        self.spec = scene.legend

        # An opaque plate covering the strip: a unit plane that `_aim` scales each frame.
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
        # An explicit depth range is required: pygfx would otherwise derive one from camera
        # dimensions set later, and the backdrop would silently fall outside it and not
        # draw. Layering is by `render_order`, not z.
        self.camera.depth_range = (1.0, CAMERA_Z * 10.0)
        self.main = _Part(self, 0)
        self.viewport = _Part(self, 1)
        self.columns = 1
        self.measure()
        self.relayout(1)

    # -- the rows ---------------------------------------------------------------

    def build_entries(self, strict: bool = True) -> None:
        """(Re)group the scene's drawables into rows, discarding any rows there were.

        Relabelling can merge and split rows, so every label change rebuilds them here.
        Highlights carry over by member: a rebuilt row is lit if every member was, so a
        split keeps both halves lit and a lit row merged into an unlit one goes dark.

        Group rows (``Legend.groups``) come first, carry their highlight over by text, and
        resolve members by name. With ``strict``, an unknown member or a group text that
        collides with a label row raises; :meth:`sync` passes ``False``, since it runs
        inside a draw, and warns and skips instead.
        """
        lit_rows = {i for e in self.entries if e.highlighted and not e.is_group
                    for i in e.indices}
        lit_groups = {e.text for e in self.entries if e.highlighted and e.is_group}
        for entry in self.entries:
            self.scene.remove(entry.group)
        self._structure_at, self._state_at = self._structure(), self._state()

        drawables = self.scene_data.drawables
        rows: dict[str, list[int]] = {}
        for index, drawable in enumerate(drawables):
            label = label_of(drawable)
            if label is not None:
                rows.setdefault(str(label), []).append(index)

        by_name = {d.name: i for i, d in enumerate(drawables) if d.name is not None}
        declared: dict[str, list[int]] = {}
        for text, members in (self.spec.groups or {}).items():
            problems = []
            if text in rows:
                problems.append(f"its text is also a label row's, and rows are addressed "
                                f"by text")
            unknown = [m for m in members if m not in by_name]
            if unknown:
                problems.append(f"it names drawables that are not in the scene: {unknown}")
            indices = [by_name[m] for m in members if m in by_name]
            if not indices and not problems:
                problems.append("it has no members")
            if problems:
                message = f"legend group {text!r}: " + "; ".join(problems)
                if strict:
                    raise ValueError(message)
                warnings.warn(message + " — skipping it", stacklevel=2)
                if text in rows or not indices:
                    continue
            declared[text] = indices

        objects = list(self.group.children)
        self.entries = []
        for is_group, table in ((True, declared), (False, rows)):
            for label, indices in table.items():
                entry = LegendEntry(
                    label, indices,
                    [drawables[i] for i in indices],
                    [objects[i] if i < len(objects) else None for i in indices],
                    self.spec, is_group=is_group)
                entry.highlighted = (label in lit_groups if is_group
                                     else bool(lit_rows) and set(indices) <= lit_rows)
                self.entries.append(entry)
                self.scene.add(entry.group)
                self._bind(entry)
        self._refresh_all()

    def _lit(self) -> set:
        """Scene indices drawn in the highlight colour: a member of ANY lit row."""
        return {i for e in self.entries if e.highlighted for i in e.indices}

    def _refresh_all(self) -> None:
        """Repaint every row from the scene, since one drawable may be on several rows."""
        lit = self._lit()
        for entry in self.entries:
            entry.refresh(lit)

    def _structure(self) -> tuple:
        """A fingerprint of the rows' structure: each drawable's label and name, in order,
        and the groups. Any merge, split, rename, addition or reorder changes it."""
        groups = tuple((text, tuple(members))
                       for text, members in (self.spec.groups or {}).items())
        return (tuple((label_of(d), d.name) for d in self.scene_data.drawables), groups)

    def _state(self) -> tuple:
        """A fingerprint of the rows' appearance: visibility, colour and opacity."""
        return tuple((bool(d.visible), tuple(d.color), float(d.alpha))
                     for d in self.scene_data.drawables)

    def sync(self) -> bool:
        """Catch up with the scene if it changed since the last draw; cheap when it did not.

        Called from :meth:`draw`, so direct field edits such as
        ``scene.get("a").label = "Tm2"`` show up, in snapshots too. A changed structure
        rebuilds the rows; a changed appearance only reassigns materials. Returns whether
        anything was done.
        """
        structure = self._structure()
        if structure != self._structure_at:
            self.build_entries(strict=False)
            self.layout()
            return True

        state = self._state()
        if state != self._state_at:
            self._refresh_all()
            self._state_at = state
            return True
        return False

    # -- geometry --------------------------------------------------------------

    def measure(self) -> None:
        """Re-read the label widths, which the grid is derived from. Independent of the
        canvas size, unlike :meth:`relayout`."""
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

        As many columns as the entries need to fit the height, up to what the width budget
        allows; any remainder is shrunk to fit. (A single tall column shrunk to fit would
        leave tiny text beside empty margin.) An explicit ``Legend.width`` fixes the strip,
        and the column count follows.
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
        """Point the ortho camera at the content: 1 world unit = 1 px, shrunk uniformly to
        fit rather than clipped."""
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

        Pass a renderer for an offscreen snapshot. Re-reads the scene first (:meth:`sync`)
        and re-lays out only when the column count changes.
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
        """Left-click a row to hide or show it; right-click to highlight it."""
        def on_click(event) -> None:
            button = getattr(event, "button", 1)
            if button == 1:
                entry.toggle()
            elif button == 2:
                entry.toggle_highlight()
            else:
                return
            self._refresh_all()
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
        """This legend's rows, in order. The methods here take these labels, not drawable
        names."""
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

        Giving two rows the same new label merges them::

            view.legend.relabel({"1401 mesh": "Tm2", "1402 mesh": "Tm2"})

        Sets the members' ``label`` and leaves their names alone, so
        ``scene.get("1401 mesh")`` still works.
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
        """Recolour every drawable on a row. On a highlighted row the new colour appears
        when the highlight comes off."""
        rgba = to_rgba(color)
        for drawable in self.entry(label).drawables:
            drawable.color = rgba
        self._refresh_all()
        self._request_draw()
        return self

    def set_visible(self, label: str, visible: bool) -> "LegendOverlay":
        """Show or hide a whole row, by label."""
        entry = self.entry(label)
        for drawable in entry.drawables:
            drawable.visible = bool(visible)
        self._refresh_all()
        self._request_draw()
        return self

    def toggle(self, label: str) -> bool:
        state = self.entry(label).toggle()
        self._refresh_all()
        self._request_draw()
        return state

    # -- highlighting ----------------------------------------------------------

    @property
    def highlighted(self) -> list[str]:
        """The row labels currently drawn in the highlight colour."""
        return [e.text for e in self.entries if e.highlighted]

    def highlight(self, *labels: str, exclusive: bool = False) -> "LegendOverlay":
        """Draw these rows in the highlight colour, leaving their real colours alone.

        ``exclusive=True`` clears other highlights first; by default highlights accumulate,
        as right-clicking several rows does.
        """
        if exclusive:
            self.clear_highlights()
        for label in labels:
            self.entry(label).highlighted = True
        self._refresh_all()
        self._request_draw()
        return self

    def unhighlight(self, *labels: str) -> "LegendOverlay":
        """Put these rows back to their own colours. No labels means all of them."""
        chosen = [self.entry(x) for x in labels] if labels else list(self.entries)
        for entry in chosen:
            entry.highlighted = False
        self._refresh_all()
        self._request_draw()
        return self

    def clear_highlights(self) -> "LegendOverlay":
        return self.unhighlight()

    def toggle_highlight(self, label: str) -> bool:
        """What right-clicking a row does."""
        state = self.entry(label).toggle_highlight()
        self._refresh_all()
        self._request_draw()
        return state

    def refresh(self) -> "LegendOverlay":
        """Rebuild the rows from the scene, including their text, for changes made to the
        scene directly (such as ``scene.rename``)."""
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
