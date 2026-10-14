"""What to draw, described without reference to any renderer.

A :class:`Scene` is data: a list of drawables, each with geometry, a colour, an alpha and
a name, plus camera framing and legend settings. It imports nothing that renders, so what
a figure will contain can be checked without a GPU: fetch with ``sources``, build a
Scene, then render it with :func:`neu_draw.show`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import reduce
from typing import Any, Hashable, Iterable, Mapping, Optional, Sequence, Union

import numpy as np
from neu_lib import BBox, Mesh, Skeleton, Vec3

from .colors import RGBA, assign_colors, to_rgba

#: pygfx's ``MarkerShape`` names, written out so this module needs no renderer, and checked
#: here because pygfx rejects an unknown marker only at draw time. There is no plain
#: "triangle"; the four are directional.
MARKERS = (
    "circle", "square", "diamond", "ring", "cross", "plus", "tick",
    "triangle_up", "triangle_down", "triangle_left", "triangle_right",
    "asterisk6", "asterisk8", "heart", "spade", "club", "pin",
)

#: matplotlib's scatter shorthands (``'^'``, ``'s'``, ...).
MARKER_ALIASES = {
    "o": "circle", "s": "square", "D": "diamond", "d": "diamond",
    "^": "triangle_up", "v": "triangle_down", "<": "triangle_left",
    ">": "triangle_right", "+": "plus", "x": "cross", "*": "asterisk8",
    ".": "circle", "|": "tick",
}


def resolve_marker(marker: str) -> str:
    """Map a matplotlib shorthand to a pygfx marker name, and validate either way."""
    name = MARKER_ALIASES.get(marker, marker)
    if name not in MARKERS:
        raise ValueError(
            f"unknown marker {marker!r}; known: {', '.join(MARKERS)} "
            f"(or shorthands {', '.join(sorted(MARKER_ALIASES))})")
    return name


def label_of(drawable: Any) -> Optional[str]:
    """A drawable's legend row: its ``label``, or failing that its ``name``.

    A ``name`` is unique identity (what :meth:`Scene.get` and :meth:`Scene.set_color`
    resolve). A ``label`` is a classification that many drawables may share: forty bodies
    of one cell type can share a label, one legend row and one colour, and still be
    addressed individually by name.
    """
    label = getattr(drawable, "label", None)
    return label if label is not None else getattr(drawable, "name", None)


#: Where on a drawable's box an alignment measures from.
ANCHORS = ("center", "min", "max")


def anchor_of(box: BBox, anchor: str = "center") -> Vec3:
    """The point of ``box`` an alignment refers to."""
    if anchor not in ANCHORS:
        raise ValueError(f"unknown anchor {anchor!r}; known: {', '.join(ANCHORS)}")
    if anchor == "min":
        return Vec3.of(box.lo)
    if anchor == "max":
        return Vec3.of(box.hi)
    return Vec3.of(box.center)


class Placed:
    """Where a drawable is shown, as distinct from where its data says it is.

    The offset is a display transform; the geometry is never changed, so a mesh's ``bbox``
    still reports where the tissue is, and moving a large mesh copies nothing. Use
    :meth:`~Scene.bake` when the shifted coordinates themselves are the output.
    """

    #: Subclasses supply this; it is the box of the geometry as stored.
    @property
    def data_bbox(self) -> BBox:                                  # pragma: no cover
        raise NotImplementedError

    @property
    def bbox(self) -> BBox:
        """Where it is drawn: the data box moved by the offset."""
        return self.data_bbox.translate(self.offset_zyx_nm)       # type: ignore[attr-defined]

    def offset_by(self, delta: Any) -> "Placed":
        """A copy shifted by ``delta``, in addition to any existing offset."""
        current = self.offset_zyx_nm                              # type: ignore[attr-defined]
        return replace(self, offset_zyx_nm=current + Vec3.of(delta))  # type: ignore[type-var]

    def placed_at(self, offset: Any) -> "Placed":
        """A copy with the offset set outright, discarding any it had."""
        return replace(self, offset_zyx_nm=Vec3.of(offset))       # type: ignore[type-var]

    def centered(self, at: Any = (0.0, 0.0, 0.0), *, anchor: str = "center") -> "Placed":
        """A copy moved so its ``anchor`` sits at ``at`` (the origin by default)."""
        return self.offset_by(Vec3.of(at) - anchor_of(self.bbox, anchor))

    def aligned_to(self, other: Any, *, anchor: str = "center") -> "Placed":
        """A copy moved so its ``anchor`` coincides with ``other``'s.

        ``other`` is anything with a ``bbox``: another drawable, or a whole :class:`Scene`.
        """
        target = other.bbox if hasattr(other, "bbox") else other
        return self.offset_by(anchor_of(target, anchor) - anchor_of(self.bbox, anchor))


@dataclass
class MeshDrawable(Placed):
    """A surface. ``mesh`` is an :class:`~neu_lib.Mesh`, already in nm/zyx."""
    mesh: Mesh
    color: RGBA = (0.5, 0.5, 0.5, 1.0)
    alpha: float = 1.0
    name: Optional[str] = None
    label: Optional[str] = None
    visible: bool = True
    offset_zyx_nm: Vec3 = Vec3.zero()

    def __post_init__(self) -> None:
        self.offset_zyx_nm = Vec3.of(self.offset_zyx_nm)

    @property
    def data_bbox(self) -> BBox:
        return self.mesh.bbox

    def baked(self) -> "MeshDrawable":
        return replace(self, mesh=self.mesh.translate(self.offset_zyx_nm),
                       offset_zyx_nm=Vec3.zero())


@dataclass
class LinesDrawable(Placed):
    """A skeleton, drawn as independent segments, one per edge."""
    skeleton: Skeleton
    color: RGBA = (0.5, 0.5, 0.5, 1.0)
    alpha: float = 1.0
    thickness: float = 1.5
    name: Optional[str] = None
    label: Optional[str] = None
    visible: bool = True
    offset_zyx_nm: Vec3 = Vec3.zero()

    def __post_init__(self) -> None:
        self.offset_zyx_nm = Vec3.of(self.offset_zyx_nm)

    @property
    def data_bbox(self) -> BBox:
        return self.skeleton.bbox

    def baked(self) -> "LinesDrawable":
        return replace(self, skeleton=self.skeleton.translate(self.offset_zyx_nm),
                       offset_zyx_nm=Vec3.zero())


@dataclass
class PointsDrawable(Placed):
    """Synapses, or any other point set. zyx nm, like everything else."""
    positions_zyx_nm: np.ndarray
    color: RGBA = (0.5, 0.5, 0.5, 1.0)
    alpha: float = 1.0
    size: float = 8.0
    marker: str = "circle"
    name: Optional[str] = None
    label: Optional[str] = None
    visible: bool = True
    offset_zyx_nm: Vec3 = Vec3.zero()

    def __post_init__(self) -> None:
        arr = np.ascontiguousarray(self.positions_zyx_nm, dtype=np.float32)
        arr = arr.reshape(0, 3) if arr.size == 0 else arr
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"positions must be (N, 3) zyx, got shape {arr.shape}")
        self.positions_zyx_nm = arr
        self.marker = resolve_marker(self.marker)
        self.offset_zyx_nm = Vec3.of(self.offset_zyx_nm)

    @property
    def data_bbox(self) -> BBox:
        if not len(self.positions_zyx_nm):
            return BBox.empty(3)
        return BBox.from_points(self.positions_zyx_nm)

    def baked(self) -> "PointsDrawable":
        return replace(self,
                       positions_zyx_nm=self.positions_zyx_nm + self.offset_zyx_nm.array,
                       offset_zyx_nm=Vec3.zero())


@dataclass
class VolumeDrawable:
    """Reserved, not implemented: a future drawable for scalar fields (image slices,
    probability maps). Constructing one raises."""
    def __post_init__(self) -> None:
        raise NotImplementedError(
            "volumetric drawables are not implemented; see NEU-DRAW-PLAN.md. Meshes and "
            "skeletons cover segment morphology, and this slot is reserved for scalar "
            "fields (image slices, probability maps).")


Drawable = Union[MeshDrawable, LinesDrawable, PointsDrawable]

#: Drawable kinds by name, as used by `set_alpha`, the toolbar's opacity sliders and
#: `build_scene`'s name suffixes.
DRAWABLE_KINDS = {"mesh": MeshDrawable, "skeleton": LinesDrawable, "points": PointsDrawable}


@dataclass
class Camera:
    """How to frame the scene (intent, not a camera object).

    ``view_angle`` is the direction the camera looks from, in **xyz** like the renderer:
    the one non-zyx value in the package. ``target=None`` fits the scene's bounding box.
    """
    view_angle: tuple[float, float, float] = (-1.0, -1.0, -1.0)
    target: Optional[BBox] = None
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    zoom: float = 1.0


#: Which edge the legend strip docks against. Only vertical edges: names are long.
LEGEND_LOCATIONS = ("right", "left")


@dataclass
class Legend:
    """Whether to draw a legend, where, and how it looks. Intent, not objects.

    One row per distinct :func:`label_of`, in first-appearance order, hidden drawables
    included so they can be turned back on. Drawables sharing a label share a row.

    Sizes are in logical pixels: the legend is a fixed-size panel beside the scene.
    ``width=None`` fits the longest label. A legend is shown by default; pass
    ``Legend(visible=False)``, or ``show(scene, legend=False)``, to omit it.

    ``groups`` adds rows above the per-label ones, ``{row text: [drawable names]}``. A
    drawable may be in any number of groups as well as its own row, so one click can hide
    every Mi1 synapse while each cell's Mi1 set keeps its row. Members are names, so a
    relabel leaves groups intact and :meth:`Scene.rename` updates them.
    """
    visible: bool = True
    location: str = "right"
    font_size: float = 13.0
    width: Optional[float] = None
    #: Label colour. Light, because the default panel is dark.
    text_color: RGBA = (0.93, 0.93, 0.93, 1.0)
    #: The plate behind each row, which is also its click target.
    row_color: RGBA = (0.30, 0.30, 0.34, 0.55)
    #: The colour a right-clicked entry is drawn in, until right-clicked again. Change it
    #: if the palette already uses white, or a white body will not visibly highlight.
    highlight_color: RGBA = (1.0, 1.0, 1.0, 1.0)
    #: The strip's background; ``None`` takes the scene's ``background``, or near-black.
    #: Opaque, since a translucent strip would composite against the notebook page.
    panel_color: Optional[RGBA] = None
    #: Extra rows, each toggling a set of drawables by **name**. See the class docstring.
    groups: Optional[Mapping[str, Sequence[str]]] = None
    #: The plate behind a group row, lighter than ``row_color`` so groups read as headers.
    group_row_color: RGBA = (0.42, 0.42, 0.50, 0.75)

    def __post_init__(self) -> None:
        if self.groups is not None:
            if isinstance(self.groups, (str, bytes)) or not isinstance(self.groups, Mapping):
                raise TypeError("groups is a mapping {row text: [drawable names]}, not "
                                f"{type(self.groups).__name__}")
            groups = {}
            for text, members in self.groups.items():
                # A bare string would iterate as one-letter names that match nothing.
                if isinstance(members, str):
                    raise TypeError(f"group {text!r}: members must be a list of names, "
                                    f"not the string {members!r}")
                groups[str(text)] = tuple(str(m) for m in members)
            self.groups = groups
        self.group_row_color = to_rgba(self.group_row_color)
        if self.location not in LEGEND_LOCATIONS:
            raise ValueError(
                f"unknown legend location {self.location!r}; known: "
                f"{', '.join(LEGEND_LOCATIONS)}. A legend of body names needs a column, "
                f"so 'top' and 'bottom' are deliberately not offered.")
        if self.font_size <= 0:
            raise ValueError(f"font_size must be positive, got {self.font_size}")
        self.text_color = to_rgba(self.text_color)
        self.row_color = to_rgba(self.row_color)
        self.highlight_color = to_rgba(self.highlight_color)
        if self.panel_color is not None:
            self.panel_color = to_rgba(self.panel_color)


@dataclass
class Scene:
    """Everything a backend needs, and nothing it does not."""

    drawables: list[Drawable] = field(default_factory=list)
    camera: Camera = field(default_factory=Camera)
    legend: Legend = field(default_factory=Legend)
    background: Optional[RGBA] = None
    axes_visible: bool = False
    #: How an exported SVG/PDF nests drawables under their kind's group: ``drawable ->
    #: tuple of group names``, or ``None`` for no nesting (see ``backends/vector.py``).
    export_groups: Optional[Any] = None

    def __post_init__(self) -> None:
        # Not a field: kept out of __init__, __eq__ and repr, and a scene made by
        # `replace()` starts with no listeners rather than another view's.
        self._listeners: list = []

    # -- change notification ---------------------------------------------------

    def on_change(self, callback) -> "Scene":
        """Call ``callback()`` whenever a mutating method here changes the scene.

        The callback takes no arguments; the listener re-reads the scene. It fires for the
        mutating methods (:meth:`add`, :meth:`rename`, :meth:`relabel`, :meth:`set_color`,
        :meth:`recolor`, ...), not for a field set directly, such as
        ``scene.get("a").visible = False``; the backend re-reads the scene each frame to
        cover that.
        """
        self._listeners.append(callback)
        return self

    def _changed(self) -> None:
        for callback in list(self._listeners):
            callback()

    # -- assembly --------------------------------------------------------------

    def add(self, drawable: Drawable, rename: bool = False) -> "Scene":
        """Append a drawable. A duplicate name raises, or is renamed if asked.

        A name is identity, so two drawables may not share one; to share a legend row, give
        them the same ``label`` (see :func:`label_of`). ``rename=True`` appends a number
        instead (``"1401 (2)"``); a label is left alone. :func:`build_scene` avoids the
        common collision of a body's mesh and skeleton by suffixing ``mesh`` and
        ``skeleton``.
        """
        if drawable.name is not None:
            taken = {d.name for d in self.drawables if d.name is not None}
            if drawable.name in taken:
                if not rename:
                    raise ValueError(
                        f"a drawable named {drawable.name!r} is already here. A name is "
                        f"identity, so this is a collision, not a duplicate — pass "
                        f"rename=True, or give it a name of its own. To put two drawables "
                        f"on ONE legend row, give them the same `label` instead.")
                stem, n = drawable.name, 2
                while f"{stem} ({n})" in taken:
                    n += 1
                drawable = replace(drawable, name=f"{stem} ({n})")
        self.drawables.append(drawable)
        self._changed()
        return self

    def add_mesh(self, mesh: Mesh, *, rename: bool = False, **kwargs) -> "Scene":
        kwargs.setdefault("name", mesh.name)
        return self.add(MeshDrawable(mesh, **_normalized(kwargs)), rename=rename)

    def add_skeleton(self, skeleton: Skeleton, *, rename: bool = False,
                     **kwargs) -> "Scene":
        kwargs.setdefault("name", skeleton.name)
        return self.add(LinesDrawable(skeleton, **_normalized(kwargs)), rename=rename)

    def add_points(self, positions_zyx_nm: Any, *, rename: bool = False,
                   **kwargs) -> "Scene":
        return self.add(PointsDrawable(positions_zyx_nm, **_normalized(kwargs)),
                        rename=rename)

    # -- placement -------------------------------------------------------------

    def _mapped(self, fn) -> "Scene":
        """A new Scene with every drawable replaced by ``fn(drawable)``. Placement methods
        return new scenes, so each attempt at a layout leaves the original intact."""
        return replace(self, drawables=[fn(d) for d in self.drawables])

    def offset_by(self, delta: Any) -> "Scene":
        """Shift every drawable by ``delta``, keeping their relative positions."""
        return self._mapped(lambda d: d.offset_by(delta))

    def centered(self, at: Any = (0.0, 0.0, 0.0), *, anchor: str = "center") -> "Scene":
        """Move the scene as a whole, keeping the arrangement, so its anchor sits at
        ``at``."""
        return self.offset_by(Vec3.of(at) - anchor_of(self.bbox, anchor))

    def aligned_to(self, other: Any, *, anchor: str = "center") -> "Scene":
        """Move the scene as a whole so its anchor coincides with ``other``'s."""
        target = other.bbox if hasattr(other, "bbox") else other
        return self.offset_by(anchor_of(target, anchor) - anchor_of(self.bbox, anchor))

    # -- laying out a SET, as opposed to placing one thing ----------------------

    def _laid_out(self, offsets_for, *, key=None) -> "Scene":
        """Apply a layout to the visible drawables, in ``key`` order. Hidden drawables keep
        their offset and take no slot."""
        indices = [i for i, d in enumerate(self.drawables) if d.visible]
        if key is not None:
            indices.sort(key=lambda i: key(self.drawables[i]))
        offsets = offsets_for([self.drawables[i].bbox for i in indices])

        moved = list(self.drawables)
        for i, delta in zip(indices, offsets):
            moved[i] = moved[i].offset_by(delta)
        return replace(self, drawables=moved)

    def superimpose(self, *, anchor: str = "center", axes: Any = None,
                    at: Any = None, key=None) -> "Scene":
        """Put every drawable's ``anchor`` on one point, by default the first's.
        ``axes="z"`` aligns depth only and leaves the other axes as the data put them."""
        from .layout import superimpose_offsets

        return self._laid_out(
            lambda boxes: superimpose_offsets(boxes, anchor=anchor, axes=axes, at=at),
            key=key)

    def arrange(self, *, along: Any = "x", anchor: str = "center",
                spacing: Optional[float] = None, gap: float = 0.1,
                wrap: Optional[int] = None, down: Any = "z", origin: Any = None,
                align_cross: bool = True, key=None) -> "Scene":
        """Lay the drawables out along an axis, wrapping into a grid if ``wrap`` is given.

        Packed by each object's extent unless ``spacing`` gives a fixed pitch in nm. Combine
        with :meth:`superimpose`::

            scene.superimpose(axes="z").arrange(along="x")

        ``key`` sorts the drawables first; ``key=lambda d: -d.bbox.size`` puts the largest
        first.
        """
        from .layout import arrange_offsets

        return self._laid_out(
            lambda boxes: arrange_offsets(
                boxes, along=along, anchor=anchor, spacing=spacing, gap=gap, wrap=wrap,
                down=down, origin=origin, align_cross=align_cross),
            key=key)

    def bake(self) -> "Scene":
        """Fold every offset into the geometry, leaving all offsets zero.

        For when the shifted coordinates are the output, such as writing the arrangement as
        meshes. Once baked, the geometry no longer reports where the tissue is.
        """
        return self._mapped(lambda d: d.baked())

    # -- queries ---------------------------------------------------------------

    @property
    def bbox(self) -> BBox:
        """Union over visible drawables, in nm, including their offsets."""
        boxes = [d.bbox for d in self.drawables if d.visible]
        return reduce(BBox.union, boxes, BBox.empty(3))

    @property
    def names(self) -> list[str]:
        """Every drawable's name, in scene order. Unique by construction."""
        return [d.name for d in self.drawables if d.name is not None]

    @property
    def labels(self) -> list[str]:
        """The legend's rows: each distinct :func:`label_of`, in first-appearance order."""
        seen: dict[str, None] = {}
        for drawable in self.drawables:
            label = label_of(drawable)
            if label is not None:
                seen.setdefault(label, None)
        return list(seen)

    def __len__(self) -> int:
        return len(self.drawables)

    def __iter__(self):
        return iter(self.drawables)

    def get(self, name: str) -> Drawable:
        for drawable in self.drawables:
            if drawable.name == name:
                return drawable
        raise KeyError(f"no drawable named {name!r}; have {self.names}")

    def by_label(self, label: str) -> list[Drawable]:
        """Every drawable in one legend row."""
        found = [d for d in self.drawables if label_of(d) == label]
        if not found:
            raise KeyError(f"no drawable labelled {label!r}; have {self.labels}")
        return found

    def rename(self, old: Union[str, Mapping[str, str]],
               new: Optional[str] = None) -> "Scene":
        """Change drawables' names (their unique identity).

        Takes a mapping or one ``(old, new)`` pair. A mapping is checked as one batch, so a
        swap (``{"a": "b", "b": "a"}``) works. To change legend text, use :meth:`relabel`;
        a rename changes a row only for a drawable without a label.
        """
        mapping = dict(old) if isinstance(old, Mapping) else {old: new}
        drawables = [self.get(name) for name in mapping]

        after = [mapping.get(d.name, d.name) for d in self.drawables
                 if d.name is not None]
        clashes = {n for n in after if after.count(n) > 1}
        if clashes:
            raise ValueError(
                f"renaming would leave {len(clashes)} name(s) shared: "
                f"{sorted(clashes)}. Names are a drawable's identity, so this is a "
                f"collision — a shared legend row is what `relabel` is for.")

        for drawable in drawables:
            drawable.name = mapping[drawable.name]
        # Groups are keyed on names, so they follow a rename rather than losing a member.
        if self.legend is not None and self.legend.groups:
            self.legend.groups = {text: tuple(mapping.get(m, m) for m in members)
                                  for text, members in self.legend.groups.items()}
        self._changed()
        return self

    def relabel(self, names: Union[str, Mapping[str, str]],
                label: Optional[str] = None) -> "Scene":
        """Set drawables' legend **labels**, keyed on their names.

        Takes a mapping or one ``(name, label)`` pair. Giving two drawables the same label
        merges their legend rows:

            scene.relabel({"1401 mesh": "Tm2", "1402 mesh": "Tm2"})

        A label of ``None`` puts a drawable back on a row of its own.
        """
        mapping = dict(names) if isinstance(names, Mapping) else {names: label}
        for name, value in mapping.items():
            self.get(name).label = value
        self._changed()
        return self

    def group(self, text: str, members: Union[Iterable[str], Any]) -> "Scene":
        """Add (or replace) a legend **group** row: one click for many drawables.

        ``members`` is drawable names, or a predicate on drawables, such as
        ``scene.group("Mi1", lambda d: d.name.endswith(": Mi1"))``, resolved to names now.
        A group matching nothing raises. See :class:`Legend`.
        """
        if callable(members):
            names = [d.name for d in self.drawables if d.name is not None and members(d)]
        else:
            if isinstance(members, str):
                raise TypeError(f"members must be a list of names or a predicate, not "
                                f"the string {members!r}")
            names = [str(m) for m in members]
            unknown = [m for m in names if m not in set(self.names)]
            if unknown:
                raise KeyError(f"group {text!r} names drawables that are not here: "
                               f"{unknown}; have {self.names}")
        if not names:
            raise ValueError(f"group {text!r} matches no drawables")
        groups = dict(self.legend.groups or {})
        groups[str(text)] = tuple(names)
        self.legend.groups = groups
        self._changed()
        return self

    def set_alpha(self, alpha: float, *, kind: Optional[str] = None,
                  names: Optional[Iterable[str]] = None) -> "Scene":
        """Set the opacity of every drawable of one ``kind``, or of the ``names`` given.

        ``kind`` is ``"mesh"``, ``"skeleton"`` or ``"points"``; with neither argument, all
        drawables. Each is set to ``alpha`` outright, not scaled, and the change is stored
        on the drawables, so ``view.reset()`` and saved figures keep it.
        """
        alpha = float(alpha)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        chosen = self.of_kind(kind) if kind is not None else list(self.drawables)
        if names is not None:
            wanted = {str(n) for n in names}
            unknown = wanted - set(self.names)
            if unknown:
                raise KeyError(f"no drawables named {sorted(unknown)}")
            chosen = [d for d in chosen if d.name in wanted]
        for drawable in chosen:
            drawable.alpha = alpha
        self._changed()
        return self

    def of_kind(self, kind: str) -> list[Drawable]:
        """The drawables of one kind: ``"mesh"``, ``"skeleton"`` or ``"points"``."""
        if kind not in DRAWABLE_KINDS:
            raise ValueError(f"kind is one of {', '.join(DRAWABLE_KINDS)}, not {kind!r}")
        return [d for d in self.drawables if isinstance(d, DRAWABLE_KINDS[kind])]

    def set_color(self, name: str, color: Any) -> "Scene":
        """Recolour one drawable, leaving every other colour unchanged (unlike
        :meth:`recolor`, which reassigns the palette over the whole scene)."""
        self.get(name).color = to_rgba(color)
        self._changed()
        return self

    def recolor(self, colors: Optional[Mapping[Hashable, Any] | Any] = None,
                palette: Optional[Sequence[Any]] = None,
                by: str = "name") -> "Scene":
        """Reassign every drawable's colour in one pass.

        Colours are chosen over the whole set at once, so the palette's distinct colours go
        to distinct drawables. ``by="label"`` gives one colour per legend row, so a row's
        swatch matches all its drawables; unlabelled drawables fall back to their name.
        ``colors`` is keyed by name or label accordingly.
        """
        if by not in ("name", "label"):
            raise ValueError(f"`by` is 'name' or 'label', not {by!r}")
        key = (lambda d: d.name) if by == "name" else label_of

        chosen = assign_colors(self.names if by == "name" else self.labels,
                               explicit=colors, palette=palette)
        for i, drawable in enumerate(self.drawables):
            if key(drawable) in chosen:
                self.drawables[i] = replace(drawable, color=chosen[key(drawable)])
        self._changed()
        return self

    def __repr__(self) -> str:
        kinds: dict[str, int] = {}
        for d in self.drawables:
            kinds[type(d).__name__] = kinds.get(type(d).__name__, 0) + 1
        body = ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())) or "empty"
        return f"Scene({body}, bbox={self.bbox!r})"


def build_scene(meshes: Iterable[Mesh] = (), skeletons: Iterable[Skeleton] = (),
                points: Mapping[str, Any] | None = None, *,
                colors: Optional[Mapping[Hashable, Any] | Any] = None,
                palette: Optional[Sequence[Any]] = None,
                alpha: Optional[float] = None,
                labels: Optional[Mapping[Hashable, str]] = None,
                color_by: Optional[str] = None,
                point_color: Any = None, **scene_kwargs) -> Scene:
    """Assemble a scene and colour it in one call.

    ``alpha`` defaults to 1 for a single body and 0.8 for several, so overlapping surfaces
    do not hide each other.

    A body's mesh and skeleton are both named after the body; when a name occurs more than
    once, both get the representation appended (``"1401 mesh"``, ``"1401 skeleton"``).

    ``labels`` groups the legend, keyed on each item's own name (the body id, before any
    suffix), matched by value or by ``str()``. Unlisted items keep their own row::

        build_scene(meshes=ms, skeletons=sks, labels={10014014: "Tm2", 10017394: "Tm2"})

    Passing ``labels`` also colours by label (see :meth:`Scene.recolor`); ``color_by="name"``
    overrides that.
    """
    scene = Scene(**scene_kwargs)
    meshes, skeletons = list(meshes), list(skeletons)
    if alpha is None:
        alpha = 1.0 if len(meshes) + len(skeletons) <= 1 else 0.8

    counts: dict[Any, int] = {}
    for item in (*meshes, *skeletons):
        if item.name is not None:
            counts[item.name] = counts.get(item.name, 0) + 1
    for name in (points or {}):
        counts[name] = counts.get(name, 0) + 1

    # Both the raw key and its string form, so {10014014: "Tm2"} finds a mesh named
    # "10014014" and vice versa.
    wanted: dict[Any, str] = {}
    for key, value in (labels or {}).items():
        wanted[key] = value
        wanted[str(key)] = value

    def name_for(item, kind: str) -> Optional[str]:
        if item.name is None:
            return None
        return f"{item.name} {kind}" if counts.get(item.name, 0) > 1 else item.name

    for mesh in meshes:
        scene.add_mesh(mesh, alpha=alpha, name=name_for(mesh, "mesh"),
                       label=wanted.get(mesh.name), rename=True)
    for skeleton in skeletons:
        scene.add_skeleton(skeleton, alpha=alpha, name=name_for(skeleton, "skeleton"),
                           label=wanted.get(skeleton.name), rename=True)
    for name, positions in (points or {}).items():
        scene.add_points(positions, name=name, label=wanted.get(name), rename=True,
                         **({"color": point_color} if point_color is not None else {}))

    if color_by is None:
        color_by = "label" if labels else "name"
    return scene.recolor(colors, palette=palette, by=color_by)


def _normalized(kwargs: dict) -> dict:
    if "color" in kwargs and kwargs["color"] is not None:
        kwargs["color"] = to_rgba(kwargs["color"])
    else:
        kwargs.pop("color", None)
    return kwargs
