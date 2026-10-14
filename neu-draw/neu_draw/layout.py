"""Arranging a set of objects: boxes (:class:`~neu_lib.BBox`) in, offsets
(:class:`~neu_lib.Vec3`) out. Knows nothing about drawables or renderers.

:func:`superimpose_offsets` sends every object's anchor to one point;
:func:`arrange_offsets` lays them out on a row or grid. Both act only on the axes asked
for, so they compose without discarding coordinates that carry meaning, such as depth::

    scene.superimpose(axes="z").arrange(along="x")

Spacing is either packed by each box's real extent (``spacing=None``, the default) or a
fixed pitch in nm, for figures where uneven spacing would look meaningful.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from neu_lib import BBox, Vec3

#: Axis letters, zyx. Indices are accepted too.
AXES = {"z": 0, "y": 1, "x": 2}

#: Where on a box a layout measures from. Matches ``scene.ANCHORS``.
ANCHORS = ("center", "min", "max")

#: Gap between packed boxes, as a fraction of the mean extent rather than nm, since cell
#: sizes vary by an order of magnitude between datasets.
DEFAULT_GAP = 0.1


def axis_index(axis: Any) -> int:
    """``"x"`` or ``2`` to an index into a zyx triple."""
    if isinstance(axis, str):
        key = axis.lower()
        if key not in AXES:
            raise ValueError(f"unknown axis {axis!r}; use one of {', '.join(AXES)}")
        return AXES[key]
    index = int(axis)
    if not 0 <= index < 3:
        raise ValueError(f"axis index {index} is out of range for a zyx triple")
    return index


def anchor_point(box: BBox, anchor: str = "center") -> Vec3:
    """The point of ``box`` a layout places."""
    if anchor not in ANCHORS:
        raise ValueError(f"unknown anchor {anchor!r}; known: {', '.join(ANCHORS)}")
    if anchor == "min":
        return Vec3.of(box.lo)
    if anchor == "max":
        return Vec3.of(box.hi)
    return Vec3.of(box.center)


def _axis_set(axes: Any) -> tuple[int, ...]:
    """``"zx"``, ``("z", "x")`` or ``(0, 2)`` to sorted, de-duplicated indices."""
    if axes is None:
        return (0, 1, 2)
    items: Iterable[Any] = axes if not isinstance(axes, str) else tuple(axes)
    return tuple(sorted({axis_index(a) for a in items}))


def _masked(delta: Vec3, axes: tuple[int, ...]) -> Vec3:
    """``delta`` with every axis outside ``axes`` zeroed."""
    return Vec3.of([delta[i] if i in axes else 0.0 for i in range(3)])


def superimpose_offsets(boxes: Sequence[BBox], *, anchor: str = "center",
                        axes: Any = None, at: Any = None) -> list[Vec3]:
    """Offsets putting every box's ``anchor`` on the same point.

    The point is ``at``, by default the first box's anchor, so the first object stays put
    and the others move onto it. ``axes`` restricts which axes move: ``axes="z"`` aligns
    depth and leaves the rest where the data put them.
    """
    if not boxes:
        return []
    wanted = _axis_set(axes)
    target = Vec3.of(at) if at is not None else anchor_point(boxes[0], anchor)
    return [_masked(target - anchor_point(box, anchor), wanted) for box in boxes]


def arrange_offsets(boxes: Sequence[BBox], *, along: Any = "x", anchor: str = "center",
                    spacing: float | None = None, gap: float = DEFAULT_GAP,
                    wrap: int | None = None, down: Any = "z",
                    origin: Any = None, align_cross: bool = True) -> list[Vec3]:
    """Offsets laying the boxes out along ``along``, wrapping to ``down`` every ``wrap``.

    ``spacing`` is a fixed pitch in nm. Left ``None``, the boxes are packed by their own
    extents with ``gap`` between them (a fraction of the mean extent), so unequal sizes do
    not overlap. ``wrap`` turns the row into a grid.

    ``align_cross=True`` also lines the anchors up on the other axes, so a row reads as a
    row even for objects from different datasets; ``False`` leaves those axes untouched.
    ``origin`` is where the first anchor lands, by default where it already is.
    """
    if not boxes:
        return []
    if gap < 0:
        raise ValueError(f"gap is a fraction of the step and must not be negative: {gap}")
    if spacing is not None and spacing <= 0:
        raise ValueError(f"spacing is a pitch in nm and must be positive: {spacing}")
    if wrap is not None and wrap < 1:
        raise ValueError(f"wrap is a count per row and must be at least 1: {wrap}")

    major, minor = axis_index(along), axis_index(down)
    if wrap is not None and major == minor:
        raise ValueError(
            f"a grid needs two different axes, got along={along!r} and down={down!r}. "
            f"With one axis leave `wrap` unset and it is a row.")

    start = Vec3.of(origin) if origin is not None else anchor_point(boxes[0], anchor)
    anchors = [anchor_point(box, anchor) for box in boxes]
    shapes = [box.shape for box in boxes]

    # Where each box's anchor sits inside its own extent, so packing can place boxes
    # edge-to-edge whatever anchor was asked for.
    lead = [a[major] - box.lo[major] for a, box in zip(anchors, boxes)]
    trail = [box.hi[major] - a[major] for a, box in zip(anchors, boxes)]

    rows = _row_indices(len(boxes), wrap)
    positions: list[Vec3] = [Vec3.zero()] * len(boxes)
    row_extent = 0.0
    cursor_minor = 0.0

    for row in rows:
        cursor_major = 0.0
        for place, i in enumerate(row):
            if spacing is not None:
                cursor_major = place * float(spacing)
            elif place:
                cursor_major += trail[row[place - 1]] + lead[i] + _gap_of(shapes, gap, major)
            target = list(start)
            target[major] = start[major] + cursor_major
            target[minor] = start[minor] + cursor_minor
            positions[i] = Vec3.of(target)
        row_extent = max((shapes[i][minor] for i in row), default=0.0)
        cursor_minor += row_extent + _gap_of(shapes, gap, minor)

    if align_cross:
        wanted = (0, 1, 2)
    else:
        wanted = (major, minor) if wrap is not None else (major,)
    return [_masked(target - here, wanted)
            for target, here in zip(positions, anchors)]


def _gap_of(shapes: Sequence[Sequence[int]], gap: float, axis: int) -> float:
    """The gap in nm: a fraction of the mean extent on that axis, so gaps are uniform
    along a row."""
    if not shapes:
        return 0.0
    return gap * (sum(float(s[axis]) for s in shapes) / len(shapes))


def _row_indices(count: int, wrap: int | None) -> list[list[int]]:
    if wrap is None:
        return [list(range(count))]
    return [list(range(i, min(i + wrap, count))) for i in range(0, count, wrap)]
