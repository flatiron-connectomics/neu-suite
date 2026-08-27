"""Arranging a set of objects: boxes in, offsets out.

Every layout here is the same operation — **assign each object's anchor a target point** —
and the two functions differ only in how those points are generated. :func:`superimpose_offsets`
sends every anchor to one point; :func:`arrange_offsets` sends them to a lattice.

Nothing in this module knows what a drawable is, let alone a renderer. It takes
:class:`~neu_lib.BBox` and returns :class:`~neu_lib.Vec3`, which is what lets the arithmetic
be tested without building a scene — the same seam the rest of the package rests on.

## They compose, and that is the point

    scene.superimpose(axes="z").arrange(along="x")

Align on depth, spread horizontally. **A layout that regularises all three axes throws away
whatever the axes meant** — if soma depth or layer position carries information, tiling in
every direction destroys it. So ``superimpose`` takes an axis subset and ``arrange`` only
touches the axes it lays out, which is what lets one axis stay honest while the objects are
separated for viewing.

## Two spacing policies, because they answer different questions

- **packed** (``spacing=None``, the default) uses each box's real extent, so objects of
  different sizes do not overlap and a big one simply takes more room.
- **fixed pitch** (``spacing=<nm>``) puts them on a regular lattice. What a figure with
  labels or a scale bar wants, where irregular spacing would read as meaningful.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from neu_lib import BBox, Vec3

#: Axis letters, in the package's zyx order. Indices are accepted too, but a letter is what
#: a caller means — ``along="x"`` rather than ``along=2``.
AXES = {"z": 0, "y": 1, "x": 2}

#: Where on a box a layout measures from. Kept in step with ``scene.ANCHORS``, which is the
#: same vocabulary applied to one object rather than a set.
ANCHORS = ("center", "min", "max")

#: Gap between packed boxes, as a fraction of the step. A fraction rather than nm because
#: the objects here are cells whose size varies by an order of magnitude between datasets,
#: and an absolute default would be either invisible or enormous.
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
    """``delta`` with every axis outside ``axes`` zeroed — the opt-out that keeps an axis
    at its true coordinates while the others are laid out."""
    return Vec3.of([delta[i] if i in axes else 0.0 for i in range(3)])


def superimpose_offsets(boxes: Sequence[BBox], *, anchor: str = "center",
                        axes: Any = None, at: Any = None) -> list[Vec3]:
    """Offsets putting every box's ``anchor`` on the same point.

    ``at`` defaults to the **first** box's anchor rather than the origin or the mean: the
    usual reason to superimpose is "show the others against this one", and moving the
    reference object as well makes the result harder to relate to anything. Pass ``at`` to
    say otherwise.

    ``axes`` restricts which axes move, so ``axes="z"`` aligns depth and leaves the rest
    where the data put them.
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

    ``spacing`` is a fixed pitch in nm; left ``None`` the boxes are **packed** by their own
    extents with ``gap`` between them as a fraction of the step, so unequal sizes do not
    overlap. ``wrap`` turns the row into a grid.

    ``align_cross`` puts every anchor on the layout line — which is what makes a row read as
    a row. Objects from different datasets have unrelated coordinates on the axes not being
    laid out, so leaving those alone scatters them; ``align_cross=False`` does that
    deliberately, for when an untouched axis is the point.

    ``origin`` is where the first anchor lands, defaulting to the first box's anchor so that
    an arrangement starts where the objects already are rather than jumping to nm zero.
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
    """The gap in nm: a fraction of the **mean** extent on that axis.

    The mean rather than each box's own size, so the spacing of a row is uniform — a gap
    that grew beside the big cells would read as grouping.
    """
    if not shapes:
        return 0.0
    return gap * (sum(float(s[axis]) for s in shapes) / len(shapes))


def _row_indices(count: int, wrap: int | None) -> list[list[int]]:
    if wrap is None:
        return [list(range(count))]
    return [list(range(i, min(i + wrap, count))) for i in range(0, count, wrap)]
