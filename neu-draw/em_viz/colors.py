"""Colour normalisation and assignment. Pure — no matplotlib, no cmap, no renderer.

The predecessor reached into matplotlib for its palette (``plt.get_cmap('tab10')``) and
into ``cmap`` for every single colour, which meant assigning colours to a scene pulled a
plotting library and a colormap library into the import graph of code that does no
plotting. A palette is *data*; the sixteen values below are written out, and everything
here is arithmetic on RGBA tuples.

Names resolve from a small table covering matplotlib's single-letter shorthands and the
tab10 names — which is what the old call sites actually used (``'r'``, ``'b'``). Anything
unrecognised is handed to the ``cmap`` library **if it happens to be installed**, so the
full CSS vocabulary works in a rendering environment without being required in a headless
one.
"""

from __future__ import annotations

from itertools import cycle
from typing import Any, Hashable, Iterable, Mapping, Optional, Sequence

RGBA = tuple[float, float, float, float]

#: tab10 followed by Dark2, the pairing the predecessor used. Qualitative, and ordered so
#: the first few are maximally distinct — a scene usually shows two or three bodies.
QUALITATIVE: tuple[RGBA, ...] = (
    (0.1216, 0.4667, 0.7059, 1.0),   # tab:blue
    (1.0000, 0.4980, 0.0549, 1.0),   # tab:orange
    (0.1725, 0.6275, 0.1725, 1.0),   # tab:green
    (0.8392, 0.1529, 0.1569, 1.0),   # tab:red
    (0.5804, 0.4039, 0.7412, 1.0),   # tab:purple
    (0.5490, 0.3373, 0.2941, 1.0),   # tab:brown
    (0.8902, 0.4667, 0.7608, 1.0),   # tab:pink
    (0.4980, 0.4980, 0.4980, 1.0),   # tab:gray
    (0.7373, 0.7412, 0.1333, 1.0),   # tab:olive
    (0.0902, 0.7451, 0.8118, 1.0),   # tab:cyan
    (0.1059, 0.6196, 0.4667, 1.0),   # Dark2 teal
    (0.8510, 0.3725, 0.0078, 1.0),   # Dark2 orange
    (0.4588, 0.4392, 0.7020, 1.0),   # Dark2 purple
    (0.9059, 0.1608, 0.5412, 1.0),   # Dark2 magenta
    (0.4000, 0.6510, 0.1176, 1.0),   # Dark2 green
    (0.9020, 0.6706, 0.0078, 1.0),   # Dark2 gold
)

_NAMED: dict[str, RGBA] = {
    # matplotlib single-letter shorthands, which is what the old call sites passed
    "b": (0.0, 0.0, 1.0, 1.0), "g": (0.0, 0.5, 0.0, 1.0), "r": (1.0, 0.0, 0.0, 1.0),
    "c": (0.0, 0.75, 0.75, 1.0), "m": (0.75, 0.0, 0.75, 1.0),
    "y": (0.75, 0.75, 0.0, 1.0), "k": (0.0, 0.0, 0.0, 1.0), "w": (1.0, 1.0, 1.0, 1.0),
    "blue": (0.0, 0.0, 1.0, 1.0), "green": (0.0, 0.5, 0.0, 1.0),
    "red": (1.0, 0.0, 0.0, 1.0), "cyan": (0.0, 1.0, 1.0, 1.0),
    "magenta": (1.0, 0.0, 1.0, 1.0), "yellow": (1.0, 1.0, 0.0, 1.0),
    "black": (0.0, 0.0, 0.0, 1.0), "white": (1.0, 1.0, 1.0, 1.0),
    "orange": (1.0, 0.4980, 0.0549, 1.0), "purple": (0.5804, 0.4039, 0.7412, 1.0),
    "gray": (0.4980, 0.4980, 0.4980, 1.0), "grey": (0.4980, 0.4980, 0.4980, 1.0),
}
_NAMED.update({f"tab:{n}": QUALITATIVE[i] for i, n in enumerate(
    ("blue", "orange", "green", "red", "purple",
     "brown", "pink", "gray", "olive", "cyan"))})


def to_rgba(color: Any, alpha: Optional[float] = None) -> RGBA:
    """Normalise a colour to ``(r, g, b, a)`` floats in ``[0, 1]``.

    Accepts a name, ``#rgb``/``#rrggbb``/``#rrggbbaa``, or a 3- or 4-sequence of floats
    in ``[0, 1]``. ``alpha`` overrides whatever the colour carried.
    """
    rgba = _resolve(color)
    if alpha is not None:
        rgba = (*rgba[:3], _unit(alpha, "alpha"))
    return rgba


def _resolve(color: Any) -> RGBA:
    if isinstance(color, str):
        key = color.strip().lower()
        if key in _NAMED:
            return _NAMED[key]
        if key.startswith("#"):
            return _from_hex(key)
        # Optional, and only for the long tail. The error is raised HERE either way:
        # whether `cmap` happens to be installed decides which names work, and it must
        # not also decide what a caller has to catch.
        try:
            import cmap as cmap_lib
        except ImportError:
            pass
        else:
            try:
                return tuple(float(v) for v in cmap_lib.Color(color).rgba)  # type: ignore[return-value]
            except Exception:
                pass
        raise ValueError(
            f"unknown colour {color!r}. Known names: {', '.join(sorted(_NAMED))}. "
            f"Hex (#rgb, #rrggbb, #rrggbbaa) and RGB(A) sequences always work; with "
            f"`cmap` installed the full CSS vocabulary does too.")

    values = tuple(float(v) for v in color)
    if len(values) == 3:
        values = (*values, 1.0)
    if len(values) != 4:
        raise ValueError(f"a colour is 3 or 4 components, got {len(values)}: {color!r}")
    return tuple(_unit(v, "colour component") for v in values)  # type: ignore[return-value]


def _from_hex(text: str) -> RGBA:
    digits = text[1:]
    if len(digits) == 3:
        digits = "".join(c * 2 for c in digits)
    if len(digits) not in (6, 8):
        raise ValueError(f"hex colour must be #rgb, #rrggbb or #rrggbbaa, got {text!r}")
    try:
        parts = [int(digits[i:i + 2], 16) / 255.0 for i in range(0, len(digits), 2)]
    except ValueError:
        raise ValueError(f"not a hex colour: {text!r}") from None
    return (*parts, 1.0)[:4] if len(parts) == 3 else tuple(parts)  # type: ignore[return-value]


def _unit(value: Any, label: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be in [0, 1], got {number}")
    return number


def assign_colors(names: Iterable[Hashable],
                  explicit: Optional[Mapping[Hashable, Any] | Any] = None,
                  palette: Optional[Sequence[Any]] = None) -> dict[Hashable, RGBA]:
    """One colour per name: explicit where given, otherwise cycling the palette.

    ``explicit`` may be a mapping, a single colour for everything, or a sequence to use
    as the palette instead. This replaces the branching the predecessor grew in
    ``render_bodies_rois_3d``, where the same three cases were disentangled inline
    against a ``colors`` argument that could be any of them.

    **Palette position depends only on the names that are not explicit.** Fixing one
    body's colour therefore does not shift every other body's, which is what makes a
    figure reproducible while you iterate on it.
    """
    names = list(dict.fromkeys(names))          # de-duplicate, keep order
    fixed: dict[Hashable, RGBA] = {}

    if explicit is None:
        pass
    elif isinstance(explicit, Mapping):
        fixed = {k: to_rgba(v) for k, v in explicit.items()}
    elif isinstance(explicit, str):
        fixed = {n: to_rgba(explicit) for n in names}
    else:
        candidate = list(explicit)
        # A single RGB(A) sequence is a colour, not a palette of numbers.
        if candidate and all(isinstance(v, (int, float)) for v in candidate):
            fixed = {n: to_rgba(candidate) for n in names}
        else:
            palette = candidate if palette is None else palette

    wheel = cycle([to_rgba(c) for c in (palette if palette else QUALITATIVE)])
    return {name: fixed[name] if name in fixed else next(wheel) for name in names}
