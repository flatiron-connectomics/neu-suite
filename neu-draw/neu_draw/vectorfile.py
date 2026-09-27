"""A vector figure as a tree of named groups, and its two writers: SVG and PDF.

No renderer and no dependency beyond numpy and the standard library (plus imageio, only
to encode an SVG's embedded PNGs). :mod:`neu_draw.backends.vector` builds the tree from a
live view; everything here is about writing it down, so the two formats cannot drift — a
group is a group in both, and a shape means the same thing in both.

Coordinates are **logical pixels, y down**, as on the canvas, and one pixel is written as
one point. Colours are RGBA floats in 0-1; an alpha below 1 is a real opacity in both
formats, and both composite in sRGB, which is what the mesh images are solved against.

## Which format for which editor

**Illustrator keeps named groups from the SVG and not from the PDF.** A PDF's layers
(optional content groups) are Acrobat's; Illustrator opens a PDF it did not write onto a
single layer. So the SVG is written for Illustrator's importer in particular — SVG 1.1,
``xlink:href`` on images (a bare SVG 2 ``href`` leaves them blank), no ``clipPath`` wrapper
(it would become a clip group around everything), a baseline offset instead of
``dominant-baseline``, and group ids in Illustrator's own ``_xHH_`` escaping, which it
decodes back into the layer name. The PDF carries the same hierarchy as nested optional
content, which Acrobat shows as a layer tree.
"""

from __future__ import annotations

import base64
import html
import zlib
from dataclasses import dataclass, field
from typing import Iterable, Optional, Union

import numpy as np

RGBA = tuple

#: The PDF writes Helvetica (one of the 14 standard fonts, so nothing is embedded);
#: the SVG names what the canvas actually uses first.
FONT_FAMILY = "'Noto Sans', Helvetica, Arial, sans-serif"
#: Where a text baseline sits below the vertical centre of its line, in ems — what
#: ``dominant-baseline: central`` would do, which Illustrator ignores.
BASELINE = 0.35
#: Bezier handle length for a quarter circle.
KAPPA = 0.5522847498


# --------------------------------------------------------------------------- #
# the tree
# --------------------------------------------------------------------------- #

@dataclass
class Rect:
    x: float
    y: float
    w: float
    h: float
    fill: RGBA


@dataclass
class Markers:
    """Many point markers of one shape and style — one drawable's synapse set."""
    shape: str
    xy: np.ndarray                    # (N, 2)
    size: float                       # diameter, px
    fill: RGBA
    stroke: Optional[RGBA] = None
    stroke_width: float = 0.0


@dataclass
class Segments:
    """Independent line segments, one stroked path — a skeleton."""
    segments: np.ndarray              # (N, 2, 2): start and end, xy
    stroke: RGBA
    width: float
    cap: str = "round"


@dataclass
class Image:
    rgba: np.ndarray                  # (h, w, 4) uint8, STRAIGHT alpha
    x: float
    y: float
    w: float
    h: float


@dataclass
class Text:
    x: float
    y: float                          # the vertical CENTRE of the line
    text: str
    size: float
    fill: RGBA


Shape = Union[Rect, Markers, Segments, Image, Text]


@dataclass
class Group:
    name: str
    children: list = field(default_factory=list)

    def child(self, name: str) -> "Group":
        """The sub-group called ``name``, created at the end if it is not there yet."""
        for c in self.children:
            if isinstance(c, Group) and c.name == name:
                return c
        made = Group(name)
        self.children.append(made)
        return made

    def nested(self, path: Iterable[str]) -> "Group":
        group = self
        for name in path:
            group = group.child(str(name))
        return group

    def prune(self) -> "Group":
        """Drop empty sub-groups, which an editor shows as layers holding nothing."""
        self.children = [c.prune() if isinstance(c, Group) else c for c in self.children]
        self.children = [c for c in self.children
                         if not (isinstance(c, Group) and not c.children)]
        return self

    def walk(self):
        yield self
        for c in self.children:
            if isinstance(c, Group):
                yield from c.walk()


def _circle_polygons(shape: str, r: float) -> Optional[list[tuple[float, float]]]:
    return {
        "diamond": [(0, -r), (r, 0), (0, r), (-r, 0)],
        "triangle_up": [(0, -r), (r, r), (-r, r)],
        "triangle_down": [(0, r), (r, -r), (-r, -r)],
        "triangle_left": [(-r, 0), (r, -r), (r, r)],
        "triangle_right": [(r, 0), (-r, -r), (-r, r)],
        "square": [(-r, -r), (r, -r), (r, r), (-r, r)],
    }.get(shape)


# --------------------------------------------------------------------------- #
# SVG
# --------------------------------------------------------------------------- #

def _num(v: float) -> str:
    return f"{float(v):.2f}".rstrip("0").rstrip(".")


def _hex(color: RGBA) -> str:
    return "#{:02x}{:02x}{:02x}".format(
        *(int(round(min(max(float(c), 0.0), 1.0) * 255)) for c in tuple(color)[:3]))


def _alpha(color: RGBA) -> float:
    color = tuple(color)
    return float(color[3]) if len(color) > 3 else 1.0


def _svg_paint(attr: str, color: Optional[RGBA]) -> str:
    if color is None:
        return f'{attr}="none"'
    a = _alpha(color)
    return f'{attr}="{_hex(color)}"' + (f' {attr}-opacity="{a:.4g}"' if a < 0.999 else "")


def illustrator_id(name: str) -> str:
    """``name`` in Illustrator's id escaping: anything but a letter, digit, ``-`` or
    ``.`` becomes ``_xHH_``, as does a leading digit. Illustrator writes ids this way and
    decodes them on import, so the layer panel reads ``T4a 78044864 : Mi1 (17)``."""
    out = []
    for i, ch in enumerate(str(name)):
        if ch.isascii() and (ch.isalpha() or ch in "-." or (ch.isdigit() and i)):
            out.append(ch)
        else:
            out.append(f"_x{ord(ch):X}_")
    return "".join(out) or "_x5F_"


class _Unique:
    def __init__(self) -> None:
        self.used: set[str] = set()

    def __call__(self, stem: str) -> str:
        candidate, n = stem, 1
        while candidate in self.used:
            candidate, n = f"{stem}_{n}_", n + 1
        self.used.add(candidate)
        return candidate


def _svg_shape(shape: Shape) -> str:
    if isinstance(shape, Rect):
        return (f'<rect x="{_num(shape.x)}" y="{_num(shape.y)}" width="{_num(shape.w)}" '
                f'height="{_num(shape.h)}" {_svg_paint("fill", shape.fill)}/>')
    if isinstance(shape, Segments):
        d = "".join(f"M{_num(a[0])} {_num(a[1])}L{_num(b[0])} {_num(b[1])}"
                    for a, b in shape.segments)
        return (f'<path d="{d}" {_svg_paint("fill", None)} '
                f'{_svg_paint("stroke", shape.stroke)} stroke-width="{_num(shape.width)}" '
                f'stroke-linecap="{shape.cap}" stroke-linejoin="round"/>')
    if isinstance(shape, Markers):
        paint = _svg_paint("fill", shape.fill)
        if shape.stroke is not None and shape.stroke_width > 0:
            paint += (f' {_svg_paint("stroke", shape.stroke)} '
                      f'stroke-width="{_num(shape.stroke_width)}"')
        r = shape.size / 2.0
        poly = _circle_polygons(shape.shape, r)
        items = []
        for x, y in shape.xy:
            if poly is None:
                items.append(f'<circle cx="{_num(x)}" cy="{_num(y)}" r="{_num(r)}" {paint}/>')
            else:
                pts = " ".join(f"{_num(x + dx)},{_num(y + dy)}" for dx, dy in poly)
                items.append(f'<polygon points="{pts}" {paint}/>')
        return "\n".join(items)
    if isinstance(shape, Image):
        from imageio import v3 as iio

        data = base64.b64encode(iio.imwrite("<bytes>", shape.rgba, extension=".png"))
        return (f'<image x="{_num(shape.x)}" y="{_num(shape.y)}" width="{_num(shape.w)}" '
                f'height="{_num(shape.h)}" preserveAspectRatio="none" '
                f'xlink:href="data:image/png;base64,{data.decode("ascii")}"/>')
    if isinstance(shape, Text):
        return (f'<text x="{_num(shape.x)}" y="{_num(shape.y + BASELINE * shape.size)}" '
                f'font-size="{_num(shape.size)}" font-family="{FONT_FAMILY}" '
                f'{_svg_paint("fill", shape.fill)}>{html.escape(shape.text)}</text>')
    raise TypeError(f"no SVG for {type(shape).__name__}")


def to_svg(root: Group, width: float, height: float) -> str:
    """The tree as SVG 1.1. Every group is a ``<g>`` with an Illustrator-style id and an
    ``inkscape:label``; the top-level groups are marked as Inkscape layers too."""
    unique = _Unique()

    def emit(group: Group, depth: int) -> list[str]:
        label = html.escape(group.name, quote=True)
        layer = ' inkscape:groupmode="layer"' if depth == 1 else ""
        lines = [f'<g id="{unique(illustrator_id(group.name))}"{layer} '
                 f'inkscape:label="{label}">']
        for c in group.children:
            lines.extend(emit(c, depth + 1) if isinstance(c, Group) else [_svg_shape(c)])
        lines.append("</g>")
        return lines

    body = [line for c in root.children
            for line in (emit(c, 1) if isinstance(c, Group) else [_svg_shape(c)])]
    return "\n".join([
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
        f'width="{_num(width)}" height="{_num(height)}" '
        f'viewBox="0 0 {_num(width)} {_num(height)}">',
        *body, "</svg>", ""])


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #

class _Pdf:
    """Just enough PDF: a page, flate streams, ExtGStates for opacity, Helvetica, image
    XObjects with soft masks, and nested optional content for the group tree."""

    def __init__(self) -> None:
        self.objects: list[bytes] = []
        self.gstates: dict[tuple, str] = {}
        self.images: list[tuple[str, int]] = []
        self.ocgs: list[tuple[str, int]] = []

    def add(self, body: Union[bytes, str]) -> int:
        self.objects.append(body.encode("latin-1") if isinstance(body, str) else body)
        return len(self.objects)

    def reserve(self) -> int:
        return self.add(b"")

    def set(self, number: int, body: Union[bytes, str]) -> None:
        self.objects[number - 1] = body.encode("latin-1") if isinstance(body, str) else body

    def stream(self, data: bytes, extra: str = "") -> int:
        packed = zlib.compress(data, 6)
        return self.add(f"<< /Length {len(packed)} /Filter /FlateDecode {extra}>>\nstream\n"
                        .encode("latin-1") + packed + b"\nendstream")

    def gstate(self, fill: float, stroke: float) -> str:
        key = (round(fill, 4), round(stroke, 4))
        if key not in self.gstates:
            self.gstates[key] = f"GS{len(self.gstates)}"
        return self.gstates[key]

    def image(self, rgba: np.ndarray) -> str:
        h, w = rgba.shape[:2]
        mask = self.stream(np.ascontiguousarray(rgba[..., 3]).tobytes(),
                           f"/Type /XObject /Subtype /Image /Width {w} /Height {h} "
                           f"/ColorSpace /DeviceGray /BitsPerComponent 8 ")
        ref = self.stream(np.ascontiguousarray(rgba[..., :3]).tobytes(),
                          f"/Type /XObject /Subtype /Image /Width {w} /Height {h} "
                          f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /SMask {mask} 0 R ")
        name = f"Im{len(self.images)}"
        self.images.append((name, ref))
        return name

    def ocg(self, name: str) -> tuple[str, int]:
        ref = self.add(f"<< /Type /OCG /Name {_pdf_string(name)} >>")
        tag = f"oc{len(self.ocgs)}"
        self.ocgs.append((tag, ref))
        return tag, ref


def _pdf_string(text: str) -> str:
    """A literal string in PDFDocEncoding/WinAnsi; anything outside Latin-1 becomes '?'."""
    raw = str(text).encode("latin-1", "replace").decode("latin-1")
    return "(" + raw.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") + ")"


def _f(v: float) -> str:
    return f"{float(v):.3f}".rstrip("0").rstrip(".") or "0"


def _rgb(color: RGBA) -> str:
    return " ".join(_f(min(max(float(c), 0.0), 1.0)) for c in tuple(color)[:3])


def _pdf_shape(pdf: _Pdf, shape: Shape, out: list[str]) -> None:
    def opacity(fill: Optional[RGBA], stroke: Optional[RGBA]) -> None:
        fa = _alpha(fill) if fill is not None else 1.0
        sa = _alpha(stroke) if stroke is not None else 1.0
        if fa < 0.999 or sa < 0.999:
            out.append(f"/{pdf.gstate(fa, sa)} gs")

    if isinstance(shape, Rect):
        out.append("q")
        opacity(shape.fill, None)
        out.append(f"{_rgb(shape.fill)} rg {_f(shape.x)} {_f(shape.y)} {_f(shape.w)} "
                   f"{_f(shape.h)} re f Q")
    elif isinstance(shape, Segments):
        out.append("q")
        opacity(None, shape.stroke)
        cap = {"butt": 0, "round": 1, "square": 2}.get(shape.cap, 1)
        out.append(f"{_rgb(shape.stroke)} RG {_f(shape.width)} w {cap} J 1 j")
        out.extend(f"{_f(a[0])} {_f(a[1])} m {_f(b[0])} {_f(b[1])} l"
                   for a, b in shape.segments)
        out.append("S Q")
    elif isinstance(shape, Markers):
        stroked = shape.stroke is not None and shape.stroke_width > 0
        out.append("q")
        opacity(shape.fill, shape.stroke if stroked else None)
        out.append(f"{_rgb(shape.fill)} rg")
        if stroked:
            out.append(f"{_rgb(shape.stroke)} RG {_f(shape.stroke_width)} w")
        paint = "B" if stroked else "f"
        r = shape.size / 2.0
        poly = _circle_polygons(shape.shape, r)
        for x, y in shape.xy:
            if poly is None:
                k = KAPPA * r
                out.append(
                    f"{_f(x + r)} {_f(y)} m "
                    f"{_f(x + r)} {_f(y + k)} {_f(x + k)} {_f(y + r)} {_f(x)} {_f(y + r)} c "
                    f"{_f(x - k)} {_f(y + r)} {_f(x - r)} {_f(y + k)} {_f(x - r)} {_f(y)} c "
                    f"{_f(x - r)} {_f(y - k)} {_f(x - k)} {_f(y - r)} {_f(x)} {_f(y - r)} c "
                    f"{_f(x + k)} {_f(y - r)} {_f(x + r)} {_f(y - k)} {_f(x + r)} {_f(y)} c "
                    f"h {paint}")
            else:
                (x0, y0), *rest = [(x + dx, y + dy) for dx, dy in poly]
                out.append(f"{_f(x0)} {_f(y0)} m " + " ".join(
                    f"{_f(px)} {_f(py)} l" for px, py in rest) + f" h {paint}")
        out.append("Q")
    elif isinstance(shape, Image):
        name = pdf.image(shape.rgba)
        # The page is flipped to y-down, so an image (drawn y-up in its unit square)
        # needs a second flip: height negative, origin at its bottom edge.
        out.append(f"q {_f(shape.w)} 0 0 {_f(-shape.h)} {_f(shape.x)} "
                   f"{_f(shape.y + shape.h)} cm /{name} Do Q")
    elif isinstance(shape, Text):
        out.append("q")
        opacity(shape.fill, None)
        # Text also un-flips: [s 0 0 -s x y] puts glyphs upright on a y-down page.
        out.append(f"BT /F1 1 Tf {_f(shape.size)} 0 0 {_f(-shape.size)} {_f(shape.x)} "
                   f"{_f(shape.y + BASELINE * shape.size)} Tm {_rgb(shape.fill)} rg "
                   f"{_pdf_string(shape.text)} Tj ET Q")
    else:
        raise TypeError(f"no PDF for {type(shape).__name__}")


def to_pdf(root: Group, width: float, height: float) -> bytes:
    """The tree as a one-page PDF, each group a nested optional-content layer."""
    pdf = _Pdf()
    catalog, pages, page = pdf.reserve(), pdf.reserve(), pdf.reserve()
    content: list[str] = [f"1 0 0 -1 0 {_f(height)} cm"]       # y down, as the tree is

    def emit(group: Group) -> list:
        tag, ref = pdf.ocg(group.name)
        content.append(f"/OC /{tag} BDC")
        order: list = []
        for c in group.children:
            if isinstance(c, Group):
                order.extend(emit(c))
            else:
                _pdf_shape(pdf, c, content)
        content.append("EMC")
        return [ref, order] if order else [ref]

    order: list = []
    for c in root.children:
        if isinstance(c, Group):
            order.extend(emit(c))
        else:
            _pdf_shape(pdf, c, content)

    stream = pdf.stream("\n".join(content).encode("latin-1", "replace"))
    font = pdf.add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
                   "/Encoding /WinAnsiEncoding >>")
    gstates = " ".join(f"/{name} << /Type /ExtGState /ca {fa} /CA {sa} >>"
                       for (fa, sa), name in pdf.gstates.items())
    xobjects = " ".join(f"/{name} {ref} 0 R" for name, ref in pdf.images)
    props = " ".join(f"/{tag} {ref} 0 R" for tag, ref in pdf.ocgs)
    pdf.set(page, f"<< /Type /Page /Parent {pages} 0 R /MediaBox [0 0 {_f(width)} "
                  f"{_f(height)}] /Contents {stream} 0 R /Resources << /Font << /F1 {font} "
                  f"0 R >> /ExtGState << {gstates} >> /XObject << {xobjects} >> "
                  f"/Properties << {props} >> >> >>")
    pdf.set(pages, f"<< /Type /Pages /Kids [{page} 0 R] /Count 1 >>")

    def order_text(items: list) -> str:
        return "[" + " ".join(order_text(i) if isinstance(i, list) else f"{i} 0 R"
                              for i in items) + "]"

    all_ocgs = " ".join(f"{ref} 0 R" for _, ref in pdf.ocgs)
    pdf.set(catalog, f"<< /Type /Catalog /Pages {pages} 0 R /OCProperties << /OCGs "
                     f"[{all_ocgs}] /D << /Order {order_text(order)} /ON [{all_ocgs}] >> "
                     f">> >>")

    out = bytearray(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for number, body in enumerate(pdf.objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode("latin-1") + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(offsets) + 1}\n0000000000 65535 f \n".encode("latin-1")
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode("latin-1")
    out += (f"trailer\n<< /Size {len(offsets) + 1} /Root {catalog} 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n").encode("latin-1")
    return bytes(out)


def write(root: Group, width: float, height: float, path: str) -> str:
    """Write ``.svg`` or ``.pdf`` by extension."""
    lower = str(path).lower()
    if lower.endswith(".pdf"):
        with open(path, "wb") as fh:
            fh.write(to_pdf(root, width, height))
    elif lower.endswith(".svg"):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(to_svg(root, width, height))
    else:
        raise ValueError(f"a vector figure is .svg or .pdf, not {path!r}")
    return path
