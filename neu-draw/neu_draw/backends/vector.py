"""Figure export: meshes as one transparent image layer each, everything else as vectors.

pygfx rasterises on the GPU and has no vector output, so this does the projection itself,
from the same camera, and builds a :class:`neu_draw.vectorfile.Group` tree that
:mod:`neu_draw.vectorfile` writes as SVG or PDF. What is vector is what you would restyle
in a figure: synapse points, skeleton lines and the legend (as real text). **Meshes stay
raster**, because as vectors they are tens of thousands of depth-sorted triangles that
open slowly, stack their own translucency and show seams — so each mesh is rendered
**alone** into an RGBA image with a transparent background and placed as its own group.

## The group tree

``background``, ``meshes``, ``skeletons``, ``points``, ``legend``. Under each of the three
drawable groups, every drawable is a group of its own, optionally nested deeper by a
``groups`` function — ``drawable -> path`` — so a figure can say *points by cell, then by
input type*::

    view.save("fig.svg", groups=lambda d: (d.name.split(" : ")[0],))

The legend splits into its ``panel``, the declared ``group rows``, and the ``rows``, which
are nested by kind and by the same function applied to each row's first member, so a
cell's legend rows sit together the way its drawables do. Every row is a group of its
plate, glyph and text.

## What is given up, and why that is the right trade

**Occlusion between groups.** Each mesh is rendered on its own, so it no longer hides
another mesh, a skeleton or a point behind it. Meshes are stacked back to front by their
centres' depth, and every vector sits above every mesh. With the faint meshes this is
meant for, that reads correctly; an opaque mesh in front of synapses will show them
through it. What is exact is everything *within* a group, including a translucent mesh
over itself.

## Two details that are easy to get wrong

- **The snapshot is PREMULTIPLIED in linear light, and editors composite in sRGB.** Red at
  alpha 0.5 comes back as (187, 0, 0, 127) — 187 being sRGB(0.5), not 255 — and even the
  physically correct un-premultiplied colour composites far too dark in an editor,
  because the GPU blended in linear light and the editor blends in sRGB (a 0.2-alpha mesh
  came out barely visible). :func:`layer_pixels` instead solves for the colour that
  reproduces the canvas's pixel over the figure's own background; see there.
- **Vector colours are read off the built MATERIALS, not the drawables**, so the export
  shows what the canvas shows: a highlighted body is exported highlighted, and alpha is
  the drawable's own times its colour's, which ``display_color`` already folded in.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import numpy as np
import pygfx

from ..scene import LinesDrawable, MeshDrawable, PointsDrawable
from ..vectorfile import Group, Image, Markers, Rect, Segments, Text, write
from .legend import GAP, GLYPH_WIDTH

#: The canvas colour pygfx clears to when a scene sets no background.
DEFAULT_BACKGROUND = (0.0, 0.0, 0.0, 1.0)
#: The top-level group each drawable kind goes in.
KIND_GROUPS = {MeshDrawable: "meshes", LinesDrawable: "skeletons", PointsDrawable: "points"}


# --------------------------------------------------------------------------- #
# pure helpers: projection and compositing
# --------------------------------------------------------------------------- #

def project(xyz: np.ndarray, matrix: np.ndarray, rect: Sequence[float]
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World xyz -> ``(pixels (N, 2), depth (N,), in_front (N,) bool)`` inside ``rect``.

    ``matrix`` is the camera's ``projection @ view`` (pygfx's ``camera_matrix``, column
    vectors). Pixel y runs **down**, as SVG's does. A point behind the camera has a clip
    ``w <= 0`` and a meaningless divide, so it is flagged rather than drawn mirrored.
    """
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    clip = np.c_[xyz, np.ones(len(xyz))] @ np.asarray(matrix, dtype=np.float64).T
    w = clip[:, 3]
    in_front = w > 1e-12
    safe = np.where(in_front, w, 1.0)
    ndc = clip[:, :3] / safe[:, None]
    x0, y0, width, height = (float(v) for v in rect)
    px = np.c_[x0 + (ndc[:, 0] + 1.0) / 2.0 * width,
               y0 + (1.0 - ndc[:, 1]) / 2.0 * height]
    return px, ndc[:, 2], in_front


def _transform(obj: pygfx.WorldObject, xyz: np.ndarray) -> np.ndarray:
    """An object's local positions in world space — its offset is its transform."""
    matrix = np.asarray(obj.world.matrix, dtype=np.float64)
    return (np.c_[xyz, np.ones(len(xyz))] @ matrix.T)[:, :3]


def _srgb_decode(v: np.ndarray) -> np.ndarray:
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def _srgb_encode(linear: np.ndarray) -> np.ndarray:
    return np.where(linear <= 0.0031308, linear * 12.92,
                    1.055 * np.clip(linear, 0.0, None) ** (1 / 2.4) - 0.055)


def layer_pixels(rgba: np.ndarray, background: Sequence[float]) -> np.ndarray:
    """A mesh render as a straight-alpha PNG that composites like the canvas does.

    Two corrections, both measured (see the module docstring):

    - pygfx hands back colour **premultiplied in linear light**, sRGB-encoded; and
    - an SVG viewer composites **in sRGB**, where the GPU composited in linear light.

    So even the physically right straight colour, ``srgb(linear(v) / alpha)``, composites
    too dark: at alpha 0.2 a mid-grey came out at 0.16 against the canvas's 0.38. Instead
    this solves for the colour ``S`` that, blended the way SVG blends, reproduces the pixel
    the canvas would show over ``background`` (sRGB, 0-1)::

        S * a + bg * (1 - a)  ==  srgb(linear(v) + linear(bg) * (1 - a))

    Exact for one layer over the figure's own background, whatever that background is —
    with the alpha raised where the colour alone cannot get there (see the body). Where
    layers stack, or the background is changed afterwards, it is close rather than exact:
    no single straight colour is right over every backdrop.
    """
    rgba = np.asarray(rgba)
    v = rgba[..., :3].astype(np.float64) / 255.0
    alpha = rgba[..., 3:4].astype(np.float64) / 255.0
    bg = np.asarray(background, dtype=np.float64)[:3]
    target = _srgb_encode(_srgb_decode(v) + _srgb_decode(bg) * (1.0 - alpha))

    # The solved colour can leave [0, 1]: a light 0.2-alpha mesh over black shows as
    # srgb(0.2 * linear) ~ 0.45, which an sRGB blend at alpha 0.2 can reach only with a
    # colour of 2.2. So the alpha is RAISED, per pixel, to the least that keeps every
    # channel in range — still exact over this background. A channel above the background
    # needs alpha >= (T - bg) / (1 - bg); one below it needs alpha >= (bg - T) / bg.
    with np.errstate(divide="ignore", invalid="ignore"):
        above = np.where(target > bg, (target - bg) / np.where(bg < 1, 1 - bg, 1), 0.0)
        below = np.where(target < bg, (bg - target) / np.where(bg > 0, bg, 1), 0.0)
    needed = np.max(np.maximum(above, below), axis=-1, keepdims=True)
    alpha = np.where(alpha > 0, np.clip(np.maximum(alpha, needed), 0.0, 1.0), 0.0)

    colour = np.divide(target - bg * (1.0 - alpha), alpha,
                       out=np.zeros_like(v), where=alpha > 0)
    out = np.empty(rgba.shape, dtype=np.uint8)
    out[..., :3] = np.round(np.clip(colour, 0.0, 1.0) * 255.0)
    out[..., 3] = np.round(alpha[..., 0] * 255.0)
    return out


def _rgba(color: Any) -> tuple[float, float, float, float]:
    values = tuple(float(c) for c in tuple(color))
    return values if len(values) == 4 else (*values[:3], 1.0)


def _inside(xy: np.ndarray, rect: Sequence[float], margin: float = 0.0) -> np.ndarray:
    """Which points fall in ``rect`` (grown by ``margin``). Used in place of a clip path,
    which Illustrator turns into a clip group wrapped around everything."""
    x0, y0, w, h = rect
    return ((xy[:, 0] >= x0 - margin) & (xy[:, 0] <= x0 + w + margin)
            & (xy[:, 1] >= y0 - margin) & (xy[:, 1] <= y0 + h + margin))


# --------------------------------------------------------------------------- #
# one drawable at a time
# --------------------------------------------------------------------------- #

def _points(obj: pygfx.Points, matrix, rect) -> Optional[Markers]:
    positions = np.asarray(obj.geometry.positions.data, dtype=np.float64)
    px, depth, front = project(_transform(obj, positions), matrix, rect)
    material = obj.material
    keep = np.flatnonzero(front & _inside(px, rect, float(material.size) / 2))
    if not len(keep):
        return None
    keep = keep[np.argsort(-depth[keep], kind="stable")]         # far first
    edge = float(getattr(material, "edge_width", 0.0) or 0.0)
    return Markers(str(material.marker), px[keep], float(material.size),
                   _rgba(material.color),
                   _rgba(material.edge_color) if edge > 0 else None, edge)


def _lines(obj: pygfx.Line, matrix, rect) -> Optional[Segments]:
    positions = np.asarray(obj.geometry.positions.data, dtype=np.float64)
    px, _, front = project(_transform(obj, positions), matrix, rect)
    n = len(px) // 2 * 2
    a, b = px[0:n:2], px[1:n:2]
    inside = _inside(px, rect)
    # A segment is kept if either end is on screen, so a line leaving the frame still
    # reaches its edge; one wholly off screen, or behind the camera, is dropped.
    ok = front[0:n:2] & front[1:n:2] & (inside[0:n:2] | inside[1:n:2])
    if not ok.any():
        return None
    material = obj.material
    return Segments(np.stack([a[ok], b[ok]], axis=1), _rgba(material.color),
                    float(material.thickness))


def _render_alone(view, index: int, camera, size: tuple[int, int],
                  pixel_ratio: float) -> np.ndarray:
    """RGBA of ``view.group.children[index]`` alone, lights on, no background.

    Visibility is flipped on the view's own scene and put back in a ``finally``, rather
    than the object being moved to a scene of its own: a pygfx object has one parent, and
    the backend pairs drawables with objects by position in ``view.group``.
    """
    from rendercanvas.offscreen import RenderCanvas as Offscreen

    saved = [(obj, obj.visible) for obj in view.group.children]
    others = [(obj, obj.visible) for obj in view.scene.children
              if obj is not view.group and not isinstance(obj, pygfx.Light)]
    try:
        for i, (obj, _) in enumerate(saved):
            obj.visible = i == index
        for obj, _ in others:                       # background, axes
            obj.visible = False
        canvas = Offscreen(size=size)
        renderer = pygfx.renderers.WgpuRenderer(canvas, pixel_ratio=pixel_ratio)
        renderer.render(view.scene, camera)
        return np.asarray(renderer.snapshot())
    finally:
        for obj, visible in saved + others:
            obj.visible = visible


def _mesh(view, index: int, camera, rect, pixel_ratio: float,
          background: Sequence[float]) -> Optional[Image]:
    """The mesh as a cropped, straight-alpha image, placed where it sits in the frame.

    Cropped to its own pixels, so a small body in a large frame costs a small image —
    and an editor's bounding box for it is the body rather than the whole canvas.
    """
    x0, y0, width, height = rect
    rgba = _render_alone(view, index, camera, (int(width), int(height)), pixel_ratio)
    alpha = rgba[..., 3]
    rows, cols = np.flatnonzero(alpha.any(axis=1)), np.flatnonzero(alpha.any(axis=0))
    if not len(rows):
        return None                                 # entirely off screen
    top, bottom, left, right = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    scale = rgba.shape[1] / float(width)            # image px per logical px
    return Image(layer_pixels(rgba[top:bottom, left:right], background),
                 x0 + left / scale, y0 + top / scale,
                 (right - left) / scale, (bottom - top) / scale)


# --------------------------------------------------------------------------- #
# the legend, redrawn from the rows the canvas shows
# --------------------------------------------------------------------------- #

def _legend(legend, size: Sequence[float], strip: Sequence[float],
            path_of: Callable[[Any], Sequence[str]]) -> Optional[Group]:
    """The strip as rects, glyphs and text, in the canvas's own layout.

    Positions come from the entries' laid-out transforms and the same fit-to-strip scale
    ``LegendOverlay._aim`` uses, and colours from their materials — so hidden rows are
    dimmed and lit rows tinted exactly as on screen.
    """
    sx, sy, sw, sh = (float(v) for v in strip)
    if sw <= 0 or sh <= 0 or not legend.entries:
        return None
    columns, _ = legend.plan(size)
    if columns != legend.columns:
        legend.relayout(columns)
    scale = min(1.0, sw / legend.content_width, sh / legend.content_height)
    visible_width = sw / scale

    def at(x: float, y: float) -> tuple[float, float]:
        # The ortho camera centres the content horizontally and hangs it from the top.
        return (sx + (x - legend.content_width / 2.0 + visible_width / 2.0) * scale,
                sy + (-y) * scale)

    root = Group("legend", [Group("panel", [Rect(sx, sy, sw, sh,
                                                 _rgba(legend.backdrop.material.color))])])
    size_px = legend.spec.font_size
    height = legend.row_height
    for entry in legend.entries:
        ox, oy = entry.group.local.position[:2]
        left, top = at(ox, oy)
        row = Group(entry.text, [Rect(left, top, legend.column_width * scale,
                                      height * scale, _rgba(entry.plate.material.color))])
        gx, gy = at(ox + size_px * GLYPH_WIDTH / 2.0, oy - height / 2.0)
        glyph = entry.glyph
        colour = _rgba(glyph.material.color)
        if isinstance(glyph, pygfx.Line):
            half = size_px * GLYPH_WIDTH / 2.0 * scale
            row.children.append(Segments(np.array([[[gx - half, gy], [gx + half, gy]]]),
                                         colour, float(glyph.material.thickness) * scale,
                                         cap="butt"))
        elif isinstance(glyph, pygfx.Points):
            row.children.append(Markers(str(glyph.material.marker), np.array([[gx, gy]]),
                                        float(glyph.material.size) * scale, colour))
        else:
            row.children.append(Markers("square", np.array([[gx, gy]]),
                                        size_px * 0.95 * scale, colour))
        tx, ty = at(ox + size_px * GLYPH_WIDTH + GAP, oy - height / 2.0)
        row.children.append(Text(tx, ty, entry.row_text, size_px * scale,
                                 _rgba(entry.label.material.color)))

        if entry.is_group:
            parent = root.nested(["group rows"])
        else:
            kinds = {KIND_GROUPS.get(type(d), "other") for d in entry.drawables}
            kind = kinds.pop() if len(kinds) == 1 else "mixed"
            parent = root.nested(["rows", kind, *path_of(entry.drawables[0])])
        parent.children.append(row)
    return root


# --------------------------------------------------------------------------- #
# the whole figure
# --------------------------------------------------------------------------- #

def figure(view, *, pixel_ratio: float = 2.0, legend: bool = True,
           size: Optional[tuple[int, int]] = None,
           groups: Optional[Callable[[Any], Sequence[str]]] = None
           ) -> tuple[Group, int, int]:
    """``(tree, width, height)`` for the view. See the module docstring for the tree.

    ``pixel_ratio`` sets the mesh images' resolution (2 = twice the logical size, which
    is what the canvas renders at too). ``size`` defaults to the view's logical size.
    ``groups`` nests drawables deeper, ``drawable -> tuple of group names``; it defaults
    to the scene's ``export_groups``. Only **visible** drawables are exported; the legend
    keeps every row, as on screen.
    """
    width, height = (int(v) for v in (size or view.logical_size()))
    if view.legend is not None and legend:
        main, strip = view.legend.rects_for((width, height))
    else:
        main, strip = (0.0, 0.0, float(width), float(height)), None
    path_of = groups or view.scene_data.export_groups or (lambda d: ())

    camera = pygfx.PerspectiveCamera(view.camera.fov)
    camera.set_state(view.camera.get_state())
    camera.set_view_size(main[2], main[3])
    matrix = np.asarray(camera.camera_matrix, dtype=np.float64)
    view_matrix = np.asarray(camera.view_matrix, dtype=np.float64)

    background = _rgba(view.scene_data.background or DEFAULT_BACKGROUND)
    root = Group("figure", [Group("background", [Rect(0, 0, width, height, background)])])
    for name in KIND_GROUPS.values():
        root.child(name)

    pairs = [(i, d, o) for i, (d, o) in
             enumerate(zip(view.scene_data.drawables, view.group.children)) if d.visible]

    def depth(obj) -> float:
        box = obj.get_world_bounding_box()
        centre = np.r_[(np.asarray(box[0]) + np.asarray(box[1])) / 2.0, 1.0]
        return float((view_matrix @ centre)[2])     # camera looks down -z: smaller is farther

    meshes = sorted(((i, d, o) for i, d, o in pairs if isinstance(d, MeshDrawable)),
                    key=lambda t: depth(t[2]))
    others = [(i, d, o) for i, d, o in pairs if not isinstance(d, MeshDrawable)]
    for i, drawable, obj in meshes + others:
        if isinstance(drawable, MeshDrawable):
            shape = _mesh(view, i, camera, main, pixel_ratio, background)
        elif isinstance(drawable, LinesDrawable):
            shape = _lines(obj, matrix, main)
        else:
            shape = _points(obj, matrix, main)
        if shape is None:
            continue
        name = drawable.name or f"{KIND_GROUPS[type(drawable)]} {i}"
        parent = root.child(KIND_GROUPS[type(drawable)]).nested(path_of(drawable))
        parent.children.append(Group(str(name), [shape]))

    if strip is not None:
        drawn = _legend(view.legend, (width, height), strip, path_of)
        if drawn is not None:
            root.children.append(drawn)
    return root.prune(), width, height


def save(view, path: str, **kwargs) -> str:
    """Write the figure to ``path`` — ``.svg`` or ``.pdf``, by extension."""
    tree, width, height = figure(view, **kwargs)
    return write(tree, width, height, path)
