"""What to draw, described without reference to any renderer.

A :class:`Scene` is data: a list of drawables, each holding geometry, a colour, an alpha
and a name, plus how the camera should be framed and whether a legend is wanted. It
renders nothing and imports nothing that renders.

That is the seam. The predecessor put a ``.render()`` on every geometry class and grew a
350-line ``render_bodies_rois_3d`` with ~50 keyword arguments around them, so the only
way to test what a figure would contain was to build one on a GPU. Here everything up to
the draw call is a value you can assert on, and swapping the renderer touches one module.

The pipeline it replaces splits four ways: **resolve** inputs → **fetch** through
``sources`` → **build** a Scene → **render** it. Only the last needs a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import reduce
from typing import Any, Hashable, Iterable, Mapping, Optional, Sequence, Union

import numpy as np
from em_volume_tools import BBox

from .colors import RGBA, assign_colors, to_rgba
from .geometry import Mesh, Skeleton

#: pygfx's ``MarkerShape`` vocabulary, written out so this module needs no renderer, and
#: checked here because pygfx rejects an unknown marker only at *draw* time. **There is
#: no plain "triangle"** — the four are directional. ``test_backend_pygfx`` cross-checks
#: this against the live enum.
MARKERS = (
    "circle", "square", "diamond", "ring", "cross", "plus", "tick",
    "triangle_up", "triangle_down", "triangle_left", "triangle_right",
    "asterisk6", "asterisk8", "heart", "spade", "club", "pin",
)

#: matplotlib's scatter shorthands, which is what the predecessor's call sites passed
#: (``presyn_marker='^'``, ``postsyn_marker='s'``).
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


@dataclass
class MeshDrawable:
    """A surface. ``mesh`` is an :class:`~em_viz.geometry.Mesh`, already in nm/zyx."""
    mesh: Mesh
    color: RGBA = (0.5, 0.5, 0.5, 1.0)
    alpha: float = 1.0
    name: Optional[str] = None
    visible: bool = True

    @property
    def bbox(self) -> BBox:
        return self.mesh.bbox


@dataclass
class LinesDrawable:
    """A skeleton, drawn as independent segments — one per edge.

    Carries the :class:`~em_viz.geometry.Skeleton` rather than a positions buffer, so
    the backend calls ``segments()`` itself and nothing here decides array layout.
    """
    skeleton: Skeleton
    color: RGBA = (0.5, 0.5, 0.5, 1.0)
    alpha: float = 1.0
    thickness: float = 1.5
    name: Optional[str] = None
    visible: bool = True

    @property
    def bbox(self) -> BBox:
        return self.skeleton.bbox


@dataclass
class PointsDrawable:
    """Synapses, or any other point set. zyx nm, like everything else."""
    positions_zyx_nm: np.ndarray
    color: RGBA = (0.5, 0.5, 0.5, 1.0)
    alpha: float = 1.0
    size: float = 8.0
    marker: str = "circle"
    name: Optional[str] = None
    visible: bool = True

    def __post_init__(self) -> None:
        arr = np.ascontiguousarray(self.positions_zyx_nm, dtype=np.float32)
        arr = arr.reshape(0, 3) if arr.size == 0 else arr
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"positions must be (N, 3) zyx, got shape {arr.shape}")
        self.positions_zyx_nm = arr
        self.marker = resolve_marker(self.marker)

    @property
    def bbox(self) -> BBox:
        if not len(self.positions_zyx_nm):
            return BBox.empty(3)
        return BBox.from_points(self.positions_zyx_nm)


@dataclass
class VolumeDrawable:
    """**Reserved, and not implemented.** A named slot rather than a missing concept.

    Volumetric masks were deliberately left out of this package — meshes and skeletons
    carry the same information far more cheaply, and a large binary mask needs chunking
    and compression to be workable at all. What this slot is really for is *continuous
    scalar* data: image slices and probability fields, which suit a volume renderer far
    better than a binary mask ever did. See EM-VIZ-PLAN.md.
    """
    def __post_init__(self) -> None:
        raise NotImplementedError(
            "volumetric drawables are not implemented; see EM-VIZ-PLAN.md. Meshes and "
            "skeletons cover segment morphology, and this slot is reserved for scalar "
            "fields (image slices, probability maps).")


Drawable = Union[MeshDrawable, LinesDrawable, PointsDrawable]


@dataclass
class Camera:
    """How to frame the scene. Intent, not a camera object.

    ``view_angle`` is the direction the camera looks *from*, in xyz, matching the
    renderer's convention — it is the one place in the package that is not zyx, because
    it names a viewing direction rather than data. ``target`` of ``None`` means fit to
    the scene's bounding box, which is what almost every caller wants.
    """
    view_angle: tuple[float, float, float] = (-1.0, -1.0, -1.0)
    target: Optional[BBox] = None
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    zoom: float = 1.0


@dataclass
class Legend:
    """Whether to draw a legend, and where. Backends may ignore it."""
    visible: bool = True
    location: str = "right"


@dataclass
class Scene:
    """Everything a backend needs, and nothing it does not."""

    drawables: list[Drawable] = field(default_factory=list)
    camera: Camera = field(default_factory=Camera)
    legend: Legend = field(default_factory=Legend)
    background: Optional[RGBA] = None
    axes_visible: bool = False

    # -- assembly --------------------------------------------------------------

    def add(self, drawable: Drawable, rename: bool = False) -> "Scene":
        """Append a drawable. A duplicate name raises, or is renamed if asked.

        Names key the legend and are how a caller refers to a layer afterwards, so two
        drawables sharing one is a collision rather than a duplicate — the same reason
        em-ngl renames a colliding layer instead of keeping two under one name.

        Raising is the right default for a hand-built scene, but note that the most
        ordinary thing a caller does — a body's mesh *and* its skeleton — collides,
        because both are named after the body. :func:`build_scene` handles that by
        suffixing with the representation; ``rename=True`` is the blunter fallback.
        """
        if drawable.name is not None:
            taken = {d.name for d in self.drawables if d.name is not None}
            if drawable.name in taken:
                if not rename:
                    raise ValueError(
                        f"a drawable named {drawable.name!r} is already here. Names key "
                        f"the legend, so this is a collision, not a duplicate — pass "
                        f"rename=True, or give it a name of its own.")
                stem, n = drawable.name, 2
                while f"{stem} ({n})" in taken:
                    n += 1
                drawable = replace(drawable, name=f"{stem} ({n})")
        self.drawables.append(drawable)
        return self

    def add_mesh(self, mesh: Mesh, *, rename: bool = False, **kwargs) -> "Scene":
        kwargs.setdefault("name", mesh.name)
        return self.add(MeshDrawable(mesh, **_normalised(kwargs)), rename=rename)

    def add_skeleton(self, skeleton: Skeleton, *, rename: bool = False,
                     **kwargs) -> "Scene":
        kwargs.setdefault("name", skeleton.name)
        return self.add(LinesDrawable(skeleton, **_normalised(kwargs)), rename=rename)

    def add_points(self, positions_zyx_nm: Any, *, rename: bool = False,
                   **kwargs) -> "Scene":
        return self.add(PointsDrawable(positions_zyx_nm, **_normalised(kwargs)),
                        rename=rename)

    # -- queries ---------------------------------------------------------------

    @property
    def bbox(self) -> BBox:
        """Union over visible drawables, in nm. What the camera frames."""
        boxes = [d.bbox for d in self.drawables if d.visible]
        return reduce(BBox.union, boxes, BBox.empty(3))

    @property
    def names(self) -> list[str]:
        return [d.name for d in self.drawables if d.name is not None]

    def __len__(self) -> int:
        return len(self.drawables)

    def __iter__(self):
        return iter(self.drawables)

    def get(self, name: str) -> Drawable:
        for drawable in self.drawables:
            if drawable.name == name:
                return drawable
        raise KeyError(f"no drawable named {name!r}; have {self.names}")

    def recolor(self, colors: Optional[Mapping[Hashable, Any] | Any] = None,
                palette: Optional[Sequence[Any]] = None) -> "Scene":
        """Reassign every named drawable's colour in one pass.

        Colours are chosen over the **whole set at once** rather than as each drawable
        is added, which is the only way a palette can guarantee distinct neighbours —
        assigning on `add` cannot know what is coming.
        """
        chosen = assign_colors(self.names, explicit=colors, palette=palette)
        for i, drawable in enumerate(self.drawables):
            if drawable.name in chosen:
                self.drawables[i] = replace(drawable, color=chosen[drawable.name])
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
                point_color: Any = None, **scene_kwargs) -> Scene:
    """Assemble a scene and colour it in one call.

    ``alpha`` defaults by count, reproducing the predecessor's rule: a single body opaque,
    several semi-transparent, because overlapping opaque surfaces hide each other.

    **A body's mesh and its skeleton are both named after the body**, so showing them
    together would collide. Where that happens the representation is appended to *both*
    (``"1401 mesh"``, ``"1401 skeleton"``) rather than only to the second — so a name is
    the same whichever order the scene was assembled in, and the legend says which is
    which. Names that appear once are left alone.
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

    def label(item, kind: str) -> Optional[str]:
        if item.name is None:
            return None
        return f"{item.name} {kind}" if counts.get(item.name, 0) > 1 else item.name

    for mesh in meshes:
        scene.add_mesh(mesh, alpha=alpha, name=label(mesh, "mesh"), rename=True)
    for skeleton in skeletons:
        scene.add_skeleton(skeleton, alpha=alpha, name=label(skeleton, "skeleton"),
                           rename=True)
    for name, positions in (points or {}).items():
        scene.add_points(positions, name=name, rename=True,
                         **({"color": point_color} if point_color is not None else {}))

    return scene.recolor(colors, palette=palette)


def _normalised(kwargs: dict) -> dict:
    if "color" in kwargs and kwargs["color"] is not None:
        kwargs["color"] = to_rgba(kwargs["color"])
    else:
        kwargs.pop("color", None)
    return kwargs
