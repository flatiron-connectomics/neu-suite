"""A skeleton as vertices and edges, in nm, zyx: :class:`Skeleton`.

An edge list, never decomposed into branch paths, so any topology works: cycles,
self-loops, thousands of disconnected components. It draws directly as line segments, in
one buffer per body::

    positions = to_xyz(skeleton.segments())
    pygfx.Line(pygfx.Geometry(positions=positions), pygfx.LineSegmentMaterial(...))

Precomputed skeletons (vertices, edges, radii) and DVID's SWC tables (nodes and parents)
both map onto it directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import numpy as np
from .grid import BBox

from .frame import Frame

#: Called with an ``(N, 3)`` zyx nm array, returns an ``(N,)`` bool array.
InsideFn = Callable[[np.ndarray], Any]


@dataclass
class Skeleton:
    """Vertices in nm (zyx), the edges between them, and optional per-vertex radii."""

    vertices_zyx_nm: np.ndarray            # (N, 3) float32
    edges: np.ndarray                      # (M, 2) int32
    radii_nm: Optional[np.ndarray] = None  # (N,) float32
    name: Optional[str] = None

    def __post_init__(self) -> None:
        verts = np.ascontiguousarray(self.vertices_zyx_nm, dtype=np.float32)
        verts = verts.reshape(0, 3) if verts.size == 0 else verts
        if verts.ndim != 2 or verts.shape[1] != 3:
            raise ValueError(f"vertices must be (N, 3) zyx, got shape {verts.shape}")
        self.vertices_zyx_nm = verts

        edges = np.ascontiguousarray(self.edges, dtype=np.int64).reshape(-1, 2)
        if len(edges) and (edges.min() < 0 or edges.max() >= len(verts)):
            raise ValueError(
                f"edge indices out of range for {len(verts)} vertices "
                f"(saw {edges.min()}..{edges.max()})")
        self.edges = edges

        if self.radii_nm is not None:
            radii = np.ascontiguousarray(self.radii_nm, dtype=np.float32).ravel()
            if len(radii) != len(verts):
                raise ValueError(f"{len(radii)} radii for {len(verts)} vertices")
            self.radii_nm = radii

    # -- construction ----------------------------------------------------------

    @classmethod
    def from_precomputed(cls, vertices_xyz_nm: Any, edges: Any,
                         radii_nm: Any = None, name: Optional[str] = None) -> "Skeleton":
        """From xyz vertices, as the precomputed format stores them and
        ``neu_vol.readback.read_body_skeleton`` returns them, flipped to zyx."""
        verts = np.asarray(vertices_xyz_nm, dtype=np.float32)
        if verts.size == 0:
            verts = verts.reshape(0, 3)
        if verts.ndim != 2 or verts.shape[1] != 3:
            raise ValueError(f"vertices must be (N, 3), got shape {verts.shape}")
        return cls(np.ascontiguousarray(verts[:, ::-1]), edges, radii_nm, name)

    @classmethod
    def from_swc(cls, table: Any, *, frame: Frame, node_col: str = "node",
                 parent_col: str = "parent",
                 coord_cols: Sequence[str] = ("x", "y", "z"),
                 radius_col: Optional[str] = "radius",
                 name: Optional[str] = None) -> "Skeleton":
        """From a node/parent table, DVID's skeleton form.

        ``coord_cols`` are read in the order given and then reversed, so the SWC default
        ``("x", "y", "z")`` becomes zyx. Coordinates are voxel indices, converted to nm
        with ``frame``. A parent id absent from the table (SWC's ``-1`` root, or a
        dangling reference) adds no edge; the result need not be a tree.
        """
        nodes = np.asarray(table[node_col], dtype=np.int64)
        parents = np.asarray(table[parent_col], dtype=np.int64)
        coords = np.column_stack([np.asarray(table[c], dtype=np.float64)
                                  for c in coord_cols])
        if not (len(nodes) == len(parents) == len(coords)):
            raise ValueError(f"ragged table: {len(nodes)} nodes, {len(parents)} "
                             f"parents, {len(coords)} coordinates")

        # A duplicate id would make the parent lookup ambiguous and wire wrong edges.
        unique, counts = np.unique(nodes, return_counts=True)
        if len(unique) != len(nodes):
            raise ValueError(f"duplicate node ids: {unique[counts > 1][:5].tolist()}")

        index_of = {int(n): i for i, n in enumerate(nodes)}
        edges = [(i, index_of[int(p)]) for i, p in enumerate(parents)
                 if int(p) in index_of]

        vertices_nm = frame.to_nm(coords[:, ::-1])
        radii = None
        if radius_col is not None and _has_column(table, radius_col):
            # A radius is one length, so an anisotropic voxel can only be summarised:
            # use the mean voxel size, which is exact for isotropic data.
            radii = (np.asarray(table[radius_col], dtype=np.float64)
                     * float(np.mean(frame.voxel_size_nm)))
        return cls(vertices_nm, np.asarray(edges, dtype=np.int64).reshape(-1, 2),
                   radii, name)

    # -- geometry --------------------------------------------------------------

    @property
    def bbox(self) -> BBox:
        if not len(self.vertices_zyx_nm):
            return BBox.empty(3)
        return BBox.from_points(self.vertices_zyx_nm)

    def segments(self) -> np.ndarray:
        """``(2M, 3)`` zyx nm, the two endpoints of each edge in turn: pass through
        :func:`~neu_lib.frame.to_xyz` to draw with ``pygfx.LineSegmentMaterial``."""
        return np.ascontiguousarray(self.vertices_zyx_nm[self.edges.reshape(-1)])

    def translate(self, offset_zyx_nm: Sequence[float]) -> "Skeleton":
        offset = np.asarray(offset_zyx_nm, dtype=np.float64)
        if offset.shape != (3,):
            raise ValueError(f"offset must be zyx, got shape {offset.shape}")
        return Skeleton(self.vertices_zyx_nm + offset, self.edges,
                        None if self.radii_nm is None else self.radii_nm.copy(),
                        self.name)

    # -- cropping --------------------------------------------------------------

    def crop(self, inside: InsideFn, *, keep_inside: bool = True,
             tolerance_nm: float = 8.0) -> "Skeleton":
        """Keep the part of the skeleton on one side of a region boundary.

        ``inside`` takes an ``(N, 3)`` zyx nm array and returns an ``(N,)`` bool array (see
        :mod:`neu_lib.predicates`). An edge with one endpoint kept is bisected to within
        ``tolerance_nm`` and ends at a new vertex on the boundary, so branches stop at the
        region surface. Every kept vertex is retained, including ones left with no edges.

        Limitation: an edge whose endpoints are on the same side is assumed not to cross,
        so a region thinner than the edge spacing can be missed.
        """
        verts = self.vertices_zyx_nm
        keep = np.asarray(inside(verts), dtype=bool).ravel()
        if keep.shape != (len(verts),):
            raise ValueError(
                f"inside() returned {keep.shape} for {len(verts)} vertices; it must "
                f"return one bool per point")
        if not keep_inside:
            keep = ~keep

        kept_ix = np.flatnonzero(keep)
        remap = np.full(len(verts), -1, dtype=np.int64)
        remap[kept_ix] = np.arange(len(kept_ix))

        edges = self.edges
        a_in, b_in = keep[edges[:, 0]], keep[edges[:, 1]]
        whole = edges[a_in & b_in]
        crossing = edges[a_in ^ b_in]

        vert_parts = [verts[kept_ix]]
        radii_parts = ([self.radii_nm[kept_ix]] if self.radii_nm is not None else None)
        edge_parts = [remap[whole]]

        if len(crossing):
            # Orient every crossing edge so column 0 is the endpoint being kept.
            flip = ~keep[crossing[:, 0]]
            crossing = np.where(flip[:, None], crossing[:, ::-1], crossing)
            p_in = verts[crossing[:, 0]].astype(np.float64)
            p_out = verts[crossing[:, 1]].astype(np.float64)
            t = _bisect(p_in, p_out, inside, keep_inside, tolerance_nm)
            vert_parts.append((p_in + t[:, None] * (p_out - p_in)).astype(np.float32))
            if radii_parts is not None:
                r_in = self.radii_nm[crossing[:, 0]].astype(np.float64)
                r_out = self.radii_nm[crossing[:, 1]].astype(np.float64)
                radii_parts.append((r_in + t * (r_out - r_in)).astype(np.float32))
            base = len(kept_ix)
            edge_parts.append(np.stack(
                [remap[crossing[:, 0]], base + np.arange(len(crossing))], axis=1))

        return Skeleton(
            np.concatenate(vert_parts) if vert_parts else np.zeros((0, 3), np.float32),
            np.concatenate(edge_parts) if edge_parts else np.zeros((0, 2), np.int64),
            None if radii_parts is None else np.concatenate(radii_parts),
            self.name)

    def exclude(self, inside: InsideFn, **kwargs) -> "Skeleton":
        """The complement of :meth:`crop` — drop what is inside the region."""
        return self.crop(inside, keep_inside=False, **kwargs)

    def split_at(self, inside: InsideFn, *, tolerance_nm: float = 8.0
                 ) -> tuple["Skeleton", np.ndarray, np.ndarray]:
        """Insert a vertex where each edge crosses a region boundary, keeping both sides.

        Returns ``(skeleton, inside, boundary)``: the skeleton with each crossing edge split
        in two, plus one bool per vertex for "in the region" and "is an inserted boundary
        vertex". Original vertices keep their indices; boundary vertices are appended, on
        the inside, within ``tolerance_nm`` of the surface (the bisection :meth:`crop`
        uses).

        The edge order is guaranteed, so per-edge data can follow the split: uncrossed
        edges in their original order, then one ``(inside endpoint, boundary)`` edge per
        crossing edge, then one ``(boundary, outside endpoint)`` edge per crossing edge.
        Same limitation as :meth:`crop`.
        """
        verts = self.vertices_zyx_nm
        keep = np.asarray(inside(verts), dtype=bool).ravel()
        if keep.shape != (len(verts),):
            raise ValueError(
                f"inside() returned {keep.shape} for {len(verts)} vertices; it must "
                f"return one bool per point")
        edges = self.edges
        is_crossing = keep[edges[:, 0]] ^ keep[edges[:, 1]]
        crossing = edges[is_crossing]
        if not len(crossing):
            return (Skeleton(verts.copy(), edges.copy(),
                             None if self.radii_nm is None else self.radii_nm.copy(),
                             self.name),
                    keep, np.zeros(len(verts), dtype=bool))

        # Orient every crossing edge so column 0 is the inside endpoint.
        flip = ~keep[crossing[:, 0]]
        crossing = np.where(flip[:, None], crossing[:, ::-1], crossing)
        p_in = verts[crossing[:, 0]].astype(np.float64)
        p_out = verts[crossing[:, 1]].astype(np.float64)
        t = _bisect(p_in, p_out, inside, True, tolerance_nm)
        new_ix = len(verts) + np.arange(len(crossing))

        radii = None
        if self.radii_nm is not None:
            r_in = self.radii_nm[crossing[:, 0]].astype(np.float64)
            r_out = self.radii_nm[crossing[:, 1]].astype(np.float64)
            radii = np.concatenate([self.radii_nm, r_in + t * (r_out - r_in)])
        out = Skeleton(
            np.concatenate([verts, (p_in + t[:, None] * (p_out - p_in))]),
            np.concatenate([edges[~is_crossing],
                            np.stack([crossing[:, 0], new_ix], axis=1),
                            np.stack([new_ix, crossing[:, 1]], axis=1)]),
            radii, self.name)
        n_new = len(crossing)
        return (out, np.concatenate([keep, np.ones(n_new, dtype=bool)]),
                np.concatenate([np.zeros(len(verts), dtype=bool),
                                np.ones(n_new, dtype=bool)]))

    def __repr__(self) -> str:
        return (f"Skeleton(name={self.name!r}, vertices={len(self.vertices_zyx_nm)}, "
                f"edges={len(self.edges)}, radii={self.radii_nm is not None})")


def _bisect(p_in: np.ndarray, p_out: np.ndarray, inside: InsideFn,
            keep_inside: bool, tolerance_nm: float) -> np.ndarray:
    """Fraction along each ``p_in -> p_out`` of the last point still on the kept side,
    bisecting all segments together with one ``inside`` call per round."""
    if tolerance_nm <= 0:
        raise ValueError(f"tolerance_nm must be positive, got {tolerance_nm}")
    if not len(p_in):
        return np.zeros(0)
    longest = float(np.linalg.norm(p_out - p_in, axis=1).max())
    rounds = int(np.clip(np.ceil(np.log2(max(longest, tolerance_nm) / tolerance_nm)),
                         1, 32))
    lo = np.zeros(len(p_in))
    hi = np.ones(len(p_in))
    for _ in range(rounds):
        mid = 0.5 * (lo + hi)
        probe = np.asarray(inside(p_in + mid[:, None] * (p_out - p_in)), dtype=bool)
        on_kept_side = probe if keep_inside else ~probe
        lo = np.where(on_kept_side, mid, lo)
        hi = np.where(on_kept_side, hi, mid)
    return lo


def _has_column(table: Any, column: str) -> bool:
    try:
        return column in table
    except TypeError:
        return False
