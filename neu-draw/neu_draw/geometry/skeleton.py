"""A skeleton as vertices and edges, in nm, zyx.

**No polylines.** Decomposing a skeleton into branch paths needs adjacency lists, edge-id
bookkeeping and a terminal/junction/interior walk order — several hundred lines that say
nothing about the data, and that grow once it becomes clear how disconnected real
segmentations are.

None of it is necessary here. pygfx's ``LineSegmentMaterial`` "renders line segments
between each two subsequent points", so an edge list draws directly::

    positions = to_xyz(skeleton.segments())
    pygfx.Line(pygfx.Geometry(positions=positions), pygfx.LineSegmentMaterial(...))

One buffer and one draw call per body, whatever the topology — cycles, self-loops and
thousands of disconnected components all work, because nothing here ever interprets the
topology. That also means one legend entry per body rather than per branch, and one
graphics object where a fragmented body used to make hundreds.

Both sources hand over exactly this shape already: precomputed skeletons are vertices,
edges and radii; a DVID SWC table is nodes and parents, which is an edge list written
down differently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import numpy as np
from neu_vol import BBox

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
        """From what ``neu_morpho.readback.read_body_skeleton`` returns.

        That reader hands back **xyz** — the order the precomputed format stores — so
        this is where the one flip into zyx happens. It has already rejected the
        all-``-1`` sentinel radii that an ``info`` declaring no radius attribute
        produces, so nothing here re-checks that.
        """
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
        """From a node/parent table — DVID's skeleton form.

        ``coord_cols`` is read **in the order given** and then reversed, so the SWC
        default ``("x", "y", "z")`` lands as zyx. Coordinates are voxel indices, which
        is why ``frame`` is required and not optional: the only way to reach nm is the
        level's real voxel size.

        A parent id that is absent from the table — SWC's ``-1`` root, or a dangling
        reference — simply contributes no edge. Nothing is required to be a tree.
        """
        nodes = np.asarray(table[node_col], dtype=np.int64)
        parents = np.asarray(table[parent_col], dtype=np.int64)
        coords = np.column_stack([np.asarray(table[c], dtype=np.float64)
                                  for c in coord_cols])
        if not (len(nodes) == len(parents) == len(coords)):
            raise ValueError(f"ragged table: {len(nodes)} nodes, {len(parents)} "
                             f"parents, {len(coords)} coordinates")

        # Kept from the old implementation: a duplicate id makes the parent lookup
        # ambiguous, and silently resolving to the last one wires edges to the wrong
        # place. Cheap to check, invisible when wrong.
        unique, counts = np.unique(nodes, return_counts=True)
        if len(unique) != len(nodes):
            raise ValueError(f"duplicate node ids: {unique[counts > 1][:5].tolist()}")

        index_of = {int(n): i for i, n in enumerate(nodes)}
        edges = [(i, index_of[int(p)]) for i, p in enumerate(parents)
                 if int(p) in index_of]

        vertices_nm = frame.to_nm(coords[:, ::-1])
        radii = None
        if radius_col is not None and _has_column(table, radius_col):
            # A radius is a length, so it scales with the voxel — but a single number
            # cannot represent an anisotropic one. The mean is the honest summary, and
            # it is exact for the isotropic pyramids this actually meets.
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
        """``(2M, 3)`` zyx nm — consecutive pairs, one per edge.

        The whole rendering path. Pass through :func:`~neu_draw.geometry.to_xyz` and hand
        it to ``pygfx.LineSegmentMaterial``.
        """
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

        ``inside`` takes an ``(N, 3)`` zyx nm array and returns an ``(N,)`` bool array.
        Taking a **predicate rather than a mask object** is what keeps this module free
        of any volume type: whatever eventually backs a region — an RLE mask, a coarse
        dense array — plugs in here without geometry knowing.

        An edge with one endpoint kept is **bisected**, and a vertex is inserted at the
        boundary, so branches terminate at the region surface instead of at the last
        vertex inside. All crossing edges bisect together, one vectorised ``inside``
        call per round, rather than one call per point.

        Kept vertices are all retained, including any left with no edges — a vertex is
        inside or it is not, and dropping the isolated ones would quietly change the
        bounding box.

        Limitation, inherited and unchanged: an edge whose endpoints are on the same
        side is assumed not to cross. A region thinner than the edge spacing can be
        missed entirely.
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

    def __repr__(self) -> str:
        return (f"Skeleton(name={self.name!r}, vertices={len(self.vertices_zyx_nm)}, "
                f"edges={len(self.edges)}, radii={self.radii_nm is not None})")


def _bisect(p_in: np.ndarray, p_out: np.ndarray, inside: InsideFn,
            keep_inside: bool, tolerance_nm: float) -> np.ndarray:
    """Fraction along each ``p_in -> p_out`` of the last point still on the kept side.

    Bisects every segment at once: one ``inside`` call per round over all of them, and
    the round count is set by the longest, so a short segment simply converges early.
    """
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
