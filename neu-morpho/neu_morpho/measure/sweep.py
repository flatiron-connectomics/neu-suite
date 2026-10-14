"""Per-body volume and local calibre, by counting voxels against a published centreline.

The measurement is ``V/L``: a body's voxel count gives its volume, and dividing by the
cable length of the centreline through it gives the **mean cross-sectional area**. Binning
the voxels by nearest centreline node turns the same pass into a *length-weighted
distribution* of local cross-section rather than one number per body.

The defaults (read block, the 32 nm level, dropping zeros before counting) were each
chosen by measurement; re-measure before changing one.

- **``V/L`` is a TUBE measure and under-reads blobs** (a sphere comes back at
  ``sqrt(2/3)`` of its radius); on a body with a soma the thick tail is "``V/L`` over a
  blob", not a diameter. :func:`blob_signal` flags it per body.
- **Per-node values are noisy (±40%)**: this is an instrument for distributions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

#: Read unit, in voxels per side.
DEFAULT_BLOCK = 512

#: Level to sweep. Volume at 32 nm is within 1-3% of 8 nm, and the recovered radius is flat
#: from 8 to 64 nm, so finer buys nothing.
DEFAULT_VOXEL_NM = 32.0


# --------------------------------------------------------------------------- #
# which blocks to read
# --------------------------------------------------------------------------- #
def roi_block_mask(roi: np.ndarray, labels: Sequence[int], factor: int,
                   dilate: int = 1) -> np.ndarray:
    """Boolean grid of blocks intersecting ``labels``, from a coarse ROI volume.

    ``factor`` is how many ROI voxels span one block on a side. Reduced by ``any``: a
    block holding one ROI voxel still holds tissue.

    **Dilated by one block by default**: a process near the ROI boundary has voxels just
    outside, and a missed block silently truncates a body while an extra one costs a read.
    The dilation does not wrap around the grid.
    """
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    keep = np.isin(roi, np.asarray(labels))
    pad = [(0, (-s) % factor) for s in keep.shape]
    padded = np.pad(keep, pad)
    grid = padded.reshape(padded.shape[0] // factor, factor,
                          padded.shape[1] // factor, factor,
                          padded.shape[2] // factor, factor).any(axis=(1, 3, 5))
    n = max(0, int(dilate))
    if n:
        from scipy.ndimage import binary_dilation
        grid = binary_dilation(grid, iterations=n)
    return grid


def blocks_from_mask(grid: np.ndarray, block: int, shape: Sequence[int]) -> list[tuple]:
    """``[(z0, y0, x0), …]`` block origins, in voxels, clipped to ``shape``."""
    shape = tuple(int(s) for s in shape)
    out = []
    for idx in zip(*np.nonzero(grid)):
        origin = tuple(int(i) * block for i in idx)
        if all(o < s for o, s in zip(origin, shape)):
            out.append(origin)
    return out


# --------------------------------------------------------------------------- #
# the per-block work — pure, so it is testable without a store
# --------------------------------------------------------------------------- #
def count_labels(block: np.ndarray, background: int = 0) -> dict[int, int]:
    """``{label: voxel_count}`` for one block, background excluded.

    Zeros are dropped before ``np.unique``, which is 17x faster on mostly empty volumes.
    """
    flat = np.asarray(block).reshape(-1)
    nz = flat[flat != background]
    if not nz.size:
        return {}
    labels, counts = np.unique(nz, return_counts=True)
    return {int(a): int(b) for a, b in zip(labels, counts)}


def bin_to_nodes(block: np.ndarray, origin_vox: Sequence[int], voxel_nm: float,
                 nodes_zyx_nm: np.ndarray, body_id: int,
                 node_ids: Optional[Sequence[int]] = None) -> dict:
    """Voxels of ``body_id`` assigned to their nearest centreline node.

    Returns ``{"counts": {node: n}, "dist_sum": {node: nm}}``; the summed voxel-to-node
    distance feeds :func:`blob_signal`. Coarse node spacing (to ~800 nm) leaves the
    distribution unchanged.
    """
    arr = np.asarray(block)
    hit = arr == body_id
    if not hit.any():
        return {"counts": {}, "dist_sum": {}}
    zz, yy, xx = np.nonzero(hit)
    pts = np.stack([(origin_vox[0] + zz + 0.5) * voxel_nm,
                    (origin_vox[1] + yy + 0.5) * voxel_nm,
                    (origin_vox[2] + xx + 0.5) * voxel_nm], axis=1)
    from scipy.spatial import cKDTree

    dist, near = cKDTree(np.asarray(nodes_zyx_nm, dtype=float)).query(pts, k=1)
    n = len(nodes_zyx_nm)
    counts = np.bincount(near, minlength=n)
    dsum = np.bincount(near, weights=dist, minlength=n)
    ids = list(range(n)) if node_ids is None else list(node_ids)
    return {"counts": {int(ids[i]): int(c) for i, c in enumerate(counts) if c},
            "dist_sum": {int(ids[i]): float(dsum[i]) for i, c in enumerate(counts) if c}}


# --------------------------------------------------------------------------- #
# accumulation
# --------------------------------------------------------------------------- #
@dataclass
class SweepTotals:
    """Running per-body and per-node counts across blocks. The driver owns one."""

    voxel_nm: float = DEFAULT_VOXEL_NM
    body_voxels: dict[int, int] = field(default_factory=dict)
    node_voxels: dict[int, dict[int, int]] = field(default_factory=dict)
    node_dist_sum: dict[int, dict[int, float]] = field(default_factory=dict)
    blocks: int = 0

    @property
    def voxel_volume_nm3(self) -> float:
        return float(self.voxel_nm) ** 3

    def add_block(self, per_label: Mapping[int, int]) -> None:
        for label, n in per_label.items():
            self.body_voxels[int(label)] = self.body_voxels.get(int(label), 0) + int(n)
        self.blocks += 1

    def add_nodes(self, body_id: int, binned: Mapping[str, Mapping[int, float]]) -> None:
        counts = self.node_voxels.setdefault(int(body_id), {})
        for node, n in binned.get("counts", {}).items():
            counts[int(node)] = counts.get(int(node), 0) + int(n)
        dsum = self.node_dist_sum.setdefault(int(body_id), {})
        for node, d in binned.get("dist_sum", {}).items():
            dsum[int(node)] = dsum.get(int(node), 0.0) + float(d)

    def volume_nm3(self, body_id: int) -> float:
        return self.body_voxels.get(int(body_id), 0) * self.voxel_volume_nm3


# --------------------------------------------------------------------------- #
# derived quantities
# --------------------------------------------------------------------------- #
def mean_cross_section(volume_nm3: float, cable_nm: float) -> tuple[float, float]:
    """``(area_nm2, equivalent_radius_nm)`` from a volume and the cable through it.

    NaN for zero cable rather than infinity, which would propagate into every aggregate.
    """
    if not cable_nm or cable_nm <= 0:
        return float("nan"), float("nan")
    area = float(volume_nm3) / float(cable_nm)
    return area, float(np.sqrt(area / np.pi))


def node_radii(node_counts: Mapping[int, int], cable_share_nm: Mapping[int, float],
               voxel_nm: float = DEFAULT_VOXEL_NM) -> tuple[np.ndarray, np.ndarray]:
    """``(radii_nm, weights_nm)`` per node — the local ``V/L`` and its cable weight.

    Weights are cable length: node density is a property of the skeletonizer, not the
    neuron.
    """
    radii, weights = [], []
    vol = float(voxel_nm) ** 3
    for node, n in node_counts.items():
        share = float(cable_share_nm.get(node, 0.0))
        if share <= 0 or n <= 0:
            continue
        radii.append(np.sqrt((n * vol / share) / np.pi))
        weights.append(share)
    return np.asarray(radii, dtype=float), np.asarray(weights, dtype=float)


def blob_signal(node_counts: Mapping[int, int], node_dist_sum: Mapping[int, float],
                cable_share_nm: Mapping[int, float],
                voxel_nm: float = DEFAULT_VOXEL_NM, tolerance: float = 1.25) -> float:
    """Share of cable whose voxels are spread more widely than a tube can explain.

    A soma, a bulb, or two merged branches put voxels further from their node than a
    cylinder would. The comparison is against a cylinder's expected mean distance, which
    has an axial part from the node spacing: for radius ``r`` and node cable share ``s``,

        E[dist] ~ sqrt(E[rho^2] + E[zeta^2]) = sqrt(r^2/2 + s^2/12)

    Comparing against ``r`` alone flags ordinary tubes whenever ``s`` is comparable to it.
    """
    vol = float(voxel_nm) ** 3
    blob = total = 0.0
    for node, n in node_counts.items():
        share = float(cable_share_nm.get(node, 0.0))
        if share <= 0 or n <= 0:
            continue
        r = np.sqrt((n * vol / share) / np.pi)
        expected = np.sqrt(r * r / 2.0 + share * share / 12.0)
        mean_d = float(node_dist_sum.get(node, 0.0)) / n
        total += share
        if expected > 0 and mean_d > tolerance * expected:
            blob += share
    return float(blob / total) if total else float("nan")


def cable_shares(vertices_zyx_nm: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Cable length attributable to each node: half of every incident edge."""
    v = np.asarray(vertices_zyx_nm, dtype=float)
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    out = np.zeros(len(v))
    if not len(e):
        return out
    seg = np.linalg.norm(v[e[:, 1]] - v[e[:, 0]], axis=1)
    np.add.at(out, e[:, 0], seg / 2.0)
    np.add.at(out, e[:, 1], seg / 2.0)
    return out


# --------------------------------------------------------------------------- #
# the histogram
# --------------------------------------------------------------------------- #
def log_bin_edges(lo_nm: float, hi_nm: float, nbins: int) -> np.ndarray:
    """``nbins + 1`` log-spaced edges: datasets compared on one axis differ by decades."""
    if not (0 < lo_nm < hi_nm):
        raise ValueError(f"need 0 < lo < hi, got lo={lo_nm} hi={hi_nm}")
    if nbins < 1:
        raise ValueError(f"nbins must be >= 1, got {nbins}")
    return np.geomspace(float(lo_nm), float(hi_nm), int(nbins) + 1)


def weighted_histogram(values: np.ndarray, weights: np.ndarray,
                       edges: np.ndarray) -> np.ndarray:
    """``nbins + 2`` bins: one **underflow**, the ``nbins`` interior, one **overflow**.

    The catch-alls make ``hist.sum() == weights.sum()`` hold exactly, so no cable is lost
    to clipping and the total checks the bin range.
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if v.shape != w.shape:
        raise ValueError(f"values {v.shape} and weights {w.shape} differ")
    edges = np.asarray(edges, dtype=float)
    out = np.zeros(len(edges) + 1)
    if not v.size:
        return out
    good = np.isfinite(v)
    out[0] = w[good & (v < edges[0])].sum()
    out[-1] = w[good & (v >= edges[-1])].sum()
    mid = good & (v >= edges[0]) & (v < edges[-1])
    if mid.any():
        which = np.searchsorted(edges, v[mid], side="right") - 1
        out[1:-1] = np.bincount(which, weights=w[mid], minlength=len(edges) - 1)
    return out
