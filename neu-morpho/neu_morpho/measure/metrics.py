"""Per-body measurements from a skeleton. Pure: arrays in, numbers out.

Everything takes a :class:`neu_lib.Skeleton`, so a *published* skeleton can be measured
without rerunning anything, and so can an excluded one (``skel.exclude(inside)``).

**Diameter statistics are weighted by edge length, never per vertex**: node density is a
property of the skeletonizer, and a densely sampled region would dominate a vertex mean.

**The radii are INSCRIBED radii**: exact at the node, but systematically small wherever a
cross-section is not round. Volume goes as r^2, so :func:`frustum_volume_nm3` is a lower
bound whose tightness depends on how round the neurites are.

**Missing radii give NaN, not zero**: an unknown calibre must not read as "infinitely thin"
in an average.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from neu_lib import Skeleton


def _edges(skeleton: Skeleton) -> tuple[np.ndarray, np.ndarray]:
    """``(edge_lengths_nm, edge_index)``. Zero-length edges are kept: they weigh nothing,
    and dropping them would change the topology counts."""
    v = np.asarray(skeleton.vertices_zyx_nm, dtype=float)
    e = np.asarray(skeleton.edges, dtype=np.int64).reshape(-1, 2)
    if not len(e):
        return np.zeros(0), e
    return np.linalg.norm(v[e[:, 1]] - v[e[:, 0]], axis=1), e


def cable_length_nm(skeleton: Skeleton) -> float:
    """Total edge length, in nm.

    Sensitive to skeleton postprocessing (joins add cable, tick removal drops it), so it is
    only comparable across datasets skeletonized with the same settings.
    """
    lengths, _ = _edges(skeleton)
    return float(lengths.sum())


def max_radius_nm(skeleton: Skeleton) -> float:
    """The largest vertex radius; NaN without radii.

    Non-positive radii are skipped: kimimaro writes -1 where a radius is unknown.
    """
    if skeleton.radii_nm is None:
        return float("nan")
    r = np.asarray(skeleton.radii_nm, dtype=float)
    r = r[r > 0]
    return float(r.max()) if r.size else float("nan")


def weighted_quantile(values: Any, weights: Any, q: Any) -> np.ndarray:
    """Quantiles of ``values`` weighted by ``weights``. **Step, not interpolated.**

    Each edge is a length of cable with one diameter, so the weighted median is "the
    diameter at which half the cable is thinner": an empirical distribution over cable.
    Interpolating would let a few sparse samples drag the answer away from where the cable
    is. Weights of zero are allowed and do not count.
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    q = np.atleast_1d(np.asarray(q, dtype=float))
    if v.shape != w.shape:
        raise ValueError(f"values {v.shape} and weights {w.shape} differ in shape")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    keep = np.isfinite(v) & (w > 0)
    if not keep.any():
        return np.full(len(q), np.nan)
    v, w = v[keep], w[keep]
    order = np.argsort(v)
    v, w = v[order], w[order]
    cum = np.cumsum(w)
    idx = np.clip(np.searchsorted(cum, q * cum[-1], side="left"), 0, len(v) - 1)
    return v[idx]


def diameter_stats(skeleton: Skeleton) -> dict:
    """Edge-length-weighted diameter statistics, in nm.

    Each edge contributes the mean of its two endpoint radii, weighted by its length; NaN
    throughout without radii.

    **p10-p90 describes the twigs, not the trunk**: the thick part of a cell is a small
    length fraction and sits above p90, so read ``p99`` / ``max`` for the thick end.
    Percentiles are steps (:func:`weighted_quantile`), so where one edge dominates the
    length they collapse onto its diameter.
    """
    lengths, e = _edges(skeleton)
    keys = ("diameter_nm_mean", "diameter_nm_p10", "diameter_nm_median",
            "diameter_nm_p90", "diameter_nm_p99", "diameter_nm_max")
    nan = {k: np.nan for k in keys}
    if skeleton.radii_nm is None or not len(e):
        return nan

    r = np.asarray(skeleton.radii_nm, dtype=float)
    edge_diameter = r[e[:, 0]] + r[e[:, 1]]          # (r0 + r1)/2 * 2 == r0 + r1
    keep = lengths > 0
    if not keep.any():
        return nan
    d, w = edge_diameter[keep], lengths[keep]
    p10, median, p90, p99 = weighted_quantile(d, w, [0.1, 0.5, 0.9, 0.99])
    return {
        "diameter_nm_mean": float(np.average(d, weights=w)),
        "diameter_nm_p10": float(p10),
        "diameter_nm_median": float(median),
        "diameter_nm_p90": float(p90),
        "diameter_nm_p99": float(p99),
        "diameter_nm_max": float(d.max()),
    }


def frustum_volume_nm3(skeleton: Skeleton) -> float:
    """Volume of the tube the radii imply: one truncated cone per edge.

    ``V = (pi/3) * L * (r0^2 + r0*r1 + r1^2)`` per edge, summed. Cones overlapping at a
    junction are double-counted: the opposite sign to the inscribed-radius bias, but not a
    cancellation of it. NaN without radii.
    """
    lengths, e = _edges(skeleton)
    if skeleton.radii_nm is None or not len(e):
        return float("nan")
    r = np.asarray(skeleton.radii_nm, dtype=float)
    r0, r1 = r[e[:, 0]], r[e[:, 1]]
    return float((np.pi / 3.0 * lengths * (r0 * r0 + r0 * r1 + r1 * r1)).sum())


def frustum_area_nm2(skeleton: Skeleton) -> float:
    """Lateral surface area of that same tube: ``pi * (r0 + r1) * slant`` per edge.

    End caps are excluded (a neurite's ends are cuts, not surface). NaN without radii.
    """
    lengths, e = _edges(skeleton)
    if skeleton.radii_nm is None or not len(e):
        return float("nan")
    r = np.asarray(skeleton.radii_nm, dtype=float)
    r0, r1 = r[e[:, 0]], r[e[:, 1]]
    slant = np.sqrt(lengths * lengths + (r1 - r0) ** 2)
    return float((np.pi * (r0 + r1) * slant).sum())


def topology(skeleton: Skeleton) -> dict:
    """Tip, branch-point and connected-component counts from the edge list.

    Segmented bodies are often genuinely fragmented, so more than one component is correct
    reporting, not a defect.
    """
    _, e = _edges(skeleton)
    n = len(np.asarray(skeleton.vertices_zyx_nm, dtype=float))
    if not len(e):
        return {"n_vertices": n, "n_edges": 0, "n_tips": 0, "n_branches": 0,
                "n_components": n}

    degree = np.bincount(e.ravel(), minlength=n)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    w = np.ones(len(e))
    graph = coo_matrix((w, (e[:, 0], e[:, 1])), shape=(n, n))
    n_components, _ = connected_components(graph, directed=False)
    return {
        "n_vertices": n,
        "n_edges": int(len(e)),
        "n_tips": int((degree == 1).sum()),
        "n_branches": int((degree > 2).sum()),
        "n_components": int(n_components),
    }


def measure_skeleton(skeleton: Skeleton, *, variant: str = "all",
                     body_id: Optional[int] = None,
                     dataset: Optional[str] = None) -> dict:
    """Every skeleton-derived measurement for one body, as one flat row.

    ``variant`` labels what was measured (``"all"``, ``"minus_nucleus"``,
    ``"minus_soma_nucleus"``, ``"neurite_positive"``) and is carried into the row, since
    rows measured differently cannot be compared safely.
    """
    row: dict[str, Any] = {
        "dataset": dataset,
        "body_id": None if body_id is None else int(body_id),
        "variant": str(variant),
        "cable_length_nm": cable_length_nm(skeleton),
        "volume_nm3_frustum": frustum_volume_nm3(skeleton),
        "area_nm2_frustum": frustum_area_nm2(skeleton),
    }
    row.update(diameter_stats(skeleton))
    row.update(topology(skeleton))
    return row
