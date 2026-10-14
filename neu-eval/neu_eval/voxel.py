"""Voxel-level agreement metrics. Every one is a pure function of a :class:`Contingency`.

No arrays and no I/O, so computing several metrics is cheap; Plaza & Funke advise reading
several together rather than trusting one.

**Which half of VOI is "split" and which is "merge" depends on which side is the
reference**, so nothing here returns a bare pair. With side ``a`` as the reference: if
``b`` fuses two ``a`` bodies, a voxel's ``b`` label no longer tells you its ``a`` body, so
``H(a|b)`` grows, and that is ``voi_merge``; if one ``a`` body is spread over several
``b`` segments, ``H(b|a)`` grows, and that is ``voi_split``. Plaza & Funke call these
under-segmentation ``H(G|S)`` and over-segmentation ``H(S|G)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .overlap import Contingency


@dataclass(frozen=True)
class Voi:
    """Variation of information, decomposed, in **bits**.

    Named with side ``a`` as the reference: ``merge = H(a|b)`` (``b`` fusing ``a``
    bodies), ``split = H(b|a)`` (``b`` fragmenting one). ``total`` is their sum.
    """

    split: float
    merge: float

    @property
    def total(self) -> float:
        return self.split + self.merge

    def as_dict(self) -> dict[str, float]:
        return {"voi_split": self.split, "voi_merge": self.merge, "voi": self.total}


@dataclass(frozen=True)
class Rand:
    """Adapted Rand, as a precision/recall pair plus its F-score and the error.

    ``precision``: of the voxel pairs ``b`` puts in one segment, the fraction ``a`` also
    puts together. ``recall``: of the pairs ``a`` puts together, the fraction ``b`` keeps
    together. ``error = 1 - f_score`` is the usual "adapted Rand error".
    """

    precision: float
    recall: float

    @property
    def f_score(self) -> float:
        denom = self.precision + self.recall
        return 0.0 if denom == 0 else 2 * self.precision * self.recall / denom

    @property
    def error(self) -> float:
        return 1.0 - self.f_score

    def as_dict(self) -> dict[str, float]:
        return {"rand_precision": self.precision, "rand_recall": self.recall,
                "rand_f_score": self.f_score, "rand_error": self.error}


def voi(c: Contingency) -> Voi:
    """Variation of information between the two sides of ``c``, in bits.

    Zero when the labelings agree up to renaming. An empty table also scores zero; check
    ``frac_scored``, which shows when nothing was scored.
    """
    total = c.n_scored
    if total == 0:
        return Voi(0.0, 0.0)

    p_ab = c.counts.astype(np.float64) / total
    _, a_counts = c.a_totals()
    _, b_counts = c.b_totals()
    p_a = a_counts.astype(np.float64) / total
    p_b = b_counts.astype(np.float64) / total

    h_a = _entropy(p_a)
    h_b = _entropy(p_b)
    h_joint = _entropy(p_ab)
    # H(a|b) = H(a,b) - H(b); clamped at zero against float cancellation.
    return Voi(split=max(h_joint - h_a, 0.0), merge=max(h_joint - h_b, 0.0))


def per_label_voi(c: Contingency) -> tuple[dict[int, float], dict[int, float]]:
    """Each label's contribution to the two halves of :func:`voi`.

    ``(merge_by_a, split_by_b)``: each ``a`` id's share of ``H(a|b)`` and each ``b`` id's
    share of ``H(b|a)``; each sums to the matching field of :func:`voi`. VOI contribution
    ranks errors better than voxel counts do: halving a large body and shaving a thousand
    voxels off it score the same in voxels.
    """
    total = c.n_scored
    if total == 0:
        return {}, {}

    p_ab = c.counts.astype(np.float64) / total
    a_ids, a_counts = c.a_totals()
    b_ids, b_counts = c.b_totals()
    p_a = a_counts.astype(np.float64) / total
    p_b = b_counts.astype(np.float64) / total

    a_index = np.searchsorted(a_ids, c.a_ids)
    b_index = np.searchsorted(b_ids, c.b_ids)

    # H(a|b) = -sum_ij p_ij log2(p_ij / p_j), gathered onto a rather than onto b.
    merge_terms = -p_ab * np.log2(p_ab / p_b[b_index])
    split_terms = -p_ab * np.log2(p_ab / p_a[a_index])

    merge_by_a = np.bincount(a_index, weights=merge_terms, minlength=a_ids.size)
    split_by_b = np.bincount(b_index, weights=split_terms, minlength=b_ids.size)
    return (
        {int(i): float(v) for i, v in zip(a_ids, merge_by_a)},
        {int(i): float(v) for i, v in zip(b_ids, split_by_b)},
    )


def pair_voi(c: Contingency) -> tuple[np.ndarray, np.ndarray]:
    """Per-**pair** VOI contributions, parallel to ``c``'s arrays.

    ``(merge_term, split_term)`` for each ``(a_id, b_id)`` row, so a caller can rank pairs
    without re-deriving the entropy. Their sums are :func:`voi`'s two fields.
    """
    total = c.n_scored
    if total == 0:
        return np.zeros(0), np.zeros(0)

    p_ab = c.counts.astype(np.float64) / total
    a_ids, a_counts = c.a_totals()
    b_ids, b_counts = c.b_totals()
    p_a = (a_counts.astype(np.float64) / total)[np.searchsorted(a_ids, c.a_ids)]
    p_b = (b_counts.astype(np.float64) / total)[np.searchsorted(b_ids, c.b_ids)]
    return -p_ab * np.log2(p_ab / p_b), -p_ab * np.log2(p_ab / p_a)


def rand(c: Contingency) -> Rand:
    """Adapted Rand precision, recall and F-score for the two sides of ``c``.

    Counts of *pairs of voxels*, computed from the sums of squares rather than by
    enumeration: ``sum_ij n_ij^2`` for pairs the two sides agree on, against ``sum_j n_j^2``
    and ``sum_i n_i^2`` for the pairs each side claims.
    """
    total = c.n_scored
    if total <= 1:
        return Rand(1.0, 1.0)

    n_ij = c.counts.astype(np.float64)
    _, a_counts = c.a_totals()
    _, b_counts = c.b_totals()

    # Unordered pairs within each group: n*(n-1)/2. The -n/2 terms cancel a group of one,
    # which otherwise contributes a pair it does not have.
    agree = float((n_ij * (n_ij - 1)).sum()) / 2
    in_b = float((b_counts.astype(np.float64) * (b_counts.astype(np.float64) - 1)).sum()) / 2
    in_a = float((a_counts.astype(np.float64) * (a_counts.astype(np.float64) - 1)).sum()) / 2

    precision = 1.0 if in_b == 0 else agree / in_b
    recall = 1.0 if in_a == 0 else agree / in_a
    return Rand(precision=precision, recall=recall)


def summary(c: Contingency) -> dict[str, float | int]:
    """Every scalar in this module, plus what was scored, as one flat row (so many
    comparisons make a DataFrame). ``frac_scored`` is included because a VOI over 3% of a
    box is not comparable to one over all of it."""
    n_total = c.n_scored + c.n_ignored
    row: dict[str, float | int] = {
        f"n_{c.labels[0]}": c.n_a,
        f"n_{c.labels[1]}": c.n_b,
        "n_pairs": c.n_pairs,
        "n_scored": c.n_scored,
        "n_ignored": c.n_ignored,
        "frac_scored": (c.n_scored / n_total) if n_total else 0.0,
    }
    row.update(voi(c).as_dict())
    row.update(rand(c).as_dict())
    return row


def _entropy(p: np.ndarray) -> float:
    """``-sum p log2 p`` over the nonzero entries, in bits."""
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log2(p)).sum())
