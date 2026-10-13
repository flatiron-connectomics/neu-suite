"""Voxel-level agreement metrics. Every one is a pure function of a :class:`Contingency`.

No arrays, no I/O, no backend — the expensive pass already happened. That is what makes
computing several metrics affordable, which matters because the connectomics literature's
own advice is not to trust any single one: Plaza & Funke's survey declines to recommend a
metric and recommends the practice of reading several together, each catching what another
flatters.

**The direction of VOI is the trap this module is arranged around.** ``VI(A, B) =
H(A|B) + H(B|A)``, and which half is "split" and which is "merge" depends on which side is
the reference — a fact that is invisible in a two-tuple and has been getting people wrong
for as long as the metric has been used. So nothing here returns a bare pair. :class:`Voi`
names its fields, ``voi_split`` and ``voi_merge`` are defined against side ``a`` as the
reference, and two tests pin the direction with a hand-built pure-split case and a
hand-built pure-merge case rather than trusting the algebra to be read correctly.

Reading it off: if side ``b`` fused two of side ``a``'s bodies, then knowing a voxel's
``b`` label does not tell you which ``a`` body it came from, so ``H(a|b)`` is what grows —
that is ``voi_merge``. Symmetrically ``H(b|a)`` grows when one ``a`` body is scattered
across many ``b`` segments, which is ``voi_split``. Plaza & Funke label the same two
quantities "under-segmentation ``H(G|S)``" and "over-segmentation ``H(S|G)``".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .overlap import Contingency


@dataclass(frozen=True)
class Voi:
    """Variation of information, decomposed, in **bits**.

    ``split`` and ``merge`` are named against side ``a`` of the table as the reference:
    ``merge = H(a|b)`` counts side ``b`` fusing distinct ``a`` bodies, ``split = H(b|a)``
    counts side ``b`` fragmenting one. ``total`` is their sum, the metric proper.

    Bits rather than nats because the unit gets quoted and "1 bit of split error" has a
    reading — one binary decision per voxel — where 0.69 nats has none.
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

    Pair-counting: of all voxel pairs that side ``b`` puts in one segment, what fraction
    does side ``a`` agree belong together (``precision``), and of all pairs side ``a`` puts
    together, what fraction does ``b`` keep together (``recall``). ``error = 1 - f_score``
    is the number usually reported as "adapted Rand error".
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

    Zero when the two labelings agree up to a renaming; growing with both kinds of
    disagreement. An empty table scores zero, on the grounds that no voxels were scored and
    a metric should not invent a disagreement out of nothing — a report says how much was
    scored alongside, which is where an empty region becomes visible.
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
    # H(a|b) = H(a,b) - H(b), and mutual information is never negative, so both
    # conditionals are clamped at zero against float cancellation on a near-perfect match.
    return Voi(split=max(h_joint - h_a, 0.0), merge=max(h_joint - h_b, 0.0))


def per_label_voi(c: Contingency) -> tuple[dict[int, float], dict[int, float]]:
    """Each label's contribution to the two halves of :func:`voi`.

    ``(merge_by_a, split_by_b)``: the first maps an ``a`` id to its share of ``H(a|b)``,
    the second a ``b`` id to its share of ``H(b|a)``. Both sum to the corresponding field
    of :func:`voi`, which a test asserts — a decomposition that does not add up to the
    total is a decomposition nobody can act on.

    This is what localizes error, and it is also the ranking key the disagreement table
    uses. A pair's VOI contribution is a better severity measure than its voxel count,
    because splitting a large body into two halves and shaving a thousand voxels off it are
    different failures that a voxel count scores the same.
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
    """Every scalar in this module, plus what was scored, as one flat row.

    The flat-row shape is deliberate and matches ``neu_morpho.measure.measure_skeleton``:
    one dict per comparison, so a set of comparisons is a DataFrame without reshaping.

    ``frac_scored`` is here rather than left to a reporting layer because a VOI computed
    over 3% of a box is not a comparable number, and the metric and its denominator
    travelling separately is how it gets quoted as if it were.
    """
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
