"""Matching one side's labels to the other's, and the completeness metrics that follow.

Also pure functions of a :class:`Contingency`. Two families live here:

**Matching.** Which segment *is* which body. Plaza & Funke's connectivity metrics need a
one-to-one assignment and are explicit about why the constraint matters: without it a
segmentation can claim credit for the same body twice, and an unmatched segment keeps none
of it. Greedy by overlap is the default; the optimal assignment is available and opt-in,
because it is the one thing in this package whose cost is not linear in anything.

**Completeness.** How much of the volume a given number of segments accounts for.
``fragmentation`` and its thresholded form are the parameter-free edit-distance estimates
from the same survey — "how many segments would a proofreader have to merge" — and
``covering`` is the size-weighted version of "how much of each body did its best segment
capture".

None of this presumes which side is the reference. ``fragmentation`` is signed with side
``b`` as the thing being assessed (more ``b`` labels than ``a`` labels is positive, i.e.
over-segmentation), which is the only asymmetry, and ``c.T`` reverses it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .overlap import Contingency

#: Refuse the optimal assignment above this many labels on either side. The dense cost
#: matrix is one thing (``n_a * n_b`` float64), but the cubic time is the real wall: this is
#: a guard against an apparent hang, not a measured break-even, and greedy is what to use
#: instead. For a sparse table `scipy.sparse.csgraph.min_weight_full_bipartite_matching` is
#: the better algorithm, but it requires a full matching to exist and raises when one does
#: not — which is the common case here, since most segments match nothing.
HUNGARIAN_MAX_LABELS = 5000


@dataclass(frozen=True)
class Match:
    """A one-to-one assignment between the two sides.

    ``a_to_b`` and ``b_to_a`` are inverse mappings over the matched labels only; a label
    absent from both is unmatched, which is a result and not a gap. ``method`` is carried
    because a greedy and an optimal assignment can differ and the number that came out of
    one should not be compared with the number that came out of the other.
    """

    a_to_b: dict[int, int]
    b_to_a: dict[int, int] = field(default_factory=dict)
    method: str = "greedy"
    min_iou: float = 0.0

    @property
    def n_matched(self) -> int:
        return len(self.a_to_b)

    def unmatched_a(self, c: Contingency) -> list[int]:
        ids, _ = c.a_totals()
        return [int(i) for i in ids if int(i) not in self.a_to_b]

    def unmatched_b(self, c: Contingency) -> list[int]:
        ids, _ = c.b_totals()
        return [int(i) for i in ids if int(i) not in self.b_to_a]


@dataclass(frozen=True)
class Completeness:
    """How many segments account for a given share of each side.

    ``fragmentation`` is ``n_b_kept - n_a_kept``: the count of extra segments a proofreader
    would have to merge to recover the kept reference bodies. Parameter-free once the
    threshold is chosen, and an estimate rather than a true edit distance — it assumes the
    ``b`` side is an over-segmented refinement of the ``a`` side, which is the regime worth
    estimating in and is worth remembering when it is not.
    """

    at: float
    n_a_kept: int
    n_b_kept: int
    a_ids: np.ndarray
    b_ids: np.ndarray

    @property
    def fragmentation(self) -> int:
        return self.n_b_kept - self.n_a_kept

    def as_dict(self) -> dict[str, float | int]:
        return {"threshold": self.at, "n_a_kept": self.n_a_kept,
                "n_b_kept": self.n_b_kept, "fragmentation": self.fragmentation}


def iou(c: Contingency) -> np.ndarray:
    """Intersection over union for each pair in ``c``, parallel to its arrays.

    ``|a ∩ b| / (|a| + |b| - |a ∩ b|)`` using each label's total over the *scored* region,
    so a body clipped by the mask is measured against the part of it that was scored. The
    alternative — its full extent — would score every boundary-clipped body as a partial
    match and make the metric a function of where the crop was cut.
    """
    if c.n_pairs == 0:
        return np.zeros(0)
    a_ids, a_counts = c.a_totals()
    b_ids, b_counts = c.b_totals()
    a_total = a_counts[np.searchsorted(a_ids, c.a_ids)].astype(np.float64)
    b_total = b_counts[np.searchsorted(b_ids, c.b_ids)].astype(np.float64)
    inter = c.counts.astype(np.float64)
    return inter / (a_total + b_total - inter)


def best_match(c: Contingency, *, method: str = "greedy", min_iou: float = 0.0,
               by: str = "iou") -> Match:
    """A one-to-one assignment between the two sides of ``c``.

    ``method="greedy"`` takes the best remaining pair repeatedly — O(pairs log pairs), and
    what to use at volume scale. ``method="hungarian"`` maximises the total instead, which
    can differ when a segment is a good-but-not-best match for two bodies; it densifies the
    table, so it is capped at :data:`HUNGARIAN_MAX_LABELS` per side.

    ``by`` chooses what "best" means: ``"iou"`` (default) or ``"overlap"`` raw voxels. They
    disagree exactly where size does — a huge segment overlapping a small body wins on
    voxels and loses on IoU — and IoU is the one that matches how a human reads "is this
    the same object".

    ``min_iou`` refuses a pair below that overlap. The default of 0 admits any co-occurring
    pair, which is right for VOI-style accounting and too permissive for reporting a body
    as "found"; 0.5 is the usual choice there, and it makes the match unique without a
    constraint, a one-voxel-over-half overlap being possible with only one partner.
    """
    if method not in ("greedy", "hungarian"):
        raise ValueError(f"method must be 'greedy' or 'hungarian', got {method!r}")
    if by not in ("iou", "overlap"):
        raise ValueError(f"by must be 'iou' or 'overlap', got {by!r}")

    if c.n_pairs == 0:
        return Match({}, {}, method=method, min_iou=min_iou)

    scores = iou(c) if by == "iou" else c.counts.astype(np.float64)
    admissible = iou(c) >= min_iou if min_iou > 0 else np.ones(c.n_pairs, dtype=bool)

    if method == "greedy":
        a_to_b: dict[int, int] = {}
        b_to_a: dict[int, int] = {}
        # Descending score; ties broken by id so the result does not depend on the sort's
        # internal order, for the same reason the contingency table is canonical.
        order = np.lexsort((c.b_ids, c.a_ids, -scores))
        for k in order:
            if not admissible[k]:
                continue
            a_id, b_id = int(c.a_ids[k]), int(c.b_ids[k])
            if a_id in a_to_b or b_id in b_to_a:
                continue
            a_to_b[a_id] = b_id
            b_to_a[b_id] = a_id
        return Match(a_to_b, b_to_a, method=method, min_iou=min_iou)

    a_ids, _ = c.a_totals()
    b_ids, _ = c.b_totals()
    if max(a_ids.size, b_ids.size) > HUNGARIAN_MAX_LABELS:
        raise ValueError(
            f"{a_ids.size} and {b_ids.size} labels: the optimal assignment densifies the "
            f"table and is cubic, so it is capped at {HUNGARIAN_MAX_LABELS} per side. Use "
            f"method='greedy', or restrict the table first "
            f"(Contingency.restrict / thresholded completeness).")

    from scipy.optimize import linear_sum_assignment

    cost = np.zeros((a_ids.size, b_ids.size))
    ai = np.searchsorted(a_ids, c.a_ids)
    bi = np.searchsorted(b_ids, c.b_ids)
    cost[ai[admissible], bi[admissible]] = -scores[admissible]
    rows, cols = linear_sum_assignment(cost)
    a_to_b, b_to_a = {}, {}
    for r, col in zip(rows, cols):
        if cost[r, col] == 0:
            # linear_sum_assignment fills the rectangle, so it pairs labels that share no
            # voxels at all. Those are not matches; dropping them is what keeps
            # `n_matched` meaningful.
            continue
        a_to_b[int(a_ids[r])] = int(b_ids[col])
        b_to_a[int(b_ids[col])] = int(a_ids[r])
    return Match(a_to_b, b_to_a, method=method, min_iou=min_iou)


def fragmentation(c: Contingency) -> int:
    """``n_b - n_a`` over the scored region: extra segments, as an edit-distance estimate."""
    return c.n_b - c.n_a


def completeness(c: Contingency, at: float = 0.9) -> Completeness:
    """The smallest set of labels on each side covering ``at`` of that side's voxels.

    Answers "how many segments reconstruct 90% of this volume", which is the form of the
    question a proofreading plan can act on — where a raw label count is dominated by
    specks. Both sides are thresholded independently, because the point is to compare *how
    many* each needs, not to pair them up.
    """
    if not 0 < at <= 1:
        raise ValueError(f"at must be in (0, 1], got {at}")
    a_keep = _top_labels(*c.a_totals(), at)
    b_keep = _top_labels(*c.b_totals(), at)
    return Completeness(at=at, n_a_kept=a_keep.size, n_b_kept=b_keep.size,
                        a_ids=a_keep, b_ids=b_keep)


def covering(c: Contingency) -> float:
    """Size-weighted mean, over side ``a``'s labels, of the best single ``b`` overlap.

    For each ``a`` body, the IoU of its best-matching ``b`` segment; averaged with each
    body weighted by its size. 1.0 when every body is captured exactly by one segment.
    Unlike VOI this is bounded and reads as a fraction, which is why it survives into
    figures; unlike VOI it is also blind to what happens to the *rest* of the best-matching
    segment, so it does not notice a merge that leaves the body itself intact.
    """
    if c.n_pairs == 0:
        return 0.0
    scores = iou(c)
    a_ids, a_counts = c.a_totals()
    index = np.searchsorted(a_ids, c.a_ids)
    best = np.zeros(a_ids.size)
    np.maximum.at(best, index, scores)
    weights = a_counts.astype(np.float64)
    return float((best * weights).sum() / weights.sum())


def summary(c: Contingency, *, at: tuple[float, ...] = (0.5, 0.9)) -> dict[str, float | int]:
    """Matching and completeness as one flat row, to merge with :func:`voxel.summary`."""
    row: dict[str, float | int] = {
        "fragmentation": fragmentation(c),
        "covering": covering(c),
    }
    match = best_match(c, min_iou=0.5)
    a_ids, _ = c.a_totals()
    row["n_matched_iou50"] = match.n_matched
    row[f"frac_{c.labels[0]}_matched_iou50"] = (
        (match.n_matched / a_ids.size) if a_ids.size else 0.0)
    for threshold in at:
        comp = completeness(c, threshold)
        tag = f"{int(round(threshold * 100))}"
        row[f"n_{c.labels[0]}_at{tag}"] = comp.n_a_kept
        row[f"n_{c.labels[1]}_at{tag}"] = comp.n_b_kept
        row[f"fragmentation_at{tag}"] = comp.fragmentation
    return row


def _top_labels(ids: np.ndarray, counts: np.ndarray, at: float) -> np.ndarray:
    """The largest labels whose sizes first reach ``at`` of the total, largest first."""
    if ids.size == 0:
        return ids
    order = np.argsort(counts)[::-1]
    cumulative = np.cumsum(counts[order].astype(np.float64))
    target = at * float(counts.sum())
    # searchsorted gives the first index at or past the target; +1 makes it a count.
    k = int(np.searchsorted(cumulative, target)) + 1
    return ids[order[:min(k, ids.size)]]
