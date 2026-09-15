"""One call from two registered pieces to everything a comparison produces.

``compare(a, b)`` is the entry point: it builds the table once, computes every metric off
it, ranks and locates the disagreements, and — if a reviewed verdict file is supplied —
rescores without the pairs the reviewer judged to be reference defects. A :class:`Report`
carries the lot, raw and adjudicated side by side.

**The two pieces must be registered, and there is exactly one right way to get that.**
``read_piece(seg, level=…, crop=gt_piece)`` converts the reference's physical box using the
target level's own voxel size and origin, so the two crops describe the same nanometres by
construction. Any other route — matching shapes by hand, assuming a ``2**level`` factor —
is how a half-voxel shift becomes a metric nobody can explain. This module checks the
frames agree and says so when they do not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from . import assign, disagree, voxel
from .overlap import Contingency, contingency

#: A reference label whose bounding box holds this many times more voxels than the label
#: itself is flagged as possibly scattered. Heuristic, and reported rather than refused —
#: see :func:`scattered_labels`.
SCATTER_FACTOR = 1000.0

#: Labels smaller than this are never flagged as scattered. A two-voxel label at opposite
#: corners has an enormous scatter ratio and tells you nothing.
SCATTER_MIN_VOXELS = 100


@dataclass(frozen=True)
class Report:
    """Everything one comparison produced.

    ``summary`` and ``adjudicated_summary`` are flat dicts, one row each, so a set of
    comparisons is a DataFrame without reshaping. ``adjudicated_*`` are ``None`` when no
    verdicts were supplied — distinct from "supplied and changed nothing", which gives a
    summary equal to the raw one and a ``verdict_tally`` saying why.
    """

    contingency: Contingency
    summary: dict
    rows: list[dict]
    labels: tuple[str, str]
    warnings: list[str] = field(default_factory=list)
    adjudicated: Contingency | None = None
    adjudicated_summary: dict | None = None
    verdict_tally: dict | None = None

    @property
    def kind_counts(self) -> dict[str, int]:
        return disagree.counts_by_kind(self.rows)

    def headline(self) -> str:
        """A one-line summary for a terminal or a log."""
        s = self.summary
        kinds = self.kind_counts
        text = (f"{self.labels[0]} vs {self.labels[1]}: "
                f"VOI {s['voi']:.4f} (split {s['voi_split']:.4f}, merge {s['voi_merge']:.4f}), "
                f"Rand err {s['rand_error']:.4f}, covering {s['covering']:.4f} | "
                f"{s['n_scored']:,} voxels scored ({s['frac_scored']:.1%}) | "
                f"{kinds['split']} split, {kinds['merge']} merge, {kinds['tangle']} tangle")
        if self.adjudicated_summary is not None:
            a = self.adjudicated_summary
            text += (f"\nadjudicated: VOI {a['voi']:.4f}, Rand err {a['rand_error']:.4f}, "
                     f"{a['frac_scored']:.1%} of the region scored")
            if self.verdict_tally:
                # Spelled out because "the verdicts changed nothing" and "no verdict
                # matched anything" read identically on the numbers alone, and the second
                # is a bug that already shipped once.
                tally = dict(self.verdict_tally)
                excluded = tally.pop("excluded_pairs", None)
                counted = {k: v for k, v in tally.items() if v}
                text += ("  [" + ", ".join(f"{v} {k}" for k, v in counted.items()) + "]"
                         if counted else "  [no verdicts recognised]")
                asked = sum(v for k, v in counted.items() if k.endswith("_correct"))
                if excluded == 0 and asked:
                    text += (" -- but no pair in this table matched, so nothing was "
                             "excluded")
        return text


def summarize(c: Contingency, *, at: tuple[float, ...] = (0.5, 0.9)) -> dict:
    """Every scalar metric for one table, as a single flat row."""
    row = voxel.summary(c)
    row.update(assign.summary(c, at=at))
    return row


def compare(a: Any, b: Any, *,
            labels: Sequence[str] = ("gt", "seg"),
            ignore_a: Sequence[int] = (0,),
            ignore_b: Sequence[int] = (),
            mask: Any = None,
            top: int | None = 200,
            min_frac: float = 0.05,
            min_voxels: int = 1,
            rank: str = "severity",
            locate: bool = True,
            verdicts: Mapping[str, str] | None = None,
            check_frames: bool = True,
            check_scattered: bool = True) -> Report:
    """Compare two registered pieces and return everything the comparison produced.

    ``a`` is the reference side and ``b`` the one being assessed — which only affects the
    default ``ignore_a=(0,)``, the names in the output, and the direction of ``voi_split`` /
    ``voi_merge``. The table underneath is symmetric.

    ``locate=False`` skips the per-pair distance transform, which is the only part whose
    cost scales with ``top`` rather than with the table. Worth turning off when scoring
    many crops in a loop and only the numbers are wanted.

    ``verdicts`` maps ``pair_key`` to a verdict from a reviewed disagreement file; supplying
    it adds the adjudicated score *alongside* the raw one, never instead of it.
    """
    if check_frames:
        _check_registered(a, b)

    c = contingency(a.array, b.array, ignore_a=ignore_a, ignore_b=ignore_b, mask=mask,
                    labels=labels)
    warnings: list[str] = []
    if c.n_pairs == 0:
        warnings.append(
            "no labels co-occur: the pieces may not overlap, or everything was ignored "
            f"({c.n_ignored:,} of {c.n_ignored + c.n_scored:,} voxels dropped)")

    rows = disagree.rows(c, min_frac=min_frac, min_voxels=min_voxels, top=top, rank=rank)
    if locate and rows:
        rows = disagree.locate(rows, a, b, labels=labels)

    if check_scattered:
        warnings.extend(_scatter_warnings(a, c, labels))

    report_kwargs: dict = {}
    if verdicts is not None:
        from . import adjudicate

        adjusted = adjudicate.apply(c, verdicts, labels=labels)
        tally = adjudicate.summarize(verdicts, labels=labels)
        # How many pairs the verdicts actually removed, which is NOT the same as how many
        # exclusion verdicts were written: a verdict file outlives the table it came from,
        # so it can be full of valid verdicts that match nothing here. Counting only what
        # was supplied is how an adjudicated score that changed nothing looks identical to
        # one that was silently ignored.
        tally["excluded_pairs"] = c.n_pairs - adjusted.n_pairs
        report_kwargs = {
            "adjudicated": adjusted,
            "adjudicated_summary": summarize(adjusted),
            "verdict_tally": tally,
        }

    return Report(contingency=c, summary=summarize(c), rows=rows,
                  labels=(str(labels[0]), str(labels[1])), warnings=warnings,
                  **report_kwargs)


def scattered_labels(piece: Any, c: Contingency, *,
                     factor: float = SCATTER_FACTOR,
                     min_voxels: int = SCATTER_MIN_VOXELS) -> list[dict]:
    """Reference labels whose bounding box is far larger than the label itself.

    **What this is really looking for is an un-relabelled multi-crop reference volume.**
    Every ground-truth region in this suite numbers its bodies from 1, so the same integer
    names a different cell in each crop — measured on one dataset, 3,637 label-instances
    over 12 regions but only 1,824 distinct ids, with 496 shared. Score that untreated and a
    low-numbered "body" is a chimera of a dozen unrelated cells scattered across the volume;
    every metric comes out meaningless and nothing looks wrong. ``neu-vol relabel`` is the
    fix and it is not optional.

    **It reports rather than refuses, and that is a deliberate weakening of the plan.** The
    condition that actually matters — "this reference was never relabelled" — is not
    decidable from the voxels: a genuinely scattered annotation and a chimera look the same,
    and bodies in these volumes really are fragmented (one had 344 components at scale 1).
    So this is a heuristic with a tunable threshold, and raising on a heuristic inside a
    measurement tool trades a silent wrong number for a loud wrong refusal. The warning
    carries the numbers so the reader can judge.
    """
    from scipy.ndimage import find_objects
    import fastremap

    from neu_proc.ops.backend import to_cpu

    arr = to_cpu(piece.array)
    dense, mapping = fastremap.renumber(np.ascontiguousarray(arr), in_place=False)
    back = {new: original for original, new in mapping.items()}
    boxes = find_objects(dense.astype(np.int64))

    a_ids, a_counts = c.a_totals()
    sizes = {int(i): int(n) for i, n in zip(a_ids, a_counts)}

    flagged = []
    for index, box in enumerate(boxes, start=1):
        if box is None:
            continue
        label = int(back.get(index, index))
        n = sizes.get(label)
        if n is None or n < min_voxels:
            continue
        extent = [int(s.stop - s.start) for s in box]
        bbox_volume = int(np.prod(extent))
        ratio = bbox_volume / n
        if ratio >= factor:
            flagged.append({
                "label_id": label, "n_voxels": n, "bbox_extent": tuple(extent),
                "bbox_volume": bbox_volume, "scatter": float(ratio),
            })
    flagged.sort(key=lambda r: r["scatter"], reverse=True)
    return flagged


def _scatter_warnings(piece: Any, c: Contingency, labels: Sequence[str]) -> list[str]:
    flagged = scattered_labels(piece, c)
    if not flagged:
        return []
    worst = flagged[0]
    return [
        f"{len(flagged)} {labels[0]} label(s) are scattered far beyond their own size — "
        f"worst is id {worst['label_id']}: {worst['n_voxels']:,} voxels spread over a "
        f"{'x'.join(str(v) for v in worst['bbox_extent'])} box "
        f"({worst['scatter']:.0f}x). If this reference is several annotated crops written "
        f"into one volume, the ids collide across crops and a low-numbered body is a "
        f"chimera of unrelated cells — run `neu-vol relabel` first. If the annotation is "
        f"genuinely this diffuse, ignore this."
    ]


def _check_registered(a: Any, b: Any) -> None:
    """Refuse two pieces that do not describe the same physical box."""
    if a.array.shape != b.array.shape:
        raise ValueError(
            f"the two pieces have different shapes, {a.array.shape} vs {b.array.shape}. "
            f"Read the second as `read_piece(src, level=..., crop=first_piece)`, which "
            f"converts the physical box with the target level's own voxel size and origin.")
    a_lo, a_hi = a.bounds_nm
    b_lo, b_hi = b.bounds_nm
    # A voxel of slack: read_piece grows a physical box outward to whole voxels, so two
    # levels of one volume legitimately differ by less than a voxel of the coarser.
    slack = max(max(a.voxel_size_nm), max(b.voxel_size_nm))
    drift = max(abs(np.asarray(a_lo) - np.asarray(b_lo)).max(),
                abs(np.asarray(a_hi) - np.asarray(b_hi)).max())
    if drift > slack:
        raise ValueError(
            f"the two pieces describe different physical boxes: {a.bounds_nm} vs "
            f"{b.bounds_nm}, off by {drift:.1f} nm against a {slack:.1f} nm tolerance. "
            f"Comparing them measures the misregistration, not the segmentation.")
