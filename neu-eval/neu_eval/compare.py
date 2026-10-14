"""One call from two registered pieces to everything a comparison produces.

``compare(a, b)`` builds the table once, computes every metric from it, ranks and locates
the disagreements, and, given a reviewed verdict file, adds an adjudicated score. A
:class:`Report` holds it all, raw and adjudicated side by side.

The two pieces must describe the same physical box. Read the second with
``read_piece(seg, level=..., crop=gt_piece)``, which converts the box using that level's
own voxel size and origin; matching shapes by hand or assuming a ``2 ** level`` factor can
shift it by part of a voxel. :func:`compare` checks the frames agree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from . import assign, disagree, voxel
from .overlap import Contingency, contingency

#: Flag a reference label whose bounding box holds this many times more voxels than the
#: label itself (see :func:`scattered_labels`).
SCATTER_FACTOR = 1000.0

#: Never flag labels smaller than this; a tiny label's scatter ratio means nothing.
SCATTER_MIN_VOXELS = 100


@dataclass(frozen=True)
class Report:
    """Everything one comparison produced.

    ``summary`` and ``adjudicated_summary`` are flat dicts, so many comparisons make a
    DataFrame directly. The ``adjudicated_*`` fields are ``None`` when no verdicts were
    given; verdicts that changed nothing give a summary equal to the raw one, and
    ``verdict_tally`` shows why.
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
                # "Changed nothing" and "matched nothing" look identical in the numbers.
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
            point_at: str = "seam",
            verdicts: Mapping[str, str] | None = None,
            check_frames: bool = True,
            check_scattered: bool = True) -> Report:
    """Compare two registered pieces and return everything the comparison produced.

    ``a`` is the reference and ``b`` the side assessed; that sets only the default
    ``ignore_a=(0,)``, the output names, and the direction of ``voi_split`` / ``voi_merge``.

    ``locate=False`` skips the per-pair distance transforms, the one cost that scales with
    ``top``; useful when only the numbers are wanted. ``point_at`` is passed to
    :func:`neu_eval.disagree.locate`. ``verdicts`` (``pair_key`` -> verdict, from a reviewed
    disagreement file) adds an adjudicated score alongside the raw one.
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
        # Pass the ignore sets on: a seam is contact with another labelled thing, and
        # counting ignored membrane (0) would put split points on every cell boundary.
        rows = disagree.locate(rows, a, b, labels=labels, at=point_at,
                               ignore_a=ignore_a, ignore_b=ignore_b)

    if check_scattered:
        warnings.extend(_scatter_warnings(a, c, labels))

    report_kwargs: dict = {}
    if verdicts is not None:
        from . import adjudicate

        adjusted = adjudicate.apply(c, verdicts, labels=labels)
        tally = adjudicate.summarize(verdicts, labels=labels)
        # Pairs actually removed, which can be fewer than the exclusion verdicts written:
        # a verdict file can outlive its table and match nothing here.
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

    This mainly catches a reference assembled from several annotated crops that each
    number their bodies from 1: without ``neu-vol relabel``, one id names a different cell
    in each crop, a low-numbered "body" is a chimera of unrelated cells, and every metric is
    meaningless with no error.

    It warns rather than raises, because a genuinely fragmented annotation looks the same
    from the voxels; the warning carries the numbers so the reader can judge.
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
    # One coarse voxel of slack: read_piece rounds a physical box outward to whole voxels.
    slack = max(max(a.voxel_size_nm), max(b.voxel_size_nm))
    drift = max(abs(np.asarray(a_lo) - np.asarray(b_lo)).max(),
                abs(np.asarray(a_hi) - np.asarray(b_hi)).max())
    if drift > slack:
        raise ValueError(
            f"the two pieces describe different physical boxes: {a.bounds_nm} vs "
            f"{b.bounds_nm}, off by {drift:.1f} nm against a {slack:.1f} nm tolerance. "
            f"Comparing them measures the misregistration, not the segmentation.")
