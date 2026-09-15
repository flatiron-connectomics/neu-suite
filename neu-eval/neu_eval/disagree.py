"""Where the two labelings disagree, ranked worst-first, with a place to look.

A score tells you how much two labelings differ. It does not tell you *whether the
reference was right*, and in this suite that question is live: a manual annotation and an
algorithm can disagree because the algorithm is wrong, and they can disagree because the
annotation is. The only way to tell is to look at the voxels, so this module's job is to
make looking cheap — a short list, ordered by how much each disagreement actually costs,
each row carrying a coordinate you can paste into a viewer.

**Severity is a VOI contribution, not a voxel count** — what the metric itself says a
disagreement costs, straight out of the table that was already built. It has VOI's size
bias, so ``rank="fraction"`` is there for when you want the most-broken bodies rather than
the most metric-moving ones; :func:`rows` documents the measured difference.

**Specks are filtered before anything is classified, and the filter must be asymmetric.**
Bodies in these volumes are *genuinely* fragmented — one measured body had 344 connected
components at scale 1, only 7 of which held ten voxels or more — so an unfiltered "is this
body split?" answers yes for every body. The filter that suggests itself, "the pair is a
real fraction of either side", does not work: a one-voxel stray segment inside a body is
100% *of itself*, so it passes and the body still reads as split twenty-one ways. Whether a
body is split has to be judged on fractions **of that body**, and whether a segment is a
merge on fractions **of that segment**.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

import numpy as np

from .overlap import Contingency
from .voxel import pair_voi

#: What a row's ``kind`` can be. ``tangle`` is both at once — the reference body is spread
#: over several segments *and* those segments each cover several reference bodies — which is
#: worth its own name because it is the case a proofreader cannot fix with one action.
KINDS = ("match", "split", "merge", "tangle")

#: Above this many voxels in a pair's bounding box, :func:`locate` strides the box down
#: before the distance transform rather than running it at full resolution. The point it
#: returns is then approximate — still inside the pair, just not necessarily its deepest
#: voxel — which is the right trade for a click target.
MAX_EDT_VOXELS = 1 << 24


def rows(c: Contingency, *, min_frac: float = 0.05, min_voxels: int = 1,
         top: int | None = None, include_matches: bool = False,
         rank: str = "severity") -> list[dict]:
    """One row per structurally significant pair, worst first.

    **Significance is asymmetric, and it has to be.** Whether a reference body is *split*
    depends on how many segments take a real share **of that body**; whether a segment is a
    *merge* depends on how many bodies contribute a real share **of that segment**. An
    ``or`` over the two fractions looks equivalent and is not: a one-voxel stray segment
    sitting inside a body is 0.1% of the body but **100% of itself**, so it passes on its own
    fraction and every body comes back "split into 21 pieces". Measured on the real case —
    one body, one good segment, twenty single-voxel strays — that is the difference between
    no rows and twenty-one.

    ``min_voxels`` is a floor on the pair regardless of either fraction, for the case where a
    percentage of a small thing is still nothing.

    ``rank`` chooses the order:

    - ``"severity"`` (default) — the pair's contribution to VOI. What the metric itself says
      the disagreement costs, and therefore what to fix first to move the number.
    - ``"fraction"`` — the larger of the two fractions, i.e. how much of a body or a segment
      this one pair accounts for. **Use it when the list matters more than the metric.**
      Severity inherits VOI's size weighting, which is the standard criticism of VOI: a
      600-voxel body cut clean in half scores 0.003 while a 100,000-voxel body losing a 6%
      sliver scores 0.242, so a severity-ranked list is dominated by large bodies and a
      badly-broken small one can sit a hundred rows down. Both numbers are in every row, so
      the choice is only about the order.

    ``include_matches`` keeps the pairs classified ``match`` — one structural partner each
    way, i.e. the two sides agreeing. Off by default because this is a disagreement report.

    Returns plain dicts rather than a DataFrame so this module stays free of pandas; see
    :mod:`neu_eval.tables` for the writers.
    """
    if not 0 <= min_frac <= 1:
        raise ValueError(f"min_frac must be in [0, 1], got {min_frac}")
    if rank not in ("severity", "fraction"):
        raise ValueError(f"rank must be 'severity' or 'fraction', got {rank!r}")
    if c.n_pairs == 0:
        return []

    a_ids, a_counts = c.a_totals()
    b_ids, b_counts = c.b_totals()
    a_total = a_counts[np.searchsorted(a_ids, c.a_ids)].astype(np.float64)
    b_total = b_counts[np.searchsorted(b_ids, c.b_ids)].astype(np.float64)
    counts = c.counts.astype(np.float64)
    frac_a = counts / a_total
    frac_b = counts / b_total

    big_enough = c.counts >= min_voxels
    # Asymmetric, per the docstring: a pair counts toward "is this body split?" only if it
    # takes a real share OF THE BODY, and toward "is this segment a merge?" only if it
    # contributes a real share OF THE SEGMENT.
    counts_for_a = (frac_a >= min_frac) & big_enough
    counts_for_b = (frac_b >= min_frac) & big_enough
    a_partners = Counter(int(v) for v in c.a_ids[counts_for_a])
    b_partners = Counter(int(v) for v in c.b_ids[counts_for_b])

    reportable = (counts_for_a | counts_for_b)
    if not reportable.any():
        return []

    merge_terms, split_terms = pair_voi(c)
    severity = merge_terms + split_terms

    out: list[dict] = []
    for k in np.nonzero(reportable)[0]:
        a_id, b_id = int(c.a_ids[k]), int(c.b_ids[k])
        n_b_of_a, n_a_of_b = a_partners[a_id], b_partners[b_id]
        if n_b_of_a > 1 and n_a_of_b > 1:
            kind = "tangle"
        elif n_b_of_a > 1:
            kind = "split"
        elif n_a_of_b > 1:
            kind = "merge"
        else:
            kind = "match"
        if kind == "match" and not include_matches:
            continue
        out.append({
            "pair_key": f"{a_id}:{b_id}",
            "kind": kind,
            f"{c.labels[0]}_id": a_id,
            f"{c.labels[1]}_id": b_id,
            "n_voxels": int(c.counts[k]),
            f"frac_of_{c.labels[0]}": float(frac_a[k]),
            f"frac_of_{c.labels[1]}": float(frac_b[k]),
            f"n_{c.labels[1]}_partners": n_b_of_a,
            f"n_{c.labels[0]}_partners": n_a_of_b,
            "severity": float(severity[k]),
            "worst_fraction": float(max(frac_a[k], frac_b[k])),
        })

    key = "severity" if rank == "severity" else "worst_fraction"
    # Ties broken by pair_key so the order is a property of the table, not of the sort.
    out.sort(key=lambda r: (-r[key], r["pair_key"]))
    return out[:top] if top is not None else out


def locate(report_rows: Sequence[dict], a: Any, b: Any, *,
           labels: Sequence[str] = ("a", "b"),
           max_edt_voxels: int = MAX_EDT_VOXELS) -> list[dict]:
    """Add a world-coordinate click target to each row.

    ``a`` and ``b`` are the two :class:`~neu_lib.Piece` objects the table was built from.
    The point returned is **the deepest interior voxel of the overlap**, not its centroid:
    a centroid can fall outside a curved or C-shaped region, and one that lands on a
    boundary puts you on the disagreement's edge rather than in it, which is the difference
    between seeing the problem and hunting for it.

    Cost is one pass over the arrays per row, so bound it with ``rows(..., top=N)``. The
    distance transform runs on the pair's bounding box, not the whole array, and dispatches
    to cupyx on a GPU array like everything else here.

    Coordinates come back as ``z_nm``/``y_nm``/``x_nm`` through ``Piece.to_nm``, because the
    voxel index means nothing outside its own frame and a report gets read next to a viewer
    that speaks nanometres. The voxel index is kept too, for indexing back into the arrays.
    """
    from neu_proc.ops.backend import ndimage_for, to_cpu

    if a.array.shape != b.array.shape:
        raise ValueError(f"the two pieces must be the same shape: "
                         f"{a.array.shape} vs {b.array.shape}")
    a_key, b_key = f"{labels[0]}_id", f"{labels[1]}_id"
    aa, ba = a.array, b.array
    out: list[dict] = []
    for row in report_rows:
        mask = (aa == aa.dtype.type(row[a_key])) & (ba == ba.dtype.type(row[b_key]))
        point = _deepest_voxel(mask, max_edt_voxels=max_edt_voxels, ndimage_for=ndimage_for,
                               to_cpu=to_cpu)
        enriched = dict(row)
        if point is None:
            # The pair is in the table, so it has voxels; an empty mask here means the
            # pieces are not the ones the table was built from. Say so rather than emit a
            # coordinate of zeros, which looks like a location.
            raise ValueError(
                f"pair {row['pair_key']} is in the table but not in these arrays — the "
                f"pieces are not the ones the table was built from")
        z, y, x = point
        enriched.update({"z_vox": int(z), "y_vox": int(y), "x_vox": int(x)})
        nm = np.asarray(a.to_nm([[int(z), int(y), int(x)]])).reshape(-1)
        enriched.update({"z_nm": float(nm[0]), "y_nm": float(nm[1]), "x_nm": float(nm[2])})
        out.append(enriched)
    return out


def _deepest_voxel(mask, *, max_edt_voxels, ndimage_for, to_cpu):
    """The voxel of ``mask`` furthest from its boundary, as a ``(z, y, x)`` index."""
    # Per-axis reductions give the bounding box without materialising a coordinate list,
    # which for a body-sized region would be tens of megabytes.
    hits = [to_cpu(mask.any(axis=tuple(j for j in range(3) if j != i))) for i in range(3)]
    spans = []
    for axis_hits in hits:
        where = np.nonzero(axis_hits)[0]
        if where.size == 0:
            return None
        spans.append((int(where[0]), int(where[-1]) + 1))

    box = tuple(slice(lo, hi) for lo, hi in spans)
    cropped = mask[box]
    size = int(np.prod([hi - lo for lo, hi in spans]))

    stride = 1
    while size // (stride ** 3) > max_edt_voxels:
        stride *= 2
    if stride > 1:
        cropped = cropped[::stride, ::stride, ::stride]

    # One voxel of False all round, so the transform measures distance to the pair's own
    # boundary rather than to the crop's edge — without it a region touching the box face
    # reports its deepest point on that face.
    ndi, work = ndimage_for(cropped, "distance_transform_edt")
    dist = ndi.distance_transform_edt(_pad_false(work))
    flat = int(to_cpu(dist.reshape(-1).argmax()))
    local = np.unravel_index(flat, tuple(dist.shape))
    index = [(int(l) - 1) * stride + lo for l, (lo, _) in zip(local, spans)]

    if stride > 1:
        # The strided grid may not land on a voxel of the pair. Snap to one that is, near
        # the answer, so the coordinate is always inside the disagreement.
        index = _snap_into_mask(mask, index, spans, stride, to_cpu=to_cpu)
    return tuple(index)


def _pad_false(arr):
    """``arr`` with a one-voxel False shell, in whichever array library it belongs to."""
    module = type(arr).__module__.split(".")[0]
    if module == "cupy":
        import cupy

        return cupy.pad(arr, 1, mode="constant", constant_values=False)
    return np.pad(arr, 1, mode="constant", constant_values=False)


def _snap_into_mask(mask, index, spans, stride, *, to_cpu):
    """Move ``index`` to the nearest voxel that is actually in ``mask``."""
    if bool(to_cpu(mask[tuple(index)])):
        return index
    window = tuple(
        slice(max(lo, i - stride), min(hi, i + stride + 1))
        for i, (lo, hi) in zip(index, spans))
    local = np.argwhere(to_cpu(mask[window]))
    if local.size == 0:                             # widen once to the whole bounding box
        local = np.argwhere(to_cpu(mask[tuple(slice(lo, hi) for lo, hi in spans)]))
        return [int(v) + lo for v, (lo, _) in zip(local[0], spans)]
    return [int(v) + w.start for v, w in zip(local[0], window)]


def counts_by_kind(report_rows: Sequence[dict]) -> dict[str, int]:
    """How many rows of each kind, for the summary line."""
    tally = Counter(row["kind"] for row in report_rows)
    return {kind: tally.get(kind, 0) for kind in KINDS}
