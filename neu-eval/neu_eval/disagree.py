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
from typing import Any, Iterable, Mapping, Sequence

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
         rank: str = "severity", max_per_group: int | None = None,
         exclude: Iterable[str] = ()) -> list[dict]:
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

    **One failure is many rows, and ``max_per_group`` is what stops it flooding the list.**
    A merge is a single event but the table holds one row per ``(body, segment)`` pair, so
    a segment swallowing twenty bodies yields twenty rows. Measured on real deliveries:
    1,049 merge rows came from 565 distinct segments, 71% of them in multi-row groups, and
    one segment alone contributed 21 of a top-100. Every row carries ``group`` (the
    segment for a merge or tangle, the body for a split), ``group_size`` and
    ``group_rank``, so a reviewer can see that twenty-one rows are one problem.

    ``max_per_group`` then keeps only the worst *n* of each, applied **after** ranking so
    the survivors are each group's most severe. Deliberately not a deduplication: the rows
    are not copies. 17% of within-group neighbours sit within 2 voxels — the two sides of
    one interface — but the median is 33 voxels apart, because a segment usually merges
    bodies at several genuinely distinct places, and collapsing would hide them.

    ``include_matches`` keeps the pairs classified ``match`` — one structural partner each
    way, i.e. the two sides agreeing. Off by default because this is a disagreement report.

    ``exclude`` is a set of ``pair_key`` values to treat as **not structural at all** —
    neither reported nor counted as anyone's partner. It is how :func:`thin_pairs` removes
    boundary disagreements: dropping only the strip's own row would leave the body it pokes
    into still classified "split", because the strip would still count as its second piece.

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
    exclude = set(exclude)
    if exclude:
        keys = np.array([f"{int(x)}:{int(y)}" for x, y in zip(c.a_ids, c.b_ids)])
        big_enough &= ~np.isin(keys, list(exclude))
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

    # A split is one body coming apart, so the body groups it; a merge is one segment
    # swallowing several, so the segment does. A tangle is both at once and is grouped
    # with the merges, which is the half a proofreader acts on first.
    a_key, b_key = f"{c.labels[0]}_id", f"{c.labels[1]}_id"
    for row in out:
        row["group"] = str(row[a_key] if row["kind"] == "split" else row[b_key])
    counts: Counter = Counter(row["group"] for row in out)
    seen: Counter = Counter()
    kept = []
    for row in out:
        g = row["group"]
        seen[g] += 1
        row["group_size"] = counts[g]
        row["group_rank"] = seen[g]
        if max_per_group is None or seen[g] <= max_per_group:
            kept.append(row)
    return kept[:top] if top is not None else kept


def thin_pairs(c: Contingency, a: Any, b: Any, *, radius: float = 3.0,
               min_frac: float = 0.05, min_voxels: int = 1,
               ignore_a: Iterable[int] = (0,), ignore_b: Iterable[int] = ()) -> dict[str, float]:
    """``{pair_key: interior radius}`` for the significant pairs with NO real interior.

    Two labelings draw the same cell boundary a voxel or two apart, so each body carries a
    thin strip of its neighbour's segment. By fraction alone such a strip can be
    "significant" -- a few percent of a small body -- and it then reads as a merge (the
    neighbour's segment reaching into this body) or a split. What it does not have is an
    interior: its **largest inscribed radius** (the peak of the distance transform of the
    pair's overlap) is a voxel or two. Measured on one delivery's merges, 63% of rows were
    at most 3 voxels deep, against none of its splits. A pair is thin when that radius is
    at most ``radius`` voxels.

    Only pairs that would pass :func:`rows`'s significance test are measured -- the rest
    cannot become rows -- each inside the intersection of its two labels' bounding boxes.
    Pass the keys to ``rows(..., exclude=...)``.
    """
    from scipy import ndimage as ndi

    from neu_proc.ops.backend import to_cpu

    if c.n_pairs == 0:
        return {}
    aa, ba = to_cpu(a.array), to_cpu(b.array)
    a_ids, a_counts = c.a_totals()
    b_ids, b_counts = c.b_totals()
    fa = c.counts / a_counts[np.searchsorted(a_ids, c.a_ids)]
    fb = c.counts / b_counts[np.searchsorted(b_ids, c.b_ids)]
    candidate = ((fa >= min_frac) | (fb >= min_frac)) & (c.counts >= min_voxels)
    drop_a = {int(v) for v in ignore_a}
    drop_b = {int(v) for v in ignore_b}
    boxes_a = _label_boxes(aa, to_cpu=to_cpu)
    boxes_b = _label_boxes(ba, to_cpu=to_cpu)
    out = {}
    for k in np.nonzero(candidate)[0]:
        x, y = int(c.a_ids[k]), int(c.b_ids[k])
        if x in drop_a or y in drop_b or x not in boxes_a or y not in boxes_b:
            continue
        box = tuple(slice(max(p.start, q.start), min(p.stop, q.stop))
                    for p, q in zip(boxes_a[x], boxes_b[y]))
        if any(s.stop <= s.start for s in box):
            continue
        m = (aa[box] == aa.dtype.type(x)) & (ba[box] == ba.dtype.type(y))
        r = float(ndi.distance_transform_edt(np.pad(m, 1)).max()) if m.any() else 0.0
        if r <= radius:
            out[f"{x}:{y}"] = r
    return out


def locate(report_rows: Sequence[dict], a: Any, b: Any, *,
           labels: Sequence[str] = ("a", "b"),
           at: str = "seam",
           ignore_a: Iterable[int] = (0,), ignore_b: Iterable[int] = (),
           crop: bool = True,
           max_edt_voxels: int = MAX_EDT_VOXELS,
           shapes: bool = False, anchor_radius: int = 24,
           gap: int = 2) -> list[dict]:
    """Add a world-coordinate click target to each row.

    ``a`` and ``b`` are the two :class:`~neu_lib.Piece` objects the table was built from.

    **``at="seam"`` (default) points at the error; ``at="overlap"`` points at the
    agreement.** This function shipped doing the latter and it was the wrong choice: the
    deepest interior voxel of ``a ∩ b`` is the middle of the part the two labelings agree
    about, which is the least informative voxel in the pair. What you want to look at is
    the surface where they part company:

    - a **split** row is a segment that stops while the reference body continues, so its
      seam is the boundary of ``a ∩ b`` *inside* ``a`` — the false cut;
    - a **merge** row is a segment that continues while the reference body stops, so its
      seam is the boundary of ``a ∩ b`` *inside* ``b`` — the false join;
    - a **tangle** is both, so both are computed and the larger contact wins;
    - a **match** has no seam and falls back to the overlap interior.

    Of the seam's voxels it returns the one **deepest inside the containing region** (the
    body for a split, the segment for a merge), so the point sits in the middle of the
    false cut rather than where that cut grazes the object's own surface.

    **``ignore_a`` / ``ignore_b`` must match the table's, and on real data this decides
    whether the answer is useful at all.** The seam is a contact with *another labelled
    thing*, so ignored values cannot count as the other side. Where a pipeline marks
    membranes 0 and that 0 is ignored, counting it would put every "split point" on an
    ordinary cell boundary — the seam would be the body's own surface, and every row would
    point somewhere correct and uninteresting.

    A pair whose pieces do not touch at all — a genuinely disconnected fragment — has no
    seam; those fall back to the overlap interior and say so in ``point_at``, because a
    caller comparing points across rows should know which question each answers.

    Cost is a few passes over the arrays per row, so bound it with ``rows(..., top=N)``.
    The distance transforms run on the containing region's bounding box, not the whole
    array, and dispatch to cupyx on a GPU array like everything else here.

    Coordinates come back as ``z_nm``/``y_nm``/``x_nm`` through ``Piece.to_nm``, because the
    voxel index means nothing outside its own frame and a report gets read next to a viewer
    that speaks nanometres. The voxel index is kept too, for indexing back into the arrays.

    **``shapes=True`` adds a drawable description of the error** under ``row["shapes"]`` --
    a point says *where* but not *what*, and on real data a single seam voxel can sit on an
    incidental contact rather than the false boundary itself. See :func:`_shapes`; the
    point above is computed exactly as before either way.
    """
    from neu_proc.ops.backend import ndimage_for, to_cpu

    if at not in ("seam", "overlap"):
        raise ValueError(f"at must be 'seam' or 'overlap', got {at!r}")
    if a.array.shape != b.array.shape:
        raise ValueError(f"the two pieces must be the same shape: "
                         f"{a.array.shape} vs {b.array.shape}")

    a_key, b_key = f"{labels[0]}_id", f"{labels[1]}_id"
    aa, ba = a.array, b.array
    drop_a = [aa.dtype.type(v) for v in ignore_a]
    drop_b = [ba.dtype.type(v) for v in ignore_b]
    out: list[dict] = []

    # Per-label bounding boxes, computed ONCE for each side, so a row's work is bounded by
    # the pair rather than by the volume. Every mask this function builds -- the body, the
    # segment, their intersection, the other side, the dilation -- used to be a full-array
    # pass, and a row needs none of the array beyond the two labels it names. Measured on a
    # 15 Mvoxel crop with 100 rows: the boxes cost 0.4 s once, against ~1 s *per row*
    # before, and the pair occupies a median 2.6% of the volume.
    boxes_a = _label_boxes(aa, to_cpu=to_cpu) if crop else {}
    boxes_b = _label_boxes(ba, to_cpu=to_cpu) if crop else {}
    # Whole-array label sizes, for telling a piece OF a body from a strip of a neighbour
    # (see _shapes). Counted once; a window holds only part of most neighbours.
    sizes_a = _label_sizes(aa, to_cpu=to_cpu) if shapes else {}
    sizes_b = _label_sizes(ba, to_cpu=to_cpu) if shapes else {}

    for row in report_rows:
        a_id, b_id = aa.dtype.type(row[a_key]), ba.dtype.type(row[b_key])
        # The window must contain BOTH containers whole, not just their intersection: a
        # split's depth is measured inside the body and a merge's inside the segment, so
        # clipping either would report a point as shallow that is actually deep. One voxel
        # of margin so the dilation that finds the other side is correct at the edge.
        window = _window(boxes_a.get(int(row[a_key])), boxes_b.get(int(row[b_key])),
                         aa.shape)
        sub = tuple(slice(lo, hi) for lo, hi in window) if window else ...
        origin = [lo for lo, _ in window] if window else [0, 0, 0]
        caa, cba = (aa[sub], ba[sub]) if window else (aa, ba)

        body, segment = caa == a_id, cba == b_id
        inter = body & segment

        point, how = None, "overlap"
        if at == "seam":
            point, how = _seam_voxel(
                row.get("kind", "tangle"), inter, body, segment, caa, cba,
                a_id, b_id, drop_a, drop_b,
                max_edt_voxels=max_edt_voxels, ndimage_for=ndimage_for, to_cpu=to_cpu)
        if point is None:
            point = _deepest_voxel(inter, max_edt_voxels=max_edt_voxels,
                                   ndimage_for=ndimage_for, to_cpu=to_cpu)
            how = "overlap"
        if point is not None:
            point = tuple(int(v) + o for v, o in zip(point, origin))
        if point is None:
            # The pair is in the table, so it has voxels; an empty mask here means the
            # pieces are not the ones the table was built from. Say so rather than emit a
            # coordinate of zeros, which looks like a location.
            raise ValueError(
                f"pair {row['pair_key']} is in the table but not in these arrays — the "
                f"pieces are not the ones the table was built from")

        z, y, x = point
        enriched = dict(row)
        enriched.update({"z_vox": int(z), "y_vox": int(y), "x_vox": int(x),
                         "point_at": how})
        nm = np.asarray(a.to_nm([[int(z), int(y), int(x)]])).reshape(-1)
        enriched.update({"z_nm": float(nm[0]), "y_nm": float(nm[1]), "x_nm": float(nm[2])})
        if shapes:
            enriched["shapes"] = _shapes(
                row.get("kind", "tangle"), to_cpu(caa), to_cpu(cba), a_id, b_id,
                drop_a, drop_b, origin, a, radius=anchor_radius, gap=gap,
                sizes_a=sizes_a, sizes_b=sizes_b)
        out.append(enriched)
    return out


#: The sides a row's kind has. A tangle is both errors at once, so it gets both shapes.
_SIDES = {"split": ("split",), "merge": ("merge",), "tangle": ("split", "merge"),
          "match": ()}


def _label_sizes(arr, *, to_cpu):
    ids, n = np.unique(to_cpu(arr), return_counts=True)
    return dict(zip(ids.tolist(), n.tolist()))


#: A piece counts as part of the container when at least this share of it lies inside.
#: Between the two measured cases it must separate: boundary strips of a neighbouring cell
#: sit at 0.1-1% inside (specimen 5), while a tangle's partner -- itself a merger spanning
#: bodies -- sits at 40-60%.
FRAGMENT_SHARE = 0.25


def _shapes(kind, caa, cba, a_id, b_id, drop_a, drop_b, origin, piece, *, radius, gap,
            sizes_a=None, sizes_b=None):
    """One drawable shape per side of the error: ``[{side, centre_nm, radii_nm, p0, p1, ...}]``.

    For a **split** (``a``'s body cut between this ``b`` segment and another) and a **merge**
    (``b``'s segment spanning this ``a`` body and another) alike:

    - ``other_id`` is the specific piece across the false boundary, in the other side's
      labelling. **It must be a FRAGMENT of the container** -- at least
      :data:`FRAGMENT_SHARE` of it inside -- and the largest such by overlap. Not the piece
      with the most contact: two labelings draw one cell boundary a voxel or two apart, so a
      strip of the NEIGHBOURING cell's segment lies inside the body along its whole
      boundary, and on real data that strip out-touched the true cut and put the ellipsoid
      on an ordinary cell boundary. Where no piece qualifies the side is not drawn (the row
      falls back to a point) rather than drawn on a boundary. ``other_how`` records it;
    - the **patch** is the largest connected part of the contact with it, plus its mirror
      on the far side, so ``centre`` / ``radii`` describe the false cut or join itself.
      Largest, because the contact is often several patches and the deepest single voxel
      (what the point uses) can belong to a stray one;
    - ``p0`` is inside this pair's overlap and ``p1`` inside the other piece, each the voxel
      nearest the centroid of its region within ``radius`` voxels of the patch centre, so
      both are guaranteed to lie IN their region. For a merge they are a line across the
      join; for a split, one point in each of the pieces that should be one.

    **``gap`` bridges ignored voxels.** A delivery that draws membranes as an ignored 0
    separates its two pieces of a split body by that membrane, so they never touch and a
    contact-only rule finds no cut -- or finds a stray place where they happen to touch
    and draws the error there. The other side may therefore be reached through up to
    ``gap`` ignored voxels.

    Radii are 1.5 standard deviations of the patch per axis, at least 2 voxels. Viewer
    ellipsoids are axis-aligned, so an oblique cut is drawn as the ellipsoid bounding it.
    """
    from scipy import ndimage as ndi

    body, segment = caa == a_id, cba == b_id
    inter = body & segment
    voxel = np.asarray(piece.frame.voxel_size_nm, dtype=float)
    full = np.ones((3, 3, 3), bool)
    out = []
    for side in _SIDES.get(kind, ()):
        if side == "split":
            container, other_arr, own_id, drop = body, cba, b_id, drop_b
            totals = sizes_b or {}
        else:
            container, other_arr, own_id, drop = segment, caa, a_id, drop_a
            totals = sizes_a or {}
        ignored = (np.isin(other_arr, np.asarray(drop, dtype=other_arr.dtype))
                   if drop else np.zeros(other_arr.shape, bool))
        others = container & (other_arr != own_id) & ~ignored
        if not others.any():
            continue

        def reach_of(mask):
            grown = mask.copy()
            for _ in range(gap):
                grown |= ndi.binary_dilation(grown) & ignored & container
            return ndi.binary_dilation(grown)

        values, inside = np.unique(other_arr[others], return_counts=True)
        share = np.array([n / max(totals.get(int(v), n), 1) for v, n in zip(values, inside)])
        # Qualifying pieces by overlap, largest first; the first that actually TOUCHES this
        # pair wins. A fragment elsewhere in the body cannot mark this row's cut.
        order = [i for i in np.argsort(-inside) if share[i] >= FRAGMENT_SHARE]
        pick = contact = one = None
        for i in order:
            cand = container & (other_arr == values[i])
            hit = inter & reach_of(cand)
            if hit.any():
                pick, one, contact = i, cand, hit
                break
        if pick is None:
            continue
        other_id = values[pick]
        labelled, n = ndi.label(contact, structure=full)
        if n == 0:
            continue
        sizes = np.bincount(labelled.ravel())
        sizes[0] = 0
        patch = labelled == int(sizes.argmax())
        mirror = one & ndi.binary_dilation(patch, iterations=gap + 1)
        where = np.argwhere(patch | mirror).astype(float)
        centre = where.mean(axis=0)
        radii = np.maximum(1.5 * where.std(axis=0), 2.0)
        p0, p1 = _anchor(inter, centre, radius), _anchor(one, centre, radius)

        shift = np.asarray(origin, dtype=float)
        to_nm = lambda v: [float(x) for x in                                # noqa: E731
                           np.asarray(piece.to_nm([list(v)])).reshape(-1)]
        c_abs = centre + shift
        v0 = [int(v) for v in p0 + shift]
        v1 = [int(v) for v in p1 + shift]
        out.append({
            "side": side, "other_id": str(int(other_id)),
            "other_share": float(share[pick]),
            "n_voxels": int(len(where)),
            "centre_nm": to_nm(c_abs), "radii_nm": [float(r) for r in radii * voxel],
            "p0_vox": v0, "p1_vox": v1, "p0_nm": to_nm(v0), "p1_nm": to_nm(v1),
        })
    return out


def _anchor(mask, centre, radius):
    """The voxel of ``mask`` nearest the centroid of ``mask`` within ``radius`` of ``centre``.

    Nearest-to-centroid rather than the centroid itself, because a curved region's centroid
    can fall outside it and a line end in the wrong object names the wrong segment.
    """
    lo = np.maximum(np.floor(centre - radius).astype(int), 0)
    hi = np.minimum(np.ceil(centre + radius).astype(int) + 1, mask.shape)
    sub = mask[tuple(slice(a, b) for a, b in zip(lo, hi))]
    pts = np.argwhere(sub) + lo
    if len(pts):
        inside = pts[((pts - centre) ** 2).sum(axis=1) <= radius ** 2]
        pts = inside if len(inside) else pts
    else:
        pts = np.argwhere(mask)
    c = pts.mean(axis=0)
    return pts[((pts - c) ** 2).sum(axis=1).argmin()]


def sample_ids(report_rows: Sequence[dict], pieces: Mapping[str, Any], *,
               ignore: Mapping[str, Iterable[int]] | None = None) -> list[dict]:
    """Add ``shape["ids"] = {name: [ids at p0 and p1]}`` for every labelling in ``pieces``.

    This is what lets an annotation name, in EVERY layer, the objects at its two ends: the
    reference bodies, the segment that merged them, and whatever a third delivery has there
    -- two segments if it got it right, one if it made the same merge. The rule needs no
    knowledge of which layer is "the" answer, which is the point.

    ``pieces`` must be on the grid ``locate`` was given (``p0_vox`` / ``p1_vox`` index
    it), and should hold the labels **as delivered** -- the ids a viewer shows -- even
    when the table was built from connected-components ids. Ignored values (membrane) are
    left out: they are not objects anyone can select.
    """
    ignore = {k: set(int(v) for v in vals) for k, vals in (ignore or {}).items()}
    shape = None
    for p in pieces.values():
        if shape is not None and p.array.shape != shape:
            raise ValueError("every piece must be on one grid: "
                             f"{shape} vs {p.array.shape}")
        shape = p.array.shape
    out = []
    for row in report_rows:
        new = dict(row)
        if row.get("shapes"):
            new["shapes"] = []
            for s in row["shapes"]:
                ids = {}
                for name, p in pieces.items():
                    got = {int(p.array[tuple(s[k])]) for k in ("p0_vox", "p1_vox")}
                    ids[name] = [str(v) for v in sorted(got - ignore.get(name, set()))]
                new["shapes"].append({**s, "ids": ids})
        out.append(new)
    return out


def _label_boxes(arr, *, to_cpu):
    """``{label: (slice, slice, slice)}`` for every label, in one pass over the array.

    Densely renumbered first because ``find_objects`` indexes its result by label value,
    so sparse uint64 body ids would ask it for an array of 10^18 entries.
    """
    import fastremap
    from scipy.ndimage import find_objects

    host = to_cpu(arr)
    dense, mapping = fastremap.renumber(np.ascontiguousarray(host), in_place=False)
    back = {new: old for old, new in mapping.items()}
    found = find_objects(dense.astype(np.int64))
    return {int(back.get(i, i)): sl
            for i, sl in enumerate(found, start=1) if sl is not None}


def _window(box_a, box_b, shape, margin=1):
    """The union of two per-label boxes, grown by ``margin`` and clipped to ``shape``.

    ``None`` when either box is unknown, which means fall back to the whole array rather
    than guess a window that might not contain the label.
    """
    if box_a is None or box_b is None:
        return None
    out = []
    for sa, sb, n in zip(box_a, box_b, shape):
        lo = max(0, min(sa.start, sb.start) - margin)
        hi = min(n, max(sa.stop, sb.stop) + margin)
        if hi <= lo:
            return None
        out.append((lo, hi))
    return out


def _other_side(arr, own_id, drop):
    """``arr`` is some *other* labelled thing: not this id, and not an ignored value."""
    other = arr != own_id
    for value in drop:
        other &= arr != value
    return other


def _seam_voxel(kind, inter, body, segment, aa, ba, a_id, b_id, drop_a, drop_b, *,
                max_edt_voxels, ndimage_for, to_cpu):
    """``((z, y, x), how)`` on the surface where the two labelings part company.

    ``how`` is ``"seam-split"`` or ``"seam-merge"``, naming which side's boundary the point
    sits on — a tangle has both and the wider contact wins, so the row has to say which one
    it answered.
    """
    candidates = []
    if kind in ("split", "tangle", "match"):
        # The segment stops, the body carries on. The other side must be another segment
        # **inside this body** — `& body` is load-bearing. Without it a fragment separated
        # by a gap in the body reports a seam against something outside the body entirely,
        # which is not a false cut and not where anyone should be sent to look.
        candidates.append(("seam-split", body, body & _other_side(ba, b_id, drop_b)))
    if kind in ("merge", "tangle", "match"):
        # Mirror image: another body inside this segment.
        candidates.append(("seam-merge", segment,
                           segment & _other_side(aa, a_id, drop_a)))

    best = None
    for how, container, other in candidates:
        contact = inter & _grow(other, ndimage_for=ndimage_for)
        n = int(to_cpu(contact.sum()))
        if n == 0:
            continue
        if best is None or n > best[0]:
            best = (n, how, container, contact)
    if best is None:
        return None, "overlap"

    _, how, container, contact = best
    point = _deepest_in(container, contact, max_edt_voxels=max_edt_voxels,
                        ndimage_for=ndimage_for, to_cpu=to_cpu)
    return point, how


def _grow(mask, *, ndimage_for):
    """``mask`` dilated by one voxel, 6-connected — scipy's default 3D structure."""
    ndi, work = ndimage_for(mask, "binary_dilation")
    return ndi.binary_dilation(work)


def _deepest_in(container, contact, *, max_edt_voxels, ndimage_for, to_cpu):
    """The ``contact`` voxel furthest from the outside of ``container``.

    Scored by the container's own distance transform, so the point lands mid-seam rather
    than where the seam meets the object's surface — which is where a naive pick on a thin
    contact sheet would land, since a one-voxel-thick surface has no interior of its own.
    """
    spans = _bbox(container, to_cpu=to_cpu)
    if spans is None:
        return None
    box = tuple(slice(lo, hi) for lo, hi in spans)
    size = int(np.prod([hi - lo for lo, hi in spans]))

    stride = 1
    while size // (stride ** 3) > max_edt_voxels:
        stride *= 2
    sub = tuple(slice(lo, hi, stride) for lo, hi in spans)

    ndi, held = ndimage_for(container[sub], "distance_transform_edt")
    depth = ndi.distance_transform_edt(_pad_false(held))[1:-1, 1:-1, 1:-1]
    here = contact[sub]
    scored = to_cpu(depth * here)
    if not scored.any():
        return None
    local = np.unravel_index(int(scored.argmax()), scored.shape)
    return tuple(int(v) * stride + lo for v, (lo, _) in zip(local, spans))


def _bbox(mask, *, to_cpu):
    """Per-axis ``(lo, hi)`` of ``mask``, or None when it is empty."""
    spans = []
    for axis in range(3):
        hits = to_cpu(mask.any(axis=tuple(j for j in range(3) if j != axis)))
        where = np.nonzero(hits)[0]
        if where.size == 0:
            return None
        spans.append((int(where[0]), int(where[-1]) + 1))
    return spans


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
