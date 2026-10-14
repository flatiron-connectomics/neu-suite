"""Where the two labelings disagree, ranked worst-first, with a place to look.

A score cannot say whether the reference or the segmentation was wrong; only looking at
the voxels can. This module makes looking cheap: a short list of disagreements, ordered by
what each costs, each with a coordinate to paste into a viewer.

Severity is the pair's VOI contribution; ``rank="fraction"`` orders by how broken a body
or segment is instead (see :func:`rows`).

**Specks are filtered asymmetrically.** Bodies here are genuinely fragmented into many
tiny pieces, and a one-voxel stray segment is 100% of itself, so a filter on "a real
fraction of either side" would still call every body split. Whether a body is split is
judged on fractions of that body, and whether a segment is a merge on fractions of that
segment.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .overlap import Contingency
from .voxel import pair_voi

#: What a row's ``kind`` can be. ``tangle`` is split and merge at once (the body is spread
#: over several segments, which each cover several bodies), not fixable in one action.
KINDS = ("match", "split", "merge", "tangle")

#: Above this many voxels in a region's bounding box, :func:`locate` strides it before the
#: distance transform; the point is then still inside the pair but approximate.
MAX_EDT_VOXELS = 1 << 24


def rows(c: Contingency, *, min_frac: float = 0.05, min_voxels: int = 1,
         top: int | None = None, include_matches: bool = False,
         rank: str = "severity", max_per_group: int | None = None,
         exclude: Iterable[str] = ()) -> list[dict]:
    """One row per structurally significant pair, worst first.

    A pair counts toward "this body is split" only if it holds at least ``min_frac`` of
    the body, and toward "this segment is a merge" only if it holds ``min_frac`` of the
    segment (see the module docstring). ``min_voxels`` is a floor on the pair's size.

    ``rank`` sets the order:

    - ``"severity"`` (default): the pair's VOI contribution, what to fix first to move the
      number. It inherits VOI's size weighting: a 600-voxel body cut in half scores 0.003,
      a 100,000-voxel body losing a 6% sliver 0.242.
    - ``"fraction"``: the larger of the two fractions, so a badly broken small body ranks
      high. Both values are in every row.

    One error can be many rows: a segment merging twenty bodies gives twenty pairs. Each row
    has ``group`` (the segment for a merge or tangle, the body for a split), ``group_size``
    and ``group_rank``; ``max_per_group`` keeps each group's worst *n*, after ranking. The
    rows are not duplicates, since a segment often merges bodies at several places.

    ``include_matches`` keeps pairs where the two sides agree. ``exclude`` is a set of
    ``pair_key`` values treated as not there at all, neither reported nor counted as a
    partner, as :func:`thin_pairs` needs. Returns plain dicts; see :mod:`neu_eval.tables`.
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
    # Asymmetric: a share of the body for splits, a share of the segment for merges.
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
    # Ties broken by pair_key, so the order is deterministic.
    out.sort(key=lambda r: (-r[key], r["pair_key"]))

    # A split is grouped by its body, a merge by its segment; a tangle with the merges.
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

    Two labelings draw a cell boundary a voxel or two apart, so each body carries a thin
    strip of its neighbour's segment, which can be a large enough fraction to read as a
    merge or split. Such a strip has no interior: its largest inscribed radius is a voxel
    or two. A pair is thin when that radius is at most ``radius`` voxels.

    Only pairs that could become :func:`rows` are measured. Pass the keys to
    ``rows(..., exclude=...)``.
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

    ``at="seam"`` (default) points at the error, the surface where the labelings part:

    - **split**: the boundary of ``a ∩ b`` inside the body, the false cut;
    - **merge**: the boundary of ``a ∩ b`` inside the segment, the false join;
    - **tangle**: both, and the larger contact wins;
    - **match**: no seam; falls back to the overlap.

    The point is the seam voxel deepest inside the body (split) or segment (merge), so it
    sits mid-cut. ``at="overlap"`` instead gives the deepest voxel of ``a ∩ b``, the middle
    of what the two agree on. A pair with no seam (a disconnected fragment) falls back to
    the overlap, and ``point_at`` records which was used.

    **``ignore_a`` / ``ignore_b`` must match the table's.** The seam is contact with
    another labelled thing, so if membrane is 0 and ignored but counted here, every split
    point lands on an ordinary cell boundary.

    Cost is a few passes per row over the two labels' bounding box, so bound it with
    ``rows(..., top=N)``. Adds ``z_nm``/``y_nm``/``x_nm`` (via ``Piece.to_nm``) and the voxel
    index. ``shapes=True`` also adds a drawable description of the error under
    ``row["shapes"]`` (see :func:`_shapes`), since a single seam voxel can sit on an
    incidental contact.
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

    # Per-label bounding boxes, computed once per side, so each row's work is bounded by its
    # two labels rather than by the volume.
    boxes_a = _label_boxes(aa, to_cpu=to_cpu) if crop else {}
    boxes_b = _label_boxes(ba, to_cpu=to_cpu) if crop else {}
    # Whole-array label sizes, for telling a piece of a body from a neighbour's strip.
    sizes_a = _label_sizes(aa, to_cpu=to_cpu) if shapes else {}
    sizes_b = _label_sizes(ba, to_cpu=to_cpu) if shapes else {}

    for row in report_rows:
        a_id, b_id = aa.dtype.type(row[a_key]), ba.dtype.type(row[b_key])
        # Both labels whole (depth is measured inside each), plus one voxel of margin for
        # the dilation that finds the other side.
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
            # The pair is in the table, so these must be the wrong arrays.
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
#: Measured, a neighbouring cell's boundary strip has 0.1-1% inside; a tangle's partner
#: (itself a merger) 40-60%.
FRAGMENT_SHARE = 0.25


def _shapes(kind, caa, cba, a_id, b_id, drop_a, drop_b, origin, piece, *, radius, gap,
            sizes_a=None, sizes_b=None):
    """One drawable shape per side of the error: ``[{side, centre_nm, radii_nm, p0, p1, ...}]``.

    For a **split** (``a``'s body cut between this ``b`` segment and another) and a **merge**
    (``b``'s segment spanning this ``a`` body and another) alike:

    - ``other_id`` is the piece across the false boundary, in the other labelling: the
      largest by overlap that is a fragment of the container (at least
      :data:`FRAGMENT_SHARE` inside) and touches this pair. Picking by most contact instead
      would choose a neighbouring cell's boundary strip. If none qualifies, the side is not
      drawn and the row falls back to a point.
    - The **patch** is the largest connected part of the contact, plus its mirror on the
      far side; ``centre`` and ``radii`` describe it (radii are 1.5 standard deviations per
      axis, at least 2 voxels; viewer ellipsoids are axis-aligned).
    - ``p0`` lies in this pair's overlap and ``p1`` in the other piece, each near the patch
      centre: a line across a merge, or one point in each half of a split.

    ``gap`` lets the other side be reached through up to that many ignored voxels, because
    a segmentation that draws membrane as an ignored 0 separates the two halves of a split
    body, which then never touch.
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
        # Qualifying pieces, largest overlap first; the first that touches this pair wins.
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
    """The voxel of ``mask`` nearest the centroid of ``mask`` within ``radius`` of ``centre``
    (not the centroid itself, which can fall outside a curved region)."""
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

    So an annotation names the objects at its two ends in every layer: the reference
    bodies, the merging segment, and whatever another segmentation has there.

    ``pieces`` must be on the grid ``locate`` was given (``p0_vox`` / ``p1_vox`` index it)
    and should hold the ids a viewer shows, even if the table was built from
    connected-component ids. Ignored values are left out.
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

    Renumbered densely first, because ``find_objects`` indexes by label value.
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

    ``None`` when either box is unknown, meaning use the whole array.
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

    ``how`` is ``"seam-split"`` or ``"seam-merge"``; for a tangle, the wider contact wins.
    """
    candidates = []
    if kind in ("split", "tangle", "match"):
        # The segment stops, the body continues. The other side must be another segment
        # inside this body, or a fragment would report a seam outside the body.
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
    """``mask`` dilated by one voxel, 6-connected."""
    ndi, work = ndimage_for(mask, "binary_dilation")
    return ndi.binary_dilation(work)


def _deepest_in(container, contact, *, max_edt_voxels, ndimage_for, to_cpu):
    """The ``contact`` voxel furthest from the outside of ``container``.

    Scored by the container's distance transform, so the point lands mid-seam rather than
    where the seam meets the object's surface.
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
    # Bounding box by per-axis reduction, without listing coordinates.
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

    # Pad with False, so a region touching the crop face is not measured to that face.
    ndi, work = ndimage_for(cropped, "distance_transform_edt")
    dist = ndi.distance_transform_edt(_pad_false(work))
    flat = int(to_cpu(dist.reshape(-1).argmax()))
    local = np.unravel_index(flat, tuple(dist.shape))
    index = [(int(l) - 1) * stride + lo for l, (lo, _) in zip(local, spans)]

    if stride > 1:
        # A strided point may miss the pair; snap to a nearby voxel that is in it.
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
