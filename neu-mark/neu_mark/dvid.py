"""Fetching annotations out of DVID, for a given set of bodies.

Addressing and version resolution come from ``neu_vol.dvid``: a
``dvid://server/uuid/instance`` URL, resolved once to a concrete node that the provenance
record names.

Point annotations are read per body from ``/label/<body>``, which exists only when the
annotation instance is synced to a labelmap; that is checked up front. neuclease's
``fetch_elements_for_bodies`` is not used, because it cannot return relationships. Bodies
are fetched on a thread pool (the work is I/O bound), each with retry, and a body that
still fails is recorded rather than raised. The default of 8 threads is a courtesy to a
shared server, which answers overload with 503.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping, Sequence

# Called through the module, so a test that patches `neu_vol.dvid` also patches the calls
# made here.
from neu_vol import dvid as _vdvid
from neu_vol.dvid import MISSING
from neu_vol.retry import with_retry

from . import tables

logger = logging.getLogger(__name__)

#: Default pool size, kept modest because DVID is usually a shared server.
DEFAULT_THREADS = 8

#: DVID instance types this module can read, by what they hold.
POINT_INSTANCE = "annotation"
BODY_INSTANCE = "keyvalue"
COUNT_INSTANCE = "labelsz"


def open_source(spec: Mapping[str, Any], *, expect: str | tuple[str, ...],
                prefer_locked: bool = False) -> dict[str, Any]:
    """Resolve the ref, check the instance type, and return the pinned spec.

    The returned spec carries the concrete uuid, never the ref, so a branch that moves
    mid-run cannot leave a table half from one node and half from the next.
    """
    node = _vdvid.resolve_node(spec, prefer_locked=prefer_locked)
    pinned = {**dict(spec), "uuid": node["uuid"],
              "requested_ref": spec.get("requested_ref", node["ref"]),
              "ancestors_walked": node.get("walked", 0)}
    info = _vdvid.instance_info(pinned)
    expected = (expect,) if isinstance(expect, str) else tuple(expect)
    _vdvid.check_instance_type(info, pinned, *expected)
    if not node["locked"]:
        logger.warning(
            "%s is an OPEN node, so it is still being written to and this pull is not "
            "reproducible. Pass --dvid-locked for the newest locked node.",
            node["uuid"])
    return {**pinned, "node": node, "instance_info": info}


def require_sync(source: Mapping[str, Any]) -> list[str]:
    """The labelmap(s) an annotation instance is synced to, or raise explaining why not.

    ``/label/<body>`` relies on the sync's index, so without one every per-body fetch
    would fail.
    """
    synced = _vdvid.synced_instances(source.get("instance_info") or {})
    if not synced:
        raise ValueError(
            f"{_vdvid.spec_url(source)} is not synced to a labelmap, so DVID cannot say "
            f"which elements belong to a body — the /label endpoint that every fetch "
            f"here uses does not exist for an unsynced instance. Sync it in DVID (its "
            f"labelsz/annotation sync), or fetch by region instead of by body.")
    return synced


def _fetch_one(source: Mapping[str, Any], body: int) -> list[dict]:
    try:
        from neuclease.dvid.annotation import fetch_label
    except ImportError as exc:
        raise ImportError(MISSING) from exc

    server, uuid, instance = _vdvid.address(source)

    def go():
        return fetch_label(server, uuid, instance, body, relationships=True,
                           format="list")

    # The request is idempotent, so the whole body is the retry unit.
    return with_retry(go, label=f"annotation body {body}") or []


def fetch_points(source: Mapping[str, Any], bodies: Sequence[int], *,
                 threads: int = DEFAULT_THREADS,
                 progress_every: int = 500) -> dict[str, Any]:
    """Fetch every element of every body, and join relationships to bodies.

    Returns ``{"points", "relationships", "connections", "match", "failures",
    "elements"}``. ``failures`` is ``{body: message}`` for bodies that still failed after
    retries; the run continues without them.
    """
    results: list[tuple] = []
    failures: dict[int, str] = {}
    done = 0

    def work(body: int):
        return body, _fetch_one(source, body)

    with ThreadPoolExecutor(max_workers=max(1, int(threads))) as pool:
        for body, elements in pool.map(work, bodies):
            done += 1
            try:
                results.append(tables.elements_to_frames(elements, body))
            except Exception as exc:                            # noqa: BLE001
                failures[body] = f"{type(exc).__name__}: {exc}"
            if progress_every and done % progress_every == 0:
                logger.info("fetched %d/%d bodies", done, len(bodies))

    points, rels = tables.combine(results)
    conns = tables.connections(rels)
    return {"points": points, "relationships": rels, "connections": conns,
            "match": tables.match_rate(conns), "failures": failures,
            "elements": int(len(points))}


def fetch_body_annotations(source: Mapping[str, Any], bodies: Sequence[int], *,
                           batch_size: int = 10_000) -> dict[str, Any]:
    """Fetch the per-body records from a keyvalue instance.

    Uses ``fetch_keyvalues`` rather than neuclease's ``fetch_body_annotations``, which
    raises on any ``status`` outside a fixed vocabulary; statuses stay plain strings.
    Bodies with no record are absent from the result and listed in ``missing``.
    """
    try:
        from neuclease.dvid.keyvalue import fetch_keyvalues
    except ImportError as exc:
        raise ImportError(MISSING) from exc

    server, uuid, instance = _vdvid.address(source)
    keys = [str(b) for b in bodies]

    def go():
        return fetch_keyvalues(server, uuid, instance, keys, as_json=True,
                               batch_size=batch_size, show_progress=False)

    values = with_retry(go, label=f"body annotations for {len(keys)} bodies") or {}
    frame = tables.keyvalues_to_frame(values)
    return {"bodies": frame, "requested": len(keys), "found": int(len(frame)),
            "missing": [int(k) for k in keys if int(k) not in set(frame["body"].tolist())]}


# --------------------------------------------------------------------------- #
# choosing which bodies to fetch, from DVID's own ranked synapse index
# --------------------------------------------------------------------------- #
#: DVID's page size for a `labelsz` threshold query: it caps a response at 10,000 and
#: pages by rank via `offset`.
_LABELSZ_PAGE = 10_000

#: How deep to page. Each page costs more than the last (4.8 s at offset 0, 11.5 s at
#: 50,000), so an unbounded walk down a low threshold looks like a hang.
_MAX_SELECT = 500_000


def resolve_labelsz(source: Mapping[str, Any]) -> dict[str, Any]:
    """Point a source at the ``labelsz`` instance to query, from whatever was given.

    Accepts the ``labelsz`` instance itself, or the ``annotation`` instance it indexes
    (e.g. ``synapses``), found through the labelsz instance's ``Base.Syncs``.
    """
    info = source.get("instance_info") or _vdvid.instance_info(source)
    kind = _vdvid.instance_type(info)
    if kind == COUNT_INSTANCE:
        return {**dict(source), "indexes": _vdvid.synced_instances(info) or None}

    # An annotation instance: find the labelsz that syncs to it.
    try:
        from neuclease.dvid import fetch_repo_info
    except ImportError as exc:
        raise ImportError(MISSING) from exc

    server, uuid, instance = _vdvid.address(source)
    repo = fetch_repo_info(server, uuid)
    matches = [name for name, d in (repo.get("DataInstances") or {}).items()
               if d.get("Base", {}).get("TypeName") == COUNT_INSTANCE
               and instance in (d.get("Base", {}).get("Syncs") or [])]
    if not matches:
        raise ValueError(
            f"no 'labelsz' instance on this node indexes {instance!r}, so DVID has no "
            f"per-body synapse counts to rank by. Point --src at a labelsz instance "
            f"directly, or have one created and synced to {instance!r} in DVID.")
    if len(matches) > 1:
        raise ValueError(
            f"several labelsz instances index {instance!r} ({', '.join(sorted(matches))}); "
            f"name the one you want in --src.")
    logger.info("using labelsz instance %r (it indexes %r)", matches[0], instance)
    return {**dict(source), "instance": matches[0], "indexes": [instance]}


def fetch_synapse_counts(source: Mapping[str, Any], *, min_total: int = 10,
                         min_pre: int = 0, min_post: int = 0,
                         limit: int | None = None):
    """Bodies with at least this many synapses, with their pre/post counts.

    One ranked ``threshold`` query on ``AllSyn`` at the largest of the three minimums
    (``AllSyn`` counts every type, so it is at least each one), then exact pre and post
    counts for the bodies it returns.

    Returns a DataFrame indexed 0..n with ``body``, ``pre``, ``post``, ``syn``, ranked by
    ``syn`` descending.
    """
    import pandas as pd

    from neuclease.dvid import labelsz

    server, uuid, instance = _vdvid.address(source)
    threshold = max(int(min_total), int(min_pre), int(min_post), 1)

    # Page explicitly, up to _MAX_SELECT: per-page cost grows with offset.
    pages, offset = [], 0
    while offset < _MAX_SELECT:
        want = min(_LABELSZ_PAGE, _MAX_SELECT - offset)
        page = with_retry(
            lambda o=offset, w=want: labelsz.fetch_threshold(
                server, uuid, instance, threshold, "AllSyn", offset=o, n=w),
            label=f"labelsz AllSyn>={threshold} offset={offset}")
        if page is None or not len(page):
            break
        pages.append(page)
        offset += len(page)
        if len(page) < want:
            break
        logger.info("selected %d bodies so far (AllSyn >= %d)", offset, threshold)
    else:
        logger.warning(
            "stopped at the %d-body cap with more bodies still above the threshold. "
            "Raise --min-synapses, or --limit what you need: paging deeper costs more per "
            "page than the last.", _MAX_SELECT)

    if not pages:
        # DVID answers an unindexed element type with an empty result, not an error.
        raise ValueError(
            f"no bodies have AllSyn >= {threshold} in {_vdvid.spec_url(source)}. If that "
            f"is surprising, check that this labelsz instance is synced to an annotation "
            f"instance — DVID answers an unindexed element type with an EMPTY result "
            f"rather than an error.")

    syn = pd.concat(pages)
    bodies = [int(b) for b in syn.index]
    pre = with_retry(
        lambda: labelsz.fetch_counts(server, uuid, instance, bodies, "PreSyn"),
        label="labelsz PreSyn counts")
    post = with_retry(
        lambda: labelsz.fetch_counts(server, uuid, instance, bodies, "PostSyn"),
        label="labelsz PostSyn counts")

    df = pd.DataFrame({"syn": syn.astype("int64")})
    df["pre"] = pre.reindex(df.index).fillna(0).astype("int64")
    df["post"] = post.reindex(df.index).fillna(0).astype("int64")
    if min_pre:
        df = df[df["pre"] >= int(min_pre)]
    if min_post:
        df = df[df["post"] >= int(min_post)]
    df = df.sort_values(["syn", "pre"], ascending=False)
    if limit is not None:
        df = df.head(int(limit))
    out = df.reset_index()
    out = out.rename(columns={out.columns[0]: "body"})
    out["body"] = out["body"].astype(tables.BODY_DTYPE)
    return out[["body", "pre", "post", "syn"]]


# --------------------------------------------------------------------------- #
# which neuropil each synapse is in
# --------------------------------------------------------------------------- #
#: What neuclease names a point inside none of the given ROIs; stored as null here.
ROI_UNSPECIFIED = "<unspecified>"

# Listing and fetching ROIs live in `neu_vol.dvid`; this module only samples them at
# synapse positions.


#: What to do when the chosen ROIs intersect: report how many synapses are affected
#: ("warn", the default) or refuse ("error").
ON_OVERLAP = ("warn", "error")


def label_point_rois(source: Mapping[str, Any], points, rois: Sequence[str], *,
                     on_overlap: str = "warn", processes: int = 0) -> dict[str, Any]:
    """Add a ``roi`` column to a points frame: which neuropil each synapse falls in.

    Wraps ``neuclease.dvid.roi.determine_point_rois``, which builds one combined label
    volume from the ROIs at scale 5 and samples it at each point's scale-0 ``x``/``y``/``z``
    columns.

    **Overlapping ROIs:** the combined volume is built by writing each ROI in turn, so in an
    intersection the ROI passed last wins. With ``on_overlap="warn"`` the volume is
    unpacked again in reverse order (the fetch is not repeated), and the result reports how
    many synapses would be attributed differently (``ambiguous``, ``ambiguous_pairs``).
    ``on_overlap="error"`` refuses instead; some datasets have subtracted variants such as
    ``INP(-ATL)(L)`` for a strict partition.
    """
    import pandas as pd

    try:
        from neuclease.dvid.roi import (determine_point_rois,
                                        unpack_roi_ranges_to_combined_volume)
    except ImportError as exc:
        raise ImportError(MISSING) from exc

    if on_overlap not in ON_OVERLAP:
        raise ValueError(f"on_overlap must be one of {', '.join(ON_OVERLAP)}; "
                         f"got {on_overlap!r}")

    server, uuid, _instance = _vdvid.address(source)

    logger.info("fetching %d ROIs", len(rois))
    # Fetched once; only the cheap unpack is repeated below.
    names, ranges, boxes = _vdvid.fetch_roi_ranges(source, rois, processes=processes)
    volume, box, overlaps = unpack_roi_ranges_to_combined_volume(names, ranges, boxes)

    overlapping = [] if overlaps is None or not len(overlaps) else [
        (str(a), str(b), int(n)) for a, b, n in
        overlaps[["roi_a", "roi_b", "overlap"]].itertuples(index=False)]
    if overlapping and on_overlap == "error":
        raise ValueError(
            f"{len(overlapping)} of the given ROIs overlap, e.g. {overlapping[:4]}. The "
            f"combined volume is built by overwriting, so every point in an intersection "
            f"is attributed to whichever ROI was passed last. Choose a non-overlapping set "
            f"(this dataset has subtracted variants such as INP(-ATL)(L) and PENP(-AMMC) "
            f"for exactly this), or use on_overlap='warn' to proceed and have the affected "
            f"synapse count reported.")

    if not len(points):
        out = points.copy()
        out["roi"] = pd.Series(dtype="string")
        return {"points": out, "rois": names, "labeled": 0, "unlabeled": 0,
                "overlapping": overlapping, "ambiguous": 0, "counts": {}}

    # determine_point_rois mutates in place and needs a unique index.
    work = points.reset_index(drop=True).copy()
    determine_point_rois(server, uuid, names, work, combined_vol=volume, combined_box=box)
    roi = work["roi"].astype("string")
    roi = roi.where(roi != ROI_UNSPECIFIED, pd.NA)

    ambiguous, ambiguous_counts = 0, {}
    if overlapping:
        # Reverse the ROI order and see which points change hands.
        reversed_names = list(reversed(names))
        alt_vol, alt_box, _ = unpack_roi_ranges_to_combined_volume(
            reversed_names, ranges, boxes, box_zyx=box)
        alt = points.reset_index(drop=True).copy()
        determine_point_rois(server, uuid, reversed_names, alt,
                             combined_vol=alt_vol, combined_box=alt_box)
        other = alt["roi"].astype("string")
        other = other.where(other != ROI_UNSPECIFIED, pd.NA)
        differs = (roi.fillna("") != other.fillna(""))
        ambiguous = int(differs.sum())
        # Name the competing pair for each ambiguous point.
        pairs = pd.Series([f"{x} | {y}" for x, y in
                           zip(roi[differs].fillna("(none)"),
                               other[differs].fillna("(none)"))], dtype="object")
        ambiguous_counts = ({str(k): int(v) for k, v in pairs.value_counts().items()}
                            if len(pairs) else {})
        logger.warning(
            "%d ROI pairs intersect; %d of %d synapses (%.2f%%) sit in an intersection and "
            "are attributed by ROI ORDER alone: %s",
            len(overlapping), ambiguous, len(roi), 100 * ambiguous / max(len(roi), 1),
            ambiguous_counts or "none of yours")

    out = points.reset_index(drop=True).copy()
    out["roi"] = roi
    # `roi_label` (an index into the ROI list) is not kept; the list goes into provenance.
    unlabeled = int(roi.isna().sum())
    counts = {str(k): int(v) for k, v in roi.value_counts().items()}
    logger.info("%d of %d points fell inside an ROI (%d outside every one)",
                len(roi) - unlabeled, len(roi), unlabeled)
    return {"points": out, "rois": names, "labeled": int(len(roi) - unlabeled),
            "unlabeled": unlabeled, "overlapping": overlapping,
            "ambiguous": ambiguous, "ambiguous_pairs": ambiguous_counts,
            "counts": counts}


# --------------------------------------------------------------------------- #
# the whole keyvalue instance, without asking for its key list
# --------------------------------------------------------------------------- #
#: DVID's own full-range bounds for a keyvalue query, which is what neuclease defaults to.
_RANGE_LO = " "
_RANGE_HI = chr(ord("~") + 1)

#: Boundaries covering the printable key space; each consecutive pair is one request, and
#: together they cover ``" "`` to ``"~"``, so no total key count is needed. Body-id keys are
#: decimal strings; the outer ranges catch non-numeric keys.
DEFAULT_KEY_BOUNDARIES = (_RANGE_LO, *"0123456789", ":", _RANGE_HI)


def key_ranges(boundaries: Sequence[str] = DEFAULT_KEY_BOUNDARIES
               ) -> list[tuple[str, str]]:
    """Consecutive pairs of ``boundaries`` as ``(key1, key2)`` request bounds.

    **DVID's key ranges are inclusive at both ends**, so consecutive ranges overlap by one
    key where a key equals a boundary (body ``3`` is in both ``'2'..'3'`` and ``'3'..'4'``).
    Collecting into a dict de-duplicates, but summing per-range counts overcounts.
    """
    bounds = list(boundaries)
    if len(bounds) < 2:
        raise ValueError("need at least two boundaries to make a range")
    if sorted(bounds) != bounds:
        raise ValueError(f"boundaries must be sorted; got {bounds}")
    return [(a, b) for a, b in zip(bounds, bounds[1:]) if a != b]


#: How many times a failing key range may be subdivided. Each level multiplies the requests
#: by up to eleven, so a deeper limit would turn a dead server into an hours-long failure.
#: A dead server still fails fast: the first sub-range of a split cannot be split again.
MAX_SPLIT_DEPTH = 2


def refine(lo: str, hi: str) -> list[str]:
    """Boundaries subdividing ``[lo, hi]`` by appending one digit to ``lo``.

    ``'1'``..``'2'`` becomes ``'1', '10', '11', … '19', '2'``, since decimal-string keys sort
    lexicographically. A range that times out is split this way rather than retried, because
    the proxy's time limit is fixed while the server's speed varies with load.

    Candidates outside ``(lo, hi)`` are dropped, so boundaries stay sorted; ``[lo, hi]`` is
    returned unchanged when none falls strictly between them (it cannot be split further).
    """
    inner = [lo + d for d in "0123456789"]
    return [lo, *(m for m in inner if lo < m < hi), hi]


def fetch_all_body_annotations(source: Mapping[str, Any], *,
                               boundaries: Sequence[str] = DEFAULT_KEY_BOUNDARIES
                               ) -> dict[str, Any]:
    """Every record in a keyvalue instance, fetched as a series of bounded key ranges.

    Never calls ``/keys``, which on a large instance times out at DVID's proxy (504). The
    instance is read as ``keyrangevalues`` over ranges that cover the whole key space, so
    no total is needed, and values cost little more than keys. A range that fails is split
    (:func:`refine`) up to :data:`MAX_SPLIT_DEPTH` times. Boundary keys arrive twice, since
    DVID's bounds are inclusive; a repeat with a different value raises, as the snapshot
    would not be coherent.
    """
    try:
        from neuclease.dvid.keyvalue import fetch_keyrangevalues
    except ImportError as exc:
        raise ImportError(MISSING) from exc

    server, uuid, instance = _vdvid.address(source)
    records: dict[str, Any] = {}
    per_range: list[dict] = []

    def fetch(lo: str, hi: str, depth: int) -> None:
        try:
            # One attempt: a range that timed out fails again if repeated, so split it.
            got = fetch_keyrangevalues(server, uuid, instance, lo, hi, as_json=True) or {}
        except Exception as exc:                                  # noqa: BLE001
            parts = key_ranges(refine(lo, hi))
            if depth >= MAX_SPLIT_DEPTH or len(parts) < 2:
                raise RuntimeError(
                    f"[{lo!r},{hi!r}] still fails and cannot usefully be split further "
                    f"(depth {depth} of {MAX_SPLIT_DEPTH}): "
                    f"{type(exc).__name__}: {exc}") from exc
            logger.warning("[%r,%r] failed (%s); splitting it and fetching the parts",
                           lo, hi, type(exc).__name__)
            for sub_lo, sub_hi in parts:
                fetch(sub_lo, sub_hi, depth + 1)
            return

        # Boundary keys arrive twice (inclusive bounds); a repeat with a different value
        # would mean the instance changed mid-read.
        inconsistent = [k for k in set(got) & set(records) if got[k] != records[k]]
        if inconsistent:
            raise RuntimeError(
                f"key(s) {sorted(inconsistent)[:3]} came back with different values from "
                f"two key ranges. The instance changed mid-read, or the server answered "
                f"inconsistently — either way this snapshot is not coherent.")
        records.update(got)
        per_range.append({"lo": lo, "hi": hi, "records": len(got), "depth": depth})
        logger.info("fetched %d records from [%r,%r] — %d so far",
                    len(got), lo, hi, len(records))

    for lo, hi in key_ranges(boundaries):
        fetch(lo, hi, 0)

    return {"records": records, "ranges": per_range}


def fetch_synapse_counts_for(source: Mapping[str, Any], bodies: Sequence[int]):
    """Exact ``pre``/``post``/``syn`` for specific bodies, ignoring any threshold.

    :func:`fetch_synapse_counts` finds the bodies above a threshold; this counts a given
    list.
    """
    import pandas as pd

    from neuclease.dvid import labelsz

    server, uuid, instance = _vdvid.address(source)
    ids = [int(b) for b in bodies]
    pre = with_retry(
        lambda: labelsz.fetch_counts(server, uuid, instance, ids, "PreSyn"),
        label="labelsz PreSyn counts")
    post = with_retry(
        lambda: labelsz.fetch_counts(server, uuid, instance, ids, "PostSyn"),
        label="labelsz PostSyn counts")
    out = pd.DataFrame({"pre": pre.reindex(ids).fillna(0).astype("int64"),
                        "post": post.reindex(ids).fillna(0).astype("int64")})
    out["syn"] = out["pre"] + out["post"]
    return out.rename_axis("body").reset_index()


def fetch_voxel_counts(source: Mapping[str, Any], bodies: Sequence[int],
                       *, batch_size: int = 1000) -> dict[int, int]:
    """``{body: voxels}`` at full resolution, from the labelmap's own ``/sizes``.

    Not from ``labelsz``, whose ``'Voxels'`` type returns an empty result, not an error,
    without a labelvol sync. About a minute for 20,000 bodies.
    """
    try:
        from neuclease.dvid import fetch_sizes
    except ImportError as exc:
        raise ImportError(MISSING) from exc

    server, uuid, instance = _vdvid.address(source)
    ids = [int(b) for b in bodies]
    series = with_retry(
        lambda: fetch_sizes(server, uuid, instance, ids, batch_size=batch_size),
        label=f"sizes for {len(ids)} bodies")
    return {int(b): int(v) for b, v in series.items()}


def node_record(source: Mapping[str, Any], **extra: Any) -> dict[str, Any]:
    """The provenance record for this source, with whatever the caller wants added."""
    node = source.get("node") or _vdvid.resolve_node(source)
    rec = _vdvid.node_provenance(
        source,
        {**node,
         "ref": source.get("requested_ref", node.get("ref")),
         "walked": source.get("ancestors_walked", 0)})
    info = source.get("instance_info") or {}
    rec["instance_type"] = info.get("Base", {}).get("TypeName")
    if synced := _vdvid.synced_instances(info):
        # Which labelmap the body ids belong to.
        rec["synced_to"] = synced
    rec.update({k: v for k, v in extra.items() if v is not None})
    return rec
