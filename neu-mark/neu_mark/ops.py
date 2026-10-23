"""The write operations, as functions the CLI is a thin shell over.

Each takes an already-opened source (from an ``open_*_source`` function in
:mod:`neu_mark.dvid`), which
carries the concrete uuid rather than a ref. The caller resolves the node once, so the
tables, the provenance record and a destination name such as ``--dst {uuid:8}`` all
describe the same node.

Each fetches for an explicit body list, writes tables, then writes provenance. There is
no dask or manifest: 20,000 bodies take about 16 minutes in one process.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

from . import dvid as _dvid
from . import io as _io
from . import tables as _tables

logger = logging.getLogger(__name__)

#: Table names written by `fetch_points`, and what each holds.
POINT_TABLES = ("points", "relationships")
BODY_TABLES = ("bodies",)


def write_selected_bodies(source: Mapping[str, Any], path: str, *, min_synapses: int = 10,
                          min_pre: int = 0, min_post: int = 0,
                          limit: int | None = None) -> dict[str, Any]:
    """Choose which bodies are worth fetching annotations for, and write the list.

    The list goes to ``path``, in the format its extension names (``.csv``, ``.parquet``
    or ``.feather``), and its provenance record beside it as ``<name>.provenance.json``.

    Uses DVID's ranked ``labelsz`` index, which takes seconds, and selects on synapses
    rather than size.

    The default threshold is on the total, not on pre and post separately: sensory neurons
    may have no postsynapses, and a neuron leaving the volume may have no presynapses.
    ``min_pre`` / ``min_post`` add per-type minimums.

    The provenance record names the node the list was computed from, since proofreading
    changes body ids.
    """
    from .notebook import select_bodies

    df = select_bodies(source, min_synapses=min_synapses, min_pre=min_pre,
                       min_post=min_post, limit=limit)
    run = {"min_synapses": min_synapses, "min_pre": min_pre or None,
           "min_post": min_post or None, "limit": limit,
           "bodies_selected": int(len(df)),
           "indexes": source.get("indexes"),
           "synapses_total": int(df["syn"].sum()) if len(df) else 0}
    directory, name, fmt = _io.split_table_path(path)
    record = _record(source, path, run=run)
    _io.write_table(df, directory, name, fmt=fmt, metadata=record)
    _io.write_provenance(directory, record, name=f"{name}.provenance")
    return {"bodies": df, "path": path, "record": record}


def _record(source: Mapping[str, Any], dst: str, *, run: Mapping[str, Any]) -> dict:
    from neu_vol.ops.provenance import build_record

    record = build_record(src_spec={"backend": "dvid", **dict(source)}, dst=dst, **run)
    record["tool"] = "neu-mark"
    record["source"] = _dvid.node_record(source)
    return record


def fetch_points(source: Mapping[str, Any], dst: str, bodies: Sequence[int] | None, *,
                 threads: int = _dvid.DEFAULT_THREADS, fmt: str = "parquet",
                 drop_unmatched: bool = False, rois: Sequence[str] | None = None,
                 on_roi_overlap: str = "warn", roi_source: Mapping[str, Any] | None = None,
                 write_connections: bool = False) -> dict[str, Any]:
    """Point annotations for ``bodies`` -> ``points`` + ``relationships`` tables.

    ``source`` comes from :func:`neu_mark.dvid.open_points_source`; it carries a
    concrete uuid. ``bodies=None`` fetches the whole instance by region, with every
    element's body from the labelmap (:func:`neu_mark.dvid.fetch_all_points`).

    ``drop_unmatched`` removes relationships whose partner body was not resolved. It is off
    by default: an unfetched partner looks the same as a missing synapse, so dropping them
    at low coverage makes an incomplete connectome look complete. Check the match rate first.

    ``roi_source`` is the node whose ``rois`` label the points (default ``source``'s), for
    ROIs drawn on a later node than the points and body ids; the record names it.
    """
    synced = _dvid.require_sync(source)
    if bodies is None:
        logger.info("%s: every element, %d processes (bodies from %s)",
                    source["instance"], threads, synced[0])
        result = _dvid.fetch_all_points(source, threads=threads)
    else:
        logger.info("%s: %d bodies, %d threads (bodies from sync: %s)",
                    source["instance"], len(bodies), threads, ", ".join(synced))
        result = _dvid.fetch_points(source, bodies, threads=threads)
    points, rels, match = result["points"], result["relationships"], result["match"]

    roi_stats = None
    if rois:
        if roi_source is None:
            roi_source = source
        # Only points get a `roi`: a relationship's two ends may be in different neuropils.
        labelled = _dvid.label_point_rois(roi_source, points, rois,
                                          on_overlap=on_roi_overlap)
        points = labelled["points"]
        roi_stats = {k: labelled.get(k) for k in
                     ("rois", "labeled", "unlabeled", "overlapping", "ambiguous",
                      "ambiguous_pairs", "counts")}
        roi_stats["node"] = {"server": roi_source["server"], "uuid": roi_source["uuid"]}

    if drop_unmatched:
        before = len(rels)
        rels = rels[rels["to_body"].notna()].reset_index(drop=True)
        logger.info("dropped %d relationships with an unresolved partner",
                    before - len(rels))

    run = {"bodies_requested": None if bodies is None else len(bodies),
           "scope": "all" if bodies is None else "bodies", "threads": threads,
           "elements": int(len(points)), "relationships": int(len(rels)),
           "match": match, "synced_to": synced,
           "drop_unmatched": drop_unmatched or None,
           "rois": roi_stats,
           "failures": result["failures"] or None}
    record = _record(source, dst, run=run)

    written = [_io.write_table(points, dst, "points", fmt=fmt, metadata=record),
               _io.write_table(rels, dst, "relationships", fmt=fmt, metadata=record)]
    if write_connections:
        written.append(_io.write_table(result["connections"], dst, "connections",
                                       fmt=fmt, metadata=record))
    _io.write_provenance(dst, record)

    return {**result, "points": points, "relationships": rels, "rois": roi_stats,
            "written": written, "record": record}


def annotation_source(dst: str, *, connections=None, points=None,
                      source: Mapping[str, Any] | None = None,
                      bodies: Sequence[int] | None = None,
                      voxel_size_xyz: Sequence[float] = (8.0, 8.0, 8.0),
                      bounds: tuple[Sequence[float], Sequence[float]] | None = None,
                      include_partial: bool = True, per_cell: int = 4_000,
                      roi_spec: str | None = None, listed: Sequence[int] | None = None,
                      threads: int = _dvid.DEFAULT_THREADS) -> dict[str, Any]:
    """Write a precomputed LINE annotation source: one line per synaptic connection.

    Pass ``connections`` and ``points`` from an earlier ``neu-mark points`` run, or a
    ``source`` and ``bodies`` to fetch now. ``bounds`` (the format's required
    ``lower_bound``/``upper_bound``) defaults to the data's extent plus one voxel.
    ``roi_spec`` (a ``roi-layers`` spec) adds the ``roi`` and ``neuropil`` properties, by
    the spec's ids, from the points' ``roi`` column. ``listed`` (body ids) adds
    ``pre_listed`` / ``post_listed``.
    """
    import numpy as np

    from . import annsource, tables as _t

    if connections is None:
        if source is None or not bodies:
            raise ValueError("pass connections+points, or source+bodies to fetch")
        fetched = _dvid.fetch_points(source, bodies, threads=threads)
        connections, points = fetched["connections"], fetched["points"]
        logger.info("fetched %d elements -> %d connections",
                    len(points), len(connections))
    if points is None:
        raise ValueError("points are needed to attach each endpoint's confidence")

    enriched, rois = enrich_for_source(connections, points, roi_spec)

    if bounds is None:
        corners = np.vstack([_t.positions_xyz(enriched, "pre_"),
                             _t.positions_xyz(enriched, "post_")]).astype(float)
        bounds = (corners.min(axis=0), corners.max(axis=0) + 1.0)

    built = annsource.build(enriched, lower_bound=bounds[0], upper_bound=bounds[1],
                            voxel_size_xyz=voxel_size_xyz,
                            include_partial=include_partial, per_cell=per_cell,
                            listed=listed, **rois)
    written = annsource.write(dst, built)

    run = {"lines": built["report"]["lines"], "connections": built["report"]["connections"],
           "both_bodies": built["report"]["both_bodies"],
           "one_body": built["report"]["one_body"],
           "include_partial": include_partial, "roi_spec": roi_spec,
           "listed": None if listed is None else len(listed),
           "levels": len(built["info"]["spatial"]),
           "stride": built["report"]["stride"]}
    if source is not None:
        record = _record(source, dst, run=run)
    else:
        from neu_vol.ops.provenance import build_record
        record = build_record(src_spec={"backend": "tables"}, dst=dst, **run)
        record["tool"] = "neu-mark"
    _io.write_provenance(dst, record)
    # `table` is the enriched frame the source was built from.
    return {**built, "written": written, "record": record, "table": enriched}


def enrich_for_source(connections, points, roi_spec: str | None = None):
    """``(connections with each end's conf (and roi), build kwargs)`` for
    :func:`neu_mark.annsource.build`; the CLI's dry run shares it."""
    from . import tables as _t

    if roi_spec is None:
        return _t.enrich_connections(connections, points, columns=("conf",)), {}
    from .roi_layers import load_spec, top_level

    spec = load_spec(roi_spec)
    enriched = _t.enrich_connections(connections, points, columns=("conf", "roi"))
    return enriched, {"roi_ids": spec.ids, "top_level": top_level(spec)}


def segment_properties(bodies_source: Mapping[str, Any], dst: str,
                       bodies: Sequence[int], *,
                       counts_source: Mapping[str, Any] | None = None,
                       labelmap_source: Mapping[str, Any] | None = None,
                       synapses: str | None = None, roi_spec: str | None = None,
                       rules_path: str | None = None, rules_only: bool = False,
                       max_tags: float | None = None,
                       keep_glia: bool = True, link: bool = True,
                       dry_run: bool = False) -> dict[str, Any]:
    """Build a ``segment_properties`` source and, by default, link the volume to it.

    ``dst`` is the segmentation volume. The document goes to
    ``<dst>/segment_properties/info``, and linking adds a ``"segment_properties"`` key to
    the volume's own ``info``, which is what makes neuroglancer show names and tags.
    ``link=False`` writes the document without touching the volume; ``dry_run`` builds and
    reports without writing anything.

    ``rules_path`` loads a rules module that overrides the builtins by name (``rules_only``
    uses it alone). ``counts_source`` (a labelsz instance) adds ``pre``/``post``/``syn``,
    and ``labelmap_source`` adds ``voxels``.

    ``synapses`` is a directory an earlier ``points --rois`` run wrote; it gives the rules a
    ``synapse_neuropils`` column (:func:`neu_mark.tables.body_neuropil_fractions`), and
    ``roi_spec`` (a ``roi-layers`` spec) a ``synapse_neuropils_top`` column grouping those
    by top-level neuropil.
    """
    from . import segprops

    ruleset = None
    rules_record = None
    if rules_path:
        from . import rules as _rules

        ruleset = _rules.from_module(rules_path)
        logger.info("loaded %d rule(s) from %s: %s", len(ruleset), rules_path,
                    ", ".join(ruleset.names))
        rules_record = _rules.provenance(ruleset)

    frame = _dvid.fetch_body_annotations(bodies_source, bodies)["bodies"]
    logger.info("%d of %d bodies have a property record", len(frame), len(bodies))
    if synapses is not None:
        frame = _with_neuropils(frame, synapses, roi_spec)
    elif roi_spec is not None:
        raise ValueError("roi_spec groups the synapse neuropils, so it needs synapses")

    counts = None
    if counts_source is not None:
        counts = _dvid.fetch_synapse_counts_for(counts_source, bodies)
    sizes = None
    if labelmap_source is not None:
        sizes = _dvid.fetch_voxel_counts(labelmap_source, bodies)

    extra = {} if max_tags is None else {"max_tags": float(max_tags)}
    built = segprops.build(frame, rules=ruleset, rules_only=rules_only, counts=counts,
                           sizes=sizes, keep_glia=keep_glia, **extra)

    if dry_run:
        logger.info("dry run: nothing written to %s", str(dst).rstrip("/"))
        return {**built, "written": [], "linked": None, "record": None, "dry_run": True}

    written = [segprops.write(dst, built["info"])]

    linked = None
    if link:
        from neu_vol.ops.subresources import link_subresources

        linked = link_subresources(dst, segment_properties=segprops.SUBDIR)
        logger.info("linked %s into %s/info", linked, str(dst).rstrip("/"))

    run = {"bodies_requested": len(bodies), "bodies_with_records": int(len(frame)),
           "segments": built["report"]["bodies"], "excluded": built["report"]["excluded"],
           "tags": built["report"]["tags"], "tag_limit": built["report"]["tag_limit"],
           "max_tags": built["report"]["max_tags"],
           "facets": built["report"]["facets"], "bytes": built["report"]["bytes"],
           "coverage": built["report"]["coverage"],
           "numbers": built["report"]["numbers"], "keep_glia": keep_glia,
           # The rules decide what the document says; see `rules.provenance`.
           "rules": rules_record, "rules_only": bool(rules_only),
           "synapses": synapses, "roi_spec": roi_spec,
           "linked": linked}
    record = _record(bodies_source, dst, run=run)
    _io.write_provenance(dst, record, name=f"{segprops.SUBDIR}/provenance")

    return {**built, "written": written, "linked": linked, "record": record,
            "dry_run": False}


def _with_neuropils(frame, synapses: str, roi_spec: str | None):
    """``frame`` with each body's synapse neuropil shares; a body with no synapses gets
    empty ones, so a rule never meets a missing value."""
    from . import tables as _tables

    (points,) = _io.read_tables(synapses, ("points",))
    top = None
    if roi_spec is not None:
        from .roi_layers import load_spec, top_level

        top = top_level(load_spec(roi_spec))
    shares = _tables.body_neuropil_fractions(points, top_level=top)
    shares["body"] = shares["body"].astype(frame["body"].dtype)
    out = frame.merge(shares, on="body", how="left")
    for column in shares.columns.drop("body"):
        out[column] = [v if isinstance(v, dict) else {"pre": {}, "post": {}, "all": {}}
                       for v in out[column]]
    logger.info("synapse neuropils for %d of %d bodies, from %s", int(frame["body"].isin(
        shares["body"]).sum()), len(frame), synapses)
    return out


def fetch_bodies(source: Mapping[str, Any], dst: str,
                 bodies: Sequence[int] | None = None, *,
                 fmt: str = "parquet", everything: bool = False) -> dict[str, Any]:
    """Body annotations -> a ``bodies`` table, for a body list or the whole instance.

    ``source`` comes from :func:`neu_mark.dvid.open_bodies_source`; it carries a
    concrete uuid.

    ``everything=True`` reads every record in the instance instead
    (:func:`neu_mark.dvid.fetch_all_body_annotations`), a different population from a
    synapse-selected list. Written to its own destination, apart from the point tables.
    """
    if everything:
        if bodies:
            raise ValueError(
                "pass a body list or everything=True, not both — they describe different "
                "populations and silently intersecting them would be a surprise.")
        pulled = _dvid.fetch_all_body_annotations(source)
        frame = _tables.keyvalues_to_frame(pulled["records"])
        run = {"whole_instance": True, "key_ranges": pulled["ranges"],
               "bodies_found": int(len(frame))}
        record = _record(source, dst, run=run)
        written = [_io.write_table(frame, dst, "bodies", fmt=fmt, metadata=record)]
        _io.write_provenance(dst, record)
        return {"bodies": frame, "requested": None, "found": int(len(frame)),
                "missing": [], "ranges": pulled["ranges"], "written": written,
                "record": record}

    if not bodies:
        raise ValueError("fetch_bodies needs a body list, or everything=True")
    result = _dvid.fetch_body_annotations(source, bodies)
    frame = result["bodies"]

    run = {"bodies_requested": result["requested"], "bodies_found": result["found"],
           "bodies_missing": len(result["missing"]) or None}
    record = _record(source, dst, run=run)
    written = [_io.write_table(frame, dst, "bodies", fmt=fmt, metadata=record)]
    _io.write_provenance(dst, record)
    return {**result, "written": written, "record": record}
