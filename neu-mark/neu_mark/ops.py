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
                          limit: int | None = None, min_pairs: int | None = None,
                          connections: str | None = None, named: str | None = None,
                          rules_path: str | None = None,
                          annotations_source: Mapping[str, Any] | None = None,
                          exclude: Sequence[str] = (),
                          labelmap_source: Mapping[str, Any] | None = None,
                          ) -> dict[str, Any]:
    """Choose which bodies are worth fetching annotations for, and write the list.

    The list goes to ``path``, in the format its extension names (``.csv``, ``.parquet``
    or ``.feather``), and its provenance record beside it as ``<name>.provenance.json``.

    Uses DVID's ranked ``labelsz`` index, which takes seconds, and selects on synapses
    rather than size.

    The default threshold is on the total, not on pre and post separately: sensory neurons
    may have no postsynapses, and a neuron leaving the volume may have no presynapses.
    ``min_pre`` / ``min_post`` add per-type minimums to that count.

    Two more criteria each add bodies the count missed (:mod:`neu_mark.selection`):

    - ``min_pairs`` with ``connections``, a directory ``points --connections`` wrote: bodies
      taking part in at least that many T-bar -> PSD connections. Adds a ``pairs`` column.
    - ``named``, a rule of the module at ``rules_path``, with ``annotations_source`` (the
      body annotation keyvalue): bodies whose record that rule gives a value and no drop
      rule excludes. Every record is read (about a minute). Adds a ``named`` column (0/1).

    ``exclude`` names rules of the same module: a body whose record any of them gives a
    value is left out, whichever criterion admitted it. ``labelmap_source`` gives every
    record a ``voxels`` field (full-resolution voxels, from DVID's ``/sizes``) for the
    rules to read.

    These must come from the node ``source`` is on, which is checked: body ids change with
    proofreading. ``limit`` cannot be combined with them.

    The provenance record names the node the list was computed from, since proofreading
    changes body ids.
    """
    import pandas as pd

    from . import selection as _selection
    from .notebook import select_bodies

    if (min_pairs is None) != (connections is None):
        raise ValueError("min_pairs and connections go together: the pairs are counted "
                         "from the connections table")
    exclude = list(exclude)
    reads_records = named is not None or bool(exclude)
    if reads_records and (rules_path is None or annotations_source is None):
        raise ValueError("named and exclude need rules_path (the rules) and "
                         "annotations_source (the records they read)")
    if labelmap_source is not None and not reads_records:
        raise ValueError("labelmap_source gives the rules a voxels field, so it needs "
                         "named or exclude")
    if limit is not None and (min_pairs is not None or reads_records):
        raise ValueError("limit keeps the top N by synapse count, which says nothing about "
                         "bodies added by pairs or by name, or taken out by a rule; use "
                         "one or the other")

    pairs = names = ruleset = None
    excluded = pd.Series(dtype="object")
    if connections is not None:
        _require_same_node(source, _tables_node(connections), f"the tables at {connections}")
        (conns,) = _io.read_tables(connections, ("connections",))
        pairs = _selection.pair_counts(conns)
    if reads_records:
        from . import rules as _rules

        _require_same_node(source, annotations_source.get("uuid"),
                           "the annotation source")
        ruleset = _rules.from_module(rules_path)
        for name in [named, *exclude]:               # an unknown rule fails before a fetch
            if name is not None:
                ruleset[name]
        records = _tables.keyvalues_to_frame(
            _dvid.fetch_all_body_annotations(annotations_source)["records"])
        if labelmap_source is not None:
            _require_same_node(source, labelmap_source.get("uuid"), "the labelmap sizes")
            sizes = _dvid.fetch_voxel_counts(labelmap_source, records["body"].tolist())
            records["voxels"] = [sizes.get(int(b)) for b in records["body"]]
        if named is not None:
            names = _selection.named_bodies(records, ruleset, named)
            logger.info("%d of %d annotation records have a %s", len(names), len(records),
                        named)
        if exclude:
            excluded = _selection.excluded_bodies(records, ruleset, exclude)

    df = select_bodies(source, min_synapses=min_synapses, min_pre=min_pre,
                       min_post=min_post, limit=limit)
    added = _selection.admitted(df, pairs=pairs, min_pairs=min_pairs, named=names)
    gone = set(int(b) for b in excluded.index)
    kept = ~df["body"].isin(excluded.index)
    dropped = {"from_count": int((~kept).sum()),
               "from_added": len([b for b in added["extra"] if b in gone])}
    df = df[kept].reset_index(drop=True)
    added = {k: [b for b in v if b not in gone] for k, v in added.items()}
    if added["extra"]:
        more = _dvid.fetch_synapse_counts_for(source, added["extra"])
        more["body"] = more["body"].astype(_tables.BODY_DTYPE)
        df = (pd.concat([df, more[["body", "pre", "post", "syn"]]], ignore_index=True)
              .sort_values(["syn", "pre"], ascending=False, ignore_index=True))
    if pairs is not None:
        df["pairs"] = pairs.reindex(df["body"].to_numpy()).fillna(0).astype("int64").to_numpy()
    if names is not None:
        df["named"] = df["body"].isin(names.index).astype("int64")

    counted = int(len(df)) - len(added["extra"])
    run = {"min_synapses": min_synapses, "min_pre": min_pre or None,
           "min_post": min_post or None, "limit": limit,
           "min_pairs": min_pairs, "connections": connections, "named": named,
           "exclude": exclude or None,
           "labelmap": (labelmap_source or {}).get("instance"),
           "rules": _rules_record(ruleset),
           "bodies_selected": int(len(df)),
           "bodies_by_synapses": counted,
           "bodies_added": ({"by_pairs": len(set(added["extra"]) & set(added["by_pairs"])),
                             "by_name": len(set(added["extra"]) & set(added["by_name"])),
                             "total": len(added["extra"])}
                            if (pairs is not None or names is not None) else None),
           "bodies_excluded": ({**dropped, "records_by_rule":
                                excluded.value_counts().to_dict()} if exclude else None),
           "indexes": source.get("indexes"),
           "synapses_total": int(df["syn"].sum()) if len(df) else 0}
    directory, name, fmt = _io.split_table_path(path)
    record = _record(source, path, run=run)
    _io.write_table(df, directory, name, fmt=fmt, metadata=record)
    _io.write_provenance(directory, record, name=f"{name}.provenance")
    return {"bodies": df, "path": path, "record": record, "added": added,
            "excluded": excluded, "dropped": dropped}


def write_criteria_bodies(source: Mapping[str, Any], path: str, *, criteria,
                          rules_path: str, annotations_source: Mapping[str, Any],
                          connections: str | None = None,
                          labelmap_source: Mapping[str, Any] | None = None,
                          sizes_cache: str | None = None) -> dict[str, Any]:
    """Write the bodies one list of a criteria file includes (:mod:`neu_mark.criteria`).

    ``criteria`` is a loaded :class:`~neu_mark.criteria.Criteria`. Each annotation record's
    category comes from its ``category_rule`` in the module at ``rules_path``. Bodies with no
    record get the category that rule gives an empty record; when the list can include
    those, each sufficient condition says where to find them: labelsz above its ``sites``,
    the pair counts above its ``pairs``, or, for ``size`` alone, a listing of every label
    of at least that size (``labelmap_source``; ~30 minutes, kept in ``sizes_cache`` for a
    locked node, :func:`neu_mark.dvid.large_bodies`).

    ``size`` needs ``labelmap_source``, ``pairs`` needs ``connections`` (a directory
    ``points --connections`` wrote); all must come from ``source``'s node, which is
    checked. The list has ``body``, ``pre``, ``post``, ``syn``, ``pairs`` and ``voxels``
    (when their sources are given), ``category`` and ``admitted_by`` (``all`` or the first
    sufficient condition that held), ranked by ``syn``; its provenance record keeps the
    whole list section and ``[common]``.
    """
    import pandas as pd

    from . import criteria as _criteria
    from . import rules as _rules
    from . import selection as _selection

    fields = criteria.fields
    if "size" in fields and labelmap_source is None:
        raise ValueError(f"[{criteria.name}] has a size condition: it needs a labelmap for "
                         f"the body sizes")
    if "pairs" in fields and connections is None:
        raise ValueError(f"[{criteria.name}] has a pairs condition: it needs the "
                         f"connections table the pairs are counted from")
    _require_same_node(source, annotations_source.get("uuid"), "the annotation source")
    if labelmap_source is not None:
        _require_same_node(source, labelmap_source.get("uuid"), "the labelmap sizes")

    ruleset = _rules.from_module(rules_path)
    rule = ruleset[criteria.category_rule]
    reporters = [ruleset[name] for name in criteria.report]   # unknown: fails before a fetch
    unrecorded = rule.scalar({"body": 0, **{n: None for n in rule.needs}})
    if unrecorded is None:
        raise ValueError(f"the rule {rule.name!r} gives no category for a body with no "
                         f"record; it must, so such bodies can be listed or left out")
    look = _criteria.candidate_sources(criteria, unrecorded)

    pairs = None
    if connections is not None:
        _require_same_node(source, _tables_node(connections), f"the tables at {connections}")
        (conns,) = _io.read_tables(connections, ("connections",))
        pairs = _selection.pair_counts(conns)

    records = _tables.keyvalues_to_frame(
        _dvid.fetch_all_body_annotations(annotations_source)["records"])
    cats = _selection.categories(records, rule)
    criteria.check_categories(cats.unique())
    logger.info("%d records: %s", len(cats), cats.value_counts().to_dict())
    reports = {r.name: _selection.reported(records, r) for r in reporters}
    for name, frame in reports.items():
        logger.info("%s: %d records", name, len(frame))

    found: dict[str, int] = {}
    extra: set[int] = set()
    recorded = set(int(b) for b in cats.index)
    if look["sites"] is not None:
        ranked = _dvid.ranked_synapse_counts(source, look["sites"])
        new = set(int(b) for b in ranked.index) - recorded
        found["sites"] = len(new)
        extra |= new
    if look["pairs"] is not None:
        new = set(int(b) for b in pairs.index[pairs >= look["pairs"]]) - recorded
        found["pairs"] = len(new)
        extra |= new
    listed = listed_from = None
    if look["size"] is not None:
        # Listed down to the smallest size any condition asks for, so a body missing from
        # the listing is below every size threshold.
        floor = _criteria.smallest(criteria, "size")
        listed, listed_from = _dvid.large_bodies(labelmap_source, floor, cache=sizes_cache)
        new = set(listed) - recorded
        found["size"] = len(new)
        extra |= new

    kept = cats[cats.map(criteria.modes) != "none"]
    table = pd.DataFrame({"body": [*kept.index.astype("int64"), *sorted(extra)],
                          "category": [*kept.to_list(), *[unrecorded] * len(extra)]})
    table = table.drop_duplicates("body").reset_index(drop=True)
    bodies = table["body"].tolist()
    if "sites" in fields:
        allsyn = _dvid.fetch_allsyn_counts(source, bodies)
        table["sites"] = table["body"].map(allsyn).fillna(0).astype("int64")
    if pairs is not None:
        table["pairs"] = pairs.reindex(table["body"].to_numpy()).fillna(0).astype(
            "int64").to_numpy()
    if "size" in fields:
        sizes = (listed if listed is not None
                 else _dvid.fetch_voxel_counts(labelmap_source, bodies))
        table["voxels"] = table["body"].map(sizes).fillna(0).astype("int64")

    table["admitted_by"] = criteria.select(table)
    chosen = table[table["admitted_by"].notna()].reset_index(drop=True)
    counts = _dvid.fetch_synapse_counts_for(source, chosen["body"].tolist())
    df = chosen[["body"]].merge(counts.astype({"body": "int64"}), on="body", how="left")
    if pairs is not None:
        df["pairs"] = chosen["pairs"].to_numpy()
    if labelmap_source is not None:
        have = (dict(zip(chosen["body"], chosen["voxels"])) if "voxels" in chosen else {})
        if listed is not None:
            have = {b: v for b, v in have.items() if b in listed}
        missing = [int(b) for b in chosen["body"] if int(b) not in have]
        if missing:
            have.update(_dvid.fetch_voxel_counts(labelmap_source, missing))
        df["voxels"] = chosen["body"].map(have).fillna(0).astype("int64").to_numpy()
    df["category"] = chosen["category"].to_numpy()
    df["admitted_by"] = chosen["admitted_by"].to_numpy()
    df["body"] = df["body"].astype(_tables.BODY_DTYPE)
    df = df.sort_values(["syn", "pre"], ascending=False, ignore_index=True)

    by_category = {}
    for category, group in table.groupby("category"):
        picked = group["admitted_by"].dropna()
        by_category[str(category)] = {"mode": criteria.modes[category],
                                      "candidates": int(len(group)),
                                      "included": int(len(picked)),
                                      "by": picked.value_counts().to_dict()}
    run = {"criteria": criteria.record(), "rules": _rules_record(ruleset),
           "connections": connections,
           "labelmap": (labelmap_source or {}).get("instance"),
           "unrecorded_category": unrecorded,
           "unrecorded_found": found or None,
           "sizes": (f"listed >= {_criteria.smallest(criteria, 'size')} from {listed_from}"
                     if listed is not None else
                     "per body" if "size" in fields else None),
           "bodies_selected": int(len(df)), "by_category": by_category,
           "indexes": source.get("indexes"),
           "synapses_total": int(df["syn"].sum()) if len(df) else 0}
    directory, name, fmt = _io.split_table_path(path)
    run["reported"] = {rule_name: {"records": int(len(frame)),
                                   "table": f"{name}.{rule_name}.{fmt}"}
                       for rule_name, frame in reports.items()} or None
    record = _record(source, path, run=run)
    _io.write_table(df, directory, name, fmt=fmt, metadata=record)
    for rule_name, frame in reports.items():
        _io.write_table(frame, directory, f"{name}.{rule_name}", fmt=fmt)
    _io.write_provenance(directory, record, name=f"{name}.provenance")
    return {"bodies": df, "path": path, "record": record, "by_category": by_category,
            "unrecorded": unrecorded, "found": found, "reports": reports}


def _tables_node(directory: str) -> str | None:
    """The DVID node a ``points`` run wrote ``directory`` from, per its provenance."""
    from neu_vol.location import read_json

    record = read_json(directory, "provenance.json") or {}
    return (record.get("source") or {}).get("uuid")


def _require_same_node(source: Mapping[str, Any], uuid: str | None, what: str) -> None:
    if uuid is None:
        raise ValueError(f"{what} name no DVID node (no provenance), so they cannot be "
                         f"checked against the node the counts come from "
                         f"({source.get('uuid')}). Body ids change with proofreading.")
    if str(uuid) != str(source.get("uuid")):
        raise ValueError(f"{what} come from node {uuid}, the synapse counts from "
                         f"{source.get('uuid')}. Body ids change with proofreading, so "
                         f"both must be read at the same node.")


def _rules_record(ruleset) -> dict | None:
    if ruleset is None:
        return None
    from . import rules as _rules

    return _rules.provenance(ruleset)


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
                       rules_path: str | None = None, label: str | None = None,
                       drop: Sequence[str] = (), max_tags: float | None = None,
                       link: bool = True,
                       table: str | None = None, dry_run: bool = False) -> dict[str, Any]:
    """Build a ``segment_properties`` source and, by default, link the volume to it.

    ``dst`` is the segmentation volume. The document goes to
    ``<dst>/segment_properties/info``, and linking adds a ``"segment_properties"`` key to
    the volume's own ``info``, which is what makes neuroglancer show names and tags.
    ``link=False`` writes the document without touching the volume; ``dry_run`` builds and
    reports without writing anything but the table.

    ``table`` is a path (``.parquet``, ``.csv`` or ``.feather``) for a row per requested
    body (:func:`neu_mark.segprops.as_table`), written with its provenance, dry run or not.

    ``rules_path`` loads a rules module (without one the document has labels and numbers
    only); ``label`` names the label column, overriding the module's ``LABEL``; ``drop``
    names rules whose bodies are excluded rather than tagged. ``counts_source`` (a labelsz
    instance) adds ``pre``/``post``/``syn``, and ``labelmap_source`` adds ``voxels``.

    ``synapses`` is a directory an earlier ``points --rois`` run wrote; it gives the rules a
    ``synapse_neuropils`` column (:func:`neu_mark.tables.body_neuropil_fractions`), and
    ``roi_spec`` (a ``roi-layers`` spec) a ``synapse_neuropils_top`` column grouping those
    by top-level neuropil.
    """
    from . import segprops

    if table is not None:
        _io.split_table_path(table)          # refuse a bad extension before any fetch
    ruleset = None
    rules_record = None
    if rules_path:
        from . import rules as _rules

        ruleset = _rules.from_module(rules_path)
        logger.info("loaded %d rule(s) from %s: %s", len(ruleset), rules_path,
                    ", ".join(ruleset.names))
        rules_record = _rules.provenance(ruleset)
    if drop:
        from .rules import RuleSet

        (ruleset or RuleSet([])).dropping(drop)       # an unknown name fails before a fetch

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
    built = segprops.build(frame, rules=ruleset, label=label, drop=tuple(drop),
                           counts=counts, sizes=sizes, table=table is not None,
                           requested=bodies, **extra)

    written, linked = [], None
    if dry_run:
        logger.info("dry run: nothing written to %s", str(dst).rstrip("/"))
    else:
        written.append(segprops.write(dst, built["info"]))
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
           "numbers": built["report"]["numbers"], "strings": built["report"]["strings"],
           "label": built["report"]["label"],
           "drop": list(drop),
           # The rules decide what the document says; see `rules.provenance`.
           "rules": rules_record,
           "synapses": synapses, "roi_spec": roi_spec,
           "linked": linked, "table": table, "dry_run": True if dry_run else None}
    record = _record(bodies_source, dst, run=run)
    if not dry_run:
        _io.write_provenance(dst, record, name=f"{segprops.SUBDIR}/provenance")

    if table is not None:
        directory, name, fmt = _io.split_table_path(table)
        _io.write_table(built["table"], directory, name, fmt=fmt, metadata=record)
        _io.write_provenance(directory, record, name=f"{name}.provenance")

    return {**built, "written": written, "linked": linked,
            "record": None if dry_run else record, "dry_run": bool(dry_run)}


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
