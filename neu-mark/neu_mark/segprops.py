"""Building a neuroglancer ``segment_properties`` source from the body annotations.

The format is one inline JSON document, written through ``neu_vol.location``. What the
spec allows shapes it:

- At most one ``label``, one ``description`` and one ``tags`` property, so every facet
  pools into the one ``tags`` property as ``prefix:value`` (``side:L``, ``group:Mi1``).
- A segment's ``tags`` values are indices into the ``tags`` array, in increasing order.
- A tag may contain no spaces (they are hyphenated here) and no leading ``#``. Matching is
  case-insensitive, so values differing only in case show as one chip in the viewer
  (:func:`case_collisions` reports them); case is preserved.

Facets come from rules (:class:`neu_mark.rules.Rule`): the builtins in
:mod:`neu_mark.segprops_rules`, overridden by name by a dataset's rules module. The
``label`` defaults to the raw ``instance`` string; a reserved ``label`` rule replaces it.

The document is inline, so every viewer downloads all of it: ``max_tags`` and ``max_facets``
refuse a vocabulary large enough to make the layer unusable (:func:`resolve_tag_limit`).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)

#: The volume subdirectory holding the document (also its `info` key's value).
SUBDIR = "segment_properties"

AT_TYPE = "neuroglancer_segment_properties"

#: Joins a facet prefix to its value. It is what distinguishes a facet (`side:L`, which can
#: be grouped on) from a bare flag (`fragment`), and matches published datasets
#: (`superclass:ol_intrinsic`). A rule with ``prefix=""`` mints a bare flag.
SEPARATOR = ":"

#: Cap on the tag vocabulary: a fraction of the segment count below 1, an absolute count at 1
#: or above (:func:`resolve_tag_limit`). A fraction, because a reasonable size depends on
#: the dataset (11,752 types over ~165k bodies is 7%).
DEFAULT_MAX_TAGS = 0.10

#: Floor under a fractional limit only (10% of 40 bodies is too small to mean anything).
MIN_TAG_ALLOWANCE = 100

#: Cap on distinct facets (`side:`, `group:`, ..., plus one for bare flags), which catches
#: rules minting a separate facet per value.
MAX_FACETS = 24


def normalize_tag(value: Any) -> str | None:
    """A tag as the format requires: spaces hyphenated, no leading ``#``, case preserved.
    Missing values (including ``pd.NA``) give ``None``, via :func:`neu_mark.explore.normalize`.
    """
    from .explore import normalize

    text = normalize(value)
    if text is None:
        return None
    text = text.lstrip("#").strip()
    if not text:
        return None
    return re.sub(r"\s+", "-", text)


def case_collisions(vocabulary: Iterable[str]) -> dict[str, list[str]]:
    """``{lowercased: [tags]}`` for tags that differ only in case.

    The viewer matches tags case-insensitively, so ``group:Tm2`` and ``group:TM2`` show as
    one chip while remaining two values in the table.
    """
    seen: dict[str, list[str]] = {}
    for tag in vocabulary:
        seen.setdefault(tag.lower(), []).append(tag)
    return {low: tags for low, tags in seen.items() if len(tags) > 1}


def tag_for(rule, value: Any) -> str | None:
    """One rule's value as a tag: ``prefix:value``, or a bare tag where prefix is empty.

    A trailing colon on the prefix is tolerated (``prefix="col:"`` gives ``col:C2``).
    """
    prefix = getattr(rule, "prefix", "") or ""
    prefix = prefix.rstrip(SEPARATOR)
    text = f"{prefix}{SEPARATOR}{value}" if prefix else str(value)
    return normalize_tag(text)


def facets(record: Mapping[str, Any], ruleset=None) -> dict[str, list[str]]:
    """Every facet for one body, as ``{rule name: [tags]}``, already tag-normalized.

    Uses the built-in rules unless given a :class:`~neu_mark.rules.RuleSet`, used as is.
    Drop rules are not consulted; exclusion happens in :func:`build`.
    """
    from .explore import _plain
    from .rules import RESERVED

    if ruleset is None:
        from .segprops_rules import BUILTIN

        ruleset = BUILTIN
    row = _plain(_as_series(record))
    out: dict[str, list[str]] = {}
    for r in ruleset.tagged:
        if r.name in RESERVED:
            continue
        tags = [tag_for(r, v) for v in r.values(row)]
        kept = [t for t in dict.fromkeys(tags) if t]
        if kept:
            out[r.name] = kept
    return out


def _as_series(record: Mapping[str, Any]):
    import pandas as pd

    return record if isinstance(record, pd.Series) else pd.Series(dict(record),
                                                                  dtype="object")


def _resolve_rules(rules=None, *, only: bool = False, keep_glia: bool = True):
    """The rules to run: the builtins, the builtins overridden by name by ``rules``, or
    (``only=True``) ``rules`` alone."""
    from .rules import Rule, RuleSet
    from .segprops_rules import BUILTIN, DROP_GLIA

    if rules is None:
        resolved = BUILTIN
    elif only:
        resolved = rules if isinstance(rules, RuleSet) else RuleSet(rules)
    else:
        resolved = BUILTIN.override(rules)
    if not keep_glia:
        # Turn whichever rule is currently named `glia` (possibly a dataset's own) into a
        # drop rule, so --drop-glia uses the same definition of glia as the tags.
        current = next((r for r in resolved if r.name == "glia"), None)
        dropper = DROP_GLIA if current is None else Rule(
            current.func, name="glia", drop=True, needs=current.needs,
            description=f"{current.description} (excluded)")
        resolved = resolved.override([dropper])
    return resolved


def build(bodies, *, rules=None, rules_only: bool = False, counts=None, sizes=None,
          keep_glia: bool = True, max_tags: float = DEFAULT_MAX_TAGS,
          max_facets: int = MAX_FACETS) -> dict[str, Any]:
    """The ``segment_properties`` info document, plus a report of what went into it.

    ``bodies`` is a frame with ``body`` and whatever property fields exist; ``counts`` an
    optional frame with ``body``/``pre``/``post``/``syn``; ``sizes`` an optional
    ``{body: voxels}``.

    ``rules`` is a :class:`~neu_mark.rules.RuleSet` (or anything it accepts) overriding the
    builtins by name; ``rules_only=True`` uses it alone. Evaluation order: drop rules,
    then tag rules, then the reserved ``label``/``description`` rules, then any other
    ``tag=False`` rule (reported, emits nothing).

    Returns ``{"info": …, "report": …}``. Check the report's coverage: a facet that fired on
    2% of bodies looks the same in the viewer as one the data does not have.
    """
    import json

    from .explore import _plain, normalize
    from .rules import RESERVED

    frame = bodies.copy()
    if "body" not in frame.columns:
        raise KeyError("bodies frame needs a 'body' column")

    ruleset = _resolve_rules(rules, only=rules_only, keep_glia=keep_glia)

    # A declared column absent from every record (e.g. `type` on a list where no body has
    # one) is filled with None and reported, rather than refused.
    absent = sorted({c for r in ruleset for c in r.needs} - set(frame.columns))
    for column in absent:
        frame[column] = None
    if absent:
        logger.warning("the body records carry no %s column, so the rule(s) needing it "
                       "can never fire", ", ".join(absent))

    named = {r.name: r for r in ruleset}
    label_rule = named.get("label")
    description_rule = named.get("description")
    tag_rules = [r for r in ruleset.tagged if r.name not in RESERVED]
    quiet_rules = [r for r in ruleset
                   if not r.tag and not r.drop and r.name not in RESERVED]

    fired = {r.name: 0 for r in ruleset}
    distinct: dict[str, set] = {r.name: set() for r in ruleset}
    described: dict[str, str] = {}
    excluded: dict[str, int] = {}
    rows = []

    for _index, raw in frame.iterrows():
        row = _plain(raw)

        reason = None
        for r in ruleset.drops:
            got = r.scalar(row)
            if got:
                reason = got if isinstance(got, str) else r.name
                fired[r.name] += 1
                distinct[r.name].add(reason)
                break
        if reason is not None:
            excluded[reason] = excluded.get(reason, 0) + 1
            continue

        tags: list[str] = []
        for r in tag_rules:
            minted = [t for t in dict.fromkeys(tag_for(r, v) for v in r.values(row)) if t]
            if not minted:
                continue
            fired[r.name] += 1
            distinct[r.name].update(minted)
            described.update(dict.fromkeys(minted, r.description))
            tags.extend(minted)

        for r in quiet_rules:
            values = r.values(row)
            if values:
                fired[r.name] += 1
                distinct[r.name].update(str(v) for v in values)

        # `normalize`, not `normalize_tag`: labels may contain spaces.
        label = normalize(label_rule.scalar(row)) if label_rule is not None else None
        if label is not None:
            fired["label"] += 1
        else:
            label = normalize(row.get("instance"))

        description = (normalize(description_rule.scalar(row))
                       if description_rule is not None else None)
        if description is not None:
            fired["description"] += 1

        rows.append((int(raw["body"]), label, description, tags))

    if not rows:
        raise ValueError("every body was excluded; nothing to write")

    ids = [str(body) for body, _l, _d, _t in rows]
    # A segment with no label is labelled with its id.
    labels = [label if label else str(body) for body, label, _d, _t in rows]

    vocabulary: list[str] = []
    index_of: dict[str, int] = {}
    per_body: list[list[int]] = []
    for _body, _label, _desc, tags in rows:
        indices = []
        for tag in tags:
            if tag not in index_of:
                index_of[tag] = len(vocabulary)
                vocabulary.append(tag)
            indices.append(index_of[tag])
        # The spec requires the indices of one segment in increasing order.
        per_body.append(sorted(set(indices)))

    facet_names = sorted({t.split(SEPARATOR, 1)[0] if SEPARATOR in t else ""
                          for t in vocabulary})
    limit = check_vocabulary(vocabulary, distinct, segments=len(ids),
                             max_tags=max_tags, max_facets=max_facets,
                             facets=facet_names)

    properties: list[dict[str, Any]] = [
        {"id": "instance", "type": "label", "values": labels},
    ]
    if description_rule is not None:
        properties.append({"id": "description", "type": "description",
                           "values": [d or "" for _b, _l, d, _t in rows]})
    properties.append(
        {"id": "tags", "type": "tags", "tags": vocabulary,
         # Parallel to `tags`: the description of the rule that minted each.
         "tag_descriptions": [described.get(tag, tag) for tag in vocabulary],
         "values": per_body})

    numbers = _number_properties(ids, counts=counts, sizes=sizes)
    properties.extend(numbers)

    collisions = case_collisions(vocabulary)
    if collisions:
        logger.warning(
            "%d tag(s) differ only in case and will be ONE chip in the viewer: %s",
            len(collisions), sorted(v for vs in collisions.values() for v in vs)[:10])

    info = {"@type": AT_TYPE, "inline": {"ids": ids, "properties": properties}}
    report = {
        "bodies": len(ids),
        "excluded": excluded,
        "tags": len(vocabulary),
        # The resolved ceiling, and the setting it came from.
        "tag_limit": limit,
        "max_tags": max_tags,
        "facets": facet_names,
        # What every viewer downloads.
        "bytes": len(json.dumps(info)),
        "case_collisions": {low: tags for low, tags in sorted(collisions.items())},
        "coverage": _coverage_report(ruleset, fired, distinct, segments=len(ids),
                                     total=len(frame)),
        "rules": {"source": ruleset.source, "names": ruleset.names},
        "absent_columns": absent,
        "numbers": [p["id"] for p in numbers],
        "untagged": sum(1 for v in per_body if not v),
    }
    return {"info": info, "report": report}


def _coverage_report(ruleset, fired: Mapping[str, int], distinct: Mapping[str, set], *,
                     segments: int, total: int) -> dict[str, dict]:
    """Per rule: how many bodies it fired on, and how many distinct values it produced.

    A drop rule's fraction is over the input bodies; every other rule's is over the
    segments written.
    """
    out = {}
    for r in ruleset:
        n = fired.get(r.name, 0)
        denominator = total if r.drop else segments
        kind = "drop" if r.drop else "tag" if r.tag else "property"
        out[r.name] = {"bodies": n,
                       "fraction": (n / denominator) if denominator else 0.0,
                       "distinct": len(distinct.get(r.name, ())),
                       "kind": kind,
                       "description": r.description}
    return out


def resolve_tag_limit(max_tags: float, segments: int) -> tuple[int, str]:
    """``max_tags`` as a concrete tag ceiling: ``(limit, how)``.

    Below 1 is a fraction of the segment count, 1 or above an absolute count, decided by
    value (``1`` and ``1.0`` are the same to TOML and JSON). :data:`MIN_TAG_ALLOWANCE`
    floors a fraction only; ``max_tags=25`` means 25.
    """
    if not max_tags > 0:
        raise ValueError(
            f"max_tags must be positive; got {max_tags!r}. Below 1 is a fraction of the "
            f"segment count, 1 or above an absolute number of tags.")
    if max_tags < 1:
        return max(MIN_TAG_ALLOWANCE, int(max_tags * segments)), "fraction"
    return int(max_tags), "count"


def check_vocabulary(vocabulary: Sequence[str], distinct: Mapping[str, set], *,
                     segments: int, max_tags: float = DEFAULT_MAX_TAGS,
                     max_facets: int = MAX_FACETS,
                     facets: Sequence[str] | None = None) -> int:
    """Refuse a vocabulary big enough to make the layer unusable. Returns the limit applied.

    Two checks: too many tags (typically a rule echoing the whole instance string) and too
    many facets (a rule minting a separate facet per value). See :func:`resolve_tag_limit`.
    """
    if facets is None:
        facets = sorted({t.split(SEPARATOR, 1)[0] if SEPARATOR in t else ""
                         for t in vocabulary})
    if len(facets) > max_facets:
        raise ValueError(
            f"{len(facets)} distinct tag facets, over the {max_facets} allowed: "
            f"{', '.join(repr(f) for f in list(facets)[:12])}…. A facet per value means "
            f"nothing can be grouped on — check that the rules minting these meant to "
            f"declare prefix=\"\" for a bare flag rather than a namespace each.")

    limit, how = resolve_tag_limit(max_tags, segments)
    if len(vocabulary) > limit:
        worst, count = max(((n, len(v)) for n, v in distinct.items()),
                           key=lambda kv: kv[1], default=("", 0))

        applied = (f"{max_tags:.0%} of {segments}, floored at {MIN_TAG_ALLOWANCE}"
                   if how == "fraction" else f"an explicit count of {limit}")
        raise ValueError(
            f"{len(vocabulary)} tags for {segments} segments, over the limit of {limit} "
            f"({applied}). The rule producing most of them is {worst!r} with {count} "
            f"distinct values — if it is returning the whole instance string, that is the "
            f"bug; if the dataset really is this diverse, raise max_tags.")
    return limit


def _number_properties(ids: Sequence[str], *, counts=None, sizes=None) -> list[dict]:
    """`pre`/`post`/`syn`/`voxels` as uint32 number properties, for the ones supplied."""
    out: list[dict] = []
    order = [int(i) for i in ids]

    if counts is not None and len(counts):
        indexed = counts.set_index("body")
        for name, blurb in (("pre", "presynaptic sites (T-bars)"),
                            ("post", "postsynaptic sites (PSDs)"),
                            ("syn", "synapses in total, pre + post")):
            if name not in indexed.columns:
                continue
            series = indexed[name].reindex(order).fillna(0)
            out.append({"id": name, "type": "number", "data_type": "uint32",
                        "description": blurb,
                        "values": [int(v) for v in series]})

    if sizes:
        # uint32, not float32: float32 is exact only to 2^24, and bodies can exceed 86
        # million voxels.
        out.append({"id": "voxels", "type": "number", "data_type": "uint32",
                    "description": "size in voxels at full resolution",
                    "values": [int(sizes.get(b, 0)) for b in order]})
    return out


def write(dst: str, info: Mapping[str, Any], *, subdir: str = SUBDIR) -> str:
    """Write the document to ``<dst>/<subdir>/info`` through the kvstore."""
    from neu_vol.location import write_json

    write_json(dst, dict(info), subdir, "info")
    logger.info("wrote %s/%s/info (%d segments)", str(dst).rstrip("/"), subdir,
                len(info["inline"]["ids"]))
    return f"{subdir}/info"


def format_report(report: Mapping[str, Any]) -> list[str]:
    """The report as lines, for a CLI to print."""
    head = f"segments: {report['bodies']}   tags: {report['tags']}"
    if report.get("tag_limit"):
        head += f" of {report['tag_limit']} allowed"
    if report.get("bytes"):
        head += f"   document: {report['bytes'] / 1024:,.0f} KiB"
    lines = [head]
    if (report.get("rules") or {}).get("source"):
        lines.append(f"rules: {report['rules']['source']}")
    if report.get("absent_columns"):
        lines.append(f"absent from the records: {', '.join(report['absent_columns'])} — "
                     f"rules needing them cannot fire")
    if report.get("excluded"):
        detail = ", ".join(f"{k} ({v})" for k, v in sorted(report["excluded"].items()))
        lines.append(f"excluded: {sum(report['excluded'].values())} — {detail}")
    for name, stat in report.get("coverage", {}).items():
        kind = stat.get("kind", "tag")
        suffix = "" if kind == "tag" else f"   [{kind}]"
        lines.append(f"  {name:<13} {stat['bodies']:>7} bodies ({stat['fraction']:>6.1%})"
                     f"  {stat.get('distinct', 0):>5} distinct{suffix}")
    if report.get("numbers"):
        lines.append(f"  numbers: {', '.join(report['numbers'])}")
    if report.get("untagged"):
        lines.append(f"  {report['untagged']} segments carry no tag at all")
    return lines
