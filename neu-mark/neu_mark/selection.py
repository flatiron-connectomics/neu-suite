"""Which bodies to select, beyond a synapse count: connectivity and names.

``select-bodies`` ranks bodies by their synapse count (T-bars plus PSDs, from ``labelsz``).
Two more criteria can each admit a body the count missed:

- **pairs**: the T-bar -> PSD connections a body takes part in, at either end. A T-bar has
  several PSDs, so a body with a few T-bars can be highly connected while holding few
  synapses. Counted from the ``connections`` table ``neu-mark points --connections`` writes.
- **a name**: the body's annotation record gets a value from a chosen rule of a rules
  module, and no drop rule excludes it. Which names count is the rules module's decision.

Rules can also take bodies OUT of the list, whichever criterion admitted them
(:func:`excluded_bodies`).

Pure functions over tables; the fetching is in :mod:`neu_mark.ops`.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .tables import BODY_DTYPE


def pair_counts(connections: pd.DataFrame) -> pd.Series:
    """``{body: pairs}``: the connections each body takes part in, as pre or post.

    A connection from a body to itself counts twice, once at each end. Body 0 (an end with
    no body) is left out.
    """
    ends = pd.concat([connections["pre_body"], connections["post_body"]],
                     ignore_index=True)
    ends = ends[ends.notna()]
    ends = ends.astype(BODY_DTYPE)
    counts = ends[ends != 0].value_counts()
    counts.index = counts.index.astype(BODY_DTYPE)
    return counts.rename_axis("body").rename("pairs").astype("int64")


def _rows(records: pd.DataFrame, used):
    """Each record as the plain dict a rule reads, with every field ``used`` needs: one no
    record has is treated as missing, as :func:`neu_mark.segprops.build` does."""
    from .explore import _plain

    frame = records.copy()
    for column in sorted({c for r in used for c in r.needs} - set(frame.columns)):
        frame[column] = None
    for _index, raw in frame.iterrows():
        yield _plain(raw)


def _by_body(values: dict, name: str) -> pd.Series:
    series = pd.Series(values, dtype="object", name=name)
    series.index = series.index.astype(BODY_DTYPE)
    return series.rename_axis("body")


def named_bodies(records: pd.DataFrame, ruleset, rule: str) -> pd.Series:
    """``{body: value}`` for each record ``rule`` gives a value and no drop rule excludes.

    ``records`` has a ``body`` column and the fields the rules read
    (:func:`neu_mark.tables.keyvalues_to_frame`).
    """
    chosen = ruleset[rule]
    if chosen.drop:
        raise ValueError(f"rule {rule!r} is a drop rule: it excludes bodies, so it cannot "
                         f"be what names one")
    out: dict[int, Any] = {}
    for row in _rows(records, [*ruleset.drops, chosen]):
        if any(r.scalar(row) for r in ruleset.drops):
            continue
        value = chosen.scalar(row)
        if value:
            out[int(row["body"])] = value
    return _by_body(out, rule)


def excluded_bodies(records: pd.DataFrame, ruleset, rules) -> pd.Series:
    """``{body: rule}`` for each record one of ``rules`` gives a value, naming the first.

    Any rule may exclude, a drop rule or a tag (``glia``); unlike :func:`named_bodies`, the
    module's other drop rules play no part.
    """
    chosen = [ruleset[name] for name in rules]
    out: dict[int, Any] = {}
    for row in _rows(records, chosen):
        hit = next((r.name for r in chosen if r.scalar(row)), None)
        if hit:
            out[int(row["body"])] = hit
    return _by_body(out, "excluded")


def categories(records: pd.DataFrame, rule) -> pd.Series:
    """``{body: category}``: what ``rule`` gives each record (``select-bodies --criteria``).

    Every record must get one; a rule giving none is an error naming a few of them.
    """
    out: dict[int, Any] = {}
    missing = []
    for row in _rows(records, [rule]):
        value = rule.scalar(row)
        if value is None:
            missing.append(row.get(rule.label) if rule.label else row["body"])
        out[int(row["body"])] = value
    if missing:
        raise ValueError(f"the rule {rule.name!r} gives no category for {len(missing)} "
                         f"records, e.g. {missing[:5]!r}: it must give every record one")
    return _by_body(out, "category")


def reported(records: pd.DataFrame, rule) -> pd.DataFrame:
    """What ``rule`` gives each record it gives a value: ``body``, the label column (when
    the rule reads one) and the rule's value, one row per record, sorted by body."""
    rows = []
    for row in _rows(records, [rule]):
        value = rule.scalar(row)
        if value is not None:
            rows.append({"body": int(row["body"]),
                         **({rule.label: row.get(rule.label)} if rule.label else {}),
                         rule.name: value})
    columns = ["body", *([rule.label] if rule.label else []), rule.name]
    out = pd.DataFrame(rows, columns=columns).sort_values("body", ignore_index=True)
    out["body"] = out["body"].astype(BODY_DTYPE)
    return out


def admitted(counted: pd.DataFrame, *, pairs: pd.Series | None = None,
             min_pairs: int | None = None, named: pd.Series | None = None) -> dict:
    """The bodies the extra criteria admit on top of ``counted`` (the synapse-count list).

    Returns ``{"by_pairs": [...], "by_name": [...], "extra": [...]}``: bodies with at least
    ``min_pairs`` pairs, bodies ``named`` lists, and those of either not in ``counted``,
    each sorted.
    """
    have = set(int(b) for b in counted["body"])
    by_pairs = (sorted(int(b) for b in pairs.index[pairs >= int(min_pairs)])
                if pairs is not None and min_pairs is not None else [])
    by_name = sorted(int(b) for b in named.index) if named is not None else []
    extra = sorted((set(by_pairs) | set(by_name)) - have)
    return {"by_pairs": by_pairs, "by_name": by_name, "extra": extra}
