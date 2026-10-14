"""Looking at annotation strings before writing rules to parse them.

Every function takes a DataFrame and returns one; nothing prints, writes or needs DVID, so
this is meant for a notebook: look at the vocabulary, write a rule, see what it missed,
repeat. (Named ``explore`` so it does not shadow the standard library's ``inspect``.)

``rules`` is a ``{name: callable}`` mapping. Each callable takes one row, a pandas Series,
and returns a scalar, a sequence (for multi-valued facets), or ``None`` / an empty sequence
when it did not fire. Read columns as ``r["instance"]``, not ``r.instance``: a column named
``name`` or ``size`` would return the Series attribute instead. A
:class:`~neu_mark.rules.Rule` is such a callable, and :meth:`RuleSet.coverage
<neu_mark.rules.RuleSet.coverage>` and :meth:`~neu_mark.rules.RuleSet.unparsed` wrap the
functions here.

Counts are by body, not by distinct string, so a vocabulary decision is weighted by how
much data it touches.
"""

from __future__ import annotations

import difflib
import re
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

#: A trailing '?' or '!' (an annotator's doubt, as in `truncated?`), removed with
#: surrounding whitespace before any vocabulary comparison.
_DOUBT = re.compile(r"[?!]+$")

#: Token separators in an instance string: ``_`` and parentheses (`LN_C5(L)_NCL`).
_SPLIT = re.compile(r"[_)(]+")


def normalize(value: Any) -> str | None:
    """Strip whitespace and trailing doubt marks; ``None`` for anything missing,
    including ``pd.NA`` (which would otherwise become the string ``"<NA>"``)."""
    if value is None or (pd.api.types.is_scalar(value) and pd.isna(value)):
        return None
    text = _DOUBT.sub("", str(value).strip()).strip()
    return text or None


def split_tokens(value: Any) -> list[str]:
    """One string's tokens, normalized, for use in rule bodies:
    ``"LN_C5(L)_NCL"`` -> ``['LN', 'C5', 'L', 'NCL']``; missing values give ``[]``."""
    return [t for t in _SPLIT.split(normalize(value) or "") if t]


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise KeyError(f"no {column!r} column; frame has "
                       f"{', '.join(map(str, frame.columns))}")
    return frame[column].map(normalize)


def instances(frame: pd.DataFrame, column: str = "instance") -> pd.DataFrame:
    """Distinct strings and how many bodies carry each, most common first. The first
    thing to look at: typically a few thousand strings, a reviewable vocabulary."""
    counts = _series(frame, column).value_counts(dropna=True)
    return (counts.rename("bodies").rename_axis(column).reset_index())


def tokens(frame: pd.DataFrame, column: str = "instance", *,
           position: str = "trailing", drop: str | None = None) -> pd.DataFrame:
    """Token frequency, for hunting controlled vocabulary and its misspellings.

    ``position`` is ``"trailing"`` (the last token, where completeness markers and flags
    live), ``"leading"``, or ``"all"``. A string with no separator is its own token.

    ``drop`` is a regex removed before tokenizing, for a facet already accounted for; since
    a parenthesized side makes ``L``/``R`` the trailing token of most strings,
    ``drop=r"\\((L|R)\\)"`` is the usual first move.

    Returns ``token``, ``bodies`` and ``strings`` (how many distinct strings contain it).
    """
    if position not in ("trailing", "leading", "all"):
        raise ValueError(f"position must be trailing, leading or all; got {position!r}")
    pattern = re.compile(drop) if drop else None

    body_counts: dict[str, int] = {}
    string_counts: dict[str, int] = {}
    for text, n in _series(frame, column).value_counts(dropna=True).items():
        if pattern is not None:
            text = pattern.sub("_", text)
        parts = [p for p in _SPLIT.split(text) if p]
        if not parts:
            continue
        picked = (parts[-1:] if position == "trailing"
                  else parts[:1] if position == "leading" else parts)
        for token in dict.fromkeys(picked):          # de-dup within one string
            body_counts[token] = body_counts.get(token, 0) + int(n)
            string_counts[token] = string_counts.get(token, 0) + 1

    out = pd.DataFrame({"token": list(body_counts),
                        "bodies": list(body_counts.values()),
                        "strings": [string_counts[t] for t in body_counts]})
    return out.sort_values(["bodies", "token"], ascending=[False, True],
                          ignore_index=True)


def near(token_frame: pd.DataFrame, target: str, *, cutoff: float = 0.72,
         include_exact: bool = False) -> pd.DataFrame:
    """Tokens close enough to ``target`` to be misspellings of it, with their weight.

    Uses ``difflib`` similarity, since real misspellings cannot be listed in advance.
    """
    rows = []
    for token, bodies, strings in token_frame[["token", "bodies", "strings"]].itertuples(
            index=False):
        if not include_exact and str(token).lower() == target.lower():
            continue
        ratio = difflib.SequenceMatcher(None, str(token).lower(), target.lower()).ratio()
        if ratio >= cutoff:
            rows.append({"token": token, "bodies": bodies, "strings": strings,
                         "ratio": round(ratio, 3)})
    return pd.DataFrame(rows, columns=["token", "bodies", "strings", "ratio"]).sort_values(
        ["bodies", "ratio"], ascending=False, ignore_index=True)


def variants(frame: pd.DataFrame, column: str = "instance") -> pd.DataFrame:
    """Raw strings that :func:`normalize` repaired, and what they became.

    Whitespace and trailing ``?`` are repaired before :func:`tokens` and :func:`near` see
    a string, so this is where they show up. A trailing ``?`` is an annotator's doubt,
    worth watching. :func:`near` finds genuine misspellings (``fragmnet``).

    Returns ``raw``, ``normalized``, ``bodies`` for every string that changed.
    """
    if column not in frame.columns:
        raise KeyError(f"no {column!r} column")
    rows = []
    for raw, n in frame[column].value_counts(dropna=True).items():
        clean = normalize(raw)
        if clean != raw:
            rows.append({"raw": raw, "normalized": clean, "bodies": int(n)})
    return pd.DataFrame(rows, columns=["raw", "normalized", "bodies"]).sort_values(
        "bodies", ascending=False, ignore_index=True)


def _plain(row: pd.Series) -> pd.Series:
    """A row with pandas' missing values replaced by ``None``, so a rule can write
    ``r["instance"] or ""`` (``pd.NA or ""`` raises)."""
    return row.where(row.notna(), None)


def _as_tuple(value: Any) -> tuple:
    """A rule's return value as a tuple of values, empty when it did not fire."""
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        # Scalars, including numpy scalars and bools. NaN counts as "did not fire".
        if isinstance(value, float) and pd.isna(value):
            return ()
        return (value,)
    return tuple(v for v in value if v is not None)


def apply_rules(frame: pd.DataFrame, rules: Mapping[str, Callable],
                *, key: str = "body") -> pd.DataFrame:
    """Run every rule over every row. One column per rule, values as tuples.

    An empty tuple means the rule did not fire. A rule that raises is an error: it should
    return ``None`` for a value it cannot parse.
    """
    rows = [_plain(row) for _index, row in frame.iterrows()]
    out = {}
    if key in frame.columns:
        out[key] = frame[key].to_numpy()
    for name, rule in rules.items():
        try:
            out[name] = [_as_tuple(rule(row)) for row in rows]
        except Exception as exc:                                    # noqa: BLE001
            raise RuntimeError(
                f"rule {name!r} raised on some row: {type(exc).__name__}: {exc}. A rule "
                f"that cannot parse a value should return None, not raise — the whole "
                f"point is that unparsed values stay visible.") from exc
    # A positional index, so a caller's own index cannot misalign these against `frame`.
    return pd.DataFrame(out).reset_index(drop=True)


def coverage(frame: pd.DataFrame, rules: Mapping[str, Callable],
             *, top: int = 3) -> pd.DataFrame:
    """Per-rule: how often it fired, how many distinct values, and the commonest.

    A rule that under-fires on a messy vocabulary looks the same as data that lacks the
    facet; coverage is how to tell.
    """
    applied = apply_rules(frame, rules)
    total = len(frame)
    rows = []
    for name in rules:
        col = applied[name]
        fired = col.map(bool)
        values = [v for tup in col for v in tup]
        counts = pd.Series(values, dtype="object").value_counts() if values else pd.Series(
            dtype="int64")
        rows.append({
            "rule": name,
            "bodies": int(fired.sum()),
            "coverage": (float(fired.sum()) / total) if total else None,
            "values": len(values),
            "distinct": int(counts.size),
            "multi_valued": int((col.map(len) > 1).sum()),
            "top": ", ".join(f"{v} ({n})" for v, n in counts.head(top).items()),
        })
    return pd.DataFrame(rows)


def unparsed(frame: pd.DataFrame, rules: Mapping[str, Callable], rule: str, *,
             column: str = "instance") -> pd.DataFrame:
    """The strings a given rule did not fire on, ranked by how many bodies they cover.

    The top of this list is the next rule to write, or the missing vocabulary entry.
    """
    if rule not in rules:
        raise KeyError(f"no rule named {rule!r}; have {', '.join(rules)}")
    applied = apply_rules(frame, {rule: rules[rule]})
    missed = frame[~applied[rule].map(bool).to_numpy()]
    return instances(missed, column) if len(missed) else pd.DataFrame(
        columns=[column, "bodies"])


#: How a derived value relates to a curated field, from agreement to conflict.
#: `derived_is_longer` means the parse found something the curated field lacks.
RELATIONS = ("exact", "case_only", "derived_is_longer", "field_is_longer",
             "substring", "unrelated", "field_missing", "derived_missing")


def _relation(field: str | None, derived: str | None) -> str:
    if field is None and derived is None:
        return "field_missing"
    if field is None:
        return "field_missing"
    if derived is None:
        return "derived_missing"
    if field == derived:
        return "exact"
    if field.lower() == derived.lower():
        return "case_only"
    if derived.startswith(field + "_") or derived.startswith(field + "("):
        return "derived_is_longer"
    if field.startswith(derived + "_") or field.startswith(derived + "("):
        return "field_is_longer"
    if field in derived or derived in field:
        return "substring"
    return "unrelated"


def compare(frame: pd.DataFrame, field: str, against: Callable, *,
            key: str = "body", column: str = "instance") -> pd.DataFrame:
    """Row-by-row: how a curated field relates to what a rule derives.

    One row per body with a ``relation`` from :data:`RELATIONS`; summarise with
    ``.relation.value_counts()``. A disagreement is not necessarily an error: ``type`` is
    often the coarser label (``LMC`` for an instance of ``L1/L3_D4(R)``).
    """
    rows = []
    for _index, raw_row in frame.iterrows():
        row = _plain(raw_row)
        got = _as_tuple(against(row))
        one = normalize(got[0]) if got else None
        raw_field = normalize(row.get(field))
        rows.append({
            key: row.get(key),
            field: raw_field,
            "derived": one,
            column: row.get(column),
            "relation": _relation(raw_field, one),
        })
    return pd.DataFrame(rows, columns=[key, field, "derived", column, "relation"])
