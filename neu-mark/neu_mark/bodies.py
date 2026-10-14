"""Reading the body list, which every command here requires.

DVID can list the elements in a body but cannot cheaply say which bodies are worth
asking about (most label ids are single-voxel fragments), so the caller supplies the set,
normally a curated list of traced neurons. Accepted forms:

- an iterable of ints;
- a path to ``.csv`` / ``.parquet`` / ``.feather``: a body-id column, or the only column;
- a path to anything else: one id per line, ``#`` comments allowed;
- otherwise an inline list, ``123,456,789``, comma or whitespace separated.

The tables this package writes are valid inputs.
"""

from __future__ import annotations

import os
import re
from typing import Iterable, Sequence

#: Column names tried, in order, when no column is named (DVID's, neuprint's and ours).
_BODY_COLUMNS = ("body", "bodyid", "body_id", "segment", "segment_id", "id")

_SEPARATORS = re.compile(r"[,\s]+")


def _from_text(text: str, *, where: str) -> list[int]:
    out: list[int] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        for token in _SEPARATORS.split(line):
            if not token:
                continue
            try:
                out.append(int(token))
            except ValueError:
                raise ValueError(
                    f"{where}: line {lineno} has {token!r}, which is not a body id. "
                    f"Expected integers, one per line or comma separated; "
                    f"'#' starts a comment.") from None
    return out


def _column(df, column: str | None, *, where: str):
    if column is not None:
        if column not in df.columns:
            raise ValueError(
                f"{where} has no column {column!r}; it has "
                f"{', '.join(map(str, df.columns))}")
        return df[column]
    for name in _BODY_COLUMNS:
        if name in df.columns:
            return df[name]
    if df.index.name in _BODY_COLUMNS:
        return df.index.to_series()
    if len(df.columns) == 1:
        return df[df.columns[0]]
    raise ValueError(
        f"{where} has several columns and none is named a body id "
        f"({', '.join(_BODY_COLUMNS)}). Name the column with --body-column.")


def load(source: str | Iterable[int], *, column: str | None = None) -> list[int]:
    """Resolve ``source`` to a de-duplicated, sorted list of body ids.

    Sorted, so output order depends only on the set requested; de-duplicated, so no body
    is fetched or written twice.
    """
    if not isinstance(source, str):
        ids = [int(b) for b in source]
        return sorted(set(ids))

    if os.path.exists(source):
        ext = os.path.splitext(source)[1].lower()
        if ext in (".parquet", ".pq", ".feather", ".arrow"):
            import pandas as pd

            df = (pd.read_parquet(source) if ext in (".parquet", ".pq")
                  else pd.read_feather(source))
            values = _column(df, column, where=source)
        elif ext == ".csv":
            import pandas as pd

            df = pd.read_csv(source)
            values = _column(df, column, where=source)
        else:
            with open(source) as fh:
                return sorted(set(_from_text(fh.read(), where=source)))
        ids = [int(v) for v in values.dropna()]
        if not ids:
            raise ValueError(f"{source} holds no body ids")
        return sorted(set(ids))

    # Not a path: an inline list (a single number is a one-body list).
    if any(ch.isdigit() for ch in source):
        return sorted(set(_from_text(source, where="--bodies")))
    raise FileNotFoundError(
        f"--bodies {source!r} is neither an existing file nor a list of body ids")


def summarise(bodies: Sequence[int]) -> str:
    """One line for the log: how many, and the range they span."""
    if not bodies:
        return "0 bodies"
    return (f"{len(bodies)} bodies ({min(bodies)}..{max(bodies)})"
            if len(bodies) > 1 else f"1 body ({bodies[0]})")
