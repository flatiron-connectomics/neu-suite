"""Turn a measure DB into tidy tables, for cohort selection and comparison.

The notebook-facing half of `measure`: DataFrame in, DataFrame out, no printing or
plotting. pandas and pyarrow (``neu-morpho[measure]``) are imported inside functions.

Variants: ``all`` is the whole body, ``minus_nucleus`` removes the nucleus,
``minus_soma_nucleus`` the whole cell body. **Which matches depends on the other cohort**:
removing a compartment from one side that is present on the other is a silent bias.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence

#: Compartment names of one semantic source. Pass a dataset's own mapping
#: (`compartments.semantic_label_names` reads it) rather than assuming this one.
DEFAULT_COMPARTMENTS = {0: "unlabelled", 1: "neuropil", 2: "fiber_bundle", 3: "nucleus",
                        4: "glia", 5: "soma", 6: "trachea", 7: "muscle",
                        8: "do_not_merge"}

#: variant -> the compartments subtracted from the total.
VARIANTS = {"all": (),
            "minus_nucleus": ("nucleus",),
            "minus_soma_nucleus": ("nucleus", "soma")}


def load_bodies(db_path: str, *, voxel_nm: float = 32.0,
                dataset: Optional[str] = None,
                compartments: Optional[Mapping[int, str]] = None):
    """One row per body, with volume, cable, per-compartment volume and the variants.

    Opens the DB read-only, so it is safe against a run in flight. Bodies with no voxel
    count were never measured and are dropped.
    """
    import pandas as pd

    from ..metrics_db import MetricsDB

    names = dict(DEFAULT_COMPARTMENTS if compartments is None else compartments)
    db = MetricsDB(db_path, read_only=True)
    try:
        bodies = pd.read_sql_query(
            "SELECT body_id, voxel_count, volume_nm3, cable_length_nm, n_tips, "
            "n_branches, max_radius_nm FROM bodies WHERE voxel_count > 0", db.con)
        has_comp = db._has_table("body_compartments")
        comp = (pd.read_sql_query(
            "SELECT body_id, label, voxel_count FROM body_compartments", db.con)
            if has_comp else None)
    finally:
        db.close()

    vox = float(voxel_nm) ** 3
    out = bodies.copy()
    out["volume_um3"] = out["volume_nm3"] / 1e9
    out["cable_um"] = out["cable_length_nm"] / 1e3

    if comp is not None and len(comp):
        wide = (comp.pivot_table(index="body_id", columns="label",
                                 values="voxel_count", aggfunc="sum", fill_value=0)
                .rename(columns=lambda c: f"{names.get(int(c), f'label_{int(c)}')}_um3"))
        wide = wide * vox / 1e9
        out = out.merge(wide, on="body_id", how="left")
        for col in wide.columns:
            out[col] = out[col].fillna(0.0)

    for variant, drop in VARIANTS.items():
        cols = [f"{d}_um3" for d in drop if f"{d}_um3" in out.columns]
        if len(cols) != len(drop):
            continue                        # that compartment was not measured
        vol = out["volume_um3"] - (out[cols].sum(axis=1) if cols else 0.0)
        out[f"volume_um3_{variant}"] = vol
        out[f"diameter_nm_{variant}"] = _diameter_nm(vol, out["cable_um"])

    if dataset is not None:
        out.insert(0, "dataset", str(dataset))
    return out


def _diameter_nm(volume_um3, cable_um):
    """Area-equivalent diameter from V/L. NaN where there is no cable, never inf.

    The minus-variants divide a reduced volume by the full cable length, soma included,
    so they under-read neurite calibre.
    """
    import numpy as np
    import pandas as pd

    vol_nm3 = pd.Series(volume_um3).astype(float) * 1e9
    cable_nm = pd.Series(cable_um).astype(float) * 1e3
    area = vol_nm3.where(cable_nm > 0) / cable_nm.where(cable_nm > 0)
    return 2.0 * np.sqrt(area / np.pi)


def _slug(tag: str) -> str:
    """A tag turned into a column-name-safe suffix."""
    import re

    return re.sub(r"[^0-9a-zA-Z]+", "_", tag).strip("_").lower() or "tag"


def load_segment_properties(base: str, *sources: str):
    """Wide table of every inline property across one or more properties sources.

    ``tags`` become a boolean column per bare tag and, for ``namespace:value`` tags, one
    column per namespace. A namespace with several values for one body keeps the first,
    and the whole tuple goes in ``<namespace>_all``.
    """
    import pandas as pd

    from neu_vol import location
    from neu_vol.logs import quiet_reads

    frames = []
    for sub in sources:
        # A notebook entry point, so it filters the store's benign credential logging
        # itself; `neu_vol.logs.reads_quiet = False` opts out.
        with quiet_reads():
            inline = location.read_json(base, sub, "info")["inline"]
        ids = [int(i) for i in inline["ids"]]
        # Columns are accumulated and the frame built once; per-column assignment
        # fragments the frame and warns per tag.
        cols: dict[str, Any] = {"body_id": ids}
        for prop in inline["properties"]:
            kind, pid = prop.get("type"), prop.get("id", "value")
            if kind == "tags":
                vocab = prop["tags"]
                rows = [[vocab[j] for j in row] for row in prop["values"]]
                # Two passes, so a namespace first seen on a later row still gets a value
                # for every row.
                parsed, bare = [], []
                for row in rows:
                    seen: dict[str, list[str]] = {}
                    plain: list[str] = []
                    for tag in row:
                        ns, sep, val = tag.partition(":")
                        if sep:
                            seen.setdefault(ns, []).append(val)
                        else:
                            plain.append(tag)
                    parsed.append(seen)
                    bare.append(plain)
                for ns in sorted({k for s in parsed for k in s}):
                    col = [tuple(s.get(ns, ())) for s in parsed]
                    cols[ns] = [v[0] if v else None for v in col]
                    if any(len(v) > 1 for v in col):
                        # Tuples, not lists, so the column stays hashable for groupby.
                        cols[f"{ns}_all"] = col
                if any(bare):
                    # Bare tags are not mutually exclusive: one boolean column per tag.
                    cols[f"{pid}_all"] = [tuple(b) for b in bare]
                    for tag in sorted({t for b in bare for t in b}):
                        cols[f"{pid}_{_slug(tag)}"] = [tag in b for b in bare]
            else:
                cols[pid] = prop["values"]
        frames.append(pd.DataFrame(cols))

    if not frames:
        raise ValueError("no properties sources given")
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="body_id", how="outer")
    return out


def merge_properties(bodies, props, *, how: str = "left"):
    """Join properties onto a bodies table, keeping boolean columns BOOLEAN.

    Use this rather than ``bodies.merge(props, on='body_id')``. A plain merge widens a
    ``bool`` column to ``object`` where a body has no properties row, and ``~`` on that
    column is arithmetic (``~True`` is ``-2``), so ``~col.fillna(False)`` silently masks
    nothing. A body with no properties row gets ``False`` for every flag.
    """
    import pandas as pd

    flags = [c for c in props.columns
             if c != "body_id" and pd.api.types.is_bool_dtype(props[c])]
    out = bodies.merge(props, on="body_id", how=how)
    for col in flags:
        out[col] = out[col].fillna(False).astype(bool)
    return out


def summarize(df, columns: Sequence[str], *, by: Optional[str] = None,
              quantiles: Sequence[float] = (0.1, 0.5, 0.9)):
    """Percentile table for ``columns``, optionally grouped by a column.

    Percentiles rather than mean and sd: these distributions are heavy-tailed.
    """
    import pandas as pd

    cols = [c for c in columns if c in df.columns]
    if by is None:
        rows = {"n": len(df)}
        for c in cols:
            for q in quantiles:
                rows[f"{c}_p{int(q * 100)}"] = df[c].quantile(q)
        return pd.DataFrame([rows])
    out = []
    for key, part in df.groupby(by, dropna=False):
        row = {by: key, "n": len(part)}
        for c in cols:
            for q in quantiles:
                row[f"{c}_p{int(q * 100)}"] = part[c].quantile(q)
        out.append(row)
    return pd.DataFrame(out).sort_values("n", ascending=False).reset_index(drop=True)


def compare(left, right, columns: Sequence[str], *,
            labels: Sequence[str] = ("left", "right"),
            statistic: str = "median"):
    """One row per column: each cohort's statistic and the ratio between them.

    The ratio is of the statistics: the cohorts are unpaired.
    """
    import pandas as pd

    def stat(frame, col):
        series = frame[col].dropna()
        if not len(series):
            return float("nan")
        return series.median() if statistic == "median" else getattr(series, statistic)()

    rows = []
    for c in columns:
        if c not in left.columns or c not in right.columns:
            continue
        a, b = stat(left, c), stat(right, c)
        rows.append({"column": c, labels[0]: a, labels[1]: b,
                     "ratio": (a / b) if b else float("nan")})
    return pd.DataFrame(rows)


def log_histogram(df, column: str, *, lo: float, hi: float, nbins: int = 48,
                  weight: Optional[str] = None):
    """Log-spaced histogram as a DataFrame, with under/overflow rows.

    The catch-alls keep the weight total exact; a fat overflow row means the range is
    wrong.
    """
    import numpy as np
    import pandas as pd

    from .sweep import log_bin_edges, weighted_histogram

    edges = log_bin_edges(lo, hi, nbins)
    values = df[column].to_numpy(dtype=float)
    weights = (np.ones(len(values)) if weight is None
               else df[weight].to_numpy(dtype=float))
    good = np.isfinite(values) & np.isfinite(weights)
    counts = weighted_histogram(values[good], weights[good], edges)
    lows = np.concatenate([[-np.inf], edges])
    highs = np.concatenate([edges, [np.inf]])
    return pd.DataFrame({"lo": lows, "hi": highs, "weight": counts})


def write_table(df, path: str) -> str:
    """Write parquet, which unlike csv round-trips a nullable integer column."""
    df.to_parquet(path, index=False)
    return path
