"""Fetches that return DataFrames and write nothing. The interactive entry point.

:mod:`neu_mark.ops` runs the same fetches (both call :mod:`neu_mark.dvid`) and writes
tables and a provenance record. Everything here is re-exported at the package top level:

    >>> from neu_mark import source, select_bodies, points, body_annotations
    >>> src   = source("dvid://dvid.example.org/93fdbc:main/synapses",
    ...               locked=True)
    >>> sel   = select_bodies(src, min_synapses=10)      # DataFrame: body, pre, post, syn
    >>> pts, rels = points(src, sel.head(50))           # DataFrames; nothing written
    >>> ann   = body_annotations("@bodies", sel.head(50))

A source is a ``dvid://`` URL, an ``@name`` config reference (:mod:`neu_mark.config`), an
already-opened source dict, or a bare instance name when a config supplies the server and
uuid.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from . import dvid as _dvid
from . import ops as _ops

#: Which `open_*_source` to use for each kind of read.
_OPENERS = {
    "points": _ops.open_points_source,
    "bodies": _ops.open_bodies_source,
    "counts": _ops.open_counts_source,
}


def _url(location: str, *, config=None) -> str:
    """Resolve an ``@name`` reference or a bare instance name into a full DVID URL."""
    from neu_vol.dvid import is_url

    from . import config as _config

    if is_url(location):
        return location
    cfg = config if config is not None else _config.load()
    if location.startswith(_config.REFERENCE_PREFIX):
        return cfg.resolve(location)
    # A bare instance name needs a config for the server and uuid.
    if not cfg.server:
        raise ValueError(
            f"{location!r} is not a dvid:// URL, and no config was found to build one "
            f"from. Either pass the full URL, or create a config — see "
            f"neu_mark.config for the search path and format.")
    return cfg.url(location)


def source(location: str | Mapping[str, Any], *, kind: str = "points",
           locked: bool = False, config=None) -> dict[str, Any]:
    """Resolve a location to an opened source: concrete uuid, validated instance type.

    ``kind`` is ``"points"`` (an annotation instance), ``"bodies"`` (keyvalue) or
    ``"counts"`` (a labelsz index, or the annotation instance it indexes). The node is
    pinned once, so later calls read the same version even if the branch moves.
    """
    if kind not in _OPENERS:
        raise ValueError(f"kind must be one of {', '.join(_OPENERS)}; got {kind!r}")
    if isinstance(location, Mapping):
        # A spec or opened source; re-opening is cheap and validates the instance type.
        spec = dict(location)
    else:
        from neu_vol.dvid import parse_url

        spec = {"backend": "dvid", **parse_url(_url(location, config=config))}
    return _OPENERS[kind](spec, prefer_locked=locked)


def _as_source(location, kind: str, locked: bool, config) -> dict[str, Any]:
    if isinstance(location, Mapping) and location.get("node") is not None:
        return dict(location)                     # already opened; do not re-resolve
    return source(location, kind=kind, locked=locked, config=config)


def body_ids(bodies) -> list[int]:
    """Body ids from a DataFrame (its ``body``, ``bodyid`` or ``body_id`` column), a Series,
    an iterable, a body-list file path, or a single id."""
    import pandas as pd

    from . import bodies as _bodies

    if isinstance(bodies, pd.DataFrame):
        for name in ("body", "bodyid", "body_id"):
            if name in bodies.columns:
                return [int(v) for v in bodies[name]]
        raise KeyError(
            f"no body column in the frame (has {', '.join(map(str, bodies.columns))})")
    if isinstance(bodies, pd.Series):
        return [int(v) for v in bodies.dropna()]
    if isinstance(bodies, str):
        return _bodies.load(bodies)
    if isinstance(bodies, Iterable):
        return [int(v) for v in bodies]
    return [int(bodies)]


def select_bodies(location="counts", *, min_synapses: int = 10, min_pre: int = 0,
                  min_post: int = 0, limit: int | None = None, locked: bool = False,
                  config=None):
    """Bodies ranked by synapse count. Returns ``body``, ``pre``, ``post``, ``syn``.

    The query behind ``neu-mark select-bodies``, without writing anything.
    """
    src = _as_source(location, "counts", locked, config)
    return _dvid.fetch_synapse_counts(src, min_total=min_synapses, min_pre=min_pre,
                                      min_post=min_post, limit=limit)


def points(location, bodies, *, threads: int = _dvid.DEFAULT_THREADS,
           locked: bool = False, config=None, connections: bool = False,
           rois: Sequence[str] | str | None = None,
           on_roi_overlap: str = "warn"):
    """Point annotations for ``bodies``. Returns ``(points, relationships)``.

    With ``connections=True`` returns ``(points, relationships, connections)``, the last
    being de-duplicated ``(tbar, psd)`` pairs; check how much of it resolved with
    :func:`neu_mark.tables.match_rate`.

    ``rois`` (ROI names, or ``"@name"`` for a set in the config) adds a ``roi`` column to the
    points frame only, since a relationship's two ends may be in different neuropils.
    :func:`neu_mark.tables.body_roi_counts` aggregates it per body.
    """
    src = _as_source(location, "points", locked, config)
    result = _dvid.fetch_points(src, body_ids(bodies), threads=threads)
    if rois:
        labelled = _dvid.label_point_rois(
            src, result["points"], roi_set(rois, config=config),
            on_overlap=on_roi_overlap)
        result["points"] = labelled["points"]
        result["rois"] = labelled
    if connections:
        return result["points"], result["relationships"], result["connections"]
    return result["points"], result["relationships"]


def roi_set(rois: Sequence[str] | str, *, config=None) -> list[str]:
    """An ROI name list from a list, a comma-separated string, or an ``@name`` config set."""
    from . import config as _config

    if isinstance(rois, str):
        if rois.startswith(_config.REFERENCE_PREFIX):
            cfg = config if config is not None else _config.load()
            return cfg.roi_set(rois[len(_config.REFERENCE_PREFIX):])
        return [r.strip() for r in rois.split(",") if r.strip()]
    return [str(r).strip() for r in rois if str(r).strip()]


def body_annotations(location="bodies", bodies=None, *, locked: bool = False, config=None):
    """The per-body property records for ``bodies``, as one row per body."""
    if bodies is None:
        raise ValueError(
            "body_annotations needs a body list: DVID cannot cheaply enumerate the bodies "
            "worth asking about. Get one from select_bodies().")
    src = _as_source(location, "bodies", locked, config)
    return _dvid.fetch_body_annotations(src, body_ids(bodies))["bodies"]


def all_body_annotations(location="bodies", *, locked: bool = False, config=None):
    """Every record in the property keyvalue instance, as one row per body.

    A different population from a synapse-selected body list: most glia, for instance, sit
    below any synapse threshold. About a minute for ~58,000 records (see
    :func:`neu_mark.dvid.fetch_all_body_annotations`).
    """
    src = _as_source(location, "bodies", locked, config)
    from . import tables as _tables

    return _tables.keyvalues_to_frame(
        _dvid.fetch_all_body_annotations(src)["records"])


def synapse_counts(location="counts", bodies=None, *, locked: bool = False, config=None):
    """Exact ``PreSyn``/``PostSyn`` counts for specific bodies, ignoring any threshold.

    :func:`select_bodies` finds the bodies above a threshold; this counts a given list.
    """
    import pandas as pd

    if bodies is None:
        raise ValueError("synapse_counts needs a body list")
    src = _as_source(location, "counts", locked, config)
    ids = body_ids(bodies)
    from neuclease.dvid import labelsz

    from neu_vol.dvid import address

    server, uuid, instance = address(src)
    pre = labelsz.fetch_counts(server, uuid, instance, ids, "PreSyn")
    post = labelsz.fetch_counts(server, uuid, instance, ids, "PostSyn")
    out = pd.DataFrame({"pre": pre.reindex(ids).fillna(0).astype("int64"),
                        "post": post.reindex(ids).fillna(0).astype("int64")})
    out["syn"] = out["pre"] + out["post"]
    return out.rename_axis("body").reset_index()
