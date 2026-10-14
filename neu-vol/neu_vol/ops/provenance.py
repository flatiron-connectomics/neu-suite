"""What a volume was made from, written beside it as ``provenance.json``.

Written for every source. For DVID it records the resolved node: a branch ref like
``93fdbc:main`` names a different node tomorrow.

A sidecar rather than a field in ``info`` or the OME attributes, which viewers and
validators read and may reject. Written through ``location``, so local or object store,
and **after** the data, so a run that died half way leaves no record.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

logger = logging.getLogger(__name__)

#: The sidecar's name at the destination root.
FILENAME = "provenance.json"


def source_provenance(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Per-backend source facts, or a generic description of the spec.

    Backends that can say something specific do; everything else is described by its
    spec, which already names the location it read.
    """
    backend = spec.get("backend")
    if backend == "dvid":
        from ..backends.dvid import provenance
        from ..dvid import resolve_node

        # A concrete node by now; `requested_ref` keeps what was typed.
        node = resolve_node(spec)
        node = {**node, "ref": spec.get("requested_ref", node["ref"]),
                "walked": spec.get("ancestors_walked", node["walked"])}
        return provenance(spec, node)
    if backend == "webknossos":
        # The resolved spec identifies the read (never the token); `mapping` is explicit.
        return {"source": backend, **{k: v for k, v in spec.items() if k != "backend"},
                "mapping": spec.get("mapping"),
                "ids": "agglomerates" if spec.get("mapping") else "supervoxels (raw ids)"}
    if backend == "mapped":
        # A view of another volume: record both halves.
        from ..mapped import read_descriptor

        rec = {"source": "mapped", "mapping": spec.get("mapping"),
               "table": spec.get("table"), "descriptor": spec.get("descriptor"),
               "of": source_provenance(spec["source"])}
        try:
            rec["origin"] = read_descriptor(spec["descriptor"]).get("origin")
        except Exception:                                      # noqa: BLE001
            pass
        return rec
    return {"source": backend, **{k: v for k, v in spec.items() if k != "backend"}}


def build_record(*, src_spec: Mapping[str, Any], dst: str, **run: Any) -> dict[str, Any]:
    """The record for one conversion. ``run`` carries whatever the caller thinks matters."""
    import datetime

    return {
        "written": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "tool": "neu-vol",
        "destination": str(dst),
        "source": source_provenance(src_spec),
        "run": {k: v for k, v in run.items() if v is not None},
    }


def write(dst: str, record: Mapping[str, Any]) -> None:
    """Write ``provenance.json`` at ``dst``. Never fails the run.

    The data is already written, so a failure here is logged, not raised.
    """
    from ..location import write_json

    try:
        write_json(dst, dict(record), FILENAME)
        logger.info("provenance written to %s/%s", str(dst).rstrip("/"), FILENAME)
    except Exception as exc:                                  # noqa: BLE001
        logger.warning("could not write %s at %s (%s: %s). The data is fine; only the "
                       "record of where it came from is missing.",
                       FILENAME, dst, type(exc).__name__, exc)
