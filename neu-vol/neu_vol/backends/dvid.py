"""Read-only backend over a DVID ``labelmap`` instance.

Addressed by ``server``/``uuid``/``instance`` rather than a path (``neu_vol.dvid.parse_url``);
addressing and version resolution are in ``neu_vol/dvid.py``, and this module is the
labelmap geometry and array view.

DVID omits blocks that hold no labels, so an empty region costs a round trip and no
bandwidth, and no occupancy prefilter is needed. Reads request ``?compression=blocks``
(DVID's stored form) and inflate here with libdvid, on the dask worker. DVID is typically a
shared service and answers overload with 503, which ``retry`` treats as transient.

Read-only: segmentation is never written back to DVID.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .. import dvid as _dvid
from .base import Region, register_backend

TAG = "dvid"

# Calls go through `_dvid.<name>` so a test that stubs the server patches one module
# attribute and covers every call site.
_MISSING = _dvid.MISSING


def provenance(spec: Mapping[str, Any], node: Mapping[str, Any]) -> dict:
    """What this labelmap pull came from — the shared node record plus label facts.

    ``maxlabel`` advances as proofreading creates bodies, so with the node it characterises
    the snapshot; DVID's mutation id and lastmod are per body, so not usable here.
    """
    out = _dvid.node_provenance(spec, node)
    out["supervoxels"] = bool(spec.get("supervoxels", False))
    # Nice to have, not worth failing a completed run over.
    try:
        from neuclease.dvid import fetch_maxlabel

        out["maxlabel"] = int(fetch_maxlabel(out["server"], node["uuid"],
                                             out["instance"]))
    except Exception as exc:                                   # noqa: BLE001
        out["maxlabel_error"] = f"{type(exc).__name__}: {exc}"[:200]
    return out


def geometry(info: Mapping[str, Any], spec: Mapping[str, Any]) -> dict:
    """Shape / chunks / voxel size / max scale, all zyx, from DVID's ``/info``.

    Presented **origin-anchored**: shape is ``MaxPoint + 1``, so a voxel index means the
    same as in DVID. The empty margin below ``MinPoint`` costs a quick round trip per task
    and nothing on disk.
    """
    ext = info.get("Extended", {})
    _dvid.check_instance_type(info, spec, "labelmap")

    lo_xyz, hi_xyz = ext.get("MinPoint"), ext.get("MaxPoint")
    if lo_xyz is None or hi_xyz is None:
        raise ValueError(
            f"{_dvid.spec_url(spec)} reports no extents (MinPoint/MaxPoint are null), "
            f"which DVID does for an instance that has been created but never written "
            f"to. There is nothing to read.")
    lo = tuple(int(v) for v in lo_xyz[::-1])
    if any(v < 0 for v in lo):
        raise ValueError(
            f"{_dvid.spec_url(spec)} has a negative MinPoint {lo_xyz!r}. This backend "
            f"presents DVID's own coordinates origin-anchored, which would clip data "
            f"at negative indices. Support for a translated frame is not implemented.")

    block = tuple(int(v) for v in ext["BlockSize"][::-1])
    voxel = tuple(float(v) for v in ext["VoxelSize"][::-1])
    return {
        "shape0": tuple(int(v) + 1 for v in hi_xyz[::-1]),
        "min_point": lo,
        "chunks": block,
        "voxel_size": voxel,
        "max_scale": int(ext.get("MaxDownresLevel", 0)),
        "dtype": np.dtype("uint64"),
    }


def level_shape(shape0, scale: int) -> tuple[int, ...]:
    """Shape at a DVID downres level.

    DVID halves every axis per level, so here, unlike a general pyramid, the factor
    really is ``2**scale``. Do not copy this to another backend.
    """
    return tuple(-(-int(s) // 2 ** int(scale)) for s in shape0)


class DVIDBackend:
    """Read-only ``(z, y, x)`` view over one scale of a DVID labelmap instance."""

    def __init__(self, spec: Mapping[str, Any]):
        self._spec = dict(spec)
        self._server, self._uuid, self._instance = _dvid.address(spec)
        self._scale = int(spec.get("scale_index", 0))
        self._supervoxels = bool(spec.get("supervoxels", False))

        geom = geometry(_dvid.instance_info(spec), spec)
        if self._scale > geom["max_scale"]:
            raise ValueError(
                f"{_dvid.spec_url(spec)} has scales 0..{geom['max_scale']} "
                f"(MaxDownresLevel), but scale {self._scale} was requested")
        self._shape = level_shape(geom["shape0"], self._scale)
        self._chunks = geom["chunks"]
        self._dtype = geom["dtype"]
        self.voxel_size = tuple(v * 2 ** self._scale for v in geom["voxel_size"])

    @classmethod
    def open(cls, spec: Mapping[str, Any]) -> "DVIDBackend":
        return cls(spec)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def chunks(self) -> tuple[int, ...]:
        return self._chunks

    def read_region(self, region: Region) -> np.ndarray:
        try:
            from neuclease.dvid import fetch_labelmap_voxels
        except ImportError as exc:
            raise ImportError(_MISSING) from exc

        box = np.array([[int(s.start) for s in region],
                        [int(s.stop) for s in region]], dtype=np.int64)
        # Rounded out to DVID's 64-voxel blocks and truncated back: an unaligned region is
        # correct but wasteful. Task shapes are whole multiples of `chunks`.
        out = fetch_labelmap_voxels(
            self._server, self._uuid, self._instance, box,
            scale=self._scale, supervoxels=self._supervoxels)
        return np.ascontiguousarray(out)

    def write_region(self, region: Region, data: np.ndarray) -> None:
        raise TypeError(
            "DVID labelmap instances are read-only here. Proofreading edits belong in "
            "the proofreading tool; this package only pulls snapshots out.")

    def to_spec(self) -> dict[str, Any]:
        return dict(self._spec)


register_backend(TAG, DVIDBackend.open)
