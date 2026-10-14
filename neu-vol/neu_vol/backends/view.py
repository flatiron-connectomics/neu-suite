"""Read-only view backends that wrap another backend.

``CropBackend`` is a shifted/resized window over a source, padding out-of-bounds regions
with a fill value; ``MaskBackend`` blanks boxes to a fill value; ``RemapBackend`` replaces
values. Each opens from a spec, so the engine can crop, pad, mask and pyramid a region as it
would any other source.

Masking on the read side holds at every scale: each pyramid level is derived from the level
below it in the output, so a hole in level 0 propagates upward on its own.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import Region, open_backend, register_backend

TAG = "crop"


class CropBackend:
    """A window ``[origin, origin+shape)`` over a source backend (canonical order).

    ``origin`` may be negative and ``shape`` may extend past the source; regions
    outside the source are filled with ``pad_value``.
    """

    def __init__(self, spec: Mapping[str, Any]):
        self._spec = dict(spec)
        self._source_spec = dict(spec["source"])
        self._origin = tuple(int(o) for o in spec["origin"])
        self._shape = tuple(int(s) for s in spec["shape"])
        self._pad_value = spec.get("pad_value", 0)
        self._source = open_backend(self._source_spec)
        if not (len(self._origin) == len(self._shape) == len(self._source.shape)):
            raise ValueError("origin/shape/source rank mismatch")

    @classmethod
    def open(cls, spec: Mapping[str, Any]) -> "CropBackend":
        return cls(spec)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._source.dtype

    @property
    def chunks(self) -> tuple[int, ...]:
        return self._source.chunks

    def read_region(self, region: Region) -> np.ndarray:
        out_shape = tuple(s.stop - s.start for s in region)
        out = np.full(out_shape, self._pad_value, dtype=self.dtype)
        src_slices, dst_slices = [], []
        for a, s in enumerate(region):
            src_start = s.start + self._origin[a]
            src_stop = s.stop + self._origin[a]
            cs = max(src_start, 0)
            ce = min(src_stop, self._source.shape[a])
            if ce <= cs:
                return out  # this axis is fully out of bounds -> all padding
            d0 = cs - src_start
            src_slices.append(slice(cs, ce))
            dst_slices.append(slice(d0, d0 + (ce - cs)))
        out[tuple(dst_slices)] = self._source.read_region(tuple(src_slices))
        return out

    def write_region(self, region: Region, data: np.ndarray) -> None:
        raise TypeError("crop views are read-only")

    def to_spec(self) -> dict[str, Any]:
        return dict(self._spec)


register_backend(TAG, CropBackend.open)


MASK_TAG = "mask"


class MaskBackend:
    """A source with every box in ``boxes`` replaced by ``value`` (canonical order).

    Boxes are full-ndim half-open ``[lo, hi)`` in the *source's* own index space, so a
    mask composes with a crop laid over it without either knowing about the other: the
    crop shifts coordinates, and this sits underneath.
    """

    def __init__(self, spec: Mapping[str, Any]):
        self._spec = dict(spec)
        self._source = open_backend(dict(spec["source"]))
        self._value = spec.get("value", 0)
        self._boxes = tuple((tuple(int(v) for v in lo), tuple(int(v) for v in hi))
                            for lo, hi in spec["boxes"])
        for lo, hi in self._boxes:
            if not (len(lo) == len(hi) == len(self._source.shape)):
                raise ValueError(f"mask box {lo}:{hi} does not match the source's rank "
                                 f"{len(self._source.shape)}")

    @classmethod
    def open(cls, spec: Mapping[str, Any]) -> "MaskBackend":
        return cls(spec)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._source.shape

    @property
    def dtype(self) -> np.dtype:
        return self._source.dtype

    @property
    def chunks(self) -> tuple[int, ...]:
        return self._source.chunks

    def read_region(self, region: Region) -> np.ndarray:
        data = self._source.read_region(region)
        if not data.flags.writeable:
            # A backend is free to hand back a read-only view; blanking in place would
            # then raise, or worse, silently mutate something shared.
            data = data.copy()
        for lo, hi in self._boxes:
            local = []
            for axis, s in enumerate(region):
                start, stop = max(s.start, lo[axis]), min(s.stop, hi[axis])
                if stop <= start:
                    break                      # this box misses the region entirely
                local.append(slice(start - s.start, stop - s.start))
            else:
                data[tuple(local)] = self._value
        return data

    def write_region(self, region: Region, data: np.ndarray) -> None:
        raise TypeError("mask views are read-only")

    def to_spec(self) -> dict[str, Any]:
        return dict(self._spec)


register_backend(MASK_TAG, MaskBackend.open)


REMAP_TAG = "remap"


class RemapBackend:
    """A source with ``values`` replaced by ``to`` (default 0) as it is read.

    For a segmentation whose background is not 0: **a block of background 1s is not
    all-fill, so it gets stored**, and "which chunks exist" stops answering "where is the
    data", which occupancy, ``relabel`` and ``--sparse`` rely on. Remapping on the read
    side corrects it before the storage decision.
    """

    def __init__(self, spec: Mapping[str, Any]):
        self._spec = dict(spec)
        self._source = open_backend(dict(spec["source"]))
        self._values = np.asarray(spec["values"]).ravel()
        self._to = spec.get("to", 0)
        if self._values.size == 0:
            raise ValueError("remap needs at least one value to replace")

    @classmethod
    def open(cls, spec: Mapping[str, Any]) -> "RemapBackend":
        return cls(spec)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._source.shape

    @property
    def dtype(self) -> np.dtype:
        return self._source.dtype

    @property
    def chunks(self) -> tuple[int, ...]:
        return self._source.chunks

    def read_region(self, region: Region) -> np.ndarray:
        data = self._source.read_region(region)
        if not data.flags.writeable:
            data = data.copy()
        values = self._values.astype(data.dtype, copy=False)
        # `isin`: a lookup table would need one entry per possible uint64 value.
        data[np.isin(data, values)] = self._to
        return data

    def write_region(self, region: Region, data: np.ndarray) -> None:
        raise TypeError("remap views are read-only")

    # A remap changes values, not geometry, so the source's recorded frame still holds and
    # is forwarded. A crop moves the data, so it must not forward the source's offset.
    def stored_offset(self, name: str = "voxel_offset"):
        return self._forward("stored_offset", name)

    def stored_axes(self, name: str = "axes"):
        return self._forward("stored_axes", name)

    def stored_voxel_size(self, name: str = "voxel_size"):
        return self._forward("stored_voxel_size", name)

    def _forward(self, capability: str, name: str):
        read = getattr(self._source, capability, None)
        return read(name) if read is not None else None

    def to_spec(self) -> dict[str, Any]:
        return dict(self._spec)


register_backend(REMAP_TAG, RemapBackend.open)
