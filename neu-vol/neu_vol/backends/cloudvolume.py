"""Read precomputed volumes written by CloudVolume, whose chunks are ``.gz``-suffixed.

CloudVolume gzips each chunk and appends ``.gz`` to the key, which the precomputed spec
does not address: tensorstore requests the unsuffixed key and returns the fill value, so
**the volume reads as all zeros and nothing raises**. ``detect.detect_backend`` returns
:data:`~neu_vol.detect.PRECOMPUTED_GZ` for such volumes, which routes here, to the library
that wrote them.

**cloud-volume is optional and not in the main environment**: it pins ``DracoPy<2``,
which would downgrade the Draco codec neu-morpho's meshes need. Use a separate
environment holding blockrun and neu-vol but not neu-morpho. Read-only.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import Region, register_backend

TAG = "neuroglancer_precomputed_gz"

_MISSING = (
    "This volume was written by CloudVolume: its chunk objects carry a '.gz' suffix, "
    "which tensorstore cannot address — it would read the whole volume as zeros "
    "without raising. Reading it needs the `cloud-volume` package, which is not "
    "installed here.\n\n"
    "cloud-volume is kept out of the main environment on purpose: it pins DracoPy<2 "
    "and would downgrade the DracoPy that neu-morpho's mesh stage depends on. "
    "Use the separate environment instead:\n"
    "    mamba env create -f neu-vol-cv-environment.yml\n"
    "    conda activate neu-vol-cv\n"
    "    pip install cloud-volume\n"
    "    pip install --no-deps -e ./blockrun -e ./neu-vol"
)


def _url(spec: Mapping[str, Any]) -> str:
    """A CloudVolume URL for this spec's location.

    Through ``spec_kvstore``, which resolves all three spec forms: ``convert`` passes a
    ``data_spec`` carrying a ``kvstore`` and no ``path``.
    """
    from ..location import spec_kvstore

    kv = spec_kvstore(spec)
    driver = kv.get("driver")
    path = str(kv.get("path", "")).strip("/")
    if driver == "file":
        return "precomputed://file://" + "/" + path
    if driver == "s3":
        bucket = kv.get("bucket") or ""
        if not bucket:
            raise ValueError(f"s3 kvstore has no bucket: {kv!r}")
        return f"precomputed://s3://{bucket}/{path}"
    raise ValueError(
        f"CloudVolume sources support local paths and s3:// only; got driver "
        f"{driver!r} from {kv!r}")


class CloudVolumeBackend:
    """Read-only ``(z, y, x)`` view over a CloudVolume-written precomputed volume."""

    def __init__(self, spec: Mapping[str, Any]):
        try:
            from cloudvolume import CloudVolume
        except ImportError as exc:                       # fail fast and explain
            raise ImportError(_MISSING) from exc

        self._spec = dict(spec)
        # fill_missing: an absent chunk reads as the fill value, as in every other backend.
        self._cv = CloudVolume(_url(spec), mip=int(spec.get("scale_index", 0)),
                               fill_missing=True, progress=False, parallel=False)

    @classmethod
    def open(cls, spec: Mapping[str, Any]) -> "CloudVolumeBackend":
        return cls(spec)

    # CloudVolume is xyz(+channel) throughout; this package is canonical zyx. Every
    # boundary below flips, and getting it wrong transposes the volume silently.
    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(s) for s in self._cv.shape[:3][::-1])

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self._cv.dtype)

    @property
    def chunks(self) -> tuple[int, ...]:
        return tuple(int(c) for c in self._cv.chunk_size[::-1])

    def read_region(self, region: Region) -> np.ndarray:
        zs, ys, xs = region
        block = self._cv[xs.start:xs.stop, ys.start:ys.stop, zs.start:zs.stop]
        arr = np.asarray(block)
        if arr.ndim == 4:                # (x, y, z, channel); single-channel only
            if arr.shape[3] != 1:
                raise ValueError(
                    f"multi-channel CloudVolume sources are not supported "
                    f"(got {arr.shape[3]} channels)")
            arr = arr[..., 0]
        return np.ascontiguousarray(arr.transpose(2, 1, 0))

    def write_region(self, region: Region, data: np.ndarray) -> None:
        raise TypeError(
            "CloudVolume-written precomputed is a read-only source here. Convert it "
            "to zarr or standard precomputed rather than writing more of it.")

    def to_spec(self) -> dict[str, Any]:
        return dict(self._spec)


register_backend(TAG, CloudVolumeBackend.open)
