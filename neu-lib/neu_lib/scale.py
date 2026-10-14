"""One pyramid level: its index, its shape, and what a voxel measures.

For example, scale 2 might be 2815x2250x3437 voxels at 32x32x32 nm, stored under the key
``32_32_32``. Levels are ordered finest first, so index 0 is full resolution, as in a
precomputed ``info["scales"]``. The voxel size comes from the source's metadata (read by
``neu_vol.read_scales``), never from ``2 ** index``: real pyramids are anisotropic.
"""

from __future__ import annotations

from dataclasses import dataclass

from .frame import Frame


@dataclass(frozen=True)
class ScaleInfo:
    """One pyramid level, zyx. Its :class:`~neu_lib.frame.Frame` gives the voxel size and
    the origin, so a cropped volume's levels keep their true position."""

    index: int
    shape: tuple[int, int, int]              # voxels (z, y, x)
    frame: Frame                             # voxel -> nm for this level
    key: str = ""                            # precomputed scale key, if any

    @property
    def voxel_size(self) -> tuple[float, float, float]:
        """nm per voxel (z, y, x)."""
        return self.frame.voxel_size_nm

    @property
    def origin_nm(self) -> tuple[float, float, float]:
        """Where this level's voxel ``(0, 0, 0)`` sits, in nm."""
        return self.frame.origin_nm

    def factor_from(self, finest: "ScaleInfo") -> tuple[float, float, float]:
        """Full-resolution voxels per voxel of this level, per axis (not assumed to be
        ``2 ** index``)."""
        return tuple(self.voxel_size[a] / finest.voxel_size[a] for a in range(3))
