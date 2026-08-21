"""One pyramid scale: which level it is, how big, and what a voxel measures.

It answers "what is scale 2?" — *2815x2250x3437 voxels at 32x32x32 nm, stored under
key ``32_32_32``*. The name matches the vocabulary everything else already uses:
the precomputed spec's own ``info["scales"]``, where ``scales[2]`` *is* level 2, and
the ``--mesh-scale`` / ``--roi-scale`` / ``scale_index`` arguments the CLIs take.
Calling the type ``Level`` was considered and rejected for exactly that reason — it
would put a second word for one concept into the suite, with the flags saying "scale"
and the types saying "level".

Nothing here uses "scale" to mean a *factor*: that is :meth:`ScaleInfo.factor_from`,
named so the distinction is explicit, and ``2 ** index`` appears in this codebase only
in warnings against assuming it. The ``-Info`` suffix marks the record as distinct from
the concept.

**The voxel size must come from the source's own metadata, never from ``2 ** index``**
— that is the whole reason this type carries a per-axis tuple instead of a level
number. Real pyramids are anisotropic: halving x and y while leaving z alone is
ordinary, and an assumed factor silently misaligns meshes against skeletons. Reading
it out of an ``info`` or a zarr group is a job for a layer that can open a store, so
it does not live here.

Levels are ordered finest-first, so index 0 is full resolution and matches
``scale_index`` in a precomputed spec.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScaleInfo:
    """One pyramid level, in canonical zyx."""

    index: int
    shape: tuple[int, int, int]              # voxels (z, y, x)
    voxel_size: tuple[float, float, float]   # nm (z, y, x)
    key: str = ""                            # precomputed scale key, if any

    def factor_from(self, finest: "ScaleInfo") -> tuple[float, float, float]:
        """Full-res voxels per voxel of this scale (NOT assumed to be 2**index)."""
        return tuple(self.voxel_size[a] / finest.voxel_size[a] for a in range(3))
