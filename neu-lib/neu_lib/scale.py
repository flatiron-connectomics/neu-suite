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

from .frame import Frame


@dataclass(frozen=True)
class ScaleInfo:
    """One pyramid level, in canonical zyx.

    The level **carries** its :class:`~neu_lib.frame.Frame` rather than a bare voxel
    size, so "what does a voxel at this level mean in nm" has exactly one answer and
    callers stop rebuilding one. The two types stay separate because each has callers
    that need only it: a synapse table or an SWC in voxel units wants a ``Frame`` and has
    no level, while building a read spec wants ``index`` and ``key`` and converts no
    coordinates.

    Carrying the frame is also what lets a level's **origin** survive. It used to be
    dropped on the floor: the reader took ``resolution`` and ``size`` and never looked at
    ``voxel_offset``, so every level came back claiming to start at nm zero. Harmless on
    a volume written from the origin, wrong for anything cropped — and invisible either
    way, since nothing raises.
    """

    index: int
    shape: tuple[int, int, int]              # voxels (z, y, x)
    frame: Frame                             # voxel -> nm for this level
    key: str = ""                            # precomputed scale key, if any

    @property
    def voxel_size(self) -> tuple[float, float, float]:
        """nm per voxel (z, y, x) — the frame's, spelled the way a level reads."""
        return self.frame.voxel_size_nm

    @property
    def origin_nm(self) -> tuple[float, float, float]:
        """Where this level's voxel ``(0, 0, 0)`` sits, in nm."""
        return self.frame.origin_nm

    def factor_from(self, finest: "ScaleInfo") -> tuple[float, float, float]:
        """Full-res voxels per voxel of this scale (NOT assumed to be 2**index)."""
        return tuple(self.voxel_size[a] / finest.voxel_size[a] for a in range(3))
