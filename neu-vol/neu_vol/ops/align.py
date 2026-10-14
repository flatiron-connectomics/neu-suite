"""The block grid a box should be aligned to, from a volume's own chunking and pyramid."""

from __future__ import annotations

from typing import Any, Mapping

GRIDS = ("write-unit", "read-chunk", "pyramid", "both")


def align_grid(described: Mapping[str, Any], volume: str, *, level: int = 0,
               to: str = "write-unit") -> tuple[tuple[int, ...], str]:
    """``(block in level-0 voxels, where it came from)`` for one of :data:`GRIDS`.

    ``described`` is :func:`neu_vol.describe`'s output for ``volume``. ``write-unit`` is
    the level's chunk, or its shard where sharded (the object a partial write rewrites);
    ``read-chunk`` the inner chunk; ``pyramid`` the deepest level's cumulative factor,
    the grid every level shares; ``both`` the LCM of the write unit and that factor.
    Raises ``ValueError`` if the volume does not report what the grid needs.
    """
    from neu_lib.grid import lcm_grid

    from ..scales import level_factor
    from ..source_metadata import level0_chunking

    if to not in GRIDS:
        raise ValueError(f"unknown grid {to!r}; expected one of {GRIDS}")
    per_level, levels = described["level_voxel_sizes"], described["levels"]

    def factor(lv):
        try:
            return level_factor(per_level, lv)
        except ValueError as e:
            raise ValueError(f"{volume} {e}") from None

    if level == 0 and not levels:
        # A bare array reports no levels; its chunking is on the array itself.
        read_chunk, write_unit = level0_chunking(described, volume)
        at = (1, 1, 1)
    else:
        if level not in levels:
            raise ValueError(f"{volume} has no level {level} "
                             f"(present: {sorted(levels) or 'none'})")
        lvl = levels[level]
        write_unit, read_chunk = lvl.get("chunks"), lvl.get("read_chunks")
        # A level's own chunk is in ITS voxels; the box is in level-0 ones.
        at = (1, 1, 1) if level == 0 else factor(level)
    sharded = bool(read_chunk and write_unit and read_chunk != write_unit)

    def scaled(block, what):
        if not block:
            raise ValueError(f"level {level} of {volume} reports no {what}")
        return tuple(int(c) * f for c, f in zip(block, at)), (
            f"level-{level} {what}"
            + (f", x{at} to level-0 voxels" if any(f != 1 for f in at) else ""))

    def pyramid_grid():
        if not per_level or len(per_level) == 1:
            return (1, 1, 1), "no pyramid: a single level constrains nothing"
        deepest = len(per_level) - 1
        return factor(deepest), f"cumulative factor of level {deepest}, the deepest"

    if to == "write-unit":
        return scaled(write_unit, "shard" if sharded else "chunk")
    if to == "read-chunk":
        return scaled(read_chunk or write_unit, "read chunk")
    if to == "pyramid":
        return pyramid_grid()
    unit, unit_from = scaled(write_unit, "shard" if sharded else "chunk")
    deepest, _ = pyramid_grid()
    return lcm_grid(unit, deepest), (f"LCM of the {unit_from} and the pyramid factor "
                                     f"{deepest}")
