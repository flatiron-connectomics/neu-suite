"""Give every occupied region of a sparse volume its own range of label ids.

Ground truth annotated chunk by chunk comes back with each chunk numbered from 1, so the
same integer means a different cell in every chunk, and a mesh of one id is a dozen
unrelated cells. This renumbers each occupied region into a range of its own.

- **The regions come from stored-chunk occupancy** (:mod:`neu_vol.ops.occupancy`), so they
  are pairwise disjoint; the run is serial, each range starting where the last ended.
- **Boxes are not tightened**: mode downsampling can drop a stray voxel, so a tightened box
  could leave level-0 data with its old id. The chunk-aligned box covers every stored
  chunk, and its writes are whole chunks.
- **The mapping is an output**, written by default: it is the only way back from a new id
  to its region and original label.

Single-scale, like :mod:`neu_vol.ops.write`: run ``neu-vol downsample --start-level
<level>`` afterwards.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Sequence

import numpy as np

from ..backends.base import open_backend
from ..detect import detect_backend, require_chunked_volume
from ..source_metadata import existing_levels, level_spec
from .occupancy import labeled_regions

# A region is read whole, to collect its ids in one pass. A region over this is a sign
# the volume is not the sparse kind this op is for.
DEFAULT_MAX_REGION_BYTES = 8 * 1024 ** 3


def _apply_map(data: np.ndarray, old: np.ndarray, new: np.ndarray) -> np.ndarray:
    """Relabel ``data`` by the sorted ``old`` -> ``new`` correspondence, keeping 0 as 0.

    ``old`` is ``np.unique(data)`` minus 0, so sorted and complete: a searchsorted works.
    """
    flat = data.reshape(-1)
    out = np.zeros_like(flat)
    nz = flat != 0
    out[nz] = new[np.searchsorted(old, flat[nz])]
    return out.reshape(data.shape)


def plan_relabel(volume: str, *, out: str | None = None, in_place: bool = False,
                 level: int = 0, block_size: int | None = None,
                 max_region_bytes: int = DEFAULT_MAX_REGION_BYTES,
                 fmt: str | None = None) -> dict:
    """Resolve the geometry of the relabel. Reads chunk keys only; writes nothing.

    The id mapping needs the voxels, so :func:`apply_relabel` resolves it (and reports it
    without writing when ``dry_run=True``).
    """
    volume = volume.rstrip("/")
    if bool(out) == bool(in_place):
        raise ValueError("give exactly one of out=<new volume> or in_place=True — "
                         "there is no safe default: one publishes beside the original, "
                         "the other overwrites the ids it is derived from")
    if block_size is not None and block_size < 1:
        raise ValueError(f"block_size must be positive, got {block_size}")

    fmt = fmt or detect_backend(volume)
    if fmt is None:
        raise FileNotFoundError(f"no volume found at {volume}")
    require_chunked_volume(fmt, volume, "neu-vol relabel")
    levels = existing_levels(volume, fmt)
    if level not in levels:
        raise ValueError(f"{volume} has no level {level}; present: {sorted(levels)}")

    regions, ctx = labeled_regions(volume, level=level, tighten_level=None, fmt=fmt)
    if not regions:
        raise ValueError(f"{volume} stores no chunks at level {level} — nothing to "
                         f"relabel")

    src = open_backend(level_spec(volume, fmt, level))
    itemsize = np.dtype(src.dtype).itemsize
    for i, r in enumerate(regions):
        extent = [hi - lo for lo, hi in zip(r["lo"], r["hi"])]
        nbytes = math.prod(extent) * itemsize
        if nbytes > max_region_bytes:
            raise ValueError(
                f"region {i} is {extent} = {nbytes / 1024 ** 3:.1f} GiB at "
                f"{itemsize} bytes/voxel, over the {max_region_bytes / 1024 ** 3:.1f} "
                f"GiB limit. Every id in a region has to be collected in one pass, so "
                f"the region is read whole; raise max_region_bytes if the machine has "
                f"the memory.")

    stale = [i for i in sorted(levels) if i > level]
    return {
        "volume": volume, "format": fmt, "level": level,
        "destination": out.rstrip("/") if out else volume,
        "in_place": bool(in_place),
        "block_size": block_size,
        "dtype": str(src.dtype),
        "chunk": ctx["cell"],
        "n_chunks": ctx["n_chunks"],
        "regions": regions,
        # Levels above keep the old ids; reported, not fixed.
        "stale_levels": stale,
    }


def apply_relabel(plan: dict, *, dry_run: bool = False, overwrite: bool = False,
                  map_path: str | None = None) -> dict:
    """Carry out a :func:`plan_relabel`, returning the mapping and per-region summary.

    With ``dry_run`` the reads still happen — the mapping is only knowable from the
    voxels — but nothing is written and no destination is created.
    """
    from ..location import write_json

    volume, fmt, level = plan["volume"], plan["format"], plan["level"]
    dest, block = plan["destination"], plan["block_size"]
    src = open_backend(level_spec(volume, fmt, level))

    dst = None
    if not dry_run:
        if not plan["in_place"]:
            # Empty, with the source's frame copied verbatim.
            from .create import create_volume
            create_volume(dest, like=volume, overwrite=overwrite)
        dst = open_backend(level_spec(dest, fmt, level))

    dtype = np.dtype(plan["dtype"])
    offset, entries, id_sets = 0, [], []
    for i, r in enumerate(plan["regions"]):
        region = tuple(slice(int(lo), int(hi)) for lo, hi in zip(r["lo"], r["hi"]))
        data = src.read_region(region)
        ids = np.unique(data)
        ids = ids[ids != 0]
        n = int(ids.size)
        base = offset if block is None else i * block
        if block is not None and n > block:
            raise ValueError(
                f"region {i} holds {n} labels, more than block_size={block}, so its "
                f"range would run into region {i + 1}'s. Raise block_size to at least "
                f"{n}.")
        new = np.arange(base + 1, base + 1 + n, dtype=dtype)
        if n and not dry_run:
            dst.write_region(region, _apply_map(data, ids, new))
        id_sets.append(set(int(v) for v in ids))
        entries.append({
            "index": i,
            "lo_zyx": [int(v) for v in r["lo"]],
            "hi_zyx": [int(v) for v in r["hi"]],
            "chunks": int(r["cells"]),
            "n_labels": n,
            "new_id_range": [base + 1, base + n] if n else None,
            "map": {str(int(o)): int(v) for o, v in zip(ids, new)},
        })
        offset = base + n

    shared = [v for v in set().union(*id_sets) if sum(v in s for s in id_sets) > 1]
    pairs = [{"regions": [i, j], "shared": len(id_sets[i] & id_sets[j])}
             for i, j in itertools.combinations(range(len(id_sets)), 2)
             if id_sets[i] & id_sets[j]]
    result = {
        "source": volume, "destination": dest, "in_place": plan["in_place"],
        "level": level, "block_size": block, "dtype": plan["dtype"],
        "n_regions": len(entries),
        "n_labels_in": sum(len(s) for s in id_sets),
        "n_labels_out": sum(e["n_labels"] for e in entries),
        "n_distinct_in": len(set().union(*id_sets)),
        "collisions_resolved": len(shared),
        "colliding_region_pairs": pairs,
        "stale_levels": plan["stale_levels"],
        "regions": entries,
        "dry_run": bool(dry_run),
    }
    if map_path and not dry_run:
        # Through `location`, so it can live beside a remote volume.
        write_json(map_path, result, indent=1)
        result["map_path"] = map_path
    return result


def default_map_path(destination: str, level: int) -> str:
    """``<last path component>.relabel-<level>.json``, in the working directory.

    A default, so the mapping cannot be lost by forgetting a flag.
    """
    name = destination.rstrip("/").rsplit("/", 1)[-1]
    return f"{name}.relabel-{level}.json"


def relabel(volume: str, *, out: str | None = None, in_place: bool = False,
            level: int = 0, block_size: int | None = None,
            max_region_bytes: int = DEFAULT_MAX_REGION_BYTES,
            dry_run: bool = False, overwrite: bool = False,
            map_path: str | None = None) -> dict:
    """Plan and apply in one call. See :func:`plan_relabel` and :func:`apply_relabel`."""
    plan = plan_relabel(volume, out=out, in_place=in_place, level=level,
                        block_size=block_size, max_region_bytes=max_region_bytes)
    if map_path is None:
        map_path = default_map_path(plan["destination"], level)
    return apply_relabel(plan, dry_run=dry_run, overwrite=overwrite,
                         map_path=map_path)
