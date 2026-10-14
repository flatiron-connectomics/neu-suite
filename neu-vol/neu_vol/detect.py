"""Which format a location holds, and whether a precomputed scale actually stores data.

Detection reads marker objects (``info``, ``zarr.json``, ``.zarray``) and URL schemes,
never volume data. The exceptions are O(1) chunk-existence probes: CloudVolume writes
``.gz``-suffixed chunk keys, which read as zeros unless detected, and an ``info`` may
declare scales that were never written.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Mapping, Sequence

from .location import as_directory, read_bytes, spec_kvstore, to_kvstore

logger = logging.getLogger(__name__)


#: Precomputed written by CloudVolume, whose chunk keys carry a ``.gz`` suffix.
#: tensorstore requests the unsuffixed key, finds nothing, and returns the fill value,
#: so the volume reads as all zeros with no error anywhere.
PRECOMPUTED_GZ = "neuroglancer_precomputed_gz"


def _chunk_key(scale: Mapping[str, Any], index_xyz: Sequence[int]) -> str | None:
    """Key of the chunk at grid position ``index_xyz`` of a precomputed scale.

    ``x0-x1_y0-y1_z0-z1`` in xyz, clipped to the scale's extent (a scale smaller than one
    chunk stores ``0-88_0-71_0-108``, not ``0-128_...``).
    """
    try:
        off = [int(v) for v in scale.get("voxel_offset", [0, 0, 0])]
        size = [int(v) for v in scale["size"]]
        chunk = [int(v) for v in scale["chunk_sizes"][0]]
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    lo = [off[a] + int(index_xyz[a]) * chunk[a] for a in range(3)]
    return "_".join(f"{lo[a]}-{min(lo[a] + chunk[a], off[a] + size[a])}" for a in range(3))


def _first_chunk_key(scale: Mapping[str, Any]) -> str | None:
    """Key of a precomputed scale's origin chunk, from its metadata alone."""
    return _chunk_key(scale, (0, 0, 0))


def _probe_keys(scale: Mapping[str, Any]) -> list[str]:
    """Chunk keys to probe beyond the origin: the chunk holding the scale's centre, then
    those holding the centres of its eight octants."""
    try:
        size = [int(v) for v in scale["size"]]
        chunk = [int(v) for v in scale["chunk_sizes"][0]]
    except (KeyError, IndexError, TypeError, ValueError):
        return []
    points = [[n // 2 for n in size]]
    points += [[(size[a] * (2 * k[a] + 1)) // 4 for a in range(3)]
               for k in [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)]]
    keys = []
    for p in points:
        key = _chunk_key(scale, [p[a] // chunk[a] for a in range(3)])
        if key and key not in keys and key != _first_chunk_key(scale):
            keys.append(key)
    return keys


def precomputed_chunks_are_gzipped(
        location: str | Mapping[str, Any],
        scales: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> bool:
    """True if these precomputed scales' chunk objects are ``.gz``-suffixed.

    CloudVolume gzips chunks and appends ``.gz`` to the key; tensorstore asks for the
    unsuffixed key and reads the fill value, so every block comes back as zeros.

    ``scales`` is one entry of ``info["scales"]`` or several in preference order. Each
    scale's origin chunk is probed by name first (two existence checks), and only if no
    scale answers is a prefix listed, scale by scale in the order given. A listing is
    never bounded by its ``limit`` — it enumerates the whole prefix — so
    :func:`detect_backend` passes the scales coarsest-first, keeping any listing on the
    smallest prefix.
    """
    from .location import exists, list_keys

    entries = [scales] if isinstance(scales, Mapping) else list(scales)

    for scale in entries:
        key = _first_chunk_key(scale)
        if key is None:
            continue
        prefix = scale.get("key", "")
        if exists(location, prefix, key):
            return False
        if exists(location, prefix, key + ".gz"):
            return True

    # Every scale until one yields a chunk name, not just the first: a volume can be
    # sparse at the origin of every scale, and its cheapest scale may be one that was
    # declared but never written. An unwritten prefix lists instantly.
    for scale in entries:
        for name in (k.rsplit("/", 1)[-1]
                     for k in list_keys(location, scale.get("key", ""))):
            # A chunk key looks like `0-2048_0-2048_0-128`; anything else (a shard
            # index, a stray file) is not evidence either way.
            if name and "_" in name and "-" in name:
                return name.endswith(".gz")
    return False


def scale_stores_chunks(location: str | Mapping[str, Any],
                        scale: Mapping[str, Any]) -> bool:
    """True if this precomputed scale's origin chunk object is stored, under either suffix.

    A True is evidence of a written scale; a False is not proof of an empty one, since a
    sparse volume may store nothing at its origin.
    """
    from .location import exists

    key = _first_chunk_key(scale)
    if key is None:
        return False
    prefix = scale.get("key", "")
    return bool(exists(location, prefix, key) or exists(location, prefix, key + ".gz"))


def scale_is_empty(location: str | Mapping[str, Any],
                   scale: Mapping[str, Any]) -> bool:
    """True if this precomputed scale's prefix holds no chunk object at all.

    The conclusive form of :func:`scale_stores_chunks`, at the cost of a listing. Cheap
    when the scale was never written (its prefix is empty); expensive on a sparse
    volume whose origin is background, where the whole prefix is enumerated.
    """
    from .location import list_keys

    for key in list_keys(location, scale.get("key", "")):
        name = key.rsplit("/", 1)[-1]
        # A chunk key looks like `0-2048_0-2048_0-128`; a shard index or a stray file
        # is not a chunk and says nothing about whether this scale holds data.
        if name and "_" in name and "-" in name:
            return False
    return True


def scale_holds_data(location: str | Mapping[str, Any],
                     scale: Mapping[str, Any]) -> bool:
    """Whether a precomputed scale stores any chunk: conclusive, like :func:`scale_is_empty`.

    Probes the origin chunk, then up to eight more predictable chunk keys (either
    suffix), and lists the prefix only if all of them miss, since a listing enumerates
    every object of a dense scale.
    """
    from .location import exists

    if scale_stores_chunks(location, scale):
        return True
    prefix = scale.get("key", "")
    for key in _probe_keys(scale):
        if exists(location, prefix, key) or exists(location, prefix, key + ".gz"):
            return True
    return not scale_is_empty(location, scale)


def finest_populated_scale(location: str | Mapping[str, Any],
                           scales: Sequence[Mapping[str, Any]]) -> int | None:
    """Index into ``scales`` of the finest one that actually holds data, or ``None``.

    An ``info`` may declare scales that were never written; such a scale opens, reports
    its declared extent, and reads as the fill value everywhere. Three tiers, cheapest
    first:

    1. Probe origin chunks finest-first. An ordinary volume answers on the first check.
    2. If a coarser scale has an origin chunk and finer ones do not, the finer ones are
       either unwritten or sparse at the origin; :func:`scale_holds_data` decides.
    3. If no scale has an origin chunk, apply :func:`scale_holds_data` to every scale,
       finest-first: a volume can be sparse at the origin of all its scales.
    """
    from .scales import finest_first

    order = finest_first(scales)
    for pos, i in enumerate(order):
        if not scale_stores_chunks(location, scales[i]):
            continue
        if pos == 0:
            return i                     # the finest scale holds data: nothing to weigh
        # Finer scales exist on paper but had no origin chunk. Sparse, or never written?
        for j in order[:pos]:
            if scale_holds_data(location, scales[j]):
                return j                 # sparse, and genuinely the finest data present
        return i

    # Tier 3: nothing at any origin, so only the conclusive test separates "sparse here"
    # from "never written".
    for i in order:
        if scale_holds_data(location, scales[i]):
            return i
    return None


def require_populated_scale(spec: Mapping[str, Any], *, op: str = "convert") -> None:
    """Raise if ``spec`` names a precomputed scale this volume never wrote.

    Reading an unwritten scale fails silently: every block is the fill value and the
    output is a correct-looking volume of zeros. This raises only on conclusive evidence
    — the scale's prefix holds no chunk while another scale's does — because a sparse
    volume may legitimately store nothing at the origin.
    """
    if not str(spec.get("backend", "")).startswith("neuroglancer_precomputed"):
        return
    from .scales import precomputed_info

    kv = as_directory(spec_kvstore(spec))
    info = precomputed_info(kv)
    if info is None:
        return
    scales = info["scales"]
    idx = int(spec.get("scale_index") or 0)
    if not 0 <= idx < len(scales):
        raise ValueError(
            f"`{op}`: source level {idx} does not exist; its `info` declares "
            f"{len(scales)} scale(s), 0-{len(scales) - 1}")
    if scale_holds_data(kv, scales[idx]):
        return

    populated = [i for i in range(len(scales))
                 if i != idx and scale_holds_data(kv, scales[i])]
    if not populated:
        logger.warning(
            "source scale %s (%s) holds no chunk objects, and neither does any other "
            "scale — this volume appears to be entirely empty, so reading it will "
            "produce nothing but the fill value.",
            idx, scales[idx].get("key"))
        return

    listed = ", ".join(f"{i} ({scales[i].get('key')})" for i in populated)
    raise ValueError(
        f"`{op}`: source level {idx} ({scales[idx].get('key')}) stores no data — its "
        f"`info` declares the scale, but nothing was ever written there, so every "
        f"block would read as the fill value and the output would be a volume of "
        f"zeros with nothing raised.\n\n"
        f"  levels that DO store data: {listed}\n\n"
        f"Pass --src-level to choose one explicitly.")


#: The object whose presence identifies each volume format at a location. ``info`` is
#: checked first, so a directory holding both markers reads as precomputed.
FORMAT_MARKERS = {"neuroglancer_precomputed": "info", "zarr3": "zarr.json"}

#: Formats that are one array in one container: no pyramid and no chunk keyspace. They
#: can be inspected and read; the ops that rewrite a volume in place refuse them through
#: :func:`require_chunked_volume`.
SINGLE_LEVEL_FORMATS = frozenset({"hdf5", "image_stack"})


def other_format_markers(location: str | Mapping[str, Any], fmt: str) -> list[str]:
    """Markers of volume formats *other* than ``fmt`` already present at ``location``.

    Creating a volume writes only its own marker, so a precomputed volume made where a
    zarr one lives leaves ``info`` and ``zarr.json`` side by side, and detection then
    never reaches the zarr.
    """
    # Servers, descriptor files and single containers have no keyspace for a second
    # volume to hide in; probing them would read a URL or a file as a directory.
    if fmt in ("dvid", "webknossos", "mapped") or fmt in SINGLE_LEVEL_FORMATS:
        return []
    kv = as_directory(to_kvstore(location))
    fmt = "neuroglancer_precomputed" if str(fmt).startswith("neuroglancer_precomputed") \
        else fmt
    return [marker for f, marker in FORMAT_MARKERS.items()
            if f != fmt and read_bytes(kv, marker) is not None]


def detect_file_backend(location: str | Mapping[str, Any]) -> str | None:
    """``hdf5`` or ``image_stack`` for a local file, glob or directory of slices.

    Neither has a marker object, so detection uses the name plus one ``stat``, and runs
    only after every marker probe has missed:

    * a **glob** is an image stack;
    * a **file** is judged by extension; an unknown one returns ``None``;
    * a **directory** must actually contain a slice, so a typo is not mistaken for an
      empty stack.

    Local paths only: neither reader has an object-store driver.
    """
    import os
    from glob import has_magic

    from .backends.hdf5 import HDF5_EXTENSIONS
    from .backends.imagestack import IMAGE_EXTENSIONS
    from .location import is_local, local_path

    if not is_local(location):
        return None
    path = local_path(location).rstrip("/")
    if not path:
        return None
    if has_magic(path):
        return "image_stack"
    if os.path.isfile(path):
        low = path.lower()
        if low.endswith(HDF5_EXTENSIONS):
            return "hdf5"
        if low.endswith(IMAGE_EXTENSIONS):
            return "image_stack"      # a single multipage TIFF is a stack
        return None
    if not os.path.isdir(path):
        return None
    try:
        names = os.listdir(path)
    except OSError:
        return None
    found = {os.path.splitext(n)[1].lower() for n in names} & set(IMAGE_EXTENSIONS)
    if not found:
        return None
    if len(found) > 1:
        # The reader sorts every known slice extension together, so a mixed directory
        # interleaves two stacks into one volume.
        import logging

        logging.getLogger(__name__).warning(
            "%s holds slices of more than one type (%s); they will be read as ONE stack, "
            "sorted together. If they are two stacks, point at a glob of one of them",
            path, ", ".join(sorted(found)))
    return "image_stack"


def detect_backend(location: str | Mapping[str, Any]) -> str | None:
    """Detect a source's format from its marker file (no data read).

    ``info`` -> neuroglancer-precomputed; ``zarr.json`` -> zarr v3;
    ``.zarray``/``.zgroup`` -> zarr v2; a ``dvid://`` URL -> ``dvid``; ``wk://`` ->
    ``webknossos``; a mapped-view descriptor -> ``mapped``. Returns ``None`` if none
    match.

    Precomputed whose chunks are ``.gz``-suffixed reports :data:`PRECOMPUTED_GZ`
    instead, so callers fail loudly rather than reading zeros.

    A marker always wins: ``hdf5`` and ``image_stack`` have no marker and are
    recognised by :func:`detect_file_backend` only after every marker probe has missed.
    """
    # Servers are decided by scheme, before `to_kvstore`, which would read
    # `dvid://server/uuid/instance` as a local path and report "no format". From `.dvid`,
    # not `.backends.dvid`, so this common first step does not import numpy.
    from .dvid import is_url as _is_dvid_url

    if _is_dvid_url(location):
        return "dvid"
    from .webknossos import is_url as _is_wk_url

    if _is_wk_url(location):
        return "webknossos"
    # A mapped view is a descriptor FILE, which the marker probes below would miss.
    from .mapped import is_descriptor

    if is_descriptor(location):
        return "mapped"

    from .scales import finest_first

    kv = as_directory(to_kvstore(location))
    raw = read_bytes(kv, "info")
    if raw is not None:
        try:
            scales = json.loads(raw)["scales"]
            # Coarsest first: any scale that stores a chunk answers for the whole volume,
            # and the coarsest keeps a listing fallback on the smallest prefix.
            ordered = [scales[i] for i in reversed(finest_first(scales))]
        except Exception:
            return "neuroglancer_precomputed"
        if precomputed_chunks_are_gzipped(kv, ordered):
            return PRECOMPUTED_GZ
        return "neuroglancer_precomputed"
    if read_bytes(kv, "zarr.json") is not None:
        return "zarr3"
    if read_bytes(kv, ".zarray") is not None or read_bytes(kv, ".zgroup") is not None:
        return "zarr2"
    # `location`, not `kv`: the trailing slash added above would fail `os.path.isfile`.
    return detect_file_backend(location)


def require_chunked_volume(fmt: str, location: str, op: str) -> str:
    """``fmt`` if it is a chunked multiscale volume; a useful error if it is not.

    Called by the ops that rewrite a volume in place. Without it an HDF5 file or image
    stack fails deep inside an occupancy listing that finds no chunk keys, which reads
    as "no data" rather than "not that kind of volume".
    """
    if fmt in SINGLE_LEVEL_FORMATS:
        raise ValueError(
            f"`{op}` needs a chunked multiscale volume (zarr or neuroglancer-"
            f"precomputed), and {location} is {fmt}: one array in one container, with no "
            f"pyramid and no chunk objects to count. Convert it into a volume first — "
            f"`neu-vol convert {location} <dst> --voxel-size z,y,x` — and run `{op}` on "
            f"that. To go the other way, `neu-vol write` places an HDF5 piece INTO a "
            f"volume.")
    return fmt
