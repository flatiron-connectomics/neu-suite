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
#: tensorstore requests the *unsuffixed* key, finds nothing, and returns the fill
#: value — so the volume reads as all zeros with no error anywhere. A full 1.9 M-block
#: conversion was lost to exactly this before it was detected.
PRECOMPUTED_GZ = "neuroglancer_precomputed_gz"
def _first_chunk_key(scale: Mapping[str, Any]) -> str | None:
    """The key of a precomputed scale's origin chunk, from its metadata alone.

    ``x0-x1_y0-y1_z0-z1`` in xyz, where the extent is clipped to the scale's size —
    a scale smaller than one chunk stores `0-88_0-71_0-108`, not `0-128_...`.
    """
    try:
        off = [int(v) for v in scale.get("voxel_offset", [0, 0, 0])]
        size = [int(v) for v in scale["size"]]
        chunk = [int(v) for v in scale["chunk_sizes"][0]]
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    return "_".join(f"{off[a]}-{off[a] + min(chunk[a], size[a])}" for a in range(3))
def precomputed_chunks_are_gzipped(
        location: str | Mapping[str, Any],
        scales: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> bool:
    """True if these precomputed scales' chunk objects are ``.gz``-suffixed.

    CloudVolume gzips chunks and appends ``.gz`` to the key, which is legal for
    something that serves them over HTTP with ``Content-Encoding: gzip`` but is not what
    the precomputed spec addresses — tensorstore asks for the unsuffixed key and reads
    the fill value, so every block comes back as zeros with nothing raised.

    **Answered with two existence checks, not a listing.** This used to list the scale
    prefix with ``limit=4``, which reads as bounded and is not: ``list_keys`` awaits the
    whole listing before Python truncates it, so on a dense image volume's finest scale
    it enumerated millions of keys. Measured on an 8-level EM volume on S3: **51 s, in
    `detect_backend`, which nearly every op calls before doing anything.** Probing the
    one key the metadata predicts is O(1) and needs no enumeration at all.

    Falls back to a listing only when no origin chunk is found — legitimate on a sparse
    volume — and then over the *first* scale given, which is why :func:`detect_backend`
    orders them coarsest-first: that one holds the fewest chunks.

    ``scales`` is one entry of ``info["scales"]`` or several in preference order. **With
    several, every O(1) origin probe is tried before any listing**, which is what makes
    a volume whose ``info`` declares scales it never wrote answerable at all. A
    ``sample3`` neuropil mask declares seven scales and stores exactly one (``64_64_64``,
    18 chunks): asked about the coarsest alone this found no origin chunk, listed an
    empty prefix, and reported the volume as un-gzipped — so every block read as zeros,
    the precise failure the two-check probe exists to prevent. Sweeping the origins
    first costs 14 existence checks and 53 ms there, and still answers a dense volume on
    the very first probe, so the 51 s enumeration stays gone.
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

    # Every scale, in the order given, until one yields a chunk name — NOT just the
    # first. Stopping at the first was the same origin-chunk assumption in another
    # costume: a volume sparse at the origin of every scale falls through pass 1
    # entirely, and if its cheapest scale is one of the unwritten ones the listing
    # finds nothing and reports the volume as un-gzipped. That is what happened to a
    # registered neuropil mask — declared five scales, wrote only `64_64_64`, and none
    # of its 26 chunks at the origin. Detection answered `neuroglancer_precomputed`,
    # tensorstore asked for unsuffixed keys, and all 80 level-0 tasks recorded "empty":
    # a full run that wrote nothing and raised nothing.
    #
    # Cost is unchanged where it matters. An unwritten scale's prefix is empty and
    # lists instantly, and a volume whose chunks ARE at the origin never reaches here.
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
    """True if this precomputed scale's ORIGIN chunk object is stored.

    The same two O(1) existence checks :func:`precomputed_chunks_are_gzipped` makes,
    asked for the other reason — either suffix counts, since a CloudVolume-written
    scale is still a written one.

    **A True is evidence of a written scale; a False is not proof of an empty one.** A
    sparse volume may legitimately store nothing at its origin, which is why
    :func:`finest_populated_scale` falls back rather than raising and why
    :func:`require_populated_scale` needs a second scale to disagree with before it
    will refuse anything.
    """
    from .location import exists

    key = _first_chunk_key(scale)
    if key is None:
        return False
    prefix = scale.get("key", "")
    return bool(exists(location, prefix, key) or exists(location, prefix, key + ".gz"))
def scale_is_empty(location: str | Mapping[str, Any],
                   scale: Mapping[str, Any]) -> bool:
    """True if this precomputed scale's prefix holds NO chunk object at all.

    The conclusive form of the question :func:`scale_stores_chunks` answers cheaply,
    and the two exist as a pair because **the origin chunk is not a sound test for an
    unwritten scale**. A sparse volume elides all-background chunks, so a ground-truth
    crop sitting away from the origin stores nothing there while being perfectly
    written — reading the origin alone, such a scale is indistinguishable from one that
    was only ever declared.

    Costs a listing, so it is used only after the cheap probe has already failed. Note
    it is **cheap exactly when it matters**: an unwritten scale's prefix is empty and
    the listing returns immediately. The expensive case is a genuinely sparse volume
    whose origin happens to be background, where the full prefix is enumerated
    (``limit`` does not bound a listing — invariant STORE-ACCESS). Naming the level with
    ``--src-level`` skips the probe entirely and costs nothing.
    """
    from .location import list_keys

    for key in list_keys(location, scale.get("key", "")):
        name = key.rsplit("/", 1)[-1]
        # A chunk key looks like `0-2048_0-2048_0-128`; a shard index or a stray file
        # is not a chunk and says nothing about whether this scale holds data.
        if name and "_" in name and "-" in name:
            return False
    return True
def finest_populated_scale(location: str | Mapping[str, Any],
                           scales: Sequence[Mapping[str, Any]]) -> int | None:
    """Index into ``scales`` of the finest one that actually holds data.

    ``None`` when no scale stores anything findable, which is a caller's problem to
    decide about rather than an answer.

    **An `info` may declare scales that were never written**, and level 0 is then a
    scale that does not exist. A sample3 neuropil mask declares seven (8 nm through
    512 nm) and stores exactly one — ``64_64_64``, 18 chunk objects. The finest scale
    opened without complaint and reported the full ``(11260, 9000, 13750)`` extent its
    ``info`` claims, so ``convert`` read *that*, every block came back as the fill
    value, and the copy would have been a structurally perfect volume of zeros.

    Three tiers, cheapest first, and an ordinary volume never leaves the first:

    1. Probe origin chunks finest-first. The finest scale is normally written like
       every other, so this answers in ONE existence check and nothing else runs.
    2. If some coarser scale has an origin chunk and finer ones do not, those finer
       scales are either unwritten or merely sparse at the origin — which the origin
       cannot distinguish. :func:`scale_is_empty` settles it, and settles it instantly
       in the unwritten case, because an unwritten prefix is empty.
    3. If NO scale has an origin chunk, fall back to the conclusive test across every
       scale, finest-first. A volume can be sparse at the origin of *all* of its
       scales and still be perfectly written — a registered neuropil mask whose data
       starts 512 voxels in stores 26 chunks at 64 nm, none of them the origin one, so
       tier 1 sees nothing anywhere. Giving up here fell back to the finest declared
       scale, which is the very failure this function exists to prevent.

    Tier 2 is what stops a sparse ground-truth volume — no origin chunk at scale 0, one
    at scale 1, every scale correctly written — from being read at the wrong scale.
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
            if not scale_is_empty(location, scales[j]):
                return j                 # sparse, and genuinely the finest data present
        return i

    # Tier 3. Nothing was found at any origin, so every scale is still in question and
    # only a listing can separate "sparse here" from "never written". Finest-first, so
    # the answer is the same one tiers 1-2 would have given.
    for i in order:
        if not scale_is_empty(location, scales[i]):
            return i
    return None
def require_populated_scale(spec: Mapping[str, Any], *, op: str = "convert") -> None:
    """Raise if ``spec`` names a precomputed scale this volume never wrote.

    The failure this prevents is entirely silent. An unwritten scale opens, reports the
    extent its ``info`` declares, and reads as the fill value at every block — so the
    run succeeds, the manifest fills up, and the output is a correct-looking volume of
    zeros. Nothing downstream can tell it from a volume that really is empty.

    **Raises only on conclusive evidence**: this scale's prefix holds no chunk object
    at all, while some other scale's does. An absent *origin* chunk is not enough —
    a sparse volume elides background chunks and may legitimately store nothing there,
    and refusing one of those would be a worse failure than the one being prevented.
    The cheap origin probe runs first and returns immediately on any ordinary volume,
    so the listing behind the conclusive form is only reached when there is already
    something to explain.
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
    if scale_stores_chunks(kv, scales[idx]) or not scale_is_empty(kv, scales[idx]):
        return

    populated = [i for i in range(len(scales))
                 if i != idx and (scale_stores_chunks(kv, scales[i])
                                  or not scale_is_empty(kv, scales[i]))]
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
#: The object whose presence identifies each volume format at a location. Order
#: matters and matches :func:`detect_backend`: ``info`` is checked first, so a
#: directory holding both markers reads as precomputed.
FORMAT_MARKERS = {"neuroglancer_precomputed": "info", "zarr3": "zarr.json"}
#: Formats that are ONE array in ONE container: no pyramid, and no chunk keyspace to
#: enumerate. Detection reaches them (see :func:`detect_file_backend`) so they can be
#: *inspected* and *read*, but every op that rewrites a volume in place —
#: ``downsample``, ``relabel``, ``mask-by-value``, ``write``, ``progress`` — needs the
#: other kind and says so through :func:`require_chunked_volume`. Without that guard
#: they fail deep inside an occupancy listing that finds no chunk keys, which reads as
#: "this volume is empty" rather than "this is not that kind of volume".
SINGLE_LEVEL_FORMATS = frozenset({"hdf5", "image_stack"})
def other_format_markers(location: str | Mapping[str, Any], fmt: str) -> list[str]:
    """Markers of volume formats *other* than ``fmt`` already present at ``location``.

    Two volumes in one directory is not hypothetical: creating a volume writes only
    its own marker, so making a precomputed volume where a zarr one already lives
    leaves ``info`` and ``zarr.json`` side by side. :func:`detect_backend` checks
    ``info`` first, so from then on the zarr is unreachable through every path in this
    package while its chunks still occupy the store — and nothing says so.
    """
    # A DVID instance has no keyspace to shadow: the question this answers — "is there
    # a second volume of another format underneath this one" — cannot arise, and asking
    # it would mean reading `dvid://...` as a local path. Nor does a single container:
    # an HDF5 file is a file, and probing `piece.h5/info` is two reads that can only
    # miss. An image stack's directory *could* hold a marker, but then detection would
    # have returned that format instead, so there is nothing left to shadow.
    if fmt in ("dvid", "webknossos", "mapped") or fmt in SINGLE_LEVEL_FORMATS:
        return []
    kv = as_directory(to_kvstore(location))
    fmt = "neuroglancer_precomputed" if str(fmt).startswith("neuroglancer_precomputed") \
        else fmt
    return [marker for f, marker in FORMAT_MARKERS.items()
            if f != fmt and read_bytes(kv, marker) is not None]
def detect_file_backend(location: str | Mapping[str, Any]) -> str | None:
    """``hdf5`` or ``image_stack`` for a local file, glob or directory of slices.

    These two have **no marker object**, so they cannot be detected the way the store
    formats are: HDF5's signature is inside the file, and a stack of PNGs is a directory
    with nothing in it that says so. What is left is the name plus one ``stat`` — which
    is why this runs *last*, after every marker probe has missed, and why it is
    deliberately narrow:

    * a **glob** is an image stack and nothing else — no other format is ever addressed
      with one;
    * a **file** is judged by extension, and only the two known sets count. An unknown
      extension returns ``None`` rather than being guessed at;
    * a **directory** must actually contain a slice. "Any directory is an image stack"
      is what ``ops/write.py`` used to assume, and it turns every typo into a stack whose
      reader then reports "no image files matched" — so the listing is the evidence, and
      an empty or unrelated directory is still ``None``.

    **Local paths only.** Neither reader has an object-store driver — h5py has none at
    all, and the stack reader globs the filesystem — so detecting either on ``s3://``
    would name a backend that cannot open it. The probes are also ``os.stat`` calls,
    which is why the locality test comes first rather than being a detail of the readers.
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
        # The reader takes every slice extension it knows and sorts them together, so a
        # mixed directory interleaves two stacks into one volume. It is a legal thing to
        # ask for and almost never what anyone means, and nothing later can see it.
        import logging

        logging.getLogger(__name__).warning(
            "%s holds slices of more than one type (%s); they will be read as ONE stack, "
            "sorted together. If they are two stacks, point at a glob of one of them",
            path, ", ".join(sorted(found)))
    return "image_stack"
def detect_backend(location: str | Mapping[str, Any]) -> str | None:
    """Detect a source's format from its marker file (no data read).

    ``info`` -> neuroglancer-precomputed; ``zarr.json`` -> zarr v3;
    ``.zarray``/``.zgroup`` -> zarr v2; a ``dvid://`` URL -> ``dvid``. Returns ``None``
    if none match.

    Precomputed whose chunks are ``.gz``-suffixed reports :data:`PRECOMPUTED_GZ`
    instead, so callers fail loudly rather than reading zeros.

    **A marker always wins.** The two container formats — ``hdf5`` and ``image_stack``
    — have no marker to probe for and are recognised from the name by
    :func:`detect_file_backend`, which runs only after every marker probe has missed. So
    a directory holding both an ``info`` and a stray TIFF is still precomputed, and the
    order cannot be reversed by adding a file.
    """
    # DVID is decided by scheme, BEFORE `to_kvstore`, because it is not a store at all:
    # `to_kvstore` would read `dvid://server/uuid/instance` as a *local file path*,
    # probe the filesystem for `info`/`zarr.json`, find nothing and return None — which
    # reaches the user as "could not detect source format", pointing nowhere near the
    # real problem.
    # From `.dvid`, not `.backends.dvid`: this runs on nearly every op's first step, and
    # the backend module pulls in numpy and registers itself just to answer a string test.
    from .dvid import is_url as _is_dvid_url

    if _is_dvid_url(location):
        return "dvid"
    # Same reason: a webknossos layer is a server, and `wk://` must not reach to_kvstore.
    from .webknossos import is_url as _is_wk_url

    if _is_wk_url(location):
        return "webknossos"
    # A mapped view is a descriptor FILE, which every marker probe below would miss
    # (they look for `info`/`zarr.json` under it as a directory).
    from .mapped import is_descriptor

    if is_descriptor(location):
        return "mapped"

    from .scales import finest_first

    kv = as_directory(to_kvstore(location))
    raw = read_bytes(kv, "info")
    if raw is not None:
        try:
            scales = json.loads(raw)["scales"]
            # COARSEST FIRST, then the rest. Gzipping is a property of how the whole
            # volume was written, so any scale that STORES a chunk answers the question
            # — but one that stores nothing answers nothing, and an `info` may well
            # declare scales that were never written. So all of them are offered, in the
            # order that keeps the listing fallback on the cheapest prefix.
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
    # `location`, not `kv`: the trailing slash added above makes an ordinary file path
    # fail every `os.path.isfile` test, and a spec's own kvstore has already been
    # normalised past the form the filesystem answers about.
    return detect_file_backend(location)
def require_chunked_volume(fmt: str, location: str, op: str) -> str:
    """``fmt`` if it is a chunked multiscale volume; a useful error if it is not.

    Detection now reaches HDF5 files and image stacks (:func:`detect_file_backend`), so
    an op that rewrites a volume in place can be handed one — and every such op finds
    out the same confusing way, deep inside an occupancy listing that finds no chunk
    keys and reports the volume as holding no data. The formats are not deficient
    volumes, they are a different thing: one array in one container, with no pyramid to
    downsample and no keyspace whose contents answer "where is the data".

    Called by the ops that *write*; the read-only ones (``info``, ``align-bbox``,
    ``create --like``, ``convert``, ``to-hdf5``) deliberately do not, since inspecting
    and reading these is the whole point of detecting them.
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
