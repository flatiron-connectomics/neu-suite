"""Read coordinate metadata + the level-0 data location from a source volume.

``convert`` uses this so you don't have to re-specify ``voxel_size``/``offset``
when the source already carries them (OME-NGFF zarr groups, precomputed ``info``,
and an HDF5 file that records its own frame — which is what ``neu-vol to-hdf5``
writes). Any field the caller passes explicitly overrides what's read here. Sources
that record nothing (image stacks, bare arrays, an HDF5 file with no frame attributes)
return ``None``.

Returned dict keys: ``data_spec`` (the array/scale spec to actually read),
``voxel_size``/``offset`` (canonical z,y,x), ``units``, ``spatial_axes``,
``has_channels``.

**Two tiers, and the difference is cost.** Everything above ``describe`` reads only
small metadata documents through the kvstore — one object per call. ``describe`` and
``existing_levels`` at the bottom compose those into a whole-volume picture, and to do
it they *open every level* through a backend. That is not free: measured on an
8-level S3 zarr, the array opens took 15.8 s against 1.4 s for the metadata. Ask the
narrow question when the narrow answer will do.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Mapping

from .detect import (PRECOMPUTED_GZ, SINGLE_LEVEL_FORMATS, detect_backend,
                     finest_populated_scale, other_format_markers)
from .location import as_directory, join, spec_kvstore

logger = logging.getLogger(__name__)

if TYPE_CHECKING:                       # the class is imported inside `describe`, where
    from .report import Description     # it is used, so the module stays import-light


def location_spec(location: str, fmt: str, *,
                  dataset: str | None = None) -> dict[str, Any]:
    """The backend spec addressing ``location``, in the form ``fmt`` expects.

    Four forms, because not every source is a store: most take ``path``, an image
    stack takes ``source`` (a directory or glob), HDF5 takes ``path`` plus the
    ``dataset`` inside the container, and DVID takes
    ``server``/``uuid``/``instance``. Everything that turns a user-supplied location
    into a spec goes through here, so a new non-store backend is one branch rather
    than a search for every place that wrote ``{"backend": fmt, "path": ...}``.

    ``dataset`` names the array inside an HDF5 file and is resolved by
    :func:`~neu_vol.backends.hdf5.sole_dataset` when it is not given — which answers
    only when the file holds exactly one candidate and otherwise raises listing them.
    That is deliberate: a container is ambiguous in a way a path is not, and guessing
    which array someone meant is worse than asking. Ignored for every other format.
    """
    if fmt == "dvid":
        from .dvid import parse_url

        return {"backend": fmt, **parse_url(location)}
    if fmt == "webknossos":
        from .webknossos import parse_url as parse_wk_url

        return {"backend": fmt, **parse_wk_url(location)}
    if fmt == "mapped":
        from .mapped import location_spec as mapped_spec

        return mapped_spec(location)
    if fmt == "image_stack":
        return {"backend": fmt, "source": location}
    if fmt == "hdf5":
        if dataset is None:
            from .backends.hdf5 import sole_dataset

            dataset = sole_dataset(location)
        return {"backend": fmt, "path": location, "dataset": dataset}
    return {"backend": fmt, "path": location}


def read_source_metadata(spec: Mapping[str, Any]) -> dict | None:
    backend = spec.get("backend")
    if backend == "zarr3":
        return _read_zarr_ome(spec)
    if backend == "dvid":
        return _read_dvid(spec)
    if backend == "webknossos":
        return _read_webknossos(spec)
    if backend == "mapped":
        return _read_mapped(spec)
    # PRECOMPUTED_GZ reads the SAME `info` document — only the chunk keys differ, and
    # metadata does not live in chunks. Omitting it here silently loses voxel_size,
    # offset and kind, which makes `convert` demand --voxel-size and, worse, lose the
    # image/segmentation distinction that decides mean vs mode downsampling.
    if backend in ("neuroglancer_precomputed", PRECOMPUTED_GZ):
        return _read_precomputed(spec)
    if backend == "hdf5":
        return _read_hdf5(spec)
    # An image stack records no physical scale at all — an image file has pixels and
    # nothing else — so there is genuinely nothing to read, and `convert` asking for
    # --voxel-size is the correct outcome rather than a shortcoming.
    return None


def read_level_voxel_sizes(spec: Mapping[str, Any]) -> list[tuple[float, ...]] | None:
    """Each level's own voxel size, from the volume's metadata. ``None`` if absent.

    **Not derived from shape ratios.** Shapes are ceil-divided, so a level-0 extent of
    13750 over a factor of 4 stores 3438, and 13750/3438 is 3.9994 — deriving from that
    reports 31.9953 nm for a level that is exactly 32. Real pyramids are also
    anisotropic, so the ``2**level`` shortcut is wrong for a different reason. Both
    formats record the true value per level; read it.

    precomputed lists a ``resolution`` per entry of ``info["scales"]``; OME-NGFF lists
    a scale transform per ``multiscales[0]["datasets"]`` entry. Returned finest-first
    in ``(z, y, x)``, spatial axes only.
    """
    backend = spec.get("backend")
    if backend == "dvid":
        # DVID's downres really is a factor of 2 on every axis at every level — that is
        # its documented model, not the `2**level` assumption invariant 1 warns about,
        # and it must not be generalised to any other backend. The pyramid depth is
        # whatever the instance recorded in MaxDownresLevel.
        from .backends.dvid import geometry
        from .dvid import instance_info

        geom = geometry(instance_info(spec), spec)
        return [tuple(v * 2 ** i for v in geom["voxel_size"])
                for i in range(geom["max_scale"] + 1)]
    if backend == "webknossos":
        from .backends.webknossos import geometry as wk_geometry
        from .webknossos import dataset_info, resolve_dataset_id

        dataset_id = spec.get("dataset_id") or resolve_dataset_id(str(spec["dataset"]))
        return [lv["voxel_size"] for lv in
                wk_geometry(dataset_info(dataset_id), str(spec["layer"]))["levels"]]
    if backend == "mapped":
        # A lookup changes ids, not geometry: the levels are the source's.
        return read_level_voxel_sizes(spec["source"])
    if backend in ("neuroglancer_precomputed", PRECOMPUTED_GZ):   # same `info`
        from .scales import finest_first, precomputed_info, precomputed_level

        info = precomputed_info(spec_kvstore(spec))
        if info is None:
            return None
        scales = info["scales"]
        return [precomputed_level(scales[i])[1].voxel_size_nm for i in finest_first(scales)]
    if backend == "hdf5":
        # One level, and it is the file's own. Returned as a list of one so a caller
        # asking "what does level N measure" gets the same shape of answer for every
        # format — and so `--bbox-scale 2` against an HDF5 piece reports "the source
        # records only 1 level(s)" rather than "records none", which is a different
        # and misleading complaint.
        meta = _read_hdf5(spec)
        return [tuple(meta["voxel_size"])] if meta else None
    if backend == "zarr3":
        from .scales import ome_levels, ome_multiscale, read_zarr_json

        ms = ome_multiscale(read_zarr_json(spec_kvstore(spec)))
        return [frame.voxel_size_nm for _, frame in ome_levels(ms)] if ms else None
    return None


def _read_zarr_ome(spec: Mapping[str, Any]) -> dict | None:
    from .scales import ome_levels, ome_multiscale, read_zarr_json, spatial_axes

    ms = ome_multiscale(read_zarr_json(spec_kvstore(spec)))
    if ms is None:
        return None  # absent, a bare array, or a non-OME group: no coordinate metadata
    axes = ms["axes"]
    path0, frame0 = ome_levels(ms)[0]
    spatial = spatial_axes(ms)
    has_channels = any(a.get("type") == "channel" for a in axes)
    return {
        "data_spec": {"backend": "zarr3", "kvstore": join(spec_kvstore(spec), path0)},
        "voxel_size": frame0.voxel_size_nm,
        "offset": frame0.origin_nm,
        "units": axes[spatial[0]].get("unit"),
        "spatial_axes": tuple(axes[i]["name"] for i in spatial),
        "has_channels": has_channels,
        # The multiscales "type" is the downsampling method; we write `kind` there,
        # but another writer may put something else, so only trust our own values.
        "kind": ms.get("type") if ms.get("type") in ("image", "segmentation") else None,
    }


def _read_hdf5(spec: Mapping[str, Any]) -> dict | None:
    """The frame an HDF5 file records about itself, or ``None`` if it records none.

    An HDF5 file is a container, not a format with a metadata document, so there is no
    ``info`` to read — the numbers live in attributes beside the array, and
    ``HDF5Backend.stored_*`` is what finds them (dataset attributes, then the root
    group's, then a top-level dataset of that name; writers use all three).
    ``neu-vol to-hdf5`` writes ``voxel_size``, ``units``, ``axes`` and ``voxel_offset``,
    so a file this package packed round-trips through ``neu-vol info``, ``create --like``
    and ``convert`` without anyone retyping a coordinate.

    **``voxel_size`` is the gate.** Without it there is no frame — an offset in voxels
    means nothing physical and ``units`` describes numbers that are not there — so the
    whole dict is withheld rather than half-filled, which is what keeps ``convert``'s
    "voxel_size is required" the honest answer for a bare array.

    **The axis order is read, never assumed, and an uninterpretable one raises.** The
    field name ``voxel_offset`` is precomputed's, where it means *xyz*, while everything
    in this package is zyx — so the numbers alone cannot say. A recorded ``axes``
    settles it; nothing recorded falls back to zyx, which is what the reader already
    assumes of the array itself, and warns when the fallback could matter (an
    anisotropic vector is a different volume reversed; an isotropic one is not). An
    ``axes`` that is neither order is refused outright: it is a *stated* fact that this
    cannot honour, and quietly substituting zyx for it would mirror the data through the
    z=x diagonal with nothing downstream able to tell.
    """
    import logging

    from .backends.base import open_backend

    backend = open_backend(dict(spec))
    size = backend.stored_voxel_size(spec.get("voxel_size_field", "voxel_size"))
    if not size:
        return None
    log = logging.getLogger(__name__)

    stated = backend.stored_axes()
    if stated and stated[0] not in ("zyx", "xyz"):
        raise ValueError(
            f"{spec.get('path')} records axes {stated[0]!r} ({stated[1]}), which is "
            f"neither 'zyx' nor 'xyz', so nothing here can say which way round its "
            f"voxel_size and voxel_offset are. Fix the attribute, or read the file with "
            f"an explicit spec")
    order = stated[0] if stated else "zyx"

    voxel_size = tuple(float(v) for v in size[0])
    if not stated and voxel_size != voxel_size[::-1]:
        log.warning(
            "%s records voxel_size %s but no `axes`, so it is read as zyx (this "
            "package's convention). Reversed it would be %s — a different volume. "
            "`neu-vol to-hdf5` records `axes`; add one to remove the doubt",
            spec.get("path"), voxel_size, voxel_size[::-1])
    if order == "xyz":
        voxel_size = voxel_size[::-1]

    found = backend.stored_offset(spec.get("offset_field", "voxel_offset"))
    voxel_offset = tuple(int(v) for v in found[0]) if found else (0,) * len(voxel_size)
    if found and order == "xyz":
        voxel_offset = voxel_offset[::-1]

    units = backend.stored_units()
    shape = tuple(int(s) for s in backend.shape)
    axes = tuple(order)

    # An HDF5 file has no agreed-on place to say image or segmentation, so a `kind` this
    # package did not write is not evidence — `kind` is a plain word another tool may have
    # used for something else. Only a member of the vocabulary is trusted, the same rule
    # `_read_zarr_ome` applies to the multiscales `type`, and anything else reads as
    # unrecorded rather than raising: it costs a `convert --kind`, where trusting it wrongly
    # would average label ids into ids that were never in the data.
    from neu_lib import KINDS

    stated_kind = backend.stored_kind()
    kind = stated_kind[0] if stated_kind and stated_kind[0] in KINDS else None
    if stated_kind and kind is None:
        log.info("%s records kind %r (%s), which is not one of %s, so it is ignored",
                 spec.get("path"), stated_kind[0], stated_kind[1], ", ".join(KINDS))
    return {
        # The spec as given: an HDF5 dataset IS the array, so there is no level or scale
        # to descend into, and `data_spec` carries the resolved `dataset` through the
        # same way every other reader carries the caller's backend (invariant 9).
        "data_spec": dict(spec),
        "voxel_size": voxel_size,
        "offset": tuple(float(o) * v for o, v in zip(voxel_offset, voxel_size)),
        "voxel_offset": voxel_offset,
        "units": units[0] if units else None,
        "spatial_axes": axes,
        "has_channels": len(shape) > len(axes),
        # Recorded when the file says so in this package's own vocabulary, and `None`
        # otherwise — never inferred from the dtype, which is the guess that averages label
        # ids into ids that were never in the data. Unrecorded means `convert --kind`.
        "kind": kind,
    }


def _read_dvid(spec: Mapping[str, Any]) -> dict | None:
    """Coordinate metadata for a DVID labelmap, from one ``/info`` request.

    ``kind`` is **always** ``segmentation``: a labelmap is one by definition, and the
    consequence of getting it wrong is silent — the pyramid would average label ids
    into ids that were never in the data (CLAUDE.md invariant 9's sibling hazard, and
    the reason `copy` exists at all).

    ``offset`` is 0: the backend presents DVID's coordinates origin-anchored, so the
    output frame already matches DVID's own. ``MinPoint`` is a fact about where data
    *starts*, not a translation to apply.

    **The branch ref is resolved to a concrete uuid here**, and that is what goes into
    ``data_spec`` — so every worker reads the node the driver chose. Leaving the ref in
    place would let a lock-and-spawn landing mid-run move HEAD, and the output would be
    stitched from two versions of the segmentation with nothing to show it. Reading from
    an *open* node is legal but not reproducible, so it is warned about.
    """
    import logging

    from .backends.dvid import geometry
    from .dvid import instance_info, resolve_node

    node = resolve_node(spec, prefer_locked=bool(spec.get("prefer_locked")))
    # Everything downstream addresses the concrete node. `prefer_locked` is dropped
    # because it has now been *applied* — carrying it would invite a second resolution
    # on a worker, which is the drift this exists to prevent.
    resolved = {k: v for k, v in spec.items() if k != "prefer_locked"}
    resolved["uuid"] = node["uuid"]
    # Keep what was actually asked for. "the newest locked node on main" and "this node"
    # are different provenance claims even when they name the same uuid today, and the
    # resolved spec is what the record is built from.
    resolved["requested_ref"] = node["ref"]
    resolved["ancestors_walked"] = node["walked"]

    # Resolution is reported; the fact that a node is OPEN is *warned about* by the op
    # that is going to export it, not here. This function is also on the path of
    # read-only inspection (`neu-vol info`, the CLI's crop peeks), where a warning about
    # reproducibility is noise — `info` prints the open/locked status in its own table.
    log = logging.getLogger(__name__)
    if node["uuid"] != str(spec.get("uuid")):
        log.info("dvid %s resolved to node %s (%s)%s", spec.get("uuid"), node["uuid"],
                 "locked" if node["locked"] else "OPEN",
                 f", {node['walked']} ancestor(s) back" if node["walked"] else "")

    geom = geometry(instance_info(resolved), resolved)
    return {
        # Carries the caller's backend through, like every other reader here
        # (invariant 9): `data_spec` selects the reader and must not contradict what
        # detection decided.
        "data_spec": {**resolved, "backend": spec.get("backend") or "dvid",
                      "scale_index": int(spec.get("scale_index", 0))},
        "provenance_spec": resolved,
        "voxel_size": geom["voxel_size"],
        "offset": (0.0,) * len(geom["voxel_size"]),
        "units": "nm",
        "spatial_axes": ("z", "y", "x"),
        "has_channels": False,
        "kind": "segmentation",
    }


def _read_webknossos(spec: Mapping[str, Any]) -> dict | None:
    """Coordinate metadata for a webknossos layer, from the dataset's one record.

    **The dataset NAME is resolved to its id here**, and the id, the server and the
    datastore go into ``data_spec`` — the same discipline as DVID's ref: names are not
    unique across organizations, so workers and the provenance must address the one the
    driver chose. The token does not go in; each process reads the credentials itself.

    A requested ``mapping`` is checked against the layer's own list before any work, since
    an unknown name would otherwise fail every task of the run.
    """
    from .backends.webknossos import geometry
    from .webknossos import (check_mapping, credentials, dataset_info,
                             resolve_dataset_id)

    creds = credentials()
    dataset_id = spec.get("dataset_id") or resolve_dataset_id(str(spec["dataset"]),
                                                              creds=creds)
    info = dataset_info(dataset_id, creds=creds)
    resolved = {**spec, "dataset_id": dataset_id, "url": creds["url"],
                "datastore": info["dataStore"]["url"],
                "organization": info.get("owningOrganization")}
    if resolved.get("mapping"):
        check_mapping(resolved, str(resolved["mapping"]))
    else:
        resolved.pop("mapping", None)
    level = int(spec.get("scale_index", 0))
    geom = geometry(info, str(spec["layer"]))
    if not 0 <= level < len(geom["levels"]):
        raise ValueError(f"{spec.get('dataset')}/{spec.get('layer')} has "
                         f"{len(geom['levels'])} mags; level {level} was requested")
    return {
        "data_spec": {**resolved, "backend": spec.get("backend") or "webknossos",
                      "scale_index": level},
        "provenance_spec": resolved,
        "voxel_size": geom["levels"][level]["voxel_size"],
        # Origin-anchored, as the backend presents it: voxel indices are webknossos' own.
        "offset": (0.0, 0.0, 0.0),
        # The layer's bounding box at this level, zyx [lo, hi). Everything outside reads
        # as zeros, so `convert` crops to it by default.
        "data_box": (geom["levels"][level]["box_lo"], geom["levels"][level]["box_hi"]),
        "units": "nm",
        "spatial_axes": ("z", "y", "x"),
        "has_channels": False,
        "kind": geom["kind"],
        "default_mapping": geom["default_mapping"],
    }


def _read_mapped(spec: Mapping[str, Any]) -> dict | None:
    """The source's metadata, with its ``data_spec`` wrapped in the table.

    The source's resolution happens first and is kept — which scale stores data, which
    level is 0, its frame — so a mapped view inherits all of it rather than re-deciding.
    ``kind`` is ``segmentation``: a table maps labels to labels by construction.
    """
    source = dict(spec["source"])
    if "scale_index" in spec:
        # `--src-level` / `describe(level=)` set it on the outer spec; it means the source's.
        source["scale_index"] = int(spec["scale_index"])
        spec = {k: v for k, v in spec.items() if k != "scale_index"}
    meta = read_source_metadata(source)
    if not meta:
        raise ValueError(f"{spec.get('descriptor')} maps {spec.get('source_location')}, "
                         f"which records no metadata; a mapped view needs a volume")
    wrapped = {**spec, "source": meta["data_spec"]}
    return {**meta, "data_spec": wrapped, "provenance_spec": wrapped,
            "kind": "segmentation"}


def _read_precomputed(spec: Mapping[str, Any]) -> dict | None:
    from .scales import finest_first, precomputed_info, precomputed_level

    kv = as_directory(spec_kvstore(spec))
    info = precomputed_info(kv)
    if info is None:
        return None
    scales = info["scales"]
    finest = finest_first(scales)[0]

    # Level 0 = the finest scale that was actually WRITTEN, not simply the finest one
    # declared. `info` is a claim about the volume's layout and the chunk objects are
    # the volume; where they disagree, believe the store. An explicit `scale_index`
    # (what `--src-level` sets) always wins, so the caller can name a scale the probe
    # would not have chosen — including one it thinks is empty, which `neu-vol write`
    # legitimately does when filling a level that does not exist yet.
    idx = spec.get("scale_index")
    if idx is None:
        idx = finest_populated_scale(kv, scales)
        if idx is None:
            idx = finest
        elif idx != finest:
            logger.warning(
                "level 0 is scale %s (%s): the %s finer scale(s) this volume's `info` "
                "declares store no data. Reading %s would have returned the fill value "
                "everywhere. Override with --src-level.",
                idx, scales[idx].get("key"), idx, scales[finest].get("key"))
    idx = int(idx)
    if not 0 <= idx < len(scales):
        raise ValueError(
            f"source level {idx} does not exist: this volume's `info` declares "
            f"{len(scales)} scale(s), 0-{len(scales) - 1}")
    _, frame = precomputed_level(scales[idx])
    nch = int(info.get("num_channels", 1))
    return {
        # Carry the CALLER's backend through. Hardcoding "neuroglancer_precomputed"
        # here silently routed .gz volumes back to tensorstore — detection said
        # PRECOMPUTED_GZ, but the spec the workers actually read through said
        # otherwise, so every block read as zeros. `data_spec` selects the reader;
        # it must not contradict what detection decided.
        "data_spec": {"backend": spec.get("backend") or "neuroglancer_precomputed",
                      "kvstore": spec_kvstore(spec), "scale_index": idx},
        "voxel_size": frame.voxel_size_nm,
        "offset": frame.origin_nm,
        "units": "nm",
        "spatial_axes": ("z", "y", "x"),
        "has_channels": nch > 1,
        # precomputed records the volume type directly, which is what decides
        # whether downsampling may average (image) or must take a mode (labels).
        "kind": "segmentation" if info.get("type") == "segmentation" else "image",
    }


# --------------------------------------------------------------------------- #
# The whole-volume picture: format + coordinates + the levels actually present.
#
# These compose the readers above and additionally OPEN each level, which is the
# expensive part (see the module docstring). `info`, `downsample`, `create --like`
# and `write` all go through here, so none of them can disagree about what is on disk.
# --------------------------------------------------------------------------- #


def level_spec(volume: str, fmt: str, level: int,
               *, dataset: str | None = None) -> dict[str, Any]:
    """The backend spec for one level of a multiscale volume.

    The two formats address a level differently: zarr v3 puts each level in its own
    subdirectory, while precomputed keeps every scale under one ``info`` and selects
    with ``scale_index``. Everything that opens a level goes through here rather than
    building the path itself.

    A single-container format has exactly one level and any other is an **error**, not
    a level that happens to be missing. That matters because the specs are permissive:
    ``HDF5Backend`` ignores a key it does not know, so a spec carrying ``scale_index=7``
    opened level 0 and reported it as level 7 — which made :func:`existing_levels` probe
    twelve levels, open the same array twelve times, and report a one-level file as a
    twelve-level pyramid of identical shapes.
    """
    if fmt in SINGLE_LEVEL_FORMATS:
        if int(level) != 0:
            raise ValueError(
                f"{volume} is {fmt}, which is a single array and has only level 0; "
                f"level {level} does not exist")
        return location_spec(volume, fmt, dataset=dataset)
    if fmt == "zarr3":
        return {"backend": fmt, "path": f"{volume.rstrip('/')}/{level}"}
    if fmt in ("dvid", "webknossos"):
        # A DVID level is a downres scale of one instance (a webknossos level, one mag of
        # a layer), addressed the way precomputed addresses a scale — by index.
        return {**location_spec(volume, fmt), "scale_index": int(level)}
    if fmt == "mapped":
        # The same table over the source's own level (approximate above 0; see mapped.py).
        spec = location_spec(volume, fmt)
        return {**spec, "source": level_spec(spec["source_location"],
                                             spec["source_format"], level)}
    return {"backend": fmt, "path": volume, "scale_index": int(level)}


def describe(volume: str, *, dataset: str | None = None,
             level: int | None = None) -> "Description":
    """Detected format, coordinate metadata and the levels actually present.

    Returns a :class:`~neu_vol.report.Description`, which **is** a dict — subscript it as
    before — that additionally renders itself as the table ``neu-vol info`` prints, so
    ``print(describe(v))`` in a notebook is legible and ``describe(v).frame()`` is a
    DataFrame of the per-level (or per-dataset) rows.

    Works for every format detection reaches, which includes an HDF5 file and a
    directory or glob of 2D slices. Those have one level and may record no coordinate
    metadata at all, so ``meta`` is ``None`` and ``level_voxel_sizes`` empty for them —
    ``shape`` and ``dtype`` come from opening the array itself.

    **An HDF5 file may be a CONTAINER of several arrays, and that is described too.** A
    bag of annotated crops, each keeping its own ``voxel_offset`` so it can be placed
    back, is what an HDF5 file here usually is — so refusing to describe one until a
    ``dataset`` is named would refuse exactly the file you run this on to *find out* the
    names. Instead:

    * ``datasets`` is ``{name: {"shape", "dtype", "chunks", <recorded frame>}}`` for
      every HDF5 file, however many it holds, and ``None`` for other formats. Frame
      values there are **as recorded**, each with its own ``axes``;
    * ``dataset`` is the array the rest of the dict describes, or ``None``;
    * with one candidate — or with ``dataset`` given — that array is resolved and every
      field is filled as usual;
    * with several and none named, ``shape`` / ``dtype`` / ``meta`` / ``level_voxel_sizes``
      are ``None`` and ``levels`` is empty, because thirteen differently-shaped arrays
      have no single answer to any of them. Callers that need one array should say so
      through :func:`require_one_array`, which names the datasets and how to choose.

    Raises ``FileNotFoundError`` if nothing at ``volume`` looks like a volume.
    """
    from .backends.base import open_backend
    from .logs import quiet_reads
    from .report import Description

    # Wrapped because `describe` IS the entry point from a notebook's point of view —
    # there is no `main()` to wrap, and an S3 open logs two `AuthCredentialsProvider`
    # lines at `E` severity per prefix that are not failures, only the two providers that
    # missed before the environment one succeeded. Erik's rule: a user must not have to
    # invoke anything to get legible output, and a manual call is worse than the noise,
    # because forgetting it after a kernel restart looks identical to a broken filter.
    # `quiet_reads` nests and is thread-safe, so the CLI wrapping `main()` too is fine.
    with quiet_reads():
        return _describe(volume, dataset=dataset, level=level)


def _describe(volume: str, *, dataset: str | None = None,
              level: int | None = None) -> "Description":
    from .backends.base import open_backend
    from .report import Description

    fmt = detect_backend(volume)
    if fmt is None:
        raise FileNotFoundError(f"no volume found at {volume}")

    entries = None
    if fmt == "hdf5":
        from .backends.hdf5 import describe_datasets

        entries = describe_datasets(volume)
        if not entries:
            raise FileNotFoundError(f"{volume} contains no 3D+ dataset")
        if dataset is None and len(entries) > 1:
            # Everything below needs ONE array. Reported rather than raised: the listing
            # is the useful answer here, and it is the only way to learn the names.
            return Description(
                location=volume, format=fmt, meta=None, shape=None, dtype=None,
                has_channels=False, level_voxel_sizes=None, levels={},
                spec={"backend": fmt, "path": volume},
                dataset=None, datasets=entries, other_markers=[])

    spec = location_spec(volume, fmt, dataset=dataset)
    if level is not None:
        # `shape`, `dtype` and `meta` then describe THAT scale, which is what a caller
        # deriving a copy's parameters from the source needs. `levels` below still maps
        # the volume's whole declared pyramid — it answers a different question.
        spec["scale_index"] = int(level)
    meta = read_source_metadata(spec)
    level0 = open_backend(meta["data_spec"] if meta else spec)
    shape = tuple(int(s) for s in level0.shape)
    # A single-container format's one level IS `meta`, so it is reused rather than read
    # again: `read_level_voxel_sizes` would reopen the file and re-emit any warning the
    # first read already made (an HDF5 file recording no `axes` warns about the zyx
    # fallback, and hearing it twice reads as two problems).
    if fmt in SINGLE_LEVEL_FORMATS:
        per_level = [tuple(meta["voxel_size"])] if meta else None
    else:
        per_level = read_level_voxel_sizes(spec)
    return Description(
        # What was described. Carried so the result can render its own first line, and
        # so a caller holding one no longer has to remember what it asked about.
        location=volume,
        format=fmt, meta=meta, shape=shape,
        dtype=str(getattr(level0, "dtype", "?")),
        has_channels=bool(meta["has_channels"]) if meta else False,
        level_voxel_sizes=per_level,
        levels=existing_levels(volume, fmt, dataset=dataset),
        # The resolved spec, so a caller that has already paid for the detection can
        # read the volume without repeating it — and so `info` can say WHICH dataset
        # of an HDF5 container it just described.
        spec=spec,
        dataset=spec.get("dataset"),
        # WHICH source scale `shape`/`dtype`/`meta` above describe. Usually 0, but a
        # volume whose `info` declares scales it never wrote resolves to the finest one
        # that stores data, and a caller reading level-0 properties out of `levels`
        # needs to look them up under this rather than under 0.
        source_level=int((meta or {}).get("data_spec", {}).get("scale_index") or 0),
        datasets=entries,
        # One extra small read, against several array opens — worth it to notice
        # a second volume shadowed underneath this one.
        other_markers=other_format_markers(volume, fmt))


def require_one_array(described: Mapping[str, Any], location: str, op: str) -> dict:
    """``described`` if it resolved a single array; a useful error if it did not.

    :func:`describe` describes a multi-array HDF5 container rather than refusing to look
    at one, which means ``shape`` / ``dtype`` / ``meta`` can legitimately be ``None``.
    Anything that goes on to *read* the array has to say so, or it hits a ``TypeError``
    on ``None`` somewhere in its own arithmetic — which says nothing about the container.
    """
    if described.get("shape") is not None:
        # The same object, not a copy: it is a `Description`, and copying it into a plain
        # dict would take the rendering off a value a caller may still want to print.
        return described
    listed = ", ".join(sorted(described.get("datasets") or ()))
    raise ValueError(
        f"`{op}` needs one array and {location} is a container of "
        f"{len(described.get('datasets') or ())}: {listed}. `neu-vol info {location}` "
        f"lists them with their shapes and offsets; pass --dataset where the command "
        f"takes one, or point at a file holding a single array.")


def level0_chunking(described: Mapping[str, Any], location: str):
    """``(read chunk, write chunk)`` of a described source's level 0, either possibly ``None``.

    :func:`describe` reports chunking only for levels it finds; a bare array has none,
    so its chunking is read from the array itself. The two differ when the level is
    sharded: the write chunk is the shard.
    """
    from .backends.base import open_backend

    lvl0 = described["levels"].get(described.get("source_level") or 0) or {}
    if lvl0.get("read_chunks") or lvl0.get("chunks"):
        return lvl0.get("read_chunks"), lvl0.get("chunks")
    meta = described["meta"] or {}
    try:
        be = open_backend(meta.get("data_spec")
                          or {"backend": described["format"], "path": location})
    except Exception as e:
        # The cloudvolume backend needs a package only its own environment carries, and
        # workers there can still read the source.
        logger.info("could not read the source's chunking (%s); using the profile default", e)
        return None, None

    def _maybe(attr):
        try:
            return tuple(int(c) for c in getattr(be, attr))
        except Exception:
            return None

    return _maybe("read_chunks"), _maybe("chunks")


def existing_levels(volume: str, fmt: str, probe: int = 12,
                    *, dataset: str | None = None) -> dict[int, dict]:
    """``{level: {"shape", "chunks", "read_chunks"}}`` for the levels that open.

    Probes upward until one misses. The multiscale group metadata is written at the
    very end of a conversion, so an in-flight volume has levels but no group metadata
    — probing is what makes this work on a run that is still going.

    A single-container format is not probed at all: it has one level by construction,
    and :func:`level_spec` refuses any other, so the loop below would stop after one
    iteration anyway — ``probe`` just stops being a number of store opens.

    Chunking is **per level**, not a property of the volume: it lives in each level's
    own array metadata (zarr's ``zarr.json``, precomputed's per-scale ``chunk_sizes``),
    and a conversion is free to chunk levels differently. So it is read here, where
    each level is opened anyway, rather than assumed from level 0.

    ``chunks`` is the *write* chunk and ``read_chunks`` the *read* chunk. They differ
    only when the level is sharded, where the write chunk is the shard and the read
    chunk is the unit actually fetched — which is the number that governs read
    amplification, so both are worth having.
    """
    from .backends.base import open_backend

    out: dict[int, dict] = {}
    for i in range(1 if fmt in SINGLE_LEVEL_FORMATS else probe):
        try:
            be = open_backend(level_spec(volume, fmt, i, dataset=dataset))
            shape = tuple(int(s) for s in be.shape)
        except Exception:
            break
        # chunks come from the same open, but a backend need not expose them
        def _maybe(attr):
            try:
                return tuple(int(c) for c in getattr(be, attr))
            except Exception:
                return None
        out[i] = {"shape": shape, "chunks": _maybe("chunks"),
                  "read_chunks": _maybe("read_chunks")}
    return out
