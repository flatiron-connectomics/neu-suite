"""Read coordinate metadata and the level-0 data location from a source volume.

``convert`` uses this so ``voxel_size`` / ``offset`` need not be re-specified when the
source carries them (OME-NGFF zarr groups, precomputed ``info``, DVID, webknossos, an
HDF5 file that records its own frame). Explicit arguments override what is read here;
sources that record nothing return ``None``.

Returned dict keys: ``data_spec`` (the array/scale spec to read), ``voxel_size`` /
``offset`` (z, y, x), ``units``, ``spatial_axes``, ``has_channels``, ``kind``.

Two tiers, differing in cost: the readers read one small metadata document each;
:func:`describe` and :func:`existing_levels` additionally open every level, which on an
object store costs about ten times as much (15.8 s against 1.4 s on an 8-level zarr).
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

    Most formats take ``path``; an image stack takes ``source`` (a directory or glob),
    HDF5 takes ``path`` plus the ``dataset`` inside the container, and DVID and
    webknossos take the fields of their URLs. Everything that turns a user-supplied
    location into a spec goes through here.

    ``dataset`` names the array inside an HDF5 file. When omitted it is resolved by
    :func:`~neu_vol.backends.hdf5.sole_dataset`, which raises, listing the candidates,
    unless the file holds exactly one. Ignored for every other format.
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
    """Coordinate metadata and the level-0 ``data_spec`` for ``spec``; ``None`` if absent.

    See the module docstring for the keys. An image stack records no physical scale, so
    it always returns ``None``.
    """
    backend = spec.get("backend")
    if backend == "zarr3":
        return _read_zarr_ome(spec)
    if backend == "dvid":
        return _read_dvid(spec)
    if backend == "webknossos":
        return _read_webknossos(spec)
    if backend == "mapped":
        return _read_mapped(spec)
    # PRECOMPUTED_GZ reads the same `info`: only the chunk keys differ.
    if backend in ("neuroglancer_precomputed", PRECOMPUTED_GZ):
        return _read_precomputed(spec)
    if backend == "hdf5":
        return _read_hdf5(spec)
    return None


def read_level_voxel_sizes(spec: Mapping[str, Any]) -> list[tuple[float, ...]] | None:
    """Each level's own voxel size, from the volume's metadata. ``None`` if absent.

    Read per level, never derived from shape ratios (shapes are ceil-divided, so the
    ratio is inexact) or from ``2**level`` (pyramids may be anisotropic). Finest-first,
    ``(z, y, x)``, spatial axes only.
    """
    backend = spec.get("backend")
    if backend == "dvid":
        # DVID's downres is documented as a factor of 2 on every axis at every level;
        # the depth is the instance's MaxDownresLevel. True of DVID only.
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
        # One level, as a list of one, so "level N" errors say "only 1 level", not "none".
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

    The values live in attributes found by ``HDF5Backend.stored_*`` (dataset attributes,
    then the root group's, then a top-level dataset of that name). ``neu-vol to-hdf5``
    writes ``voxel_size``, ``units``, ``axes`` and ``voxel_offset``, so a packed file
    round-trips through ``info``, ``create --like`` and ``convert``.

    ``voxel_size`` is required: without it nothing is returned. The axis order comes from
    a recorded ``axes``; with none, zyx is assumed (and warned about when the voxel size
    is anisotropic, since reversed it would be a different volume). An ``axes`` that is
    neither zyx nor xyz raises rather than risk mirroring the data.
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

    # Only a `kind` from this package's vocabulary is trusted; another tool may use the
    # word for something else, and a wrong kind would average label ids.
    from neu_lib import KINDS

    stated_kind = backend.stored_kind()
    kind = stated_kind[0] if stated_kind and stated_kind[0] in KINDS else None
    if stated_kind and kind is None:
        log.info("%s records kind %r (%s), which is not one of %s, so it is ignored",
                 spec.get("path"), stated_kind[0], stated_kind[1], ", ".join(KINDS))
    return {
        # The dataset IS the array: no level to descend into.
        "data_spec": dict(spec),
        "voxel_size": voxel_size,
        "offset": tuple(float(o) * v for o, v in zip(voxel_offset, voxel_size)),
        "voxel_offset": voxel_offset,
        "units": units[0] if units else None,
        "spatial_axes": axes,
        "has_channels": len(shape) > len(axes),
        # Never inferred from the dtype; unrecorded means `convert --kind`.
        "kind": kind,
    }


def _read_dvid(spec: Mapping[str, Any]) -> dict | None:
    """Coordinate metadata for a DVID labelmap, from one ``/info`` request.

    ``kind`` is always ``segmentation``, since a labelmap is one by definition. ``offset``
    is 0: coordinates are presented origin-anchored, as DVID's own, and ``MinPoint`` is
    where data starts, not a translation.

    A branch ref is resolved to a concrete uuid here and that goes into ``data_spec``, so
    every worker reads the same node even if the branch moves mid-run.
    """
    import logging

    from .backends.dvid import geometry
    from .dvid import instance_info, resolve_node

    node = resolve_node(spec, prefer_locked=bool(spec.get("prefer_locked")))
    # `prefer_locked` has been applied; carrying it would invite a second resolution.
    resolved = {k: v for k, v in spec.items() if k != "prefer_locked"}
    resolved["uuid"] = node["uuid"]
    # What was asked for, for provenance: "newest locked on main" is not "this node".
    resolved["requested_ref"] = node["ref"]
    resolved["ancestors_walked"] = node["walked"]

    # Reported, not warned: an open node is warned about by the op that exports it,
    # since this also runs for read-only inspection.
    log = logging.getLogger(__name__)
    if node["uuid"] != str(spec.get("uuid")):
        log.info("dvid %s resolved to node %s (%s)%s", spec.get("uuid"), node["uuid"],
                 "locked" if node["locked"] else "OPEN",
                 f", {node['walked']} ancestor(s) back" if node["walked"] else "")

    geom = geometry(instance_info(resolved), resolved)
    return {
        # The caller's backend, carried through: it selects the reader.
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

    The dataset name is resolved to its id, and the id, server and datastore go into
    ``data_spec``, since names are not unique across organizations. The token does not;
    each process reads the credentials itself. A requested ``mapping`` is checked against
    the layer's list before any work.
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

    # Level 0 is the finest scale actually written, not the finest declared. An explicit
    # `scale_index` (`--src-level`) always wins, even naming an empty scale, which
    # `neu-vol write` does when filling a new level.
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
        # The caller's backend, carried through: `data_spec` selects the reader, and a
        # .gz volume read as plain precomputed reads as zeros.
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
# The whole-volume picture: format, coordinates and the levels present. These open
# each level, which is the expensive part (see the module docstring).
# --------------------------------------------------------------------------- #


def level_spec(volume: str, fmt: str, level: int,
               *, dataset: str | None = None) -> dict[str, Any]:
    """The backend spec for one level of a multiscale volume.

    zarr v3 puts each level in its own subdirectory; precomputed, DVID and webknossos
    select one with ``scale_index``. A single-container format has only level 0, and any
    other raises, since its backend would ignore ``scale_index`` and open level 0 anyway.
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

    Returns a :class:`~neu_vol.report.Description`: a dict that also renders as the table
    ``neu-vol info`` prints, with ``.frame()`` giving the per-level rows as a DataFrame.
    ``level`` describes that source scale instead of level 0.

    An HDF5 file or image stack has one level and may record no metadata, so ``meta`` can
    be ``None``. An HDF5 file may hold several arrays:

    * ``datasets`` is ``{name: {"shape", "dtype", "chunks", <recorded frame>}}`` for every
      HDF5 file, and ``None`` for other formats;
    * ``dataset`` is the array the rest of the dict describes, or ``None``;
    * with several arrays and none named, ``shape`` / ``dtype`` / ``meta`` /
      ``level_voxel_sizes`` are ``None`` and ``levels`` is empty; callers needing one
      array use :func:`require_one_array`.

    Raises ``FileNotFoundError`` if nothing at ``volume`` looks like a volume.
    """
    from .backends.base import open_backend
    from .logs import quiet_reads
    from .report import Description

    # A notebook entry point, so it filters tensorstore's benign credential-chain lines
    # itself. `quiet_reads` nests, so the CLI wrapping `main()` as well is fine.
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
            # Several arrays and none named: the listing is the answer.
            return Description(
                location=volume, format=fmt, meta=None, shape=None, dtype=None,
                has_channels=False, level_voxel_sizes=None, levels={},
                spec={"backend": fmt, "path": volume},
                dataset=None, datasets=entries, other_markers=[])

    spec = location_spec(volume, fmt, dataset=dataset)
    if level is not None:
        # `shape`, `dtype` and `meta` describe that scale; `levels` is still the whole
        # declared pyramid.
        spec["scale_index"] = int(level)
    meta = read_source_metadata(spec)
    level0 = open_backend(meta["data_spec"] if meta else spec)
    shape = tuple(int(s) for s in level0.shape)
    # A single container's one level is `meta`; re-reading would repeat its warnings.
    if fmt in SINGLE_LEVEL_FORMATS:
        per_level = [tuple(meta["voxel_size"])] if meta else None
    else:
        per_level = read_level_voxel_sizes(spec)
    return Description(
        location=volume,
        format=fmt, meta=meta, shape=shape,
        dtype=str(getattr(level0, "dtype", "?")),
        has_channels=bool(meta["has_channels"]) if meta else False,
        level_voxel_sizes=per_level,
        levels=existing_levels(volume, fmt, dataset=dataset),
        spec=spec,                      # resolved, so a caller need not detect again
        dataset=spec.get("dataset"),
        # Which source scale `shape`/`dtype`/`meta` describe: the finest that stores data.
        source_level=int((meta or {}).get("data_spec", {}).get("scale_index") or 0),
        datasets=entries,
        # A second volume shadowed underneath this one, if any.
        other_markers=other_format_markers(volume, fmt))


def require_one_array(described: Mapping[str, Any], location: str, op: str) -> dict:
    """``described`` if it resolved a single array; otherwise an error naming the datasets.

    For callers that go on to read the array, which would otherwise fail on a ``None``
    ``shape`` from a multi-array HDF5 container.
    """
    if described.get("shape") is not None:
        return described                # the Description itself, so it still renders
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

    Probes upward until a level fails to open, so it works on a conversion still in
    progress, whose group metadata is written last. A single-container format has only
    level 0. Chunking is read per level, since levels may be chunked differently;
    ``chunks`` is the write chunk (the shard, when sharded) and ``read_chunks`` the unit
    actually fetched.
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
