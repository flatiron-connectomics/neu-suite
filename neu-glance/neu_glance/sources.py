"""Layers built from something on a store: a volume, or a precomputed annotation source.

The only module in this package that reads a store, so :mod:`neu_glance.state` and
:mod:`neu_glance.layers` stay pure and work for locally served sources that have no store.

Two choices here fail silently if made wrong:

- **The source scheme** (``precomputed://`` or ``zarr://``) comes from
  :func:`~neu_vol.source_metadata.detect_backend`, not from the path.
- **Volume or annotation source.** Both are ``precomputed://`` with an ``info`` at the
  root, and an annotation layer pointed at a volume loads and draws nothing. Only the
  ``@type`` tells them apart, so :func:`read_annotation_info` checks it.
"""

from __future__ import annotations

import functools
from typing import Any, Mapping, Sequence

from .shaders import SPLIT_CONTROLS, ShaderProblem, pick_shader

#: How neuroglancer addresses each backend: the classic ``scheme://`` form, which older
#: viewers read too, rather than the newer ``kvstore|adapter:`` syntax.
def _scheme_map() -> dict[str, str]:
    from neu_vol.source_metadata import PRECOMPUTED_GZ

    return {"neuroglancer_precomputed": "precomputed",
            PRECOMPUTED_GZ: "precomputed",
            "zarr3": "zarr"}


#: The `@type` a precomputed annotation source declares.
ANNOTATION_TYPE = "neuroglancer_annotations_v1"

#: ``(side, relationship)`` for the two halves of a synapse source. Filtering on ``body_pre``
#: gives the selected body's outputs; on ``body_post``, its inputs.
SPLIT_SIDES = (("pre", "body_pre"), ("post", "body_post"))


class SourceProblem(RuntimeError):
    """A named volume or annotation source could not be used as a layer source."""


def _quiet(fn):
    """Filter the benign credential log lines every S3/GCS open emits, so callers need not.
    Set ``neu_vol.logs.reads_quiet = False`` to see them."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        from neu_vol.logs import quiet_reads

        with quiet_reads():
            return fn(*args, **kwargs)
    return wrapper


@_quiet
def volume_extent(volume: str, fmt: str) -> tuple[tuple, tuple] | None:
    """``(extent_zyx, offset_zyx)`` in level-0 voxels, or None if not determinable.

    Precomputed answers from the root ``info``; zarr opens the level-0 array once.
    """
    from neu_vol.location import read_json
    from neu_vol.source_metadata import read_source_metadata

    if str(fmt).startswith("neuroglancer_precomputed"):
        info = read_json(volume.rstrip("/") + "/info") or {}
        scales = info.get("scales")
        if not scales:
            return None
        finest = min(scales, key=lambda s: tuple(s["resolution"]))
        size = [int(v) for v in finest["size"]][::-1]                    # xyz -> zyx
        off = [int(v) for v in finest.get("voxel_offset", [0, 0, 0])][::-1]
        return tuple(size), tuple(off)

    meta = read_source_metadata({"backend": fmt, "path": volume})
    if not meta:
        return None
    try:
        from neu_vol.backends.base import open_backend

        shape = tuple(int(s) for s in open_backend(meta["data_spec"]).shape)
    except Exception:
        return None
    spatial = shape[-3:] if len(shape) > 3 else shape       # drop a channel axis
    return spatial, (0, 0, 0)


@_quiet
def volume_layer(volume: str, *, kind: str | None = None, name: str | None = None,
                 segments: Sequence[int] | None = None,
                 opacity: float | None = None) -> tuple[dict, dict]:
    """One layer for ``volume``, plus what its metadata says about the frame.

    ``kind`` (``"image"`` or ``"segmentation"``) overrides what the volume records; without
    it the recorded kind decides, defaulting to image.
    """
    from neu_vol.source_metadata import detect_backend, read_source_metadata

    volume = volume.rstrip("/")
    fmt = detect_backend(volume)
    if fmt is None:
        raise SourceProblem(f"no volume found at {volume}")
    scheme = _scheme_map().get(fmt)
    if scheme is None:
        raise SourceProblem(f"{volume} is {fmt}, which neuroglancer cannot read")

    # read_source_metadata, not describe: everything needed is in `info`, and describe
    # opens every level.
    meta = read_source_metadata({"backend": fmt, "path": volume}) or {}
    resolved = kind or meta.get("kind") or "image"
    layer: dict[str, Any] = {
        "type": "segmentation" if resolved == "segmentation" else "image",
        "name": name or volume.rsplit("/", 1)[-1],
        "source": f"{scheme}://{volume}",
    }
    if segments:
        # Strings: segment ids are uint64, and a JSON number above 2**53 is rounded.
        layer["segments"] = [str(int(s)) for s in segments]
    if opacity is not None:
        layer["opacity"] = float(opacity)
    return layer, {"voxel_size": meta.get("voxel_size"), "units": meta.get("units"),
                   "format": fmt}


@_quiet
def read_annotation_info(source: str) -> dict:
    """The ``info`` of a precomputed annotation source, refusing anything else."""
    from neu_vol.location import read_json

    source = source.rstrip("/")
    info = read_json(source, "info")
    if info is None:
        raise SourceProblem(f"no info at {source}")
    if info.get("@type") != ANNOTATION_TYPE:
        raise SourceProblem(
            f"{source} is {info.get('@type') or 'not an annotation source'}, not "
            f"{ANNOTATION_TYPE}. Volumes go to --image or --seg; --annotations wants a "
            f"precomputed ANNOTATION source (what `neu-mark annotation-source` writes).")
    return info


@_quiet
def annotation_layer(source: str, *, name: str | None = None,
                     shader: str | None = None,
                     linked_segmentation: str | None = None,
                     filter_by_segmentation: bool = True,
                     filter_relationships: Sequence[str] | None = None,
                     controls: Mapping[str, Any] | None = None,
                     info: Mapping[str, Any] | None = None) -> tuple[dict, dict]:
    """One annotation layer for a precomputed annotation source, plus its frame.

    ``linked_segmentation`` names a segmentation layer in the same state. Without it the
    source's ``relationships`` index does nothing: neuroglancer only uses a relationship once
    it is bound to a layer whose selected segments it can read, and otherwise draws every
    annotation.

    Every relationship is bound; ``filter_relationships`` chooses which ones restrict the
    view, e.g. only ``body_pre`` for the selected body's outputs.
    """
    source = source.rstrip("/")
    info = dict(info) if info is not None else read_annotation_info(source)
    layer: dict[str, Any] = {
        "type": "annotation",
        "name": name or source.rsplit("/", 1)[-1],
        "source": f"precomputed://{source}",
    }
    try:
        shader_source, why = pick_shader(info, shader)
    except ShaderProblem as e:
        raise SourceProblem(str(e)) from None
    if shader_source:
        layer["shader"] = shader_source
    if controls:
        layer["shaderControls"] = dict(controls)

    relationships = [r["id"] for r in info.get("relationships", []) or []]
    if linked_segmentation and relationships:
        layer["linkedSegmentationLayer"] = {r: linked_segmentation for r in relationships}
        if filter_by_segmentation:
            picked = [r for r in (filter_relationships or relationships)
                      if r in relationships]
            if filter_relationships and not picked:
                raise SourceProblem(
                    f"none of {', '.join(filter_relationships)} is a relationship of "
                    f"{source}; it declares " + (", ".join(relationships) or "none"))
            layer["filterBySegmentation"] = picked

    return layer, {"info": info, "shader": why, "relationships": relationships}


@_quiet
def annotation_layer_pair(source: str, *, name: str | None = None,
                          shader: str | None = None,
                          linked_segmentation: str | None = None,
                          filter_by_segmentation: bool = True) -> tuple[list[dict], dict]:
    """Two layers on one source: the selected body's outputs, then its inputs.

    One layer filtered on both relationships mixes the two directions, and their markers
    overlap. Split, each half has its own visibility, colour and marker size. The ``info`` is
    read once for both.
    """
    base = (name or source.rstrip("/").rsplit("/", 1)[-1])
    info = read_annotation_info(source)
    layers, detail = [], None
    for side, relationship in SPLIT_SIDES:
        layer, detail = annotation_layer(
            source, name=f"{base}-{side}", shader=shader,
            linked_segmentation=linked_segmentation,
            filter_by_segmentation=filter_by_segmentation,
            filter_relationships=[relationship], controls=SPLIT_CONTROLS[side],
            info=info)
        layers.append(layer)
    return layers, detail


@_quiet
def annotation_source_extent(info: Mapping[str, Any]) -> tuple[tuple, tuple] | None:
    """``(extent_zyx, offset_zyx)`` from an annotation source's declared bounds.

    Lets an annotations-only link open framed on its data, as :func:`volume_extent` does
    for a volume.
    """
    lower, upper = info.get("lower_bound"), info.get("upper_bound")
    if not lower or not upper or len(lower) != 3 or len(upper) != 3:
        return None
    lo = [float(v) for v in reversed(lower)]            # stored xyz, wanted zyx
    hi = [float(v) for v in reversed(upper)]
    return tuple(max(1.0, h - l) for l, h in zip(lo, hi)), tuple(lo)


@_quiet
def annotation_source_voxel_size(info: Mapping[str, Any]) -> tuple | None:
    """Level-0 voxel size in nm, zyx, from the source's ``dimensions``.

    ``dimensions`` is ``{axis: [scale, unit]}`` in SI units (metres). None if absent or not
    in metres.
    """
    dims = info.get("dimensions")
    if not isinstance(dims, Mapping):
        return None
    try:
        scales = [dims[axis] for axis in ("x", "y", "z")]
    except KeyError:
        return None
    out = []
    for scale, unit in scales:
        if unit != "m":
            return None
        out.append(float(scale) * 1e9)
    return tuple(reversed(out))                        # xyz read, zyx returned


def _readback():
    """``neu_morpho.readback`` (the ``framing`` extra), imported on first use, or a clear
    error naming the extra. The reader needs only numpy, none of neu-morpho's conda-only
    meshing dependencies."""
    try:
        from neu_morpho import readback
    except ImportError as e:
        raise SourceProblem(
            f"reading an object's bounds needs neu-morpho's mesh reader, which is not "
            f"installed: {e}. Install it with `pip install 'neu-glance[framing]'`, or pass "
            f"boxes to Timeline.frame_on() yourself — it takes any (lo_zyx, hi_zyx).") from None
    return readback


@_quiet
def segment_boxes(volume: str, segment_ids: Sequence[Any], *, lod: int | None = None,
                  mesh_dir: str | None = None, threads: int = 8,
                  skip_missing: bool = True) -> dict[str, tuple[tuple, tuple]]:
    """``{segment_id: (lo_zyx, hi_zyx)}`` in **nanometres**, from each segment's mesh.

    For :meth:`neu_glance.animate.Timeline.frame_on`. Bounds come from the mesh rather than
    a coarse level of the segmentation, where a thin arbor can shrink or vanish and the box
    would be wrong. ``lod=None`` reads the coarsest level of detail, which is small and
    bounds the object just as well. Sharded sources and a source's own ``transform`` are
    handled, so the result is in nm.

    ``skip_missing`` drops a segment with no mesh; a structurally broken source still raises.
    """
    readback = _readback()
    ids = [str(s).lstrip("!") for s in segment_ids]
    boxes: dict[str, tuple[tuple, tuple]] = {}
    # Resolved from `info`: passing None to read_body_mesh would look for a directory named
    # "None" and report every body as having no mesh.
    directory = mesh_dir or _subresource(volume, "mesh")

    from concurrent.futures import ThreadPoolExecutor

    def one(sid: str):
        raw = readback.read_body_mesh(volume, int(sid), mesh_dir=directory, lod=lod)
        if raw is None:
            return sid, None
        vertices = raw[0]
        lo = tuple(float(v) for v in vertices.min(axis=0)[::-1])   # stored xyz, wanted zyx
        hi = tuple(float(v) for v in vertices.max(axis=0)[::-1])
        return sid, (lo, hi)

    with ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
        for sid, box in pool.map(one, ids):
            if box is not None:
                boxes[sid] = box
            elif not skip_missing:
                raise SourceProblem(f"{volume} has no mesh for segment {sid}")
    return boxes


def _subresource(volume: str, key: str, info: Mapping[str, Any] | None = None) -> str:
    """The subdirectory a volume's ``info`` names for ``key``. Raises if there is none,
    rather than guessing a name that would find nothing."""
    from neu_vol import read_json

    info = info if info is not None else read_json(volume, "info")
    name = (info or {}).get(key)
    if not name:
        present = ", ".join(k for k in ("mesh", "skeletons", "segment_properties")
                            if (info or {}).get(k)) or "none"
        raise SourceProblem(
            f"{volume} declares no {key!r} subresource in its info (it has: {present}), so "
            f"there is nothing to read")
    return str(name)


@_quiet
def segment_labels(volume: str, *, info: Mapping[str, Any] | None = None) -> dict[str, str]:
    """``{segment_id: label}`` from a volume's ``segment_properties`` subresource.

    Lets code name segments (``"LAL(L)"``) rather than ids. The format pairs ids and values
    by position, so arrays of different lengths raise rather than mislabel segments.
    """
    from neu_vol import read_json

    directory = _subresource(volume, "segment_properties", info)
    document = read_json(f"{volume.rstrip('/')}/{directory}", "info")
    declared = (document or {}).get("@type")
    if declared != "neuroglancer_segment_properties":
        raise SourceProblem(
            f"{volume}/{directory} declares @type {declared!r}, not "
            f"'neuroglancer_segment_properties' — that is not a segment properties source")
    inline = (document or {}).get("inline")
    if not inline:
        raise SourceProblem(
            f"{volume}/{directory} carries no `inline` block; only inline segment properties "
            f"are readable here")
    ids = [str(i) for i in inline.get("ids") or []]
    labels = next((p.get("values") for p in inline.get("properties") or []
                   if p.get("type") == "label"), None)
    if labels is None:
        kinds = ", ".join(sorted({str(p.get("type")) for p in inline.get("properties") or []}))
        raise SourceProblem(
            f"{volume}/{directory} has no property of type 'label' (it has: {kinds or 'none'})")
    if len(labels) != len(ids):
        raise SourceProblem(
            f"{volume}/{directory} lists {len(ids)} ids and {len(labels)} labels; the format "
            f"pairs them by position, so a mismatch would mislabel segments")
    return dict(zip(ids, [str(v) for v in labels]))


@_quiet
def segment_ids(volume: str, names: Sequence[str], *,
                info: Mapping[str, Any] | None = None) -> list[str]:
    """Segment ids for ``names``, in order. Raises on a name the source does not carry.

    A mistyped name raises (with close matches suggested) rather than being dropped. Labels
    are not unique in this format, so a name matching several segments also raises.
    """
    labels = segment_labels(volume, info=info)
    by_name: dict[str, list[str]] = {}
    for sid, label in labels.items():
        by_name.setdefault(label, []).append(sid)

    out, unknown, ambiguous = [], [], []
    for name in names:
        found = by_name.get(str(name))
        if not found:
            unknown.append(str(name))
        elif len(found) > 1:
            ambiguous.append(f"{name} ({', '.join(found)})")
        else:
            out.append(found[0])
    if unknown:
        import difflib
        hints = {n: difflib.get_close_matches(n, by_name, n=3) for n in unknown}
        detail = "; ".join(f"{n!r}" + (f" — did you mean {', '.join(h)}?" if h else "")
                           for n, h in hints.items())
        raise SourceProblem(
            f"{volume} has no segment labelled {detail}. It has {len(by_name)} labels: "
            f"{', '.join(sorted(by_name))}")
    if ambiguous:
        raise SourceProblem(
            f"{volume} labels more than one segment {'; '.join(ambiguous)} — labels are not "
            f"unique in this format, so name the id instead")
    return out


@_quiet
def read_segment_properties(volume: str, *, info: Mapping[str, Any] | None = None) -> dict:
    """Everything a volume's ``segment_properties`` carries, in ONE read.

    Returns ``{"ids", "labels", "tags", "numbers", "vocabulary"}``: ids as strings, labels
    parallel to them, tags resolved from indices to strings, and each numeric property under
    its own name. Pass the result to :func:`select_segments` to query it without re-reading.
    Properties are paired with ids by position, so a length mismatch raises.
    """
    from neu_vol import read_json

    directory = _subresource(volume, "segment_properties", info)
    document = read_json(f"{volume.rstrip('/')}/{directory}", "info")
    declared = (document or {}).get("@type")
    if declared != "neuroglancer_segment_properties":
        raise SourceProblem(
            f"{volume}/{directory} declares @type {declared!r}, not "
            f"'neuroglancer_segment_properties' — that is not a segment properties source")
    inline = (document or {}).get("inline")
    if not inline:
        raise SourceProblem(
            f"{volume}/{directory} carries no `inline` block; only inline segment properties "
            f"are readable here")

    ids = [str(i) for i in inline.get("ids") or []]
    out: dict[str, Any] = {"ids": ids, "labels": [], "tags": [], "numbers": {},
                           "vocabulary": []}
    for prop in inline.get("properties") or []:
        values = prop.get("values") or []
        if len(values) != len(ids):
            raise SourceProblem(
                f"{volume}/{directory} property {prop.get('id')!r} has {len(values)} values "
                f"for {len(ids)} ids; the format pairs them by position, so a mismatch would "
                f"attach them to the wrong segments")
        kind = prop.get("type")
        if kind == "label":
            out["labels"] = [str(v) for v in values]
        elif kind == "tags":
            vocabulary = [str(t) for t in prop.get("tags") or []]
            out["vocabulary"] = vocabulary
            out["tags"] = [tuple(vocabulary[i] for i in row) for row in values]
        elif kind == "number":
            out["numbers"][str(prop.get("id"))] = list(values)
    if not out["labels"]:
        out["labels"] = list(ids)
    if not out["tags"]:
        out["tags"] = [()] * len(ids)
    return out


@_quiet
def segment_tags(volume: str, *, info: Mapping[str, Any] | None = None) -> dict[str, tuple]:
    """``{segment_id: (tag, ...)}``, tags resolved from their vocabulary indices."""
    read = read_segment_properties(volume, info=info)
    return dict(zip(read["ids"], read["tags"]))


@_quiet
def select_segments(volume: str, *, tags: Sequence[str] | None = None,
                    any_tags: Sequence[str] | None = None,
                    without: Sequence[str] | None = None,
                    label: str | None = None, limit: int | None = None,
                    order_by: str | None = "voxels", properties: Mapping | None = None
                    ) -> list[str]:
    """Segment ids matching a tag and/or label query.

    - ``tags``: all must be present;
    - ``any_tags``: at least one must be;
    - ``without``: none may be (e.g. ``fragment``, ``truncated``);
    - ``label``: a regular expression, searched (not anchored) in the label.

    Conditions combine with "and", so ``tags=["side:L"], label=r"^KC"`` is left-side KCs.
    No match returns an empty list; an unknown tag raises. ``limit`` keeps the largest by
    ``order_by`` (``voxels``); ``order_by=None`` keeps the source's order. Sets can be very
    large (thousands of bodies), so check the count before rendering.
    """
    import re

    read = properties if properties is not None else read_segment_properties(volume)
    pattern = re.compile(label) if label else None
    tags, any_tags, without = set(tags or ()), set(any_tags or ()), set(without or ())

    known = set(read.get("vocabulary") or ())
    unknown = (tags | any_tags | without) - known
    if unknown and known:
        # A misspelled tag would otherwise silently empty the set.
        raise SourceProblem(
            f"{volume} has no tag(s) {', '.join(sorted(unknown))}. It has {len(known)}, "
            f"e.g. {', '.join(sorted(known)[:8])}")

    chosen = []
    for sid, name, has in zip(read["ids"], read["labels"], read["tags"]):
        have = set(has)
        if tags and not tags <= have:
            continue
        if any_tags and not (any_tags & have):
            continue
        if without and (without & have):
            continue
        if pattern is not None and not pattern.search(name):
            continue
        chosen.append(sid)

    if limit is not None and len(chosen) > limit:
        if order_by and order_by in (read.get("numbers") or {}):
            size = dict(zip(read["ids"], read["numbers"][order_by]))
            chosen.sort(key=lambda s: size.get(s, 0), reverse=True)
        chosen = chosen[:limit]
    return chosen


@_quiet
def segments_with_meshes(volume: str, segment_ids: Sequence[Any], *,
                         mesh_dir: str | None = None, threads: int = 32) -> list[str]:
    """The subset of ``segment_ids`` whose meshes are actually in the store, in order.

    **One segment with no mesh hangs a render indefinitely.** Neuroglancer waits for its
    mesh chunk forever, so the viewer never reports ready and the capture stalls at
    ``N-1 / N chunks``. ``segment_properties`` lists every segment, including unmeshed ones,
    so filter a set through this before rendering it.

    One existence check per body (cheap for unsharded meshes). A sharded source has no
    per-body object to check, so every id is returned unchecked.
    """
    from concurrent.futures import ThreadPoolExecutor

    from neu_vol import read_json
    from neu_vol.location import exists

    volume = volume.rstrip("/")
    directory = mesh_dir or _subresource(volume, "mesh")
    info = read_json(f"{volume}/{directory}", "info") or {}
    ids = [str(s).lstrip("!") for s in segment_ids]

    if info.get("sharding") is not None:
        return ids
    legacy = info.get("@type") == "neuroglancer_legacy_mesh"
    key = (lambda sid: f"{sid}:0") if legacy else (lambda sid: f"{sid}.index")

    with ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
        present = list(pool.map(lambda sid: exists(volume, directory, key(sid)), ids))
    return [sid for sid, found in zip(ids, present) if found]
