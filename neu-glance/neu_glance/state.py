"""Assemble a viewer state, encode it into a URL, and read one back.

Neuroglancer keeps its whole state in the URL fragment, so a link is the state. A wrong
``dimensions`` block still loads, with every layer in the wrong place, so it is derived
from the volume's recorded voxel size rather than assumed. The fragment (after ``#!``) is
never sent to a server, but a large inline annotation layer makes a long link (see
:data:`LONG_URL`).

Pure: :func:`build_state` takes plain layer dicts and never reads a store.
"""

from __future__ import annotations

import json
import urllib.parse
from typing import Any, Mapping, Sequence

from .layers import output_dimensions
from .shaders import as_hex_color

DEFAULT_VIEWER = "https://neuroglancer-demo.appspot.com/"

# Length past which a URL is awkward to paste (mail clients wrap, chat tools truncate).
# Worth a warning, not an error.
LONG_URL = 8000

# Neuroglancer's layout names. Checked, because the viewer silently ignores a typo.
LAYOUTS = ("4panel", "xy", "yz", "xz", "xy-3d", "yz-3d", "xz-3d", "3d")

# A nominal viewport edge in pixels, for turning "fit the volume" into a zoom. The real
# panel size is unknown here; erring large errs zoomed out, which is harmless.
NOMINAL_VIEWPORT_PX = 1000

# Leave a little space around the volume rather than cropping it to the panel edge.
FIT_MARGIN = 1.15


class StateProblem(RuntimeError):
    """A state or layer could not be read, or would not be well formed."""


def annotation_extent(layers: Sequence[dict]) -> tuple[tuple, tuple] | None:
    """``(extent_zyx, offset_zyx)`` covering the inline annotations in ``layers``.

    For framing a state with no volume, e.g. one built from a bounding-box layer file.
    """
    lo: list[float | None] = [None, None, None]
    hi: list[float | None] = [None, None, None]
    for layer in layers:
        for ann in layer.get("annotations", []) or []:
            pts = [ann[k] for k in ("pointA", "pointB", "point", "center") if k in ann]
            for p in pts:
                for a in range(3):
                    v = float(p[2 - a])                     # stored xyz, wanted zyx
                    lo[a] = v if lo[a] is None else min(lo[a], v)
                    hi[a] = v if hi[a] is None else max(hi[a], v)
    if any(v is None for v in lo):
        return None
    extent = tuple(max(1.0, hi[a] - lo[a]) for a in range(3))
    return extent, tuple(lo)


def default_view(extent_zyx: Sequence[float],
                 offset_zyx: Sequence[float] = (0, 0, 0)) -> tuple[list, float, float]:
    """``(centre_zyx, cross_section_scale, projection_scale)`` framing a whole volume.

    Without these, neuroglancer opens at the origin corner at one voxel per pixel, a view
    of a large volume's empty edge. Both scales are in voxels: per pixel for the cross
    sections, across the viewport for the projection.
    """
    centre = [float(o) + float(e) / 2 for o, e in zip(offset_zyx, extent_zyx)]
    span = max(float(e) for e in extent_zyx) * FIT_MARGIN
    return centre, span / NOMINAL_VIEWPORT_PX, span


def build_state(layers: Sequence[dict], *, voxel_size_zyx=None, units: str | None = None,
                position_zyx: Sequence[float] | None = None,
                layout: str = "4panel",
                cross_section_scale: float | None = None,
                projection_scale: float | None = None,
                selected: str | None = None,
                show_slices: bool | None = None,
                projection_orientation: Sequence[float] | None = None,
                frame: tuple | None = None) -> tuple[dict, str | None]:
    """A viewer state. Coordinates are given zyx and written xyz.

    ``projection_orientation`` is the 3D view's quaternion as neuroglancer writes it
    (``projectionOrientation``, x, y, z, w).

    ``dimensions`` comes from ``voxel_size_zyx`` and ``units``, and must agree with the
    layers or nothing lines up. ``frame``, an ``(extent_zyx, offset_zyx)`` pair, fills in
    whichever of the position and the two zooms are not given (:func:`default_view`).

    ``show_slices=False`` hides the cross-section planes inside the 3D panel, useful for
    meshes and skeletons; the 2D panels need ``layout="3d"`` instead. ``None`` leaves the
    viewer's default (shown).
    """
    dims, warning = output_dimensions(voxel_size_zyx, units)
    state: dict[str, Any] = {"dimensions": dims, "layers": list(layers),
                             "layout": layout}

    fitted = default_view(*frame) if frame else (None, None, None)
    position = position_zyx if position_zyx is not None else fitted[0]
    cross = cross_section_scale if cross_section_scale is not None else fitted[1]
    projection = projection_scale if projection_scale is not None else fitted[2]

    if position is not None:
        state["position"] = [float(v) for v in tuple(position)[::-1]]
    if cross is not None:
        state["crossSectionScale"] = float(cross)
    if projection is not None:
        state["projectionScale"] = float(projection)
    if selected:
        state["selectedLayer"] = {"visible": True, "layer": selected}
    if show_slices is not None:
        state["showSlices"] = bool(show_slices)
    if projection_orientation is not None:
        state["projectionOrientation"] = [float(v) for v in projection_orientation]
    return state, warning


def state_url(state: Mapping[str, Any], viewer: str = DEFAULT_VIEWER) -> str:
    """``<viewer>#!<url-encoded state>``.

    Percent-encoded, so the quotes and braces survive chat clients, wikis and shells.
    """
    encoded = urllib.parse.quote(json.dumps(dict(state), separators=(",", ":")), safe="")
    return f"{viewer.rstrip('/')}/#!{encoded}"


def parse_url(url: str) -> dict:
    """The state back out of a neuroglancer URL, for inspecting or editing one."""
    if "#!" not in url:
        raise ValueError(
            "not a neuroglancer state URL: no '#!' fragment. If you passed it in double "
            "quotes, bash history expansion may have eaten the '!' (a link starting "
            "'#7B' has lost '!%'); put the URL in single quotes instead")
    fragment = url.split("#!", 1)[1]
    return json.loads(urllib.parse.unquote(fragment))


def load_layer(path: str, read_bytes=None) -> list[dict]:
    """A layer (or a whole state's worth of layers) from a JSON file.

    Accepts a bare layer object, a list of layers, or a full state (whose ``layers`` are
    taken), as ``neu-glance bboxes`` and ``neu-glance annotate`` write either.
    """
    obj = _read_json(path, read_bytes)
    if isinstance(obj, dict) and "layers" in obj:
        return list(obj["layers"])
    if isinstance(obj, list):
        return list(obj)
    if not isinstance(obj, dict) or "type" not in obj:
        raise StateProblem(
            f"{path} is not a neuroglancer layer or state: expected an object with a "
            f"'type' (a layer) or a 'layers' array (a state)")
    return [obj]


def load_state(source: str, read_bytes=None) -> dict:
    """An existing state, from a neuroglancer URL **or** a JSON file.

    An ``http(s)://`` source is always treated as a URL, so a link that lost its ``#!``
    (bash expands ``!`` inside double quotes) fails as a broken URL rather than as "no such
    file".
    """
    if "#!" in source or source.startswith(("http://", "https://")):
        return parse_url(source)
    obj = _read_json(source, read_bytes)
    if not isinstance(obj, dict) or "layers" not in obj:
        raise StateProblem(
            f"{source} is not a neuroglancer state: expected an object with a 'layers' "
            f"array. A bare layer file goes to --layer, not --into.")
    return obj


def split_segment_layer(state: Mapping[str, Any], layer: str, *,
                        segments: Sequence[Any] | None = None,
                        names: Mapping[str, str] | None = None,
                        name_template: str = "{layer} {segment}",
                        object_alpha: float | None = None,
                        ) -> tuple[dict, dict[str, str], list[str]]:
    """One layer per segment, so each can be shown, coloured or FADED on its own.

    Returns ``(state, {segment_id: layer_name}, notes)``.

    Neuroglancer has no per-segment alpha (``objectAlpha`` belongs to the layer), so fading
    one neuron while its neighbour stays needs one layer per neuron. The copies share the
    ``source``, and the viewer fetches each mesh once however many layers name it.

    Each copy is the original layer verbatim, then overridden, so its other settings carry
    over. Colours are unchanged: an explicit ``segmentColors`` entry is kept (for that
    segment only), and an unlisted segment's hash colour depends only on ``colorSeed`` and
    its id, both preserved.

    - A ``!``-prefixed (starred, hidden) id is not split; it stays on the original.
    - The original layer is kept, emptied and hidden, because ``selectedLayer``,
      ``linkedSegmentationLayer`` or a tool palette may name it, and a dangling name
      resolves to nothing silently. A synapse layer filtering on it now filters on an
      empty set; re-point it at a copy if needed.
    - The copies replace the original in place, since neuroglancer draws in layer order.

    Segment ids stay strings: a uint64 through a JSON number is rounded.
    """
    out = json.loads(json.dumps(dict(state)))            # deep copy; never mutate the caller's
    existing = list(out.get("layers") or [])
    index = next((i for i, lyr in enumerate(existing)
                  if isinstance(lyr, dict) and lyr.get("name") == layer), None)
    if index is None:
        known = ", ".join(repr(lyr.get("name")) for lyr in existing if isinstance(lyr, dict))
        raise StateProblem(f"no layer named {layer!r} in this state. It has: {known}")

    source = existing[index]
    if source.get("type") != "segmentation":
        raise StateProblem(
            f"layer {layer!r} is a {source.get('type')!r} layer; only a segmentation layer "
            f"has segments to split")

    raw = [str(s) for s in (source.get("segments") or [])]
    starred = [s for s in raw if s.startswith("!")]
    visible = [s for s in raw if not s.startswith("!")]
    if segments is not None:
        wanted = [str(s) for s in segments]
        missing = [s for s in wanted if s not in visible]
        if missing:
            raise StateProblem(
                f"layer {layer!r} has no visible segment(s) {', '.join(missing)}. Its visible "
                f"segments are: {', '.join(visible) or '(none)'}"
                + (f". {', '.join(starred)} are starred but hidden ('!'), so they cannot be "
                   f"split" if starred else ""))
        visible = wanted

    notes: list[str] = []
    if not visible:
        notes.append(f"layer {layer!r} has no visible segments, so nothing was split")

    colors = {str(k): v for k, v in (source.get("segmentColors") or {}).items()}
    taken = {lyr.get("name") for lyr in existing if isinstance(lyr, dict)}
    created: dict[str, str] = {}
    copies: list[dict] = []

    for i, sid in enumerate(visible):
        copy = json.loads(json.dumps(source))
        copy["segments"] = [sid]
        if sid in colors:
            copy["segmentColors"] = {sid: colors[sid]}
        else:
            copy.pop("segmentColors", None)
        if object_alpha is not None:
            copy["objectAlpha"] = float(object_alpha)
        copy.pop("visible", None)

        name = (names or {}).get(sid) or name_template.format(
            layer=layer, segment=sid, index=i)
        if name in taken:
            suffix = 2
            while f"{name}-{suffix}" in taken:
                suffix += 1
            notes.append(f"renamed split layer {name!r} to {name}-{suffix}: the state already "
                         f"has a layer by that name")
            name = f"{name}-{suffix}"
        taken.add(name)
        copy["name"] = name
        created[sid] = name
        copies.append(copy)

    emptied = json.loads(json.dumps(source))
    emptied["segments"] = starred
    emptied.pop("segmentColors", None)
    emptied["visible"] = False
    out["layers"] = existing[:index] + copies + [emptied] + existing[index + 1:]
    return out, created, notes


def subset_layers(state: Mapping[str, Any], layer: str,
                  subsets: Mapping[str, Sequence[Any]], *,
                  colors: Mapping[str, str] | None = None,
                  object_alpha: float | None = None,
                  name_template: str = "{subset}",
                  ) -> tuple[dict, dict[str, str], list[str]]:
    """One layer per NAMED SET of segments. Returns ``(state, {set: layer_name}, notes)``.

    Like :func:`split_segment_layer`, but one layer per set, for fading whole populations.
    A set's colour (``colors``) is its ``segmentDefaultColor``. The inherited
    ``segmentColors`` is dropped, since any entry in it would override the default for that
    body. The original layer is kept, emptied and hidden, and the copies replace it in
    place, as in :func:`split_segment_layer`.
    """
    out = json.loads(json.dumps(dict(state)))            # deep copy; never mutate the caller's
    existing = list(out.get("layers") or [])
    index = next((i for i, lyr in enumerate(existing)
                  if isinstance(lyr, dict) and lyr.get("name") == layer), None)
    if index is None:
        known = ", ".join(repr(lyr.get("name")) for lyr in existing if isinstance(lyr, dict))
        raise StateProblem(f"no layer named {layer!r} in this state. It has: {known}")

    source = existing[index]
    if source.get("type") != "segmentation":
        raise StateProblem(
            f"layer {layer!r} is a {source.get('type')!r} layer; only a segmentation layer "
            f"has segments to group")

    notes: list[str] = []
    taken = {lyr.get("name") for lyr in existing if isinstance(lyr, dict)}
    created: dict[str, str] = {}
    copies: list[dict] = []

    for i, (subset, ids) in enumerate(subsets.items()):
        members = [str(s).lstrip("!") for s in ids]
        if not members:
            notes.append(f"set {subset!r} is empty, so no layer was made for it")
            continue
        copy = json.loads(json.dumps(source))
        copy["segments"] = members
        copy.pop("segmentColors", None)
        copy.pop("visible", None)
        if colors and subset in colors:
            # `segmentDefaultColor` must be `#rrggbb`; anything else is silently ignored and
            # the bodies fall back to hash colours. as_hex_color also accepts names and
            # RGB triples.
            copy["segmentDefaultColor"] = as_hex_color(colors[subset], f"colour for {subset!r}")
        if object_alpha is not None:
            copy["objectAlpha"] = float(object_alpha)

        name = name_template.format(subset=subset, layer=layer, index=i)
        if name in taken:
            suffix = 2
            while f"{name}-{suffix}" in taken:
                suffix += 1
            notes.append(f"renamed set layer {name!r} to {name}-{suffix}: the state already "
                         f"has a layer by that name")
            name = f"{name}-{suffix}"
        taken.add(name)
        copy["name"] = name
        created[subset] = name
        copies.append(copy)

    emptied = json.loads(json.dumps(source))
    emptied["segments"] = []
    emptied.pop("segmentColors", None)
    emptied["visible"] = False
    out["layers"] = existing[:index] + copies + [emptied] + existing[index + 1:]
    return out, created, notes


def colors_by_property(properties: Mapping[str, Any], keys: Sequence[str], *,
                       seed: int = 0) -> tuple[dict[str, str], dict[str, Any]]:
    """``({segment_id: "#rrggbb"}, report)``: each segment coloured by the first of ``keys``
    it has a value for, for a segmentation layer's ``segmentColors``.

    ``properties`` is what :func:`neu_glance.sources.read_segment_properties` returns. A key
    is ``label``, or a tag facet (``col`` or ``col:``, taking the value after the colon), or
    a bare flag tag (``glia``). A segment with several values for a facet takes the first.
    Each ``key:value`` gets a fixed colour (:func:`neu_glance.shaders.value_color`), so a
    value is the same colour in every state. Segments with no value keep their hash colour.
    """
    from .shaders import value_color

    ids = list(properties.get("ids") or [])
    vocabulary = list(properties.get("vocabulary") or [])
    numbers = set((properties.get("numbers") or {}))
    facets = sorted({t.split(":", 1)[0] for t in vocabulary if ":" in t})
    flags = sorted(t for t in vocabulary if ":" not in t)

    resolved: list[tuple[str, str]] = []           # (key, "label" | "facet" | "flag")
    for raw in keys:
        key = str(raw).strip().rstrip(":")
        if key == "label":
            resolved.append((key, "label"))
        elif key in facets:
            resolved.append((key, "facet"))
        elif key in flags:
            resolved.append((key, "flag"))
        elif key in numbers:
            raise StateProblem(f"--color-by {key!r} is a number property; colouring is by "
                               f"category: a tag facet, a flag or label")
        else:
            raise StateProblem(
                f"--color-by {key!r} is not a property of this layer. Tag facets: "
                f"{', '.join(facets) or '(none)'}; flags: {', '.join(flags) or '(none)'}; "
                f"or label")

    labels = list(properties.get("labels") or [])
    tags = list(properties.get("tags") or [()] * len(ids))
    colors: dict[str, str] = {}
    by_key = {key: 0 for key, _ in resolved}
    values: set[str] = set()
    several = 0
    for i, sid in enumerate(ids):
        for key, kind in resolved:
            if kind == "label":
                # read_segment_properties labels a segment with no label by its id
                label = labels[i] if i < len(labels) else ""
                found = [label] if label and label != sid else []
            elif kind == "facet":
                found = [t.split(":", 1)[1] for t in tags[i] if t.startswith(f"{key}:")]
            else:
                found = [key] if key in tags[i] else []
            if not found:
                continue
            several += len(found) > 1
            value = f"{key}:{found[0]}"
            colors[sid] = value_color(value, seed)
            values.add(value)
            by_key[key] += 1
            break
    report = {"segments": len(ids), "colored": len(colors), "by_key": by_key,
              "values": len(values), "several_values": several}
    return colors, report


def _read_json(path: str, read_bytes=None):
    if read_bytes is None:
        from neu_vol.location import read_bytes as read_bytes_impl
        read_bytes = read_bytes_impl
    raw = read_bytes(path)
    if raw is None:
        raise StateProblem(f"no such file: {path}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise StateProblem(f"{path} is not valid JSON: {e}") from None


def merge_into(base: Mapping[str, Any], layers: Sequence[dict], *,
               layout: str | None = None,
               position_zyx: Sequence[float] | None = None,
               cross_section_scale: float | None = None,
               projection_scale: float | None = None,
               selected: str | None = None,
               show_slices: bool | None = None,
               projection_orientation: Sequence[float] | None = None
               ) -> tuple[dict, list[str]]:
    """``base`` with ``layers`` appended. Returns the state and any notes worth printing.

    - The base's ``dimensions``, position and zooms are kept unless overridden here, so
      adding a layer does not move the view.
    - Layer names stay unique: neuroglancer keys layers by name, so a clashing incoming
      layer is renamed with a ``-2`` suffix, and the rename is reported.
    """
    state = json.loads(json.dumps(dict(base)))      # deep copy; never mutate the caller's
    existing = list(state.get("layers") or [])
    taken = {lyr.get("name") for lyr in existing if isinstance(lyr, dict)}
    notes: list[str] = []

    added = []
    for layer in layers:
        layer = dict(layer)
        name = layer.get("name")
        if name in taken:
            suffix = 2
            while f"{name}-{suffix}" in taken:
                suffix += 1
            layer["name"] = f"{name}-{suffix}"
            notes.append(f"renamed incoming layer {name!r} to {layer['name']!r}: the state "
                         f"already has a layer by that name")
        taken.add(layer["name"])
        added.append(layer)

    state["layers"] = existing + added
    if not state.get("dimensions"):
        notes.append("the state being added to declares no `dimensions`, so the layers' own "
                     "coordinates may not line up")

    if layout is not None:
        state["layout"] = layout
    if position_zyx is not None:
        state["position"] = [float(v) for v in tuple(position_zyx)[::-1]]
    if cross_section_scale is not None:
        state["crossSectionScale"] = float(cross_section_scale)
    if projection_scale is not None:
        state["projectionScale"] = float(projection_scale)
    if selected:
        state["selectedLayer"] = {"visible": True, "layer": selected}
    if show_slices is not None:
        state["showSlices"] = bool(show_slices)
    if projection_orientation is not None:
        state["projectionOrientation"] = [float(v) for v in projection_orientation]
    return state, notes
