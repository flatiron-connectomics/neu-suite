"""Site defaults for ``neu-glance gen --config``: display choices kept out of the package.

Pure: parses TOML into checked settings and applies them to layer dicts. Nothing is read
unless ``--config`` names a file, so a state is reproducible from its command line. A flag
on the command line wins over the file.

::

    [volume]
    bounds = false              # draw each volume's bounding box

    [annotations.shader_controls]
    line_color = "#ffff00"      # any #uicontrol of the layer's shader

    [view]                      # as neuroglancer writes them: copy from a saved state
    position = [6476.7, 4508.8, 5568.2]          # x, y, z, in the state's voxels
    projection_orientation = [-0.1, 0.715, 0.682, 0.116]   # the 3D view's quaternion
    projection_scale = 15812.5  # 3D zoom
    cross_section_scale = 15.8  # 2D zoom
    layout = "4panel"
    select = "seg"              # the layer the side panel opens on

    [layer.head]                # by layer name
    silhouette = 5              # meshSilhouetteRendering, 0-10
    opacity_2d = 0              # selectedAlpha
    opacity_3d = 1              # objectAlpha
    color = "white"             # segmentDefaultColor: every segment without its own colour
    pick = false                # clicks go through it, to what is inside
    tab = "segments"            # the side panel's tab for this layer
    hide = true                 # added switched off
    all_segments = true         # every id its segment_properties lists
    color_by = ["col", "group"] # segmentColors by property, first one present (--color-by)

A ``[layer.NAME]`` the state has no layer for is reported, not refused: one site file serves
many states. An unknown key is refused, since neuroglancer ignores what it does not know.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

#: ``[layer.NAME]`` display keys and the state key each sets, with its range.
LAYER_DISPLAY = {
    "silhouette": ("meshSilhouetteRendering", 0.0, 10.0),
    "opacity_2d": ("selectedAlpha", 0.0, 1.0),
    "opacity_3d": ("objectAlpha", 0.0, 1.0),
}
#: ``[layer.NAME]`` colour keys and the state key each sets, as ``#rrggbb``.
LAYER_COLORS = {"color": "segmentDefaultColor"}
#: ``[layer.NAME]`` true/false keys written to the layer as they are.
LAYER_SWITCHES = {"pick": "pick"}
#: Side-panel tabs a ``[layer.NAME] tab`` may name (neuroglancer's own ids).
LAYER_TABS = ("source", "rendering", "annotations", "segments")
LAYER_FLAGS = ("hide", "all_segments")
#: Keys the caller applies, as they need a store read or merge with flags.
CALLER_KEYS = (*LAYER_FLAGS, "color_by")
#: ``[view]`` keys: (how many numbers, or None for a string).
VIEW_KEYS = {"position": 3, "projection_orientation": 4, "projection_scale": 1,
             "cross_section_scale": 1, "layout": None, "select": None}
_SECTIONS = {"volume", "annotations", "view", "layer"}


class ConfigProblem(ValueError):
    """A config file that cannot be applied as written."""


def load(path: str, read_bytes=None) -> dict[str, Any]:
    """Parse and check a config file. Returns ``{"volume", "shader_controls", "layers"}``."""
    import tomllib

    if read_bytes is None:
        from neu_vol.location import read_bytes
    raw = read_bytes(path)
    if raw is None:
        raise ConfigProblem(f"no config at {path}")
    try:
        doc = tomllib.loads(raw.decode())
    except tomllib.TOMLDecodeError as e:
        raise ConfigProblem(f"{path}: {e}") from None
    return parse(doc, where=path)


def parse(doc: Mapping[str, Any], *, where: str = "config") -> dict[str, Any]:
    unknown = set(doc) - _SECTIONS
    if unknown:
        raise ConfigProblem(f"{where}: unknown section(s) {sorted(unknown)}; expected "
                            f"{sorted(_SECTIONS)}")

    volume = dict(doc.get("volume") or {})
    bad = set(volume) - {"bounds"}
    if bad:
        raise ConfigProblem(f"{where}: [volume] has unknown key(s) {sorted(bad)}")
    if "bounds" in volume and not isinstance(volume["bounds"], bool):
        raise ConfigProblem(f"{where}: [volume] bounds must be true or false")

    annotations = dict(doc.get("annotations") or {})
    bad = set(annotations) - {"shader_controls"}
    if bad:
        raise ConfigProblem(f"{where}: [annotations] has unknown key(s) {sorted(bad)}")
    controls = dict(annotations.get("shader_controls") or {})

    view = dict(doc.get("view") or {})
    bad = set(view) - set(VIEW_KEYS)
    if bad:
        raise ConfigProblem(f"{where}: [view] has unknown key(s) {sorted(bad)}; known: "
                            f"{', '.join(VIEW_KEYS)}")
    for key, value in view.items():
        size = VIEW_KEYS[key]
        if size is None:
            if not isinstance(value, str) or not value:
                raise ConfigProblem(f"{where}: [view] {key} must be a name")
            continue
        values = [value] if size == 1 else value
        if (not isinstance(values, list) or len(values) != size
                or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                           for v in values)):
            raise ConfigProblem(f"{where}: [view] {key} must be "
                                + ("a number" if size == 1 else f"{size} numbers"))
        view[key] = float(value) if size == 1 else [float(v) for v in values]

    layers: dict[str, dict[str, Any]] = {}
    for name, entry in (doc.get("layer") or {}).items():
        if not isinstance(entry, Mapping):
            raise ConfigProblem(f"{where}: layer.{name} must be a table")
        known = [*LAYER_DISPLAY, *LAYER_COLORS, *LAYER_SWITCHES, "tab", *CALLER_KEYS]
        bad = set(entry) - set(known)
        if bad:
            raise ConfigProblem(
                f"{where}: [layer.{name}] has unknown key(s) {sorted(bad)}; known: "
                f"{', '.join(known)}")
        out: dict[str, Any] = {}
        for key, value in entry.items():
            if key == "tab":
                if value not in LAYER_TABS:
                    raise ConfigProblem(f"{where}: [layer.{name}] tab must be one of "
                                        f"{', '.join(LAYER_TABS)}, not {value!r}")
                out["tab"] = value
                continue
            if key == "color_by":
                keys = [value] if isinstance(value, str) else value
                if (not isinstance(keys, list) or not keys
                        or not all(isinstance(k, str) and k for k in keys)):
                    raise ConfigProblem(f"{where}: [layer.{name}] color_by must be a property "
                                        f"name or a list of them")
                out[key] = keys
                continue
            if key in LAYER_FLAGS or key in LAYER_SWITCHES:
                if not isinstance(value, bool):
                    raise ConfigProblem(f"{where}: [layer.{name}] {key} must be true or false")
                out[LAYER_SWITCHES.get(key, key)] = value
                continue
            if key in LAYER_COLORS:
                from .shaders import ShaderProblem, as_hex_color

                try:
                    out[LAYER_COLORS[key]] = as_hex_color(value, f"[layer.{name}] {key}")
                except ShaderProblem as e:
                    raise ConfigProblem(f"{where}: {e}") from None
                continue
            state_key, lo, hi = LAYER_DISPLAY[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigProblem(f"{where}: [layer.{name}] {key} must be a number")
            if not lo <= float(value) <= hi:
                raise ConfigProblem(f"{where}: [layer.{name}] {key} must be between "
                                    f"{lo:g} and {hi:g}")
            out[state_key] = float(value)
        layers[str(name)] = out
    return {"volume": volume, "shader_controls": controls, "view": view, "layers": layers}


def shader_control_names(shader: str) -> set[str]:
    """The ``#uicontrol`` names a shader declares."""
    return set(re.findall(r"^#uicontrol\s+\w+\s+(\w+)", shader, flags=re.M))


def apply(config: Mapping[str, Any], layers: list[dict]) -> list[str]:
    """Apply ``[annotations]`` and ``[layer.*]`` display settings in place. Returns notes.

    ``hide``, ``all_segments`` and ``color_by`` are not applied here: the caller merges them
    with its flags, and ``color_by`` reads the layer's segment properties.
    """
    notes: list[str] = []
    by_name = {lyr["name"]: lyr for lyr in layers}

    controls = config.get("shader_controls") or {}
    if controls:
        for lyr in layers:
            if lyr["type"] != "annotation" or "shader" not in lyr:
                continue
            missing = sorted(set(controls) - shader_control_names(lyr["shader"]))
            if missing:
                raise ConfigProblem(
                    f"[annotations.shader_controls] sets {', '.join(missing)}, which the "
                    f"shader of layer {lyr['name']!r} does not declare; neuroglancer would "
                    f"ignore it. Its controls: "
                    + ", ".join(sorted(shader_control_names(lyr["shader"]))))
            lyr["shaderControls"] = {**controls, **(lyr.get("shaderControls") or {})}

    for name, settings in (config.get("layers") or {}).items():
        layer = by_name.get(name)
        if layer is None:
            notes.append(f"config [layer.{name}]: no such layer in this state, skipped")
            continue
        display = {k: v for k, v in settings.items() if k not in CALLER_KEYS}
        segmentation_only = sorted(set(display) - {"tab"})
        if segmentation_only and layer["type"] != "segmentation":
            raise ConfigProblem(f"config [layer.{name}] sets "
                                f"{', '.join(segmentation_only)}, which only a segmentation "
                                f"layer has; {name!r} is a {layer['type']} layer")
        if display.get("tab") == "segments" and layer["type"] != "segmentation":
            raise ConfigProblem(f"config [layer.{name}] tab 'segments' exists only on a "
                                f"segmentation layer; {name!r} is a {layer['type']} layer")
        layer.update(display)
    view = config.get("view") or {}
    if view.get("select") and view["select"] not in by_name:
        notes.append(f"config [view] select {view['select']!r}: no such layer in this state, "
                     f"skipped")
    return notes
