"""GLSL shaders for annotation layers, and the rule for choosing one.

A shader lives in the viewer state, not in the annotation source, so it ships in a link.

**A shader naming a ``prop_`` the source does not declare fails to compile, and the layer
draws nothing**, with the error shown only in the layer's shader tab. :func:`pick_shader`
therefore checks a shader's properties against the source's ``info`` and refuses a
mismatch.
"""

from __future__ import annotations

from typing import Any, Mapping

#: Synapse markers. ``show_pre`` / ``show_post`` toggle the two endpoint markers, and the line
#: (sub-pixel at most zooms) is drawn only when both are on; this is also what lets one source
#: serve as two layers.
#:
#: The confidence test must stay **discard-if-below**. An unknown confidence is NaN, which
#: fails every comparison, so ``discard if conf < min`` keeps those synapses at any threshold
#: while ``keep if conf >= min`` would hide them all. Unknown confidence usually means a
#: synapse added by a proofreader, among the most trustworthy.
SYNAPSE_SHADER = """\
#uicontrol bool show_pre checkbox(default=true)
#uicontrol bool show_post checkbox(default=true)

#uicontrol float pre_size slider(min=0.0, max=20.0, default=6.0)
#uicontrol float post_size slider(min=0.0, max=20.0, default=4.0)

#uicontrol vec3 pre_color color(default="#ff2000")
#uicontrol vec3 post_color color(default="#00c0ff")
#uicontrol vec3 line_color color(default="#ffffff")

#uicontrol float min_conf slider(min=0.0, max=1.0, default=0.0, step=0.01)

void main() {
  // An UNKNOWN confidence is NaN, which fails this comparison and so is never hidden.
  // That is deliberate: a synapse with no confidence value was usually added by hand by a
  // proofreader, making it MORE trustworthy rather than less. Keep this as discard-if-below.
  if (prop_conf_pre() < min_conf || prop_conf_post() < min_conf) discard;

  setEndpointMarkerColor(vec4(pre_color, 1.0), vec4(post_color, 1.0));
  setLineColor(line_color);

  // The line is only meaningful when both ends are shown; otherwise it points at something
  // deliberately hidden, and at these scales its colour would drown the markers.
  if (show_pre && show_post) {
    setLineWidth(1.0);
    setEndpointMarkerSize(pre_size, post_size);
  } else if (show_pre) {
    setLineWidth(0.0);
    setEndpointMarkerSize(pre_size, 0.0);
  } else if (show_post) {
    setLineWidth(0.0);
    setEndpointMarkerSize(0.0, post_size);
  } else {
    setLineWidth(0.0);
    setEndpointMarkerSize(0.0, 0.0);
  }
}
"""

#: Built-in shaders by name, each declaring the properties it reads so
#: :func:`pick_shader` can refuse an unrunnable pairing.
SHADERS: dict[str, dict[str, Any]] = {
    "synapse": {
        "properties": ("conf_pre", "conf_post"),
        "doc": "pre/post endpoint markers, independently toggleable, with a confidence "
               "threshold",
        "source": SYNAPSE_SHADER,
    },
}

#: ``shaderControls`` for the two halves of a split pair; both layers share one shader.
SPLIT_CONTROLS = {
    "pre": {"show_pre": True, "show_post": False},
    "post": {"show_pre": False, "show_post": True},
}


class ShaderProblem(RuntimeError):
    """A shader was named that this source cannot feed, or cannot be found."""


def pick_shader(info: Mapping[str, Any], name: str | None,
                read_bytes=None) -> tuple[str | None, str | None]:
    """``(shader source, why)`` for an annotation layer.

    ``name`` is a built-in name, a path to a GLSL file, ``"none"``, or ``None`` for the
    first built-in whose properties the source declares. ``read_bytes`` reads a shader file
    (default :func:`neu_vol.location.read_bytes`).
    """
    declared = {p["id"] for p in info.get("properties", []) or []}
    if name == "none":
        return None, None
    if name is None:
        for key, entry in SHADERS.items():
            if declared.issuperset(entry["properties"]):
                return entry["source"], f"auto-selected {key!r} ({entry['doc']})"
        return None, ("no built-in shader matches this source's properties "
                      + (f"({', '.join(sorted(declared))})" if declared else "(none)"))
    if name in SHADERS:
        entry = SHADERS[name]
        missing = sorted(set(entry["properties"]) - declared)
        if missing:
            raise ShaderProblem(
                f"shader {name!r} reads {', '.join(missing)}, which this source does not "
                f"declare. A shader naming a property that is absent fails to compile and "
                f"the layer draws nothing. Declared: "
                + (", ".join(sorted(declared)) or "no properties"))
        return entry["source"], f"built-in {name!r}"

    reader = read_bytes
    if reader is None:
        from neu_vol.location import read_bytes as reader

    raw = reader(name)
    if raw is None:
        raise ShaderProblem(
            f"shader {name!r} is neither a built-in name ("
            + ", ".join(SHADERS) + ", none) nor a readable file")
    return raw.decode(), f"from {name}"


# --------------------------------------------------------------------------- #
# Image shaders: chosen by data kind and channel count. They read getDataValue(channel),
# so the only thing to check is the number of channels.
# --------------------------------------------------------------------------- #

#: Window/level over one channel, the default for images. Neuroglancer's own default maps
#: the whole dtype range, which washes out EM data.
GRAYSCALE_SHADER = """\
#uicontrol invlerp normalized
#uicontrol float gamma slider(min=0.05, max=3.0, default=1.0, step=0.05)

void main() {
  float v = pow(normalized(), gamma);
  emitGrayscale(v);
}
"""

#: One colour with opacity carrying the probability, the default for probability maps: EM
#: drawn underneath stays visible where the model is unsure. The ``color`` uicontrol
#: defaults to red, and ``ServedLayer(color=...)`` sets it.
#:
#: Keep the threshold discard-if-below, so NaN voxels stay visible (see
#: :data:`SYNAPSE_SHADER`), and keep NaN pinned to full alpha, since ``clamp(NaN)`` is
#: undefined.
PROBABILITY_SHADER = """\
#uicontrol invlerp normalized
#uicontrol vec3 color color(default="#ff0000")
#uicontrol float threshold slider(min=0.0, max=1.0, default=0.0, step=0.01)
#uicontrol float gain slider(min=0.0, max=4.0, default=1.0, step=0.05)

void main() {
  float v = normalized();
  // discard-if-below, so a NaN voxel stays visible at every threshold
  if (v < threshold) discard;
  // v != v is true only for NaN. The alpha comes from the value here, so an unscored voxel
  // would otherwise be kept by the line above and then drawn at an undefined alpha.
  float a = (v != v) ? 1.0 : clamp(v * gain, 0.0, 1.0);
  emitRGBA(vec4(color, a));
}
"""

#: One channel through a two-colour gradient, with a threshold (discard-if-below, so NaN
#: voxels stay visible).
COLORMAP_SHADER = """\
#uicontrol invlerp normalized
#uicontrol vec3 low_color color(default="#000060")
#uicontrol vec3 high_color color(default="#ffe000")
#uicontrol float threshold slider(min=0.0, max=1.0, default=0.0, step=0.01)
#uicontrol float opacity slider(min=0.0, max=1.0, default=1.0, step=0.01)

void main() {
  float v = normalized();
  // discard-if-below, so a NaN voxel stays visible at every threshold
  if (v < threshold) discard;
  emitRGBA(vec4(mix(low_color, high_color, v), opacity));
}
"""

#: Three channels to three colours with independent gains, e.g. affinities.
RGB_SHADER = """\
#uicontrol vec3 color0 color(default="#ff2000")
#uicontrol vec3 color1 color(default="#00c000")
#uicontrol vec3 color2 color(default="#2060ff")
#uicontrol float gain0 slider(min=0.0, max=4.0, default=1.0, step=0.05)
#uicontrol float gain1 slider(min=0.0, max=4.0, default=1.0, step=0.05)
#uicontrol float gain2 slider(min=0.0, max=4.0, default=1.0, step=0.05)
#uicontrol float opacity slider(min=0.0, max=1.0, default=1.0, step=0.01)

void main() {
  vec3 rgb = color0 * (toNormalized(getDataValue(0)) * gain0)
           + color1 * (toNormalized(getDataValue(1)) * gain1)
           + color2 * (toNormalized(getDataValue(2)) * gain2);
  emitRGBA(vec4(clamp(rgb, 0.0, 1.0), opacity));
}
"""

#: Built-in image shaders by name, with the channel count each reads. ``color`` names the
#: shader's single colour uicontrol, which ``ServedLayer(color=...)`` sets; it is absent for
#: shaders with none or several. Neuroglancer silently ignores an unknown shaderControl.
IMAGE_SHADERS: dict[str, dict[str, Any]] = {
    "grayscale": {"channels": 1, "doc": "window/level with gamma, single channel",
                  "source": GRAYSCALE_SHADER},
    "probability": {"channels": 1, "color": "color",
                    "doc": "one colour with opacity carrying the value, and a "
                           "discard-if-below threshold",
                    "source": PROBABILITY_SHADER},
    "colormap": {"channels": 1, "color": "high_color",
                 "doc": "one channel through a two-colour gradient, with "
                        "a discard-if-below threshold",
                 "source": COLORMAP_SHADER},
    "rgb": {"channels": 3, "doc": "three channels to three colours with independent gains",
            "source": RGB_SHADER},
}

#: The image shader a layer gets when none is named, by kind.
_DEFAULT_IMAGE_SHADER = {"image": "grayscale", "probability": "probability"}


#: Common colour names that work without matplotlib. Each value is matplotlib's own, so
#: installing it never changes a colour. Single letters (``"c"``) are left to matplotlib,
#: whose ``"c"`` differs from CSS ``"cyan"``.
BUILTIN_COLORS = {
    "red": "#ff0000",
    "green": "#008000",
    "blue": "#0000ff",
    "cyan": "#00ffff",
    "magenta": "#ff00ff",
    "yellow": "#ffff00",
    "orange": "#ffa500",
    "purple": "#800080",
    "lime": "#00ff00",
    "white": "#ffffff",
    "black": "#000000",
    "gray": "#808080",
    "grey": "#808080",
}


def _hex_from_rgb(values: Sequence[float], scale: float) -> str:
    channels = [int(round(min(255.0, max(0.0, v * scale)))) for v in values]
    return "#" + "".join(f"{c:02x}" for c in channels)


def _named_color(color: str, what: str) -> str:
    """A named colour as ``#rrggbb``: from :data:`BUILTIN_COLORS`, else from matplotlib if
    it is installed (it is not a dependency, and is imported only here)."""
    known = BUILTIN_COLORS.get(color.lower())
    if known:
        return known
    try:
        from matplotlib.colors import to_hex
    except ImportError:
        raise ShaderProblem(
            f"{what} {color!r} is not a #rrggbb colour or one of the built-in names "
            f"({', '.join(sorted(set(BUILTIN_COLORS)))}). Any colour matplotlib names "
            f"('forestgreen', 'tab:blue') works when matplotlib is installed, which it is "
            f"not here and is not a dependency of this package — pass '#rrggbb' or a "
            f"3-sequence instead") from None
    try:
        return to_hex(color)
    except ValueError as e:
        raise ShaderProblem(f"{what} {color!r} is not a colour matplotlib knows: {e}") \
            from None


def as_hex_color(color: Any, what: str = "color") -> str:
    """A colour as ``#rrggbb``, which is the only form a shaderControl takes.

    Accepts:

    * ``"#ff0000"`` or ``"ff0000"``;
    * a name from :data:`BUILTIN_COLORS` (``"red"``, ``"cyan"``, ...);
    * any other name matplotlib knows (``"tab:blue"``), if matplotlib is installed;
    * a 3-sequence, read as 0..1 if every value is at most 1, else as 0..255 (so
      ``(1, 1, 1)`` is white).
    """
    if isinstance(color, str):
        text = color.strip().lstrip("#")
        if len(text) == 6 and all(c in "0123456789abcdefABCDEF" for c in text):
            return "#" + text.lower()
        return _named_color(color.strip(), what)
    try:
        values = [float(v) for v in color]
    except TypeError:
        raise ShaderProblem(
            f"{what} must be a colour name, '#rrggbb', or a 3-sequence, got "
            f"{type(color).__name__}") from None
    if len(values) != 3:
        extra = (" — alpha is not settable here: for a probability layer the opacity "
                 "carries the value" if len(values) == 4 else "")
        raise ShaderProblem(f"{what} needs 3 components, got {len(values)}{extra}")
    return _hex_from_rgb(values, 255.0 if all(v <= 1.0 for v in values) else 1.0)


def shader_color_control(name: str | None, *, kind: str = "image",
                         channels: int = 1) -> str:
    """The uicontrol a caller's ``color`` should set, for the shader that will be chosen.

    Raises when that shader has no single colour, since neuroglancer would silently ignore
    the control.
    """
    resolved = name
    if resolved in (None, "none"):
        resolved = "rgb" if channels == 3 else _DEFAULT_IMAGE_SHADER.get(kind, "grayscale")
    entry = IMAGE_SHADERS.get(resolved) or {}
    control = entry.get("color")
    if control is None:
        colourable = ", ".join(k for k, e in IMAGE_SHADERS.items() if e.get("color"))
        raise ShaderProblem(
            f"shader {resolved!r} has no single colour to set"
            + (" (it has three: color0/color1/color2)" if resolved == "rgb" else "")
            + f", so color= would be a shaderControl neuroglancer ignores in silence. "
              f"Shaders that take one: {colourable}. Or pass the whole shader with "
              f"shader=<glsl>.")
    return control


def image_shader(name: str | None, *, kind: str = "image", channels: int = 1) -> str | None:
    """GLSL for an image-like layer, or ``None`` for neuroglancer's own default.

    ``name`` is a built-in name, ``"none"``, or ``None`` for the default for ``kind``:
    ``grayscale`` for an image, ``probability`` for a probability map, and ``rgb`` for any
    3-channel array. A shader whose channel count does not match the volume raises.
    """
    if name == "none":
        return None
    if name is None:
        name = "rgb" if channels == 3 else _DEFAULT_IMAGE_SHADER.get(kind, "grayscale")
    entry = IMAGE_SHADERS.get(name)
    if entry is None:
        raise ShaderProblem(
            f"unknown image shader {name!r}; built-ins are "
            + ", ".join(IMAGE_SHADERS) + ", none")
    if entry["channels"] != channels:
        raise ShaderProblem(
            f"shader {name!r} reads {entry['channels']} channel(s) and this volume has "
            f"{channels}. Asking for a channel that is not there fails to compile, and "
            f"reading one of three shows a third of the data with nothing to say so.")
    return entry["source"]
