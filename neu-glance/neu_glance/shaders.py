"""GLSL shaders for annotation layers, and the rule for choosing one.

A shader lives in the viewer state, not in the annotation source, so it ships in a link.

**A shader naming a ``prop_`` the source does not declare fails to compile, and the layer
draws nothing**, with the error shown only in the layer's shader tab. :func:`pick_shader`
therefore checks a shader's properties against the source's ``info`` and refuses a
mismatch.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

#: The synapse shader's optional colourings, in the order the first one ticked wins:
#: ``(checkbox, properties read, GLSL setting pre/post)``. Each is included only when the
#: source declares its properties, since a shader naming an absent one fails to compile.
#: An id colour is neuroglancer's own segment colour for that id (colour seed 0), so a
#: synapse coloured by neuropil matches that neuropil's mesh in the ROI layers.
SYNAPSE_COLORINGS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("color_by_neuropil", ("neuropil",),
     "pre = post = id_color(uint(prop_neuropil()));"),
    ("color_by_neuropil_part", ("roi",),
     "pre = post = id_color(uint(prop_roi()));"),
    ("color_by_neuron_id", ("body_pre_u32", "body_post_u32"),
     "pre = id_color(uint(prop_body_pre_u32()));\n"
     "    post = id_color(uint(prop_body_post_u32()));"),
    ("color_by_tbar_fanout", ("tbar_fanout",),
     "pre = viridis(min(float(prop_tbar_fanout()), max_fanout) / max_fanout);"),
)

#: The synapse shader's optional filters: ``(checkbox, properties read, GLSL)``, included
#: like :data:`SYNAPSE_COLORINGS`. ``*_listed`` is 1 where that end's body is one the
#: segmentation shows (``neu-mark annotation-source --listed``).
SYNAPSE_FILTERS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("hide_unlisted", ("pre_listed", "post_listed"),
     "if (hide_unlisted && prop_pre_listed() == 0u && prop_post_listed() == 0u) { discard; }"),
    ("hide_unlisted_partner", ("pre_listed", "post_listed"),
     "if (hide_unlisted_partner && (prop_pre_listed() == 0u || prop_post_listed() == 0u)) "
     "{ discard; }"),
)

_SYNAPSE_CONTROLS = """\
#uicontrol bool show_pre checkbox(default=true)
#uicontrol bool show_post checkbox(default=true)

#uicontrol float pre_size slider(min=0.0, max=20.0, default=6.0)
#uicontrol float post_size slider(min=0.0, max=20.0, default=4.0)

#uicontrol vec3 pre_color color(default="#ff2000")
#uicontrol vec3 post_color color(default="#00c0ff")
#uicontrol vec3 line_color color(default="#ffff00")

#uicontrol float min_conf slider(min=0.0, max=1.0, default=0.0, step=0.01)
"""

# neuroglancer's SegmentColorHash: murmur3 mixing of the id's low then high 32 bits, then
# HSV with the first byte as hue. Ids here are 32-bit, so the high word is 0.
_ID_COLOR_GLSL = """\
uint hash_combine(uint state, uint value) {
  value *= 0xcc9e2d51u;
  value = (value << 15) | (value >> 17);
  value *= 0x1b873593u;
  state ^= value;
  state = (state << 13) | (state >> 19);
  return state * 5u + 0xe6546b64u;
}

vec3 hsv_to_rgb(float h, float s, float v) {
  h *= 6.0;
  float i = floor(h);
  float f = h - i;
  float p = v * (1.0 - s);
  float q = v * (1.0 - s * f);
  float t = v * (1.0 - s * (1.0 - f));
  int k = int(i) % 6;
  if (k == 0) return vec3(v, t, p);
  if (k == 1) return vec3(q, v, p);
  if (k == 2) return vec3(p, v, t);
  if (k == 3) return vec3(p, q, v);
  if (k == 4) return vec3(t, p, v);
  return vec3(v, p, q);
}

// Id 0 is "none": white, or not drawn with hide_uncolored.
vec3 id_color(uint id) {
  if (id == 0u) {
    if (hide_uncolored) { discard; }
    return vec3(1.0);
  }
  uint h = hash_combine(hash_combine(0u, id), 0u);
  return hsv_to_rgb(float(h & 255u) / 255.0, 0.5 + 0.5 * float((h >> 8) & 255u) / 255.0, 1.0);
}
"""

# A polynomial fit to matplotlib's viridis.
_VIRIDIS_GLSL = """\
vec3 viridis(float t) {
  const vec3 c0 = vec3(0.2777273272234177, 0.005407344544966578, 0.3340998053353061);
  const vec3 c1 = vec3(0.1050930431085774, 1.404613529898575, 1.384590162594685);
  const vec3 c2 = vec3(-0.3308618287255563, 0.214847559468213, 0.09509516302823659);
  const vec3 c3 = vec3(-4.634230498983486, -5.799100973351585, -19.33244095627987);
  const vec3 c4 = vec3(6.228269936347081, 14.17993336680509, 56.69055260068105);
  const vec3 c5 = vec3(4.776384997670288, -13.74514537774601, -65.35303263337234);
  const vec3 c6 = vec3(-5.435455855934631, 4.645852612178535, 26.3124352495832);
  return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}
"""

_SYNAPSE_MAIN = """\
void main() {
  // An UNKNOWN confidence is NaN, which fails this comparison and so is never hidden.
  // That is deliberate: a synapse with no confidence value was usually added by hand by a
  // proofreader, making it MORE trustworthy rather than less. Keep this as discard-if-below.
  if (prop_conf_pre() < min_conf || prop_conf_post() < min_conf) discard;
{filters}
  vec3 pre = pre_color;
  vec3 post = post_color;
{coloring}
  setEndpointMarkerColor(vec4(pre, 1.0), vec4(post, 1.0));
{border}  setLineColor(line_color);

  // The line is only meaningful when both ends are shown; otherwise it points at something
  // deliberately hidden.
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


def synapse_colorings(declared: Iterable[str]) -> list[str]:
    """The checkboxes of :data:`SYNAPSE_COLORINGS` a source declaring ``declared`` gets."""
    have = set(declared)
    return [name for name, props, _ in SYNAPSE_COLORINGS if have.issuperset(props)]


def synapse_filters(declared: Iterable[str]) -> list[str]:
    """The checkboxes of :data:`SYNAPSE_FILTERS` a source declaring ``declared`` gets."""
    have = set(declared)
    return [name for name, props, _ in SYNAPSE_FILTERS if have.issuperset(props)]


def synapse_shader(declared: Iterable[str] = ()) -> str:
    """The synapse shader for a source whose properties are ``declared``.

    ``show_pre`` / ``show_post`` toggle the two endpoint markers, and the line is drawn only
    when both are on. A ``color_by_*`` checkbox is added for each colouring the properties
    allow; the first one ticked wins.

    The confidence test must stay **discard-if-below**. An unknown confidence is NaN, which
    fails every comparison, so ``discard if conf < min`` keeps those synapses at any
    threshold while ``keep if conf >= min`` would hide them all. Unknown confidence usually
    means a synapse added by a proofreader, among the most trustworthy.
    """
    declared = set(declared)
    chosen = set(synapse_colorings(declared))
    entries = [(name, glsl) for name, _, glsl in SYNAPSE_COLORINGS if name in chosen]
    filtering = set(synapse_filters(declared))
    filters = [(name, glsl) for name, _, glsl in SYNAPSE_FILTERS if name in filtering]
    parts = [_SYNAPSE_CONTROLS]
    if filters:
        parts.append("\n".join(f"#uicontrol bool {name} checkbox(default=false)"
                               for name, _ in filters) + "\n")
    if entries:
        controls = [f"#uicontrol bool {name} checkbox(default=false)" for name, _ in entries]
        controls.append("#uicontrol float border_width slider(min=0.0, max=6.0, step=0.5, "
                        "default=0.5)")
        if "color_by_tbar_fanout" in chosen:
            controls.append("#uicontrol float max_fanout slider(min=1.0, max=40.0, step=1.0, "
                            "default=20.0)")
        if chosen - {"color_by_tbar_fanout"}:
            controls.append("#uicontrol bool hide_uncolored checkbox(default=false)")
        parts.append("\n".join(controls) + "\n")
        if chosen - {"color_by_tbar_fanout"}:
            parts.append(_ID_COLOR_GLSL)
        if "color_by_tbar_fanout" in chosen:
            parts.append(_VIRIDIS_GLSL)
        branches = "  " + " else ".join(f"if ({name}) {{\n    {glsl}\n  }}"
                                        for name, glsl in entries) + "\n"
        # With a colouring on, the fill no longer says which end is which, so the border
        # keeps the pre / post colour.
        border = ("  if (" + " || ".join(name for name, _ in entries) + ") {\n"
                  "    setEndpointMarkerBorderColor(pre_color, post_color);\n"
                  "    setEndpointMarkerBorderWidth(border_width, border_width);\n"
                  "  }\n")
    else:
        branches = border = ""
    parts.append(_SYNAPSE_MAIN.replace("{coloring}", branches).replace(
        "{filters}", "".join(f"  {glsl}\n" for _, glsl in filters)).replace(
        "{border}", border))
    return "\n".join(parts)


#: Every colouring and filter included, as `neu-glance shaders synapse` prints it.
SYNAPSE_SHADER = synapse_shader(p for _, props, _ in (*SYNAPSE_COLORINGS, *SYNAPSE_FILTERS)
                                for p in props)

#: Built-in shaders by name, each declaring the properties it requires so
#: :func:`pick_shader` can refuse an unrunnable pairing. ``build`` makes the source from the
#: properties declared, for a shader with optional parts.
SHADERS: dict[str, dict[str, Any]] = {
    "synapse": {
        "properties": ("conf_pre", "conf_post"),
        "doc": "pre/post endpoint markers, independently toggleable, with a confidence "
               "threshold and, where the source has the properties, colouring by "
               "neuropil, neuron or T-bar fanout",
        "source": SYNAPSE_SHADER,
        "build": synapse_shader,
    },
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
                return (_built(entry, declared),
                        f"auto-selected {key!r} ({entry['doc']})" + _extras(key, declared))
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
        return _built(entry, declared), f"built-in {name!r}" + _extras(name, declared)

    reader = read_bytes
    if reader is None:
        from neu_vol.location import read_bytes as reader

    raw = reader(name)
    if raw is None:
        raise ShaderProblem(
            f"shader {name!r} is neither a built-in name ("
            + ", ".join(SHADERS) + ", none) nor a readable file")
    return raw.decode(), f"from {name}"


def _built(entry: Mapping[str, Any], declared: set[str]) -> str:
    build = entry.get("build")
    return build(declared) if build else entry["source"]


def _extras(name: str, declared: set[str]) -> str:
    if name != "synapse":
        return ""
    found = synapse_colorings(declared)
    text = f"; colourings: {', '.join(found)}" if found else "; no colourings"
    hides = synapse_filters(declared)
    return text + (f"; filters: {', '.join(hides)}" if hides else "")


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
#: NaN voxels (no prediction) are discarded, so they never cover the EM.
PROBABILITY_SHADER = """\
#uicontrol invlerp normalized
#uicontrol vec3 color color(default="#ff0000")
#uicontrol float threshold slider(min=0.0, max=1.0, default=0.0, step=0.01)
#uicontrol float gain slider(min=0.0, max=4.0, default=1.0, step=0.05)

void main() {
  float v = normalized();
  // v != v is true only for NaN: a voxel with no prediction is not drawn.
  if (v != v || v < threshold) discard;
  emitRGBA(vec4(color, clamp(v * gain, 0.0, 1.0)));
}
"""

#: One channel through a two-colour gradient, with a threshold. NaN voxels are discarded.
COLORMAP_SHADER = """\
#uicontrol invlerp normalized
#uicontrol vec3 low_color color(default="#000060")
#uicontrol vec3 high_color color(default="#ffe000")
#uicontrol float threshold slider(min=0.0, max=1.0, default=0.0, step=0.01)
#uicontrol float opacity slider(min=0.0, max=1.0, default=1.0, step=0.01)

void main() {
  float v = normalized();
  // v != v is true only for NaN: a voxel with no prediction is not drawn.
  if (v != v || v < threshold) discard;
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


def _hash_combine(state: int, value: int) -> int:
    value = (value * 0xcc9e2d51) & 0xFFFFFFFF
    value = ((value << 15) | (value >> 17)) & 0xFFFFFFFF
    value = (value * 0x1b873593) & 0xFFFFFFFF
    state ^= value
    state = ((state << 13) | (state >> 19)) & 0xFFFFFFFF
    return (state * 5 + 0xe6546b64) & 0xFFFFFFFF


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    h *= 6.0
    i = int(h // 1)
    f = h - i
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    return [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i % 6]


def segment_color(segment_id: int, seed: int = 0) -> str:
    """``#rrggbb`` neuroglancer gives ``segment_id`` with colour seed ``seed``.

    A port of its ``SegmentColorHash.compute``, the same function the synapse shader's
    ``id_color`` implements in GLSL.
    """
    segment_id = int(segment_id)
    h = _hash_combine(_hash_combine(int(seed) & 0xFFFFFFFF, segment_id & 0xFFFFFFFF),
                      (segment_id >> 32) & 0xFFFFFFFF)
    rgb = _hsv_to_rgb((h & 255) / 255, 0.5 + ((h >> 8) & 255) / 255 * 0.5, 1.0)
    return _hex_from_rgb(rgb, 255.0)


def value_color(value: str, seed: int = 0) -> str:
    """A stable ``#rrggbb`` for a property value, in neuroglancer's segment palette: the value
    is hashed to a 64-bit id (blake2b), which :func:`segment_color` colours."""
    import hashlib

    digest = hashlib.blake2b(str(value).encode(), digest_size=8).digest()
    return segment_color(int.from_bytes(digest, "little"), seed)


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
