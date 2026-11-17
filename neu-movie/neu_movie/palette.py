"""Colours: the palette sets are drawn from, and the conversions neuVid needs."""

import math

#: Matplotlib's tab10 and Dark2, as hex.
TAB10 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2",
         "#7f7f7f", "#bcbd22", "#17becf"]
DARK2 = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d",
         "#666666"]

#: Colours dimmer than this (WCAG relative luminance, 0 black to 1 white) vanish against a
#: black background, as cells and as legend text. 0.15 drops tab10's brown (0.127) and
#: Dark2's grey (0.133) and keeps the rest of both.
MIN_LUMINANCE = 0.15


def rgb(hex_color):
    return tuple(int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))


def to_hex(rgb_floats):
    return "#" + "".join(f"{round(c * 255):02x}" for c in rgb_floats[:3])


def _linear(v):
    return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4


def luminance(hex_color):
    """WCAG relative luminance of an sRGB hex colour."""
    r, g, b = (_linear(v) for v in rgb(hex_color))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


#: Set colours, assigned in order of first appearance: tab10 then Dark2, without dim ones.
SETS = [c for c in TAB10 + DARK2 if luminance(c) >= MIN_LUMINANCE]

#: Matplotlib's Set2, as hex.
SET2 = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494",
        "#b3b3b3"]

#: What neuropils draw from, in order.
NEUROPILS = TAB10 + DARK2 + SET2

#: Nearer than this (CIE76 delta E), two colours read as one, e.g. Dark2's orange (14 from
#: tab10's) or Set2's pink (11).
MIN_DELTA_E = 25.0


def lab(hex_color):
    """CIE L*a*b* (D65) of an sRGB hex colour."""
    r, g, b = (_linear(v) for v in rgb(hex_color))
    xyz = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047, \
        0.2126 * r + 0.7152 * g + 0.0722 * b, \
        (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883
    fx, fy, fz = (t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116 for t in xyz)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def delta_e(a, b):
    """CIE76 colour difference: about 2 is just noticeable, 25 a different colour."""
    return sum((p - q) ** 2 for p, q in zip(lab(a), lab(b))) ** 0.5


def chroma(hex_color):
    """CIE LCh chroma: 0 for a grey, 40 and more for a strong colour."""
    _, a, b = lab(hex_color)
    return (a * a + b * b) ** 0.5


def hue_gap(a, b):
    """Degrees between two colours' CIE LCh hues, 0 to 180."""
    (_, a1, b1), (_, a2, b2) = lab(a), lab(b)
    d = abs(math.degrees(math.atan2(b1, a1) - math.atan2(b2, a2))) % 360
    return min(d, 360 - d)


#: A neuropil is drawn muted and nearly transparent; below this chroma, once muted, it
#: reads as grey, the head's colour (tab10's brown muted by half is 12).
MIN_CHROMA = 20.0

#: Hue a neuropil keeps from the neurons shown with it, which sit inside it. tab10's cyan is
#: 58 degrees from its blue and the two read as one blue.
MIN_OWN_HUE = 90.0

#: Hue between neuropils shown with the same set, which that set's neurons join, so they
#: are near each other. delta E alone passes Dark2's gold and brown (25 apart, 5 degrees of
#: hue), which look alike.
MIN_SIBLING_HUE = 30.0


def neuropil_colours(regions, given, avoid, own=None, amount=0.5):
    """``{region: "#rrggbb"}``, in the order of `regions`.

    `given` holds colours chosen for regions (the spec's), kept as they are. `own` maps a
    region to the colours of the sets shown with it, `avoid` lists every set's colour, and
    `amount` is how far neuropils are muted toward grey. A colour fits a region if, once
    muted, it keeps `MIN_CHROMA` (it does not read as the head's grey) and `MIN_OWN_HUE`
    from its own sets (it does not wear the colour of the neurons inside it). Any other
    region takes the `NEUROPILS` colour that fits and is, in order of preference:
    `MIN_SIBLING_HUE` from the colours of its own sets' other neuropils (given or already
    chosen), the least used, `MIN_DELTA_E` from every set, the earliest."""
    avoid, own = list(avoid), own or {}

    def fits(c, region):
        m = mute(c, amount)
        return chroma(m) >= MIN_CHROMA and all(
            hue_gap(m, o) >= MIN_OWN_HUE for o in own.get(region, ()) if chroma(o) >= MIN_CHROMA)

    def near_a_set(c):
        return any(delta_e(c, a) < MIN_DELTA_E for a in avoid)

    kept = {r: given[r] for r in regions if given.get(r)}
    used = {}
    for c in kept.values():
        used[c] = used.get(c, 0) + 1
    out = {}
    for region in regions:
        c = kept.get(region)
        if not c:
            siblings = [c for r, c in {**kept, **out}.items()
                        if set(own.get(r, ())) & set(own.get(region, ()))]
            pool = [c for c in NEUROPILS if fits(c, region)] or NEUROPILS
            c = min(pool, key=lambda c: (
                any(hue_gap(c, s) < MIN_SIBLING_HUE for s in siblings), used.get(c, 0),
                near_a_set(c), NEUROPILS.index(c)))
            used[c] = used.get(c, 0) + 1
        out[region] = c
    return out


def linear_hex(hex_color):
    """An sRGB `#rrggbb` re-encoded so neuVid renders it as that colour.

    neuVid reads a hex colour as `value / 255` straight into a material, i.e. as LINEAR, and
    Blender's display transform then brightens it again, so sRGB colours come out pale.
    Labels read hex the same way, so a label in a cell's `linear_hex` colour matches it.
    """
    return to_hex([_linear(v) for v in rgb(hex_color)])


def mute(hex_color, amount, grey=0.6):
    """`hex_color` blended `amount` of the way toward a mid grey."""
    return to_hex([(1 - amount) * c + amount * grey for c in rgb(hex_color)])


def normalize(color):
    """A colour from a spec as lowercase `#rrggbb`."""
    if not (isinstance(color, str) and len(color) == 7 and color.startswith("#")):
        raise ValueError(f"colour {color!r} is not #rrggbb")
    int(color[1:], 16)
    return color.lower()
