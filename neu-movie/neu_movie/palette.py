"""Colours: the palette sets are drawn from, and the conversions neuVid needs."""

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

#: Neuropils without a colour in the view, in order of first appearance.
NEUROPILS = TAB10


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
