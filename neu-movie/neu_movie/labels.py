"""Text on the movie: beat titles and a legend of the sets on screen, each set's name in its
colour. Drawn onto rendered frames afterwards (`blender/labels.py`), so changing labels
never means rendering again.

Sizes are fractions of the frame height and positions fractions of the frame from its
bottom left, neuVid's label units. Every label fades in and out over the spec's `fade`, the
same time and easing as the cells it names.

- replace: each beat shows its title at the top left and a legend of its sets at the bottom
  left, both timed like the beat's sets, so at a beat change old and new cross-fade in place.
  A set in consecutive beats keeps its label only where it keeps its row.
- accumulate: the legend grows a row per set as it appears and stays to the end; a beat's
  title shows until the next beat has faded in.

Neuropils get a legend of their own at the bottom right, right-aligned, timed like the sets
they come with, each name in the colour its neuropil is drawn in.
"""

import json

from neu_movie import schedule
from neu_movie.palette import linear_hex, mute

WHITE = "#ffffff"


def _label(start, end, text, position, size, color, fade, align="left"):
    out = {"start": start, "end": end, "text": text, "position": position, "size": size,
           "color": color, "fade": fade}
    if align != "left":
        out["align"] = align
    return out


def _row(lab, row, rows, right=False):
    """Row `row` of a `rows`-row legend, read top-down, sitting on the bottom margin; on the
    right, `legend_x` in from the right edge."""
    return [round(1 - lab["legend_x"], 4) if right else lab["legend_x"],
            round(lab["legend_bottom"] + lab["legend_step"] * (rows - 1 - row), 4)]


def _regions(spec, names):
    """The regions shown with the sets `names`, in order, each once."""
    return list(dict.fromkeys(r for n in names for r in spec["sets"][n]["regions"]))


def compose(spec, ids, colours, end, region_colours=None):
    """The labels document: neuVid `label` commands, plus a `fade` and an `align` neuVid's
    own lack. `region_colours` are the neuropils' colours before muting."""
    lab, fade = spec["labels"], spec["timing"]["fade"]
    region_colours = region_colours or {}
    if lab["neuropil_legend"] and spec["neuropils"] and region_colours:
        amount = spec["neuropils"]["mute"]

        def region_label(start, gone, region, row, rows):
            return _label(start, gone, region, _row(lab, row, rows, right=True),
                          lab["legend_size"],
                          linear_hex(mute(region_colours[region], amount)), fade, "right")
    else:
        region_label = None
    header = lab["legend_header"] if lab["legend"] else ""
    beats = spec["beats"]
    out = []
    if spec["mode"] == "replace":
        for i, beat in enumerate(beats):
            start = schedule.beat_start(spec, i)
            gone = schedule.beat_end(spec, i)
            if lab["titles"] and beat["title"]:
                out.append(_label(start, gone, beat["title"], "top", lab["title_size"], WHITE,
                                  fade))
            shown = [n for n in beat["sets"] if n in ids]
            if lab["legend"]:
                rows = len(shown) + bool(header and shown)
                if header and shown:
                    out.append(_label(start, gone, header, _row(lab, 0, rows),
                                      lab["legend_size"], WHITE, fade))
                for row, name in enumerate(shown, start=rows - len(shown)):
                    out.append(_label(start, gone, name, _row(lab, row, rows),
                                      lab["legend_size"], linear_hex(colours[name]), fade))
            if region_label:
                regions = [r for r in _regions(spec, shown) if r in region_colours]
                out += [region_label(start, gone, r, row, len(regions))
                        for row, r in enumerate(regions)]
    else:
        gone = end + fade                      # fade-out falls after the last frame
        if lab["titles"]:
            for i, beat in enumerate(beats):
                if beat["title"]:
                    until = (schedule.beat_start(spec, i + 1) + fade if i + 1 < len(beats)
                             else gone)
                    out.append(_label(schedule.beat_start(spec, i), until, beat["title"],
                                      "top", lab["title_size"], WHITE, fade))
        order = [(i, n) for i, b in enumerate(beats) for n in b["sets"] if n in ids]
        if lab["legend"]:
            rows = len(order) + bool(header)
            if header:
                out.append(_label(schedule.beat_start(spec, 0), gone, header,
                                  _row(lab, 0, rows), lab["legend_size"], WHITE, fade))
            for row, (i, name) in enumerate(order, start=bool(header)):
                out.append(_label(schedule.beat_start(spec, i), gone, name,
                                  _row(lab, row, rows), lab["legend_size"],
                                  linear_hex(colours[name]), fade))
        if region_label:
            first = {}
            for i, name in order:
                for r in _regions(spec, [name]):
                    if r in region_colours:
                        first.setdefault(r, i)
            out += [region_label(schedule.beat_start(spec, i), gone, r, row, len(first))
                    for row, (r, i) in enumerate(first.items())]
    return {"fps": spec["fps"], "animation": commands(out)}


def commands(labels):
    """Labels as neuVid `label` commands in time order. Labels that look identical and
    overlap in time become one, so a title or a set shown through consecutive beats stays
    on instead of fading out and back in."""
    merged = []
    for lab in sorted(labels, key=lambda lab: lab["start"]):
        look = (lab["text"], lab["color"], json.dumps(lab["position"]), lab["size"],
                lab.get("align"))
        prev = next((m for m in reversed(merged)
                     if m[0] == look and lab["start"] <= m[1]["end"] + 1e-9), None)
        if prev:
            prev[1]["end"] = max(prev[1]["end"], lab["end"])
        else:
            merged.append((look, dict(lab)))
    anim, now = [], 0.0
    for _, lab in merged:
        if lab["start"] > now + 1e-9:
            anim.append(["advanceTime", {"by": round(lab["start"] - now, 4)}])
            now = lab["start"]
        args = {"text": lab["text"], "position": lab["position"], "size": lab["size"],
                "color": lab["color"], "duration": round(lab["end"] - lab["start"], 4),
                "fade": lab["fade"]}
        if lab.get("align"):
            args["align"] = lab["align"]
        anim.append(["label", args])
    return anim
