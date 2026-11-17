"""When things happen, in seconds: beat starts, each set's time on screen, the end of the
content. The neuVid input, the labels and the colour check all read it from here."""


def pass_beats(spec):
    """Beats in one pass of the sequence (`spec["beats"]` holds `repeat` passes)."""
    return len(spec["beats"]) // spec["timing"]["repeat"]


def pass_seconds(spec):
    """One pass, lead-in and tail included: a repeated pass starts where the movie's own
    start would, so the seam between passes is the seam of a loop."""
    t = spec["timing"]
    return _offset(spec, pass_beats(spec) - 1) + _span(spec) + t["tail"]


def _offset(spec, j):
    return spec["timing"]["start"] + j * spec["timing"]["every"]


def _span(spec):
    t = spec["timing"]
    return t["fade"] + t["hold"] + t["fade"] if spec["mode"] == "replace" else t["fade"]


def beat_start(spec, i):
    n = pass_beats(spec)
    return (i // n) * pass_seconds(spec) + _offset(spec, i % n)


def runs(spec):
    """``{set: [(first beat, last beat), ...]}``, the CONTIGUOUS runs of beats naming it."""
    out = {}
    for i, beat in enumerate(spec["beats"]):
        for name in beat["sets"]:
            if name in out and out[name][-1][1] == i - 1:
                out[name][-1] = (out[name][-1][0], i)
            else:
                out.setdefault(name, []).append((i, i))
    return out


def intervals(spec):
    """``{set: [(fade-in start, fade-out start or None), ...]}``; None is "stays to the end"."""
    t = spec["timing"]
    out = {}
    for name, rs in runs(spec).items():
        for first, last in rs:
            start = beat_start(spec, first)
            if spec["mode"] == "replace":
                out.setdefault(name, []).append(
                    (start, beat_start(spec, last) + t["fade"] + t["hold"]))
            else:
                out.setdefault(name, []).append((start, None))
    return out


def beat_end(spec, i):
    """When beat `i`'s own sets are gone (replace) or fully in (accumulate)."""
    return beat_start(spec, i) + _span(spec)


def content_end(spec):
    return beat_end(spec, len(spec["beats"]) - 1)


def overlapping(a, b, end):
    """Whether two lists of intervals share any moment (None ends at `end`)."""
    for s1, e1 in a:
        for s2, e2 in b:
            if s1 < (e2 if e2 is not None else end) and s2 < (e1 if e1 is not None else end):
                return True
    return False
