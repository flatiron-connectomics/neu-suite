"""When things happen, in seconds: beat starts, each set's time on screen, the end of the
content. The neuVid input, the labels and the colour check all read it from here."""


def beat_start(spec, i):
    return spec["timing"]["start"] + i * spec["timing"]["every"]


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
    t = spec["timing"]
    if spec["mode"] == "replace":
        return beat_start(spec, i) + t["fade"] + t["hold"] + t["fade"]
    return beat_start(spec, i) + t["fade"]


def content_end(spec):
    return beat_end(spec, len(spec["beats"]) - 1)


def overlapping(a, b, end):
    """Whether two lists of intervals share any moment (None ends at `end`)."""
    for s1, e1 in a:
        for s2, e2 in b:
            if s1 < (e2 if e2 is not None else end) and s2 < (e1 if e1 is not None else end):
                return True
    return False
