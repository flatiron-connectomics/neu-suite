"""A movie spec: one TOML file per animation. Every key is checked, so a typo fails loudly
instead of quietly falling back to a default.

    view = "frontal_view.json"   # the opening camera, and the layers volumes are read from
    mode = "replace"             # or "accumulate"
    fps = 30

    [neurons]                    # the segmentation the sets come from
    layer = "segmentation"       # its layer in the view
    lod = 1                      # mesh level of detail (0 = full)
    query = { without = ["fragment"] }   # defaults for every set's query

    [neuropils]                  # optional: regions shown with sets (a set's `regions`)
    layer = "neuropils"

    [head]                       # optional: the animal's outline, as a faint shell
    library = "my-volume/v1"     # a volume library name, or `volume = "<url>"`
    id = 95

    [timing]
    start = 1.5                  # when the first beat begins
    every = 5.5                  # seconds between beats
    fade = 1.5

    [camera]
    style = "orbit"              # or "follow"

    [labels]
    legend_header = "Cell types:"

    [[beat]]
    title = "optic lobe"
    sets = [
        { name = "LA", label = "^L[1-9]" },
        { name = "Tm", label = "^Tm", exclude = [82159813] },
    ]

    [[beat]]                     # a beat of one set may give the set's keys directly
    name = "R7p"
    ids = [27332882]
    regions = ["LA(L)", "ME(L)"]

**Modes.** `replace`: each beat's sets fade in at its start and out `hold` seconds after
their fade-in ends, unless the next beat names them too — a set named in consecutive beats
stays on. `accumulate`: sets fade in at their beat and stay.

**Sets.** `ids` lists bodies; otherwise the set is a query on the segmentation's
segment_properties (`label` regex, `tags`, `any_tags`, `without`, `limit`), with
`[neurons].query` as defaults. `exclude` drops bodies either way. `color` overrides the
palette. A set named in several beats is ONE set, defined once.

**Camera.** `orbit`: frames the head (or every set) and turns about the screen's vertical
from `orbit_at` until the last fade ends. `follow`: each beat dollies to frame its sets with
the previous `window - 1` beats', then the camera returns to the opening view and orbits for
`orbit_seconds`.
"""

import copy
import os
import tomllib

from neu_movie import palette

MODES = ("replace", "accumulate")
CAMERA_STYLES = ("orbit", "follow")
QUERY_KEYS = ("label", "tags", "any_tags", "without", "limit")

SECTIONS = {
    "neurons": {"layer": None, "lod": 0, "query": {}, "workers": 16},
    "neuropils": {"layer": None, "alpha": 0.08, "exponent": 7.0, "mute": 0.5},
    "head": {"library": None, "volume": None, "path": "rois/head", "id": None, "lod": 0,
             "alpha": 0.035, "color": "#808080", "exponent": 7.0, "offset": 4.0,
             "intro_at": 1.0, "intro_seconds": 0.0, "lit_color": "#d9d9d9"},
    "meshes": {"smooth": 10},
    "timing": {"start": 0.5, "every": 5.5, "fade": 1.5, "hold": 4.0, "alpha": 1.0,
               "tail": 1.5},
    "camera": {"style": "orbit", "scale": 1.0, "orbit_degrees": 360.0, "orbit_at": 1.0,
               "window": 2, "move_seconds": None, "pullback_seconds": 4.0,
               "orbit_seconds": 12.0},
    "labels": {"titles": True, "legend": True, "legend_header": "", "title_size": 0.06,
               "legend_size": 0.045, "legend_x": 0.04, "legend_bottom": 0.06,
               "legend_step": 0.065},
}
TOP = {"view": None, "name": None, "fps": 30, "mode": "replace", "beat": None, **SECTIONS}
SET_KEYS = ("name", "ids", "exclude", "regions", "color", *QUERY_KEYS)


class SpecError(ValueError):
    pass


def _section(raw, defaults, where):
    unknown = set(raw) - set(defaults)
    if unknown:
        raise SpecError(f"{where}: unknown key(s) {', '.join(sorted(unknown))}; "
                        f"known: {', '.join(defaults)}")
    out = copy.deepcopy(defaults)
    out.update(raw)
    return out


def _set(raw, where, query_defaults):
    unknown = set(raw) - set(SET_KEYS)
    if unknown:
        raise SpecError(f"{where}: unknown key(s) {', '.join(sorted(unknown))}; "
                        f"a set takes {', '.join(SET_KEYS)}")
    if not raw.get("name"):
        raise SpecError(f"{where}: a set needs a `name`")
    s = {"name": str(raw["name"]), "regions": list(raw.get("regions", [])),
         "exclude": [str(b) for b in raw.get("exclude", [])], "color": None}
    if raw.get("color") is not None:
        try:
            s["color"] = palette.normalize(raw["color"])
        except ValueError as e:
            raise SpecError(f"{where}: {e}") from None
    if "ids" in raw:
        if any(k in raw for k in QUERY_KEYS):
            raise SpecError(f"{where}: give `ids` or a query, not both")
        s["ids"] = [str(b) for b in raw["ids"]]
        s["query"] = None
    else:
        s["ids"] = None
        s["query"] = {**query_defaults, **{k: raw[k] for k in QUERY_KEYS if k in raw}}
        if not any(k in s["query"] for k in ("label", "tags", "any_tags")):
            raise SpecError(f"{where}: set {s['name']!r} has neither `ids` nor a query "
                            f"(`label`, `tags` or `any_tags`)")
    return s


def load(path):
    """The spec at `path`, every default filled in. Paths in it become absolute."""
    path = os.path.abspath(path)
    with open(path, "rb") as f:
        try:
            raw = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise SpecError(f"{path}: {e}") from None
    return parse(raw, path)


def parse(raw, path):
    where = os.path.basename(path)
    spec = _section(raw, TOP, where)
    for name, defaults in SECTIONS.items():
        spec[name] = _section(raw.get(name, {}), defaults, f"{where} [{name}]")
    spec["path"] = path
    spec["name"] = spec["name"] or os.path.splitext(os.path.basename(path))[0]
    if not spec["view"]:
        raise SpecError(f"{where}: `view` (a neuroglancer state file) is required")
    spec["view"] = os.path.join(os.path.dirname(path), os.path.expanduser(spec["view"]))
    if spec["mode"] not in MODES:
        raise SpecError(f"{where}: mode {spec['mode']!r} is not one of {MODES}")
    if spec["camera"]["style"] not in CAMERA_STYLES:
        raise SpecError(f"{where}: camera style {spec['camera']['style']!r} is not one of "
                        f"{CAMERA_STYLES}")
    if not spec["neurons"]["layer"]:
        raise SpecError(f"{where} [neurons]: `layer` is required")
    head = spec["head"]
    if "head" in raw:
        if bool(head["library"]) == bool(head["volume"]):
            raise SpecError(f"{where} [head]: give one of `library` or `volume`")
        if head["id"] is None:
            raise SpecError(f"{where} [head]: `id` is required")
    else:
        spec["head"] = None
    if "neuropils" not in raw:
        spec["neuropils"] = None

    query_defaults = spec["neurons"]["query"]
    unknown = set(query_defaults) - set(QUERY_KEYS)
    if unknown:
        raise SpecError(f"{where} [neurons] query: unknown key(s) {', '.join(sorted(unknown))}")

    beats, sets = [], {}
    for i, b in enumerate(raw.get("beat") or [], 1):
        bwhere = f"{where} beat {i}"
        if "sets" in b:
            extra = set(b) - {"title", "sets"}
            if extra:
                raise SpecError(f"{bwhere}: with `sets`, a beat takes only `title` "
                                f"(found {', '.join(sorted(extra))})")
            members = b["sets"]
        else:
            members = [{k: v for k, v in b.items() if k != "title"}]
        names = []
        for j, m in enumerate(members, 1):
            name = str(m.get("name", ""))
            if set(m) == {"name"} and name in sets:     # named again: the same set
                pass
            else:
                s = _set(m, f"{bwhere} set {j}", query_defaults)
                if name in sets and s != sets[name]:
                    raise SpecError(f"{bwhere}: set {name!r} is defined differently in an "
                                    f"earlier beat; define it once and name it after")
                sets[name] = s
            if name in names:
                raise SpecError(f"{bwhere}: set {name!r} is listed twice")
            names.append(name)
        beats.append({"title": str(b.get("title", "")), "sets": names})
    if not beats:
        raise SpecError(f"{where}: no `[[beat]]`")
    spec["beats"], spec["sets"] = beats, sets
    del spec["beat"]

    if spec["mode"] == "accumulate":
        again = [n for n in sets if sum(n in b["sets"] for b in beats) > 1]
        if again:
            raise SpecError(f"{where}: in accumulate mode a set stays once shown, so naming "
                            f"it again does nothing: {', '.join(again)}")
    return spec
