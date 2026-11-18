"""A movie spec: one TOML file per animation. Every key is checked, so a typo fails loudly
instead of quietly falling back to a default::

    mode = "replace"             # or "accumulate"
    fps = 30

    [view]                       # the opening camera; `neu-movie camera STATE` prints it
    position = [53304.0, 43142.0, 53359.0]          # what it looks at, nm
    orientation = [-0.126, 0.675, 0.722, 0.084]     # neuroglancer's projectionOrientation
    height = 109974.0            # nm spanned top to bottom at `position`

    [neurons]                    # the segmentation the sets come from
    library = "my-volume/v1"     # a volume library name, or `volume = "<url>"`
    lod = 1                      # mesh level of detail (0 = full)
    query = { without = ["fragment"] }   # defaults for every set's query

    [neuropils]                  # optional: shown with sets (a set's `neuropils`)
    library = "my-volume/v1"
    path = "rois/neuropils"      # the neuropil layer, under the library volume

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
    neuron_legend_header = "Cell types:"

    [[beat]]
    title = "optic lobe"
    sets = [
        { name = "LA", label = "^L[1-9]" },
        { name = "Tm", label = "^Tm", exclude = [82159813] },
    ]

    [[beat]]                     # a beat of one set may give the set's keys directly
    name = "R7p"
    ids = [27332882]
    neuropils = ["LA(L)", "ME(L)"]

**Modes.** `replace`: each beat's sets fade in at its start and out `hold` seconds after
their fade-in ends, unless the next beat names them too — a set named in consecutive beats
stays on. `accumulate`: sets fade in at their beat and stay.

**Sets.** `ids` lists bodies; otherwise the set is a query on the segmentation's
segment_properties (`label` regex, `tags`, `any_tags`, `without`, `limit`), with
`[neurons].query` as defaults. `exclude` drops bodies either way. `color` overrides the
palette. A set named in several beats is ONE set, defined once.

**Repeats.** `[timing] repeat = N` plays the beats N times over (`replace` mode only), each
pass with its own `start` and `tail`, so the seam between passes matches a loop's.

**Sources.** `[neurons]`, `[neuropils]` and `[head]` each name a volume: a `library` name
or a `volume` URL, and a `path` under it (the head's defaults to `rois/head`).

**Camera.** `orbit`: frames the head (`frame = "sets"`, or without a head: every set) and
turns about the screen's vertical from `orbit_at` for `orbit_seconds`, by default until the
last fade ends. With `loop = true` it turns from the first frame to the last at constant
speed instead, so the movie loops without a seam. `follow`: each beat dollies to frame its
sets with the previous `window - 1` beats', then the camera returns to the opening view and
orbits for `orbit_seconds` (default 12). `glide`: opens framed on the first beat and makes
one slow move, from the first beat's start until the last beat has faded in, to frame every
set, then orbits for `orbit_seconds` (default 12). A positive `orbit_degrees` moves the
camera to its right, so the scene turns to the left. `up = [x, y, z]` keeps the
view direction but stands it upright about that vertical, so the orbit turns about it
rather than about the opening view's: for `follow` it levels the return, for `orbit` and
`glide` the whole movie.

**Light.** neuVid's lights are fixed in the world, so `[light] fill` adds one that moves with
the camera, so no side the camera comes round to is black (W/m^2; neuVid's lamps give
about 4 each; 0 for none).

**Labels.** Text drawn over the frames, each part with its own switch: `beat_titles` at the
top, `neuron_legend` (the sets) at the bottom left and `neuropil_legend` at the bottom right,
each name in its colour, under an optional `neuron_legend_header` / `neuropil_legend_header`.
"""

import copy
import os
import tomllib

from neu_movie import palette

MODES = ("replace", "accumulate")
CAMERA_STYLES = ("orbit", "follow", "glide")
FRAMES = ("head", "sets")
QUERY_KEYS = ("label", "tags", "any_tags", "without", "limit")

SECTIONS = {
    "view": {"position": None, "orientation": [0.0, 0.0, 0.0, 1.0], "height": None},
    "neurons": {"library": None, "volume": None, "path": "", "lod": 0, "query": {},
                "workers": 16},
    "neuropils": {"library": None, "volume": None, "path": None, "colors": {}, "alpha": 0.08,
                  "exponent": 7.0, "mute": 0.5},
    "head": {"library": None, "volume": None, "path": "rois/head", "id": None, "lod": 0,
             "alpha": 0.035, "color": "#808080", "exponent": 7.0, "offset": 4.0,
             "intro_at": 1.0, "intro_seconds": 0.0, "lit_color": "#d9d9d9"},
    "meshes": {"smooth": 10},
    "timing": {"start": 0.5, "every": 5.5, "fade": 1.5, "hold": 4.0, "alpha": 1.0,
               "tail": 1.5, "repeat": 1},
    "camera": {"style": "orbit", "scale": 1.0, "orbit_degrees": 360.0, "orbit_at": 1.0,
               "loop": False, "window": 2, "move_seconds": None, "pullback_seconds": 4.0,
               "orbit_seconds": None, "up": None, "frame": "head"},
    "light": {"fill": 2.0},
    "labels": {"beat_titles": True, "neuron_legend": True, "neuron_legend_header": "",
               "neuropil_legend": True, "neuropil_legend_header": "", "title_size": 0.06,
               "legend_size": 0.045, "legend_x": 0.04, "legend_bottom": 0.06,
               "legend_step": 0.065},
}
TOP = {"name": None, "fps": 30, "mode": "replace", "beat": None, **SECTIONS}
SET_KEYS = ("name", "ids", "exclude", "neuropils", "color", *QUERY_KEYS)


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
    s = {"name": str(raw["name"]), "neuropils": list(raw.get("neuropils", [])),
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


#: Earlier names of `[labels]` keys, which older builds' copies of their spec still use.
OLD_LABEL_KEYS = {"titles": "beat_titles", "legend": "neuron_legend",
                  "legend_header": "neuron_legend_header",
                  "neuropil_header": "neuropil_legend_header"}


def _new_names(raw):
    """`raw` with the earlier key names (`OLD_LABEL_KEYS`, a set's `regions`) renamed."""
    raw = copy.deepcopy(raw)
    lab = raw.get("labels", {})
    for old, new in OLD_LABEL_KEYS.items():
        if old in lab and new not in lab:
            lab[new] = lab.pop(old)
    for b in raw.get("beat") or []:
        for m in b.get("sets", [b]):
            if "regions" in m and "neuropils" not in m:
                m["neuropils"] = m.pop("regions")
    return raw


def load(path, *, old_names=False):
    """The spec at `path`, every default filled in. Paths in it become absolute.
    `old_names` reads the earlier key names too, for a build's own copy of its spec."""
    path = os.path.abspath(path)
    with open(path, "rb") as f:
        try:
            raw = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise SpecError(f"{path}: {e}") from None
    return parse(_new_names(raw) if old_names else raw, path)


def _numbers(value, n, what):
    """`value` as `n` floats, or a SpecError naming `what`."""
    if (not isinstance(value, (list, tuple)) or len(value) != n
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value)):
        raise SpecError(f"{what} must be {n} numbers")
    return [float(v) for v in value]


def _source(section, where):
    if bool(section["library"]) == bool(section["volume"]):
        raise SpecError(f"{where}: give one of `library` or `volume`")


def parse(raw, path):
    where = os.path.basename(path)
    if not isinstance(raw.get("view", {}), dict):
        raise SpecError(f"{where}: `view` is a [view] table of the opening camera (position, "
                        f"orientation, height); `neu-movie camera {raw['view']}` prints one "
                        f"from a neuroglancer state")
    spec = _section(raw, TOP, where)
    for name, defaults in SECTIONS.items():
        spec[name] = _section(raw.get(name, {}), defaults, f"{where} [{name}]")
    spec["path"] = path
    spec["name"] = spec["name"] or os.path.splitext(os.path.basename(path))[0]
    view = spec["view"]
    if "view" not in raw or view["position"] is None or view["height"] is None:
        raise SpecError(f"{where}: [view] needs `position` and `height` (and `orientation`); "
                        f"`neu-movie camera STATE` prints them from a neuroglancer state")
    view["position"] = _numbers(view["position"], 3, f"{where} [view] `position`")
    view["orientation"] = _numbers(view["orientation"], 4, f"{where} [view] `orientation`")
    h = view["height"]
    if isinstance(h, bool) or not isinstance(h, (int, float)) or h <= 0:
        raise SpecError(f"{where} [view]: `height` must be a number above 0")
    view["height"] = float(h)
    if spec["mode"] not in MODES:
        raise SpecError(f"{where}: mode {spec['mode']!r} is not one of {MODES}")
    cam = spec["camera"]
    if cam["style"] not in CAMERA_STYLES:
        raise SpecError(f"{where}: camera style {cam['style']!r} is not one of "
                        f"{CAMERA_STYLES}")
    if cam["loop"] and cam["style"] != "orbit":
        raise SpecError(f"{where} [camera]: `loop` is for the orbit camera")
    if cam["loop"] and cam["orbit_seconds"] is not None:
        raise SpecError(f"{where} [camera]: a looping orbit lasts the whole movie, so it takes "
                        f"no `orbit_seconds`")
    if cam["orbit_seconds"] is None and cam["style"] != "orbit":
        cam["orbit_seconds"] = 12.0
    o = cam["orbit_seconds"]
    if o is not None and (isinstance(o, bool) or not isinstance(o, (int, float)) or o <= 0):
        raise SpecError(f"{where} [camera]: `orbit_seconds` must be a number above 0")
    if cam["frame"] not in FRAMES:
        raise SpecError(f"{where} [camera]: frame {cam['frame']!r} is not one of {FRAMES}")
    if cam["up"] is not None:
        cam["up"] = _numbers(cam["up"], 3, f"{where} [camera] `up`")
    fill = spec["light"]["fill"]
    if isinstance(fill, bool) or not isinstance(fill, (int, float)) or fill < 0:
        raise SpecError(f"{where} [light]: `fill` must be a number, 0 or more")
    _source(spec["neurons"], f"{where} [neurons]")
    head = spec["head"]
    if "head" in raw:
        _source(head, f"{where} [head]")
        if head["id"] is None:
            raise SpecError(f"{where} [head]: `id` is required")
    else:
        spec["head"] = None
    if "neuropils" in raw:
        nps = spec["neuropils"]
        _source(nps, f"{where} [neuropils]")
        if nps["library"] and not nps["path"]:
            raise SpecError(f"{where} [neuropils]: with `library`, give `path`, the "
                            f"neuropil layer under it (e.g. \"rois/neuropils\")")
        try:
            nps["colors"] = {str(r): palette.normalize(c) for r, c in nps["colors"].items()}
        except ValueError as e:
            raise SpecError(f"{where} [neuropils] colors: {e}") from None
    else:
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
    repeat = spec["timing"]["repeat"]
    if not isinstance(repeat, int) or repeat < 1:
        raise SpecError(f"{where} [timing]: `repeat` must be a whole number, 1 or more")
    if repeat > 1 and spec["mode"] != "replace":
        raise SpecError(f"{where} [timing]: `repeat` needs mode \"replace\"; in accumulate "
                        f"mode the sets of the first pass never leave")
    spec["beats"], spec["sets"] = [dict(b) for _ in range(repeat) for b in beats], sets
    del spec["beat"]
    shown = {r for s in sets.values() for r in s["neuropils"]}
    if shown and not spec["neuropils"]:
        raise SpecError(f"{where}: sets name `neuropils` but there is no [neuropils] section")
    stray = set(spec["neuropils"]["colors"]) - shown if spec["neuropils"] else set()
    if stray:
        raise SpecError(f"{where} [neuropils] colors: no set shows {', '.join(sorted(stray))}")

    if spec["mode"] == "accumulate":
        again = [n for n in sets if sum(n in b["sets"] for b in beats) > 1]
        if again:
            raise SpecError(f"{where}: in accumulate mode a set stays once shown, so naming "
                            f"it again does nothing: {', '.join(again)}")
    return spec
