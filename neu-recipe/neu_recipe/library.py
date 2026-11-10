"""The volume library: every volume a project uses, under a stable NAME.

Recipes, configs and reports name volumes (``fly-a/seg-v2``), never locations. Data moves
and deliveries get re-cut, so the name is what everything refers to, and the location is
an attribute of it that changes in one place.

A library is a directory of TOML files, one per specimen, ``<specimen>.toml``, with one
table per volume::

    [volume."fly-a/seg-v2"]
    kind = "segmentation"
    location = "s3://my-bucket/fly-a/seg-v2"
    description = "proofread segmentation, second release"
    ignore = [0]

Required: ``kind`` (one of :data:`KINDS`) and ``location``. Common optional fields:
``description``, ``ignore`` (label values that are not bodies), ``delivered_box_zyx``
(level-0 voxels ``[[lo], [hi]]`` where a volume's data actually is, for one whose extent
is larger), ``voxel_size_nm``, ``provenance``, ``status`` (``current`` | ``superseded``)
and ``superseded_by``. Anything else is carried through untouched, and a recipe can use
it as ``{seg.<field>}``.

Checked on every load: a name starts with its file's specimen, names are unique, and
locations are unique after normalizing (a ``precomputed://`` prefix and a trailing slash
are dropped), since two names for one volume is how two analyses silently disagree about
which one they read.

The directory is passed explicitly, or comes from ``$NEU_RECIPE_LIBRARY``. A library
holds data locations, so it usually lives in a private repository beside the recipes.
"""

from __future__ import annotations

import difflib
import os
from pathlib import Path
from typing import Iterable

import tomllib

#: The environment variable naming the library directory when none is passed.
ENV_VAR = "NEU_RECIPE_LIBRARY"

KINDS = {"segmentation", "image", "skeletons", "semantic", "instance", "roi", "points",
         "segment_properties"}

#: What marks a location as a volume (or a precomputed subresource): one is enough.
METADATA_KEYS = ("info", "zarr.json", ".zattrs", ".zgroup", ".zarray")

#: A ``format`` that is a plain directory of files, with nothing to probe.
UNPROBED_FORMATS = {"swc"}


class LibraryError(ValueError):
    pass


def default_dir() -> Path:
    """The library directory named by ``$NEU_RECIPE_LIBRARY``."""
    value = os.environ.get(ENV_VAR)
    if not value:
        raise LibraryError(f"no volume library given: pass its directory, or set "
                           f"${ENV_VAR}")
    return Path(value).expanduser()


def normalize(location: str) -> str:
    loc = location.strip()
    if loc.startswith("precomputed://"):
        loc = loc[len("precomputed://"):]
    return loc.rstrip("/")


def load(directory: Path | str | None = None) -> dict[str, dict]:
    """Every volume, by name. Each entry gains ``name``, ``specimen`` and a normalized
    ``location``."""
    directory = default_dir() if directory is None else Path(directory).expanduser()
    files = sorted(directory.glob("*.toml"))
    if not files:
        raise LibraryError(f"no library files (*.toml) in {directory}")
    volumes: dict[str, dict] = {}
    where: dict[str, str] = {}          # normalized location -> name
    for f in files:
        specimen = f.stem
        with open(f, "rb") as fh:
            table = tomllib.load(fh).get("volume", {})
        for name, v in table.items():
            if not name.startswith(f"{specimen}/"):
                raise LibraryError(f"{f.name}: {name!r} must start with '{specimen}/'")
            if name in volumes:
                raise LibraryError(f"{f.name}: {name!r} is defined twice")
            missing = [k for k in ("kind", "location") if k not in v]
            if missing:
                raise LibraryError(f"{f.name}: {name!r} has no {', '.join(missing)}")
            if v["kind"] not in KINDS:
                raise LibraryError(f"{f.name}: {name!r} kind {v['kind']!r} is not one of "
                                   f"{sorted(KINDS)}")
            loc = normalize(v["location"])
            if loc in where:
                raise LibraryError(f"{f.name}: {name!r} and {where[loc]!r} are the same "
                                   f"location ({loc})")
            where[loc] = name
            volumes[name] = {**v, "name": name, "specimen": specimen, "location": loc}
    return volumes


def get(name: str, volumes: dict[str, dict] | None = None, *,
        directory: Path | str | None = None) -> dict:
    """One volume's entry, from ``volumes`` or else from the library at ``directory``."""
    volumes = load(directory) if volumes is None else volumes
    if name not in volumes:
        near = difflib.get_close_matches(name, list(volumes), n=3)
        hint = f"; did you mean {', '.join(near)}?" if near else ""
        raise LibraryError(f"no volume {name!r} in the library{hint}")
    return volumes[name]


def location(name: str, volumes: dict[str, dict] | None = None, *,
             directory: Path | str | None = None) -> str:
    return get(name, volumes, directory=directory)["location"]


def probe(loc: str) -> str | None:
    """The first metadata object found at a location, or None. Needs neu-vol."""
    try:
        from neu_vol import dvid
        from neu_vol import location as nloc
    except ImportError:
        raise LibraryError("checking locations needs neu-vol: "
                           "pip install 'neu-recipe[check]'") from None

    if dvid.is_url(loc):                     # a server, not a store: ask for its /info
        try:
            dvid.instance_info(dvid.parse_url(loc))
        except Exception:                    # noqa: BLE001 - any failure is a miss
            return None
        return "/info"
    for key in METADATA_KEYS:
        if nloc.exists(loc, key):
            return key
    return None


def listing(volumes: dict[str, dict], *, check: bool = False,
            probe_fn=None) -> Iterable[tuple[str, bool | None]]:
    """``(line, found)`` per volume for the ``library`` subcommand; ``found`` is None when
    nothing was probed. ``probe_fn`` defaults to :func:`probe`."""
    probe_fn = probe_fn or probe
    for n, v in volumes.items():
        flag = ""
        if v.get("status") == "superseded":
            flag = f"  [superseded by {v.get('superseded_by', '?')}]"
        line = f"{n:42s} {v['kind']:13s} {v['location']}{flag}"
        if not check:
            yield line, None
        elif v.get("format") in UNPROBED_FORMATS:
            yield f"--   {line}", None
        else:
            found = probe_fn(v["location"]) is not None
            yield f"{'ok  ' if found else 'MISS'} {line}", found
