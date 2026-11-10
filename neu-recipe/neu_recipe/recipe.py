"""Recipes: parse, resolve and select steps.

A recipe is a TOML file with up to four tables::

    [recipe]                  # optional; paths relative to this file
    cwd = ".."                # where steps run (default: this file's directory)
    library = "../library"    # the volume library (default: $NEU_RECIPE_LIBRARY)

    [volumes]                 # volume-library names
    seg = "fly-a/seg-v2"

    [vars]                    # strings, resolved in the order written
    work_dir = "/path/to/work/{seg.name}"
    bodies = "{work_dir}/selected_bodies.csv"

    [[step]]                  # run in file order, one at a time
    name = "mesh"
    run = '''neu-morpho run --src {seg} --dst {seg} --work-dir {work_dir}
             --allowlist {bodies}'''
    progress = "neu-morpho progress {work_dir}"      # optional, for `status`

``{seg}`` is a volume's location, ``{seg.name}`` its library name and ``{seg.<field>}``
any other field of its entry. A var may use volumes and the vars above it. Any other
``{name}`` is an error, so a typo fails before anything runs; write ``{{`` and ``}}`` for
literal braces, e.g. ``--out 'bodies_{{uuid:8}}.csv'`` for a placeholder a command
expands itself. A command is split as a shell would split it (quotes respected, newlines
are spaces) and runs without a shell, from ``cwd``, so relative paths resolve the same
however the runner is started.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Callable

import tomllib

from . import library as lib

TABLES = {"recipe", "volumes", "vars", "step"}
RECIPE_KEYS = {"cwd", "library"}
STEP_KEYS = {"name", "run", "progress"}

_REF = re.compile(r"\{([A-Za-z_][\w-]*)(?:\.([A-Za-z_][\w-]*))?\}")
#: A DVID branch ref (``uuid:branch``) means a different node tomorrow.
_DVID_BRANCH = re.compile(r"dvid://[^/\s]+/[^/\s]*:[^/\s]+")


class RecipeError(ValueError):
    pass


def _substitute(text: str, values: dict, volumes: dict, where: str) -> str:
    """``text`` with every ``{name}`` / ``{volume.field}`` replaced; ``{{ }}`` are literal."""
    def one(m):
        name, field = m.group(1), m.group(2)
        if name in volumes:
            entry = volumes[name]
            key = field or "location"
            if key not in entry:
                raise RecipeError(f"{where}: volume {name} ({entry['name']}) has no field "
                                  f"{key!r}; it has {', '.join(sorted(entry))}")
            value = entry[key]
            if not isinstance(value, (str, int, float)):
                raise RecipeError(f"{where}: {{{name}.{key}}} is a {type(value).__name__}, "
                                  f"not a string")
            return str(value)
        if field is not None:
            raise RecipeError(f"{where}: {{{name}.{field}}} — {name} is not a volume")
        if name not in values:
            raise RecipeError(f"{where}: {{{name}}} is not defined (above this point)")
        return values[name]

    parts = re.split(r"(\{\{|\}\})", text)
    return "".join({"{{": "{", "}}": "}"}.get(p) or _REF.sub(one, p) for p in parts)


def _library_lookup(directory: Path | None) -> Callable[[str], dict]:
    cache: dict = {}

    def entry(name: str) -> dict:
        try:
            if "volumes" not in cache:
                cache["volumes"] = lib.load(directory)
            return lib.get(name, cache["volumes"])
        except lib.LibraryError as e:
            raise RecipeError(f"volume {name!r}: {e}") from None
    return entry


def load(path: Path | str, *, library: Path | str | None = None,
         entry: Callable[[str], dict] | None = None) -> dict:
    """Parse and resolve a recipe. Raises :class:`RecipeError` for anything wrong in it.

    ``library`` overrides the recipe's own ``[recipe] library``; ``entry`` replaces the
    library lookup altogether (name -> entry).
    """
    path = Path(path)
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    unknown = set(raw) - TABLES
    if unknown:
        raise RecipeError(f"unknown table(s) {sorted(unknown)}; expected "
                          f"{', '.join(sorted(TABLES))}")
    settings = raw.get("recipe") or {}
    extra = set(settings) - RECIPE_KEYS
    if extra:
        raise RecipeError(f"[recipe]: unknown key(s) {sorted(extra)}; it takes "
                          f"{', '.join(sorted(RECIPE_KEYS))}")
    here = path.resolve().parent
    cwd =(here / Path(settings["cwd"]).expanduser()).resolve() if "cwd" in settings else here
    if library is not None:
        library_dir = Path(library).expanduser()
    elif "library" in settings:
        library_dir = (here / Path(settings["library"]).expanduser()).resolve()
    else:
        library_dir = None                  # $NEU_RECIPE_LIBRARY, if a volume needs it
    entry = entry or _library_lookup(library_dir)

    volumes = {k: entry(v) for k, v in (raw.get("volumes") or {}).items()}
    values: dict[str, str] = {}
    for key, value in (raw.get("vars") or {}).items():
        if key in volumes:
            raise RecipeError(f"var {key!r} shadows a volume of the same name")
        if not isinstance(value, str):
            raise RecipeError(f"var {key!r} must be a string, got {type(value).__name__}")
        values[key] = _substitute(value, values, volumes, f"var {key}")

    steps, seen = [], set()
    for i, step in enumerate(raw.get("step") or [], 1):
        extra = set(step) - STEP_KEYS
        if extra:
            raise RecipeError(f"step {i}: unknown key(s) {sorted(extra)}; a step has "
                              f"name, run and optionally progress, and steps run in "
                              f"file order")
        name = step.get("name")
        if not name or "run" not in step:
            raise RecipeError(f"step {i}: needs both name and run")
        if name in seen:
            raise RecipeError(f"step name {name!r} is used twice")
        seen.add(name)
        command = _substitute(step["run"], values, volumes, f"step {name}")
        progress = step.get("progress")
        if progress is not None:
            progress = shlex.split(_substitute(progress, values, volumes,
                                               f"step {name} progress"))
        steps.append({"index": i, "name": name, "argv": shlex.split(command),
                      "progress": progress})
    if not steps:
        raise RecipeError("no [[step]] entries")
    return {"volumes": volumes, "vars": values, "steps": steps, "cwd": cwd,
            "library": library_dir}


def select(steps: list[dict], *, start: str | None = None, only: str | None = None):
    names = [s["name"] for s in steps]
    for wanted in (start, only):
        if wanted is not None and wanted not in names:
            raise RecipeError(f"no step {wanted!r}; steps are {', '.join(names)}")
    if only is not None:
        return [s for s in steps if s["name"] == only]
    if start is not None:
        return steps[names.index(start):]
    return steps


def warnings(steps: list[dict]) -> list[str]:
    """What is legal but likely to make the run unreproducible."""
    out = []
    for s in steps:
        for arg in s["argv"]:
            if _DVID_BRANCH.search(arg):
                out.append(f"step {s['name']}: {arg} names a DVID branch, whose node moves; "
                           f"pin a node uuid")
    return out
