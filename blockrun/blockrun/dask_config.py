"""Dask cluster config: bundled templates, layered overrides, key validation.

- **Bundled templates:** ``--config dask-local`` resolves through
  ``importlib.resources``, so it works from any directory, including for an installed
  command.
- **Layering:** ``--config`` is repeatable and deep-merged left to right, so a site config
  holds only the keys that differ from a template::

      --config dask-slurm-example --config configs/my-site.yaml

- **Validation:** an unknown key would otherwise merge silently and do nothing, so each
  file is checked against dask's own config tree (:func:`unknown_keys`).
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Mapping, Sequence

import yaml

SUFFIX = ".yaml"


class ConfigKeyError(ValueError):
    """An unrecognised key in a dask config — almost always a typo."""


# --------------------------------------------------------------------------- #
# bundled templates
# --------------------------------------------------------------------------- #
def _config_dir():
    from importlib.resources import files

    return files("blockrun") / "configs"


def bundled_configs() -> list[str]:
    """Names of the dask configs shipped with this package."""
    try:
        return sorted(p.name[: -len(SUFFIX)] for p in _config_dir().iterdir()
                      if p.name.endswith(SUFFIX))
    except (FileNotFoundError, ModuleNotFoundError, NotADirectoryError):
        return []


def resolve_config(value: str) -> str:
    """One ``--config`` value -> a filesystem path.

    An existing path wins over a bundled name, so a local file can shadow a template
    and every previously-working invocation keeps working.
    """
    if os.path.exists(value):
        return value
    stem = value[: -len(SUFFIX)] if value.endswith(SUFFIX) else value
    candidate = _config_dir() / f"{stem}{SUFFIX}"
    if candidate.is_file():
        return str(candidate)
    names = bundled_configs()
    raise FileNotFoundError(
        f"--config {value!r}: no such file, and not one of the bundled configs "
        f"({', '.join(names) if names else 'none found'}). Pass a path to your own "
        f"YAML, or one of those names.")


# --------------------------------------------------------------------------- #
# merging
# --------------------------------------------------------------------------- #
def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict:
    """``overlay`` over ``base``: recurse into dicts, REPLACE everything else.

    Lists are replaced, not concatenated — ``job-extra-directives`` and
    ``worker-extra-args`` are complete argument vectors, and appending to them
    produces a command line nobody wrote. Scalars replace so that ``spill: false``
    can be overridden back to a number.
    """
    out = dict(base)
    for key, value in overlay.items():
        if (key in out and isinstance(out[key], Mapping)
                and isinstance(value, Mapping)):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def _reference_tree() -> dict:
    """dask's own known-key tree (dask and dask-jobqueue defaults). Not the bundled
    templates, which omit many valid keys, such as ``job-extra-directives``."""
    import dask.config

    try:
        import dask_jobqueue                                  # noqa: F401
    except ImportError:
        # Without it the `jobqueue` subtree is unpopulated, so every key under it
        # would look unknown. Drop the subtree instead of failing everything.
        return {k: v for k, v in dask.config.config.items() if k != "jobqueue"}
    return dask.config.config


def unknown_keys(doc: Mapping[str, Any], reference: Mapping[str, Any] | None = None,
                 _prefix: tuple[str, ...] = ()) -> list[str]:
    """Dotted paths in ``doc`` that dask's config tree does not define.

    Names are canonicalised as dask does (``log_directory`` is ``log-directory``).
    Checking stops below an empty-dict default, which dask uses for free-form mappings
    such as ``distributed.worker.resources``, and below scalar or list defaults.
    """
    from dask.config import canonical_name

    ref = _reference_tree() if reference is None else reference
    out: list[str] = []
    for key, value in (doc or {}).items():
        name = canonical_name(key, ref) if isinstance(ref, Mapping) else key
        if not isinstance(ref, Mapping) or name not in ref:
            out.append(".".join(_prefix + (str(key),)))
            continue
        child = ref[name]
        if isinstance(value, Mapping) and isinstance(child, Mapping) and child:
            out += unknown_keys(value, child, _prefix + (name,))
    return out


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_config(sources: str | Sequence[str], *,
                validate: bool = True) -> tuple[dict, list[str]]:
    """Resolve, load, validate and merge one or more ``--config`` values.

    Returns ``(merged, resolved_paths)``. Each file is validated on its own before
    merging, so an error names the file that introduced the key.
    """
    if isinstance(sources, str):
        sources = [sources]
    if not sources:
        raise ValueError("no dask config given")

    merged: dict = {}
    paths: list[str] = []
    for source in sources:
        path = resolve_config(source)
        with open(path) as f:
            doc = yaml.safe_load(f) or {}
        if not isinstance(doc, Mapping):
            raise ConfigKeyError(f"{path}: top level is {type(doc).__name__}, "
                                 f"expected a mapping")
        if validate:
            bad = unknown_keys(doc)
            if bad:
                raise ConfigKeyError(
                    f"{path}: unrecognised dask config key(s): "
                    f"{', '.join(sorted(bad))}. These would merge silently and do "
                    f"nothing. Check the spelling against dask / dask-jobqueue "
                    f"defaults (note dask accepts '-' and '_' interchangeably).")
        merged = deep_merge(merged, doc)
        paths.append(path)
    return merged, paths


def write_effective_config(dest: str, merged: Mapping[str, Any],
                           sources: Iterable[str]) -> str:
    """Record the config dask actually received. Returns the path written.

    ``dest`` is a directory (the file is named ``dask-config.effective.yaml``) or a full
    path. Kept with a run, it records what layered overrides produced.
    """
    path = (os.path.join(dest, "dask-config.effective.yaml")
            if os.path.isdir(dest) else dest)
    header = "".join(f"#   {s}\n" for s in sources)
    body = yaml.safe_dump(dict(merged), sort_keys=False, default_flow_style=False)
    with open(path, "w") as f:
        f.write(f"# Effective dask config — merged, in order, from:\n{header}"
                f"# Generated by blockrun; edit the sources, not this file.\n{body}")
    return path
