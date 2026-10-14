"""Placeholders in a destination path, resolved from the source.

``--dst /path/to/seg_{uuid:8}`` becomes ``/path/to/seg_d38898ac``: an export named after
the DVID node it came from, since a branch ref means a different node tomorrow.

Expanded once, in the caller, before anything derives a target, manifest name or resume
check from the path; idempotent. By regex, never ``str.format``, since a real path may
contain braces of its own.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

#: ``{name}`` or ``{name:spec}``. A narrow name leaves unrelated braces alone; a loose
#: spec means a malformed one like ``{uuid:-2}`` is reported, not passed through.
_FIELD = re.compile(r"\{([a-z_]+)(?::([^}]*))?\}")

#: Default length for ``{uuid}``, as short uuids are written in practice.
_UUID_DEFAULT = 8

#: What may appear in an expanded value; anything else becomes ``-``.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def has_placeholder(dst: str) -> bool:
    return bool(_FIELD.search(str(dst)))


def _branch_of(ref: str) -> str:
    """The branch part of a ``repo:branch`` ref, or the whole ref when there is none.

    From the ref the caller typed, which costs no request.
    """
    return ref.split(":", 1)[1] if ":" in ref else ref


def _values(src_spec: Mapping[str, Any]) -> dict[str, str]:
    """The substitutable values for this source, or ``{}`` if it offers none."""
    if src_spec.get("backend") != "dvid":
        return {}
    from ..dvid import resolve_node

    # Cached, so this is free when the source metadata has already been read.
    node = resolve_node(src_spec)
    ref = str(src_spec.get("requested_ref") or src_spec.get("uuid") or "")
    return {
        "uuid": str(node["uuid"]),
        "instance": str(src_spec.get("instance", "")),
        "branch": _branch_of(ref),
    }


def _apply_spec(name: str, value: str, spec: str | None) -> str:
    if spec is None:
        return value[:_UUID_DEFAULT] if name == "uuid" else value
    if spec == "full":
        return value
    if spec.isdigit() and int(spec) > 0:
        return value[:int(spec)]
    raise ValueError(
        f"bad placeholder {{{name}:{spec}}} in --dst: the part after ':' must be a "
        f"positive number of characters, or 'full'")


def expand(dst: str, src_spec: Mapping[str, Any]) -> str:
    """Resolve ``{...}`` placeholders in ``dst`` against ``src_spec``.

    Returns ``dst`` unchanged when it holds no placeholder, without resolving anything —
    so an ordinary destination costs nothing.
    """
    dst = str(dst)
    if not has_placeholder(dst):
        return dst

    values = _values(src_spec)
    if not values:
        raise ValueError(
            f"--dst {dst!r} contains a placeholder, but the source "
            f"({src_spec.get('backend')!r}) supplies none. Placeholders name an export "
            f"after the version it came from, which only DVID sources have.")

    def sub(m: re.Match) -> str:
        name, spec = m.group(1), m.group(2)
        if name not in values:
            raise ValueError(
                f"unknown placeholder {{{name}}} in --dst; this source offers "
                f"{', '.join('{' + k + '}' for k in sorted(values))}")
        value = _apply_spec(name, values[name], spec)
        if not value:
            raise ValueError(
                f"placeholder {{{name}}} in --dst resolved to an empty string, which "
                f"would leave a nameless path component")
        return _UNSAFE.sub("-", value)

    return _FIELD.sub(sub, dst)
