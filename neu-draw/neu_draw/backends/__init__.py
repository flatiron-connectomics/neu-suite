"""Renderer backends. Nothing here is imported by ``neu_draw`` itself.

A backend turns a :class:`~neu_draw.scene.Scene` into something on screen. Importing
one needs a canvas backend and a GPU adapter, which a cluster worker or CI lacks, so it is
imported on demand::

    from neu_draw.backends import get_backend
    view = get_backend().show(scene)

pygfx is the only backend.
"""

from __future__ import annotations

from typing import Any

#: Backends by name.
KNOWN = ("pygfx",)


def get_backend(name: str = "pygfx") -> Any:
    """Import and return a backend module, with a useful error when it is missing."""
    if name not in KNOWN:
        raise ValueError(f"unknown backend {name!r}; known: {', '.join(KNOWN)}")
    try:
        from . import pygfx as backend
    except ImportError as exc:
        raise ImportError(
            f"the {name} backend needs the render extra: "
            f"pip install 'neu-draw[render]'  ({exc})") from exc
    return backend
