"""Renderer backends. Nothing here is imported by ``neu_draw`` itself.

A backend turns a :class:`~neu_draw.scene.Scene` — pure data — into something on screen.
Importing one needs a canvas backend and a GPU adapter, neither of which exists on a
cluster worker or in CI, so the import stays behind this call rather than at package
level::

    from neu_draw.backends import get_backend
    view = get_backend().show(scene)

There is one backend, pygfx, which supplies every primitive in use directly. The seam
exists to keep the renderer import lazy, and so that a second backend could implement
the same interface.
"""

from __future__ import annotations

from typing import Any

#: Backends by name. A second entry is what makes the seam real rather than aspirational.
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
