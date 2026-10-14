"""Animations: a :class:`Timeline` of tweens over a viewer state, rendered frame by frame.

:mod:`~neu_glance.movie.animate` decides what each frame is: plain dicts, stdlib only, no
neuroglancer. :mod:`~neu_glance.movie.rendering` gets a real viewer to draw each frame
(the ``serve`` extra), headless or in an attended browser, and prints the ffmpeg command
that encodes them::

    from neu_glance.movie import Timeline, ffmpeg_command, record

    tl = Timeline(state, fps=30).orbit(360, seconds=12)
    record(tl, "frames/")
    print(ffmpeg_command("frames/", fps=30))
"""

from __future__ import annotations

#: Name -> module. Resolved on first use, so importing ``movie.animate`` stays stdlib-only.
_EXPORTS = {
    "Timeline": "animate", "Tween": "animate", "EASINGS": "animate",
    "AnimateProblem": "animate",
    "record": "rendering", "render_state": "rendering", "ffmpeg_command": "rendering",
    "headless_browser": "rendering", "open_viewer": "rendering",
    "wait_for_browser": "rendering", "RenderProblem": "rendering",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(f".{module}", __name__), name)


def __dir__():
    return list(__all__)
