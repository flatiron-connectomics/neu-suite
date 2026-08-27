"""Saved camera viewpoints, held **outside** any one view so they outlive it.

A viewpoint is the thing you want to reuse: line two cells up on the same angle, get
back to the framing you had before a stray drag, or reopen a figure where the last one
closed. None of that can live on the view, because the whole point is that the view it
came from is gone by the time you want it.

So this module is a dict, and deliberately nothing more. :data:`views` maps a slot name
to a :class:`ViewState`; ``View.save_view`` / ``View.restore_view`` are the two ends. It
is *pure* — no renderer, no canvas — which is what lets a notebook print it, edit it, or
pickle it without a GPU in the picture.

The camera state itself is whatever ``pygfx.Camera.get_state()`` returns, kept as an
opaque dict on purpose: it is absolute (a world position, a rotation, a zoom), so it
carries across scenes of completely different extent, which is exactly the
two-cells-same-angle case. Interpreting it here would mean tracking pygfx's camera
fields, and there is nothing to gain from that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: The slot the toolbar's "save view" button and ``save_view()`` write by default.
SAVED = "saved"

#: The slot a view records itself into as it CLOSES, which is the one nobody remembers to
#: save. Closing a figure to look at its snapshot and then wanting the angle back is the
#: ordinary case, so it is captured whether or not anyone asked.
LAST = "last"


@dataclass(frozen=True)
class ViewState:
    """A camera, and the canvas size it was framed for.

    **The size is part of the viewpoint, not decoration.** A perspective camera's
    horizontal field of view follows the aspect ratio of the rect it renders into, so the
    same camera state in a differently shaped canvas frames a different amount of the
    scene. Restoring the size is what makes "the same viewpoint" mean the same picture.
    """
    camera: dict[str, Any] = field(default_factory=dict)
    size: tuple[int, int] = (0, 0)


#: Slot name → :class:`ViewState`, for the whole session. A plain dict because that is
#: all it needs to be: ``neu_draw.views`` prints, ``views.clear()`` resets, and a caller
#: wanting more than two slots just uses more keys.
views: dict[str, ViewState] = {}
