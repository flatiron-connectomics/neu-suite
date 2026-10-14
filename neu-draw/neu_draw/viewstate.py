"""Saved camera viewpoints, kept outside any one view so they outlive it.

Use them to show two cells from the same angle, return to a framing after a stray drag,
or reopen a figure where the last one closed. :data:`views` maps a slot name to a
:class:`ViewState`; ``View.save_view`` and ``View.restore_view`` write and read it. The
camera state is whatever ``pygfx.Camera.get_state()`` returns, kept as an opaque dict.
It is absolute (a world position, rotation and zoom), so it carries across scenes of any
extent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: The slot the toolbar's "save view" button and ``save_view()`` write by default.
SAVED = "saved"

#: The slot a view records itself into when it closes, so the last angle can be restored
#: without having been saved.
LAST = "last"


@dataclass(frozen=True)
class ViewState:
    """A camera, and the canvas size it was framed for. The size matters: a perspective
    camera's horizontal field of view follows the canvas aspect ratio, so the same camera
    in a differently shaped canvas frames a different part of the scene."""
    camera: dict[str, Any] = field(default_factory=dict)
    size: tuple[int, int] = (0, 0)


#: Slot name → :class:`ViewState`, for the whole session. A plain dict: print it, clear
#: it, or add slots.
views: dict[str, ViewState] = {}
