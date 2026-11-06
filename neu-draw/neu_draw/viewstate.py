"""Saved camera viewpoints, kept outside any one view so they outlive it.

Use them to show two cells from the same angle, return to a framing after a stray drag,
or reopen a figure where the last one closed. :data:`views` maps a slot name to a
:class:`ViewState`; ``View.save_view`` and ``View.restore_view`` write and read it. The
camera state is whatever ``pygfx.Camera.get_state()`` returns, kept as an opaque dict.
It is absolute (a world position, rotation and zoom), so it carries across scenes of any
extent.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Self

#: The slot the toolbar's "save view" button and ``save_view()`` write by default.
SAVED = "saved"

#: The slot a view records itself into when it closes, so the last angle can be restored
#: without having been saved.
LAST = "last"


def _jsonable(obj: Any) -> Any:
    """``json``'s ``default`` hook: numpy arrays and scalars as plain lists and numbers."""
    import numpy as np

    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist()
    raise TypeError(f"{type(obj).__name__} is not JSON serializable")


@dataclass(frozen=True)
class ViewState:
    """A camera, and the canvas size it was framed for. The size matters: a perspective
    camera's horizontal field of view follows the canvas aspect ratio, so the same camera
    in a differently shaped canvas frames a different part of the scene."""
    camera: dict[str, Any] = field(default_factory=dict)
    size: tuple[int, int] = (0, 0)

    def save(self, path: str | Path, *, overwrite: bool = True) -> Path:
        """Write this state to a JSON file, creating its directory. Returns the path."""
        path = Path(path).expanduser().resolve()
        if path.exists() and not overwrite:
            raise FileExistsError(f"File already exists at {path}")
        # Serialize before opening, so a failure cannot truncate an existing file.
        text = json.dumps(asdict(self), default=_jsonable, indent=1)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print(f"Saved viewstate to {path}")
        return path

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Read a state written by :meth:`save`. Camera vectors come back as lists, which
        ``pygfx.Camera.set_state`` accepts as it does arrays."""
        state = json.loads(Path(path).expanduser().read_text())
        return cls(camera=state["camera"], size=tuple(state["size"]))


#: Slot name → :class:`ViewState`, for the whole session. A plain dict: print it, clear
#: it, or add slots.
views: dict[str, ViewState] = {}


def save_view_file(path: str | Path, name: str = SAVED, *,
                   overwrite: bool = True) -> Path:
    """Write the state in slot ``name`` of :data:`views` to a JSON file."""
    if name not in views:
        raise KeyError(f"no view in slot {name!r}; slots: {sorted(views)}")
    return views[name].save(path, overwrite=overwrite)


def load_view_file(path: str | Path, name: str = SAVED) -> ViewState:
    """Read a JSON file written by :func:`save_view_file` into slot ``name`` of
    :data:`views`."""
    views[name] = state = ViewState.load(path)
    return state
