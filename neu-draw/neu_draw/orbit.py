"""Camera orbits: the angle schedule and the pose at each angle. Pure numpy.

An orbit rotates the camera, never the data, about the point at the centre of the screen
at the depth of the scene's centre, so frame 0 is the current view. The axis is either a
screen axis (``up``, ``right``, ``view``, read from the camera as the orbit starts, so
``up`` is a turntable of the current view) or a data axis (``x``, ``y``, ``z`` or an xyz
vector, in nm). Either way the camera turns about a world-space axis.

An orbit is a schedule of angles, not an interpolation between orientations, which would
take the short way round and make 360° a no-op. Positive degrees turn the camera by the
right-hand rule, so the scene appears to turn the other way.
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np

#: Camera-local directions. pygfx cameras look down their local -z.
SCREEN_AXES = {"up": (0.0, 1.0, 0.0), "right": (1.0, 0.0, 0.0), "view": (0.0, 0.0, -1.0)}
#: Model-space directions, xyz, as pygfx takes them.
DATA_AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
#: Progress curves over [0, 1]. ``in-out`` eases both ends; use ``linear`` for a loop, or it
#: pauses visibly at the seam.
EASES = {
    "linear": lambda t: t,
    "in-out": lambda t: t * t * (3.0 - 2.0 * t),
}

Axis = Union[str, Sequence[float]]


def angles(degrees: float, frames: int, *, ease: str = "linear") -> np.ndarray:
    """The camera angle at each frame, in radians, starting at 0.

    A whole number of turns omits the endpoint, so the video loops without a repeated
    frame; any other sweep ends exactly at ``degrees``.
    """
    frames = int(frames)
    if frames < 1:
        raise ValueError(f"an orbit needs at least one frame, got {frames}")
    if ease not in EASES:
        raise ValueError(f"ease is one of {', '.join(EASES)}, not {ease!r}")
    loops = abs(float(degrees)) > 0 and float(degrees) % 360.0 == 0.0
    steps = frames if loops else max(frames - 1, 1)
    t = np.arange(frames, dtype=np.float64) / steps
    return np.radians(float(degrees)) * np.array([EASES[ease](v) for v in t])


def _unit(v: Sequence[float]) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("an orbit axis must not be the zero vector")
    return v / n


def _rotate(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Rotate ``v`` by the unit quaternion ``q`` (xyzw, pygfx's order)."""
    xyz, w = q[:3], q[3]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def _mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a ⊗ b``, xyzw."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])


def axis_quaternion(axis: Sequence[float], angle: float) -> np.ndarray:
    """The rotation by ``angle`` radians about the unit vector ``axis``, xyzw."""
    half = 0.5 * float(angle)
    return np.r_[_unit(axis) * np.sin(half), np.cos(half)]


def world_axis(axis: Axis, camera_rotation: Sequence[float]) -> np.ndarray:
    """``axis`` as a world direction. Checked eagerly, so a typo fails before frame 0."""
    if isinstance(axis, str):
        name = axis.lower()
        if name in SCREEN_AXES:
            q = np.asarray(camera_rotation, dtype=np.float64)
            return _unit(_rotate(np.array(SCREEN_AXES[name]), q / np.linalg.norm(q)))
        if name in DATA_AXES:
            return _unit(DATA_AXES[name])
        raise ValueError(f"unknown orbit axis {axis!r}; screen axes are "
                         f"{', '.join(SCREEN_AXES)}, data axes {', '.join(DATA_AXES)}, "
                         f"or pass an xyz 3-vector")
    return _unit(axis)


def pivot(position: Sequence[float], rotation: Sequence[float],
          centre: Sequence[float]) -> np.ndarray:
    """The point at the screen centre, at the depth of ``centre``. Not ``centre`` itself,
    which after a pan is off to one side, so orbiting it would swing the view away."""
    position = np.asarray(position, dtype=np.float64)
    q = np.asarray(rotation, dtype=np.float64)
    forward = _rotate(np.array([0.0, 0.0, -1.0]), q / np.linalg.norm(q))
    depth = float(np.dot(np.asarray(centre, dtype=np.float64) - position, forward))
    return position + forward * depth


def pose(position: Sequence[float], rotation: Sequence[float], about: Sequence[float],
         axis_world: Sequence[float], angle: float) -> tuple[np.ndarray, np.ndarray]:
    """The camera ``(position, rotation)`` after turning ``angle`` about ``about``."""
    r = axis_quaternion(axis_world, angle)
    about = np.asarray(about, dtype=np.float64)
    moved = about + _rotate(np.asarray(position, dtype=np.float64) - about, r)
    turned = _mul(r, np.asarray(rotation, dtype=np.float64))
    return moved, turned / np.linalg.norm(turned)
