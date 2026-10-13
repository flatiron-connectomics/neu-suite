"""Render an orbit of a view to numbered PNG frames, and encode them if ffmpeg is here.

The maths is :mod:`neu_draw.orbit`; this is getting frames out of the renderer. Four
choices, each the way it is because the alternative failed somewhere already:

- **One offscreen renderer for the whole run.** A fresh canvas per frame is what
  ``snapshot`` does and is fine once; for hundreds of frames it is setup cost paid per
  frame for nothing.
- **Supersampled, then averaged down.** Frames come out at the requested size, rendered at
  ``supersample`` times it — which is where the canvas's own antialiasing comes from — so a
  1500x900 video does not have 3000x1800 frames, and does not have jagged lines either.
- **Written atomically, and resumed by parameters.** A frame is written to a temporary
  name and renamed, so a run killed mid-write leaves no torn frame behind to be counted as
  done (neu-glance found that one seven hundred frames into a render). The parameters go
  in ``animation.json`` beside the frames, and a rerun into the same folder resumes only
  when they match — frames from two different orbits in one sequence would encode
  perfectly and be wrong.
- **ffmpeg is looked for, never assumed** — see :func:`find_ffmpeg`. Where one is found
  the frames are encoded; where none is, a plain ``ffmpeg`` line is printed instead, and
  nothing about any one site's way of providing it (a module system, a store path) is
  baked in: that belongs in the user's environment, via ``PATH`` or ``NEU_DRAW_FFMPEG``.
  ``-pix_fmt yuv420p`` is not optional: without it the mp4 encodes and then will not play
  in PowerPoint, Keynote or Slack — which also needs even frame dimensions, so odd ones
  are refused before any rendering rather than by libx264.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Optional

import numpy as np
import pygfx

from .. import orbit as _orbit
from ..video import (ENCODE_MODES, FFMPEG_ENV, FRAME_PATTERN, Orbit, find_ffmpeg,
                     run_encode)

PARAMS_FILE = "animation.json"
LIGHTS = ("camera", "fixed")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
def _write_png(path: str, rgb: np.ndarray) -> None:
    from imageio import v3 as iio

    tmp = os.path.join(os.path.dirname(path), "." + os.path.basename(path) + ".tmp")
    iio.imwrite(tmp, rgb, extension=".png")
    os.replace(tmp, path)


def _is_frame(path: str) -> bool:
    try:
        with open(path, "rb") as fh:
            return fh.read(8) == PNG_SIGNATURE
    except OSError:
        return False


def _downsample(rgba: np.ndarray, factor: int, size: tuple[int, int]) -> np.ndarray:
    """RGB at ``size`` (w, h) from a render ``factor`` times larger, by box averaging."""
    width, height = size
    if factor == 1:
        return np.ascontiguousarray(rgba[:height, :width, :3])
    # Integer strided sums, not a float reshape-and-mean: measured at 1500x900 from a 2x
    # render, 54 ms against 335 ms — which had been 80% of the whole per-frame cost.
    rgb = rgba[:height * factor, :width * factor, :3]
    total = np.zeros((height, width, 3), dtype=np.uint32)
    for i in range(factor):
        for j in range(factor):
            total += rgb[i::factor, j::factor]
    n = factor * factor
    return ((total + n // 2) // n).astype(np.uint8)


def orbit(view, directory: str, *, degrees: float = 360.0, seconds: float = 12.0,
          fps: float = 30.0, axis: Any = "up", light: str = "camera",
          ease: str = "linear", size: Optional[tuple[int, int]] = None,
          supersample: int = 2, legend: bool = True, overwrite: bool = False,
          encode: Any = "auto", output: Optional[str] = None,
          ffmpeg: Optional[str] = None,
          progress: Optional[Callable[[int, int], None]] = None) -> Orbit:
    """Render ``seconds * fps`` frames of the camera turning ``degrees`` about ``axis``.

    ``axis`` is a screen axis (``up``, ``right``, ``view``), a data axis (``x``, ``y``,
    ``z``), or an xyz vector — see :mod:`neu_draw.orbit`. ``light="camera"`` turns the
    scene's directional lights with the camera, so shading stays put as it spins (a
    turntable under studio lights); ``"fixed"`` leaves them in the scene, so the shading
    sweeps across the surface. The view itself is left exactly as it was.

    ``encode`` — ``"auto"`` (encode if :func:`find_ffmpeg` finds one), ``True``
    (encode or raise), ``False`` (frames only). ``output`` is the mp4, defaulting to the
    folder's name plus ``.mp4``; ``ffmpeg`` names the executable. The ffmpeg lookup runs
    **before** any rendering, so a bad path fails in a second rather than after the run.

    Returns an :class:`Orbit`: the video's path if one was written, the line that would
    write it if not.
    """
    if encode not in ENCODE_MODES:
        raise ValueError(f"encode is 'auto', True or False, not {encode!r}")
    exe = find_ffmpeg(ffmpeg) if encode is not False else None
    if encode is True and exe is None:
        raise FileNotFoundError(
            f"encode=True, but no ffmpeg was found: pass ffmpeg=, set "
            f"${FFMPEG_ENV}, or put ffmpeg on PATH")
    width, height = (int(v) for v in (size or view.logical_size()))
    if width % 2 or height % 2:
        raise ValueError(f"frame size {width}x{height} has an odd side; H.264 in yuv420p "
                         f"needs even dimensions, so pass e.g. size=({width // 2 * 2}, "
                         f"{height // 2 * 2})")
    if light not in LIGHTS:
        raise ValueError(f"light is one of {', '.join(LIGHTS)}, not {light!r}")
    supersample = int(supersample)
    if supersample < 1:
        raise ValueError(f"supersample must be at least 1, got {supersample}")
    n = int(round(float(seconds) * float(fps)))
    turn = _orbit.angles(degrees, n, ease=ease)

    start = view.camera.get_state()
    axis_world = _orbit.world_axis(axis, start["rotation"])
    sphere = _centre(view)
    about = _orbit.pivot(start["position"], start["rotation"], sphere)

    params = {"degrees": float(degrees), "frames": n, "fps": float(fps),
              "axis": axis if isinstance(axis, str) else [float(v) for v in axis],
              "light": light, "ease": ease, "size": [width, height],
              "supersample": supersample, "legend": bool(legend),
              "camera": {k: np.asarray(v).tolist() if hasattr(v, "__len__") else v
                         for k, v in start.items()}}
    _prepare(directory, params, overwrite)

    from rendercanvas.offscreen import RenderCanvas as Offscreen

    canvas = Offscreen(size=(width, height))
    renderer = pygfx.renderers.WgpuRenderer(canvas, pixel_ratio=supersample)
    camera = pygfx.PerspectiveCamera(view.camera.fov)
    camera.set_state(start)

    lights = [(obj, np.array(obj.local.position, dtype=np.float64))
              for obj in view.scene.children if isinstance(obj, pygfx.DirectionalLight)]
    rendered = 0
    try:
        for k, angle in enumerate(turn):
            path = os.path.join(directory, FRAME_PATTERN % k)
            if not overwrite and _is_frame(path):
                continue
            position, rotation = _orbit.pose(start["position"], start["rotation"],
                                             about, axis_world, angle)
            camera.set_state({**start, "position": position, "rotation": rotation})
            if light == "camera":
                for obj, original in lights:
                    target = np.asarray(obj.target.world.position, dtype=np.float64)
                    moved, _ = _orbit.pose(original, (0, 0, 0, 1), target, axis_world,
                                           angle)
                    obj.local.position = moved
            view._paint(renderer, camera, legend=legend)
            _write_png(path, _downsample(np.asarray(renderer.snapshot()), supersample,
                                         (width, height)))
            rendered += 1
            if progress is not None:
                progress(k + 1, n)
    finally:
        for obj, original in lights:
            obj.local.position = original
    result = Orbit(directory, n, rendered, (width, height), float(fps), output=output)
    if exe is not None:
        if progress is not None:
            progress(n, n)
        result.video = run_encode(directory, fps, result.output, exe)
    return result


def _centre(view) -> np.ndarray:
    from .pygfx import _visible_sphere

    sphere = _visible_sphere(view.group)
    if sphere is not None:
        return np.asarray(sphere[:3], dtype=np.float64)
    return np.asarray(view.scene_data.bbox.center.xyz, dtype=np.float64)


def _prepare(directory: str, params: dict, overwrite: bool) -> None:
    """Make the folder, and refuse to mix frames from a different orbit into it."""
    os.makedirs(directory, exist_ok=True)
    record = os.path.join(directory, PARAMS_FILE)
    frames = sorted(f for f in os.listdir(directory)
                    if f.startswith("frame_") and f.endswith(".png"))
    if overwrite:
        for name in frames:
            os.remove(os.path.join(directory, name))
    elif frames:
        try:
            with open(record) as fh:
                previous = json.load(fh)
        except (OSError, ValueError):
            previous = None
        if previous != json.loads(json.dumps(params)):
            raise FileExistsError(
                f"{directory} already holds {len(frames)} frames of a DIFFERENT orbit "
                f"(or of one with no {PARAMS_FILE}). Resuming would splice two animations "
                f"into one sequence; pass overwrite=True, or choose another folder.")
    with open(record, "w") as fh:
        json.dump(params, fh, indent=1)
