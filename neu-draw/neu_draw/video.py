"""Finding ffmpeg, encoding frames with it, and the result of an orbit. Standard library only.

Kept apart from :mod:`neu_draw.backends.animation`, which renders, so that everything about
video *files* is importable — and tested — where there is no renderer, which is CI.

**Nothing site-specific lives here.** How a machine provides ffmpeg — a module system, a
store path, a conda package — is the user's environment's business: put it on ``PATH``, or
name it with ``$NEU_DRAW_FFMPEG``. The printed fallback is a plain ``ffmpeg`` line.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

#: The frame file names, 0-based. ffmpeg's ``%05d`` and Python's agree.
FRAME_PATTERN = "frame_%05d.png"
#: Names an ffmpeg executable to use, ahead of anything on ``PATH``.
FFMPEG_ENV = "NEU_DRAW_FFMPEG"
#: What an encode is handed after the input, in order. H.264 at a visually lossless CRF,
#: in the one pixel format every player accepts.
ENCODE_ARGS = ("-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18")
ENCODE_MODES = ("auto", True, False)


def find_ffmpeg(explicit: Optional[str] = None) -> Optional[str]:
    """The ffmpeg to encode with, or ``None``. First match wins:

    1. ``explicit`` — the caller's own path, or a name to look up on ``PATH``;
    2. ``$NEU_DRAW_FFMPEG`` — the same, from the environment (a shell profile, a
       notebook's first cell);
    3. ``ffmpeg`` on ``PATH`` — which is what loading a site's ffmpeg into the shell
       that started Jupyter gives you;
    4. ``imageio-ffmpeg``'s bundled binary, if that package happens to be installed.
       Opportunistic, never a dependency.

    **An explicit choice that is not usable RAISES** rather than falling through to the
    next — the same rule as neu-mark's ``$NEU_MARK_CONFIG``. Silently encoding with a
    different binary than the one named is how "I set it and nothing changed" happens.
    """
    for source, value in (("ffmpeg=", explicit), (f"${FFMPEG_ENV}", os.environ.get(FFMPEG_ENV))):
        if value:
            found = shutil.which(value)
            if found is None:
                raise FileNotFoundError(
                    f"{source} names {value!r}, which is not an executable (checked as a "
                    f"path and on PATH)")
            return found
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:                                           # noqa: BLE001
        return None


def encode_args(ffmpeg: str, directory: str, fps: float, output: str) -> list[str]:
    """The argv for an encode — a list, run without a shell, so no path needs quoting."""
    return [ffmpeg, "-y", "-loglevel", "error", "-framerate", f"{float(fps):g}",
            "-i", os.path.join(directory, FRAME_PATTERN), *ENCODE_ARGS, output]


def run_encode(directory: str, fps: float, output: str, ffmpeg: str) -> str:
    """Run ffmpeg over the frames. Raises with ffmpeg's own last lines if it fails."""
    done = subprocess.run(encode_args(ffmpeg, directory, fps, output),
                          capture_output=True, text=True)
    if done.returncode != 0:
        tail = "\n".join((done.stderr or done.stdout or "").strip().splitlines()[-8:])
        raise RuntimeError(f"ffmpeg exited with {done.returncode} encoding {output}:\n"
                           f"{tail}")
    return output


@dataclass
class Orbit:
    """What a render produced: the frames, and the video if one was encoded."""
    directory: str
    frames: int
    rendered: int
    size: tuple[int, int]
    fps: float
    #: The mp4 path, once written; ``None`` when nothing was encoded.
    video: Optional[str] = None
    #: Where the video goes (or would go): the frames folder's name plus ``.mp4``.
    output: Optional[str] = None

    def __post_init__(self) -> None:
        if self.output is None:
            self.output = self.directory.rstrip("/") + ".mp4"

    def command(self, ffmpeg: str = "ffmpeg") -> str:
        """The encode as one line to paste into a shell."""
        return shlex.join(encode_args(ffmpeg, self.directory, self.fps, self.output))

    def __repr__(self) -> str:
        head = (f"Orbit({self.frames} frames in {self.directory!r}, "
                f"{self.rendered} rendered now")
        if self.video:
            return f"{head}; encoded to {self.video!r})"
        return (f"{head}; not encoded — no ffmpeg found. Encode with:\n  {self.command()}\n"
                f"or put ffmpeg on PATH, or set ${FFMPEG_ENV}, and it runs itself next time)")
