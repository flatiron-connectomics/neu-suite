"""Render a :class:`neu_glance.movie.animate.Timeline` to a sequence of PNGs, through a real viewer.

Each frame is a round trip: Python pushes a state, the browser loads the visible chunks,
draws, and posts the image back. :mod:`neu_glance.movie.animate` decides what each frame is.

By default a headless chrome or chromium does the drawing, started with
``subprocess.Popen`` (no selenium: the viewer's own channel carries every state change and
screenshot). With :data:`GPU_FLAGS` it is also fast: measured on one 1920x1080 mesh scene,
0.21 s/frame, against 2.17 s/frame without them and 0.57 s/frame through a browser on
another machine. ``record(..., browser="none")`` prints a URL and waits instead.

neuroglancer is imported only inside functions (a test checks), so ``import neu_glance``
does not need it. Beyond ``neuroglancer.tool.screenshot``, this:

* requeues a wrong-sized reply instead of writing it;
* writes frames atomically, and on resume checks each file's PNG signature;
* gives up after a bounded number of browser refreshes instead of hanging.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from typing import Any, Mapping, Sequence

from .animate import Timeline
from ..served_layer import _neuroglancer

#: The first eight bytes of any PNG, checked on resume.
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

#: Panels switched off in a rendered frame, set once on the base state.
CHROME_OFF = {
    "selectedLayer": {"visible": False},
    "statistics": {"visible": False},
    "layerListPanel": {"visible": False},
    "helpPanel": {"visible": False},
}


class RenderProblem(RuntimeError):
    """A render could not start, or could not finish the frame it was on."""


class _Tail(io.TextIOBase):
    """A bounded sink for the capture loop's output: keeps the last few lines. Bounded,
    because a stalled render prints a line per second indefinitely."""

    def __init__(self, keep: int = 40) -> None:
        self.lines: deque[str] = deque(maxlen=keep)
        self._partial = ""

    def write(self, text: str) -> int:
        self._partial += text
        while "\n" in self._partial:
            line, self._partial = self._partial.split("\n", 1)
            if line.strip():
                self.lines.append(line.strip())
        return len(text)


class _Progress:
    """A tqdm bar if tqdm is installed (it is not a dependency), else periodic lines with an
    ETA, or nothing when ``verbose``. While a bar is shown, other output to the terminal is
    suppressed, since it would garble the bar."""

    def __init__(self, total: int, *, verbose: bool = False, description: str = "rendering",
                 stream=None, min_interval: float = 10.0) -> None:
        self.total, self.verbose, self.min_interval = total, verbose, min_interval
        self.stream = stream or sys.stderr
        self.done = 0
        self.started = self.last = time.monotonic()
        self.bar = None
        if verbose:
            return
        try:
            from tqdm.auto import tqdm
        except ImportError:
            print(f"{description}: {total} frames", file=self.stream)
        else:
            self.bar = tqdm(total=total, unit="frame", desc=description, file=self.stream,
                            dynamic_ncols=True)

    def advance(self, _path: str | None = None) -> None:
        self.done += 1
        if self.bar is not None:
            self.bar.update(1)
            return
        if self.verbose:
            return
        now = time.monotonic()
        if now - self.last < self.min_interval and self.done < self.total:
            return
        self.last = now
        per = (now - self.started) / max(1, self.done)
        left = per * (self.total - self.done)
        print(f"  {self.done}/{self.total} frames ({self.done / max(1, self.total):.0%}), "
              f"{per:.2f}s/frame, ~{left / 60:.1f} min left", file=self.stream)

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()
            self.bar = None


# --------------------------------------------------------------------------- #
# shaping the state
# --------------------------------------------------------------------------- #
def render_state(state: Mapping[str, Any], *, show_axis_lines: bool = False,
                 show_default_annotations: bool = False,
                 show_scale_bar: bool | None = None,
                 gpu_memory_limit: int | None = None,
                 system_memory_limit: int | None = None,
                 concurrent_downloads: int | None = None) -> dict:
    """A state with the viewer's furniture switched off, ready to be a frame.

    Apply it once to the base state, before interpolation. Axis lines and default
    annotations (bounding boxes) default to off, unlike in the viewer. The memory limits and
    concurrent downloads are left at the viewer's defaults unless given; they matter for
    scenes with many mesh layers.
    """
    out = json.loads(json.dumps(dict(state)))
    for key, value in CHROME_OFF.items():
        existing = out.get(key)
        out[key] = {**existing, **value} if isinstance(existing, dict) else dict(value)
    out["showAxisLines"] = bool(show_axis_lines)
    out["showDefaultAnnotations"] = bool(show_default_annotations)
    if show_scale_bar is not None:
        out["showScaleBar"] = bool(show_scale_bar)
    for key, value in (("gpuMemoryLimit", gpu_memory_limit),
                       ("systemMemoryLimit", system_memory_limit),
                       ("concurrentDownloads", concurrent_downloads)):
        if value is not None:
            out[key] = int(value)
    # A tool palette is UI that names layers, which a split may have emptied.
    out.pop("toolPalettes", None)
    return out


# --------------------------------------------------------------------------- #
# frames on disk
# --------------------------------------------------------------------------- #
def frame_path(out_dir: str, index: int, pattern: str = "frame_%05d.png") -> str:
    """``<out_dir>/frame_00042.png``. ``pattern`` must be usable verbatim by ``ffmpeg -i``."""
    try:
        name = pattern % index
    except TypeError:
        raise RenderProblem(
            f"frame pattern {pattern!r} has no integer field; ffmpeg reads a sequence through "
            f"one, so it needs something like 'frame_%05d.png'") from None
    return os.path.join(out_dir, name)


def is_written(path: str) -> bool:
    """True only for a file that exists and begins with the PNG signature, so a frame
    killed mid-write is rendered again."""
    try:
        with open(path, "rb") as f:
            return f.read(len(PNG_MAGIC)) == PNG_MAGIC
    except OSError:
        return False


def _write_atomic(path: str, data: bytes) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def stale_frames(out_dir: str, frame_count: int,
                 pattern: str = "frame_%05d.png") -> list[str]:
    """Frame files numbered at or past ``frame_count`` — leftovers from a longer earlier run.

    ``ffmpeg -i frame_%05d.png`` reads the whole numbered sequence, so after shortening an
    animation these would be appended to the video.
    """
    matcher = re.compile("^" + re.sub(r"%0?\d*d", r"(\\d+)", re.escape(pattern)
                                      .replace("\\%", "%")) + "$")
    out = []
    for name in sorted(os.listdir(out_dir)) if os.path.isdir(out_dir) else []:
        found = matcher.match(name)
        if found and int(found.group(1)) >= frame_count:
            out.append(os.path.join(out_dir, name))
    return out


def pending_frames(timeline: Timeline, out_dir: str, *, resume: bool = True,
                   start_frame: int = 0, end_frame: int | None = None,
                   pattern: str = "frame_%05d.png") -> list[tuple[int, float, str]]:
    """``(index, seconds, path)`` for the frames still to render. States are built later.

    States are built one per frame later, rather than held for the whole timeline.
    """
    stop = timeline.frame_count if end_frame is None else min(end_frame, timeline.frame_count)
    out = []
    for i in range(max(0, start_frame), stop):
        path = frame_path(out_dir, i, pattern)
        if resume and is_written(path):
            continue
        out.append((i, i / timeline.fps, path))
    return out


# --------------------------------------------------------------------------- #
# the browser
# --------------------------------------------------------------------------- #
def open_viewer(*, bind: str | None = None, port: int = 0):
    """A viewer for a browser to attach to. Returns it; its ``.get_viewer_url()`` is the link.

    ``bind`` and ``port`` take effect only before the first viewer in the process, since
    neuroglancer reads them at server start. ``bind="0.0.0.0"`` puts the host's name in the
    URL, for a browser on another machine.
    """
    ng = _neuroglancer()
    if bind is not None or port:
        if ng.server.is_server_running():
            print(f"a viewer server is already running in this process, so bind={bind!r} "
                  f"port={port!r} are ignored — neuroglancer reads them once, at server start",
                  file=sys.stderr)
        else:
            ng.set_server_bind_address(bind or "127.0.0.1", port)
    return ng.Viewer()


#: Browser binaries to look for, in preference order.
BROWSERS = ("google-chrome", "google-chrome-stable", "chromium-browser", "chromium")

#: Flags that put WebGL on the GPU instead of the software rasteriser (SwiftShader): about
#: 10x faster per frame on a mesh scene.
GPU_FLAGS = ("--use-angle=vulkan", "--enable-features=Vulkan", "--enable-gpu",
             "--ignore-gpu-blocklist")

#: Flags that make Chrome survive being run headless as an ordinary user on a shared machine.
HEADLESS_FLAGS = ("--headless=new", "--no-sandbox", "--disable-dev-shm-usage",
                  "--disable-extensions", "--mute-audio")


def find_browser(binary: str | None = None) -> str | None:
    """A chrome/chromium executable, or ``None``. ``binary`` overrides the search."""
    if binary:
        return binary if os.path.exists(binary) else shutil.which(binary)
    for name in BROWSERS:
        found = shutil.which(name)
        if found:
            return found
    return None


@contextlib.contextmanager
def headless_browser(url: str, *, size: tuple[int, int] = (1920, 1080),
                     binary: str | None = None, gpu: bool = True,
                     extra_args: Sequence[str] = ()):
    """Run a browser on ``url`` for the duration of the block, with no window and no tab.

    No selenium is needed: the browser only has to load the URL, and the viewer's channel
    does the rest. Yields a function returning ``None`` while the browser runs, or why it
    stopped.

    The window is sized to the render, or every screenshot comes back the wrong size. The
    profile is a temporary directory, removed afterwards; sharing one would make a second
    render attach to the first browser.
    """
    executable = find_browser(binary)
    if executable is None:
        raise RenderProblem(
            f"no browser found (looked for {', '.join(BROWSERS)}). Pass binary=..., or render "
            f"attended by opening the viewer URL yourself — `record(..., browser='none')`.")
    profile = tempfile.mkdtemp(prefix="neu-glance-render-")
    runtime = os.path.join(profile, "runtime")
    os.makedirs(runtime, mode=0o700, exist_ok=True)
    command = [executable, *HEADLESS_FLAGS, f"--user-data-dir={profile}",
               f"--window-size={size[0]},{size[1]}", *(GPU_FLAGS if gpu else ()),
               *extra_args, url]

    # A private XDG_RUNTIME_DIR: under a long-lived tmux the inherited one may have been
    # deleted at logout, and chrome then fails to start.
    environment = {**os.environ, "XDG_RUNTIME_DIR": runtime}
    environment.pop("DBUS_SESSION_BUS_ADDRESS", None)

    log_path = os.path.join(profile, "browser.log")
    log = open(log_path, "wb")
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=log,
                               env=environment)
    print(f"launched {os.path.basename(executable)} headless"
          f"{' on the GPU' if gpu else ' (software rendering)'}", file=sys.stderr)

    def check() -> str | None:
        """``None`` while it is running; why it stopped once it has."""
        code = process.poll()
        if code is None:
            return None
        log.flush()
        try:
            with open(log_path, "rb") as f:
                tail = f.read()[-2000:].decode("utf-8", "replace").strip()
        except OSError:                                         # pragma: no cover
            tail = ""
        detail = ("\n    " + "\n    ".join(tail.splitlines()[-8:])) if tail else ""
        hint = ""
        if gpu:
            hint = ("\n    The GPU flags may not work here — retry with gpu=False "
                    "(--software-gl), which is slower but needs nothing from the driver.")
        return (f"{os.path.basename(executable)} exited with code {code} before the render "
                f"began.{hint}{detail}")

    try:
        yield check
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:                       # pragma: no cover
            process.kill()
        log.close()
        shutil.rmtree(profile, ignore_errors=True)


def _await_reply(viewer, *, timeout, notify_every, message, on_timeout, watch=None):
    """Block on one screenshot reply, reporting periodically; raise what `on_timeout` builds.
    `watch` is checked each tick, so a browser that died at startup fails at once."""
    ready = threading.Event()
    viewer.async_screenshot(lambda _reply: ready.set())
    waited = 0.0
    while not ready.wait(max(0.1, notify_every)):
        waited += notify_every
        if watch is not None:
            stopped = watch()
            if stopped:
                raise RenderProblem(stopped)
        if waited >= timeout:
            raise on_timeout()
        print(f"{message} ({waited:g}s)", file=sys.stderr)


def wait_for_browser(viewer, state: Mapping[str, Any], *, timeout: float = 600.0,
                     notify_every: float = 15.0, stall_timeout: float = 180.0,
                     launched: bool = False, watch=None) -> None:
    """Push ``state``, then block until a browser has it open and loaded.

    ``launched=True`` when the browser was started here, which changes the messages (no
    "open this link") and the diagnosis if it never attaches.

    Two waits: ``timeout`` for a browser to attach, then for the scene to load, failing if
    no progress is made for ``stall_timeout``. A viewer with no layers and no dimensions
    never reports ready, so it cannot answer a screenshot; attachment is therefore tested
    with ``state``'s dimensions and no layers, which is ready almost at once. The viewer
    config (such as its size) is not touched, since a small ``viewerSize`` is shown
    magnified in the user's window.
    """
    url = viewer.get_viewer_url()
    if launched:
        print(f"waiting for the headless browser\n    {url}", file=sys.stderr)
    else:
        print(f"open this in a browser and leave it open:\n    {url}", file=sys.stderr)

    # Test attachment with an empty but valid state, so "no browser" is not confused with
    # "a large scene still loading".
    empty = {"dimensions": state.get("dimensions") or {}, "layers": [],
             "layout": state.get("layout", "3d")}
    viewer.set_state(empty)
    _await_reply(viewer, timeout=timeout, notify_every=notify_every,
                 message=(f"still waiting for {'the headless browser' if launched else 'a browser'}"
                          + ("" if launched else f":\n    {url}")),
                 on_timeout=lambda: RenderProblem(
                     f"no browser attached after {timeout:g}s.\n    {url}" + (
                         "\n    The browser was started here, so it has died or cannot reach "
                         "the viewer port. Try browser='none' to open one yourself, or "
                         "gpu=False if the GPU flags are not supported." if launched else
                         "\n    The viewer is bound to loopback, so only a browser on this "
                         "machine can reach it — pass bind='0.0.0.0' or forward the port."
                         if "localhost" in url or "127.0.0.1" in url else "")),
                 watch=watch)
    print("browser attached; loading the scene", file=sys.stderr)

    viewer.set_state(dict(state))
    loaded = threading.Event()
    progress: dict[str, Any] = {"at": time.monotonic(), "loaded": 0, "total": 0, "best": 0,
                                "downloading": 0}

    def on_statistics(statistics):
        total = statistics.total
        got = total.visible_chunks_gpu_memory
        if got > progress["best"]:                 # PROGRESS, not merely a message
            progress["best"] = got
            progress["at"] = time.monotonic()
        progress.update(loaded=got, total=total.visible_chunks_total,
                        downloading=total.visible_chunks_downloading)

    viewer.async_screenshot(lambda _reply: loaded.set(), statistics_callback=on_statistics)

    bar = _Progress(0, verbose=True)               # placeholder until a total is known
    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None
    else:
        bar = tqdm(total=0, unit="chunk", desc="loading the scene", file=sys.stderr,
                   dynamic_ncols=True)
    try:
        while not loaded.wait(1.0 if tqdm else max(0.1, notify_every)):
            if watch is not None:
                stopped = watch()
                if stopped:
                    raise RenderProblem(stopped)
            # Stalled means no progress, not silence: statistics arrive every second
            # regardless.
            silent = time.monotonic() - progress["at"]
            if silent >= stall_timeout and progress["total"]:
                missing = progress["total"] - progress["loaded"]
                raise RenderProblem(
                    f"the scene stopped loading {silent:.0f}s ago at "
                    f"{progress['loaded']}/{progress['total']} chunks, "
                    f"{progress['downloading']} downloading, and the viewer will never report "
                    f"itself ready — so no frame can be captured.\n"
                    f"    {missing} chunk(s) short. The usual cause is a SEGMENT WITH NO MESH: "
                    f"neuroglancer counts it among the visible chunks and waits for it "
                    f"forever. `neu_glance.sources.segments_with_meshes(volume, ids)` filters "
                    f"those out — measured at 3s for 2200 bodies.")
            if tqdm:
                if progress["total"] and bar.total != progress["total"]:
                    bar.total = progress["total"]
                bar.n = progress["loaded"]
                bar.set_postfix_str(f"{progress['downloading']} downloading", refresh=False)
                bar.refresh()
            elif progress["total"]:
                print(f"  loading: {progress['loaded']}/{progress['total']} chunks "
                      f"({progress['downloading']} downloading)", file=sys.stderr)
    finally:
        if tqdm:
            bar.close()
    print("scene loaded; rendering", file=sys.stderr)


# --------------------------------------------------------------------------- #
# the render
# --------------------------------------------------------------------------- #
class _RequestQueue:
    """Feeds ``capture_screenshots``, requeuing a frame whose reply is the wrong size (a
    transient fault that the capture loop does not retry)."""

    def __init__(self, timeline, items, *, size, attempts, request_type, on_write=None):
        self.timeline = timeline
        self.queue = deque(items)
        self.size = size
        self.attempts = attempts
        self.request_type = request_type
        self.on_write = on_write
        self.tries: dict[int, int] = {}
        self.written: list[str] = []

    def __iter__(self):
        return self

    def __next__(self):
        if not self.queue:
            raise StopIteration
        index, seconds, path = self.queue.popleft()
        width, height = self.size

        def config_callback(s):
            s.viewer_size = (width, height)

        def response_callback(reply):
            if (reply.width, reply.height) != (width, height):
                self.tries[index] = self.tries.get(index, 0) + 1
                if self.tries[index] > self.attempts:
                    raise RenderProblem(
                        f"frame {index} came back {reply.width}x{reply.height} instead of "
                        f"{width}x{height}, {self.attempts} times running. The browser window "
                        f"is probably smaller than the requested size — enlarge it, or render "
                        f"smaller. Frames already written are kept; re-run to resume")
                print(f"frame {index}: got {reply.width}x{reply.height}, wanted "
                      f"{width}x{height} — requeued", file=sys.stderr)
                self.queue.appendleft((index, seconds, path))
                return
            _write_atomic(path, reply.image)
            self.written.append(path)
            if self.on_write is not None:
                self.on_write(path)

        return self.request_type(
            state=self.timeline.at(seconds),
            description=f"frame {index} @ {seconds:.3f}s",
            config_callback=config_callback,
            response_callback=response_callback)


def record(timeline: Timeline, out_dir: str, *, size: tuple[int, int] = (1920, 1080),
           viewer=None, bind: str | None = None, port: int = 0,
           prefetch: int = 1, refresh_timeout: float = 120.0, give_up_after: int = 5,
           connect_timeout: float = 600.0, size_attempts: int = 5,
           resume: bool = True, start_frame: int = 0, end_frame: int | None = None,
           pattern: str = "frame_%05d.png", shape: bool = True,
           verbose: bool = False, progress_interval: float = 10.0,
           browser: str = "auto", browser_binary: str | None = None,
           gpu: bool = True) -> list[str]:
    """Render ``timeline`` into ``out_dir``. Returns the frame paths written this run.

    ``browser="auto"`` runs a headless chrome or chromium if one is on the PATH, else prints
    a URL to open; ``"headless"`` requires one; ``"none"`` always waits for you to open the
    URL. ``verbose=True`` prints the capture loop's own lines; otherwise a progress bar is
    shown, and the last lines are printed only if the render fails.

    Writes ``timeline.json`` beside the frames. Interrupting is safe and re-running resumes,
    unless the timeline changed, in which case every frame is rendered again. Frames past
    the end of the timeline, left by a longer earlier run, are deleted.
    ``start_frame`` / ``end_frame`` render a slice with the same numbering.
    """
    _neuroglancer()          # fail early if the extra is missing

    width, height = (int(v) for v in size)
    for label, value in (("width", width), ("height", height)):
        if value % 2:
            raise RenderProblem(
                f"{label} must be even ({value} is not): libx264 with the yuv420p pixel "
                f"format needs even dimensions, and the error it gives otherwise is unhelpful")

    os.makedirs(out_dir, exist_ok=True)
    for note in timeline.check():
        print(note, file=sys.stderr)

    shaped = timeline
    if shape:
        shaped = Timeline(base=render_state(timeline.base), fps=timeline.fps,
                          tweens=list(timeline.tweens), groups=dict(timeline.groups),
                          cursor=timeline.cursor)
        shaped.duration = timeline.duration

    # Resuming is only correct if the timeline is unchanged; otherwise old and new frames
    # would mix silently. Compare with the timeline saved beside the frames.
    recorded = os.path.join(out_dir, "timeline.json")
    if resume and os.path.exists(recorded):
        try:
            with open(recorded) as f:
                previous = json.load(f)
        except (OSError, ValueError):
            previous = None
        if previous != shaped.to_json():
            print("the timeline differs from the one these frames were rendered with — "
                  "re-rendering all of them rather than mixing two animations", file=sys.stderr)
            resume = False

    # Frames from a longer earlier run are removed (and reported): ffmpeg would append them.
    leftovers = stale_frames(out_dir, shaped.frame_count, pattern)
    if leftovers:
        print(f"removing {len(leftovers)} frame(s) past the end of this timeline "
              f"({os.path.basename(leftovers[0])}..{os.path.basename(leftovers[-1])}) — they "
              f"are from a longer earlier run and ffmpeg would append them", file=sys.stderr)
        for path in leftovers:
            os.remove(path)

    todo = pending_frames(shaped, out_dir, resume=resume, start_frame=start_frame,
                          end_frame=end_frame, pattern=pattern)
    total = shaped.frame_count if end_frame is None else end_frame - start_frame
    _write_atomic(os.path.join(out_dir, "timeline.json"),
                  json.dumps(shaped.to_json(), indent=2).encode())
    print(f"{shaped.duration:g}s at {shaped.fps:g} fps = {shaped.frame_count} frames at "
          f"{width}x{height}; {len(todo)} to render"
          + (f" ({total - len(todo)} already present)" if resume and total > len(todo) else ""),
          file=sys.stderr)
    if not todo:
        print_encode_hint(out_dir, shaped.fps, pattern=pattern)
        return []

    # Push the opening frame first: it makes the viewer ready (needed for any screenshot
    # reply) and starts the meshes loading.
    opening = shaped.at(todo[0][1])
    capture = dict(shaped=shaped, todo=todo, out_dir=out_dir, size=(width, height),
                   pattern=pattern, prefetch=prefetch, refresh_timeout=refresh_timeout,
                   give_up_after=give_up_after, size_attempts=size_attempts,
                   verbose=verbose, progress_interval=progress_interval)

    if viewer is not None:                       # a caller driving its own viewer
        viewer.set_state(opening)
        return _capture(viewer, **capture)

    if browser not in ("auto", "headless", "none"):
        raise RenderProblem(f"browser must be 'auto', 'headless' or 'none', got {browser!r}")
    executable = None if browser == "none" else find_browser(browser_binary)
    if browser == "headless" and executable is None:
        raise RenderProblem(
            f"browser='headless' but none was found (looked for {', '.join(BROWSERS)}). Pass "
            f"browser_binary=..., or browser='none' to open one yourself.")

    viewer = open_viewer(bind=bind, port=port)
    if executable is None:
        wait_for_browser(viewer, opening, timeout=connect_timeout)
        return _capture(viewer, **capture)
    with headless_browser(viewer.get_viewer_url(), size=(width, height),
                          binary=browser_binary, gpu=gpu) as alive:
        wait_for_browser(viewer, opening, timeout=connect_timeout, launched=True, watch=alive)
        return _capture(viewer, **capture)


def _capture(viewer, *, shaped, todo, out_dir, size, pattern, prefetch, refresh_timeout,
             give_up_after, size_attempts, verbose, progress_interval) -> list[str]:
    """The loop itself, once something is drawing. Shared by every browser arrangement."""
    from neuroglancer.tool.screenshot import CaptureScreenshotRequest, capture_screenshots

    width, height = size
    refreshes = {"n": 0}

    def refresh_browser_callback():
        refreshes["n"] += 1
        if refreshes["n"] > give_up_after:
            raise RenderProblem(
                f"the browser sent no progress for {refresh_timeout:g}s, {give_up_after} times "
                f"running. Frames already written are kept — reload {viewer.get_viewer_url()} "
                f"and re-run to resume")
        print(f"no progress for {refresh_timeout:g}s ({refreshes['n']}/{give_up_after}) — "
              f"reload the tab if it has stopped responding", file=sys.stderr)

    progress = _Progress(len(todo), verbose=verbose, min_interval=progress_interval,
                         description=f"rendering {width}x{height}")
    queue = _RequestQueue(shaped, todo, size=(width, height), attempts=size_attempts,
                          request_type=CaptureScreenshotRequest, on_write=progress.advance)

    # capture_screenshots prints to stdout, including from the tornado thread, so the
    # redirect is held for the whole render. Unless verbose, its lines are kept out of the
    # terminal (they would garble a progress bar) and printed only if the render fails.
    tail = _Tail()
    try:
        with contextlib.redirect_stdout(sys.stderr if verbose else tail):
            capture_screenshots(viewer, iter(queue), refresh_browser_callback,
                                refresh_timeout, num_to_prefetch=prefetch)
    except BaseException:
        progress.close()
        if tail.lines:
            print("\n--- last lines from the capture loop ---", file=sys.stderr)
            for line in tail.lines:
                print(f"    {line}", file=sys.stderr)
        raise
    finally:
        progress.close()
        # Restore the responsive layout: a leftover `viewerSize` would show the viewer
        # scaled to fit the window, which looks like a rendering fault.
        with contextlib.suppress(Exception):
            with viewer.config_state.txn() as s:
                s.viewer_size = None
                s.show_ui_controls = True

    print(f"wrote {len(queue.written)} frame(s) to {out_dir}", file=sys.stderr)
    print_encode_hint(out_dir, shaped.fps, pattern=pattern)
    return queue.written


def print_encode_hint(out_dir: str, fps: float, *, pattern: str = "frame_%05d.png",
                      file=None) -> None:
    """Print the encode command, stating plainly that nothing has been encoded yet."""
    print(f"\nRendering is done — the output is a directory of PNG frames.\n"
          f"NOTHING HAS BEEN ENCODED: run this yourself to get a video.\n"
          f"(ffmpeg is often not on PATH — on a cluster it is usually a module.)\n\n"
          f"    {ffmpeg_command(out_dir, fps, pattern=pattern)}\n",
          file=file or sys.stderr)


#: H.264 in ``yuv420p``, the pixel format every player accepts, at a visually lossless CRF.
#: The same settings as neu-draw's ``video.ENCODE_ARGS``.
VIDEO_CODEC, PIXEL_FORMAT, DEFAULT_CRF = "libx264", "yuv420p", 18


def ffmpeg_command(out_dir: str, fps: float, *, output: str | None = None,
                   crf: int = DEFAULT_CRF, pattern: str = "frame_%05d.png") -> str:
    """The command that turns the frames into an mp4, quoted for a shell. Printed, never run.

    Not run, because ffmpeg is often not on ``PATH`` (on a cluster it is usually a module),
    and because re-encoding should never mean re-rendering. Keep ``-pix_fmt yuv420p``:
    without it the video encodes but does not play in QuickTime, PowerPoint or Slack.
    """
    import shlex

    dest = output or os.path.join(out_dir, "animation.mp4")
    return shlex.join(["ffmpeg", "-y", "-framerate", f"{fps:g}",
                       "-i", os.path.join(out_dir, pattern), "-c:v", VIDEO_CODEC,
                       "-pix_fmt", PIXEL_FORMAT, "-crf", str(crf), dest])
