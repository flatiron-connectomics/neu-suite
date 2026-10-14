"""The render driver, against a fake viewer: state shaping, frames on disk, resume, the
wrong-size retry, waiting for a browser, progress output, the ffmpeg command, and the
headless launcher.
"""

import contextlib
import json
import os
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import pytest

from neu_glance.movie.animate import Timeline

pytest.importorskip("neuroglancer", reason="the serve extra is not installed")

from neu_glance.movie.rendering import (  # noqa: E402
    CHROME_OFF, PNG_MAGIC, RenderProblem, _RequestQueue, _Tail, _write_atomic,
    ffmpeg_command, frame_path, is_written, pending_frames, print_encode_hint, record,
    render_state, stale_frames, wait_for_browser,
    GPU_FLAGS, find_browser, headless_browser)


def _state(**extra):
    state = {"dimensions": {"x": [8e-9, "m"], "y": [8e-9, "m"], "z": [8e-9, "m"]},
             "layers": [{"type": "segmentation", "name": "seg",
                         "source": "precomputed://s3://my-bucket/seg",
                         "segments": ["1", "2"]}],
             "layout": "3d", "projectionScale": 12000.0}
    state.update(extra)
    return state


def _timeline(fps=10.0):
    tl = Timeline(_state(), fps=fps)
    tl.tween(layer="seg", at=0.0, seconds=1.0, objectAlpha=0.0)
    return tl


class _Reply:
    """What the browser posts back: PNG bytes plus the size it actually rendered."""

    def __init__(self, width, height, image=PNG_MAGIC + b"fake"):
        self.width, self.height, self.image = width, height, image


class _Config:
    """The mutable half of a viewer's config_state, as `capture_screenshots` uses it."""

    def __init__(self):
        self.prefetch = []
        self.show_ui_controls = None
        self.show_panel_borders = None
        self.viewer_size = None


class _FakeViewer:
    """A viewer that answers instantly, so the capture loop can run with no browser.

    `reply_size=None` echoes whatever size was asked for; a fixed pair forces a mismatch.
    """

    def __init__(self, *, url="http://host:1/v/tok/", reply_size=None, answers=True,
                 statistics=()):
        self.url, self.reply_size, self.answers = url, reply_size, answers
        self.statistics = list(statistics)
        self.config = _Config()
        self.states = []
        self.probes = 0
        self.config_touched = 0

    def get_viewer_url(self):
        return self.url

    def set_state(self, state):
        self.states.append(state)

    @property
    def config_state(self):
        viewer = self

        class _Handle:
            @staticmethod
            @contextlib.contextmanager
            def txn():
                viewer.config_touched += 1
                yield viewer.config
        return _Handle

    def async_screenshot(self, callback, include_depth=False, statistics_callback=None):
        self.probes += 1
        # `answers` may be a count: the two-stage wait probes once to prove attachment and
        # again for the loaded scene, and those fail differently.
        answers = (self.probes <= self.answers if isinstance(self.answers, int)
                   and not isinstance(self.answers, bool) else self.answers)
        for loaded, total, downloading in self.statistics:
            if statistics_callback is not None:
                statistics_callback(SimpleNamespace(total=SimpleNamespace(
                    visible_chunks_gpu_memory=loaded, visible_chunks_total=total,
                    visible_chunks_downloading=downloading)))
        if not answers:
            return
        size = self.reply_size or self.config.viewer_size or (64, 64)
        callback(SimpleNamespace(screenshot=_Reply(*size)))


# --------------------------------------------------------------------------- #
# shaping the state
# --------------------------------------------------------------------------- #
def test_the_render_state_switches_off_every_panel_and_the_navigation_furniture():
    """Panels, axis lines and default annotations (bounding boxes) are all switched off."""
    shaped = render_state(_state())
    for key in CHROME_OFF:
        assert shaped[key]["visible"] is False
    assert shaped["showAxisLines"] is False
    assert shaped["showDefaultAnnotations"] is False


def test_the_render_state_keeps_the_scene_alone():
    """It shapes the viewer's furniture, not what is being rendered."""
    shaped = render_state(_state())
    assert shaped["layout"] == "3d"
    assert shaped["projectionScale"] == 12000.0
    assert [lyr["name"] for lyr in shaped["layers"]] == ["seg"]


def test_the_render_state_merges_into_an_existing_panel_block_rather_than_replacing_it():
    """`selectedLayer` carries a layer name as well as a visibility."""
    shaped = render_state(_state(selectedLayer={"layer": "seg", "visible": True, "size": 364}))
    assert shaped["selectedLayer"] == {"layer": "seg", "visible": False, "size": 364}


def test_a_tool_palette_is_dropped_because_it_names_layers_a_split_has_emptied():
    shaped = render_state(_state(toolPalettes={"All controls": {"row": 1, "query": "+"}}))
    assert "toolPalettes" not in shaped


def test_the_render_state_never_mutates_the_state_it_was_given():
    state = _state()
    snapshot = json.dumps(state)
    render_state(state)
    assert json.dumps(state) == snapshot


def test_the_memory_knobs_are_left_alone_unless_asked_for():
    """GPU memory limits are left at the viewer's defaults unless given."""
    assert "gpuMemoryLimit" not in render_state(_state())
    assert render_state(_state(), gpu_memory_limit=8 << 30)["gpuMemoryLimit"] == 8 << 30


# --------------------------------------------------------------------------- #
# frames on disk
# --------------------------------------------------------------------------- #
def test_a_frame_pattern_with_no_integer_field_is_refused():
    """ffmpeg reads a sequence through the printf field; without one it reads a single file."""
    with pytest.raises(RenderProblem, match="integer field"):
        frame_path("/tmp", 3, "frame.png")


def test_a_truncated_frame_is_NOT_treated_as_done(tmp_path):
    """An empty or torn PNG is not done, though `os.path.exists` would say so."""
    good, empty, junk = (tmp_path / n for n in ("g.png", "e.png", "j.png"))
    good.write_bytes(PNG_MAGIC + b"...")
    empty.write_bytes(b"")
    junk.write_bytes(b"\x89PNG")                      # a real mid-write truncation
    assert is_written(str(good))
    assert not is_written(str(empty))
    assert not is_written(str(junk))
    assert not is_written(str(tmp_path / "absent.png"))


def test_a_frame_is_written_atomically_so_a_kill_leaves_no_half_png(tmp_path):
    path = str(tmp_path / "f.png")
    _write_atomic(path, PNG_MAGIC + b"body")
    assert is_written(path)
    assert not os.path.exists(path + ".tmp")


def test_resume_skips_frames_that_are_already_written_and_keeps_their_numbering(tmp_path):
    """Resume keeps the global frame numbering."""
    tl = _timeline()
    for i in (0, 1, 2):
        _write_atomic(frame_path(str(tmp_path), i), PNG_MAGIC + b"x")
    todo = pending_frames(tl, str(tmp_path))
    assert [i for i, _, _ in todo] == list(range(3, tl.frame_count))
    assert todo[0][1] == pytest.approx(0.3)
    assert not pending_frames(tl, str(tmp_path), resume=False)[0][0]     # index 0 back again


def test_frames_left_over_past_the_END_of_the_timeline_are_REMOVED(tmp_path, capsys):
    """ffmpeg reads the whole numbered sequence, so a longer earlier run's tail would be
    appended to the video."""
    tl = _timeline()
    for i in range(tl.frame_count + 5):                  # a longer previous run
        _write_atomic(frame_path(str(tmp_path), i), PNG_MAGIC + b"x")
    record(tl, str(tmp_path), size=(64, 32), viewer=_FakeViewer())
    survivors = sorted(p.name for p in tmp_path.glob("frame_*.png"))
    assert len(survivors) == tl.frame_count
    assert "removing 5 frame(s) past the end" in capsys.readouterr().err


def test_stale_detection_reads_the_index_out_of_the_PATTERN(tmp_path):
    """Stale frames are found through a custom frame pattern too."""
    for i in (0, 1, 9, 10):
        (tmp_path / f"shot-{i:03d}.png").write_bytes(PNG_MAGIC)
    (tmp_path / "notes.txt").write_bytes(b"not a frame")
    stale = stale_frames(str(tmp_path), 9, "shot-%03d.png")
    assert [os.path.basename(p) for p in stale] == ["shot-009.png", "shot-010.png"]


def test_resuming_onto_a_CHANGED_timeline_re_renders_instead_of_mixing_two(tmp_path, capsys):
    """Frames from an earlier version of the timeline look valid; the saved timeline.json is
    what detects them."""
    first = _timeline()
    record(first, str(tmp_path), size=(64, 32), viewer=_FakeViewer())
    for path in tmp_path.glob("frame_*.png"):
        path.write_bytes(PNG_MAGIC + b"old")             # mark them as the previous render

    changed = Timeline(_state(), fps=first.fps)
    changed.tween(layer="seg", at=0.0, seconds=1.0, objectAlpha=0.5)     # a different fade
    written = record(changed, str(tmp_path), size=(64, 32), viewer=_FakeViewer(), resume=True)
    assert "differs from the one these frames were rendered with" in capsys.readouterr().err
    assert len(written) == changed.frame_count
    assert b"old" not in (tmp_path / "frame_00000.png").read_bytes()


def test_resuming_an_UNCHANGED_timeline_still_skips_the_work(tmp_path):
    """An unchanged timeline resumes with nothing to do."""
    tl = _timeline()
    record(tl, str(tmp_path), size=(64, 32), viewer=_FakeViewer())
    assert record(tl, str(tmp_path), size=(64, 32), viewer=_FakeViewer(), resume=True) == []


def test_rendering_fresh_rewrites_frames_that_are_already_there(tmp_path):
    """`resume=False` re-renders existing frames."""
    tl = _timeline()
    for i in range(tl.frame_count):
        _write_atomic(frame_path(str(tmp_path), i), PNG_MAGIC + b"old")
    assert record(tl, str(tmp_path), size=(64, 32), viewer=_FakeViewer(), resume=True) == []
    written = record(tl, str(tmp_path), size=(64, 32), viewer=_FakeViewer(), resume=False)
    assert len(written) == tl.frame_count
    assert b"old" not in (tmp_path / "frame_00000.png").read_bytes()


# --------------------------------------------------------------------------- #
# the capture queue
# --------------------------------------------------------------------------- #
def _queue(tmp_path, *, size=(64, 32), attempts=2, frames=2):
    from neuroglancer.tool.screenshot import CaptureScreenshotRequest
    tl = _timeline()
    items = [(i, i / tl.fps, frame_path(str(tmp_path), i)) for i in range(frames)]
    return _RequestQueue(tl, items, size=size, attempts=attempts,
                         request_type=CaptureScreenshotRequest)


def test_a_reply_of_the_right_size_is_written(tmp_path):
    q = _queue(tmp_path)
    request = next(iter(q))
    request.response_callback(_Reply(64, 32))
    assert is_written(q.written[0])


def test_a_wrong_sized_reply_is_NOT_written_and_the_frame_is_requeued(tmp_path):
    """`async_screenshot` has no size retry of its own, so a too-small window would yield
    frames of two sizes."""
    q = _queue(tmp_path)
    request = next(iter(q))
    request.response_callback(_Reply(60, 30))
    assert q.written == []
    assert not os.path.exists(frame_path(str(tmp_path), 0))
    assert q.queue[0][0] == 0                                # back at the front


def test_a_frame_that_keeps_coming_back_wrong_sized_gives_up_and_says_why(tmp_path):
    q = _queue(tmp_path, attempts=2)
    for _ in range(2):
        next(iter(q)).response_callback(_Reply(60, 30))
    with pytest.raises(RenderProblem, match="window is probably smaller"):
        next(iter(q)).response_callback(_Reply(60, 30))


def test_the_config_callback_asks_for_the_requested_viewer_size(tmp_path):
    class _Config:
        viewer_size = None
    config = _Config()
    next(iter(_queue(tmp_path, size=(800, 600)))).config_callback(config)
    assert config.viewer_size == (800, 600)


def test_each_request_carries_ITS_OWN_frames_state(tmp_path):
    """Prefetch buffers several requests, so each must hold its own state."""
    q = _queue(tmp_path, frames=3)
    it = iter(q)
    states = [next(it).state for _ in range(3)]
    # frame 0 is the tween's start, where the layer carries no objectAlpha (default 1.0)
    alphas = [next(lyr for lyr in s["layers"] if lyr["name"] == "seg").get("objectAlpha", 1.0)
              for s in states]
    assert len(set(alphas)) == 3


# --------------------------------------------------------------------------- #
# record: the parts reachable without a browser
# --------------------------------------------------------------------------- #
def test_an_odd_width_or_height_is_refused_with_the_reason(tmp_path):
    """libx264 with yuv420p needs even dimensions."""
    with pytest.raises(RenderProblem, match="even"):
        record(_timeline(), str(tmp_path), size=(1921, 1080))
    with pytest.raises(RenderProblem, match="yuv420p"):
        record(_timeline(), str(tmp_path), size=(1920, 1081))


def test_a_fully_rendered_directory_needs_no_browser_at_all(tmp_path, capsys):
    """A finished render opens no viewer and just prints the encode hint."""
    tl = _timeline()
    for i in range(tl.frame_count):
        _write_atomic(frame_path(str(tmp_path), i), PNG_MAGIC + b"x")
    assert record(tl, str(tmp_path), size=(64, 32)) == []
    assert "ffmpeg" in capsys.readouterr().err


def test_record_writes_the_timeline_beside_the_frames(tmp_path):
    """timeline.json beside the frames makes the render reproducible."""
    tl = _timeline()
    for i in range(tl.frame_count):
        _write_atomic(frame_path(str(tmp_path), i), PNG_MAGIC + b"x")
    record(tl, str(tmp_path), size=(64, 32))
    written = json.loads((tmp_path / "timeline.json").read_text())
    assert written["fps"] == tl.fps
    assert Timeline.from_json(written).frame_count == tl.frame_count


def test_the_state_is_shaped_ONCE_before_interpolation_not_per_frame(tmp_path):
    """The saved base is shaped; the caller's timeline is not."""
    tl = _timeline()
    for i in range(tl.frame_count):
        _write_atomic(frame_path(str(tmp_path), i), PNG_MAGIC + b"x")
    record(tl, str(tmp_path), size=(64, 32))
    written = json.loads((tmp_path / "timeline.json").read_text())
    assert written["base"]["showAxisLines"] is False          # shaped
    assert tl.base.get("showAxisLines") is None               # the caller's is untouched


# --------------------------------------------------------------------------- #
# waiting for a browser
# --------------------------------------------------------------------------- #
def test_the_probe_PUSHES_A_STATE_because_an_empty_viewer_is_never_ready(tmp_path):
    """A viewer with no dimensions never becomes ready and never answers a screenshot, so
    the attach probe pushes dimensions with no layers, then the real scene."""
    viewer = _FakeViewer()
    state = _state()
    wait_for_browser(viewer, state)
    assert len(viewer.states) == 2, "an attach probe, then the real scene"
    probe, real = viewer.states
    assert probe["layers"] == [] and probe["dimensions"] == state["dimensions"]
    assert real == state
    assert viewer.probes == 2


def test_waiting_for_a_browser_changes_NOTHING_about_the_viewers_config():
    """The probe leaves the config alone: a small `viewerSize` would be CSS-scaled up to fill
    the window and look like broken data to anyone viewing the link."""
    viewer = _FakeViewer()
    wait_for_browser(viewer, _state())
    assert viewer.config_touched == 0
    assert viewer.config.viewer_size is None
    assert viewer.config.show_ui_controls is None


def test_a_browser_that_never_attaches_times_out_and_names_the_bind_hint():
    """A loopback URL is unreachable from another machine; the timeout says so."""
    viewer = _FakeViewer(url="http://127.0.0.1:8080/v/tok/", answers=False)
    with pytest.raises(RenderProblem, match="loopback"):
        wait_for_browser(viewer, _state(), timeout=0.3, notify_every=0.1)


def test_the_bind_hint_is_omitted_when_the_url_already_names_a_host():
    viewer = _FakeViewer(url="http://workstation.example.org:8080/v/tok/", answers=False)
    with pytest.raises(RenderProblem, match="no browser attached") as excinfo:
        wait_for_browser(viewer, _state(), timeout=0.3, notify_every=0.1)
    assert "loopback" not in str(excinfo.value)


def test_A_SLOW_SCENE_IS_NOT_REPORTED_AS_A_MISSING_BROWSER():
    """Attachment and scene loading are separate waits, so slow loading is reported as such."""
    viewer = _FakeViewer(answers=1, statistics=[(300, 1900, 12)])
    with pytest.raises(RenderProblem, match="stopped loading") as excinfo:
        wait_for_browser(viewer, _state(), timeout=5.0, notify_every=0.1, stall_timeout=0.25)
    assert "300/1900" in str(excinfo.value)
    assert "no browser attached" not in str(excinfo.value)


def test_loading_progress_is_reported_rather_than_a_bare_wait(capsys, monkeypatch):
    """Chunk counts are printed while waiting (as lines, without tqdm)."""
    monkeypatch.setitem(sys.modules, "tqdm.auto", None)
    viewer = _FakeViewer(answers=1, statistics=[(300, 1900, 12)])
    with pytest.raises(RenderProblem):
        wait_for_browser(viewer, _state(), timeout=5.0, notify_every=0.05, stall_timeout=5.0)
    err = capsys.readouterr().err
    assert "browser attached" in err and "300/1900 chunks" in err


def test_a_STALL_is_no_progress_not_silence(capsys):
    """A stall is no progress, not silence: N-1/N chunks with 0 downloading is a segment with
    no mesh, and the error names the fix."""
    viewer = _FakeViewer(answers=1, statistics=[(6476, 6477, 0)])
    with pytest.raises(RenderProblem, match="stopped loading") as excinfo:
        wait_for_browser(viewer, _state(), timeout=5.0, notify_every=0.05, stall_timeout=0.2)
    message = str(excinfo.value)
    assert "6476/6477" in message and "0 downloading" in message
    assert "1 chunk(s) short" in message
    assert "segments_with_meshes" in message, "it must name the fix, not just the symptom"


# --------------------------------------------------------------------------- #
# the capture loop, end to end against a fake viewer
# --------------------------------------------------------------------------- #
def test_record_renders_every_frame(tmp_path):
    """Every frame is rendered and written."""
    tl = _timeline()
    written = record(tl, str(tmp_path), size=(64, 32), viewer=_FakeViewer())
    assert len(written) == tl.frame_count
    assert all(is_written(p) for p in written)


def test_record_restores_the_viewer_even_when_the_render_FAILS(tmp_path):
    """A failed render still resets `viewerSize` and the UI, leaving a usable tab."""
    viewer = _FakeViewer(reply_size=(11, 11))          # never the requested size
    with pytest.raises(RenderProblem, match="window is probably smaller"):
        record(_timeline(), str(tmp_path), size=(64, 32), viewer=viewer, size_attempts=1)
    assert viewer.config.viewer_size is None
    assert viewer.config.show_ui_controls is True


def test_record_pushes_the_opening_frame_before_capturing(tmp_path):
    """The shaped opening frame is pushed first, so loading starts before capture."""
    viewer = _FakeViewer()
    tl = _timeline()
    record(tl, str(tmp_path), size=(64, 32), viewer=viewer)
    shaped = Timeline(base=render_state(tl.base), fps=tl.fps, tweens=list(tl.tweens))
    assert viewer.states[0] == shaped.at(0.0)
    assert viewer.states[0]["showAxisLines"] is False


# --------------------------------------------------------------------------- #
# progress reporting
# --------------------------------------------------------------------------- #
def test_a_progress_bar_SWALLOWS_the_capture_loops_per_frame_lines(tmp_path, capsys):
    """Quiet mode drops the capture loop's lines, which would land on tqdm's bar line."""
    record(_timeline(), str(tmp_path), size=(64, 32), viewer=_FakeViewer(), verbose=False)
    err = capsys.readouterr().err
    assert "Requesting screenshot" not in err
    assert "frame" in err                       # the bar itself still says something


def test_verbose_keeps_every_line_and_shows_no_bar(tmp_path, capsys):
    record(_timeline(), str(tmp_path), size=(64, 32), viewer=_FakeViewer(), verbose=True)
    err = capsys.readouterr().err
    assert err.count("Requesting screenshot") == _timeline().frame_count
    assert "frame/s]" not in err                # no tqdm bar


def test_a_FAILED_quiet_render_still_prints_the_last_lines_it_swallowed(tmp_path, capsys):
    """A failed quiet render prints the capture loop's last lines."""
    viewer = _FakeViewer(reply_size=(11, 11))
    with pytest.raises(RenderProblem):
        record(_timeline(), str(tmp_path), size=(64, 32), viewer=viewer, size_attempts=1,
               verbose=False)
    err = capsys.readouterr().err
    assert "last lines from the capture loop" in err
    assert "Requesting screenshot" in err


def test_without_tqdm_the_fallback_is_periodic_lines_with_an_ETA(tmp_path, capsys, monkeypatch):
    """tqdm is optional; without it, progress is periodic lines with an ETA."""
    monkeypatch.setitem(sys.modules, "tqdm.auto", None)
    record(_timeline(), str(tmp_path), size=(64, 32), viewer=_FakeViewer(), verbose=False,
           progress_interval=0.0)
    err = capsys.readouterr().err
    assert "s/frame" in err and "min left" in err
    assert "frame/s]" not in err


def test_the_tail_buffer_is_BOUNDED_so_a_stalled_render_cannot_grow_it():
    """The tail buffer stays bounded, since a stalled render logs indefinitely."""
    tail = _Tail(keep=3)
    for i in range(500):
        tail.write(f"line {i}\n")
    assert list(tail.lines) == ["line 497", "line 498", "line 499"]


# --------------------------------------------------------------------------- #
# the ffmpeg command
# --------------------------------------------------------------------------- #
def test_the_ffmpeg_command_names_the_pattern_the_frame_rate_and_the_PIXEL_FORMAT():
    """Without `-pix_fmt yuv420p` the mp4 will not play in QuickTime, PowerPoint or Slack.
    No site-specific module loading in a command printed to users."""
    cmd = ffmpeg_command("/out", 24.0)
    assert "-pix_fmt yuv420p" in cmd
    assert "-framerate 24" in cmd
    assert os.path.join("/out", "frame_%05d.png") in cmd
    assert cmd.endswith(os.path.join("/out", "animation.mp4"))
    assert "module" not in cmd and "nix" not in cmd


def test_the_ffmpeg_command_survives_a_path_with_spaces():
    """It is pasted into a shell, so a path is one argument however it is spelled."""
    import shlex

    argv = shlex.split(ffmpeg_command("/my renders/take 1", 30.0))
    assert argv[argv.index("-i") + 1] == "/my renders/take 1/frame_%05d.png"
    assert argv[-1] == "/my renders/take 1/animation.mp4"


def test_the_encode_hint_says_OUT_LOUD_that_nothing_was_encoded(capsys):
    """The hint must read as an instruction: nothing was encoded."""
    print_encode_hint("/out", 30.0)
    err = capsys.readouterr().err
    assert "NOTHING HAS BEEN ENCODED" in err
    assert "run this yourself" in err
    assert ffmpeg_command("/out", 30.0) in err


# --------------------------------------------------------------------------- #
# dependencies
# --------------------------------------------------------------------------- #
def test_the_rendering_path_needs_no_selenium():
    """Neither importing nor finding a browser needs selenium (the webdriver route would)."""
    code = ("import neu_glance.movie.rendering as r, sys; "
            "r.find_browser('nope'); "
            "assert 'selenium' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_no_module_imports_neuroglancer_at_import_time():
    """Every neu_glance module imports without loading neuroglancer, so `--help` and the
    pure code paths work without the serve extra."""
    code = ("import importlib, pkgutil, sys, neu_glance; "
            "[importlib.import_module('neu_glance.' + m.name) "
            "for m in pkgutil.iter_modules(neu_glance.__path__)]; "
            "assert 'neuroglancer' not in sys.modules, sorted(sys.modules)")
    subprocess.run([sys.executable, "-c", code], check=True)


# --------------------------------------------------------------------------- #
# launching a browser
# --------------------------------------------------------------------------- #
def test_finding_a_browser_prefers_an_explicit_binary_over_the_search(tmp_path):
    fake = tmp_path / "my-chrome"
    fake.write_text("#!/bin/sh\n")
    assert find_browser(str(fake)) == str(fake)
    assert find_browser("definitely-not-a-browser-anywhere") is None


def test_the_launcher_cleans_up_its_throwaway_profile(tmp_path):
    """Each render gets its own profile directory (so renders do not collide), removed after."""
    before = set(os.listdir(tempfile.gettempdir()))
    with headless_browser("http://localhost:1/v/x/", size=(64, 64), binary="/bin/sleep",
                          gpu=False) as alive:
        assert alive() is None                      # sleep is still running
    new = [n for n in set(os.listdir(tempfile.gettempdir())) - before
           if n.startswith("neu-glance-render-")]
    assert new == []


def test_the_launcher_sizes_the_window_to_the_render_with_the_gpu_on(monkeypatch):
    """A window smaller than the frame makes every screenshot the wrong size; software
    rendering is about 10x slower, so the GPU flags are on by default."""
    seen = {}

    class _Fake:
        def __init__(self, command, **kw):
            seen["command"] = command
            self.pid = 1

        def terminate(self): pass

        def wait(self, timeout=None): return 0

    monkeypatch.setattr(subprocess, "Popen", _Fake)
    with headless_browser("http://x/", size=(1280, 720), binary="/bin/true"):
        pass
    assert "--window-size=1280,720" in seen["command"]
    assert seen["command"][-1] == "http://x/"
    assert all(flag in seen["command"] for flag in GPU_FLAGS)


def test_an_unknown_browser_mode_is_refused(tmp_path):
    with pytest.raises(RenderProblem, match="'auto', 'headless' or 'none'"):
        record(_timeline(), str(tmp_path), size=(64, 32), browser="chrome-ish")


def test_insisting_on_headless_with_none_installed_names_the_alternative(tmp_path):
    with pytest.raises(RenderProblem, match="browser='none'"):
        record(_timeline(), str(tmp_path), size=(64, 32), browser="headless",
               browser_binary="definitely-not-a-browser-anywhere")


def test_a_caller_supplying_its_own_viewer_launches_nothing(tmp_path, monkeypatch):
    """With `viewer=` given, no browser is launched."""
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("launched a browser"))
    assert record(_timeline(), str(tmp_path), size=(64, 32), viewer=_FakeViewer())


def test_a_LAUNCHED_browser_is_not_announced_as_something_to_open(capsys):
    """A launched browser is not announced as a link to open."""
    viewer = _FakeViewer()
    wait_for_browser(viewer, _state(), launched=True)
    err = capsys.readouterr().err
    assert "open this in a browser" not in err
    assert "headless browser" in err


def test_a_launched_browser_that_never_attaches_blames_the_RIGHT_thing():
    """A launched browser that never attaches is blamed on the process, not on a tab."""
    viewer = _FakeViewer(url="http://127.0.0.1:8080/v/tok/", answers=False)
    with pytest.raises(RenderProblem, match="died or cannot reach") as excinfo:
        wait_for_browser(viewer, _state(), timeout=0.3, notify_every=0.1, launched=True)
    assert "loopback" not in str(excinfo.value)


def test_a_browser_that_DIES_at_startup_is_reported_at_once_not_after_the_timeout():
    """A browser that exits at startup is reported at once, naming `gpu=False` as the likely
    fix. `/bin/false` stands in for chrome failing to start."""
    with headless_browser("http://localhost:1/v/x/", size=(64, 64), binary="/bin/false",
                          gpu=True) as alive:
        for _ in range(50):
            stopped = alive()
            if stopped:
                break
            time.sleep(0.05)
    assert stopped and "exited with code 1" in stopped
    assert "gpu=False" in stopped                  # ...and names the likeliest cause


def test_a_dead_browser_short_circuits_the_wait():
    viewer = _FakeViewer(answers=False)
    with pytest.raises(RenderProblem, match="exited with code"):
        wait_for_browser(viewer, _state(), timeout=600.0, notify_every=0.05, launched=True,
                         watch=lambda: "chrome exited with code 1")


def test_the_launched_browser_gets_a_PRIVATE_runtime_dir(monkeypatch):
    """A long-lived tmux session can carry an XDG_RUNTIME_DIR that no longer exists, so the
    browser gets a fresh one (removed afterwards) and no stale DBus address."""
    seen = {}

    class _Fake:
        def __init__(self, command, **kw):
            seen.update(env=kw.get("env"), command=command)
            self.pid = 1

        def poll(self): return None

        def terminate(self): pass

        def wait(self, timeout=None): return 0

    monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/stale-and-deleted")
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "unix:path=/gone")
    monkeypatch.setattr(subprocess, "Popen", _Fake)
    with headless_browser("http://x/", size=(64, 64), binary="/bin/true"):
        runtime = seen["env"]["XDG_RUNTIME_DIR"]
        assert runtime != "/run/user/stale-and-deleted"
        assert os.path.isdir(runtime), "it must EXIST, not merely differ"
    assert "DBUS_SESSION_BUS_ADDRESS" not in seen["env"]
    assert not os.path.exists(runtime), "and go with the profile afterwards"
