"""Orbits: the angle schedule and camera poses (pure), then frames from a real render."""

import json
import os
import shutil
import stat
import sys

import numpy as np
import pytest

from neu_draw import orbit

#: A real ffmpeg, looked up before the fixture below hides it — for the one test that
#: encodes for real. Skipped where there is none.
REAL_FFMPEG = os.environ.get("NEU_DRAW_FFMPEG") or shutil.which("ffmpeg")


@pytest.fixture(autouse=True)
def no_ffmpeg(monkeypatch):
    """Hermetic: no ffmpeg from this machine's PATH, environment or imageio-ffmpeg, so a
    test sees only the ones it installs itself."""
    monkeypatch.delenv("NEU_DRAW_FFMPEG", raising=False)
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)
    keep = [d for d in os.environ.get("PATH", "").split(os.pathsep)
            if d and not os.path.exists(os.path.join(d, "ffmpeg"))]
    monkeypatch.setenv("PATH", os.pathsep.join(keep))


def _fake_ffmpeg(directory, name="ffmpeg", fail=False):
    """An executable that records its argv and writes the output file — or fails."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, name)
    log = os.path.join(directory, "argv.json")
    body = (f"#!{sys.executable}\nimport json, sys\n"
            f"json.dump(sys.argv, open({log!r}, 'w'))\n")
    body += ("sys.stderr.write('Unknown encoder libx264\\n'); sys.exit(1)\n" if fail
             else "open(sys.argv[-1], 'wb').write(b'\\0\\0\\0\\x18ftypisom')\n")
    with open(path, "w") as fh:
        fh.write(body)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
    return path, log


# --------------------------------------------------------------------------- #
# pure
# --------------------------------------------------------------------------- #

def test_a_whole_turn_leaves_out_its_endpoint_so_the_video_loops():
    assert np.degrees(orbit.angles(360, 4)).tolist() == pytest.approx([0, 90, 180, 270])


def test_a_partial_sweep_ends_exactly_where_it_was_asked_to():
    assert np.degrees(orbit.angles(90, 4)).tolist() == pytest.approx([0, 30, 60, 90])


def test_easing_keeps_the_ends_and_moves_the_middle():
    eased = np.degrees(orbit.angles(90, 5, ease="in-out"))
    assert eased[0] == 0 and eased[-1] == pytest.approx(90)
    assert eased[1] < 22.5


def test_a_bad_schedule_is_refused():
    with pytest.raises(ValueError, match="ease"):
        orbit.angles(360, 4, ease="bounce")
    with pytest.raises(ValueError, match="at least one frame"):
        orbit.angles(360, 0)


def test_a_screen_axis_is_read_in_the_cameras_frame():
    identity = (0, 0, 0, 1)
    assert orbit.world_axis("up", identity) == pytest.approx([0, 1, 0])
    tilted = orbit.axis_quaternion((1, 0, 0), np.pi / 2)       # camera pitched up 90°
    assert orbit.world_axis("up", tilted) == pytest.approx([0, 0, 1], abs=1e-12)
    assert orbit.world_axis("z", tilted) == pytest.approx([0, 0, 1])   # data: unmoved


def test_an_axis_that_means_nothing_fails_before_any_frame():
    with pytest.raises(ValueError, match="unknown orbit axis"):
        orbit.world_axis("upward", (0, 0, 0, 1))
    with pytest.raises(ValueError, match="zero vector"):
        orbit.world_axis((0, 0, 0), (0, 0, 0, 1))
    assert orbit.world_axis((0, 0, 5), (0, 0, 0, 1)) == pytest.approx([0, 0, 1])


def test_the_pivot_is_on_the_view_ray_not_at_the_scene_centre():
    """After a pan the scene's centre is off to one side; orbiting it would swing the view
    away from what is on screen."""
    pivot = orbit.pivot((0, 0, 10), (0, 0, 0, 1), (3, 0, 0))
    assert pivot == pytest.approx([0, 0, 0])


def _forward(q):
    return orbit._rotate(np.array([0.0, 0, -1]), np.asarray(q, dtype=float))


@pytest.mark.parametrize("angle", [0.0, 0.3, np.pi / 2, np.pi, 5.0])
def test_the_camera_keeps_looking_at_the_pivot_all_the_way_round(angle):
    start_q = orbit.axis_quaternion((1, 1, 0), 0.4)
    start_p = np.array([2.0, -3.0, 9.0])
    about = orbit.pivot(start_p, start_q, (0, 0, 0))
    axis = orbit.world_axis("up", start_q)
    p, q = orbit.pose(start_p, start_q, about, axis, angle)

    to_pivot = about - p
    assert np.linalg.norm(to_pivot) == pytest.approx(np.linalg.norm(about - start_p))
    assert _forward(q) == pytest.approx(to_pivot / np.linalg.norm(to_pivot), abs=1e-9)


def test_a_full_turn_comes_back_to_the_start():
    q0 = orbit.axis_quaternion((0, 1, 1), 0.7)
    p, q = orbit.pose((1, 2, 3), q0, (0, 0, 0), (0, 0, 1), 2 * np.pi)
    assert p == pytest.approx([1, 2, 3])
    assert abs(np.dot(q, q0)) == pytest.approx(1.0)          # same rotation, up to sign


# --------------------------------------------------------------------------- #
# frames from a real render
# --------------------------------------------------------------------------- #

@pytest.fixture
def has_gpu():
    pytest.importorskip("pygfx", reason="the render extra is not installed")
    pytest.importorskip("imageio")
    import wgpu
    try:
        wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    except Exception as exc:                                    # pragma: no cover
        pytest.skip(f"no wgpu adapter available: {exc}")
    return True


@pytest.fixture
def view(has_gpu):
    from neu_lib import Mesh
    from neu_draw.backends import pygfx as backend
    from neu_draw.scene import build_scene

    # Deliberately asymmetric, so a turn changes the picture.
    verts = np.array([[0.0, 0, 0], [400, 0, 0], [0, 100, 0], [0, 0, 100]])
    mesh = Mesh(verts, np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]), name="L")
    view = backend.show(build_scene(meshes=[mesh], background=(0, 0, 0, 1)),
                        size=(120, 80), canvas="offscreen", pixel_ratio=1.0)
    yield view
    view.close()


def _frames(directory):
    from imageio import v3 as iio

    names = sorted(n for n in os.listdir(directory) if n.startswith("frame_"))
    return [iio.imread(os.path.join(directory, n)).astype(float) for n in names]


def test_frame_zero_is_the_view_you_had(view, tmp_path):
    view.orbit(str(tmp_path), seconds=1, fps=4, supersample=1)
    first = _frames(tmp_path)[0]
    assert np.abs(first - view.snapshot()[..., :3].astype(float)).mean() < 1.0


def test_a_turn_changes_the_picture_and_leaves_the_view_alone(view, tmp_path):
    before = dict(view.camera.get_state())
    lights = [tuple(o.local.position) for o in view.scene.children
              if o.__class__.__name__ == "DirectionalLight"]
    view.orbit(str(tmp_path), degrees=360, seconds=1, fps=4)
    frames = _frames(tmp_path)
    assert len(frames) == 4
    assert np.abs(frames[2] - frames[0]).mean() > 1.0
    after = view.camera.get_state()
    assert np.allclose(after["position"], before["position"])
    assert np.allclose(after["rotation"], before["rotation"])
    assert lights == [tuple(o.local.position) for o in view.scene.children
                      if o.__class__.__name__ == "DirectionalLight"]


def test_the_frames_are_the_requested_size_whatever_the_smoothing(view, tmp_path):
    view.orbit(str(tmp_path), seconds=0.25, fps=4, size=(60, 40), supersample=3)
    assert _frames(tmp_path)[0].shape[:2] == (40, 60)


def test_a_rerun_resumes_and_redoes_a_torn_frame(view, tmp_path):
    first = view.orbit(str(tmp_path), seconds=1, fps=4)
    assert first.rendered == 4
    with open(tmp_path / "frame_00002.png", "wb") as fh:
        fh.write(b"half a fr")                          # killed mid-write, pre-atomic
    again = view.orbit(str(tmp_path), seconds=1, fps=4)
    assert again.rendered == 1


def test_frames_of_a_different_orbit_are_not_spliced_in(view, tmp_path):
    view.orbit(str(tmp_path), seconds=1, fps=4)
    with pytest.raises(FileExistsError, match="DIFFERENT orbit"):
        view.orbit(str(tmp_path), seconds=1, fps=4, axis="z")
    assert view.orbit(str(tmp_path), seconds=1, fps=4, axis="z",
                      overwrite=True).rendered == 4


def test_an_odd_frame_size_is_refused_before_rendering(view, tmp_path):
    with pytest.raises(ValueError, match="even dimensions"):
        view.orbit(str(tmp_path / "x"), size=(61, 40))
    assert not (tmp_path / "x").exists()




# --------------------------------------------------------------------------- #
# finding and running ffmpeg
# --------------------------------------------------------------------------- #

def test_ffmpeg_is_looked_for_in_order(tmp_path, monkeypatch):
    from neu_draw import video

    on_path, _ = _fake_ffmpeg(tmp_path / "bin")
    by_env, _ = _fake_ffmpeg(tmp_path / "env", name="ffmpeg-site")
    explicit, _ = _fake_ffmpeg(tmp_path / "mine", name="ffmpeg-mine")
    assert video.find_ffmpeg() is None

    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    assert video.find_ffmpeg() == on_path
    monkeypatch.setenv("NEU_DRAW_FFMPEG", by_env)
    assert video.find_ffmpeg() == by_env
    assert video.find_ffmpeg(explicit) == explicit


def test_a_named_ffmpeg_that_is_not_there_raises_rather_than_falling_through(
        tmp_path, monkeypatch):
    from neu_draw import video

    _fake_ffmpeg(tmp_path / "bin")
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    with pytest.raises(FileNotFoundError, match="ffmpeg="):
        video.find_ffmpeg(str(tmp_path / "nope"))
    monkeypatch.setenv("NEU_DRAW_FFMPEG", str(tmp_path / "nope"))
    with pytest.raises(FileNotFoundError, match="NEU_DRAW_FFMPEG"):
        video.find_ffmpeg()


def test_imageio_ffmpeg_is_the_last_resort(tmp_path, monkeypatch):
    import types

    from neu_draw import video

    bundled, _ = _fake_ffmpeg(tmp_path / "bundled")
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg",
                        types.SimpleNamespace(get_ffmpeg_exe=lambda: bundled))
    assert video.find_ffmpeg() == bundled


def test_with_an_ffmpeg_the_frames_are_encoded(view, tmp_path):
    exe, log = _fake_ffmpeg(tmp_path / "bin")
    result = view.orbit(str(tmp_path / "spin"), seconds=0.5, fps=8, ffmpeg=exe)
    assert result.video == f"{tmp_path / 'spin'}.mp4" and os.path.exists(result.video)
    argv = json.load(open(log))
    assert argv[argv.index("-framerate") + 1] == "8"
    assert argv[argv.index("-pix_fmt") + 1] == "yuv420p"
    assert argv[argv.index("-i") + 1] == os.path.join(str(tmp_path / "spin"),
                                                     "frame_%05d.png")
    assert "encoded to" in repr(result)


def test_without_one_the_frames_stay_and_the_plain_command_is_shown(view, tmp_path):
    result = view.orbit(str(tmp_path / "spin"), seconds=0.25, fps=8)
    assert result.video is None and len(os.listdir(tmp_path / "spin")) > 1
    text = repr(result)
    assert "ffmpeg -y" in text and "NEU_DRAW_FFMPEG" in text
    assert "module" not in text            # nothing site-specific in the package


def test_insisting_on_an_encode_with_no_ffmpeg_fails_before_rendering(view, tmp_path):
    with pytest.raises(FileNotFoundError, match="no ffmpeg"):
        view.orbit(str(tmp_path / "spin"), encode=True)
    assert not (tmp_path / "spin").exists()


def test_frames_only_never_looks_for_ffmpeg(view, tmp_path, monkeypatch):
    monkeypatch.setenv("NEU_DRAW_FFMPEG", str(tmp_path / "broken"))   # would raise
    assert view.orbit(str(tmp_path / "spin"), seconds=0.25, fps=4,
                      encode=False).video is None


def test_a_failed_encode_reports_ffmpegs_own_words(view, tmp_path):
    exe, _ = _fake_ffmpeg(tmp_path / "bin", fail=True)
    with pytest.raises(RuntimeError, match="Unknown encoder libx264"):
        view.orbit(str(tmp_path / "spin"), seconds=0.25, fps=4, ffmpeg=exe)


@pytest.mark.skipif(REAL_FFMPEG is None, reason="no ffmpeg on this machine")
def test_a_real_encode_makes_an_mp4(view, tmp_path):
    result = view.orbit(str(tmp_path / "spin"), seconds=0.5, fps=8, size=(64, 48),
                        ffmpeg=REAL_FFMPEG)
    with open(result.video, "rb") as fh:
        assert fh.read(12)[4:8] == b"ftyp"
