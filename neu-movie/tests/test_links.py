"""The inspection links: everything, then each beat, and nothing the movie does not show."""

import numpy as np
import pytest

from conftest import make_spec
from neu_movie import compose, links, schedule
from neu_movie.palette import mute

pytest.importorskip("neu_glance")

WORLD = {"volume": "s3://my-bucket/seg", "unit": 8.0,
         "ids": {"A": ["1"], "B": ["2", "3"], "C": ["4"]},
         "colours": {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c"},
         "roi_volume": "s3://my-bucket/seg/rois", "head_volume": "s3://my-bucket/seg/rois/head",
         "regions": {"LA": {"id": "2", "color": "#1b9e77"}, "ME": {"id": "3", "color": "#e7298a"}}}


def spec(mode="accumulate", **raw):
    return make_spec([{"title": "one", "name": "A", "ids": [1], "regions": ["LA"]},
                      {"name": "B", "ids": [2, 3]},
                      {"title": "three", "name": "C", "ids": [4], "regions": ["ME", "LA"]}],
                     mode=mode, neuropils={"library": "lib", "path": "rois"},
                     head={"library": "lib", "id": 95},
                     timing={"start": 0.5, "every": 2.0, "fade": 1.0, "hold": 1.0}, **raw)


def layers(state):
    return {lyr["name"]: lyr for lyr in state["layers"]}


def test_everything_first_then_each_beat_with_what_is_on_screen():
    out = links.states(spec(), WORLD)
    assert [t for t, _ in out] == ["everything", "one", "B", "three"]
    everything = layers(out[0][1])
    assert set(everything) == {"A", "B", "C", "neuropils", "head"}
    assert everything["B"]["segments"] == ["2", "3"]
    assert everything["B"]["segmentDefaultColor"] == "#ff7f0e"
    assert everything["B"]["source"] == "precomputed://s3://my-bucket/seg"
    assert everything["neuropils"]["segments"] == ["2", "3"]
    assert everything["neuropils"]["segmentColors"] == {"2": mute("#1b9e77", 0.5),
                                                        "3": mute("#e7298a", 0.5)}
    assert everything["head"]["segments"] == ["95"]
    # Accumulating: a beat shows everything so far, and only its sets' neuropils.
    assert set(layers(out[2][1])) == {"A", "B", "neuropils", "head"}
    assert layers(out[2][1])["neuropils"]["segments"] == ["2"]


def test_a_replace_beat_shows_only_its_own_sets():
    out = links.states(spec("replace"), WORLD)
    assert set(layers(out[2][1])) == {"B", "head"}


def test_the_camera_is_the_view_in_the_volumes_units():
    state = links.states(spec(camera={"orbit_degrees": 0}), WORLD)[0][1]
    assert state["dimensions"] == {a: [8e-9, "m"] for a in "xyz"}
    assert state["position"] == [1.0, 2.0, 3.0] and state["projectionScale"] == 1000.0
    assert all(state["projectionOrientation"] == [0, 0, 0, 1]
               for _, state in links.states(spec(camera={"orbit_degrees": 0}), WORLD))


def test_an_orbit_turns_each_beats_link_as_far_as_the_movie_has_turned():
    s = spec(camera={"orbit_degrees": 90.0, "orbit_at": 0.0})
    out = links.states(s, WORLD)
    # Content ends at C's beat (4.5) + fade: 5.5. Beat "B" runs 2.5 to 4.5, so its middle,
    # 3.5, is 7/11 of the orbit, eased.
    f = 3.5 / 5.5
    expected = compose.turned([0, 0, 0, 1], 90.0 * f * f * (3 - 2 * f))
    assert np.allclose(out[2][1]["projectionOrientation"], expected)
    assert out[0][1]["projectionOrientation"] == [0, 0, 0, 1]          # everything: opening


def test_turning_moves_the_camera_to_its_right():
    # Identity: looking down +z, neuroglancer's y down the screen, so up is -y and the
    # camera's right is +x. A quarter turn to the right leaves it looking along -x.
    m = compose._matrix(compose.turned([0, 0, 0, 1], 90.0))
    assert np.allclose(m[:, 2], [-1, 0, 0]) and np.allclose(m[:, 1], [0, 1, 0])


def test_the_orbit_angle_eases_unless_it_loops():
    s = spec(camera={"orbit_degrees": 360.0, "orbit_at": 1.0})
    assert compose.orbit_degrees_at(s, 0.5) == 0.0
    assert compose.orbit_degrees_at(s, (1.0 + 5.5) / 2) == pytest.approx(180.0)
    assert compose.orbit_degrees_at(s, 9.0) == 360.0
    looped = spec("replace", camera={"loop": True})
    end = schedule.content_end(looped) + looped["timing"]["tail"]       # the movie's length
    assert compose.orbit_degrees_at(looped, end / 4) == pytest.approx(90.0)
    assert compose.orbit_degrees_at(spec(camera={"style": "follow"}), 3.0) is None
