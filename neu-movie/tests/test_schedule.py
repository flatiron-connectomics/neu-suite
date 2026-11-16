"""When each set is on screen, in both modes."""

from neu_movie import schedule


def test_runs_join_consecutive_beats_only(replace_spec):
    spec = dict(replace_spec, beats=replace_spec["beats"] + [{"title": "", "sets": ["A"]}])
    assert schedule.runs(spec) == {"A": [(0, 1), (3, 3)], "B": [(1, 1)], "C": [(2, 2)]}


def test_replace_fades_out_hold_after_the_last_beat_of_a_run(replace_spec):
    # start 1, every 5, fade 1, hold 2: a set fades out fade + hold after its run's last beat.
    assert schedule.intervals(replace_spec) == {"A": [(1.0, 9.0)], "B": [(6.0, 9.0)],
                                                "C": [(11.0, 14.0)]}
    assert schedule.beat_end(replace_spec, 0) == 1.0 + 1 + 2 + 1
    assert schedule.content_end(replace_spec) == 11.0 + 4


def test_accumulate_sets_stay(accumulate_spec):
    assert schedule.intervals(accumulate_spec) == {"A": [(0.5, None)], "B": [(2.5, None)],
                                                   "C": [(4.5, None)]}
    assert schedule.content_end(accumulate_spec) == 4.5 + 1.5


def test_overlapping_treats_none_as_the_end():
    assert schedule.overlapping([(0, 2)], [(1, 3)], end=10)
    assert not schedule.overlapping([(0, 2)], [(2, 3)], end=10)
    assert schedule.overlapping([(5, None)], [(8, 9)], end=10)
    assert not schedule.overlapping([(5, None)], [(10, 12)], end=10)
