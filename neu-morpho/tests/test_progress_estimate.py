"""`neu-morpho progress` estimates time left from the manifest records' times."""

import json
import time

import pytest

from neu_morpho._scripts.run_progress import _estimate, main

NOW = 1_792_000_000.0


def test_the_rate_is_measured_over_the_recent_window():
    # 10 a minute for the last 20 minutes; only the last 10 count
    times = [NOW - 6 * i for i in range(200)]
    assert _estimate(times, 100, NOW) == "10.0/min over the last 10 min: ~10 min left"


def test_a_group_that_started_recently_is_measured_since_its_first_completion():
    times = [NOW - 6 * i for i in range(30)]                # 29 x 6 s = 2.9 min
    assert _estimate(times, 300, NOW).startswith("10.0/min over the last 3 min")


@pytest.mark.parametrize("times, remaining, expected", [
    ([], 10, None),                                          # an old manifest: no times
    ([NOW - 1], 10, None),                                   # too soon to call it a rate
    ([NOW - 6 * i for i in range(10)], 0, None),             # nothing left
    ([NOW - 3600, NOW - 3000], 5, "none finished in the last 10 min"),
])
def test_no_estimate_without_a_rate(times, remaining, expected):
    assert _estimate(times, remaining, NOW) == expected


def test_long_estimates_read_in_hours():
    times = [NOW - 60 * i for i in range(11)]               # 1 a minute
    assert _estimate(times, 300, NOW).endswith("~5.0 h left")


def test_progress_prints_the_estimate_for_a_group_in_flight(tmp_path, capsys):
    now = time.time()
    (tmp_path / "run_plan.json").write_text(json.dumps(
        {"dst": "seg", "stages": {"skel": {"run": {"id": "r"}, "scale": 1,
                                           "planned_blocks": 100}}}))
    with open(tmp_path / "progress.skel.jsonl", "w") as f:
        for i in range(40):                                 # 40 blocks, one per 6 s
            f.write(json.dumps({"group": "skel-chunk", "key": [i, 0, 0],
                                "status": "written", "t": now - 6 * i}) + "\n")
    assert main([str(tmp_path), "--failures", "0"]) == 0
    out = capsys.readouterr().out
    assert "40/    100" in out and "/min over the last 4 min: ~6 min left" in out
