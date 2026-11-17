"""Put `tmp_path` on tmpfs when one is available.

The suites are dominated by fsync latency (`Manifest.record` fsyncs per block), which
costs ~8 ms on a disk-backed `/tmp` and ~0.004 ms on `/dev/shm`. Setting
`PYTEST_DEBUG_TEMPROOT` keeps pytest's numbered run directories and retention, unlike
`--basetemp`. An explicit `--basetemp` or an inherited `PYTEST_DEBUG_TEMPROOT` wins;
`NEU_TESTS_TMPFS=0` forces the platform default. The same code is in each package's
conftest, since the suites run independently.
"""

import os
from pathlib import Path

# tmpfs spends real memory, so only use one with room to spare. A whole suite leaves
# well under a hundred MB, counting the runs pytest retains.
_MIN_FREE_BYTES = 2 * 1024**3

_CANDIDATES = ("/dev/shm", "/run/shm")


def _tmpfs_root() -> str | None:
    """A writable tmpfs directory with room to spare, or None to leave the default."""
    if os.environ.get("NEU_TESTS_TMPFS") == "0":
        return None
    for candidate in _CANDIDATES:
        root = Path(candidate)
        if not root.is_dir() or not os.access(root, os.W_OK):
            continue
        try:
            st = os.statvfs(root)
        except OSError:
            continue
        if st.f_bavail * st.f_frsize < _MIN_FREE_BYTES:
            continue
        # Per-user, because tmpfs is node-wide and pytest insists on owning its root.
        mine = root / f"em-tests-{os.getuid()}"
        try:
            mine.mkdir(mode=0o700, exist_ok=True)
        except OSError:
            continue
        return str(mine)
    return None


_root = _tmpfs_root()
if _root is not None:
    os.environ.setdefault("PYTEST_DEBUG_TEMPROOT", _root)


# -- specs ----------------------------------------------------------------------------- #

import pytest  # noqa: E402

from neu_movie import spec as spec_mod  # noqa: E402

#: The opening camera at (1, 2, 3) and 1000 units tall, in 8 nm units (`scene`'s `unit`).
BASE = {"view": {"position": [8, 16, 24], "height": 8000}, "neurons": {"library": "lib/seg"}}


def make_spec(beats, tmp=Path("/specs"), **raw):
    """A parsed spec named `movie`, from `beats` and any top-level keys or sections."""
    return spec_mod.parse({**BASE, **raw, "beat": beats}, str(tmp / "movie.toml"))


@pytest.fixture
def replace_spec():
    """Three beats: A alone, A with B, then C; replace mode, start 1, every 5, fade 1,
    hold 2."""
    return make_spec(
        [{"title": "one", "sets": [{"name": "A", "ids": [1, 2]}]},
         {"title": "two", "sets": [{"name": "A"}, {"name": "B", "label": "^B"}]},
         {"title": "three", "name": "C", "ids": [5], "regions": ["LA(L)"]}],
        neuropils={"library": "lib/seg", "path": "rois"},
        timing={"start": 1.0, "every": 5.0, "fade": 1.0, "hold": 2.0})


@pytest.fixture
def accumulate_spec():
    return make_spec(
        [{"title": "first", "name": "A", "ids": [1]},
         {"name": "B", "ids": [2]},
         {"title": "third", "name": "C", "ids": [3]}],
        mode="accumulate", timing={"start": 0.5, "every": 2.0, "fade": 1.5},
        camera={"style": "follow"})
