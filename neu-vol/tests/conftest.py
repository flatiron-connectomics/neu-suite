"""Put `tmp_path` on tmpfs when one is available.

The suite is dominated by fsync latency: every recorded block fsyncs, which costs ~8 ms
on a RAID-backed `/tmp` and ~0.004 ms on `/dev/shm`. `PYTEST_DEBUG_TEMPROOT` moves the
temp tree without changing pytest's retention. An explicit `--basetemp` or an inherited
`PYTEST_DEBUG_TEMPROOT` wins; `NEU_TESTS_TMPFS=0` forces the platform default. The same
code is in each package's conftest, since each package's suite runs on its own; keep the
copies in step.

`pytest-xdist` (in the `dev` extra) is opt-in and not in `addopts`. For this suite
`python -m pytest -q -n 4 --dist worksteal` takes ~21 s down to ~10 s. Use a small explicit
`-n`: each worker re-pays a ~2 s import and tensorstore starts thread pools per process,
so `-n auto` on a many-core machine is no faster than serial. The other packages' suites
gain nothing from it.
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
