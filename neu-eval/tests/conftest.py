"""Put `tmp_path` on tmpfs when one is available.

The suites are dominated by fsync latency (every `Manifest.record` fsyncs), and on a
RAID-backed `/tmp` that is ~8 ms per call against ~0.004 ms on `/dev/shm`. Setting
`PYTEST_DEBUG_TEMPROOT` keeps pytest's numbered run directories and retention.
`--basetemp`, an inherited `PYTEST_DEBUG_TEMPROOT`, or `NEU_TESTS_TMPFS=0` override it.
The same code is in each package's conftest; keep the copies in step.
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
