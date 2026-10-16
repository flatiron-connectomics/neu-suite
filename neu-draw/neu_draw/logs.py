"""Silencing the object-store logging that is noise. Opt-in, always.

TensorStore's S3 stack logs at absl ``ERROR`` severity on paths that are not errors: it
reports each credential provider it could not build *before* falling through to the one
that works. So a perfectly successful read prints things like::

    E0820 08:36:34 AuthCredentialsProvider:6146] static: Profile credentials provider
        could not load a profile at .
    E0820 08:36:34 AuthCredentialsProvider:6146] Failed to resolve either region, role
        arn or token file path during sts web identity provider initialization.

These are not failures: a real problem says ``PERMISSION_DENIED`` or ``AccessDenied``.
No environment variable turns them off, and they are written by C++ straight to file
descriptor 2, where ``contextlib.redirect_stderr`` cannot reach them. There are about two
per store prefix, the first time it is opened.

Reads through ``sources`` are filtered by default (:func:`quiet_reads`). The filter drops
only specific known lines, so anything unrecognised, or mentioning denied access, still
prints. It applies for the duration of a read, not the session. Set
``neu_draw.logs.quiet = False`` to see everything. Nested and concurrent reads share one
filter, so only one thread swaps fd 2.
"""

from __future__ import annotations

import contextlib
import threading
from typing import Any, Callable, Optional

#: Filter the benign store logging (the default); False shows it raw, as does neu-vol's
#: own store-open filter while a read runs.
quiet = True

_installed: Optional[Any] = None
_scoped: Optional[Any] = None
_depth = 0
_lock = threading.RLock()


@contextlib.contextmanager
def quiet_reads():
    """Filter benign store logging for one read; used by every ``sources`` entry point.
    With :data:`quiet` false it filters nothing, neu-vol's store opens included; a no-op
    when a session-wide filter is installed or an outer read is already filtering."""
    global _depth, _scoped
    from neu_vol.logs import quiet_store_logs

    if not quiet:
        # neu-vol's store opens filter themselves; this turns that off too.
        with quiet_store_logs(False):
            yield
        return
    if _installed is not None:
        yield
        return

    with _lock:
        if _depth == 0:
            _scoped = quiet_store_logs(True)
            _scoped.__enter__()
        _depth += 1
    try:
        yield
    finally:
        with _lock:
            _depth -= 1
            if _depth == 0 and _scoped is not None:
                _scoped, done = None, _scoped
                done.__exit__(None, None, None)


@contextlib.contextmanager
def quiet_stores(enabled: bool = True):
    """Drop benign store logging for the duration of the block.

    >>> with quiet_stores():
    ...     skeletons = sources.body_skeletons(volume, bodies)
    """
    from neu_vol.logs import quiet_store_logs

    with quiet_store_logs(enabled):
        yield


def installed() -> bool:
    """Whether the filter is currently active in this process."""
    return _installed is not None


def install_quiet_stores() -> Callable[[], None]:
    """Filter benign store logging for the whole session; returns an undo.

    For store access outside ``sources``, such as a direct ``neu_vol`` call. Idempotent.
    It is per-process state, so a kernel restart clears it.
    """
    global _installed
    if _installed is None:
        from neu_vol.logs import quiet_store_logs

        _installed = quiet_store_logs(True)
        _installed.__enter__()
    return remove_quiet_stores


def remove_quiet_stores() -> None:
    """Restore unfiltered stderr. Safe to call when nothing is installed."""
    global _installed
    if _installed is not None:
        manager, _installed = _installed, None
        manager.__exit__(None, None, None)
