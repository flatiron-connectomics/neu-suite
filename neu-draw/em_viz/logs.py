"""Silencing the object-store logging that is noise. Opt-in, always.

TensorStore's S3 stack logs at absl ``ERROR`` severity on paths that are not errors: it
reports each credential provider it could not build *before* falling through to the one
that works. So a perfectly successful read prints things like::

    E0820 08:36:34 AuthCredentialsProvider:6146] static: Profile credentials provider
        could not load a profile at .
    E0820 08:36:34 AuthCredentialsProvider:6146] Failed to resolve either region, role
        arn or token file path during sts web identity provider initialization.

**These are not failures.** The marker of a real problem is ``PERMISSION_DENIED`` or
``AccessDenied``; without one of those, the read worked. No environment variable turns
them off — ``TENSORSTORE_VERBOSE_LOGGING``, ``ABSL_MIN_LOG_LEVEL``, ``AWS_CRT_LOG_LEVEL``
and ``GLOG_minloglevel`` were all checked — because they come from C++ writing straight
to file descriptor 2, where ``contextlib.redirect_stderr`` cannot see them.

Volume is bounded rather than per-read: ``em_volume_tools.location`` caches an opened
store per prefix, so it is roughly two lines the first time a prefix is touched and
nothing after. A notebook still accumulates them across cells, hence this.

**Reads are quiet by default**, via :func:`quiet_reads` around each ``sources`` entry
point. em-volume-tools' own guidance is that only entry points should touch process-wide
stderr — and from a notebook's point of view ``sources.body_skeletons`` *is* the entry
point. Expecting a user to remember an incantation before their output is legible is the
worse trade, especially since forgetting it looks exactly like the filter being broken.

Three things keep that defensible. The filter is a **deny-list of specific known strings,
not a severity filter**, so an unrecognised message always passes through and anything
mentioning denied access or a failure status prints even when it also matches a noise
pattern. It is **scoped** to the read rather than installed for the session. And it is
**defeatable**: set ``em_viz.logs.enabled = False`` to see everything.

Only the calling thread ever touches fd 2 — a depth counter makes nested and concurrent
reads no-ops, because two threads racing ``dup2`` on the same descriptor is a genuine
way to lose output rather than merely filter it.
"""

from __future__ import annotations

import contextlib
import threading
from typing import Any, Callable, Optional

#: Set to False to see raw store logging, including the benign lines.
enabled = True

_installed: Optional[Any] = None
_scoped: Optional[Any] = None
_depth = 0
_lock = threading.RLock()


@contextlib.contextmanager
def quiet_reads():
    """Filter benign store logging for one read. Used by every ``sources`` entry point.

    A no-op when :data:`enabled` is false, when a session-wide filter is already
    installed, or when an outer read is already filtering — so nesting is free and the
    fd is only ever swapped by the thread that got there first.
    """
    global _depth, _scoped
    if not enabled or _installed is not None:
        yield
        return
    from em_volume_tools.logs import quiet_store_logs

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
    from em_volume_tools.logs import quiet_store_logs

    with quiet_store_logs(enabled):
        yield


def installed() -> bool:
    """Whether the filter is currently active in this process."""
    return _installed is not None


def install_quiet_stores() -> Callable[[], None]:
    """Filter benign store logging for the whole session. Returns an undo.

    Rarely needed now that reads filter themselves — this is for covering store access
    that does not go through ``sources`` (a direct ``em_volume_tools`` call, say).
    Idempotent: calling it twice does not stack two filters, which would leave fd 2
    pointing at a pipe nobody drains.

    **Per-process state, so a kernel restart clears it.** That is precisely why reads no
    longer depend on it: "never ran" and "ran but broken" look identical in the output,
    and the first is far more likely.
    """
    global _installed
    if _installed is None:
        from em_volume_tools.logs import quiet_store_logs

        _installed = quiet_store_logs(True)
        _installed.__enter__()
    return remove_quiet_stores


def remove_quiet_stores() -> None:
    """Restore unfiltered stderr. Safe to call when nothing is installed."""
    global _installed
    if _installed is not None:
        manager, _installed = _installed, None
        manager.__exit__(None, None, None)
