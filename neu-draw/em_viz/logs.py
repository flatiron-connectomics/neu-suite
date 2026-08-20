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

**Nothing in em-viz calls these implicitly.** Filtering fd 2 is process-wide, and a
library that quietly reassigns its caller's stderr is the kind of thing that makes an
unrelated traceback vanish. A notebook or script is an entry point and may decide this
for itself; ``sources`` may not decide it for them.

The filter itself is ``em_volume_tools.logs``, and it is a **deny-list of specific known
strings, not a severity filter** — an unrecognised message always passes through, and
anything mentioning denied access or a failure status is printed even if it also matches
a noise pattern.
"""

from __future__ import annotations

import contextlib
from typing import Any, Callable, Optional

_installed: Optional[Any] = None


@contextlib.contextmanager
def quiet_stores(enabled: bool = True):
    """Drop benign store logging for the duration of the block.

    >>> with quiet_stores():
    ...     skeletons = sources.body_skeletons(volume, bodies)
    """
    from em_volume_tools.logs import quiet_store_logs

    with quiet_store_logs(enabled):
        yield


def install_quiet_stores() -> Callable[[], None]:
    """Filter benign store logging for the rest of the session. Returns an undo.

    For the top of a notebook, where wrapping every cell in a ``with`` is worse than the
    noise. Idempotent — calling it twice does not stack two filters, which would leave
    fd 2 pointing at a pipe nobody drains.
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
