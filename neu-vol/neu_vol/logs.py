"""Drop the object-store logging that is noise, and only that.

TensorStore's S3 stack logs at absl ERROR severity on every successful store open: the
credential providers that could not be built before the one that works. No environment
variable turns this off, and it comes from C++ on fd 2, so :func:`quiet_store_logs`
filters fd 2 through a pipe. Two rules keep it from hiding real problems:

1. It is a deny-list of known-noise strings, not a severity filter: an unrecognized
   message always passes through.
2. :data:`NEVER_DROP` wins over every pattern, so denied access, HTTP error codes and
   tensorstore failure statuses are always printed.

:func:`quiet_store_logs` is for entry points (scripts, drivers). :func:`quiet_reads` is
the same filter for a library function a notebook calls directly; it nests and is safe
across threads.
"""

from __future__ import annotations

import contextlib
import os
import re
import sys
import threading

# Emitted on every successful S3 open: the credential providers that could not be
# built, before the environment provider (populated by aws.ensure_aws_credentials).
BENIGN_PATTERNS = (
    "Profile credentials provider could not load a profile",
    "Failed to resolve either region, role arn or token file path",
    "sts web identity provider initialization",
    # Reached only when no credentials are in the environment; the chain probes
    # the EC2 metadata service, which does not exist off EC2, and times out.
    "AWS_IO_SOCKET_TIMEOUT",
    "AWS_IO_SOCKET_NO_ROUTE_TO_HOST",
    "IMDS Client failed to acquire a connection",
    "IMDS client failed to update the token",
    "Failed to obtain new connection from http layer",
    "Failed to complete connection acquisition",
    "Client connection failed with error",
    "timed out, shutting down.",
    # The GCS equivalent, on every anonymous read of a public bucket. A real GCS auth
    # failure says PERMISSION_DENIED or UNAUTHENTICATED, which NEVER_DROP keeps.
    "google_auth_provider.cc",
    "Could not find the credentials file in the standard gcloud location",
)

# Printed no matter what. A message that mentions any of these is diagnostic even
# if it also looks like noise.
NEVER_DROP = (
    "PERMISSION_DENIED", "AccessDenied", "Access Denied", "UNAUTHENTICATED",
    "NOT_FOUND", "INVALID_ARGUMENT", "ALREADY_EXISTS", "FAILED_PRECONDITION",
    "Traceback", "http_response_code",
)

# absl's prefix: severity, date, time, thread id. Only such lines are candidates;
# ordinary output is never examined for noise patterns.
_ABSL = re.compile(rb"^[EWIF]\d{4} \d")


def is_benign(line: bytes) -> bool:
    """True if this line is known store-stack noise and safe to drop."""
    if not _ABSL.match(line):
        return False
    text = line.decode("utf-8", "replace")
    if any(m in text for m in NEVER_DROP):
        return False
    return any(p in text for p in BENIGN_PATTERNS)


@contextlib.contextmanager
def quiet_store_logs(enabled: bool = True):
    """Filter benign object-store logging out of fd 2 for the duration.

    No-op when ``enabled`` is false, or when fd 2 cannot be duplicated (a closed
    or exotic stderr) — in that case everything is passed through unfiltered,
    which is the safe direction.
    """
    if not enabled:
        yield
        return
    try:
        saved = os.dup(2)
    except OSError:
        yield
        return

    read_fd, write_fd = os.pipe()

    def pump():
        with os.fdopen(read_fd, "rb", 0) as r:
            for line in r:
                if not is_benign(line):
                    os.write(saved, line)

    thread = threading.Thread(target=pump, name="stderr-filter", daemon=True)
    thread.start()
    try:
        sys.stderr.flush()
        os.dup2(write_fd, 2)
        os.close(write_fd)
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)          # restore before draining, so nothing is lost
        thread.join(timeout=5.0)
        os.close(saved)


# --------------------------------------------------------------------------- #
# the notebook-facing wrapper
# --------------------------------------------------------------------------- #
#: Set False to see raw store logging, including the benign lines. Consulted at call
#: time, so flipping it in a live session takes effect on the next read.
reads_quiet = True

_scoped = None
_depth = 0
_depth_lock = threading.RLock()


@contextlib.contextmanager
def quiet_reads():
    """Filter benign store logging for one read, safe to nest and to call from threads.

    For a library function a notebook calls directly::

        with quiet_reads():
            info = location.read_json(source, "info")

    A no-op when :data:`reads_quiet` is false or an outer read is already filtering. Only
    the outermost caller swaps fd 2, since racing ``dup2`` calls could lose output.
    """
    global _depth, _scoped
    if not reads_quiet:
        yield
        return
    with _depth_lock:
        first = _depth == 0
        if first:
            _scoped = quiet_store_logs(True)
            _scoped.__enter__()
        _depth += 1
    try:
        yield
    finally:
        with _depth_lock:
            _depth -= 1
            if _depth == 0 and _scoped is not None:
                manager, _scoped = _scoped, None
                manager.__exit__(None, None, None)
