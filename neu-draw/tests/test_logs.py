"""Silencing benign store noise, without silencing anything else.

The risk in a filter that reassigns fd 2 is not that it fails loudly — it is that it
swallows the one message that mattered. So the tests that matter here are the ones
asserting things still get through.
"""

import os
import re

import pytest

from em_viz import logs

BENIGN = (b"E0820 08:36:34.115913 3896997 AuthCredentialsProvider:6146] static: "
          b"Profile credentials provider could not load a profile at .\n")
REAL = (b"E0820 08:36:34.115913 3896997 Kv:1] PERMISSION_DENIED: AccessDenied on "
        b"s3://bucket/key\n")
ORDINARY = b"a plain message from something else\n"


def _through_filter(lines, use_context=True):
    """Write raw bytes to fd 2 with the filter active; return what survived."""
    read_fd, write_fd = os.pipe()
    saved = os.dup(2)
    os.dup2(write_fd, 2)
    os.close(write_fd)
    try:
        if use_context:
            with logs.quiet_stores():
                for line in lines:
                    os.write(2, line)
        else:
            logs.install_quiet_stores()
            try:
                for line in lines:
                    os.write(2, line)
            finally:
                logs.remove_quiet_stores()
    finally:
        os.dup2(saved, 2)
        os.close(saved)
    out = os.read(read_fd, 1 << 20)
    os.close(read_fd)
    return out


def test_the_credential_provider_noise_is_dropped():
    assert _through_filter([BENIGN]) == b""


def test_a_permission_error_is_never_dropped():
    """The whole point. These lines look exactly like the noise — same severity, same
    absl prefix — and one of them is the only sign that a read actually failed."""
    assert b"PERMISSION_DENIED" in _through_filter([REAL])


def test_output_that_is_not_absl_formatted_is_never_examined():
    assert _through_filter([ORDINARY]) == ORDINARY


def test_the_filter_is_a_deny_list_not_a_severity_filter():
    """An unrecognised error must survive, or the next new failure mode is invisible."""
    unknown = b"E0820 08:36:34.115913 3896997 Something:1] a brand new complaint\n"
    assert unknown in _through_filter([unknown])


def test_a_mixed_stream_keeps_exactly_the_signal():
    out = _through_filter([BENIGN, ORDINARY, BENIGN, REAL, BENIGN])
    assert out == ORDINARY + REAL


def test_disabling_it_passes_everything_through():
    read_fd, write_fd = os.pipe()
    saved = os.dup(2)
    os.dup2(write_fd, 2)
    os.close(write_fd)
    try:
        with logs.quiet_stores(enabled=False):
            os.write(2, BENIGN)
    finally:
        os.dup2(saved, 2)
        os.close(saved)
    out = os.read(read_fd, 1 << 20)
    os.close(read_fd)
    assert out == BENIGN


# --------------------------------------------------------------------------- #
# the session-wide form, for the top of a notebook
# --------------------------------------------------------------------------- #

def test_the_session_wide_install_filters_and_undoes():
    assert _through_filter([BENIGN, REAL], use_context=False) == REAL


def test_installing_twice_does_not_stack_two_filters():
    """Stacking would leave fd 2 pointing at a pipe whose pump nobody joins, so the
    second uninstall would restore the *filter's* pipe rather than the real stderr."""
    try:
        first = logs.install_quiet_stores()
        second = logs.install_quiet_stores()
        assert first is second
    finally:
        logs.remove_quiet_stores()
    assert logs._installed is None


def test_removing_when_nothing_is_installed_is_harmless():
    logs.remove_quiet_stores()
    logs.remove_quiet_stores()


# --------------------------------------------------------------------------- #
# the design constraint
# --------------------------------------------------------------------------- #

def test_reading_does_not_quietly_reassign_the_callers_stderr():
    """`sources` must never install this itself. Filtering fd 2 is process-wide, and a
    library that silently reassigns its caller's stderr is how an unrelated traceback
    goes missing. A notebook is an entry point and may choose; a read may not choose
    for it."""
    import inspect

    from em_viz import sources

    source = inspect.getsource(sources)
    assert "quiet_store" not in source
    assert logs._installed is None
