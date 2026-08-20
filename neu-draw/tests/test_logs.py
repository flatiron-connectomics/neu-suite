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
# reads filter themselves
# --------------------------------------------------------------------------- #

def _noise_through(fn):
    """Run `fn` with fd 2 captured; return what reached the real stderr."""
    read_fd, write_fd = os.pipe()
    saved = os.dup(2)
    os.dup2(write_fd, 2)
    os.close(write_fd)
    try:
        fn()
    finally:
        os.dup2(saved, 2)
        os.close(saved)
    out = os.read(read_fd, 1 << 20)
    os.close(read_fd)
    return out


def test_every_reading_entry_point_filters_itself():
    """The user should not have to know this exists. The sibling CLIs wrap `main()`;
    em-viz has no CLI, so its entry points are these functions."""
    from em_viz import sources

    for name in ("volume_info", "scales", "volume_frame", "body_skeleton",
                 "body_mesh", "body_skeletons", "body_meshes"):
        fn = getattr(sources, name)
        assert getattr(fn, "__wrapped__", None) is not None, f"{name} is not wrapped"


def test_a_read_drops_the_noise_it_provokes():
    def noisy():
        with logs.quiet_reads():
            os.write(2, BENIGN)
    assert _noise_through(noisy) == b""


def test_a_read_still_reports_a_real_failure():
    def noisy():
        with logs.quiet_reads():
            os.write(2, BENIGN)
            os.write(2, REAL)
    assert _noise_through(noisy) == REAL


def test_nesting_is_free_and_only_the_outermost_swaps_the_fd():
    """`body_skeletons` filters, then calls `body_skeleton` which filters too. If the
    inner one swapped fd 2 again its exit would restore the outer *filter's* pipe."""
    def nested():
        with logs.quiet_reads():
            depth_outer = logs._depth
            with logs.quiet_reads():
                assert logs._depth == depth_outer + 1
                os.write(2, BENIGN)
            os.write(2, BENIGN)          # still filtered after the inner exits
    assert _noise_through(nested) == b""
    assert logs._depth == 0


def test_concurrent_reads_do_not_race_the_descriptor():
    """Worker threads must never each dup2 fd 2 — that loses output rather than
    filtering it. The depth counter is what keeps it to one swap."""
    import threading

    def worker():
        for _ in range(20):
            with logs.quiet_reads():
                os.write(2, BENIGN)

    def run():
        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert _noise_through(run) == b""
    assert logs._depth == 0


def test_the_escape_hatch_shows_everything():
    """For debugging a store problem, where the benign lines are the diagnosis."""
    logs.enabled = False
    try:
        def noisy():
            with logs.quiet_reads():
                os.write(2, BENIGN)
        assert _noise_through(noisy) == BENIGN
    finally:
        logs.enabled = True


def test_a_session_wide_install_makes_the_per_read_filter_a_no_op():
    logs.install_quiet_stores()
    try:
        with logs.quiet_reads():
            assert logs._depth == 0        # deferred to the installed one
    finally:
        logs.remove_quiet_stores()
