"""Filtering benign store noise from fd 2 without swallowing real errors."""

import os
import re

import pytest

from neu_draw import logs

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


UNKNOWN = b"E0820 08:36:34.115913 3896997 Something:1] a brand new complaint\n"


@pytest.mark.parametrize("lines, survives", [
    ([BENIGN], b""),                                    # the credential-provider noise
    ([REAL], REAL),                                     # same format, but a real failure
    ([ORDINARY], ORDINARY),                             # not absl-formatted: untouched
    ([UNKNOWN], UNKNOWN),                               # a deny list, not a severity filter
    ([BENIGN, ORDINARY, BENIGN, REAL, BENIGN], ORDINARY + REAL),
])
def test_only_known_noise_is_dropped(lines, survives):
    assert _through_filter(lines) == survives


def test_disabling_it_passes_everything_through():
    def noisy():
        with logs.quiet_stores(enabled=False):
            os.write(2, BENIGN)
    assert _noise_through(noisy) == BENIGN


# --------------------------------------------------------------------------- #
# the session-wide form, for the top of a notebook
# --------------------------------------------------------------------------- #

def test_the_session_wide_install_filters_and_undoes():
    assert _through_filter([BENIGN, REAL], use_context=False) == REAL


def test_installing_twice_does_not_stack_two_filters():
    """Stacking would make the uninstall restore the filter's pipe, not the real stderr."""
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
    """neu-draw has no CLI to wrap, so each reading function filters itself."""
    from neu_draw import sources

    for name in ("volume_info", "scales", "volume_frame", "body_skeleton",
                 "body_mesh", "body_skeletons", "body_meshes"):
        fn = getattr(sources, name)
        assert getattr(fn, "__wrapped__", None) is not None, f"{name} is not wrapped"


def test_a_read_drops_its_noise_but_reports_a_real_failure():
    def noisy():
        with logs.quiet_reads():
            os.write(2, BENIGN)
            os.write(2, REAL)
    assert _noise_through(noisy) == REAL


def test_nesting_is_free_and_only_the_outermost_swaps_the_fd():
    """`body_skeletons` calls `body_skeleton`, and both filter; the inner one must not
    swap fd 2 again."""
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
    """Threads share one fd swap (a depth counter); each dup2-ing fd 2 would lose output."""
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
    """`logs.enabled = False`, for debugging a store problem."""
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
