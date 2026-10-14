"""Filtering benign object-store logging without hiding real errors."""

import os
import sys

import pytest

from neu_vol.logs import is_benign, quiet_store_logs

NOISE = [
    b"E0730 12:50:54.614346  161829 AuthCredentialsProvider:6146] static: Profile "
    b"credentials provider could not load a profile at .\n",
    b"E0730 12:50:54.614860  161829 AuthCredentialsProvider:6146] Failed to resolve "
    b"either region, role arn or token file path during sts web identity provider "
    b"initialization.\n",
    b"E0730 12:50:56.616988  161882 socket:1026] id=0x1 fd=41: timed out, shutting down.\n",
    b"W0730 12:50:56.617144  161882 aws_api.cc:6148] id=0x1: IMDS Client failed to "
    b"acquire a connection, error code 1048(socket operation timed out.)\n",
]

MUST_SURVIVE = [
    b"E0730 1 s3_key_value_store.cc:792] PERMISSION_DENIED: AccessDenied: Access "
    b"Denied [http_response_code='403']\n",
    b"E0730 1 x.cc:1] NOT_FOUND: no such object\n",
    b"E0730 1 x.cc:1] some entirely new failure mode\n",
    b"2026-07-30 12:52:13,215 INFO   level 0  (11260, 9000, 13750)\n",
    b"Traceback (most recent call last):\n",
]


@pytest.mark.parametrize("line, benign", [
    *[(line, True) for line in NOISE],
    *[(line, False) for line in MUST_SURVIVE],
    # NEVER_DROP wins over a noise pattern
    (b"E0730 1 x] static: Profile credentials provider could not load a "
     b"profile at . PERMISSION_DENIED\n", False),
    # ordinary output is never examined, even carrying a noise phrase
    (b"my script says: timed out, shutting down.\n", False),
])
def test_only_known_noise_is_benign(line, benign):
    assert is_benign(line) is benign


def test_filter_passes_real_stderr_and_drops_noise(capfd):
    with quiet_store_logs():
        os.write(2, NOISE[0])
        os.write(2, MUST_SURVIVE[0])
        os.write(2, b"plain message\n")
        print("via python stderr", file=sys.stderr)
        sys.stderr.flush()
    err = capfd.readouterr().err
    assert "could not load a profile" not in err
    assert "PERMISSION_DENIED" in err
    assert "plain message" in err and "via python stderr" in err


def test_disabled_passes_everything(capfd):
    with quiet_store_logs(False):
        os.write(2, NOISE[0])
        sys.stderr.flush()
    assert "could not load a profile" in capfd.readouterr().err


def test_stderr_is_restored_after_an_exception(capfd):
    try:
        with quiet_store_logs():
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    print("after", file=sys.stderr)
    sys.stderr.flush()
    assert "after" in capfd.readouterr().err
