"""Whole-file transfers between a local path and an object store (``s3://`` / ``gs://``).

For formats with no object-store driver (HDF5): download, work on the local copy, upload.
S3 goes through boto3 when it is installed (the ``s3`` extra): streamed to and from disk,
multipart and parallel. Otherwise, and for GCS, through tensorstore, which holds the whole
object in memory and uploads it in a single request — and S3 caps a single PUT at 5 GB.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Any

from .location import is_local, read_bytes, to_kvstore, write_bytes

#: S3's limit on a single PUT; above it only a multipart upload works.
S3_SINGLE_PUT_BYTES = 5 * 1000 ** 3


def _have_boto3() -> bool:
    return importlib.util.find_spec("boto3") is not None


def _s3_client() -> Any:
    import boto3

    from .aws import ensure_aws_credentials

    # The same credentials tensorstore reads: boto3 consults AWS_* before any profile.
    ensure_aws_credentials()
    return boto3.client("s3")


def _remote(url: str) -> dict:
    if is_local(url):
        raise ValueError(f"{url!r} is a local path; a transfer needs an s3:// or gs:// URL")
    return to_kvstore(url)


def download(url: str, path: str) -> None:
    """Copy the object at ``url`` to the local file ``path``. Raises if it does not exist."""
    kv = _remote(url)
    if kv["driver"] == "s3" and _have_boto3():
        from botocore.exceptions import ClientError

        try:
            _s3_client().download_file(kv["bucket"], kv["path"], path)
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
                raise FileNotFoundError(f"{url} does not exist") from None
            raise
        return
    data = read_bytes(url)
    if data is None:
        raise FileNotFoundError(f"{url} does not exist")
    with open(path, "wb") as f:
        f.write(data)


def upload(path: str, url: str) -> None:
    """Copy the local file ``path`` to ``url``, replacing any object there."""
    kv = _remote(url)
    if kv["driver"] == "s3":
        if _have_boto3():
            _s3_client().upload_file(path, kv["bucket"], kv["path"])
            return
        size = os.path.getsize(path)
        if size > S3_SINGLE_PUT_BYTES:
            raise ValueError(
                f"{path} is {size / 1e9:.1f} GB, over S3's 5 GB single-request limit, and "
                f"without boto3 it can only be sent as one request. Install the s3 extra "
                f"(`pip install boto3`), or upload it with another tool.")
    with open(path, "rb") as f:
        write_bytes(url, f.read())
