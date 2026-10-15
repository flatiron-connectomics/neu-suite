"""`neu_vol.transfer`: whole-file copies to and from an object store.

boto3 when present for S3 (streamed, multipart); tensorstore otherwise and for GCS, which
holds the file in memory and must refuse an S3 upload over the single-request limit.
"""

import pytest

from neu_vol import transfer


class _Client:
    def __init__(self, objects):
        self.objects = objects

    def download_file(self, bucket, key, path):
        if (bucket, key) not in self.objects:
            from botocore.exceptions import ClientError

            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        with open(path, "wb") as f:
            f.write(self.objects[bucket, key])

    def upload_file(self, path, bucket, key):
        with open(path, "rb") as f:
            self.objects[bucket, key] = f.read()


@pytest.fixture
def boto(monkeypatch):
    pytest.importorskip("botocore")
    objects = {}
    monkeypatch.setattr(transfer, "_have_boto3", lambda: True)
    monkeypatch.setattr(transfer, "_s3_client", lambda: _Client(objects))
    return objects


@pytest.fixture
def no_boto(monkeypatch):
    store = {}
    monkeypatch.setattr(transfer, "_have_boto3", lambda: False)
    monkeypatch.setattr(transfer, "read_bytes", lambda url: store.get(url))
    monkeypatch.setattr(transfer, "write_bytes",
                        lambda url, data: store.__setitem__(url, bytes(data)))
    return store


def test_s3_goes_through_boto3_when_it_is_installed(tmp_path, boto):
    local = tmp_path / "a.h5"
    local.write_bytes(b"hdf5 bytes")
    transfer.upload(str(local), "s3://my-bucket/dir/a.h5")
    assert boto == {("my-bucket", "dir/a.h5"): b"hdf5 bytes"}

    back = tmp_path / "b.h5"
    transfer.download("s3://my-bucket/dir/a.h5", str(back))
    assert back.read_bytes() == b"hdf5 bytes"


def test_a_missing_s3_object_is_file_not_found_with_boto3(tmp_path, boto):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        transfer.download("s3://my-bucket/absent.h5", str(tmp_path / "x.h5"))


def test_without_boto3_the_bytes_go_through_tensorstore(tmp_path, no_boto):
    local = tmp_path / "a.h5"
    local.write_bytes(b"payload")
    transfer.upload(str(local), "s3://my-bucket/a.h5")
    assert no_boto == {"s3://my-bucket/a.h5": b"payload"}
    transfer.download("s3://my-bucket/a.h5", str(tmp_path / "b.h5"))
    assert (tmp_path / "b.h5").read_bytes() == b"payload"
    with pytest.raises(FileNotFoundError):
        transfer.download("s3://my-bucket/absent.h5", str(tmp_path / "c.h5"))


def test_without_boto3_an_s3_upload_over_the_single_put_limit_is_refused(
        tmp_path, no_boto, monkeypatch):
    monkeypatch.setattr(transfer, "S3_SINGLE_PUT_BYTES", 4)
    local = tmp_path / "big.h5"
    local.write_bytes(b"too big")
    with pytest.raises(ValueError, match="boto3"):
        transfer.upload(str(local), "s3://my-bucket/big.h5")
    assert no_boto == {}


def test_gcs_never_uses_boto3(tmp_path, no_boto, monkeypatch):
    monkeypatch.setattr(transfer, "_have_boto3", lambda: True)
    monkeypatch.setattr(transfer, "S3_SINGLE_PUT_BYTES", 4)
    local = tmp_path / "a.h5"
    local.write_bytes(b"not limited")
    transfer.upload(str(local), "gs://my-bucket/a.h5")
    assert no_boto == {"gs://my-bucket/a.h5": b"not limited"}


def test_a_local_path_is_not_a_transfer(tmp_path):
    with pytest.raises(ValueError, match="local path"):
        transfer.upload(str(tmp_path / "a.h5"), str(tmp_path / "b.h5"))
