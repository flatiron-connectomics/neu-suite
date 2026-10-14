import json
import os

import numpy as np
import pytest

from neu_vol import convert
from neu_vol.backends.base import open_backend
from neu_vol.backends.tensorstore import TensorStoreBackend
from neu_vol.profiles import precomputed_create_spec, zarr3_create_spec
from neu_vol.source_metadata import PRECOMPUTED_GZ, detect_backend, read_source_metadata


def _zarr3(path):
    TensorStoreBackend.create(
        zarr3_create_spec("local", path, (8, 8, 8), "uint16",
                          dimension_names=("z", "y", "x"), chunk=(8, 8, 8)),
        delete_existing=True)


def _precomputed(path):
    TensorStoreBackend.create(
        precomputed_create_spec("s3-neuroglancer", path, (8, 8, 8), "uint8",
                                resolution_zyx=[8, 8, 8], scale_index=0, chunk=(8, 8, 8)),
        delete_existing=True)


def _zarr2(path, data):
    import tensorstore as ts
    s = ts.open({"driver": "zarr", "kvstore": {"driver": "file", "path": path},
                 "metadata": {"shape": list(data.shape), "chunks": [8, 8, 8],
                              "dtype": "<u2", "compressor": None, "order": "C",
                              "fill_value": 0}, "create": True, "delete_existing": True}).result()
    s[...] = data
    return path


@pytest.mark.parametrize("build, expected", [
    (_zarr3, "zarr3"),
    (_precomputed, "neuroglancer_precomputed"),
    (os.makedirs, None),
])
def test_detect_backend_by_format(tmp_path, build, expected):
    p = str(tmp_path / "vol")
    build(p)
    assert detect_backend(p) == expected


def test_detect_zarr2_and_convert(tmp_path):
    vol = np.arange(8 * 8 * 8, dtype=np.uint16).reshape(8, 8, 8)
    p = _zarr2(str(tmp_path / "v2.zarr"), vol)
    assert detect_backend(p) == "zarr2"

    dst = str(tmp_path / "out.zarr")
    convert(p, dst, voxel_size=(8, 8, 8), chunk=(8, 8, 8), multiscale=False, delete_existing=True)
    out = open_backend({"backend": "zarr3", "path": os.path.join(dst, "0")})
    np.testing.assert_array_equal(out.read_region((slice(0, 8),) * 3), vol)


# --------------------------------------------------------------------------- #
# CloudVolume-written precomputed: chunks carry a .gz suffix
# --------------------------------------------------------------------------- #
def _precomputed_info(tmp_path, chunk_name=None, key="8_8_8"):
    """A minimal precomputed volume, with one chunk object of the given name (or none)."""
    root = tmp_path / "vol"
    (root / key).mkdir(parents=True)
    (root / "info").write_text(json.dumps({
        "type": "image", "data_type": "uint8", "num_channels": 1,
        "scales": [{"key": key, "size": [256, 256, 256], "resolution": [8, 8, 8],
                    "chunk_sizes": [[64, 64, 64]], "encoding": "raw",
                    "voxel_offset": [0, 0, 0]}]}))
    if chunk_name is not None:
        (root / key / chunk_name).write_bytes(b"\x1f\x8b junk")
    return root


@pytest.mark.parametrize("chunk_name, expected", [
    ("0-64_0-64_0-64.gz", PRECOMPUTED_GZ),      # tensorstore would read these as zeros
    ("0-64_0-64_0-64", "neuroglancer_precomputed"),
    (None, "neuroglancer_precomputed"),         # no chunks yet: do not guess .gz
])
def test_gzipped_chunks_are_a_distinct_backend(tmp_path, chunk_name, expected):
    assert detect_backend(str(_precomputed_info(tmp_path, chunk_name))) == expected


def test_opening_a_gz_volume_without_cloudvolume_fails_with_guidance(tmp_path):
    """Fail fast and say what to do, not a silent zero-filled read."""
    root = _precomputed_info(tmp_path, "0-64_0-64_0-64.gz")
    try:
        import cloudvolume  # noqa: F401
        pytest.skip("cloud-volume is installed here; the missing-dep path cannot run")
    except ImportError:
        pass
    with pytest.raises(ImportError, match="cloud-volume|neu-vol-cv"):
        open_backend({"backend": PRECOMPUTED_GZ, "path": str(root)})


@pytest.mark.parametrize("chunk_name, backend", [
    ("0-64_0-64_0-64.gz", PRECOMPUTED_GZ),
    ("0-64_0-64_0-64", "neuroglancer_precomputed"),
])
def test_data_spec_keeps_the_detected_backend(tmp_path, chunk_name, backend):
    """`data_spec` selects the reader, so it must not contradict detection: a spec saying
    plain precomputed for a .gz volume reads every block as zeros."""
    root = _precomputed_info(tmp_path, chunk_name)
    meta = read_source_metadata({"backend": backend, "path": str(root)})
    assert meta["data_spec"]["backend"] == backend
    assert meta["voxel_size"] == (8.0, 8.0, 8.0)
    assert meta["kind"] == "image"


def test_cloudvolume_url_handles_every_spec_form(tmp_path):
    """`convert` hands workers a `kvstore` spec with no `path`; the URL must build from it."""
    from neu_vol.backends.cloudvolume import _url

    root = _precomputed_info(tmp_path, "0-64_0-64_0-64.gz")
    by_path = _url({"backend": PRECOMPUTED_GZ, "path": str(root)})
    assert by_path == f"precomputed://file://{root}"

    meta = read_source_metadata({"backend": PRECOMPUTED_GZ, "path": str(root)})
    assert _url(meta["data_spec"]) == by_path

    assert _url({"backend": PRECOMPUTED_GZ, "path": "s3://bkt/a/b"}) \
        == "precomputed://s3://bkt/a/b"


def test_cloudvolume_url_rejects_an_unsupported_driver():
    from neu_vol.backends.cloudvolume import _url

    with pytest.raises(ValueError, match="local paths and s3"):
        _url({"kvstore": {"driver": "gcs", "bucket": "b", "path": "p"}})
