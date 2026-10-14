"""What `detect_backend` may cost: it runs before nearly every op, so its gzip probe must
ask about a named object and, if it ever lists, list the scale with the fewest chunks
(listing the finest scale once took 51 s on S3)."""

import numpy as np
import pytest

from neu_vol import convert
from neu_vol.backends.tensorstore import TensorStoreBackend
from neu_vol.profiles import zarr3_create_spec
from neu_vol.source_metadata import (PRECOMPUTED_GZ, _first_chunk_key,
                                            detect_backend,
                                            precomputed_chunks_are_gzipped)


def _precomputed(tmp_path, name="vol", shape=(32, 32, 32)):
    data = np.ones(shape, "uint8")
    src = str(tmp_path / f"{name}.src.zarr")
    be = TensorStoreBackend.create(
        zarr3_create_spec("local", src, shape, "uint8",
                          dimension_names=("z", "y", "x"), chunk=(8, 8, 8)),
        delete_existing=True)
    be.write_region(tuple(slice(0, s) for s in shape), data)
    dst = str(tmp_path / name)
    convert(src, dst, voxel_size=(8, 8, 8), profile="local-neuroglancer",
            chunk=(8, 8, 8), factors=[(2, 2, 2)], min_dim=8, delete_existing=True)
    return dst


# --------------------------------------------------------------------------- #
# the predicted key
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scale, key", [
    ({"size": [88, 71, 108], "chunk_sizes": [[128, 128, 128]]}, "0-88_0-71_0-108"),
    ({"size": [512, 512, 512], "chunk_sizes": [[128, 128, 128]]}, "0-128_0-128_0-128"),
    ({"size": [256, 256, 256], "voxel_offset": [64, 0, 32],
      "chunk_sizes": [[128, 128, 128]]}, "64-192_0-128_32-160"),
    ({}, None),                                    # unusable metadata: no key, not a wrong one
    ({"size": [1, 2, 3]}, None),
])
def test_origin_chunk_key(scale, key):
    """Clipped to the scale's size, shifted by its voxel offset."""
    assert _first_chunk_key(scale) == key


# --------------------------------------------------------------------------- #
# the probe
# --------------------------------------------------------------------------- #
def test_a_plain_volume_is_detected_without_listing_anything(tmp_path, monkeypatch):
    """No enumeration on the happy path: the predicted origin key resolves."""
    dst = _precomputed(tmp_path)
    from neu_vol import location

    def refuse(*a, **k):
        raise AssertionError("detect_backend listed a prefix instead of probing a key")

    monkeypatch.setattr(location, "list_keys", refuse)
    monkeypatch.setattr("neu_vol.source_metadata.list_keys", refuse,
                        raising=False)
    assert detect_backend(dst) == "neuroglancer_precomputed"


def test_gz_suffixed_chunks_are_detected(tmp_path):
    """Missed, tensorstore would read the fill value and a conversion would be zeros."""
    import os
    import shutil

    dst = _precomputed(tmp_path, "gz")
    # rename every chunk object the way CloudVolume writes them
    for root, _dirs, files in os.walk(dst):
        for f in files:
            if "_" in f and "-" in f:
                shutil.move(os.path.join(root, f), os.path.join(root, f + ".gz"))
    assert detect_backend(dst) == PRECOMPUTED_GZ


def test_the_coarsest_scale_is_probed(tmp_path):
    """Any scale answers the question, so the cheap (coarsest) one may stand in."""
    from neu_vol.location import read_json

    dst = _precomputed(tmp_path, "many")
    scales = read_json(dst, "info")["scales"]
    coarsest = max(scales, key=lambda s: tuple(s["resolution"]))
    finest = min(scales, key=lambda s: tuple(s["resolution"]))
    assert coarsest["key"] != finest["key"], "test volume needs >1 scale"
    assert precomputed_chunks_are_gzipped(dst, coarsest) is False
    assert precomputed_chunks_are_gzipped(dst, finest) is False


def test_an_absent_origin_chunk_falls_back_to_a_listing(tmp_path):
    """On a sparse volume the predicted key may hold nothing."""
    import os

    dst = _precomputed(tmp_path, "sparse")
    from neu_vol.location import read_json
    coarsest = max(read_json(dst, "info")["scales"],
                   key=lambda s: tuple(s["resolution"]))
    key = _first_chunk_key(coarsest)
    origin = os.path.join(dst, coarsest["key"], key)
    assert os.path.exists(origin)
    os.remove(origin)                       # now only non-origin chunks exist
    # still answerable, via the listing fallback
    assert precomputed_chunks_are_gzipped(dst, coarsest) is False


def test_a_volume_whose_declared_scales_are_mostly_empty_is_still_detected(tmp_path):
    """An `info` may declare scales never written; the coarsest may be one of them."""
    import os
    import shutil

    from neu_vol.location import read_json

    dst = _precomputed(tmp_path, "mostly_empty")
    scales = read_json(dst, "info")["scales"]
    assert len(scales) > 1, "test volume needs >1 scale"
    finest = min(scales, key=lambda s: tuple(s["resolution"]))

    # Only the finest scale keeps chunks, .gz-suffixed; the others are declared but empty.
    for sc in scales:
        d = os.path.join(dst, sc["key"])
        if sc["key"] != finest["key"]:
            shutil.rmtree(d)
            continue
        for f in os.listdir(d):
            shutil.move(os.path.join(d, f), os.path.join(d, f + ".gz"))

    assert detect_backend(dst) == PRECOMPUTED_GZ


def test_gz_is_detected_when_no_scale_has_an_origin_chunk(tmp_path):
    """No origin chunk anywhere, and only one (non-coarsest) scale holds data."""
    import os
    import shutil

    from neu_vol.location import read_json
    from neu_vol.source_metadata import _first_chunk_key

    dst = _precomputed(tmp_path, "no_origin_gz")
    scales = read_json(dst, "info")["scales"]
    finest = min(scales, key=lambda s: tuple(s["resolution"]))

    for sc in scales:
        d = os.path.join(dst, sc["key"])
        if sc["key"] != finest["key"]:
            shutil.rmtree(d)                    # declared, never written
            continue
        os.remove(os.path.join(d, _first_chunk_key(sc)))    # and no ORIGIN chunk
        for f in os.listdir(d):
            shutil.move(os.path.join(d, f), os.path.join(d, f + ".gz"))
    assert os.listdir(os.path.join(dst, finest["key"])), "test needs surviving chunks"

    assert detect_backend(dst) == PRECOMPUTED_GZ


def test_the_listing_fallback_still_goes_to_the_coarsest_scale(tmp_path, monkeypatch):
    """The fallback lists the coarsest scale only, never a dense one."""
    import os

    from neu_vol import location
    from neu_vol.location import read_json

    dst = _precomputed(tmp_path, "no_origins")
    scales = read_json(dst, "info")["scales"]
    coarsest = max(scales, key=lambda s: tuple(s["resolution"]))
    for sc in scales:                       # no origin chunk anywhere -> pass 1 misses
        os.remove(os.path.join(dst, sc["key"], _first_chunk_key(sc)))

    listed = []
    real = location.list_keys
    monkeypatch.setattr(location, "list_keys",
                        lambda loc, prefix="", *a, **k: (listed.append(prefix),
                                                         real(loc, prefix, *a, **k))[1])
    assert detect_backend(dst) == "neuroglancer_precomputed"
    assert listed == [coarsest["key"]]


def test_an_empty_scale_is_not_gzipped(tmp_path):
    """Nothing stored is no evidence; False keeps the ordinary reader."""
    assert precomputed_chunks_are_gzipped(
        str(tmp_path), {"key": "8_8_8", "size": [8, 8, 8],
                        "chunk_sizes": [[8, 8, 8]]}) is False
