"""`neu-vol to-hdf5`: pack a small volume into an HDF5 file `neu-vol write` can place.

The round trip is the claim: packed, then written with no `--offset` and no
`--offset-order`, the piece lands where it came from. Most other tests guard the axis
order, which reversed mirrors the piece through z=x with nothing to show for it.
"""

import numpy as np
import pytest

from neu_vol import cli, create_volume, pack_hdf5
from neu_vol.backends.base import open_backend
from neu_vol.ops.pack import DEFAULT_DATASET


def _stack(tmp_path, name="stack", *, shape=(4, 6, 8), start=1):
    """A directory of ordered PNG slices, each plane distinct."""
    from PIL import Image

    d = tmp_path / name
    d.mkdir()
    data = np.arange(start, start + np.prod(shape), dtype=np.uint8).reshape(shape)
    for z in range(shape[0]):
        Image.fromarray(data[z]).save(d / f"s{z:03d}.png")
    return str(d), data


def _h5(path, dataset=DEFAULT_DATASET):
    import h5py

    with h5py.File(path, "r") as f:
        dset = f[dataset]
        return (np.asarray(dset[()]),
                {k: dset.attrs[k] for k in dset.attrs},
                {k: f.attrs[k] for k in f.attrs})


# --------------------------------------------------------------------------- #
# what lands in the file
# --------------------------------------------------------------------------- #
def test_the_stack_and_its_frame_are_written(tmp_path):
    src, data = _stack(tmp_path)
    out = str(tmp_path / "piece.h5")
    plan = pack_hdf5(src, out, voxel_size=(40, 8, 8), voxel_offset=(16, 32, 64),
                     src_format="image_stack")

    arr, dattrs, rattrs = _h5(out)
    np.testing.assert_array_equal(arr, data)
    assert plan["dataset"] == DEFAULT_DATASET, "the default the reader also assumes"
    assert list(dattrs["voxel_offset"]) == [16, 32, 64]
    assert list(dattrs["voxel_size"]) == [40, 8, 8]
    assert list(dattrs["offset"]) == [16 * 40, 32 * 8, 64 * 8], "physical, from the voxels"
    assert dattrs["axes"] == "zyx" and rattrs["axes"] == "zyx"
    assert rattrs["units"] == "nm"
    assert list(rattrs["voxel_size"]) == [40, 8, 8], "the file's frame, not just the array's"


def test_the_default_dataset_is_what_the_reader_looks_for(tmp_path):
    """Packed with no arguments, read with no arguments."""
    src, data = _stack(tmp_path)
    out = str(tmp_path / "p.h5")
    pack_hdf5(src, out, voxel_size=(8, 8, 8), src_format="image_stack")

    from neu_vol.backends.hdf5 import sole_dataset

    assert sole_dataset(out) == DEFAULT_DATASET
    be = open_backend({"backend": "hdf5", "path": out})       # no `dataset` key
    np.testing.assert_array_equal(be.read_region((slice(0, 4), slice(0, 6), slice(0, 8))),
                                  data)


def test_a_named_dataset_and_chunking_are_honoured(tmp_path):
    import h5py

    src, _ = _stack(tmp_path)
    out = str(tmp_path / "named.h5")
    pack_hdf5(src, out, voxel_size=(8, 8, 8), dataset="gt/piece0", chunk=(2, 3, 4),
              compression=None, src_format="image_stack")
    with h5py.File(out, "r") as f:
        assert f["/gt/piece0"].chunks == (2, 3, 4)
        assert f["/gt/piece0"].compression is None


def test_the_dtype_can_be_cast(tmp_path):
    src, data = _stack(tmp_path)
    out = str(tmp_path / "cast.h5")
    pack_hdf5(src, out, voxel_size=(8, 8, 8), dtype="uint32", src_format="image_stack")
    arr, _, _ = _h5(out)
    assert arr.dtype == np.uint32
    np.testing.assert_array_equal(arr, data)


def _labels(tmp_path, top, *, dtype=np.uint64, name="labels.h5"):
    """A self-describing label file whose largest id is ``top``."""
    src, _ = _stack(tmp_path, name=f"{name}.stack")
    path = str(tmp_path / name)
    pack_hdf5(src, path, voxel_size=(8, 8, 8), voxel_offset=(1, 2, 3), kind="segmentation")
    data = np.arange(4 * 6 * 8, dtype=dtype).reshape(4, 6, 8)
    data[-1, -1, -1] = top
    import h5py

    with h5py.File(path, "a") as f:
        attrs = dict(f[DEFAULT_DATASET].attrs)
        del f[DEFAULT_DATASET]
        f.create_dataset(DEFAULT_DATASET, data=data)
        f[DEFAULT_DATASET].attrs.update(attrs)
    return path, data


def test_a_narrowing_cast_whose_values_fit_is_written(tmp_path):
    src, data = _labels(tmp_path, 65535)
    out = str(tmp_path / "u16.h5")
    plan = pack_hdf5(src, out, dtype="uint16")
    arr, dattrs, _ = _h5(out)
    assert plan["narrowing"] and arr.dtype == np.uint16
    np.testing.assert_array_equal(arr, data)
    assert list(dattrs["voxel_offset"]) == [1, 2, 3] and dattrs["kind"] == "segmentation"


@pytest.mark.parametrize("top, dtype, to", [(65536, np.uint64, "uint16"),
                                            (-1, np.int16, "uint8")],
                         ids=["too-big", "negative"])
def test_a_narrowing_cast_that_does_not_fit_is_refused_and_writes_nothing(
        tmp_path, top, dtype, to):
    import os

    src, _ = _labels(tmp_path, top, dtype=dtype)
    out = str(tmp_path / "narrow.h5")
    with pytest.raises(ValueError, match=r"1 value\(s\) in block .* do not fit"):
        pack_hdf5(src, out, dtype=to)
    assert not os.path.exists(out), "a file this call created is removed"


def test_a_refused_replacement_leaves_the_old_dataset_alone(tmp_path):
    import h5py

    src, data = _labels(tmp_path, 70000)
    out = str(tmp_path / "keep.h5")
    pack_hdf5(src, out)                                       # uint64, as the source is
    with pytest.raises(ValueError, match="do not fit uint16"):
        pack_hdf5(src, out, dtype="uint16", overwrite=True)
    np.testing.assert_array_equal(_h5(out)[0], data)
    with h5py.File(out, "r") as f:
        assert list(f) == ["data"], "no partial dataset left behind"


def test_truncate_casts_what_does_not_fit(tmp_path):
    src, data = _labels(tmp_path, 65537)
    out = str(tmp_path / "trunc.h5")
    pack_hdf5(src, out, dtype="uint16", truncate=True)
    arr = _h5(out)[0]
    assert arr[-1, -1, -1] == 1, "low-order bits, as astype keeps them"
    np.testing.assert_array_equal(arr, data.astype(np.uint16))


# --------------------------------------------------------------------------- #
# object-store source and destination
# --------------------------------------------------------------------------- #
class _Bucket:
    """``s3://my-bucket/<key>`` backed by a directory, counting transfers."""

    def __init__(self, root):
        self.root = root
        self.downloads = 0
        self.uploads = 0

    def path(self, url):
        assert url.startswith("s3://my-bucket/"), url
        return self.root / url[len("s3://my-bucket/"):]

    def put(self, local, url):
        import shutil

        self.path(url).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, self.path(url))


@pytest.fixture
def bucket(tmp_path, monkeypatch):
    """A fake object store, plus a temp root that must be empty again after each call."""
    import shutil
    import tempfile

    from neu_vol import location, transfer

    b = _Bucket(tmp_path / "bucket")
    b.root.mkdir()

    def download(url, path):
        if not b.path(url).exists():
            raise FileNotFoundError(f"{url} does not exist")
        b.downloads += 1
        shutil.copyfile(b.path(url), path)

    def upload(path, url):
        b.uploads += 1
        b.put(path, url)

    monkeypatch.setattr(transfer, "download", download)
    monkeypatch.setattr(transfer, "upload", upload)
    monkeypatch.setattr(location, "exists", lambda url: b.path(url).exists())
    b.scratch = tmp_path / "scratch"
    b.scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(b.scratch))
    return b


def test_an_hdf5_source_on_an_object_store_is_read_from_a_local_copy(tmp_path, bucket):
    src, data = _labels(tmp_path, 9)
    url = "s3://my-bucket/gt/labels.h5"
    bucket.put(src, url)

    plan = pack_hdf5(url, str(tmp_path / "from-s3.h5"), dtype="uint16")
    arr, dattrs, _ = _h5(str(tmp_path / "from-s3.h5"))
    np.testing.assert_array_equal(arr, data)
    assert list(dattrs["voxel_offset"]) == [1, 2, 3], "the frame comes along"
    assert plan["src_spec"]["path"] == url, "reported as read from where it lives"
    assert list(bucket.scratch.iterdir()) == [], "the local copy is removed"

    with pytest.raises(FileNotFoundError, match="does not exist"):
        pack_hdf5("s3://my-bucket/gt/absent.h5", str(tmp_path / "x.h5"))


def test_a_remote_out_is_written_locally_then_uploaded(tmp_path, bucket):
    src, data = _labels(tmp_path, 9)
    url = "s3://my-bucket/out/piece.h5"
    plan = pack_hdf5(src, url, dtype="uint16")
    assert plan["out"] == url and bucket.uploads == 1
    np.testing.assert_array_equal(_h5(str(bucket.path(url)))[0], data)
    assert list(bucket.scratch.iterdir()) == []


def test_a_remote_out_that_exists_is_added_to(tmp_path, bucket):
    import h5py

    src, _ = _labels(tmp_path, 9)
    url = "s3://my-bucket/out/piece.h5"
    pack_hdf5(src, url, dataset="a")
    pack_hdf5(src, url, dataset="b")
    assert bucket.downloads == 1, "the second call starts from what is there"
    with h5py.File(bucket.path(url), "r") as f:
        assert sorted(f) == ["a", "b"]


def test_a_remote_out_is_not_uploaded_on_failure_or_dry_run(tmp_path, bucket):
    src, _ = _labels(tmp_path, 70000)
    url = "s3://my-bucket/out/piece.h5"
    with pytest.raises(ValueError, match="do not fit"):
        pack_hdf5(src, url, dtype="uint16")
    pack_hdf5(src, url, dry_run=True)
    assert bucket.uploads == 0 and not bucket.path(url).exists()
    assert list(bucket.scratch.iterdir()) == []


# --------------------------------------------------------------------------- #
# several datasets of one file
# --------------------------------------------------------------------------- #
def _multi(tmp_path, name="multi.h5"):
    """A file of three self-describing datasets at different offsets."""
    src, _ = _stack(tmp_path, name=f"{name}.stack")
    path = str(tmp_path / name)
    for i, d in enumerate(("a", "b", "c")):
        pack_hdf5(src, path, voxel_size=(8, 8, 8), voxel_offset=(i, 10 * i, 100 * i),
                  dataset=d, kind="segmentation")
    return path


def test_every_dataset_is_repacked_under_its_own_name(tmp_path):
    from neu_vol import pack_hdf5_datasets

    src = _multi(tmp_path)
    out = str(tmp_path / "all.h5")
    plans = pack_hdf5_datasets(src, out, dtype="uint16")
    assert [p["dataset"] for p in plans] == ["/a", "/b", "/c"]
    for i, d in enumerate(("a", "b", "c")):
        arr, dattrs, _ = _h5(out, d)
        assert arr.dtype == np.uint16
        np.testing.assert_array_equal(arr, _h5(src, d)[0])
        assert list(dattrs["voxel_offset"]) == [i, 10 * i, 100 * i]


def test_a_subset_of_datasets_can_be_named(tmp_path):
    import h5py

    from neu_vol import pack_hdf5_datasets

    src = _multi(tmp_path)
    out = str(tmp_path / "some.h5")
    pack_hdf5_datasets(src, out, ["c", "a"])
    with h5py.File(out, "r") as f:
        assert sorted(f) == ["a", "c"]


@pytest.mark.parametrize("kwargs, err", [({"dataset": "x"}, TypeError),
                                         ({"voxel_offset": (0, 0, 0)}, ValueError),
                                         ({"crop_start": (0, 0, 0)}, ValueError)])
def test_what_describes_one_dataset_is_refused_for_several(tmp_path, kwargs, err):
    from neu_vol import pack_hdf5_datasets

    with pytest.raises(err):
        pack_hdf5_datasets(_multi(tmp_path), str(tmp_path / "x.h5"), **kwargs)


def test_several_datasets_cost_one_transfer_each_way(tmp_path, bucket):
    from neu_vol import pack_hdf5_datasets

    src_url, out_url = "s3://my-bucket/in/multi.h5", "s3://my-bucket/out/multi.h5"
    bucket.put(_multi(tmp_path), src_url)
    plans = pack_hdf5_datasets(src_url, out_url, dtype="uint16")
    assert (bucket.downloads, bucket.uploads) == (1, 1)
    assert all(p["out"] == out_url and p["src_spec"]["path"] == src_url for p in plans)
    assert list(bucket.scratch.iterdir()) == []


def test_the_cli_repacks_all_datasets(tmp_path, capsys):
    src = _multi(tmp_path)
    out = str(tmp_path / "cli-all.h5")
    assert cli.main(["to-hdf5", "--src", src, "--out", out, "--all-datasets",
                     "--dtype", "uint16"]) == 0
    assert "wrote 3 datasets" in capsys.readouterr().out
    assert cli.main(["to-hdf5", "--src", src, "--out", str(tmp_path / "two.h5"),
                     "--src-dataset", "a", "--src-dataset", "b"]) == 0
    with pytest.raises(SystemExit, match="describes ONE dataset"):
        cli.main(["to-hdf5", "--src", src, "--out", str(tmp_path / "y.h5"),
                  "--all-datasets", "--dataset", "x"])


def test_the_cli_reports_and_refuses_narrowing(tmp_path, capsys):
    src, _ = _labels(tmp_path, 65536)
    out = str(tmp_path / "cli-narrow.h5")
    assert cli.main(["to-hdf5", "--src", src, "--out", out, "--dtype", "uint16",
                     "--dry-run"]) == 0
    assert "narrowing   uint64 -> uint16: refused if" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="--truncate"):
        cli.main(["to-hdf5", "--src", src, "--out", out, "--dtype", "uint16"])
    assert cli.main(["to-hdf5", "--src", src, "--out", out, "--dtype", "uint16",
                     "--truncate"]) == 0


def test_a_big_source_is_read_in_blocks(tmp_path):
    """A "small volume" that is not stays packable: the array is never held whole."""
    src, data = _stack(tmp_path, shape=(8, 6, 8))
    out = str(tmp_path / "blocked.h5")
    plan = pack_hdf5(src, out, voxel_size=(8, 8, 8), src_format="image_stack",
                     max_bytes=6 * 8)                       # one plane at a time
    assert plan["blocks"] == 8 and plan["block_shape"] == (1, 6, 8)
    np.testing.assert_array_equal(_h5(out)[0], data)


# --------------------------------------------------------------------------- #
# adding to an existing file
# --------------------------------------------------------------------------- #
def test_a_second_piece_can_join_a_file_with_the_same_frame(tmp_path, caplog):
    """Several pieces of one volume in one file, each with its own voxel_offset."""
    src, _ = _stack(tmp_path)
    out = str(tmp_path / "two.h5")
    pack_hdf5(src, out, voxel_size=(8, 8, 8), voxel_offset=(0, 0, 0), dataset="a",
              src_format="image_stack")
    with caplog.at_level("WARNING"):
        pack_hdf5(src, out, voxel_size=(8, 8, 8), voxel_offset=(4, 0, 0), dataset="b",
                  src_format="image_stack")

    import h5py

    with h5py.File(out, "r") as f:
        assert list(f["/a"].attrs["voxel_offset"]) == [0, 0, 0]
        assert list(f["/b"].attrs["voxel_offset"]) == [4, 0, 0]
    assert "readers must name one" in caplog.text, \
        "sole_dataset stops working, so say so before it does"


def test_a_frame_that_disagrees_is_refused(tmp_path):
    """One file describing two coordinate systems is not a thing worth allowing."""
    src, _ = _stack(tmp_path)
    out = str(tmp_path / "clash.h5")
    pack_hdf5(src, out, voxel_size=(8, 8, 8), dataset="a", src_format="image_stack")
    with pytest.raises(ValueError, match="already records a different frame"):
        pack_hdf5(src, out, voxel_size=(4, 4, 4), dataset="b", src_format="image_stack")
    with pytest.raises(ValueError, match="already records a different frame"):
        pack_hdf5(src, out, voxel_size=(8, 8, 8), units="um", dataset="c",
                  src_format="image_stack")


def test_reusing_a_name_needs_overwrite(tmp_path):
    src, _ = _stack(tmp_path)
    out = str(tmp_path / "same.h5")
    pack_hdf5(src, out, voxel_size=(8, 8, 8), src_format="image_stack")
    with pytest.raises(FileExistsError, match="already has a dataset"):
        pack_hdf5(src, out, voxel_size=(8, 8, 8), src_format="image_stack")

    src2, data2 = _stack(tmp_path, "other", start=100)
    plan = pack_hdf5(src2, out, voxel_size=(8, 8, 8), src_format="image_stack",
                     overwrite=True)
    assert plan["replacing"]
    np.testing.assert_array_equal(_h5(out)[0], data2)


@pytest.mark.parametrize("kwargs, match", [
    ({"voxel_size": (8, 8), "src_format": "image_stack"}, "same number of axes"),
    ({"voxel_size": (8, 8, 8), "level": 2}, "needs a multiscale volume"),
    ({}, "voxel_size is required"),                 # a stack records no scale
])
def test_a_stack_refuses_what_it_cannot_supply(tmp_path, kwargs, match):
    src, _ = _stack(tmp_path)
    with pytest.raises(ValueError, match=match):
        pack_hdf5(src, str(tmp_path / "x.h5"), **kwargs)


# --------------------------------------------------------------------------- #
# the round trip, which is the point
# --------------------------------------------------------------------------- #
def test_the_packed_piece_writes_back_with_no_offset_and_no_order(tmp_path):
    """Pack, then place with no --offset and no --offset-order."""
    from neu_vol import write_subvolume

    src, data = _stack(tmp_path)
    piece = str(tmp_path / "piece.h5")
    pack_hdf5(src, piece, voxel_size=(8, 8, 8), voxel_offset=(4, 6, 8),
              src_format="image_stack")

    vol = str(tmp_path / "vol.zarr")
    create_volume(vol, shape=(16, 16, 16), voxel_size=(8, 8, 8), dtype="uint8",
                  chunk=(8, 8, 8), levels=1)
    result = write_subvolume(vol, piece)

    assert result["offset"] == (4, 6, 8)
    assert "voxel_offset" in result["offset_from"]
    assert "recorded in the source" in result["offset_from"], \
        "the order came from the file, not from a default"
    be = open_backend({"backend": "zarr3", "path": f"{vol}/0"})
    np.testing.assert_array_equal(
        be.read_region((slice(4, 8), slice(6, 12), slice(8, 16))), data)


def test_an_xyz_file_is_read_as_xyz_without_being_told(tmp_path):
    """A file recording axes="xyz" has its offset reversed by `write` itself."""
    from neu_vol import write_subvolume

    src, data = _stack(tmp_path, shape=(4, 4, 4))
    piece = str(tmp_path / "xyz.h5")
    # the same place, written the other way round: xyz (x=8, y=6, z=4)
    pack_hdf5(src, piece, voxel_size=(8, 8, 8), voxel_offset=(8, 6, 4), axes=("x", "y", "z"),
              src_format="image_stack")

    vol = str(tmp_path / "v.zarr")
    create_volume(vol, shape=(16, 16, 16), voxel_size=(8, 8, 8), dtype="uint8",
                  chunk=(8, 8, 8), levels=1)
    result = write_subvolume(vol, piece)
    assert result["offset"] == (4, 6, 8), "reversed to zyx on the file's own authority"
    assert "read as xyz" in result["offset_from"]
    assert "recorded in the source" in result["offset_from"]


def test_an_explicit_order_still_wins_over_the_file(tmp_path):
    from neu_vol.ops.write import resolve_offset

    src, _ = _stack(tmp_path, shape=(4, 4, 4))
    piece = str(tmp_path / "o.h5")
    pack_hdf5(src, piece, voxel_size=(8, 8, 8), voxel_offset=(1, 2, 3),
              src_format="image_stack")
    be = open_backend({"backend": "hdf5", "path": piece})
    assert be.stored_axes()[0] == "zyx"
    off, prov = resolve_offset(be, None, field="voxel_offset", order="xyz", ndim=3)
    assert off == (3, 2, 1) and "read as xyz" in prov
    assert "recorded in the source" not in prov, "an explicit order is not the file's"


# --------------------------------------------------------------------------- #
# a box out of a volume, at a level
# --------------------------------------------------------------------------- #
@pytest.fixture
def pyramid(tmp_path):
    """A 3-level anisotropic zarr group, so a level's voxel size is not 2**level."""
    from neu_vol import convert
    from neu_vol.backends.tensorstore import TensorStoreBackend
    from neu_vol.profiles import zarr3_create_spec

    data = np.arange(32 * 32 * 32, dtype=np.uint16).reshape(32, 32, 32)
    src = str(tmp_path / "p.src.zarr")
    be = TensorStoreBackend.create(
        zarr3_create_spec("local", src, data.shape, "uint16",
                          dimension_names=("z", "y", "x"), chunk=(8, 8, 8)),
        delete_existing=True)
    be.write_region(tuple(slice(0, s) for s in data.shape), data)
    dst = str(tmp_path / "pyr.zarr")
    convert(src, dst, voxel_size=(40, 8, 8), profile="local", chunk=(8, 8, 8), min_dim=8,
            delete_existing=True)
    return dst, data


def test_a_zarr_group_can_be_packed_at_all(tmp_path, pyramid):
    """A zarr group is opened at a level, not as an array."""
    vol, data = pyramid
    out = str(tmp_path / "grp.h5")
    pack_hdf5(vol, out, voxel_size=(40, 8, 8))
    np.testing.assert_array_equal(_h5(out)[0], data)


def test_the_level_supplies_the_voxel_size_it_records(tmp_path, pyramid):
    """The level's recorded voxel size, never 2**level."""
    vol, _ = pyramid
    out = str(tmp_path / "lvl.h5")
    plan = pack_hdf5(vol, out, level=1)
    assert plan["voxel_size"] == (40.0, 16.0, 16.0), "z is coarse already, so (1,2,2)"
    assert list(_h5(out)[2]["voxel_size"]) == [40, 16, 16]

    assert pack_hdf5(vol, str(tmp_path / "l2.h5"), level=2)["voxel_size"] == \
        (40.0, 32.0, 32.0)


def test_a_box_is_packed_and_its_origin_becomes_the_offset(tmp_path, pyramid):
    vol, data = pyramid
    out = str(tmp_path / "box.h5")
    plan = pack_hdf5(vol, out, crop_start=(4, 8, 12), crop_stop=(12, 16, 20))

    assert plan["shape"] == (8, 8, 8)
    assert plan["voxel_offset"] == (4, 8, 12), "taken from the crop origin"
    np.testing.assert_array_equal(_h5(out)[0], data[4:12, 8:16, 12:20])
    assert list(_h5(out)[1]["voxel_offset"]) == [4, 8, 12]


def test_an_explicit_offset_overrides_the_crop_origin(tmp_path, pyramid):
    vol, _ = pyramid
    plan = pack_hdf5(vol, str(tmp_path / "o.h5"), crop_start=(4, 4, 4),
                     crop_stop=(8, 8, 8), voxel_offset=(0, 0, 0))
    assert plan["voxel_offset"] == (0, 0, 0)


def test_a_box_at_a_level_round_trips_into_that_level(tmp_path, pyramid):
    """Extract a region at level 1, write it back at level 1, unchanged."""
    from neu_vol import write_subvolume

    vol, _ = pyramid
    piece = str(tmp_path / "rt.h5")
    pack_hdf5(vol, piece, level=1, crop_start=(2, 2, 2), crop_stop=(10, 10, 10))
    before = open_backend({"backend": "zarr3", "path": f"{vol}/1"}).read_region(
        (slice(2, 10),) * 3).copy()

    result = write_subvolume(vol, piece, level=1)
    assert result["offset"] == (2, 2, 2)
    after = open_backend({"backend": "zarr3", "path": f"{vol}/1"}).read_region(
        (slice(2, 10),) * 3)
    np.testing.assert_array_equal(after, before)


def test_a_level_the_volume_does_not_have_is_refused(tmp_path, pyramid):
    vol, _ = pyramid
    with pytest.raises(ValueError, match="has no level 9"):
        pack_hdf5(vol, str(tmp_path / "x.h5"), level=9)


# --------------------------------------------------------------------------- #
# what the fields are called
# --------------------------------------------------------------------------- #
def test_the_voxel_size_field_can_be_renamed_for_both_directions(tmp_path):
    """The same field name is used to write and to read."""
    src, _ = _stack(tmp_path)
    out = str(tmp_path / "named_field.h5")
    pack_hdf5(src, out, voxel_size=(40, 8, 8), voxel_size_field="resolution")

    _, dattrs, rattrs = _h5(out)
    assert list(rattrs["resolution"]) == [40, 8, 8] and "voxel_size" not in rattrs
    assert list(dattrs["resolution"]) == [40, 8, 8]

    # and reading it back needs no --voxel-size, given the same field name
    again = str(tmp_path / "again.h5")
    plan = pack_hdf5(out, again, voxel_size_field="resolution")
    assert plan["voxel_size"] == (40.0, 8.0, 8.0)


def test_an_hdf5_source_supplies_its_own_scale(tmp_path):
    """Repacking a file that already records a voxel size must not ask for it again."""
    src, _ = _stack(tmp_path)
    first = str(tmp_path / "first.h5")
    pack_hdf5(src, first, voxel_size=(40, 8, 8), voxel_offset=(1, 2, 3))

    plan = pack_hdf5(first, str(tmp_path / "second.h5"))
    assert plan["voxel_size"] == (40.0, 8.0, 8.0)
    assert plan["axes"] == ("z", "y", "x"), "the recorded order comes along too"


def _legacy_file(tmp_path, name="legacy.h5", *, as_datasets=True):
    """A file written elsewhere: data in `main`, offset and size as top-level datasets
    (``as_datasets``) or as attributes."""
    import h5py

    path = str(tmp_path / name)
    data = np.arange(8 * 8 * 8, dtype=np.uint16).reshape(8, 8, 8)
    with h5py.File(path, "w") as f:
        f.create_dataset("main", data=data)
        if as_datasets:
            f.create_dataset("voxel_offset", data=np.array([100, 200, 300], np.int64))
            f.create_dataset("voxel_size", data=np.array([40.0, 8.0, 8.0]))
        else:
            f["main"].attrs["voxel_offset"] = np.array([100, 200, 300], np.int64)
            f["main"].attrs["voxel_size"] = np.array([40.0, 8.0, 8.0])
    return path, data


@pytest.mark.parametrize("as_datasets", [True, False], ids=["datasets", "attrs"])
def test_a_file_that_describes_itself_is_repacked_without_losing_it(tmp_path, as_datasets):
    """Dropping the offset would silently place the repacked piece at 0,0,0."""
    src, data = _legacy_file(tmp_path, as_datasets=as_datasets)
    out = str(tmp_path / "repacked.h5")
    plan = pack_hdf5(src, out)                       # no --voxel-size, no --offset

    assert plan["voxel_offset"] == (100, 200, 300)
    assert plan["voxel_size"] == (40.0, 8.0, 8.0)
    arr, dattrs, _ = _h5(out)
    np.testing.assert_array_equal(arr, data)
    assert list(dattrs["voxel_offset"]) == [100, 200, 300]


def test_a_box_out_of_a_self_describing_file_lands_at_the_sum(tmp_path):
    src, data = _legacy_file(tmp_path)
    out = str(tmp_path / "box.h5")
    plan = pack_hdf5(src, out, crop_start=(2, 3, 4), crop_stop=(6, 7, 8))

    assert plan["voxel_offset"] == (102, 203, 304)
    np.testing.assert_array_equal(_h5(out)[0], data[2:6, 3:7, 4:8])


def test_an_explicit_offset_still_overrides_what_the_file_says(tmp_path):
    src, _ = _legacy_file(tmp_path)
    plan = pack_hdf5(src, str(tmp_path / "o.h5"), voxel_offset=(0, 0, 0))
    assert plan["voxel_offset"] == (0, 0, 0)


def test_a_repacked_file_round_trips_into_a_volume(tmp_path):
    from neu_vol import write_subvolume

    src, data = _legacy_file(tmp_path)
    piece = str(tmp_path / "canonical.h5")
    pack_hdf5(src, piece)

    vol = str(tmp_path / "v.zarr")
    create_volume(vol, shape=(128, 256, 320), voxel_size=(40, 8, 8), dtype="uint16",
                  chunk=(8, 8, 8), levels=1)
    result = write_subvolume(vol, piece)
    assert result["offset"] == (100, 200, 300)
    assert "recorded in the source" in result["offset_from"], "axes came from the file"
    be = open_backend({"backend": "zarr3", "path": f"{vol}/0"})
    np.testing.assert_array_equal(
        be.read_region((slice(100, 108), slice(200, 208), slice(300, 308))), data)


def test_a_recorded_xyz_order_is_honoured_when_repacking(tmp_path):
    """If the file says its numbers are xyz, they are reversed once, not twice."""
    import h5py

    src, _ = _legacy_file(tmp_path, "xyz.h5")
    with h5py.File(src, "a") as f:
        f.attrs["axes"] = "xyz"
    plan = pack_hdf5(src, str(tmp_path / "x.h5"))
    assert plan["voxel_offset"] == (300, 200, 100), "reversed on the file's own authority"


def test_the_offset_field_can_be_renamed(tmp_path):
    src, _ = _stack(tmp_path)
    out = str(tmp_path / "off.h5")
    pack_hdf5(src, out, voxel_size=(8, 8, 8), voxel_offset=(1, 2, 3),
              offset_field="corner")
    _, dattrs, _ = _h5(out)
    assert list(dattrs["corner"]) == [1, 2, 3] and "voxel_offset" not in dattrs

    # `write` looks for voxel_offset, so it has to be told the new name
    from neu_vol.ops.write import resolve_offset

    be = open_backend({"backend": "hdf5", "path": out})
    assert be.stored_offset("voxel_offset") is None
    assert resolve_offset(be, None, field="corner", order=None, ndim=3)[0] == (1, 2, 3)


def test_a_renamed_frame_is_still_compared_on_append(tmp_path):
    src, _ = _stack(tmp_path)
    out = str(tmp_path / "cmp.h5")
    pack_hdf5(src, out, voxel_size=(8, 8, 8), voxel_size_field="resolution", dataset="a")
    with pytest.raises(ValueError, match="already records a different frame"):
        pack_hdf5(src, out, voxel_size=(4, 4, 4), voxel_size_field="resolution",
                  dataset="b")


def test_write_warns_when_the_pieces_scale_is_not_the_levels(tmp_path, pyramid, caplog):
    """A piece from another level fits and places cleanly at the wrong resolution."""
    from neu_vol import write_subvolume

    vol, _ = pyramid
    piece = str(tmp_path / "lvl1.h5")
    pack_hdf5(vol, piece, level=1, crop_start=(0, 0, 0), crop_stop=(4, 4, 4))

    with caplog.at_level("WARNING"):
        result = write_subvolume(vol, piece, level=0)      # level 1 data into level 0
    assert "records a voxel size" in caplog.text and "check --level" in caplog.text
    assert result["scale_note"]

    caplog.clear()
    with caplog.at_level("WARNING"):
        result = write_subvolume(vol, piece, level=1)      # back where it belongs
    assert result["scale_note"] is None
    assert "records a voxel size" not in caplog.text


# --------------------------------------------------------------------------- #
# the CLI
# --------------------------------------------------------------------------- #
def test_the_cli_packs_and_reports(tmp_path, capsys):
    src, data = _stack(tmp_path)
    out = str(tmp_path / "cli.h5")
    assert cli.main(["to-hdf5", "--src", src, "--out", out, "--src-format",
                     "image_stack", "--voxel-size", "40,8,8", "--offset", "2,4,6"]) == 0
    printed = capsys.readouterr().out
    assert "40x8x8 nm" in printed and "z 2  y 4  x 6" in printed
    assert "axes        zyx" in printed
    np.testing.assert_array_equal(_h5(out)[0], data)


def test_the_cli_dry_run_writes_nothing(tmp_path, capsys):
    import os

    src, _ = _stack(tmp_path)
    out = str(tmp_path / "none.h5")
    assert cli.main(["to-hdf5", "--src", src, "--out", out, "--src-format",
                     "image_stack", "--voxel-size", "8,8,8", "--dry-run"]) == 0
    assert "nothing written" in capsys.readouterr().out
    assert not os.path.exists(out)


def test_the_reader_refuses_a_remote_path_with_a_useful_error():
    """h5py has no object-store driver, and its own error does not say so. (`pack_hdf5`
    stages remote files instead; see the object-store tests above.)"""
    with pytest.raises(ValueError, match="rclone|aws s3 cp"):
        open_backend({"backend": "hdf5", "path": "s3://bucket/piece.h5"})


def test_the_cli_refuses_a_bad_axes_string(tmp_path):
    src, _ = _stack(tmp_path)
    with pytest.raises(SystemExit, match="permutation of zyx"):
        cli.main(["to-hdf5", "--src", src, "--out", str(tmp_path / "b.h5"),
                  "--src-format", "image_stack", "--voxel-size", "8,8,8",
                  "--axes", "zyz"])
