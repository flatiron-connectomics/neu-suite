"""Detecting and describing the two container formats, HDF5 and image stacks.

Neither has a marker object, so: detection is evidence-based (a directory is a stack
because it holds slices), a marker always wins, and an axis order is read, never assumed
(``voxel_offset`` is precomputed's name, where it means xyz).
"""

import os

import numpy as np
import pytest

from neu_vol.detect import (SINGLE_LEVEL_FORMATS, detect_backend, detect_file_backend,
                            require_chunked_volume)
from neu_vol.source_metadata import (describe, existing_levels, level_spec, location_spec,
                                     read_level_voxel_sizes, read_source_metadata)

h5py = pytest.importorskip("h5py")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _h5(path, datasets, root_attrs=None, chunks=None):
    """An HDF5 file. ``datasets`` is ``{name: (array, attrs)}``."""
    with h5py.File(str(path), "w") as f:
        for k, v in (root_attrs or {}).items():
            f.attrs[k] = v
        for name, (data, attrs) in datasets.items():
            dset = f.create_dataset(name, data=data, chunks=chunks)
            for k, v in (attrs or {}).items():
                dset.attrs[k] = v
    return str(path)


def _framed(path, data=None, voxel_size=(40.0, 8.0, 8.0), voxel_offset=(2, 3, 4),
            axes="zyx", units="nm"):
    """One dataset carrying the frame `neu-vol to-hdf5` records."""
    data = np.arange(2 * 3 * 4, dtype="uint16").reshape(2, 3, 4) if data is None else data
    attrs = {"voxel_size": np.asarray(voxel_size, "float64"),
             "voxel_offset": np.asarray(voxel_offset, "int64"), "units": units}
    if axes is not None:
        attrs["axes"] = axes
    return _h5(path, {"data": (data, attrs)})


def _slices(directory, n=3, ext=".tif", shape=(6, 7)):
    os.makedirs(directory, exist_ok=True)
    for z in range(n):
        arr = np.full(shape, z, "uint8")
        target = os.path.join(str(directory), f"s{z:03d}{ext}")
        if ext in (".tif", ".tiff"):
            tifffile = pytest.importorskip("tifffile")
            tifffile.imwrite(target, arr)
        else:
            iio = pytest.importorskip("imageio.v3")
            iio.imwrite(target, arr)
    return str(directory)


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["a.h5", "a.hdf5", "a.hdf", "a.he5", "A.H5"])
def test_every_hdf5_extension_is_recognised(tmp_path, name):
    assert detect_backend(_framed(tmp_path / name)) == "hdf5"


def test_a_directory_of_slices_is_a_stack(tmp_path):
    assert detect_backend(_slices(tmp_path / "stack")) == "image_stack"


def test_a_glob_is_a_stack_without_touching_the_filesystem(tmp_path):
    """A glob is only ever a stack, even before anything matches it."""
    assert detect_file_backend(str(tmp_path / "nothing-here" / "*.tif")) == "image_stack"


def test_a_single_multipage_tiff_is_a_stack(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    path = str(tmp_path / "pages.tif")
    # photometric, or tifffile stores a 3-plane array as RGB and deprecation-warns
    tifffile.imwrite(path, np.zeros((4, 5, 6), "uint8"), photometric="minisblack")
    assert detect_backend(path) == "image_stack"
    assert describe(path)["shape"] == (4, 5, 6)


def _empty_dir(tmp_path):
    (tmp_path / "empty").mkdir()
    return tmp_path / "empty"


def _unrelated_dir(tmp_path):
    d = tmp_path / "notes"
    d.mkdir()
    (d / "README.md").write_text("hello")
    (d / "run.log").write_text("hello")
    return d


def _unknown_file(tmp_path):
    path = tmp_path / "volume.dat"
    path.write_bytes(b"\x00" * 16)
    return path


@pytest.mark.parametrize("make", [_empty_dir, _unrelated_dir, _unknown_file])
def test_nothing_is_detected_without_evidence(tmp_path, make):
    """A directory is not a stack just for being a directory, and no extension is guessed."""
    assert detect_backend(str(make(tmp_path))) is None


def test_a_marker_always_wins_over_a_stray_slice(tmp_path):
    """Adding a file must not change what an existing volume is."""
    root = tmp_path / "vol"
    root.mkdir()
    (root / "zarr.json").write_text('{"node_type": "group", "attributes": {}}')
    _slices(root, n=2)
    assert detect_backend(str(root)) == "zarr3"


def test_a_remote_location_is_never_a_container(tmp_path):
    """Neither container backend can open an object-store location."""
    assert detect_file_backend("s3://my-bucket/pieces/a.h5") is None
    assert detect_file_backend("s3://my-bucket/slices") is None


def test_mixed_slice_types_are_read_as_one_stack_and_warned_about(tmp_path, caplog):
    """A mixed directory interleaves two stacks into one volume: legal, but warned."""
    d = _slices(tmp_path / "mixed", n=2, ext=".tif")
    _slices(d, n=2, ext=".png")
    with caplog.at_level("WARNING"):
        assert detect_backend(d) == "image_stack"
    assert "more than one type" in caplog.text


# --------------------------------------------------------------------------- #
# the frame an HDF5 file records about itself
# --------------------------------------------------------------------------- #
def test_a_packed_file_reports_its_own_frame(tmp_path):
    path = _framed(tmp_path / "piece.h5")
    meta = read_source_metadata(location_spec(path, "hdf5"))
    assert meta["voxel_size"] == (40.0, 8.0, 8.0)
    # physical origin = voxel_offset * voxel_size, per axis
    assert meta["offset"] == (80.0, 24.0, 32.0)
    assert meta["voxel_offset"] == (2, 3, 4)
    assert meta["units"] == "nm"
    assert meta["spatial_axes"] == ("z", "y", "x")
    assert meta["has_channels"] is False
    # never guessed: a wrong "image" would average label ids
    assert meta["kind"] is None


def test_an_xyz_frame_is_reversed_into_zyx(tmp_path):
    """Read the wrong way round, an anisotropic volume would be mirrored through z=x."""
    path = _framed(tmp_path / "xyz.h5", voxel_size=(8.0, 8.0, 40.0),
                   voxel_offset=(4, 3, 2), axes="xyz")
    meta = read_source_metadata({"backend": "hdf5", "path": path, "dataset": "/data"})
    assert meta["voxel_size"] == (40.0, 8.0, 8.0)
    assert meta["voxel_offset"] == (2, 3, 4)
    assert meta["spatial_axes"] == ("x", "y", "z"), "reported as recorded"


def test_an_uninterpretable_axis_order_raises_rather_than_falling_back(tmp_path):
    path = _framed(tmp_path / "weird.h5", axes="yxz")
    with pytest.raises(ValueError, match="'yxz'"):
        read_source_metadata({"backend": "hdf5", "path": path, "dataset": "/data"})


def test_no_recorded_axes_falls_back_to_zyx_and_says_so(tmp_path, caplog):
    """Warned only when anisotropic, since reversing an isotropic vector changes nothing."""
    aniso = _framed(tmp_path / "aniso.h5", voxel_size=(40.0, 8.0, 8.0), axes=None)
    with caplog.at_level("WARNING"):
        meta = read_source_metadata({"backend": "hdf5", "path": aniso,
                                     "dataset": "/data"})
    assert meta["voxel_size"] == (40.0, 8.0, 8.0)
    assert "no `axes`" in caplog.text

    caplog.clear()
    iso = _framed(tmp_path / "iso.h5", voxel_size=(8.0, 8.0, 8.0), axes=None)
    with caplog.at_level("WARNING"):
        read_source_metadata({"backend": "hdf5", "path": iso, "dataset": "/data"})
    assert caplog.text == ""


def test_voxel_size_is_the_gate_for_the_whole_frame(tmp_path):
    """An offset without a voxel size is withheld, not half-reported."""
    path = _h5(tmp_path / "off-only.h5",
               {"data": (np.zeros((2, 2, 2), "uint8"),
                         {"voxel_offset": np.asarray([1, 1, 1], "int64")})})
    assert read_source_metadata({"backend": "hdf5", "path": path,
                                 "dataset": "/data"}) is None


def test_a_frame_on_the_root_group_covers_the_dataset(tmp_path):
    """A file-wide voxel size on the root, the usual shape for a container of crops."""
    path = _h5(tmp_path / "root.h5", {"data": (np.zeros((2, 2, 2), "uint8"), {})},
               root_attrs={"voxel_size": np.asarray([8.0, 8.0, 8.0]), "axes": "zyx"})
    meta = read_source_metadata({"backend": "hdf5", "path": path, "dataset": "/data"})
    assert meta["voxel_size"] == (8.0, 8.0, 8.0)


def test_channels_come_from_the_ndim(tmp_path):
    path = _framed(tmp_path / "ch.h5", data=np.zeros((2, 3, 4, 5), "uint8"),
                   voxel_offset=(0, 0, 0))
    meta = read_source_metadata({"backend": "hdf5", "path": path, "dataset": "/data"})
    assert meta["has_channels"] is True


# --------------------------------------------------------------------------- #
# one level, and exactly one
# --------------------------------------------------------------------------- #
def test_a_container_has_exactly_one_level(tmp_path):
    """Not a pyramid of identical levels (the backend ignores `scale_index`)."""
    path = _framed(tmp_path / "piece.h5")
    levels = existing_levels(path, "hdf5")
    assert sorted(levels) == [0]
    assert levels[0]["shape"] == (2, 3, 4)

    stack = _slices(tmp_path / "stack", n=3)
    assert sorted(existing_levels(stack, "image_stack")) == [0]


@pytest.mark.parametrize("fmt", sorted(SINGLE_LEVEL_FORMATS))
def test_any_level_but_zero_is_an_error_not_an_absence(tmp_path, fmt):
    with pytest.raises(ValueError, match="only level 0"):
        level_spec(str(tmp_path / "x"), fmt, 3)


def test_the_one_level_carries_the_recorded_voxel_size(tmp_path):
    """A list of one, the same shape of answer as for a pyramid."""
    path = _framed(tmp_path / "piece.h5")
    assert read_level_voxel_sizes(location_spec(path, "hdf5")) == [(40.0, 8.0, 8.0)]
    assert read_level_voxel_sizes({"backend": "image_stack", "source": "x"}) is None


# --------------------------------------------------------------------------- #
# describe
# --------------------------------------------------------------------------- #
def test_describe_works_on_a_packed_hdf5_file(tmp_path):
    path = _framed(tmp_path / "piece.h5")
    d = describe(path)
    assert d["location"] == path
    assert d["dataset"] == "/data" and sorted(d["datasets"]) == ["/data"]
    assert d["format"] == "hdf5"
    assert d["shape"] == (2, 3, 4)
    assert d["dtype"] == "uint16"
    assert d["meta"]["voxel_size"] == (40.0, 8.0, 8.0)
    assert d["level_voxel_sizes"] == [(40.0, 8.0, 8.0)]
    assert sorted(d["levels"]) == [0]
    assert d["spec"]["dataset"] == "/data"
    assert d["other_markers"] == []


def test_describe_works_on_a_file_that_records_nothing(tmp_path):
    """`meta` is None; shape and dtype come from the array itself."""
    path = _h5(tmp_path / "plain.h5", {"data": (np.zeros((4, 5, 6), "uint8"), {})})
    d = describe(path)
    assert d["meta"] is None
    assert d["shape"] == (4, 5, 6) and d["dtype"] == "uint8"
    assert d["level_voxel_sizes"] is None


def test_describe_of_a_stack_is_shape_and_dtype(tmp_path):
    d = describe(_slices(tmp_path / "stack", n=3, shape=(6, 7)))
    assert d["format"] == "image_stack"
    assert d["shape"] == (3, 6, 7)
    assert d["meta"] is None, "an image file has pixels and no physical scale"


# --------------------------------------------------------------------------- #
# a container holds SEVERAL arrays, which is the ordinary case
# --------------------------------------------------------------------------- #
def _bag(tmp_path):
    """The shape a real ground-truth file has: several crops, one frame each."""
    return _h5(tmp_path / "bag.h5", {
        "z0100": (np.zeros((4, 4, 4), "uint64"),
                  {"voxel_offset": np.asarray([100, 10, 20], "int64")}),
        "z0200": (np.ones((6, 6, 6), "uint64"),
                  {"voxel_offset": np.asarray([200, 30, 40], "int64")}),
    }, root_attrs={"voxel_size": np.asarray([8.0, 8.0, 8.0]), "axes": "zyx",
                   "units": "nm"})


def test_the_container_listing_names_every_dataset_with_its_own_offset(tmp_path):
    from neu_vol.backends.hdf5 import describe_datasets

    entries = describe_datasets(_bag(tmp_path))
    assert sorted(entries) == ["/z0100", "/z0200"]
    assert entries["/z0100"]["shape"] == (4, 4, 4)
    assert entries["/z0100"]["voxel_offset"] == (100.0, 10.0, 20.0)
    assert entries["/z0200"]["shape"] == (6, 6, 6)
    # the root's frame fills in for a dataset that does not repeat it
    assert entries["/z0200"]["voxel_size"] == (8.0, 8.0, 8.0)
    assert entries["/z0200"]["axes"] == "zyx"


def test_describe_describes_a_container_rather_than_refusing(tmp_path):
    """A multi-array container is listed, with no single shape, dtype or meta."""
    path = _bag(tmp_path)
    d = describe(path)
    assert d["format"] == "hdf5"
    assert d["dataset"] is None
    assert d["shape"] is None and d["dtype"] is None and d["meta"] is None
    assert d["levels"] == {}
    assert sorted(d["datasets"]) == ["/z0100", "/z0200"]
    assert d["datasets"]["/z0100"]["voxel_offset"] == (100.0, 10.0, 20.0)


def test_naming_a_dataset_resolves_it_fully(tmp_path):
    path = _bag(tmp_path)
    d = describe(path, dataset="/z0200")
    assert d["dataset"] == "/z0200"
    assert d["shape"] == (6, 6, 6)
    assert d["meta"]["voxel_offset"] == (200, 30, 40)
    assert sorted(d["datasets"]) == ["/z0100", "/z0200"]


def test_an_hdf5_with_no_volumetric_dataset_is_not_a_volume(tmp_path):
    path = _h5(tmp_path / "flat.h5", {"table": (np.zeros((4, 2), "uint8"), {})})
    with pytest.raises(FileNotFoundError, match="no 3D\\+ dataset"):
        describe(path)


def test_the_consumers_that_need_one_array_say_so(tmp_path):
    """Consumers needing one array refuse a container, naming its datasets."""
    from neu_vol.ops.create import plan_volume
    from neu_vol.source_metadata import require_one_array

    path = _bag(tmp_path)
    with pytest.raises(ValueError) as e:
        require_one_array(describe(path), path, "create --like")
    assert "container of 2" in str(e.value)
    assert "/z0100" in str(e.value) and "neu-vol info" in str(e.value)

    # `create --like` has no dataset argument, so a container is ambiguous
    with pytest.raises(ValueError, match="needs one array"):
        plan_volume(str(tmp_path / "like"), like=path, shape=(6, 6, 6),
                    voxel_size=(8, 8, 8), dtype="uint64")
    plan = plan_volume(str(tmp_path / "like2"), like=_framed(tmp_path / "one.h5"))
    assert plan["voxel_size"] == (40.0, 8.0, 8.0)


def test_info_lists_a_container_instead_of_refusing(tmp_path, capsys):
    from neu_vol import cli

    assert cli.cmd_info(cli._parse_args(["info", _bag(tmp_path)])) == 0
    out = capsys.readouterr().out
    assert "2 volumetric datasets" in out
    assert "/z0100" in out and "/z0200" in out
    assert "(100, 10, 20)" in out, "each crop's own offset is the point of the listing"
    assert "--dataset" in out, "say how to get the full report on one"


def test_info_on_a_container_with_no_frame_says_which_names_it_looked_for(tmp_path,
                                                                         capsys):
    """Lists the attribute names it looked for, and the flag to name others."""
    from neu_vol import cli

    path = _h5(tmp_path / "plain.h5", {"a": (np.zeros((2, 2, 2), "uint8"), {}),
                                       "b": (np.zeros((2, 2, 2), "uint8"), {})})
    cli.cmd_info(cli._parse_args(["info", path]))
    out = capsys.readouterr().out
    assert "voxel_size" in out and "voxel_offset" in out and "axes" in out
    assert "--voxel-size-field" in out


def test_info_reports_which_dataset_it_described(tmp_path, capsys):
    from neu_vol import cli

    cli.cmd_info(cli._parse_args(["info", _bag(tmp_path), "--dataset", "/z0100"]))
    out = capsys.readouterr().out
    assert "dataset     /z0100" in out


# --------------------------------------------------------------------------- #
# the ops that need a chunked volume must SAY so
# --------------------------------------------------------------------------- #
def _pyramid(tmp_path):
    from neu_vol import convert
    from neu_vol.backends.tensorstore import TensorStoreBackend
    from neu_vol.profiles import zarr3_create_spec

    src = str(tmp_path / "src.zarr")
    data = np.arange(8 * 8 * 8, dtype="uint16").reshape(8, 8, 8)
    be = TensorStoreBackend.create(
        zarr3_create_spec("local", src, data.shape, "uint16",
                          dimension_names=("z", "y", "x"), chunk=(4, 4, 4)),
        delete_existing=True)
    be.write_region(tuple(slice(0, s) for s in data.shape), data)
    dst = str(tmp_path / "vol")
    convert(src, dst, voxel_size=(8, 8, 8), chunk=(4, 4, 4), factors=[(2, 2, 2)],
            min_dim=4, delete_existing=True)
    return dst


def test_require_chunked_volume_names_the_op_and_the_way_out(tmp_path):
    with pytest.raises(ValueError) as e:
        require_chunked_volume("hdf5", "/p/piece.h5", "neu-vol relabel")
    msg = str(e.value)
    assert "neu-vol relabel" in msg and "hdf5" in msg
    assert "neu-vol convert" in msg, "say what to do, not just what is wrong"
    # and it is a no-op for the formats that do have chunks
    assert require_chunked_volume("zarr3", "/p/v", "x") == "zarr3"


def test_relabel_refuses_a_container_before_reading_anything(tmp_path):
    from neu_vol.ops.relabel import plan_relabel

    with pytest.raises(ValueError, match="needs a chunked multiscale volume"):
        plan_relabel(_framed(tmp_path / "piece.h5"), in_place=True)


def test_write_refuses_a_container_as_the_DESTINATION(tmp_path):
    from neu_vol.ops.write import plan_subvolume_write

    piece = _framed(tmp_path / "piece.h5")
    with pytest.raises(ValueError, match="needs a chunked multiscale volume"):
        plan_subvolume_write(piece, piece, (0, 0, 0))


def test_downsample_refuses_a_container(tmp_path):
    """Both the op and the CLI, which plans before calling the op."""
    from neu_vol import cli
    from neu_vol.ops.rebuild import rebuild_pyramid

    piece = _framed(tmp_path / "piece.h5")
    with pytest.raises(ValueError, match="needs a chunked multiscale volume"):
        rebuild_pyramid(piece, start_level=0, kind="image")
    with pytest.raises(SystemExit, match="needs a chunked multiscale volume"):
        cli.cmd_downsample(cli._parse_args(["downsample", piece, "--start-level", "0",
                                            "--dry-run"]))


def test_the_read_only_ops_do_not_refuse(tmp_path):
    """`create --like` and `convert` accept a container (convert without --voxel-size)."""
    from neu_vol import convert
    from neu_vol.backends.base import open_backend
    from neu_vol.ops.create import plan_volume

    piece = _framed(tmp_path / "piece.h5")
    plan = plan_volume(str(tmp_path / "like"), like=piece)
    assert plan["voxel_size"] == (40.0, 8.0, 8.0)

    dst = str(tmp_path / "out")
    convert(piece, dst, chunk=(2, 2, 2), multiscale=False, delete_existing=True)
    out = open_backend({"backend": "zarr3", "path": os.path.join(dst, "0")})
    np.testing.assert_array_equal(
        out.read_region((slice(0, 2), slice(0, 3), slice(0, 4))),
        np.arange(2 * 3 * 4, dtype="uint16").reshape(2, 3, 4))
    assert describe(dst)["meta"]["voxel_size"] == (40.0, 8.0, 8.0)
    assert describe(dst)["meta"]["offset"] == (80.0, 24.0, 32.0), "origin carried through"


# --------------------------------------------------------------------------- #
# `write`'s own resolver now delegates, so the two cannot disagree
# --------------------------------------------------------------------------- #
def test_source_spec_builds_the_form_each_backend_wants(tmp_path):
    """The image-stack backend reads `source`, not `path`."""
    from neu_vol.ops.write import source_spec

    piece = _framed(tmp_path / "piece.h5")
    assert source_spec(piece) == {"backend": "hdf5", "path": piece, "dataset": "/data"}

    stack = _slices(tmp_path / "stack", n=2)
    assert source_spec(stack) == {"backend": "image_stack", "source": stack}
    assert source_spec(stack, "image_stack") == {"backend": "image_stack",
                                                "source": stack}

    with pytest.raises(ValueError, match="could not tell what"):
        source_spec(str(tmp_path / "nothing.dat"))


def test_a_named_dataset_survives_the_round_trip(tmp_path):
    from neu_vol.ops.write import source_spec

    bag = _bag(tmp_path)
    assert source_spec(bag, dataset="/z0200")["dataset"] == "/z0200"


# --------------------------------------------------------------------------- #
# a `describe` result shows itself: the same table `neu-vol info` prints
# --------------------------------------------------------------------------- #
def test_a_description_is_still_a_dict(tmp_path):
    d = describe(_framed(tmp_path / "piece.h5"))
    assert isinstance(d, dict)
    assert d["shape"] == (2, 3, 4)
    assert dict(d)["format"] == "hdf5", "dict() strips the presentation, keeps the data"


def test_printing_a_description_gives_the_table_not_the_mapping(tmp_path):
    # One file, described once: rewriting a path whose handle is cached fails in HDF5.
    d = describe(_framed(tmp_path / "piece.h5"))
    text = str(d)
    assert "format      hdf5" in text
    assert "voxel size  40x8x8 nm" in text
    assert "level" in text and "40x8x8" in text
    assert repr(d) == text, "a bare `d` in a cell too"


def test_the_cli_and_the_repr_render_from_ONE_implementation(tmp_path, capsys):
    from neu_vol import cli

    path = _framed(tmp_path / "piece.h5")
    cli.cmd_info(cli._parse_args(["info", path]))
    printed = capsys.readouterr().out
    for line in str(describe(path)).splitlines():
        assert line in printed


def test_the_repr_makes_no_store_reads(tmp_path, capsys):
    """Provenance is read by the CLI only: a `__repr__` must not make network calls."""
    dst = _pyramid(tmp_path)
    assert "provenance" not in str(describe(dst))
    from neu_vol import cli

    cli.cmd_info(cli._parse_args(["info", dst]))
    assert "provenance" in capsys.readouterr().out, "...but the CLI still shows it"


def test_a_container_frame_is_one_row_per_dataset(tmp_path):
    """Numbers, not the formatted strings of the text table."""
    pytest.importorskip("pandas")
    frame = describe(_bag(tmp_path)).frame()
    assert list(frame.index) == ["/z0100", "/z0200"]
    assert frame.loc["/z0100", "shape"] == (4, 4, 4)
    assert frame.loc["/z0100", "voxel_offset"] == (100, 10, 20)
    assert frame.loc["/z0200", "dtype"] == "uint64"


def test_a_volume_frame_is_one_row_per_level(tmp_path):
    pytest.importorskip("pandas")
    frame = describe(_pyramid(tmp_path)).frame()
    assert list(frame.index) == [0, 1]
    assert frame.loc[0, "shape"] == (8, 8, 8)
    assert frame.loc[0, "voxel_size"] == (8.0, 8.0, 8.0)
    assert frame.loc[1, "voxel_size"] == (16.0, 16.0, 16.0), "each level's OWN size"
    assert frame.loc[0, "shard"] is None, "unsharded: no shard to report"


def test_the_html_repr_needs_no_pandas(tmp_path, monkeypatch):
    """pandas is optional: without it the HTML repr is preformatted text."""
    import builtins

    real = builtins.__import__

    def blocked(name, *a, **k):
        if name == "pandas":
            raise ImportError("blocked")
        return real(name, *a, **k)

    d = describe(_framed(tmp_path / "piece.h5"))
    monkeypatch.setattr(builtins, "__import__", blocked)
    html = d._repr_html_()
    assert "40x8x8" in html and "<pre>" in html
    with pytest.raises(ImportError, match="pip install pandas"):
        d.frame()


# --------------------------------------------------------------------------- #
# open_hdf5: the front door for a file you are pointing at by hand
# --------------------------------------------------------------------------- #
def test_open_hdf5_finds_the_sole_dataset(tmp_path):
    from neu_vol import open_hdf5

    be = open_hdf5(_framed(tmp_path / "piece.h5"))
    assert be.shape == (2, 3, 4)
    np.testing.assert_array_equal(
        be.read_region((slice(0, 2), slice(0, 3), slice(0, 4))),
        np.arange(2 * 3 * 4, dtype="uint16").reshape(2, 3, 4))


def test_open_hdf5_requires_a_dataset_for_a_container(tmp_path):
    from neu_vol import open_hdf5

    path = _bag(tmp_path)
    with pytest.raises(KeyError, match="z0100.*z0200"):
        open_hdf5(path)
    assert open_hdf5(path, "/z0200").shape == (6, 6, 6)


def test_open_hdf5_shares_the_process_handle_cache(tmp_path):
    """One open h5py.File per array, not one per call."""
    from neu_vol import open_hdf5

    path = _framed(tmp_path / "piece.h5")
    assert open_hdf5(path) is open_hdf5(path)


def test_open_hdf5_refuses_a_remote_path_with_guidance(tmp_path):
    from neu_vol import open_hdf5

    with pytest.raises(ValueError, match="ordinary filesystem path"):
        open_hdf5("s3://my-bucket/pieces/a.h5")


def test_open_hdf5_needs_no_detection(tmp_path, monkeypatch):
    from neu_vol import open_hdf5, source_metadata

    def refuse(*a, **k):
        raise AssertionError("open_hdf5 ran detection")

    monkeypatch.setattr(source_metadata, "detect_backend", refuse)
    assert open_hdf5(_framed(tmp_path / "piece.h5")).shape == (2, 3, 4)


def test_read_piece_refuses_a_read_that_would_hang(tmp_path):
    """A whole production level is terabytes; a read that size looks like a hung kernel."""
    from neu_vol import read_piece

    vol = _pyramid(tmp_path)               # 8^3 uint16 = 4 KiB, two levels
    with pytest.raises(ValueError) as e:
        read_piece(vol, max_bytes=512)
    msg = str(e.value)
    assert "crop=" in msg and "max_bytes" in msg, "the error says what to do"

    assert read_piece(vol, max_bytes=None).shape == (8, 8, 8)
    assert read_piece(vol).shape == (8, 8, 8)
    assert read_piece(vol, crop=((0, 0, 0), (4, 4, 4)),
                      max_bytes=512).shape == (4, 4, 4)


def test_the_refusal_names_a_level_that_would_fit(tmp_path):
    """Predicted from the recorded per-level voxel sizes, not by opening anything."""
    from neu_vol import read_piece

    vol = _pyramid(tmp_path)
    with pytest.raises(ValueError, match="level=1 would be"):
        read_piece(vol, max_bytes=600)


def test_a_piece_is_named_after_its_source(tmp_path):
    """`stem/dataset`, or just the stem; either half alone collides across files."""
    from neu_vol.piece import piece_name
    from neu_vol import read_piece

    assert piece_name("data/gt_v1_eval.h5", "/vol_03700") == "gt_v1_eval/vol_03700"
    assert piece_name("piece.h5") == "piece"
    assert piece_name("s3://my-bucket/dataset/image_v3.zarr") == "image_v3"
    assert piece_name("/p/seg_v1") == "seg_v1"
    assert piece_name("/p/x.h5", "/g/nested/vol") == "x/vol"

    path = _framed(tmp_path / "gt.h5")
    assert read_piece(path).name == "gt/data"
    assert read_piece(path, name="mine").name == "mine"
    assert read_piece(path).with_name("later").name == "later"


def test_a_piece_can_be_read_in_the_dtype_it_will_be_used_in(tmp_path):
    from neu_vol import read_piece

    path = _framed(tmp_path / "gt.h5")
    piece = read_piece(path)
    wide = read_piece(path, dtype="uint64")

    assert piece.dtype == np.dtype("uint16") and wide.dtype == np.dtype("uint64")
    np.testing.assert_array_equal(wide.array, piece.array)
    assert wide.frame == piece.frame and wide.name == piece.name


def test_a_narrowing_read_warns_because_a_wrapped_label_id_is_another_label_id(tmp_path,
                                                                              caplog):
    """Allowed but warned: a wrapped label id is just another valid id."""
    import logging

    from neu_vol import read_piece

    path = _framed(tmp_path / "big.h5",
                   data=np.array([[[0, 1, 65538]]], dtype="uint32"))
    with caplog.at_level(logging.WARNING, logger="neu_vol.piece"):
        narrow = read_piece(path, dtype="uint16")

    assert narrow.array.tolist() == [[[0, 1, 2]]], "65538 wrapped, which is the hazard"
    assert "not a safe cast" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="neu_vol.piece"):
        read_piece(path, dtype="uint64")
    assert not caplog.text, "widening among unsigned ints loses nothing, so it says nothing"
