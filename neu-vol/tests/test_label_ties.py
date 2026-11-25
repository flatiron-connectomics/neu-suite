"""`label_ties`: whether background or an object wins a tie in a segmentation pyramid.

The default (smallest label, i.e. 0) erodes small objects in a sparse volume at every
level; "nonzero" keeps them. Nothing about the default may change.
"""

import numpy as np
import pytest

from neu_vol import cli, convert, rebuild_pyramid
from neu_vol.backends.base import open_backend
from neu_vol.backends.tensorstore import TensorStoreBackend
from neu_vol.ops._multiscale import StaleManifest
from neu_vol.profiles import zarr3_create_spec
from neu_vol.pyramid import get_reducer, mode_downsample


def _window(values):
    return np.asarray(values, np.uint64).reshape(2, 2, 2)


@pytest.mark.parametrize("values, default, nonzero", [
    ([0, 0, 0, 0, 7, 7, 7, 7], 0, 7),        # a 4-4 tie: the object wins it
    ([0, 0, 0, 0, 0, 7, 7, 7], 0, 0),        # a strict majority of background still wins
    ([0, 0, 3, 3, 5, 5, 9, 9], 0, 3),        # a four-way tie: the smallest OBJECT
    ([0, 0, 0, 3, 3, 3, 5, 5], 0, 3),        # 0 ties 3 for the most; 3 takes it
    ([3, 3, 3, 3, 5, 5, 5, 5], 3, 3),        # no background: unchanged
])
def test_the_tie_rule(values, default, nonzero):
    w = _window(values)
    assert mode_downsample(w, (2, 2, 2))[0, 0, 0] == default
    assert mode_downsample(w, (2, 2, 2), prefer_nonzero=True)[0, 0, 0] == nonzero


def test_a_tie_rule_is_refused_where_there_are_no_ties():
    with pytest.raises(ValueError, match="averaged"):
        get_reducer("image", "nonzero")
    with pytest.raises(ValueError, match="unknown label_ties"):
        get_reducer("segmentation", "largest")
    assert get_reducer("image") is get_reducer("image", "smallest")


def _source(path, data):
    be = TensorStoreBackend.create(
        zarr3_create_spec("local", path, data.shape, str(data.dtype),
                          dimension_names=("z", "y", "x"), chunk=(8, 8, 8)),
        delete_existing=True)
    be.write_region(tuple(slice(0, s) for s in data.shape), data)
    return path


def _read(volume, level):
    be = open_backend({"backend": "zarr3", "path": f"{volume}/{level}"})
    return be.read_region(tuple(slice(0, s) for s in be.shape))


@pytest.fixture
def sheet(tmp_path):
    """One-voxel-thick sheets: every 2x2x2 window holds 4 object voxels and 4 background."""
    data = np.zeros((16, 16, 16), np.uint64)
    data[::2] = 5
    return _source(str(tmp_path / "sheet_src.zarr"), data)


def _convert(src, dst, **kw):
    return convert(src, dst, voxel_size=(8, 8, 8), profile="local", chunk=(8, 8, 8),
                   kind="segmentation", min_dim=4, **kw)


def test_the_default_pyramid_loses_a_half_filled_object_and_nonzero_keeps_it(tmp_path, sheet):
    a, b = str(tmp_path / "smallest.zarr"), str(tmp_path / "nonzero.zarr")
    _convert(sheet, a, delete_existing=True)
    _convert(sheet, b, delete_existing=True, label_ties="nonzero")
    assert not _read(a, 1).any()
    assert (_read(b, 1) == 5).all()
    np.testing.assert_array_equal(_read(a, 0), _read(b, 0))     # level 0 is a copy either way


def test_downsample_rebuilds_with_the_tie_rule(tmp_path, sheet):
    dst = str(tmp_path / "rebuilt.zarr")
    _convert(sheet, dst, delete_existing=True)
    assert not _read(dst, 1).any()
    rebuild_pyramid(dst, start_level=0, kind="segmentation", min_dim=4, label_ties="nonzero")
    assert (_read(dst, 1) == 5).all()


def test_a_resume_under_another_tie_rule_is_refused(tmp_path, sheet):
    dst = str(tmp_path / "resumed.zarr")
    _convert(sheet, dst, delete_existing=True)
    with pytest.raises(StaleManifest, match="another pyramid reducer"):
        _convert(sheet, dst, resume=True, label_ties="nonzero")
    _convert(sheet, dst, resume=True)                            # the same rule resumes


def test_the_cli_passes_the_tie_rule_through(tmp_path, sheet):
    dst = str(tmp_path / "cli.zarr")
    assert cli.cmd_convert(cli._parse_args(
        ["convert", "--src", sheet, "--dst", dst, "--format", "zarr", "--kind", "segmentation",
         "--voxel-size", "8,8,8", "--chunk", "8,8,8", "--min-dim", "4",
         "--label-ties", "nonzero", "--serial"])) == 0
    assert (_read(dst, 1) == 5).all()
    args = cli._parse_args(["downsample", dst, "--label-ties", "nonzero"])
    assert args.label_ties == "nonzero"
