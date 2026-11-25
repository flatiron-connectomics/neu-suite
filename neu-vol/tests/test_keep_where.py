"""`keep_where`: a volume written as 0 wherever another volume, in the same frame, is 0.

Built for organelles labelled by their containing neuron: the neuron segmentation kept
where an organelle segmentation is nonzero.
"""

import numpy as np
import pytest

from neu_vol import cli, convert
from neu_vol.backends.base import open_backend
from neu_vol.backends.tensorstore import TensorStoreBackend
from neu_vol.ops._multiscale import StaleManifest
from neu_vol.profiles import zarr3_create_spec


def _volume(tmp_path, name, data, voxel=(8, 8, 8), offset=None):
    """An OME-zarr volume with a recorded frame, from an array."""
    raw = str(tmp_path / f"{name}.raw.zarr")
    be = TensorStoreBackend.create(
        zarr3_create_spec("local", raw, data.shape, str(data.dtype),
                          dimension_names=("z", "y", "x"), chunk=(8, 8, 8)),
        delete_existing=True)
    be.write_region(tuple(slice(0, s) for s in data.shape), data)
    out = str(tmp_path / f"{name}.zarr")
    convert(raw, out, voxel_size=voxel, offset=offset, profile="local", chunk=(8, 8, 8),
            kind="segmentation", multiscale=False, delete_existing=True)
    return out


def _read(volume, level=0):
    be = open_backend({"backend": "zarr3", "path": f"{volume}/{level}"})
    return be.read_region(tuple(slice(0, s) for s in be.shape))


@pytest.fixture
def world(tmp_path):
    rng = np.random.default_rng(0)
    neurons = rng.integers(1, 6, (16, 24, 32)).astype(np.uint64)
    # The organelle volume is deeper in z and shorter in y than the neurons, as the real
    # ones are: outside it nothing may be kept.
    organelles = np.where(rng.random((20, 16, 32)) < 0.3, 9, 0).astype(np.uint64)
    return dict(tmp_path=tmp_path, neurons=neurons, organelles=organelles,
                src=_volume(tmp_path, "neurons", neurons),
                where=_volume(tmp_path, "organelles", organelles))


def _expected(w):
    keep = np.zeros(w["neurons"].shape, bool)
    keep[:, :16, :] = w["organelles"][:16] != 0
    return np.where(keep, w["neurons"], 0)


def test_only_voxels_where_the_other_volume_is_nonzero_are_kept(world):
    dst = str(world["tmp_path"] / "kept.zarr")
    convert(world["src"], dst, profile="local", chunk=(8, 8, 8), multiscale=False,
            keep_where=world["where"], delete_existing=True)
    out = _read(dst)
    np.testing.assert_array_equal(out, _expected(world))
    assert not out[:, 16:, :].any(), "beyond the other volume's extent nothing is kept"
    assert out.any()


def test_it_composes_with_a_crop_in_source_coordinates(world):
    dst = str(world["tmp_path"] / "kept_crop.zarr")
    convert(world["src"], dst, profile="local", chunk=(8, 8, 8), multiscale=False,
            keep_where=world["where"], crop_start=(4, 8, 8), crop_stop=(12, 24, 24),
            delete_existing=True)
    np.testing.assert_array_equal(_read(dst), _expected(world)[4:12, 8:24, 8:24])


@pytest.mark.parametrize("voxel, offset", [((16, 16, 16), None), ((8, 8, 8), (8, 0, 0))])
def test_a_volume_in_another_frame_is_refused(world, voxel, offset):
    other = _volume(world["tmp_path"], f"other_{voxel[0]}_{offset}", world["organelles"],
                    voxel=voxel, offset=offset)
    with pytest.raises(ValueError, match="must share a frame"):
        convert(world["src"], str(world["tmp_path"] / "refused.zarr"), profile="local",
                chunk=(8, 8, 8), multiscale=False, keep_where=other, delete_existing=True)


def test_a_resume_keeping_by_another_volume_is_refused(world):
    dst = str(world["tmp_path"] / "resume.zarr")
    convert(world["src"], dst, profile="local", chunk=(8, 8, 8), multiscale=False,
            keep_where=world["where"], delete_existing=True)
    other = _volume(world["tmp_path"], "other_org", world["organelles"][::-1].copy())
    with pytest.raises(StaleManifest, match="another source"):
        convert(world["src"], dst, profile="local", chunk=(8, 8, 8), multiscale=False,
                keep_where=other, resume=True)


def test_the_cli_passes_keep_where_through(world):
    dst = str(world["tmp_path"] / "cli.zarr")
    assert cli.cmd_convert(cli._parse_args(
        ["convert", "--src", world["src"], "--dst", dst, "--format", "zarr",
         "--chunk", "8,8,8", "--single-level", "--keep-where", world["where"],
         "--serial"])) == 0
    np.testing.assert_array_equal(_read(dst), _expected(world))
