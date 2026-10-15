"""Stage-1 block workers retry a transient store error instead of failing the stage."""

import numpy as np
import pytest

from blockrun import iter_blocks

#: What a worker saw when DNS failed mid-run (curl_code=6), trimmed.
TRANSIENT = ('UNAVAILABLE: Error opening "neuroglancer_precomputed" driver: CURL error '
             'Could not resolve hostname [curl_code=\'6\']')


class _Flaky:
    """A backend whose first ``failures`` reads raise ``message``."""

    def __init__(self, array, failures, message=TRANSIENT):
        self.array, self.failures, self.message, self.reads = array, failures, message, 0

    def read_region(self, region):
        self.reads += 1
        if self.reads <= self.failures:
            raise ValueError(self.message)
        return self.array[region]


@pytest.fixture
def flaky(monkeypatch):
    import neu_vol.backends.base as base
    import neu_vol.retry as retry

    monkeypatch.setattr(retry.time, "sleep", lambda s: None)
    vol = np.zeros((32, 32, 32), np.uint64)
    vol[4:28, 4:28, 4:28] = 7

    def install(failures, message=TRANSIENT):
        be = _Flaky(vol, failures, message)
        monkeypatch.setattr(base, "open_backend", lambda spec: be)
        return be
    return install


def _block():
    (block,) = iter_blocks((32, 32, 32), (32, 32, 32))
    return block


def test_the_index_worker_retries_a_transient_read(flaky):
    from neu_morpho.ops.index_segments import _index_block

    be = flaky(failures=2)
    key, partials = _index_block(_block(), seg_spec={}, fullres_factor=(1, 1, 1))
    assert be.reads == 3 and set(partials) == {7}


def test_the_mesh_worker_retries_a_transient_read(flaky, tmp_path):
    pytest.importorskip("vol2mesh")
    from neu_morpho.config import MeshConfig
    from neu_morpho.ops.meshify import _chunk_block

    be = flaky(failures=1)
    index, status = _chunk_block(_block(), seg_spec={}, chunked_dir=str(tmp_path),
                                 mesh_cfg=MeshConfig(mesh_scale=0, block_shape=(32, 32, 32)),
                                 allow=None, mesh_voxel_size=(8, 8, 8))
    assert be.reads == 2 and status == "written"


def test_a_persistent_error_still_fails_the_block(flaky):
    from neu_morpho.ops.index_segments import _index_block

    be = flaky(failures=99)
    with pytest.raises(ValueError, match="UNAVAILABLE"):
        _index_block(_block(), seg_spec={}, fullres_factor=(1, 1, 1))
    assert be.reads == 5


def test_a_permanent_error_is_not_retried(flaky):
    from neu_morpho.ops.index_segments import _index_block

    be = flaky(failures=99, message="PERMISSION_DENIED: AccessDenied")
    with pytest.raises(ValueError, match="PERMISSION_DENIED"):
        _index_block(_block(), seg_spec={}, fullres_factor=(1, 1, 1))
    assert be.reads == 1
