import pytest

from neu_vol import open_backend


@pytest.mark.parametrize("spec", [{"backend": "does-not-exist"}, {"path": "/tmp/x"}])
def test_open_backend_needs_a_known_tag(spec):
    with pytest.raises(ValueError):
        open_backend(spec)


def test_open_missing_store_raises(tmp_path):
    with pytest.raises(Exception):
        open_backend({"backend": "zarr3", "path": str(tmp_path / "does-not-exist")})
