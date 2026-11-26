"""Yes3Cache keys: yes3 stores each item under its key as a file name."""
import subprocess
import sys

import pytest

from neu_draw import cache as cache_mod
from neu_draw.cache import _normalize_key

SOURCE_KEY = ("mesh", "s3://bucket/volume", 123, 1)


def test_a_safe_str_key_is_kept():
    assert _normalize_key("ann") == "ann"
    assert _normalize_key("left_t4_partners-v2.1") == "left_t4_partners-v2.1"


def test_a_source_key_becomes_a_readable_file_name():
    name = _normalize_key(SOURCE_KEY)
    assert name.startswith("mesh__123__1__")
    assert "/" not in name and ":" not in name and "'" not in name


def test_keys_differing_only_in_an_unsafe_part_stay_apart():
    other = ("mesh", "s3://bucket/other-volume", 123, 1)
    assert _normalize_key(SOURCE_KEY) != _normalize_key(other)


def test_a_key_is_the_same_in_every_process():
    code = ("from neu_draw.cache import _normalize_key; "
            f"print(_normalize_key({SOURCE_KEY!r}))")
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    assert out.stdout.strip() == _normalize_key(SOURCE_KEY)


def test_an_object_key_is_refused_rather_than_never_found_again():
    with pytest.raises(TypeError):
        _normalize_key(("mesh", object()))


def test_a_source_key_is_found_after_a_restart(tmp_path):
    pytest.importorskip("yes3")
    first = cache_mod.Yes3Cache(tmp_path / "cache")
    first[SOURCE_KEY] = [1, 2, 3]
    assert SOURCE_KEY in first
    again = cache_mod.Yes3Cache(tmp_path / "cache")
    assert SOURCE_KEY in again
    assert again[SOURCE_KEY] == [1, 2, 3]
    assert [p.name for p in (tmp_path / "cache").iterdir() if p.suffix == ".pkl"] == [
        _normalize_key(SOURCE_KEY) + ".pkl"]
