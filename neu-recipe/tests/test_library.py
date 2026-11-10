"""The volume library: loading, its checks, lookup, and the listing."""

from __future__ import annotations

import pytest

from neu_recipe import library as lib


def write(tmp_path, **files):
    for stem, text in files.items():
        (tmp_path / f"{stem}.toml").write_text(text)
    return tmp_path


SEG = '[volume."fly/seg"]\nkind = "segmentation"\nlocation = "precomputed://s3://my-bucket/seg/"\n'
IMG = '[volume."fly/img"]\nkind = "image"\nlocation = "s3://my-bucket/img"\nstatus = "superseded"\nsuperseded_by = "fly/img2"\n'


def test_entries_gain_name_and_specimen_and_a_normalized_location(tmp_path):
    volumes = lib.load(write(tmp_path, fly=SEG + IMG))
    assert volumes["fly/seg"] == {"kind": "segmentation", "location": "s3://my-bucket/seg",
                                  "name": "fly/seg", "specimen": "fly"}
    assert lib.location("fly/img", volumes) == "s3://my-bucket/img"


@pytest.mark.parametrize("files, match", [
    ({"fly": '[volume."worm/seg"]\nkind = "image"\nlocation = "x"\n'}, "must start with"),
    ({"fly": '[volume."fly/seg"]\nlocation = "x"\n'}, "has no kind"),
    ({"fly": '[volume."fly/seg"]\nkind = "movie"\nlocation = "x"\n'}, "is not one of"),
    ({"fly": SEG + SEG.replace("fly/seg", "fly/seg-copy")}, "are the same location"),
])
def test_a_bad_library_fails_on_load(tmp_path, files, match):
    with pytest.raises(lib.LibraryError, match=match):
        lib.load(write(tmp_path, **files))


def test_an_empty_directory_is_an_error(tmp_path):
    with pytest.raises(lib.LibraryError, match="no library files"):
        lib.load(tmp_path)


def test_an_unknown_name_suggests_near_ones(tmp_path):
    with pytest.raises(lib.LibraryError, match="did you mean fly/seg"):
        lib.get("fly/sge", directory=write(tmp_path, fly=SEG))


def test_the_directory_defaults_to_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv(lib.ENV_VAR, str(write(tmp_path, fly=SEG)))
    assert lib.get("fly/seg")["kind"] == "segmentation"
    monkeypatch.delenv(lib.ENV_VAR)
    with pytest.raises(lib.LibraryError, match=lib.ENV_VAR):
        lib.load()


def test_the_listing_marks_superseded_volumes_and_probe_results(tmp_path):
    volumes = lib.load(write(tmp_path, fly=SEG + IMG))
    plain = dict(lib.listing(volumes))
    assert all(found is None for found in plain.values())
    assert any("[superseded by fly/img2]" in line for line in plain)

    checked = list(lib.listing(volumes, check=True,
                               probe_fn=lambda loc: "info" if "seg" in loc else None))
    assert [found for _line, found in checked] == [True, False]
    assert checked[0][0].startswith("ok  ") and checked[1][0].startswith("MISS")
