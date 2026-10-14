"""The DVID source path without a DVID server: address resolution, detection and metadata.

Reading voxels needs a real server and is checked by hand.
"""

import numpy as np
import pytest

from neu_vol import dvid as canonical_dvid
from neu_vol import detect
from neu_vol import source_metadata as sm
from neu_vol.backends import dvid

URL = "dvid://dvid.example.org/93fdbc:main/labels"

#: A believable `/info` for a labelmap, matching the real instance's shape.
INFO = {
    "Base": {"TypeName": "labelmap"},
    "Extended": {
        "BlockSize": [64, 64, 64],
        "VoxelSize": [8, 8, 8],
        "VoxelUnits": ["nanometers"] * 3,
        "MaxDownresLevel": 3,
        "MinPoint": [1024, 1536, 0],
        "MaxPoint": [12799, 9215, 8191],
    },
}


# --------------------------------------------------------------------------- #
# URL form
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("url, parsed", [
    (URL, {"server": "dvid.example.org", "uuid": "93fdbc:main", "instance": "labels"}),
    # a port and a branch both carry ':', so the URL splits on '/' alone
    ("dvid://emdata3:8900/abc9:main/segmentation",
     {"server": "emdata3:8900", "uuid": "abc9:main", "instance": "segmentation"}),
    # neuclease prepends http:// to a bare host, so TLS keeps its scheme
    ("dvid+https://example.org/abc9/labels",
     {"server": "https://example.org", "uuid": "abc9", "instance": "labels"}),
])
def test_url_parses_and_round_trips(url, parsed):
    spec = canonical_dvid.parse_url(url)
    assert {k: spec[k] for k in parsed} == parsed
    assert canonical_dvid.spec_url(spec) == url


@pytest.mark.parametrize("bad", [
    "dvid://server/uuid",                 # no instance
    "dvid://server/uuid/instance/extra",  # too many
    "dvid://server",
    "s3://bucket/prefix",                 # not DVID at all
])
def test_a_malformed_url_raises_and_says_what_was_expected(bad):
    with pytest.raises(ValueError) as e:
        canonical_dvid.parse_url(bad)
    assert "dvid://" in str(e.value)


def test_is_url_recognises_both_schemes_and_nothing_else():
    assert canonical_dvid.is_url(URL) and canonical_dvid.is_url("dvid+https://a/b/c")
    assert not canonical_dvid.is_url("/local/path")
    assert not canonical_dvid.is_url("s3://bucket/prefix")
    assert not canonical_dvid.is_url({"driver": "file", "path": "/x"})


# --------------------------------------------------------------------------- #
# detection and spec building
# --------------------------------------------------------------------------- #
def test_detect_backend_answers_from_the_scheme_without_touching_a_store(monkeypatch):
    """`to_kvstore` would read the URL as a local path and find nothing."""
    def explode(*a, **k):
        raise AssertionError("detect_backend touched a store for a DVID URL")

    monkeypatch.setattr(detect, "to_kvstore", explode)
    assert detect.detect_backend(URL) == "dvid"


def test_location_spec_builds_the_three_forms():
    assert sm.location_spec(URL, "dvid")["backend"] == "dvid"
    assert sm.location_spec(URL, "dvid")["instance"] == "labels"
    assert sm.location_spec("/some/dir", "image_stack") == {
        "backend": "image_stack", "source": "/some/dir"}
    assert sm.location_spec("/v", "zarr3") == {"backend": "zarr3", "path": "/v"}


def test_level_spec_selects_a_dvid_scale_by_index():
    spec = sm.level_spec(URL, "dvid", 2)
    assert spec["scale_index"] == 2 and spec["instance"] == "labels"


def test_no_format_can_shadow_a_dvid_instance():
    """`other_format_markers` would otherwise read `dvid://...` as a local path."""
    assert detect.other_format_markers(URL, "dvid") == []


# --------------------------------------------------------------------------- #
# metadata, against a stubbed /info
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _no_cached_nodes():
    """Ref resolutions are memoized per process; clear them between tests."""
    canonical_dvid.clear_node_cache()
    yield
    canonical_dvid.clear_node_cache()


@pytest.fixture()
def stub_info(monkeypatch):
    """Stub both the instance info and the ref resolution, on `neu_vol.dvid`, so no test
    touches a server."""
    monkeypatch.setattr(canonical_dvid, "instance_info", lambda spec: INFO)
    monkeypatch.setattr(canonical_dvid, "resolve_node",
                        lambda spec, prefer_locked=False: {
                            "ref": str(spec.get("uuid")),
                            "uuid": "846e3a" if prefer_locked else "d38898",
                            "locked": bool(prefer_locked),
                            "walked": 1 if prefer_locked else 0})


def test_geometry_is_origin_anchored_not_translated_by_minpoint(stub_info):
    """MinPoint is recorded, not subtracted: a voxel index means the same as in DVID."""
    geom = dvid.geometry(INFO, canonical_dvid.parse_url(URL))
    assert geom["shape0"] == (8192, 9216, 12800)      # zyx, MaxPoint + 1
    assert geom["min_point"] == (0, 1536, 1024)       # recorded, not subtracted
    assert geom["chunks"] == (64, 64, 64)
    assert geom["voxel_size"] == (8.0, 8.0, 8.0)


def test_read_source_metadata_reports_a_segmentation(stub_info):
    """Never `image`, which would average label ids."""
    meta = sm.read_source_metadata(sm.location_spec(URL, "dvid"))
    assert meta["kind"] == "segmentation"
    assert meta["voxel_size"] == (8.0, 8.0, 8.0)
    assert meta["offset"] == (0.0, 0.0, 0.0)
    assert meta["units"] == "nm" and meta["has_channels"] is False


def test_data_spec_carries_the_dvid_backend_through(stub_info):
    """`data_spec` selects the reader, so it must not contradict detection."""
    meta = sm.read_source_metadata(sm.location_spec(URL, "dvid"))
    assert meta["data_spec"]["backend"] == "dvid"
    assert meta["data_spec"]["instance"] == "labels"


def test_level_voxel_sizes_follow_dvids_own_downres_depth(stub_info):
    sizes = sm.read_level_voxel_sizes(sm.location_spec(URL, "dvid"))
    assert sizes == [(8.0,) * 3, (16.0,) * 3, (32.0,) * 3, (64.0,) * 3]


def test_level_shape_halves_every_axis():
    assert dvid.level_shape((8192, 9216, 12800), 0) == (8192, 9216, 12800)
    assert dvid.level_shape((8192, 9216, 12800), 3) == (1024, 1152, 1600)
    assert dvid.level_shape((10, 10, 10), 2) == (3, 3, 3)      # ceil, not floor


# --------------------------------------------------------------------------- #
# refusals
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("info, match", [
    ({**INFO, "Base": {"TypeName": "uint8blk"}}, "labelmap"),
    # null extents: created but never written; a zero-shaped volume would "convert" fine
    ({**INFO, "Extended": {**INFO["Extended"], "MinPoint": None, "MaxPoint": None}},
     "never written"),
    ({**INFO, "Extended": {**INFO["Extended"], "MinPoint": [-64, 0, 0]}}, "negative MinPoint"),
])
def test_unusable_geometry_is_refused(info, match):
    with pytest.raises(ValueError, match=match):
        dvid.geometry(info, canonical_dvid.parse_url(URL))


def test_writing_to_dvid_is_refused(stub_info):
    be = dvid.DVIDBackend(canonical_dvid.parse_url(URL))
    with pytest.raises(TypeError, match="read-only"):
        be.write_region((slice(0, 1),) * 3, np.zeros((1, 1, 1), np.uint64))


def test_a_scale_above_maxdownreslevel_is_refused(stub_info):
    with pytest.raises(ValueError, match="MaxDownresLevel"):
        dvid.DVIDBackend({**canonical_dvid.parse_url(URL), "scale_index": 4})


@pytest.mark.parametrize("option", ["supervoxels", "prefer_locked"])
def test_a_dvid_option_on_another_source_raises_rather_than_being_ignored(tmp_path, option):
    """Ignoring `supervoxels` would silently copy agglomerated bodies instead."""
    from neu_vol import convert

    with pytest.raises(ValueError, match="DVID sources only"):
        convert({"backend": "zarr3", "path": str(tmp_path / "x")},
                str(tmp_path / "out"), voxel_size=(8, 8, 8), **{option: True})


@pytest.mark.parametrize("option", ["supervoxels", "prefer_locked"])
def test_to_hdf5_refuses_a_dvid_option_on_another_source(tmp_path, option):
    from neu_vol.ops.pack import pack_hdf5

    with pytest.raises(ValueError, match="DVID sources only"):
        pack_hdf5(str(tmp_path / "x.zarr"), str(tmp_path / "out.h5"), **{option: True})


# --------------------------------------------------------------------------- #
# to-hdf5: a box out of DVID
# --------------------------------------------------------------------------- #
def test_to_hdf5_reads_a_dvid_level_from_the_resolved_node(stub_info):
    """The level's own voxel size, the options applied, and one concrete node."""
    from neu_vol.ops.pack import resolve_source

    spec, frame = resolve_source(URL, level=2,
                                 options={"prefer_locked": True, "supervoxels": True})
    assert spec["backend"] == "dvid" and spec["scale_index"] == 2
    assert spec["uuid"] == "846e3a" and "prefer_locked" not in spec
    assert spec["supervoxels"] is True
    assert frame["voxel_size"] == (32.0,) * 3 and frame["kind"] == "segmentation"


def test_to_hdf5_refuses_a_dvid_level_the_instance_does_not_have(stub_info):
    from neu_vol.ops.pack import resolve_source

    with pytest.raises(ValueError, match=r"no level 4 \(present: 0-3\)"):
        resolve_source(URL, level=4)


# --------------------------------------------------------------------------- #
# ref resolution: a branch ref names a different node tomorrow
# --------------------------------------------------------------------------- #
@pytest.fixture()
def fake_dag(monkeypatch):
    """A lock-and-spawn repo: HEAD open, every ancestor locked."""
    pytest.importorskip("neuclease")
    import neuclease.dvid as nd

    locked = {"d38898": False, "846e3a": True, "5ad376": True,
              # The full-length form, for the placeholder-truncation tests.
              "d38898ac94c8400baeb15103bab7f850": False}
    refs = {"93fdbc:main": "d38898", "93fdbc:main~1": "846e3a",
            "93fdbc:main~2": "5ad376"}

    def resolve_ref(server, ref, expand=False, **k):
        # A concrete uuid resolves to itself, as the real one does; anything else must
        # be a ref we know about.
        return refs[ref] if ref in refs else ref if ref in locked else _unknown(ref)

    def _unknown(ref):
        raise RuntimeError(f'"{ref}" is not a known uuid')

    monkeypatch.setattr(nd, "resolve_ref", resolve_ref)
    monkeypatch.setattr(nd, "fetch_commit", lambda server, uuid, **k: locked[uuid])
    return refs


def test_a_branch_ref_resolves_to_head_and_reports_it_open(fake_dag):
    node = canonical_dvid.resolve_node(canonical_dvid.parse_url(URL))
    assert node == {"ref": "93fdbc:main", "uuid": "d38898", "locked": False, "walked": 0}


def test_prefer_locked_walks_back_to_the_newest_locked_node(fake_dag):
    node = canonical_dvid.resolve_node(canonical_dvid.parse_url(URL), prefer_locked=True)
    assert node["uuid"] == "846e3a" and node["locked"] and node["walked"] == 1


def test_prefer_locked_is_a_no_op_when_the_ref_is_already_locked(fake_dag):
    url = "dvid://dvid.example.org/93fdbc:main~1/labels"
    node = canonical_dvid.resolve_node(canonical_dvid.parse_url(url), prefer_locked=True)
    assert node["uuid"] == "846e3a" and node["walked"] == 0


def test_prefer_locked_from_a_bare_uuid_says_why_it_cannot_walk(fake_dag):
    """DVID's ref~N syntax needs repo:branch; a bare uuid already names one node."""
    url = "dvid://dvid.example.org/d38898/labels"
    with pytest.raises(ValueError, match="repo:branch"):
        canonical_dvid.resolve_node(canonical_dvid.parse_url(url), prefer_locked=True)


def test_the_resolved_uuid_reaches_the_workers_not_the_ref(fake_dag, stub_info):
    """Workers reopen from `data_spec`; a ref there could move mid-run."""
    meta = sm.read_source_metadata({**sm.location_spec(URL, "dvid"),
                                    "prefer_locked": True})
    assert meta["data_spec"]["uuid"] == "846e3a"        # not '93fdbc:main'
    assert "prefer_locked" not in meta["data_spec"]     # applied, not carried


# --------------------------------------------------------------------------- #
# what the output records, and what `copy` refuses
# --------------------------------------------------------------------------- #
@pytest.fixture()
def fake_repo(monkeypatch):
    # neuclease is conda-only and CI installs from PyPI: skip, not error, without it.
    pytest.importorskip("neuclease")
    import neuclease.dvid as nd

    monkeypatch.setattr(nd, "fetch_maxlabel", lambda *a, **k: 139103922, raising=False)
    monkeypatch.setattr(nd, "fetch_repo_info", lambda *a, **k: {
        "DAG": {"Nodes": {"846e3a": {"Branch": "main", "Note": "Locked via fivol API",
                                     "VersionID": 452, "Created": "c", "Updated": "u",
                                     "Log": ["enormous", "do not record"]}}}},
                        raising=False)


def test_provenance_names_the_resolved_node_and_what_was_asked_for(fake_dag, fake_repo):
    """The uuid is what was read, the ref is how it was chosen; both are recorded."""
    from neu_vol.ops import provenance as prov

    spec = {**sm.location_spec(URL, "dvid"), "uuid": "846e3a",
            "requested_ref": "93fdbc:main", "ancestors_walked": 1}
    rec = prov.build_record(src_spec=spec, dst="/out", kind="segmentation")
    src = rec["source"]
    assert src["uuid"] == "846e3a" and src["requested"] == "93fdbc:main"
    assert src["locked"] is True and src["ancestors_walked"] == 1
    assert src["url"].endswith("/846e3a/labels")      # pinned, not the branch ref
    assert src["maxlabel"] == 139103922
    assert src["node"]["Branch"] == "main" and src["node"]["VersionID"] == 452
    assert "Log" not in src["node"]                   # the mutation log can be enormous
    assert rec["run"]["kind"] == "segmentation"


def test_provenance_survives_a_server_that_will_not_answer(fake_dag, monkeypatch):
    """A completed conversion must not be failed by a missing extra field."""
    import neuclease.dvid as nd
    from neu_vol.ops import provenance as prov

    def boom(*a, **k):
        raise RuntimeError("nope")

    monkeypatch.setattr(nd, "fetch_maxlabel", boom, raising=False)
    monkeypatch.setattr(nd, "fetch_repo_info", boom, raising=False)
    src = prov.build_record(src_spec={**sm.location_spec(URL, "dvid"),
                                      "uuid": "846e3a"}, dst="/out")["source"]
    assert src["uuid"] == "846e3a"
    assert "nope" in src["maxlabel_error"] and "nope" in src["node_error"]


def test_provenance_falls_back_to_describing_any_other_spec():
    from neu_vol.ops import provenance as prov

    rec = prov.build_record(src_spec={"backend": "zarr3", "path": "/v"}, dst="/out")
    assert rec["source"] == {"source": "zarr3", "path": "/v"}


def test_copy_refuses_a_dvid_source_and_points_at_convert():
    from neu_vol import cli

    args = cli._parse_args(["copy", "--src", URL, "--dst", "/out"])
    with pytest.raises(SystemExit) as e:
        cli.cmd_copy(args)
    assert "convert" in str(e.value) and "not a storage format" in str(e.value)


# --------------------------------------------------------------------------- #
# what `neu-vol info` reports
# --------------------------------------------------------------------------- #
def test_node_summary_gives_both_candidate_versions(fake_dag):
    """HEAD and the newest locked node: choosing between them decides reproducibility."""
    s = canonical_dvid.node_summary(canonical_dvid.parse_url(URL))
    assert s["head"]["uuid"] == "d38898" and s["head"]["locked"] is False
    assert s["locked"]["uuid"] == "846e3a" and s["locked"]["walked"] == 1


def test_node_summary_collapses_when_the_ref_is_already_locked(fake_dag):
    url = "dvid://dvid.example.org/93fdbc:main~1/labels"
    s = canonical_dvid.node_summary(canonical_dvid.parse_url(url))
    assert s["head"]["uuid"] == s["locked"]["uuid"] == "846e3a"


def test_node_summary_reports_rather_than_raises_when_no_locked_node_is_reachable(
        fake_dag):
    """A bare uuid cannot be walked back from; `info` still prints the rest."""
    url = "dvid://dvid.example.org/d38898/labels"
    s = canonical_dvid.node_summary(canonical_dvid.parse_url(url))
    assert s["head"]["uuid"] == "d38898"
    assert s["locked"] is None and "repo:branch" in s["locked_error"]


def test_info_prints_both_uuids_in_full(fake_dag, stub_info, capsys):
    from neu_vol import cli

    cli.cmd_info(cli._parse_args(["info", URL]))
    out = capsys.readouterr().out
    assert "d38898" in out and "846e3a" in out
    assert "OPEN" in out and "--dvid-locked" in out


def test_info_summarises_a_provenance_file_and_can_print_it_whole(tmp_path, capsys):
    import json

    from neu_vol import cli

    vol = tmp_path / "v"
    vol.mkdir()
    (vol / "provenance.json").write_text(json.dumps(
        {"written": "2026-08-15T00:00:00+00:00",
         "source": {"source": "dvid", "url": "dvid://s/abc/labels", "uuid": "abc",
                    "locked": True, "maxlabel": 7}}))

    cli._print_provenance(str(vol), False)
    brief = capsys.readouterr().out
    assert "dvid://s/abc/labels" in brief and "locked" in brief
    assert "maxlabel" not in brief                 # summary, not the whole document

    cli._print_provenance(str(vol), True)
    assert "maxlabel" in capsys.readouterr().out


def test_info_says_nothing_about_provenance_when_there_is_none(tmp_path, capsys):
    from neu_vol import cli

    cli._print_provenance(str(tmp_path), False)
    assert capsys.readouterr().out == ""


# --------------------------------------------------------------------------- #
# naming a destination after the node it came from
# --------------------------------------------------------------------------- #
@pytest.fixture()
def dvid_src(fake_dag):
    """A resolved DVID spec, as `read_source_metadata` hands one on."""
    return {**sm.location_spec(URL, "dvid"), "uuid": "846e3a",
            "requested_ref": "93fdbc:main"}


def test_uuid_defaults_to_eight_characters(dvid_src):
    from neu_vol.ops.naming import expand

    assert expand("/d/seg_{uuid}", dvid_src) == "/d/seg_846e3a"   # stub uuid is short


def test_uuid_length_and_full_are_selectable(fake_dag):
    from neu_vol.ops.naming import expand

    src = {**sm.location_spec(URL, "dvid"), "uuid": "d38898ac94c8400baeb15103bab7f850"}
    assert expand("{uuid}", src) == "d38898ac"
    assert expand("{uuid:6}", src) == "d38898"
    assert expand("{uuid:full}", src) == "d38898ac94c8400baeb15103bab7f850"


def test_branch_and_instance_are_available(dvid_src):
    from neu_vol.ops.naming import expand

    assert expand("/d/{instance}_{branch}", dvid_src) == "/d/labels_main"


def test_a_destination_without_placeholders_is_untouched():
    from neu_vol.ops.naming import expand

    # Also must not resolve anything — an ordinary destination costs no request.
    assert expand("/plain/path", {"backend": "zarr3"}) == "/plain/path"


def test_expansion_is_idempotent(dvid_src):
    """The CLI expands before deriving targets; `convert` may expand again."""
    from neu_vol.ops.naming import expand

    once = expand("/d/seg_{uuid}", dvid_src)
    assert expand(once, dvid_src) == once


def test_a_placeholder_on_a_source_that_has_none_raises():
    from neu_vol.ops.naming import expand

    with pytest.raises(ValueError, match="supplies none"):
        expand("/d/seg_{uuid}", {"backend": "zarr3", "path": "/v"})


def test_an_unknown_placeholder_lists_the_real_ones(dvid_src):
    from neu_vol.ops.naming import expand

    with pytest.raises(ValueError, match=r"\{uuid\}"):
        expand("/d/{nonesuch}", dvid_src)


@pytest.mark.parametrize("bad", ["{uuid:0}", "{uuid:xyz}", "{uuid:-2}"])
def test_a_bad_length_spec_raises(dvid_src, bad):
    from neu_vol.ops.naming import expand

    with pytest.raises(ValueError, match="positive number"):
        expand(bad, dvid_src)


def test_values_are_sanitised_for_paths_and_object_keys(fake_dag):
    """A ref can carry '~'; an object-store key is happier without it."""
    from neu_vol.ops.naming import expand

    src = {**sm.location_spec(URL, "dvid"), "uuid": "846e3a",
           "requested_ref": "93fdbc:main~1"}
    assert expand("{branch}", src) == "main-1"


def test_convert_kind_defaults_to_none_so_the_source_can_decide():
    """A default of 'image' would override a source that records 'segmentation'."""
    from neu_vol import cli

    assert cli._parse_args(["convert", "--src", "a", "--dst", "b"]).kind is None


# --------------------------------------------------------------------------- #
# the addressing / labelmap split
# --------------------------------------------------------------------------- #
def test_stubbing_the_canonical_module_covers_every_call_site(monkeypatch):
    """`source_metadata` and the backend both call `neu_vol.dvid.<name>`, so one patch
    governs both and no test can quietly reach a live server."""
    calls = []

    def spy_info(spec):
        calls.append("instance_info")
        return INFO

    def spy_node(spec, prefer_locked=False):
        calls.append("resolve_node")
        return {"ref": "r", "uuid": "d38898", "locked": True, "walked": 0}

    monkeypatch.setattr(canonical_dvid, "instance_info", spy_info)
    monkeypatch.setattr(canonical_dvid, "resolve_node", spy_node)

    meta = sm.read_source_metadata(sm.location_spec(URL, "dvid"))
    assert meta["kind"] == "segmentation"
    assert "instance_info" in calls and "resolve_node" in calls

    calls.clear()
    dvid.DVIDBackend(sm.location_spec(URL, "dvid"))
    assert "instance_info" in calls


def test_instance_type_and_syncs_are_read_from_base():
    ann = {"Base": {"TypeName": "annotation", "Syncs": ["labels"]}}
    assert canonical_dvid.instance_type(ann) == "annotation"
    assert canonical_dvid.synced_instances(ann) == ["labels"]
    # A missing or null Syncs is a plain empty list, not None — callers iterate it.
    assert canonical_dvid.synced_instances({"Base": {"TypeName": "keyvalue"}}) == []
    assert canonical_dvid.synced_instances({"Base": {"Syncs": None}}) == []


def test_check_instance_type_names_what_it_wanted():
    spec = canonical_dvid.parse_url(URL)
    assert canonical_dvid.check_instance_type(
        {"Base": {"TypeName": "annotation"}}, spec, "annotation") == "annotation"
    with pytest.raises(ValueError, match="expected annotation"):
        canonical_dvid.check_instance_type({"Base": {"TypeName": "keyvalue"}},
                                          spec, "annotation")
    # More than one acceptable type reads as a list, not a tuple repr.
    with pytest.raises(ValueError, match="expected annotation or keyvalue"):
        canonical_dvid.check_instance_type({"Base": {"TypeName": "roi"}}, spec,
                                          "annotation", "keyvalue")


def test_node_provenance_carries_no_labelmap_fields(fake_repo, fake_dag):
    """`maxlabel` and `supervoxels` belong to the labelmap backend's record only."""
    spec = canonical_dvid.parse_url(URL)
    node = canonical_dvid.resolve_node(spec)
    rec = canonical_dvid.node_provenance(spec, node)
    assert "maxlabel" not in rec and "supervoxels" not in rec
    assert rec["uuid"] == "d38898" and rec["source"] == "dvid"
    # The labelmap backend is what adds them.
    assert "supervoxels" in dvid.provenance(spec, node)
