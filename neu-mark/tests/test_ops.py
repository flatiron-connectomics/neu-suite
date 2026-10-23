"""The stage-1 operations end to end, with DVID stubbed at the ``neuclease.dvid`` boundary
so the code under test is the code that runs in production."""

import json

import pandas as pd
import pytest

from neu_mark import io, ops
from neu_mark import dvid as ann_dvid

from conftest import (BODY_RECORDS, COUNTS, ELEMENTS, KV_URL, LABELSZ_INFO,
                      SYN_INFO, SZ_URL, URL, _open, _src)


# --------------------------------------------------------------------------- #
# points
# --------------------------------------------------------------------------- #
def test_points_writes_both_tables_and_a_provenance_sidecar(tmp_path, dvid_server):
    """The branch ref is resolved to a concrete uuid before any fetch, so the tables are one
    snapshot; the match rate is recorded with it."""
    from neu_vol.location import read_json

    out = str(tmp_path / "syn")
    result = ops.fetch_points(_open(), out, [1, 2])

    assert set(result["written"]) == {"points.parquet", "relationships.parquet"}
    assert len(io.read_table(out, "points")) == 2
    assert len(io.read_table(out, "relationships")) == 3

    rec = read_json(out, "provenance.json")
    assert rec["tool"] == "neu-mark"
    assert rec["source"]["uuid"] == "d38898" and rec["source"]["requested"] == "93fdbc:main"
    assert rec["run"]["match"]["both_ends"] == 1
    assert rec["run"]["synced_to"] == ["labels"]


def test_relationships_are_always_fetched(dvid_server, tmp_path):
    """neuclease's `fetch_elements_for_bodies` hardcodes `relationships=False`."""
    ops.fetch_points(_open(), str(tmp_path / "o"), [1, 2])
    assert dvid_server and all(rels is True for _body, rels in dvid_server)


def test_partner_bodies_resolve_without_touching_the_labelmap(tmp_path, dvid_server):
    out = str(tmp_path / "o")
    ops.fetch_points(_open(), out, [1, 2])
    rels = io.read_table(out, "relationships")
    matched = rels[rels["to_x"] == 11].iloc[0]
    assert matched["from_body"] == 1 and matched["to_body"] == 2
    # no `labels` instance was ever asked about
    assert all(body in (1, 2) for body, _ in dvid_server)


def test_unmatched_partners_are_kept_as_nulls_by_default(tmp_path, dvid_server):
    """Dropping silently would turn "the body list missed this partner" into "this
    synapse does not exist"."""
    out = str(tmp_path / "o")
    result = ops.fetch_points(_open(), out, [1, 2])
    rels = io.read_table(out, "relationships")
    assert rels["to_body"].isna().sum() == 1
    assert result["match"]["pairs"] == 2 and result["match"]["both_ends"] == 1


def test_drop_unmatched_is_opt_in(tmp_path, dvid_server):
    out = str(tmp_path / "o")
    ops.fetch_points(_open(), out, [1, 2], drop_unmatched=True)
    rels = io.read_table(out, "relationships")
    assert len(rels) == 2 and rels["to_body"].notna().all()


def test_connections_table_is_opt_in(tmp_path, dvid_server):
    out = str(tmp_path / "o")
    r = ops.fetch_points(_open(), out, [1, 2], write_connections=True)
    assert "connections.parquet" in r["written"]
    assert len(io.read_table(out, "connections")) == 2


def test_a_permanent_error_on_one_body_is_raised_not_swallowed(tmp_path, dvid_server,
                                                               monkeypatch):
    import neuclease.dvid.annotation as nda

    real = nda.fetch_label

    def flaky(server, uuid, instance, label, **k):
        if int(label) == 2:
            raise ValueError("PERMISSION_DENIED: nope")
        return real(server, uuid, instance, label, **k)

    monkeypatch.setattr(nda, "fetch_label", flaky)
    with pytest.raises(ValueError, match="PERMISSION_DENIED"):
        ops.fetch_points(_open(), str(tmp_path / "o"), [1, 2])


def test_an_unsynced_annotation_instance_is_refused_up_front(tmp_path, monkeypatch,
                                                             dvid_server):
    """Without a sync there is no /label endpoint: one failure per body otherwise."""
    import neu_vol.dvid as vdvid

    monkeypatch.setattr(vdvid, "instance_info",
                        lambda spec: {"Base": {"TypeName": "annotation", "Syncs": []}})
    with pytest.raises(ValueError, match="not synced to a labelmap"):
        ann_dvid.open_points_source(_src())


def test_a_keyvalue_instance_is_refused_for_points(tmp_path, dvid_server):
    with pytest.raises(ValueError, match="expected annotation"):
        ann_dvid.open_points_source(_src(KV_URL))


def test_dvid_locked_selects_the_locked_ancestor(tmp_path, dvid_server):
    out = str(tmp_path / "o")
    ops.fetch_points(_open(locked=True), out, [1])
    from neu_vol.location import read_json

    assert read_json(out, "provenance.json")["source"]["uuid"] == "846e3a"


# --------------------------------------------------------------------------- #
# bodies
# --------------------------------------------------------------------------- #
def test_bodies_writes_its_own_table_to_its_own_destination(tmp_path, dvid_server):
    out = str(tmp_path / "bodyann")
    result = ops.fetch_bodies(_open(KV_URL, "bodies"), out, [1, 2])
    assert result["written"] == ["bodies.parquet"]
    df = io.read_table(out, "bodies")
    assert df.set_index("body").loc[1, "instance"] == "CAm(L)"
    assert result["found"] == 2 and not result["missing"]


def test_bodies_reports_the_ones_with_no_record(tmp_path, dvid_server):
    result = ops.fetch_bodies(_open(KV_URL, "bodies"), str(tmp_path / "o"), [1, 2, 12345])
    assert result["missing"] == [12345] and result["found"] == 2


def test_an_annotation_instance_is_refused_for_bodies(tmp_path, dvid_server):
    with pytest.raises(ValueError, match="expected keyvalue"):
        ann_dvid.open_bodies_source(_src())


# --------------------------------------------------------------------------- #
# the whole instance, without asking for its key list
# --------------------------------------------------------------------------- #
def test_the_whole_instance_is_read_without_ever_calling_keys(tmp_path, dvid_server,
                                                              monkeypatch):
    """`/keys` times out on a large instance, so nothing may depend on it."""
    import neuclease.dvid.keyvalue as ndk

    def boom(*a, **k):
        raise AssertionError("fetch_keys must never be called")

    monkeypatch.setattr(ndk, "fetch_keys", boom)
    out = str(tmp_path / "all")
    result = ops.fetch_bodies(_open(KV_URL, "bodies"), out, everything=True)
    assert result["found"] == len(BODY_RECORDS)
    assert set(io.read_table(out, "bodies")["body"]) == {1, 2}


def test_the_key_ranges_are_recorded_in_the_provenance(tmp_path, dvid_server):
    from neu_vol.location import read_json

    out = str(tmp_path / "all")
    ops.fetch_bodies(_open(KV_URL, "bodies"), out, everything=True)
    run = read_json(out, "provenance.json")["run"]
    assert run["whole_instance"] is True
    assert len(run["key_ranges"]) >= 2 and run["bodies_found"] == len(BODY_RECORDS)


def test_a_body_list_and_all_are_mutually_exclusive(tmp_path, dvid_server):
    """They describe different populations, so silently intersecting them would surprise."""
    with pytest.raises(ValueError, match="not both"):
        ops.fetch_bodies(_open(KV_URL, "bodies"), str(tmp_path / "o"), [1],
                         everything=True)
    with pytest.raises(ValueError, match="needs a body list"):
        ops.fetch_bodies(_open(KV_URL, "bodies"), str(tmp_path / "o"))


def test_a_boundary_key_arriving_twice_is_not_an_error(dvid_server, monkeypatch):
    """DVID key ranges are inclusive at both ends, so consecutive ranges share a key."""
    import neuclease.dvid.keyvalue as ndk

    from neu_mark import dvid as ad

    def overlapping(server, uuid, instance, lo, hi, **k):
        # every range returns the same boundary record, as an inclusive bound would
        return {"3": {"bodyid": 3, "instance": "shared"}}

    monkeypatch.setattr(ndk, "fetch_keyrangevalues", overlapping)
    got = ad.fetch_all_body_annotations(_open(KV_URL, "bodies"))
    assert set(got["records"]) == {"3"}


def test_the_same_key_with_different_values_is_an_error(dvid_server, monkeypatch):
    """A repeat is expected; a repeat that disagrees means the snapshot is not coherent."""
    import neuclease.dvid.keyvalue as ndk

    from neu_mark import dvid as ad

    seen = {"n": 0}

    def inconsistent(server, uuid, instance, lo, hi, **k):
        seen["n"] += 1
        return {"3": {"bodyid": 3, "instance": f"changed-{seen['n']}"}}

    monkeypatch.setattr(ndk, "fetch_keyrangevalues", inconsistent)
    with pytest.raises(RuntimeError, match="not coherent"):
        ad.fetch_all_body_annotations(_open(KV_URL, "bodies"))


def test_a_failing_range_is_split_rather_than_retried_whole(dvid_server, monkeypatch):
    """A range that times out is split, since a retry would just time out more slowly."""
    import neuclease.dvid.keyvalue as ndk

    from neu_mark import dvid as ad

    attempted = []

    def too_big(server, uuid, instance, lo, hi, **k):
        attempted.append((lo, hi))
        if (lo, hi) == ("1", "2"):
            raise RuntimeError("504 Server Error: Gateway Time-out")
        return {"11": {"bodyid": 11}} if lo == "11" else {}

    monkeypatch.setattr(ndk, "fetch_keyrangevalues", too_big)
    got = ad.fetch_all_body_annotations(_open(KV_URL, "bodies"))
    assert set(got["records"]) == {"11"}
    assert ("1", "10") in attempted and ("11", "12") in attempted
    assert attempted.count(("1", "2")) == 1


def test_splitting_gives_up_at_a_bounded_depth(dvid_server, monkeypatch):
    import neuclease.dvid.keyvalue as ndk

    from neu_mark import dvid as ad

    monkeypatch.setattr(ad, "MAX_SPLIT_DEPTH", 1)
    monkeypatch.setattr(ndk, "fetch_keyrangevalues",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("504 nope")))
    with pytest.raises(RuntimeError, match="cannot usefully be split further"):
        ad.fetch_all_body_annotations(_open(KV_URL, "bodies"))


def test_refine_stays_sorted_and_drops_candidates_outside_the_range():
    """`hi` is not always the next character after `lo`, e.g. ' '..' 0'."""
    from neu_mark import dvid as ad

    bounds = ad.refine("1", "2")
    assert bounds == sorted(bounds)
    assert bounds[:3] == ["1", "10", "11"] and bounds[-1] == "2"
    assert len(ad.key_ranges(bounds)) == 11
    assert ad.refine(" ", " 0") == [" ", " 0"]        # nothing strictly between
    inner = ad.refine(" ", "0")
    assert inner == sorted(inner) and inner[1] == " 0"


def test_a_dead_server_fails_quickly_rather_than_recursing(dvid_server, monkeypatch):
    """Splitting answers one range being too big, not the server being down."""
    import neuclease.dvid.keyvalue as ndk

    from neu_mark import dvid as ad

    calls = {"n": 0}

    def dead(*a, **k):
        calls["n"] += 1
        raise RuntimeError("504 Server Error: Gateway Time-out")

    monkeypatch.setattr(ndk, "fetch_keyrangevalues", dead)
    with pytest.raises(RuntimeError):
        ad.fetch_all_body_annotations(_open(KV_URL, "bodies"))
    assert calls["n"] < 10, calls["n"]                 # a handful, not the full recursion


def test_a_split_that_cannot_subdivide_is_what_stops_the_recursion(dvid_server,
                                                                  monkeypatch):
    """The leading sub-range of any split (`lo`..`lo + '0'`) cannot subdivide, so this guard
    fires before any failure budget would."""
    import neuclease.dvid.keyvalue as ndk

    from neu_mark import dvid as ad

    calls = {"n": 0}

    def dead(server, uuid, instance, lo, hi, **k):
        calls["n"] += 1
        raise RuntimeError("504 Server Error: Gateway Time-out")

    monkeypatch.setattr(ndk, "fetch_keyrangevalues", dead)
    # Passed, not patched: `boundaries` is a default argument, bound at definition time.
    with pytest.raises(RuntimeError, match="cannot usefully be split further"):
        ad.fetch_all_body_annotations(_open(KV_URL, "bodies"), boundaries=("1", "2"))
    assert calls["n"] <= 3, calls["n"]


def test_unsorted_boundaries_are_refused():
    from neu_mark import dvid as ad

    with pytest.raises(ValueError, match="must be sorted"):
        ad.key_ranges(["5", "1"])


def test_cli_all_writes_the_whole_instance(tmp_path, dvid_server, capsys):
    from neu_mark.cli import main

    out = str(tmp_path / "all")
    assert main(["bodies", "--src", KV_URL, "--out", out, "--all"]) == 0
    printed = capsys.readouterr().out
    assert "records across" in printed
    assert (tmp_path / "all" / "bodies.parquet").exists()


def test_cli_refuses_both_or_neither(tmp_path, dvid_server):
    from neu_mark.cli import main

    with pytest.raises(SystemExit, match="pass one, not both"):
        main(["bodies", "--src", KV_URL, "--out", str(tmp_path / "o"),
              "--all", "--bodies", "1"])
    with pytest.raises(SystemExit, match="--bodies is required"):
        main(["bodies", "--src", KV_URL, "--out", str(tmp_path / "o")])


# --------------------------------------------------------------------------- #
# select-bodies
# --------------------------------------------------------------------------- #
def _select(tmp_path, dvid_server, url=URL, **kw):
    source = ann_dvid.open_counts_source(_src(url))
    out = str(tmp_path / "sel" / "bodies.csv")
    return out, ops.write_selected_bodies(source, out, **kw)


def test_select_thresholds_on_the_total_and_ranks_by_it(tmp_path, dvid_server):
    _out, r = _select(tmp_path, dvid_server, min_synapses=10)
    df = r["bodies"]
    # body 4 (5 synapses) is out; the rest ranked by total descending
    assert df["body"].tolist() == [1, 2, 3]
    assert df["syn"].tolist() == [105, 63, 40]
    assert df["pre"].tolist() == [5, 60, 0]


def test_a_total_threshold_keeps_sensory_and_outward_projecting_neurons(tmp_path,
                                                                       dvid_server):
    """Body 2 is sensory (almost no postsynapses) and body 3 projects out of the volume (no
    presynapses); a per-type default would drop both."""
    _out, r = _select(tmp_path, dvid_server, min_synapses=10)
    assert {2, 3} <= set(r["bodies"]["body"])

    # ...and the per-type filters, when asked for, do exclude them
    _out, only_pre = _select(tmp_path, dvid_server, min_synapses=10, min_pre=1)
    assert 3 not in set(only_pre["bodies"]["body"])          # no presynapses
    _out, only_post = _select(tmp_path, dvid_server, min_synapses=10, min_post=10)
    assert 2 not in set(only_post["bodies"]["body"])         # sensory


def test_limit_takes_the_top_n_of_the_ranking(tmp_path, dvid_server):
    _out, r = _select(tmp_path, dvid_server, min_synapses=10, limit=2)
    assert r["bodies"]["body"].tolist() == [1, 2]


def test_a_per_type_minimum_raises_the_query_threshold(tmp_path, dvid_server,
                                                       monkeypatch):
    """AllSyn >= each type, so max(minimums) is a sound narrowing threshold."""
    import neuclease.dvid.labelsz as ndsz

    seen = []
    real = ndsz.fetch_threshold
    monkeypatch.setattr(ndsz, "fetch_threshold",
                        lambda *a, **k: (seen.append(a[3]), real(*a, **k))[1])
    _select(tmp_path, dvid_server, min_synapses=10, min_post=50)
    assert seen[0] == 50


@pytest.mark.parametrize("url", [URL, SZ_URL], ids=["via-annotation", "direct"])
def test_the_labelsz_instance_is_found_or_named_directly(tmp_path, dvid_server, url):
    """`synapses` is the name people know; the index records what it syncs to."""
    source = ann_dvid.open_counts_source(_src(url))
    assert source["instance"] == "synapses_labelsz"
    assert source["indexes"] == ["synapses"]


def test_no_labelsz_for_the_instance_says_so(tmp_path, dvid_server, monkeypatch):
    import neuclease.dvid as nd

    monkeypatch.setattr(nd, "fetch_repo_info",
                        lambda server, uuid, **k: {"DataInstances": {"synapses": SYN_INFO}})
    with pytest.raises(ValueError, match="no 'labelsz' instance"):
        ann_dvid.open_counts_source(_src(URL))


def test_an_ambiguous_labelsz_asks_which_one(tmp_path, dvid_server, monkeypatch):
    import neuclease.dvid as nd

    monkeypatch.setattr(nd, "fetch_repo_info", lambda server, uuid, **k: {
        "DataInstances": {"synapses": SYN_INFO, "a_labelsz": LABELSZ_INFO,
                          "b_labelsz": LABELSZ_INFO}})
    with pytest.raises(ValueError, match="several labelsz instances"):
        ann_dvid.open_counts_source(_src(URL))


def test_an_empty_result_names_the_silent_unindexed_failure(tmp_path, dvid_server):
    """DVID answers an unindexed element type with an empty result, not an error."""
    with pytest.raises(ValueError, match="EMPTY result"):
        _select(tmp_path, dvid_server, min_synapses=10_000)


def test_paging_collects_every_body_beyond_one_page(tmp_path, dvid_server, monkeypatch):
    monkeypatch.setattr("neu_mark.dvid._LABELSZ_PAGE", 2)
    _out, r = _select(tmp_path, dvid_server, min_synapses=1)
    assert r["bodies"]["body"].tolist() == [1, 2, 3, 4]


def test_the_deep_paging_cap_warns_rather_than_silently_truncating(tmp_path, dvid_server,
                                                                  monkeypatch, caplog):
    """Per-page cost grows with offset, so there is a cap — but a capped result must say so."""
    monkeypatch.setattr("neu_mark.dvid._LABELSZ_PAGE", 1)
    monkeypatch.setattr("neu_mark.dvid._MAX_SELECT", 2)
    with caplog.at_level("WARNING"):
        _out, r = _select(tmp_path, dvid_server, min_synapses=1)
    assert len(r["bodies"]) == 2
    assert "cap" in caplog.text


def test_the_list_records_which_node_it_was_computed_from(tmp_path, dvid_server):
    """Proofreading changes body ids, so a list nobody can date is one that goes stale."""
    from neu_vol.location import read_json

    out, r = _select(tmp_path, dvid_server, min_synapses=10)
    rec = read_json(str(tmp_path / "sel"), "bodies.provenance.json")
    assert rec["source"]["uuid"] == "d38898"
    assert rec["run"]["bodies_selected"] == 3
    assert rec["run"]["indexes"] == ["synapses"]


def test_the_written_list_is_a_valid_bodies_input(tmp_path, dvid_server):
    """Closes the loop: what select-bodies writes is what --bodies reads."""
    from neu_mark import bodies as body_reader

    out, r = _select(tmp_path, dvid_server, min_synapses=10)
    assert r["path"] == out
    assert body_reader.load(out) == [1, 2, 3]


def test_csv_is_the_default_and_is_lossless_here(tmp_path, dvid_server, caplog):
    """Every column is a non-nullable integer, which csv does preserve — so no warning."""
    with caplog.at_level("WARNING"):
        out, r = _select(tmp_path, dvid_server, min_synapses=10)
    assert (tmp_path / "sel" / "bodies.csv").exists()
    assert "read them back as str or float64" not in caplog.text


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_cli_select_bodies_prints_the_command_to_run_next(tmp_path, dvid_server, capsys):
    from neu_mark.cli import main

    out = str(tmp_path / "sel" / "bodies.csv")
    assert main(["select-bodies", "--src", URL, "--out", out,
                 "--min-synapses", "10"]) == 0
    printed = capsys.readouterr().out
    assert "selected 3 bodies" in printed
    assert f"--bodies {out}" in printed



def test_cli_points_runs_and_expands_the_uuid_placeholder(tmp_path, dvid_server, capsys):
    from neu_mark.cli import main

    dst = str(tmp_path / "syn_{uuid:6}")
    assert main(["points", "--src", URL, "--out", dst, "--bodies", "1,2"]) == 0
    out = capsys.readouterr().out
    assert "syn_d38898" in out
    assert (tmp_path / "syn_d38898" / "points.parquet").exists()


def test_the_destination_is_named_after_the_node_actually_read(tmp_path, dvid_server,
                                                              capsys):
    """With --dvid-locked, {uuid} must expand to the locked node the tables were read from,
    not HEAD."""
    from neu_mark.cli import main
    from neu_vol.location import read_json

    dst = str(tmp_path / "syn_{uuid:6}")
    main(["points", "--src", URL, "--out", dst, "--bodies", "1", "--dvid-locked"])

    written = tmp_path / "syn_846e3a"
    assert written.exists(), sorted(p.name for p in tmp_path.iterdir())
    assert not (tmp_path / "syn_d38898").exists()
    assert read_json(str(written), "provenance.json")["source"]["uuid"] == "846e3a"


def test_cli_warns_loudly_when_coverage_is_low(tmp_path, dvid_server, capsys):
    from neu_mark.cli import main

    main(["points", "--src", URL, "--out", str(tmp_path / "o"), "--bodies", "1"])
    out = capsys.readouterr().out
    # body 1 alone resolves neither of its two partners
    assert "0.0%" in out and "coverage" in out


def test_cli_refuses_a_non_dvid_source(tmp_path):
    from neu_mark.cli import main

    with pytest.raises(SystemExit, match="not a DVID URL"):
        main(["points", "--src", "/local/path", "--out", str(tmp_path), "--bodies", "1"])


def test_cli_info_shows_the_type_the_sync_and_both_nodes(dvid_server, capsys):
    from neu_mark.cli import main

    assert main(["info", "--src", URL]) == 0
    out = capsys.readouterr().out
    assert "annotation" in out and "labels" in out
    assert "d38898" in out and "846e3a" in out
    assert "OPEN" in out


def test_cli_bodies_writes_elsewhere(tmp_path, dvid_server):
    from neu_mark.cli import main

    out = str(tmp_path / "ann")
    assert main(["bodies", "--src", KV_URL, "--out", out, "--bodies", "1,2"]) == 0
    assert (tmp_path / "ann" / "bodies.parquet").exists()
    assert not (tmp_path / "ann" / "points.parquet").exists()


# --------------------------------------------------------------------------- #
# segment-properties --meshed-bodies
# --------------------------------------------------------------------------- #

def _meshed_volume(tmp_path, names, mesh_info=None):
    vol = tmp_path / "seg"
    (vol / "mesh").mkdir(parents=True)
    (vol / "info").write_text(json.dumps({"@type": "neuroglancer_multiscale_volume",
                                          "type": "segmentation", "mesh": "mesh"}))
    (vol / "mesh" / "info").write_text(json.dumps(
        mesh_info or {"@type": "neuroglancer_multilod_draco"}))
    for name in names:
        (vol / "mesh" / name).write_bytes(b"x")
    return str(vol)


def test_meshed_bodies_writes_properties_for_the_meshed_bodies(tmp_path, dvid_server,
                                                               capsys):
    from neu_mark.cli import main

    vol = _meshed_volume(tmp_path, ["1", "1.index", "2", "2.index", "3"])  # 3: no manifest
    assert main(["segment-properties", "--src", KV_URL, "--dst", vol,
                 "--meshed-bodies"]) == 0
    assert "2 bodies (1..2), meshed in" in capsys.readouterr().out
    info = json.loads((tmp_path / "seg" / "segment_properties" / "info").read_text())
    assert sorted(int(i) for i in info["inline"]["ids"]) == [1, 2]


@pytest.mark.parametrize("names, mesh_info, match", [
    ([], None, "no meshes in"),
    (["0.shard"], {"@type": "neuroglancer_multilod_draco",
                   "sharding": {"@type": "neuroglancer_uint64_sharded_v1"}}, "Use --bodies"),
])
def test_meshed_bodies_refuses_what_it_cannot_list(tmp_path, names, mesh_info, match):
    from neu_mark.cli import _meshed_bodies

    with pytest.raises(SystemExit, match=match):
        _meshed_bodies(_meshed_volume(tmp_path, names, mesh_info))


@pytest.mark.parametrize("which", [["--bodies", "1", "--meshed-bodies"], []])
def test_segment_properties_takes_exactly_one_body_selection(which):
    from neu_mark.cli import main

    with pytest.raises(SystemExit):
        main(["segment-properties", "--src", KV_URL, "--dst", "x", *which])


@pytest.mark.parametrize("path, expected", [
    ("out/bodies.csv", ("out", "bodies", "csv")),
    ("s3://my-bucket/sel/b.v2.parquet", ("s3://my-bucket/sel", "b.v2", "parquet")),
    ("bodies.feather", (".", "bodies", "feather")),
])
def test_a_table_path_names_its_directory_name_and_format(path, expected):
    assert io.split_table_path(path) == expected


@pytest.mark.parametrize("path", ["out/bodies", "out/bodies.txt", "out/.csv"])
def test_a_table_path_without_a_table_extension_is_refused(path):
    with pytest.raises(ValueError, match="no table extension"):
        io.split_table_path(path)


# --------------------------------------------------------------------------- #
# points --all: the whole instance by region, bodies from the labelmap
# --------------------------------------------------------------------------- #
#: Every element of the instance: bodies 1 and 2's, a pair between bodies 7 and 8 that no
#: body list reaches, and the partner [99, 99, 99] of body 1's T-bar, on background.
ALL_ELEMENTS = [
    *ELEMENTS[1], *ELEMENTS[2],
    {"Pos": [99, 99, 99], "Kind": "PostSyn", "Prop": {"conf": "0.4"},
     "Rels": [{"Rel": "PostSynTo", "To": [10, 20, 30]}]},
    {"Pos": [50, 50, 50], "Kind": "PreSyn", "Prop": {"conf": "0.8"},
     "Rels": [{"Rel": "PreSynTo", "To": [51, 51, 51]}]},
    {"Pos": [51, 51, 51], "Kind": "PostSyn", "Prop": {},
     "Rels": [{"Rel": "PostSynTo", "To": [50, 50, 50]}]},
]
#: The labelmap's body at each x.
BODY_AT_X = {10: 1, 11: 2, 99: 0, 50: 7, 51: 8}


@pytest.fixture()
def whole_instance(dvid_server, monkeypatch):
    import neuclease.dvid.annotation as nda
    import neuclease.dvid.labelmap as ndl

    calls = {}

    def fetch_synapses_in_batches(server, uuid, instance, *, format, processes, **k):
        calls["batches"] = (uuid, instance, format, processes)
        return [json.loads(json.dumps(e)) for e in ALL_ELEMENTS]

    def fetch_bodies_for_many_points(server, uuid, seg_instance, point_df, **k):
        calls["bodies"] = (uuid, seg_instance)
        point_df["sv"] = point_df["x"] * 10
        point_df["body"] = point_df["x"].map(BODY_AT_X).astype("uint64")

    monkeypatch.setattr(nda, "fetch_synapses_in_batches", fetch_synapses_in_batches)
    monkeypatch.setattr(ndl, "fetch_bodies_for_many_points", fetch_bodies_for_many_points)
    return calls


def test_all_fetches_by_region_and_every_partner_has_its_body(tmp_path, whole_instance,
                                                              dvid_server):
    out = str(tmp_path / "o")
    result = ops.fetch_points(_open(), out, None, write_connections=True)
    assert not dvid_server, "--all must not fetch per body"
    assert whole_instance["batches"][1:3] == ("synapses", "json")
    assert whole_instance["bodies"][1] == "labels"          # the synced labelmap
    assert whole_instance["batches"][0] == whole_instance["bodies"][0]   # one node

    conns = result["connections"].set_index(["pre_x", "post_x"])
    assert int(conns.loc[(10, 11), "post_body"]) == 2
    assert int(conns.loc[(50, 51), "pre_body"]) == 7        # no body list reaches it
    assert pd.isna(conns.loc[(10, 99), "post_body"])        # background is no body
    assert result["match"]["pairs"] == 3
    assert set(result["points"]["body"]) == {0, 1, 2, 7, 8}
    assert "sv" in result["points"].columns                 # kept for a cheaper re-run
    record = json.loads((tmp_path / "o" / "provenance.json").read_text())
    assert record["run"]["scope"] == "all" and "bodies_requested" not in record["run"]


def test_complete_tables_take_each_relationships_own_body_from_its_element():
    from neu_mark import tables

    points, rels = tables.elements_to_frames(ALL_ELEMENTS, 0)
    points["body"] = points["x"].map(BODY_AT_X)
    done = tables.complete_tables(points, rels)
    by_pos = done["relationships"].set_index("x")
    assert int(by_pos.loc[50, "from_body"]) == 7
    assert int(by_pos.loc[51, "to_body"]) == 7


def test_points_takes_bodies_or_all_not_both(tmp_path):
    from neu_mark import cli

    with pytest.raises(SystemExit, match="not both"):
        cli.cmd_points(cli.build_parser().parse_args(
            ["points", "--src", URL, "--out", str(tmp_path), "--bodies", "1", "--all"]))
    with pytest.raises(SystemExit, match="--all"):
        cli.cmd_points(cli.build_parser().parse_args(
            ["points", "--src", URL, "--out", str(tmp_path)]))
