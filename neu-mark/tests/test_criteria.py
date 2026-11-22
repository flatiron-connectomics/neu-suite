"""select-bodies --criteria: the file, the selection, and the run end to end."""

import pandas as pd
import pytest

from neu_mark import criteria as crit
from neu_mark import dvid as ann_dvid
from neu_mark import ops

from conftest import BODY_RECORDS, KV_URL, URL, _src

CRITERIA = '''
[common]
noise_tokens = ["irrelevant"]

[meshes]
neurons = "all"
fragments = "conditional"
glia = "conditional"
noise = "none"
unnamed = "conditional"
necessary_conditions = []
sufficient_conditions = ["big", "synaptic"]

[meshes.conditions]
big = { size = 1000 }
synaptic = { size = 100, sites = 30 }

[skeletons]
neurons = "all"
fragments = "conditional"
glia = "none"
noise = "none"
unnamed = "none"
necessary_conditions = ["not_tiny"]
sufficient_conditions = ["connected"]

[skeletons.conditions]
not_tiny = { size = 100 }
connected = { pairs = 2 }
'''

RULES = '''
from neu_mark import rule

LABEL = "instance"


@rule(tag=False)
def category(r):
    """what kind of body"""
    name = r["instance"]
    if not name:
        return "unnamed"
    if "irrelevant" in name:
        return "noise"
    if "glia" in name:
        return "glia"
    return "fragments" if "fragment" in name else "neurons"


@rule(tag=False)
def corrected_name(r):
    """a misspelling fixed"""
    name = r["instance"] or ""
    return name.replace("fragmnet", "fragment") if "fragmnet" in name else None
'''


def _file(tmp_path, text=CRITERIA, name="criteria.toml"):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def _table(**columns):
    return pd.DataFrame(columns)


# --------------------------------------------------------------------------- #
# the file
# --------------------------------------------------------------------------- #
def test_a_list_is_loaded_with_its_conditions_in_order(tmp_path):
    c = crit.load(_file(tmp_path), "meshes")
    assert c.modes["fragments"] == "conditional"
    assert [x.name for x in c.sufficient] == ["big", "synaptic"]
    assert c.sufficient[1].minimums == {"size": 100, "sites": 30}
    assert c.fields == {"size", "sites"}
    assert crit.lists(_file(tmp_path)) == ["meshes", "skeletons"]


@pytest.mark.parametrize("change, match", [
    (('glia = "conditional"', 'glia = "some"'), "a category is"),
    (("big = { size = 1000 }", "big = { volume = 1000 }"), "the fields are"),
    (("big = { size = 1000 }", "big = { size = true }"), "whole number"),
    (("big = { size = 1000 }", "big = { size = -1 }"), "whole number"),
    (("big = { size = 1000 }", "big = {}"), "non-empty"),
    (('["big", "synaptic"]', '["big", "synaptc"]'), "synaptc"),
    (('sufficient_conditions = ["big", "synaptic"]', 'sufficient_conditions = ["big"]'),
     "defined but in neither list"),
    (('neurons = "all"\nfragments', 'neurons = 3\nfragments'), "neither a category"),
])
def test_mistakes_in_the_file_are_refused(tmp_path, change, match):
    with pytest.raises(ValueError, match=match):
        crit.load(_file(tmp_path, CRITERIA.replace(*change, 1)), "meshes")


def test_conditional_with_nothing_sufficient_is_refused(tmp_path):
    text = '[a]\nx = "conditional"\nsufficient_conditions = []\n'
    with pytest.raises(ValueError, match="could never be included"):
        crit.load(_file(tmp_path, text), "a")


def test_an_unknown_list_names_the_ones_there_are(tmp_path):
    with pytest.raises(ValueError, match="meshes, skeletons"):
        crit.load(_file(tmp_path), "segment-properties")
    with pytest.raises(ValueError, match="no list 'common'"):
        crit.load(_file(tmp_path), "common")


def test_the_record_keeps_the_list_and_common(tmp_path):
    rec = crit.load(_file(tmp_path), "skeletons").record()
    assert rec["list"] == "skeletons" and rec["common"] == {"noise_tokens": ["irrelevant"]}
    assert rec["section"]["conditions"]["connected"] == {"pairs": 2}
    assert len(rec["sha256"]) == 64


# --------------------------------------------------------------------------- #
# selecting
# --------------------------------------------------------------------------- #
def test_select_names_the_first_sufficient_condition_and_all_is_unconditional(tmp_path):
    c = crit.load(_file(tmp_path), "meshes")
    t = _table(category=["neurons", "fragments", "fragments", "fragments", "noise", "glia"],
               voxels=[1, 5000, 500, 50, 10**9, 500],
               sites=[0, 99, 30, 99, 99, 29])
    assert c.select(t).tolist() == ["all", "big", "synaptic", None, None, None]


def test_a_necessary_condition_gates_all_and_conditional_alike(tmp_path):
    c = crit.load(_file(tmp_path), "skeletons")
    t = _table(category=["neurons", "neurons", "fragments", "fragments"],
               voxels=[99, 100, 100, 99], pairs=[0, 0, 2, 5])
    assert c.select(t).tolist() == [None, "all", "connected", None]


def test_a_category_the_list_does_not_mention_is_refused(tmp_path):
    c = crit.load(_file(tmp_path), "meshes")
    with pytest.raises(ValueError, match="does not list: muscle"):
        c.select(_table(category=["muscle"], voxels=[1], sites=[1]))


def test_bodies_with_no_record_are_looked_for_by_each_sufficient_condition(tmp_path):
    c = crit.load(_file(tmp_path), "meshes")
    assert crit.candidate_sources(c, "unnamed") == {"sites": 30, "pairs": None,
                                                    "size": 1000}
    s = crit.load(_file(tmp_path), "skeletons")
    assert crit.candidate_sources(s, "unnamed") == {"sites": None, "pairs": None,
                                                    "size": None}


def test_all_for_bodies_with_no_record_is_refused(tmp_path):
    text = CRITERIA.replace('unnamed = "conditional"', 'unnamed = "all"', 1)
    with pytest.raises(ValueError, match="every body in the labelmap"):
        crit.candidate_sources(crit.load(_file(tmp_path, text), "meshes"), "unnamed")


def test_a_condition_of_only_zeros_cannot_admit_bodies_with_no_record(tmp_path):
    text = CRITERIA.replace("big = { size = 1000 }", "big = { size = 0 }", 1)
    with pytest.raises(ValueError, match="holds for every body"):
        crit.candidate_sources(crit.load(_file(tmp_path, text), "meshes"), "unnamed")


# --------------------------------------------------------------------------- #
# end to end, DVID stubbed
# --------------------------------------------------------------------------- #
#: Full-resolution sizes. Bodies 3, 7 and 8 have no record; 3 has 40 synapses, 7 and 8
#: none.
SIZES = {1: 5000, 2: 50, 3: 500, 4: 2000, 5: 10, 6: 9999, 7: 3000, 8: 20}


@pytest.fixture()
def world(tmp_path, dvid_server, monkeypatch):
    for key, name in {"4": "SNP_fragment", "5": "glia", "6": "irrelevant"}.items():
        monkeypatch.setitem(BODY_RECORDS, key, {"bodyid": int(key), "instance": name})
    asked = {"listed": [], "sized": []}

    def large(source, min_voxels, **k):
        asked["listed"].append(min_voxels)
        return {b: v for b, v in SIZES.items() if v >= min_voxels}

    def sizes(source, bodies, **k):
        asked["sized"].append(sorted(int(b) for b in bodies))
        return {int(b): SIZES.get(int(b), 0) for b in bodies}

    monkeypatch.setattr(ann_dvid, "fetch_large_bodies", large)
    monkeypatch.setattr(ann_dvid, "fetch_voxel_counts", sizes)
    rules = tmp_path / "rules.py"
    rules.write_text(RULES)
    annotations = ann_dvid.open_bodies_source(_src(KV_URL))
    return {"asked": asked, "rules": str(rules), "annotations": annotations,
            "labelmap": {**annotations, "instance": "labels"},
            "source": ann_dvid.open_counts_source(_src(URL))}


def _run(tmp_path, world, name, **kw):
    out = str(tmp_path / "sel" / f"{name}.csv")
    args = {"rules_path": world["rules"], "annotations_source": world["annotations"],
            "labelmap_source": world["labelmap"], **kw}
    return out, ops.write_criteria_bodies(world["source"], out,
                                          criteria=crit.load(_file(tmp_path), name), **args)


def test_meshes_admit_by_category_and_by_condition_including_bodies_with_no_record(
        tmp_path, world):
    out, r = _run(tmp_path, world, "meshes")
    df = r["bodies"]
    assert df["body"].tolist() == [1, 3, 4, 7]                       # ranked by synapses
    assert df["admitted_by"].tolist() == ["all", "synaptic", "big", "big"]
    assert df["category"].tolist() == ["neurons", "unnamed", "fragments", "unnamed"]
    assert df["voxels"].tolist() == [5000, 500, 2000, 3000]
    assert df.set_index("body").loc[3, ["pre", "post", "syn"]].tolist() == [0, 40, 40]
    # listed down to the smallest size any condition asks for, not just the size-only one
    assert world["asked"]["listed"] == [100]
    assert r["found"] == {"sites": 1, "size": 2}                      # 3; 3 and 7
    run = r["record"]["run"]
    assert run["criteria"]["list"] == "meshes"
    assert run["by_category"]["unnamed"]["by"] == {"synaptic": 1, "big": 1}
    assert run["by_category"]["glia"]["included"] == 0
    assert "noise" not in run["by_category"]                          # "none": not fetched
    back = pd.read_csv(out)
    assert back["admitted_by"].tolist() == ["all", "synaptic", "big", "big"]


def test_skeletons_read_pairs_and_size_per_body_when_nothing_is_listed(tmp_path, world):
    from test_ops import _connections

    conns = _connections(tmp_path, [(4, 1), (4, 2), (2, 1)])
    _out, r = _run(tmp_path, world, "skeletons", connections=conns)
    df = r["bodies"].set_index("body")
    assert sorted(df.index) == [1, 4]       # 2 is unnamed: "none"; 4 has 2 pairs, 2000 vox
    assert df.loc[4, "admitted_by"] == "connected" and df.loc[4, "pairs"] == 2
    assert world["asked"]["listed"] == []
    assert world["asked"]["sized"] == [[1, 4]]       # every candidate, once; never "none"


def test_a_pairs_condition_needs_the_connections(tmp_path, world):
    with pytest.raises(ValueError, match="connections"):
        _run(tmp_path, world, "skeletons")


def test_a_size_condition_needs_a_labelmap(tmp_path, world):
    with pytest.raises(ValueError, match="labelmap"):
        _run(tmp_path, world, "meshes", labelmap_source=None)


def test_a_labelmap_from_another_node_is_refused(tmp_path, world):
    with pytest.raises(ValueError, match="same node"):
        _run(tmp_path, world, "meshes",
             labelmap_source={**world["labelmap"], "uuid": "846e3a"})


def test_a_record_the_rule_gives_no_category_is_an_error(tmp_path, world):
    rules = tmp_path / "rules.py"
    rules.write_text(RULES.replace('return "glia"', "return None"))
    with pytest.raises(ValueError, match="no category for 1 records"):
        _run(tmp_path, world, "meshes", rules_path=str(rules))


# --------------------------------------------------------------------------- #
# the command line
# --------------------------------------------------------------------------- #
def test_cli_refuses_thresholds_beside_criteria(tmp_path, world):
    from neu_mark.cli import main

    with pytest.raises(SystemExit, match="drop --min-synapses, --named"):
        main(["select-bodies", "--src", URL, "--out", str(tmp_path / "x.csv"),
              "--criteria", _file(tmp_path), "--list", "meshes", "--min-synapses", "5",
              "--named", "category", "--rules", world["rules"], "--annotations", KV_URL])


def test_cli_needs_both_criteria_and_list(tmp_path):
    from neu_mark.cli import main

    with pytest.raises(SystemExit, match="go together"):
        main(["select-bodies", "--src", URL, "--out", str(tmp_path / "x.csv"),
              "--criteria", _file(tmp_path)])


def test_cli_writes_the_list_and_prints_each_category(tmp_path, world, capsys):
    from neu_mark.cli import main

    rc = main(["select-bodies", "--src", URL, "--out", str(tmp_path / "m.csv"),
               "--criteria", _file(tmp_path), "--list", "meshes", "--rules", world["rules"],
               "--annotations", KV_URL, "--labelmap",
               "dvid://dvid.example.org/93fdbc:main/labels"])
    assert rc == 0
    text = capsys.readouterr().out
    assert "selected 4 bodies for [meshes]" in text
    assert "left out: noise" in text
    assert "unnamed" in text and "synaptic 1" in text


def test_report_rules_are_written_beside_the_list_as_a_table(tmp_path, world, monkeypatch):
    monkeypatch.setitem(BODY_RECORDS, "4", {"bodyid": 4, "instance": "SNP_fragmnet"})
    text = 'report_rules = ["corrected_name"]\n' + CRITERIA
    out = str(tmp_path / "sel" / "meshes.csv")
    r = ops.write_criteria_bodies(
        world["source"], out, criteria=crit.load(_file(tmp_path, text), "meshes"),
        rules_path=world["rules"], annotations_source=world["annotations"],
        labelmap_source=world["labelmap"])
    table = pd.read_csv(tmp_path / "sel" / "meshes.corrected_name.csv")
    assert table.to_dict("records") == [{"body": 4, "instance": "SNP_fragmnet",
                                         "corrected_name": "SNP_fragment"}]
    assert r["record"]["run"]["reported"] == {
        "corrected_name": {"records": 1, "table": "meshes.corrected_name.csv"}}


def test_an_unknown_report_rule_fails_before_any_fetch(tmp_path, world):
    text = 'report_rules = ["nope"]\n' + CRITERIA
    with pytest.raises(KeyError, match="nope"):
        ops.write_criteria_bodies(
            world["source"], str(tmp_path / "m.csv"),
            criteria=crit.load(_file(tmp_path, text), "meshes"), rules_path=world["rules"],
            annotations_source=world["annotations"], labelmap_source=world["labelmap"])


def test_report_rules_must_be_a_list_of_names(tmp_path):
    with pytest.raises(ValueError, match="report_rules"):
        crit.load(_file(tmp_path, 'report_rules = "corrected_name"\n' + CRITERIA), "meshes")


# --------------------------------------------------------------------------- #
# the listing of large labels, and keeping it per locked node
# --------------------------------------------------------------------------- #
NODE = {"backend": "dvid", "server": "http://dvid.example.org", "uuid": "93fdbc",
        "instance": "labels"}


@pytest.fixture()
def lister(monkeypatch):
    calls = []

    def large(source, min_voxels, **k):
        calls.append(min_voxels)
        return {b: v for b, v in SIZES.items() if v >= min_voxels}

    monkeypatch.setattr(ann_dvid, "fetch_large_bodies", large)
    monkeypatch.setattr(ann_dvid, "_node_locked", lambda source: True)
    return calls


def test_a_locked_nodes_listing_is_kept_and_reused_at_that_size_or_above(tmp_path, lister):
    cache = str(tmp_path / "sizes")
    got, where = ann_dvid.large_bodies(NODE, 100, cache=cache)
    assert got == {1: 5000, 3: 500, 4: 2000, 6: 9999, 7: 3000}
    assert where.endswith("dvid.example.org_93fdbc_labels.sizes.parquet")
    again, where2 = ann_dvid.large_bodies(NODE, 2000, cache=cache)
    assert again == {1: 5000, 4: 2000, 6: 9999, 7: 3000} and where2 == where
    assert lister == [100]                                     # read back, not re-listed


def test_a_smaller_size_than_kept_lists_again_and_keeps_that(tmp_path, lister):
    import json

    cache = tmp_path / "sizes"
    ann_dvid.large_bodies(NODE, 1000, cache=str(cache))
    got, _ = ann_dvid.large_bodies(NODE, 30, cache=str(cache))
    assert 2 in got and lister == [1000, 30]
    meta = json.loads(next(cache.glob("*.sizes.json")).read_text())
    assert meta["min_voxels"] == 30 and meta["uuid"] == "93fdbc"
    ann_dvid.large_bodies(NODE, 40, cache=str(cache))
    assert lister == [1000, 30]


def test_another_node_does_not_read_a_kept_listing(tmp_path, lister):
    cache = str(tmp_path / "sizes")
    ann_dvid.large_bodies(NODE, 100, cache=cache)
    ann_dvid.large_bodies({**NODE, "uuid": "846e3a"}, 100, cache=cache)
    assert lister == [100, 100]


def test_an_open_nodes_listing_is_not_kept(tmp_path, lister, monkeypatch):
    monkeypatch.setattr(ann_dvid, "_node_locked", lambda source: False)
    cache = tmp_path / "sizes"
    _got, where = ann_dvid.large_bodies(NODE, 100, cache=str(cache))
    assert where == "DVID"
    assert not cache.exists() or not list(cache.iterdir())
    ann_dvid.large_bodies(NODE, 100, cache=str(cache))
    assert lister == [100, 100]


def test_the_listing_pages_in_one_stream_keeping_the_large(monkeypatch):
    dvid_mod = pytest.importorskip("neuclease.dvid")
    labels = pd.Series(SIZES).sort_index()
    starts = []

    def listlabels(server, uuid, instance, start, batch, sizes=False):
        starts.append(start)
        return labels[labels.index >= start].iloc[:batch]

    monkeypatch.setattr(dvid_mod, "fetch_listlabels", listlabels)
    monkeypatch.setattr(dvid_mod, "fetch_maxlabel", lambda *a: 8)
    got = ann_dvid.fetch_large_bodies(NODE, 1000, batch_size=3)
    assert got == {1: 5000, 4: 2000, 6: 9999, 7: 3000}
    assert starts == [0, 4, 7]                     # pages of 3, each after the last id


def test_the_run_record_says_where_the_sizes_came_from(tmp_path, world, monkeypatch):
    monkeypatch.setattr(ann_dvid, "_node_locked", lambda source: True)
    _out, r = _run(tmp_path, world, "meshes", sizes_cache=str(tmp_path / "sizes"))
    assert r["record"]["run"]["sizes"].startswith("listed >= 100 from ")
    assert r["record"]["run"]["sizes"].endswith(".sizes.parquet")
    _out, r2 = _run(tmp_path, world, "meshes", sizes_cache=str(tmp_path / "sizes"))
    assert world["asked"]["listed"] == [100]                       # the second run read it
    assert r2["bodies"]["body"].tolist() == r["bodies"]["body"].tolist()


@pytest.mark.parametrize("extra, message", [
    (["--criteria", "C", "--list", "meshes"], "needs --labelmap"),
    ([], "needs --criteria"),
])
def test_cli_sizes_cache_needs_criteria_and_a_labelmap(tmp_path, world, extra, message):
    from neu_mark.cli import main

    extra = [_file(tmp_path) if a == "C" else a for a in extra]
    with pytest.raises(SystemExit, match=message):
        main(["select-bodies", "--src", URL, "--out", str(tmp_path / "x.csv"),
              "--sizes-cache", str(tmp_path / "s"), "--rules", world["rules"],
              "--annotations", KV_URL, *extra])


def test_a_kept_listing_is_found_however_the_server_is_spelled(tmp_path, lister):
    cache = str(tmp_path / "sizes")
    ann_dvid.large_bodies(NODE, 100, cache=cache)
    ann_dvid.large_bodies({**NODE, "server": "dvid.example.org/"}, 100, cache=cache)
    assert lister == [100]
