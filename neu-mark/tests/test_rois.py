"""Labelling each synapse with the neuropil it falls in."""

import pandas as pd
import pytest

from conftest import URL, _open
from neu_mark import dvid as ad
from neu_mark import io, notebook as nb, ops, tables


def _src():
    from neu_vol.dvid import parse_url

    return {"backend": "dvid", **parse_url(URL)}


# --------------------------------------------------------------------------- #
# overlap
# --------------------------------------------------------------------------- #
def test_overlap_proceeds_by_default_and_is_measured_in_synapses(dvid_server, caplog):
    """Refusing is the wrong default, since small intersections are expected. The ambiguous
    count comes from labelling again with the ROI order reversed: in the stub, x=10 takes
    the first ROI and x=11 the last, so both points change hands."""
    pts, _ = nb.points(_open(), [1, 2])
    with caplog.at_level("WARNING"):
        out = ad.label_point_rois(_open(), pts, ["ME(L)", "OL(L)"])
    assert out["overlapping"]
    assert "attributed by ROI ORDER" in caplog.text
    assert out["ambiguous"] == 2
    assert out["ambiguous_pairs"] == {"ME(L) | OL(L)": 1, "OL(L) | ME(L)": 1}


def test_a_non_overlapping_set_does_no_second_unpack(dvid_server):
    pts, _ = nb.points(_open(), [1, 2])
    out = ad.label_point_rois(_open(), pts, ["ME(L)", "LO(L)"])
    assert out["overlapping"] == [] and out["ambiguous"] == 0


def test_strict_mode_refuses_and_points_at_the_subtracted_variants(dvid_server):
    pts, _ = nb.points(_open(), [1, 2])
    with pytest.raises(ValueError, match=r"INP\(-ATL\)\(L\)"):
        ad.label_point_rois(_open(), pts, ["ME(L)", "OL(L)"], on_overlap="error")


def test_an_unknown_overlap_policy_lists_the_valid_ones(dvid_server):
    pts, _ = nb.points(_open(), [1, 2])
    with pytest.raises(ValueError, match="on_overlap must be one of"):
        ad.label_point_rois(_open(), pts, ["ME(L)"], on_overlap="shrug")


def test_cli_strict_rois_refuses(tmp_path, dvid_server):
    from neu_mark.cli import main

    with pytest.raises(SystemExit, match="ROIs overlap"):
        main(["points", "--src", URL, "--out", str(tmp_path / "o"), "--bodies", "1,2",
              "--rois", "ME(L),OL(L)", "--strict-rois"])


def test_cli_reports_the_ambiguity_by_default(tmp_path, dvid_server, capsys):
    from neu_mark.cli import main

    main(["points", "--src", URL, "--out", str(tmp_path / "o"), "--bodies", "1,2",
          "--rois", "ME(L),OL(L)"])
    printed = capsys.readouterr().out
    assert "1 ROI pairs intersect; 2 synapse(s)" in printed
    assert "attributed by ROI ORDER" in printed


# --------------------------------------------------------------------------- #
# the roi column
# --------------------------------------------------------------------------- #
def test_the_points_frame_gains_a_roi_column_and_relationships_do_not(dvid_server):
    """The label index means nothing without the ROI list, so it is dropped. Relationships
    span two points that may be in different neuropils, so they get no column."""
    pts, rels = nb.points(_open(), [1, 2], rois=["ME(L)", "LO(L)"])
    by_body = pts.set_index("body")["roi"]          # x=10 -> the first ROI, x=11 -> the last
    assert by_body.loc[1] == "ME(L)" and by_body.loc[2] == "LO(L)"
    assert "roi_label" not in pts.columns
    assert "roi" not in rels.columns


def test_a_point_in_no_roi_is_null_not_a_region_called_unspecified(dvid_server):
    """`<unspecified>` is neuclease's name for label 0. "In no ROI" is missing data."""
    pts, _ = nb.points(_open(), [1, 2])
    out = ad.label_point_rois(_open(), pts.assign(x=999), ["ME(L)"])
    assert out["points"]["roi"].isna().all()
    assert ad.ROI_UNSPECIFIED not in set(out["points"]["roi"].dropna())
    assert out["unlabeled"] == len(pts) and out["labeled"] == 0


def test_counts_are_reported_and_the_callers_frame_is_not_mutated(dvid_server):
    """neuclease labels in place; that must not reach the caller's frame."""
    pts, _ = nb.points(_open(), [1, 2])
    before = list(pts.columns)
    out = ad.label_point_rois(_open(), pts, ["ME(L)", "LO(L)"])
    assert list(pts.columns) == before
    assert out["labeled"] + out["unlabeled"] == len(pts)
    assert out["counts"] == {"ME(L)": 1, "LO(L)": 1}
    assert out["rois"] == ["ME(L)", "LO(L)"]


def test_an_empty_points_frame_is_handled(dvid_server):
    pts, _ = nb.points(_open(), [999999])
    out = ad.label_point_rois(_open(), pts, ["ME(L)"])
    assert out["labeled"] == 0 and "roi" in out["points"].columns


# --------------------------------------------------------------------------- #
# the per-body aggregate
# --------------------------------------------------------------------------- #
def test_body_roi_counts_is_long_form(dvid_server):
    pts, _ = nb.points(_open(), [1, 2], rois=["ME(L)", "LO(L)"])
    out = tables.body_roi_counts(pts)
    assert set(out.columns) == {"body", "kind", "roi", "synapses"}
    assert int(out["synapses"].sum()) == 2


def test_body_roi_counts_without_a_roi_column_says_how_to_get_one(dvid_server):
    pts, _ = nb.points(_open(), [1, 2])
    with pytest.raises(KeyError, match="--rois"):
        tables.body_roi_counts(pts)


# --------------------------------------------------------------------------- #
# through ops and the CLI
# --------------------------------------------------------------------------- #
def test_ops_writes_the_roi_column_and_records_the_set(tmp_path, dvid_server):
    from neu_vol.location import read_json

    out = str(tmp_path / "syn")
    ops.fetch_points(_open(), out, [1, 2], rois=["ME(L)", "LO(L)"])
    assert io.read_table(out, "points")["roi"].notna().sum() == 2
    run = read_json(out, "provenance.json")["run"]
    assert run["rois"]["rois"] == ["ME(L)", "LO(L)"]
    assert run["rois"]["labeled"] == 2


def test_no_rois_means_no_column_and_no_record(tmp_path, dvid_server):
    from neu_vol.location import read_json

    out = str(tmp_path / "syn")
    ops.fetch_points(_open(), out, [1, 2])
    assert "roi" not in io.read_table(out, "points").columns
    assert read_json(out, "provenance.json")["run"].get("rois") is None


def test_cli_reports_how_many_synapses_landed_in_a_roi(tmp_path, dvid_server, capsys):
    from neu_mark.cli import main

    main(["points", "--src", URL, "--out", str(tmp_path / "o"), "--bodies", "1,2",
          "--rois", "ME(L),LO(L)"])
    printed = capsys.readouterr().out
    assert "rois: 2 of 2 synapses inside one of 2 ROIs" in printed


def _roi_nodes_asked(monkeypatch):
    """The uuid of every ROI fetch and point lookup, from here on."""
    import neuclease.dvid.roi as ndroi

    asked = []
    for name in ("fetch_roi_ranges_and_boxes", "determine_point_rois"):
        stub = getattr(ndroi, name)

        def recording(server, uuid, *a, _stub=stub, **k):
            asked.append(uuid)
            return _stub(server, uuid, *a, **k)

        monkeypatch.setattr(ndroi, name, recording)
    return asked


def test_rois_are_read_from_the_points_node_by_default(tmp_path, dvid_server, monkeypatch):
    from neu_vol.location import read_json

    asked = _roi_nodes_asked(monkeypatch)
    out = str(tmp_path / "syn")
    ops.fetch_points(_open(), out, [1, 2], rois=["ME(L)", "LO(L)"])
    assert set(asked) == {"d38898"}
    assert read_json(out, "provenance.json")["run"]["rois"]["node"]["uuid"] == "d38898"


def test_cli_roi_source_labels_from_another_node_and_records_it(tmp_path, dvid_server,
                                                                 monkeypatch):
    """The points and body ids stay on --src's node (HEAD, d38898); only the ROIs move."""
    from neu_mark.cli import main
    from neu_vol.location import read_json

    asked = _roi_nodes_asked(monkeypatch)
    out = str(tmp_path / "o")
    main(["points", "--src", URL, "--out", out, "--bodies", "1,2",
          "--rois", "ME(L),LO(L)", "--roi-source", "dvid://dvid.example.org/846e3a"])
    assert set(asked) == {"846e3a"}
    assert io.read_table(out, "points")["roi"].notna().sum() == 2
    record = read_json(out, "provenance.json")
    assert record["source"]["uuid"] == "d38898"
    assert record["run"]["rois"]["node"] == {"server": "dvid.example.org", "uuid": "846e3a"}


@pytest.mark.parametrize("argv, match", [
    (["--rois", "ME(L)", "--roi-source", "dvid://dvid.example.org/846e3a/synapses"],
     r"--roi-source: .*names an instance.*dvid://dvid.example.org/846e3a"),
    (["--roi-source", "dvid://dvid.example.org/846e3a"], "pass --rois too"),
])
def test_cli_roi_source_is_refused_without_a_node_or_without_rois(tmp_path, dvid_server,
                                                                  argv, match):
    from neu_mark.cli import main

    with pytest.raises(SystemExit, match=match):
        main(["points", "--src", URL, "--out", str(tmp_path / "o"), "--bodies", "1,2",
              *argv])


def test_cli_takes_a_roi_file(tmp_path, dvid_server, capsys):
    path = tmp_path / "rois.txt"
    path.write_text("# optic lobe\nME(L)\nLO(L)   # left only\n")
    from neu_mark.cli import main

    main(["points", "--src", URL, "--out", str(tmp_path / "o"), "--bodies", "1,2",
          "--rois", str(path)])
    assert "of 2 ROIs" in capsys.readouterr().out


def test_cli_resolves_a_config_roi_set(tmp_path, dvid_server, capsys, monkeypatch):
    from neu_mark import config
    from neu_mark.cli import main

    cfg = tmp_path / "c.toml"
    cfg.write_text('[dvid]\nserver = "s"\nuuid = "u"\n\n'
                   '[roi_sets]\noptic = ["ME(L)", "LO(L)"]\n')
    monkeypatch.setenv(config.ENV_VAR, str(cfg))
    main(["points", "--src", URL, "--out", str(tmp_path / "o"), "--bodies", "1,2",
          "--rois", "@optic"])
    printed = capsys.readouterr().out
    assert "--rois @optic  ->  2 ROIs" in printed


def test_an_unknown_config_roi_set_lists_the_known_ones(tmp_path, monkeypatch):
    from neu_mark import config

    cfg = tmp_path / "c.toml"
    cfg.write_text('[roi_sets]\noptic = ["ME(L)"]\n')
    monkeypatch.setenv(config.ENV_VAR, str(cfg))
    with pytest.raises(ValueError, match="Configured sets: optic"):
        nb.roi_set("@nope")


def test_a_comma_list_and_a_python_list_are_equivalent():
    assert nb.roi_set("ME(L), LO(L)") == ["ME(L)", "LO(L)"]
    assert nb.roi_set(["ME(L)", "LO(L)"]) == ["ME(L)", "LO(L)"]
