"""Bundled dask configs, layered overrides, and key validation.

The validation tests are the point of this file. An unrecognised key in an overlay
deep-merges silently and does nothing, so without a check the failure mode is a run
that uses different settings than you wrote — never an error.
"""

from __future__ import annotations

import os

import pytest
import yaml

from em_blockrun import dask_config as dc


# --------------------------------------------------------------------------- #
# bundled templates
# --------------------------------------------------------------------------- #
def test_bundled_configs_ship_and_resolve():
    """An installed command is useless if its templates are not in the wheel."""
    names = dc.bundled_configs()
    assert "dask-local" in names and "dask-slurm-example" in names
    for name in names:
        path = dc.resolve_config(name)
        assert os.path.isfile(path) and path.endswith(f"{name}.yaml")


def test_bundled_configs_are_themselves_valid():
    """Shipping a template that our own validator rejects would be absurd.

    It has happened in spirit before: the rule "reject keys absent from the base
    config" would reject `job-extra-directives`, which the SLURM template documents
    as the way to pin a node type.
    """
    for name in dc.bundled_configs():
        merged, _ = dc.load_config(name)          # validates
        assert "jobqueue" in merged, name


def test_name_with_and_without_suffix_and_explicit_path_agree():
    path = dc.resolve_config("dask-local")
    assert dc.resolve_config("dask-local.yaml") == path
    assert dc.resolve_config(path) == path


def test_a_real_path_shadows_a_bundled_name(tmp_path):
    """Order matters: a local file must win, so old invocations keep working."""
    shadow = tmp_path / "dask-local"
    shadow.write_text("jobqueue:\n  local: {}\n")
    assert dc.resolve_config(str(shadow)) == str(shadow)


def test_unknown_name_lists_the_alternatives():
    with pytest.raises(FileNotFoundError, match="dask-local"):
        dc.resolve_config("no-such-config")


# --------------------------------------------------------------------------- #
# merging
# --------------------------------------------------------------------------- #
def test_deep_merge_recurses_dicts_and_replaces_leaves():
    base = {"a": {"b": 1, "c": 2}, "list": [1, 2], "scalar": False}
    over = {"a": {"c": 99}, "list": [3], "scalar": 0.5}
    got = dc.deep_merge(base, over)
    assert got["a"] == {"b": 1, "c": 99}, "sibling key must survive"
    # Lists REPLACE: job-extra-directives is a complete argument vector, and
    # appending to it produces a command line nobody wrote.
    assert got["list"] == [3]
    # Scalars replace, so `spill: false` can be overridden back to a number.
    assert got["scalar"] == 0.5


def test_deep_merge_does_not_mutate_its_inputs():
    base = {"a": {"b": 1}}
    dc.deep_merge(base, {"a": {"b": 2}})
    assert base == {"a": {"b": 1}}


def test_overlay_keeps_template_values_it_does_not_mention(tmp_path):
    """The reason layering exists: a site file is the keys that differ, not a fork."""
    site = tmp_path / "site.yaml"
    site.write_text("jobqueue:\n  slurm:\n    account: test-account\n")
    merged, sources = dc.load_config(["dask-slurm-example", str(site)])
    slurm = merged["jobqueue"]["slurm"]
    assert slurm["account"] == "test-account"
    assert slurm["cores"] and slurm["walltime"], "template values were lost"
    assert "distributed" in merged, "sibling section was lost"
    assert len(sources) == 2


# --------------------------------------------------------------------------- #
# validation — the part that earns its keep
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("doc,expected", [
    ({"jobqueue": {"slurm": {"acount": "x"}}}, "jobqueue.slurm.acount"),
    ({"jobqeue": {"slurm": {"account": "x"}}}, "jobqeue"),
    ({"distributed": {"worker": {"memry": {"target": False}}}},
     "distributed.worker.memry"),
])
def test_typos_are_caught_at_every_depth(doc, expected):
    assert dc.unknown_keys(doc) == [expected]


def test_keys_dask_knows_but_our_templates_omit_are_accepted():
    """Validate against DASK's key tree, not against our own templates.

    `job-extra-directives` is commented out in the SLURM template as the documented
    way to pin a node type — so a template-derived whitelist would reject exactly the
    customisation we tell people to make. Same for the many valid dask-jobqueue keys
    the template does not ship.
    """
    doc = {"jobqueue": {"slurm": {
        "job-extra-directives": ["--constraint=genoa"],
        "interface": "ib0", "death-timeout": 120,
        "job-script-prologue": ["module load foo"]}}}
    assert dc.unknown_keys(doc) == []


def test_underscore_and_hyphen_spellings_both_accepted():
    """dask canonicalises them to the same key, so rejecting one is a false positive."""
    assert dc.unknown_keys({"jobqueue": {"slurm": {"log_directory": "/tmp/x"}}}) == []
    assert dc.unknown_keys({"jobqueue": {"slurm": {"log-directory": "/tmp/x"}}}) == []


def test_free_form_mappings_accept_arbitrary_keys():
    """dask ships `{}` for these and arbitrary user keys are the whole point.

    Descending into them would reject valid config: `scheduler-options` takes any
    Scheduler kwarg, `distributed.worker.resources` takes user-invented resource
    names. Recursion has to stop at an empty reference node.
    """
    assert dc.unknown_keys(
        {"jobqueue": {"slurm": {"scheduler-options": {"dashboard_address": ":0"}}}}) == []
    assert dc.unknown_keys(
        {"distributed": {"worker": {"resources": {"GPU": 1, "MY_THING": 4}}}}) == []


def test_load_config_raises_and_names_the_offending_file(tmp_path):
    """With a merge chain, 'which overlay did this come from' is the real question."""
    bad = tmp_path / "typo-site.yaml"
    bad.write_text("jobqueue:\n  slurm:\n    acount: x\n")
    with pytest.raises(dc.ConfigKeyError) as exc:
        dc.load_config(["dask-slurm-example", str(bad)])
    message = str(exc.value)
    assert "typo-site.yaml" in message and "acount" in message


def test_validation_can_be_disabled(tmp_path):
    odd = tmp_path / "odd.yaml"
    odd.write_text("jobqueue:\n  local: {}\nmy-own-thing: 1\n")
    merged, _ = dc.load_config(str(odd), validate=False)
    assert merged["my-own-thing"] == 1
    with pytest.raises(dc.ConfigKeyError):
        dc.load_config(str(odd))


def test_non_mapping_yaml_is_rejected(tmp_path):
    bad = tmp_path / "list.yaml"
    bad.write_text("- a\n- b\n")
    with pytest.raises(dc.ConfigKeyError, match="expected a mapping"):
        dc.load_config(str(bad))


# --------------------------------------------------------------------------- #
# effective-config record
# --------------------------------------------------------------------------- #
def test_effective_config_records_the_merged_result_and_its_sources(tmp_path):
    site = tmp_path / "site.yaml"
    site.write_text("jobqueue:\n  slurm:\n    account: test-account\n")
    merged, sources = dc.load_config(["dask-slurm-example", str(site)])

    out = tmp_path / "work"
    out.mkdir()
    written = dc.write_effective_config(str(out), merged, sources)
    assert os.path.dirname(written) == str(out)
    text = open(written).read()
    for source in sources:
        assert source in text, "the merge chain must be recoverable from the record"
    assert yaml.safe_load(text)["jobqueue"]["slurm"]["account"] == "test-account"


def test_effective_config_accepts_an_explicit_file_path(tmp_path):
    merged, sources = dc.load_config("dask-local")
    target = tmp_path / "chosen-name.yaml"
    assert dc.write_effective_config(str(target), merged, sources) == str(target)
    assert target.is_file()
