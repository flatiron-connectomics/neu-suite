"""SLURM workers get the config through their job script, not from the driver."""

from __future__ import annotations

import copy
import shlex
import sys
import types

import dask
import pytest
import yaml

from blockrun import dask_config as dc
from blockrun.dask_runner import start_dask, worker_config_exports


def _leaves(node, prefix=""):
    for key, value in node.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and value:
            yield from _leaves(value, path)
        else:
            yield path, value


@pytest.fixture
def config(tmp_path):
    site = tmp_path / "site.yaml"
    site.write_text(yaml.safe_dump({"distributed": {"worker": {"memory": {"target": 0.6}},
                                                    "comm": {"timeouts": {"connect": "90s"}}}}))
    merged, _ = dc.load_config(["dask-slurm-example", str(site)])
    return merged


def test_every_setting_outside_jobqueue_round_trips_through_the_environment(config):
    lines = worker_config_exports(config)
    env = dict(shlex.split(line)[1].split("=", 1) for line in lines)
    assert not any(k.startswith("DASK_JOBQUEUE") for k in env)
    collected = dask.config.collect_env(env)
    leaves = list(_leaves({k: v for k, v in config.items() if k != "jobqueue"}))
    assert ("distributed.worker.memory.pause", False) in leaves
    for path, value in leaves:
        assert dask.config.get(path, config=collected) == value, path


def test_start_dask_puts_the_exports_in_the_slurm_job_script(config, monkeypatch, tmp_path):
    seen = {}

    class Captured(Exception):
        pass

    def fake_cluster(**kwargs):
        seen.update(kwargs)
        raise Captured

    monkeypatch.setitem(sys.modules, "dask_jobqueue", types.SimpleNamespace(SLURMCluster=fake_cluster))
    monkeypatch.setattr(dask.config, "config", copy.deepcopy(dask.config.config))
    cfg = tmp_path / "slurm.yaml"
    cfg.write_text(yaml.safe_dump(config))
    with pytest.raises(Captured):
        with start_dask(2, str(cfg)):
            pass
    prologue = seen["job_script_prologue"]
    assert "export OMP_NUM_THREADS=1" in prologue
    assert "export DASK_DISTRIBUTED__WORKER__MEMORY__PAUSE=False" in prologue
    assert "export DASK_DISTRIBUTED__WORKER__MEMORY__TERMINATE=False" in prologue
