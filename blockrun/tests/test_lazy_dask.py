"""Importing blockrun must not import dask (`dask.distributed` costs ~1 s); `start_dask`
resolves on first access. Facts are gathered in one subprocess, since this session has
already imported dask."""

import json
import subprocess
import sys

import pytest

# One script, every fact. Order matters: the "before" reading has to be taken before
# anything touches `start_dask`, since touching it is what pulls dask in.
_PROBE = r"""
import json, sys

def dask_modules():
    return sorted(m for m in sys.modules
                  if m in ("dask", "distributed") or m.startswith(("dask.", "distributed.")))

import blockrun
after_import = dask_modules()
via_attr = callable(blockrun.start_dask)
after_touch = bool(dask_modules())
from blockrun import start_dask
print(json.dumps({"after_import": after_import, "via_attr": via_attr,
                  "via_from_import": callable(start_dask), "after_touch": after_touch}))
"""


@pytest.fixture(scope="session")
def probe():
    out = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True,
                         check=True)
    return json.loads(out.stdout)


def test_importing_the_package_does_not_import_dask(probe):
    assert probe["after_import"] == [], (
        f"importing blockrun now pulls in {probe['after_import']}. That is ~1 s on "
        f"every consumer's startup, including read-only ops that never build a cluster. "
        f"Keep start_dask behind the module __getattr__.")


def test_start_dask_is_still_importable_both_ways(probe):
    assert probe["via_from_import"], "`from blockrun import start_dask` broke"
    assert probe["via_attr"], "`blockrun.start_dask` broke"


def test_touching_start_dask_is_what_pulls_dask_in(probe):
    assert probe["after_touch"], "start_dask resolved without importing dask at all"


def test_a_missing_attribute_still_raises_attribute_error():
    import blockrun

    with pytest.raises(AttributeError, match="not_a_real_name"):
        blockrun.not_a_real_name


def test_dir_still_lists_the_public_api():
    import blockrun

    assert "start_dask" in dir(blockrun)
    assert set(blockrun.__all__) <= set(dir(blockrun))
