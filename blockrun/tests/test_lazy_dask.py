"""Importing blockrun must not import dask.

`dask.distributed` is ~1 s of import, and it is the single largest cost in starting any
consumer — including `neu-vol info`, which reads one metadata document, and `neu-morpho
progress`, which reads a JSONL file. Neither goes near a cluster. So `start_dask` is
resolved through a module ``__getattr__`` (PEP 562) instead of being imported eagerly,
and this file pins that: the saving evaporates the moment someone reinstates a top-level
``from .dask_runner import start_dask``, and nothing else would fail.

The facts have to be gathered in a **subprocess**, because this test session has
certainly already imported dask by way of another test. They are gathered in **one**
subprocess, shared by every test here through a session fixture: each spawn costs more
than all these assertions put together, and this suite is otherwise under a second.
"""

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
    """Laziness must not change the public API — both spellings have to keep working."""
    assert probe["via_from_import"], "`from blockrun import start_dask` broke"
    assert probe["via_attr"], "`blockrun.start_dask` broke"


def test_touching_start_dask_is_what_pulls_dask_in(probe):
    """The other half of the contract: dask is deferred, not dropped."""
    assert probe["after_touch"], "start_dask resolved without importing dask at all"


def test_a_missing_attribute_still_raises_attribute_error():
    """A hand-written __getattr__ that forgets to raise turns typos into None."""
    import blockrun

    with pytest.raises(AttributeError, match="not_a_real_name"):
        blockrun.not_a_real_name


def test_dir_still_lists_the_public_api():
    """`__getattr__` hides names from `dir()` unless `__dir__` reports them, and that is
    what tab-completion and `help()` read."""
    import blockrun

    assert "start_dask" in dir(blockrun)
    assert set(blockrun.__all__) <= set(dir(blockrun))
