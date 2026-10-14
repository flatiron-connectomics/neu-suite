"""neu-mark: annotations out of DVID, into tables and neuroglancer layers.

Two kinds of annotation:

- **point annotations**: synapses (``synapses``), and anything else stored in a DVID
  ``annotation`` instance (``labels_todo``, ``bookmark_annotations``).
- **body annotations**: the per-body records in a ``keyvalue`` instance
  (``labels_annotations``), such as neuron name, status and who traced it.

Both are fetched into columnar tables first (:mod:`neu_mark.tables`, :mod:`neu_mark.io`),
which analysis reads and precomputed layers are built from, so changing a display never
needs a new fetch.

Everything is driven by an explicit body list: most label ids are fragments, and DVID has
no cheap way to enumerate the bodies worth fetching (see :mod:`neu_mark.bodies`).
"""

__version__ = "0.1.0"

#: Re-exported from `notebook` (fetches that return DataFrames) and `rules`, resolved on
#: first access (PEP 562) so that `neu-mark --help` does not import pandas and neu-vol.
_LAZY = {
    "source": "notebook",
    "select_bodies": "notebook",
    "points": "notebook",
    "body_annotations": "notebook",
    "all_body_annotations": "notebook",
    "synapse_counts": "notebook",
    "body_ids": "notebook",
    "rule": "rules",
    "Rule": "rules",
    "RuleSet": "rules",
}

__all__ = ["__version__", *sorted(_LAZY)]


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        module = importlib.import_module(f".{_LAZY[name]}", __name__)
        value = getattr(module, name)
        globals()[name] = value          # cached
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted({*globals(), *_LAZY})
