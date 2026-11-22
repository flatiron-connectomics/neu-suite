"""Inclusion criteria: which bodies go in a list, read from a TOML file.

``select-bodies --criteria FILE --list NAME``. Every body is in one category, given by a
rule of the rules module (``category`` unless the file's top-level ``category_rule`` names
another); a body with no annotation record gets what that rule says of an empty record. Each
table of the file other than ``[common]`` is a list, saying of every category the rule can
give whether it is

- ``"none"``: never included;
- ``"all"``: included if every necessary condition holds;
- ``"conditional"``: included if every necessary condition holds and at least one
  sufficient condition does.

A condition, in ``[<list>.conditions]``, is a set of minimums that must all hold, over
``size`` (full-resolution voxels), ``sites`` (synapse elements on the body, T-bars and PSDs)
and ``pairs`` (the T-bar -> PSD connections it takes part in, at either end). Listing several
as sufficient ORs them::

    [skeletons]
    non_fragments = "all"
    type_fragments = "conditional"
    glia = "none"
    unnamed = "none"
    necessary_conditions = []
    sufficient_conditions = ["synaptic", "large"]

    [skeletons.conditions]
    synaptic = { size = 1_000_000, sites = 10 }
    large = { size = 2_000_000, pairs = 15 }

``[common]`` is not read here: it belongs to the rules module, which decides the categories.
A top-level ``report_rules = ["corrected_name"]`` writes what each of those rules gives
every record as a table beside the list (``<list>.<rule>.csv``: body, name, value), for
what the rules decided on the way, such as a corrected misspelling.

Pure functions over tables; the fetching is in :mod:`neu_mark.ops`.
"""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

MODES = ("all", "conditional", "none")
#: Condition field -> the column of the table :meth:`Criteria.select` reads.
FIELDS = {"size": "voxels", "sites": "sites", "pairs": "pairs"}
LIST_KEYS = ("conditions", "necessary_conditions", "sufficient_conditions")
COMMON = "common"
DEFAULT_CATEGORY_RULE = "category"


@dataclass(frozen=True)
class Condition:
    """Minimums that must all hold: ``{field: minimum}``."""
    name: str
    minimums: Mapping[str, int]

    def holds(self, table: pd.DataFrame) -> pd.Series:
        ok = pd.Series(True, index=table.index)
        for name, minimum in self.minimums.items():
            ok &= table[FIELDS[name]].fillna(0) >= minimum
        return ok

    def __str__(self) -> str:
        return f"{self.name} (" + ", ".join(f"{k} >= {v:,}"
                                             for k, v in self.minimums.items()) + ")"


@dataclass(frozen=True)
class Criteria:
    """One list of a criteria file."""
    name: str
    modes: Mapping[str, str]
    necessary: tuple[Condition, ...] = ()
    sufficient: tuple[Condition, ...] = ()
    category_rule: str = DEFAULT_CATEGORY_RULE
    #: Rules whose value for each record is written beside the list (``report_rules``).
    report: tuple[str, ...] = ()
    path: str | None = None
    section: Mapping[str, Any] = field(default_factory=dict)
    common: Mapping[str, Any] | None = None
    sha256: str | None = None

    @property
    def conditions(self) -> tuple[Condition, ...]:
        return (*self.necessary, *self.sufficient)

    @property
    def fields(self) -> set[str]:
        """The condition fields this list reads: ``size``, ``sites``, ``pairs``."""
        return {f for c in self.conditions for f in c.minimums}

    def mode(self, category: str) -> str:
        try:
            return self.modes[category]
        except KeyError:
            raise ValueError(self._unlisted([category])) from None

    def check_categories(self, categories) -> None:
        """Refuse categories the list does not say what to do with."""
        unknown = sorted({str(c) for c in categories} - set(self.modes))
        if unknown:
            raise ValueError(self._unlisted(unknown))

    def _unlisted(self, unknown) -> str:
        return (f"the rule {self.category_rule!r} gives categories [{self.name}] in "
                f"{self.path or 'the criteria'} does not list: {', '.join(unknown)}. Give "
                f"each one \"all\", \"conditional\" or \"none\".")

    def select(self, table: pd.DataFrame) -> pd.Series:
        """Why each row is included: ``"all"``, the first sufficient condition that held,
        or ``None`` for a row left out.

        ``table`` has a ``category`` column and the columns of :data:`FIELDS` the
        conditions read; a missing value counts as 0.
        """
        self.check_categories(table["category"].unique())
        mode = table["category"].map(self.modes)
        necessary = pd.Series(True, index=table.index)
        for c in self.necessary:
            necessary &= c.holds(table)
        why = pd.Series(None, index=table.index, dtype="object")
        for c in reversed(self.sufficient):              # the first that held wins
            why = why.mask(c.holds(table), c.name)
        out = pd.Series(None, index=table.index, dtype="object")
        out = out.mask(necessary & (mode == "all"), "all")
        out = out.mask(necessary & (mode == "conditional") & why.notna(), why)
        return out.astype(object).where(out.notna(), None)

    def record(self) -> dict[str, Any]:
        """What the provenance record keeps: the file, the list and everything it said."""
        return {"path": self.path, "sha256": self.sha256, "list": self.name,
                "category_rule": self.category_rule, "report_rules": list(self.report),
                "section": dict(self.section),
                "common": dict(self.common) if self.common is not None else None}


def lists(path: str | Path) -> list[str]:
    """The list names a criteria file defines."""
    doc = tomllib.loads(Path(path).expanduser().read_text())
    return [k for k, v in doc.items() if isinstance(v, dict) and k != COMMON]


def load(path: str | Path, name: str) -> Criteria:
    """List ``name`` of the criteria file at ``path``, checked."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"no criteria file at {path}")
    raw = path.read_bytes()
    doc = tomllib.loads(raw.decode())
    where = f"{path} [{name}]"
    available = [k for k, v in doc.items() if isinstance(v, dict) and k != COMMON]
    if name == COMMON or name not in available:
        raise ValueError(f"{path} has no list {name!r}; it has: "
                         f"{', '.join(available) or 'none'}")
    category_rule = doc.get("category_rule", DEFAULT_CATEGORY_RULE)
    if not isinstance(category_rule, str):
        raise ValueError(f"{path}: category_rule must be a rule name")
    report = doc.get("report_rules", [])
    if not isinstance(report, list) or not all(isinstance(r, str) for r in report):
        raise ValueError(f"{path}: report_rules must be a list of rule names")
    for key, value in doc.items():
        if not isinstance(value, dict) and key not in ("category_rule", "report_rules"):
            raise ValueError(f"{path}: unknown top-level key {key!r}")

    section = doc[name]
    modes: dict[str, str] = {}
    for key, value in section.items():
        if key in LIST_KEYS:
            continue
        if not isinstance(value, str):
            raise ValueError(f"{where}: {key!r} is neither a category (\"all\", "
                             f"\"conditional\" or \"none\") nor one of {', '.join(LIST_KEYS)}")
        if value not in MODES:
            raise ValueError(f"{where}: {key} = {value!r}; a category is "
                             f"{', '.join(repr(m) for m in MODES)}")
        modes[key] = value
    if not modes:
        raise ValueError(f"{where} lists no categories")

    defined = _conditions(section.get("conditions", {}), where)
    necessary = _named(section.get("necessary_conditions", []), defined,
                       "necessary_conditions", where)
    sufficient = _named(section.get("sufficient_conditions", []), defined,
                        "sufficient_conditions", where)
    conditional = sorted(k for k, v in modes.items() if v == "conditional")
    if conditional and not sufficient:
        raise ValueError(f"{where}: {', '.join(conditional)} are \"conditional\" but "
                         f"sufficient_conditions is empty, so they could never be included")
    unused = sorted(set(defined) - {c.name for c in (*necessary, *sufficient)})
    if unused:
        raise ValueError(f"{where}: conditions {', '.join(unused)} are defined but in "
                         f"neither list; remove them or name them")
    return Criteria(name=name, modes=modes, necessary=necessary, sufficient=sufficient,
                    category_rule=category_rule, report=tuple(report),
                    path=str(path), section=section,
                    common=doc.get(COMMON), sha256=hashlib.sha256(raw).hexdigest())


def _conditions(table: Any, where: str) -> dict[str, Condition]:
    if not isinstance(table, dict):
        raise ValueError(f"{where}: conditions must be a table of conditions")
    out = {}
    for name, minimums in table.items():
        if not isinstance(minimums, dict) or not minimums:
            raise ValueError(f"{where}: condition {name!r} must be a non-empty table of "
                             f"minimums, e.g. {{ size = 500_000, sites = 3 }}")
        for key, value in minimums.items():
            if key not in FIELDS:
                raise ValueError(f"{where}: condition {name!r} has {key!r}; the fields are "
                                 f"{', '.join(FIELDS)}")
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{where}: condition {name!r}: {key} must be a whole "
                                 f"number >= 0, not {value!r}")
        out[name] = Condition(name, dict(minimums))
    return out


def _named(names: Any, defined: Mapping[str, Condition], key: str,
           where: str) -> tuple[Condition, ...]:
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise ValueError(f"{where}: {key} must be a list of condition names")
    missing = [n for n in names if n not in defined]
    if missing:
        raise ValueError(f"{where}: {key} names {', '.join(missing)}, not in "
                         f"[{where.rsplit('[', 1)[-1].rstrip(']')}.conditions]")
    return tuple(defined[n] for n in names)


def candidate_sources(criteria: Criteria, category: str) -> dict[str, Any]:
    """How to find the bodies of ``category`` that have no annotation record.

    They cannot be enumerated from the records, so each sufficient condition says where to
    look: one with ``sites`` -> labelsz at that threshold; else one with ``pairs`` -> the
    pair counts; else (``size`` alone) -> a listing of every label of at least that size.
    Returns ``{"sites": min or None, "pairs": min or None, "size": min or None}``; all
    ``None`` when the category is ``"none"``. ``"all"`` is refused: it would be every body
    in the labelmap.
    """
    mode = criteria.mode(category)
    out: dict[str, Any] = {"sites": None, "pairs": None, "size": None}
    if mode == "none":
        return out
    if mode == "all":
        raise ValueError(f"[{criteria.name}] {category} = \"all\" would select every body in "
                         f"the labelmap: {category!r} is what bodies with no annotation "
                         f"record are. Use \"conditional\" or \"none\".")
    for c in criteria.sufficient:
        binding = {k: v for k, v in c.minimums.items() if v > 0}
        if not binding:
            raise ValueError(f"[{criteria.name}] condition {c.name!r} holds for every body, "
                             f"so with {category} = \"conditional\" it would select every "
                             f"body in the labelmap")
        key = next(k for k in ("sites", "pairs", "size") if k in binding)
        value = binding[key]
        out[key] = value if out[key] is None else min(out[key], value)
    return out


def smallest(criteria: Criteria, name: str) -> int | None:
    """The lowest nonzero minimum any condition sets on field ``name``."""
    values = [c.minimums[name] for c in criteria.conditions if c.minimums.get(name)]
    return min(values) if values else None
