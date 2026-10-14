"""The ``@rule`` decorator and ``RuleSet``: declaring how annotation strings become facets.

A plain rule is any callable taking one row and returning a value, a sequence, or
``None``; :mod:`neu_mark.explore` accepts those. A :class:`Rule` is that callable plus its
declarations (see :func:`rule`), which :func:`neu_mark.segprops.build` needs in order to
know which rules mint tags, which exclude a body, and what each facet is called. Explore
with plain callables; publish with a :class:`RuleSet`.

A rules module is dataset content (which prefixes are neuropils, which suffixes mark
completeness), kept with the body lists and loaded by path with :func:`from_module`.
:meth:`RuleSet.override` merges a dataset module over the builtins by name.

    >>> side.test("Tm2_A2(L)")
    'L'
    >>> RULES.explain("LN_C5(L)_NCL")
    {'side': 'L', ...}
    >>> RULES.remainder("LN_C5(L)_NCL")         # what no rule accounted for
    'LN'
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import pandas as pd

from . import explore

logger = logging.getLogger(__name__)

#: Default column a rule reads, and the one `test` fills when given a bare string.
DEFAULT_COLUMN = "instance"

#: Reserved rule names, which set a segment property rather than a tag (declared
#: ``tag=False``; see :mod:`neu_mark.segprops`). Any other ``tag=False`` rule is evaluated
#: and coverage-reported but emits nothing.
RESERVED = ("label", "description")


class Rule:
    """One named facet derived from a row. Callable exactly like the function it wraps."""

    def __init__(self, func: Callable, *, name: str | None = None,
                 needs: Sequence[str] | None = None, multi: bool = False,
                 tag: bool = True, prefix: str | None = None,
                 description: str | None = None, drop: bool = False,
                 consumes: Iterable[str] | Callable | None = None):
        self.func = func
        self.name = name or getattr(func, "__name__", "rule")
        self.needs = tuple(needs) if needs else (DEFAULT_COLUMN,)
        self.multi = bool(multi)
        #: Which tokens of the source string this rule accounts for, for :meth:`remainder`.
        #: ``None`` uses the returned values; declare it where the rule normalizes an alias
        #: (``NCL`` -> ``nucleated``). A callable takes the row.
        self._consumes = consumes
        #: Whether this rule excludes the body it fires on; its return value is the reason,
        #: counted in the build report.
        self.drop = bool(drop)
        if self.drop and self.multi:
            raise ValueError(
                f"rule {self.name!r} is declared drop=True and multi=True. A body is either "
                f"excluded or not, so a drop rule returns one reason or None.")
        #: Whether this facet becomes neuroglancer tags. A source has one `tags` property, so
        #: every tagged facet pools into it under `prefix`. Drop rules are never tagged.
        self.tag = False if self.drop else bool(tag)
        if self.name in RESERVED and self.tag:
            raise ValueError(
                f"{self.name!r} is a reserved rule name: it sets the segment's {self.name} "
                f"PROPERTY, not a tag. Declare it @rule(name={self.name!r}, tag=False). "
                f"Left tagged it would quietly mint {self.name}:… tags instead, which is a "
                f"valid document that simply never sets the property.")
        self.prefix = prefix if prefix is not None else self.name
        self.description = description or (func.__doc__ or "").strip().split("\n")[0]
        self.__doc__ = func.__doc__
        self.__name__ = self.name

    def _parts(self) -> tuple[str, str, str, str, str]:
        """``(name, kind, needs, extras, description)``, the fields of a rule's repr, also
        used by :meth:`RuleSet.__repr__`."""
        kind = "drop" if self.drop else "tag" if self.tag else "property"
        if self.tag:
            kind += f" {self.prefix}:" if self.prefix else " bare"
        extras = ["multi"] if self.multi else []
        if self._consumes is not None:
            extras.append("consumes=" + ("callable" if callable(self._consumes) else
                                         "[" + ", ".join(str(c)
                                                         for c in self._consumes) + "]"))
        return (self.name, kind, "needs=" + "+".join(self.needs), " ".join(extras),
                self.description)

    def __repr__(self) -> str:
        """Everything the rule declares, ``|``-separated on one line, then its description."""
        name, kind, needs, extras, description = self._parts()
        fields = " | ".join(p for p in (name, kind, needs, extras) if p)
        return f"<Rule {fields} — {description}>" if description else f"<Rule {fields}>"

    def __call__(self, row) -> Any:
        return self.func(row)

    def _invoke(self, row) -> Any:
        """Call the function, turning a read of an undeclared column into an error that
        names the rule and the fix, rather than a bare ``KeyError``."""
        try:
            return self.func(row)
        except KeyError as exc:
            column = exc.args[0] if exc.args else None
            if not isinstance(column, str) or column in self.needs:
                raise
            raise KeyError(
                f"rule {self.name!r} read the column {column!r}, which it does not "
                f"declare — so nothing could check for it up front. Write "
                f"@rule(needs={[*self.needs, column]!r}). Currently declared: "
                f"{', '.join(self.needs)}.") from exc

    def values(self, row) -> tuple:
        """The rule's output as a tuple, validated against ``multi``."""
        got = explore._as_tuple(self._invoke(row))
        if not self.multi and len(got) > 1:
            raise ValueError(
                f"rule {self.name!r} is declared single-valued but returned {len(got)} "
                f"values {got!r}. Either it should use re.search rather than re.findall, "
                f"or the facet really is multi-valued — in which case declare "
                f"@rule(multi=True), which is free: neuroglancer's `tags` property is a "
                f"list per segment anyway.")
        return got

    def scalar(self, row) -> Any:
        """The rule's single value, or ``None``. For the single-valued kinds: ``drop``
        reasons and the reserved ``label`` rule, where a tuple would only be unwrapped."""
        got = self.values(row)
        return got[0] if got else None

    def consumed(self, row) -> tuple[str, ...]:
        """The tokens of the source string this rule accounts for.

        A rule that does not read the source string (``instance`` is not in ``needs``)
        consumes nothing, so a rule returning the curated ``type`` cannot remove the token
        a cell-type remainder wants to keep.
        """
        if DEFAULT_COLUMN not in self.needs:
            return ()
        if self._consumes is None:
            candidates = self.values(row)
        elif callable(self._consumes):
            candidates = explore._as_tuple(self._consumes(row))
        else:
            candidates = tuple(self._consumes)
        wanted = {str(c).lower() for c in candidates}
        return tuple(t for t in explore.split_tokens(row[DEFAULT_COLUMN])
                     if t.lower() in wanted)

    def remainder(self, _value: str | None = None, /, **fields) -> str | None:
        """The source string with this rule's tokens removed, rejoined with ``_``.

        Removal is by token (the strings are ``_``- and parenthesis-delimited), not by
        substring, so ``side``'s ``"L"`` does not eat the ``L`` of ``LC10``.
        """
        row = _row({DEFAULT_COLUMN: _value, **fields} if _value is not None else fields,
                   needs=self.needs)
        drop = {t.lower() for t in self.consumed(row)}
        kept = [t for t in explore.split_tokens(row[DEFAULT_COLUMN])
                if t.lower() not in drop]
        return "_".join(kept) or None

    def test(self, _value: str | None = None, /, **fields) -> Any:
        """Evaluate on one synthetic row; ``rule.test("Tm2_A2(L)")`` fills ``instance``.

        Returns the rule's own return value. Missing fields arrive as ``None``, as in a run.
        """
        if _value is not None:
            fields = {DEFAULT_COLUMN: _value, **fields}
        row = _row(fields, needs=self.needs)
        return self._invoke(row)


def rule(func: Callable | None = None, /, *, needs: Sequence[str] | None = None,
         multi: bool = False, tag: bool = True, prefix: str | None = None,
         name: str | None = None, description: str | None = None,
         drop: bool = False, consumes: Iterable[str] | Callable | None = None):
    """Declare a function as a rule. Usable bare (``@rule``) or called (``@rule(...)``).

    - ``needs``: the columns the rule reads (default ``instance``), checked up front.
    - ``multi``: whether the facet can have several values. A single-valued rule that
      returns several raises, which catches ``re.findall`` used where ``re.search`` was meant.
    - ``prefix``: the tag namespace, defaulting to the rule's name (``glomerulus:VA1v``);
      ``prefix=""`` mints a bare tag.
    - ``drop``: the rule excludes the body it fires on, returning the reason.
    - ``consumes``: the tokens it accounts for, for :meth:`RuleSet.remainder`.
    - ``tag=False``: the rule produces a property, not a tag (see :data:`RESERVED`).

    The docstring's first line becomes the description shown in reports and in the
    published ``tag_descriptions``. A rule may read a column an earlier rule produced.
    """
    def wrap(f: Callable) -> Rule:
        return Rule(f, name=name, needs=needs, multi=multi, tag=tag, prefix=prefix,
                    description=description, drop=drop, consumes=consumes)

    return wrap(func) if func is not None else wrap


def _row(fields: Mapping[str, Any], *, needs: Iterable[str] = ()) -> pd.Series:
    """A single row as a rule sees it: declared columns present, missing ones ``None``."""
    data = {str(k): v for k, v in fields.items()}
    for column in needs:
        data.setdefault(column, None)
    return pd.Series(data, dtype="object")


class RuleSet:
    """An ordered collection of rules, plus the reports for iterating on them."""

    def __init__(self, rules: Iterable[Rule] | Mapping[str, Callable],
                 *, keep: Sequence[str] | None = None, source: str | None = None):
        if isinstance(rules, Mapping):
            items = [r if isinstance(r, Rule) else Rule(r, name=n)
                     for n, r in rules.items()]
        else:
            items = list(rules)
        for r in items:
            if not isinstance(r, Rule):
                raise TypeError(f"{r!r} is not a Rule; decorate it with @rule")
        names = [r.name for r in items]
        duplicated = {n for n in names if names.count(n) > 1}
        if duplicated:
            raise ValueError(f"duplicate rule names: {', '.join(sorted(duplicated))}")
        self.rules = items
        #: Source columns kept in a published layer (an allow-list).
        self.keep = tuple(keep) if keep is not None else ()
        self.source = source

    def __repr__(self) -> str:
        """The rules in evaluation order, one per line, aligned into columns.
        :meth:`describe` gives the same information as a DataFrame."""
        where = f", from {self.source}" if self.source else ""
        header = f"<RuleSet {len(self.rules)} rules{where}>"
        rows = [r._parts() for r in self.rules]
        if not rows:
            return header
        widths = [max(len(row[i]) for row in rows) for i in range(4)]
        lines = []
        for name, kind, needs, extras, description in rows:
            fields = "  ".join(text.ljust(width) for text, width in
                               zip((name, kind, needs, extras), widths))
            # Strip trailing padding only when no description follows, to keep alignment.
            lines.append(f"  {fields}  — {description}" if description
                         else f"  {fields}".rstrip())
        return "\n".join([header, *lines])

    def _repr_html_(self) -> str:
        """Render as :meth:`describe`'s table in a notebook (IPython's display protocol)."""
        return self.describe().to_html(index=False)

    def __len__(self) -> int:
        return len(self.rules)

    def __iter__(self):
        return iter(self.rules)

    def __getitem__(self, name: str) -> Rule:
        for r in self.rules:
            if r.name == name:
                return r
        raise KeyError(f"no rule named {name!r}; have {', '.join(self.names)}")

    @property
    def names(self) -> list[str]:
        return [r.name for r in self.rules]

    @property
    def tagged(self) -> list[Rule]:
        """The rules that mint tags, in declaration order."""
        return [r for r in self.rules if r.tag]

    @property
    def drops(self) -> list[Rule]:
        """The rules that exclude a body. Evaluated first, so an excluded body does not
        count toward other rules' coverage."""
        return [r for r in self.rules if r.drop]

    def override(self, other: "RuleSet | Iterable[Rule] | Mapping[str, Callable]",
                 ) -> "RuleSet":
        """This set with ``other``'s rules replacing same-named ones, others appended.

        A replacement keeps the position of the rule it replaces, so evaluation order is
        unchanged. (``__init__`` refuses duplicate names; overriding is how a dataset
        module redefines a builtin.)
        """
        incoming = other if isinstance(other, RuleSet) else RuleSet(other)
        by_name = {r.name: r for r in incoming.rules}
        replaced = [n for n in self.names if n in by_name]
        merged = [by_name.pop(r.name, r) for r in self.rules]
        merged.extend(by_name.values())
        if replaced:
            logger.info("rules module overrides the builtin rule(s): %s",
                        ", ".join(replaced))
        return RuleSet(merged, keep=incoming.keep or self.keep,
                       source=incoming.source or self.source)

    def as_mapping(self) -> dict[str, Callable]:
        """``{name: callable}``, for the :mod:`neu_mark.explore` functions."""
        return {r.name: r.values for r in self.rules}

    def require(self, frame: pd.DataFrame) -> None:
        """Check every rule's ``needs`` against a frame, naming the rule that is unmet."""
        missing = {r.name: [c for c in r.needs if c not in frame.columns]
                   for r in self.rules}
        missing = {k: v for k, v in missing.items() if v}
        if missing:
            detail = "; ".join(f"{k} needs {', '.join(v)}" for k, v in missing.items())
            raise KeyError(
                f"the frame is missing columns these rules declare: {detail}. It has "
                f"{', '.join(map(str, frame.columns))}.")

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Every rule over every row: one column per rule, values as tuples."""
        self.require(frame)
        return explore.apply_rules(frame, self.as_mapping())

    def describe(self) -> pd.DataFrame:
        """What every rule declares, one row per rule in evaluation order. (:meth:`coverage`
        reports what they did to a frame.)"""
        return pd.DataFrame([{
            "rule": r.name,
            "kind": "drop" if r.drop else "tag" if r.tag else "property",
            "prefix": r.prefix,
            "multi": r.multi,
            "needs": ", ".join(r.needs),
            "consumes": ("derived" if r._consumes is None else
                         "callable" if callable(r._consumes) else
                         ", ".join(str(c) for c in r._consumes)),
            "description": r.description,
        } for r in self.rules])

    def coverage(self, frame: pd.DataFrame, *, top: int = 3) -> pd.DataFrame:
        """Per-rule coverage, with each rule's declared facts alongside."""
        self.require(frame)
        out = explore.coverage(frame, self.as_mapping(), top=top)
        meta = pd.DataFrame({"rule": self.names,
                             "multi": [r.multi for r in self.rules],
                             "tag": [r.tag for r in self.rules],
                             "drop": [r.drop for r in self.rules],
                             "description": [r.description for r in self.rules]})
        return out.merge(meta, on="rule", how="left")

    def unparsed(self, frame: pd.DataFrame, name: str, *,
                 column: str = DEFAULT_COLUMN) -> pd.DataFrame:
        """The strings one rule did not fire on, ranked by bodies. The iteration loop."""
        self.require(frame)
        return explore.unparsed(frame, self.as_mapping(), name, column=column)

    def consumed(self, _value: str | None = None, /, **fields) -> dict[str, tuple]:
        """``{rule: tokens}`` for every rule that accounted for part of the string."""
        row = _row({DEFAULT_COLUMN: _value, **fields} if _value is not None else fields,
                   needs={c for r in self.rules for c in r.needs})
        return {r.name: got for r in self.rules if (got := r.consumed(row))}

    def remainder(self, _value: str | None = None, /, **fields) -> str | None:
        """The string with every token any rule accounted for removed, so a cell-type
        parse need not repeat the rules' vocabularies.

        The removed tokens are one union, so the result does not depend on rule order. The
        result is a residue, not a cell type: often a bare neuropil name for an unidentified
        body.
        """
        fields = {DEFAULT_COLUMN: _value, **fields} if _value is not None else fields
        row = _row(fields, needs={c for r in self.rules for c in r.needs})
        drop = {t.lower() for r in self.rules for t in r.consumed(row)}
        kept = [t for t in explore.split_tokens(row[DEFAULT_COLUMN])
                if t.lower() not in drop]
        return "_".join(kept) or None

    def explain(self, _value: str | None = None, /, **fields) -> dict:
        """Every rule's output for one synthetic row: a scalar for single-valued rules, a
        tuple for multi-valued ones."""
        if _value is not None:
            fields = {DEFAULT_COLUMN: _value, **fields}
        needs = {c for r in self.rules for c in r.needs}
        row = _row(fields, needs=needs)
        out = {}
        for r in self.rules:
            got = r.values(row)
            out[r.name] = got if r.multi else (got[0] if got else None)
        return out


#: Largest rules source embedded in a provenance record (a rules module is a few KB).
MAX_EMBED_BYTES = 256 * 1024


def provenance(ruleset: RuleSet, *, embed: bool = True,
               max_embed_bytes: int = MAX_EMBED_BYTES) -> dict:
    """A record of the rules that produced an output.

    Each rules source file is embedded verbatim with its sha256, so the record does not
    depend on the file still existing; git commit, remote and dirty flag are added if the
    file is in a work tree. Files are found from each rule's ``__code__.co_filename``, so
    helpers imported from sibling files are included; files inside ``neu_mark`` are not.
    """
    import hashlib

    package = Path(__file__).resolve().parent
    paths: list[Path] = []
    if ruleset.source and Path(ruleset.source).is_file():
        paths.append(Path(ruleset.source).resolve())
    for r in ruleset.rules:
        code = getattr(r.func, "__code__", None)
        name = getattr(code, "co_filename", None)
        if not name:
            continue
        candidate = Path(name)
        if not candidate.is_file():
            continue
        candidate = candidate.resolve()
        if package in candidate.parents or candidate == package:
            continue
        if candidate not in paths:
            paths.append(candidate)

    files = []
    for path in paths:
        try:
            data = path.read_bytes()
        except OSError as exc:                                    # pragma: no cover
            files.append({"path": str(path), "error": str(exc)})
            continue
        entry = {"path": str(path), "bytes": len(data),
                 "sha256": hashlib.sha256(data).hexdigest()}
        if embed and len(data) <= max_embed_bytes:
            entry["text"] = data.decode("utf-8", errors="replace")
        elif embed:
            entry["text_omitted"] = (
                f"{len(data)} bytes exceeds max_embed_bytes={max_embed_bytes}")
        git = _git_details(path)
        if git:
            entry["git"] = git
        files.append(entry)

    return {
        "source": ruleset.source,
        "names": ruleset.names,
        "keep": list(ruleset.keep),
        "rules": [{"name": r.name, "description": r.description, "needs": list(r.needs),
                   "multi": r.multi, "tag": r.tag, "drop": r.drop, "prefix": r.prefix}
                  for r in ruleset.rules],
        "files": files,
    }


def _git_details(path: Path) -> dict | None:
    """Commit, remote and dirty flag for a file in a git work tree; ``None`` if git or the
    repository is unavailable."""
    import subprocess

    def run(*args: str) -> str | None:
        try:
            done = subprocess.run(("git", "-C", str(path.parent), *args),
                                  capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):             # pragma: no cover
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    if not commit:
        return None
    out = {"commit": commit}
    for key, args in (("remote", ("config", "--get", "remote.origin.url")),
                      ("path", ("ls-files", "--full-name", "--", str(path)))):
        value = run(*args)
        if value:
            out[key] = value
    status = run("status", "--porcelain", "--", str(path))
    # Non-empty for a modified or untracked file: the commit does not describe what ran.
    out["dirty"] = bool(status)
    return out


def from_module(path: str | Path, *, variable: str = "RULES") -> RuleSet:
    """Load a rules module by file path.

    Uses the module's ``RULES`` if it defines one; otherwise every ``@rule`` at module
    level, in declaration order.
    """
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"no rules module at {path}")
    spec = importlib.util.spec_from_file_location(f"neu_mark_rules_{path.stem}", path)
    if spec is None or spec.loader is None:                       # pragma: no cover
        raise ImportError(f"cannot load a module from {path}")
    module = importlib.util.module_from_spec(spec)
    # Registered so dataclasses and pickle work inside it; removed if loading fails.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise

    declared = getattr(module, variable, None)
    if isinstance(declared, RuleSet):
        declared.source = str(path)
        return declared
    if declared is not None:
        return RuleSet(declared, keep=getattr(module, "KEEP", None), source=str(path))
    found = [v for v in vars(module).values() if isinstance(v, Rule)]
    if not found:
        raise ValueError(
            f"{path} defines no rules: expected a `{variable}` list/RuleSet, or functions "
            f"decorated with @rule at module level.")
    return RuleSet(found, keep=getattr(module, "KEEP", None), source=str(path))
