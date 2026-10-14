"""The built-in facet rules, which are ordinary :mod:`neu_mark.rules` rules.

Nothing here is privileged: these use the same ``@rule`` decorator as a dataset's own
module, and any of them can be replaced by name, so this file is also the worked example
of a rules module. There is no cell-type or column parse, because which tokens are columns
and which are part of a cell type depends on the dataset; that belongs in a rules module::

    # wasp_rules.py
    from neu_mark import rule

    @rule(multi=True)
    def column(r):
        \"\"\"Optic lobe column, e.g. C2 or A1.\"\"\"
        return re.findall(r"_([A-Z]\\d+)(?=$|[_(])", r["instance"] or "")

The vocabularies were chosen on one dataset (Megaphragma), so another dataset may need
different ones.
"""

from __future__ import annotations

import re

from .explore import normalize, split_tokens
from .rules import Rule, RuleSet, rule

#: Instance-string tokens meaning "not real data"; such a body is dropped from the output.
NOISE = ("irrelevant", "block", "chunk", "unknown")

SIDE = re.compile(r"\((L|R)\)")


@rule(drop=True)
def noise(r):
    """not real data: an `irrelevant`, `block`, `chunk` or `unknown` body"""
    lowered = [t.lower() for t in split_tokens(r["instance"])]
    return next((t for t in NOISE if t in lowered), None)


@rule
def side(r):
    """hemisphere, from the (L)/(R) in the name"""
    m = SIDE.search(normalize(r["instance"]) or "")
    return m.group(1) if m else None


@rule(needs=["type"])
def group(r):
    """cell type, from the curated `type` field"""
    return normalize(r["type"])


#: The two rules below emit a different word than the token they match (`NCL` becomes
#: `nucleated`, `CV` becomes `cervical`), so they declare `consumes` for
#: `RuleSet.remainder`.
NUCLEUS_TOKENS = ("ncl", "nucleus")
CERVICAL_TOKENS = ("cv",)


@rule(prefix="", consumes=NUCLEUS_TOKENS)
def nucleated(r):
    """cell body (nucleus) is in the volume"""
    lowered = {t.lower() for t in split_tokens(r["instance"])}
    return "nucleated" if lowered & set(NUCLEUS_TOKENS) else None


@rule(prefix="", consumes=CERVICAL_TOKENS)
def cervical(r):
    """passes through the cervical connective"""
    lowered = {t.lower() for t in split_tokens(r["instance"])}
    return "cervical" if lowered & set(CERVICAL_TOKENS) else None


@rule(prefix="")
def glia(r):
    """glia rather than a neuron"""
    return "glia" if "glia" in {t.lower() for t in split_tokens(r["instance"])} else None


@rule(prefix="")
def fragment(r):
    """a fragment, not a completely traced cell"""
    lowered = {t.lower() for t in split_tokens(r["instance"])}
    return "fragment" if "fragment" in lowered else None


@rule(prefix="")
def truncated(r):
    """traced but cut off by the volume boundary"""
    lowered = {t.lower() for t in split_tokens(r["instance"])}
    return "truncated" if "truncated" in lowered else None


#: What ``--drop-glia`` puts in place of the ``glia`` rule above. Glia are kept by default.
DROP_GLIA = Rule(lambda r: glia.func(r), name="glia", drop=True,
                 description="glia rather than a neuron (excluded)")

#: Evaluated in declaration order, drops first. Override by name; see :meth:`RuleSet.override`.
BUILTIN = RuleSet([noise, side, group, nucleated, cervical, glia, fragment, truncated],
                  source="neu_mark.segprops_rules.BUILTIN")
