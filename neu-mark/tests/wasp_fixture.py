"""A small rules module in the Megaphragma naming, for the tests: what neu-mark used to build
in. Loaded with ``rules.from_module`` where a test needs the module path, imported directly
otherwise."""

import re

from neu_mark import rule
from neu_mark.explore import normalize, split_tokens

LABEL = "instance"

NOISE = ("irrelevant", "block", "chunk", "unknown")
SIDE = re.compile(r"\((L|R)\)")


def _tokens(r):
    return {t.lower() for t in split_tokens(r["instance"])}


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


@rule(name="type", needs=["type"])
def cell_type(r):
    """cell type, from the curated `type` field"""
    return normalize(r["type"])


@rule(prefix="", consumes=("ncl", "nucleus"))
def nucleated(r):
    """cell body (nucleus) is in the volume"""
    return "nucleated" if _tokens(r) & {"ncl", "nucleus"} else None


@rule(prefix="", consumes=("cv",))
def cervical(r):
    """passes through the cervical connective"""
    return "cervical" if "cv" in _tokens(r) else None


@rule(prefix="")
def glia(r):
    """glia rather than a neuron"""
    return "glia" if "glia" in _tokens(r) else None


@rule(prefix="")
def fragment(r):
    """a fragment, not a completely traced cell"""
    return "fragment" if "fragment" in _tokens(r) else None


@rule(prefix="")
def truncated(r):
    """traced but cut off by the volume boundary"""
    return "truncated" if "truncated" in _tokens(r) else None


RULES = [noise, side, cell_type, nucleated, cervical, glia, fragment, truncated]
