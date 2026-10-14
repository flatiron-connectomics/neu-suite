"""What the shaders' discard conditions do, including for NaN, without a GLSL runtime.

Each shader's ``if (...) discard;`` condition is evaluated in Python, whose float
comparisons follow the same IEEE rule as GLSL: any comparison with NaN is false.
"""

import re

import pytest

from neu_glance.shaders import COLORMAP_SHADER, PROBABILITY_SHADER, SYNAPSE_SHADER

NAN = float("nan")


def discarded(shader: str, **values: float) -> bool:
    """Whether ``shader`` discards, with its calls and controls bound by ``values``
    (``prop_conf_pre=0.3, min_conf=0.5``; ``v=NAN, threshold=0.5``)."""
    conditions = re.findall(r"if \((.+?)\) discard;", shader)
    assert conditions, "the shader has no discard"
    expr = " or ".join(f"({c})" for c in conditions)
    expr = re.sub(r"(\w+)\(\)", r"\1", expr)               # prop_conf_pre() -> prop_conf_pre
    expr = expr.replace("||", " or ").replace("&&", " and ")
    expr = re.sub(r"!(?!=)", " not ", expr)
    return bool(eval(expr, {"__builtins__": {}}, values))


@pytest.mark.parametrize("min_conf", [0.0, 0.5, 1.0])
def test_a_synapse_with_no_confidence_is_shown_at_every_threshold(min_conf):
    """NaN confidence usually marks a synapse a proofreader added by hand."""
    assert not discarded(SYNAPSE_SHADER, prop_conf_pre=NAN, prop_conf_post=NAN,
                         min_conf=min_conf)
    assert not discarded(SYNAPSE_SHADER, prop_conf_pre=NAN, prop_conf_post=1.0,
                         min_conf=min_conf)


def test_the_confidence_threshold_hides_low_confidence_synapses():
    assert discarded(SYNAPSE_SHADER, prop_conf_pre=0.3, prop_conf_post=0.9, min_conf=0.5)
    assert discarded(SYNAPSE_SHADER, prop_conf_pre=0.9, prop_conf_post=0.3, min_conf=0.5)
    assert not discarded(SYNAPSE_SHADER, prop_conf_pre=0.6, prop_conf_post=0.9,
                         min_conf=0.5)


@pytest.mark.parametrize("shader", [PROBABILITY_SHADER, COLORMAP_SHADER],
                         ids=["probability", "colormap"])
def test_a_voxel_with_no_prediction_is_not_drawn_and_the_threshold_works(shader):
    for threshold in (0.0, 0.5):
        assert discarded(shader, v=NAN, threshold=threshold)
    assert discarded(shader, v=0.3, threshold=0.5)
    assert not discarded(shader, v=0.7, threshold=0.5)
    assert not discarded(shader, v=0.0, threshold=0.0)
