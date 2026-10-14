"""Colour normalisation and assignment, with no plotting library in sight."""

import pytest

from neu_draw.colors import QUALITATIVE, assign_colors, to_rgba


@pytest.mark.parametrize("value, expected", [
    ("r", (1.0, 0.0, 0.0, 1.0)),                    # matplotlib shorthands
    ("b", (0.0, 0.0, 1.0, 1.0)),
    ("#f00", (1.0, 0.0, 0.0, 1.0)),                 # hex in all three lengths
    ("#ff0000", (1.0, 0.0, 0.0, 1.0)),
    ("#00ff0080", (0.0, 1.0, 0.0, 128 / 255)),
    ((0.2, 0.4, 0.6), (0.2, 0.4, 0.6, 1.0)),        # an RGB triple gains opaque alpha
])
def test_accepted_colour_forms(value, expected):
    assert to_rgba(value) == pytest.approx(expected)


def test_alpha_overrides_whatever_the_colour_carried():
    assert to_rgba("#ff0000ff", alpha=0.25) == (1.0, 0.0, 0.0, 0.25)


@pytest.mark.parametrize("value, match", [
    ((255, 0, 0), r"\[0, 1\]"),                  # out-of-range components
    ("chartreuse-ish", "unknown colour"),
])
def test_invalid_colours_are_rejected(value, match):
    with pytest.raises(ValueError, match=match):
        to_rgba(value)


# --------------------------------------------------------------------------- #
# assignment
# --------------------------------------------------------------------------- #

def test_names_cycle_the_palette_in_order():
    out = assign_colors(["a", "b", "c"])
    assert [out[k] for k in "abc"] == list(QUALITATIVE[:3])


def test_the_palette_wraps_when_there_are_more_names_than_colours():
    names = [str(i) for i in range(len(QUALITATIVE) + 2)]
    out = assign_colors(names)
    assert out[names[0]] == out[names[len(QUALITATIVE)]]


def test_fixing_one_colour_does_not_shift_the_others():
    """Pinning one body's colour must not renumber everyone else's."""
    free = assign_colors(["a", "b", "c"])
    pinned = assign_colors(["a", "b", "c"], explicit={"b": "r"})
    assert pinned["b"] == (1.0, 0.0, 0.0, 1.0)
    assert pinned["a"] == free["a"] and pinned["c"] == free["b"]


RED, BLUE = (1.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0)


@pytest.mark.parametrize("explicit, expected", [
    ("k", [(0.0, 0.0, 0.0, 1.0)] * 3),                  # one colour for everything
    ((0.1, 0.2, 0.3), [(0.1, 0.2, 0.3, 1.0)] * 3),      # an RGB triple is one colour
    (["r", "b"], [RED, BLUE, RED]),                     # a list is the palette
])
def test_explicit_colour_forms(explicit, expected):
    out = assign_colors(["a", "b", "c"], explicit=explicit)
    assert [out[k] for k in "abc"] == expected


def test_duplicate_names_collapse_to_one_entry():
    assert len(assign_colors(["a", "a", "b"])) == 2
