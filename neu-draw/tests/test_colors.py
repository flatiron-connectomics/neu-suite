"""Colour normalisation and assignment, with no plotting library in sight."""

import pytest

from neu_draw.colors import QUALITATIVE, assign_colors, to_rgba


def test_the_shorthands_the_old_call_sites_used_still_work():
    """`presyn_color='r'` / `postsyn_color='b'` were the real arguments in use."""
    assert to_rgba("r") == (1.0, 0.0, 0.0, 1.0)
    assert to_rgba("b") == (0.0, 0.0, 1.0, 1.0)


@pytest.mark.parametrize("text, expected", [
    ("#f00", (1.0, 0.0, 0.0, 1.0)),
    ("#ff0000", (1.0, 0.0, 0.0, 1.0)),
    ("#00ff0080", (0.0, 1.0, 0.0, 128 / 255)),
])
def test_hex_in_all_three_lengths(text, expected):
    assert to_rgba(text) == pytest.approx(expected)


def test_a_three_sequence_gains_an_opaque_alpha():
    assert to_rgba((0.2, 0.4, 0.6)) == (0.2, 0.4, 0.6, 1.0)


def test_alpha_overrides_whatever_the_colour_carried():
    assert to_rgba("#ff0000ff", alpha=0.25) == (1.0, 0.0, 0.0, 0.25)


def test_out_of_range_components_are_rejected():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        to_rgba((255, 0, 0))


def test_an_unknown_name_names_the_alternatives():
    with pytest.raises(ValueError, match="unknown colour"):
        to_rgba("chartreuse-ish")


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
    """The property that makes a figure reproducible while you iterate: pinning one
    body's colour must not renumber everyone else's."""
    free = assign_colors(["a", "b", "c"])
    pinned = assign_colors(["a", "b", "c"], explicit={"b": "r"})
    assert pinned["b"] == (1.0, 0.0, 0.0, 1.0)
    assert pinned["a"] == free["a"] and pinned["c"] == free["b"]


def test_a_single_colour_applies_to_everything():
    out = assign_colors(["a", "b"], explicit="k")
    assert set(out.values()) == {(0.0, 0.0, 0.0, 1.0)}


def test_a_bare_rgb_sequence_is_one_colour_not_a_palette_of_numbers():
    """The ambiguity the predecessor resolved with `np.asarray(...).ndim` checks."""
    out = assign_colors(["a", "b"], explicit=(0.1, 0.2, 0.3))
    assert set(out.values()) == {(0.1, 0.2, 0.3, 1.0)}


def test_a_list_of_colours_is_used_as_the_palette():
    out = assign_colors(["a", "b", "c"], explicit=["r", "b"])
    assert [out[k] for k in "abc"] == [(1.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0),
                                       (1.0, 0.0, 0.0, 1.0)]


def test_duplicate_names_collapse_to_one_entry():
    assert len(assign_colors(["a", "a", "b"])) == 2
