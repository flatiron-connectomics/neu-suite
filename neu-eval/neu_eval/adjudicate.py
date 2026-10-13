"""Recording that a disagreement was the *reference's* fault, and rescoring without it.

The premise of this module is that ground truth is not ground truth. A manual annotation
and an algorithm disagree for two different reasons, and only one of them is a segmentation
error; scoring against an annotation known to be wrong in specific places charges the
algorithm for being right. So a disagreement table round-trips: it goes out with an empty
``verdict`` column, a human fills it in while looking at the voxels, and it comes back as
the basis for a second score alongside the first.

**Both scores are reported, always.** An adjudicated score on its own is unfalsifiable — it
is the number you get after deciding which disagreements not to count — so it only means
something next to the raw one and next to how much was excluded to get it. That is why
:func:`apply` returns a table whose ``n_ignored`` has grown rather than one that pretends
the excluded voxels were never there, and why ``frac_scored`` is in every summary.

**The vocabulary is ``correct``, not ``right``.** ``seg_right`` reads as a tuple position
everywhere else in this codebase, and a verdict that collides with left/right is the kind of
ambiguity that survives into a column name and then into a figure caption.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np

from .overlap import Contingency

#: ``<reference>_correct`` — the annotation was right, the segmentation got it wrong. The
#: ordinary case, and the one a metric is designed to count.
#:
#: ``<other>_correct`` — the segmentation was right and the annotation is a defect. Counting
#: it against the segmentation is the error this module exists to remove.
#:
#: ``ambiguous`` — looked at, genuinely unclear. Deliberately **not** excluded by default:
#: a verdict of "I could not tell" is not evidence the reference was wrong, and dropping it
#: is how an adjudicated score drifts upward one judgement call at a time.
#:
#: ``skip`` — not looked at yet. Distinct from a blank so a partly-reviewed table can say
#: which rows were considered and dismissed.
VERDICTS = ("gt_correct", "seg_correct", "ambiguous", "skip")

#: The verdict column's name in the round-tripped table.
VERDICT_COLUMN = "verdict"

def default_exclude(labels: Sequence[str] = ("gt", "seg")) -> tuple[str, ...]:
    """The verdicts whose voxels leave the denominator: only "the reference is wrong here".

    **Derived from ``labels``, never a constant.** It was a constant — ``("seg_correct",)``
    — and that silently did nothing whenever the sides were named anything else: with
    ``--labels v1,v2`` the reviewer writes ``v2_correct``, no verdict matched, and the
    adjudicated score came out exactly equal to the raw one with nothing to say why. Caught
    by running the real round trip, not by a test, because every test had used the default
    names.
    """
    return (f"{labels[1]}_correct",)


def verdict_names(labels: Sequence[str] = ("gt", "seg")) -> tuple[str, ...]:
    """The verdict vocabulary for a given pair of side names.

    ``VERDICTS`` is spelled for the usual ``("gt", "seg")`` call; when the sides are named
    something else the verdicts should follow, so a reviewer is not asked to write ``seg``
    about a column called ``v2``.
    """
    return (f"{labels[0]}_correct", f"{labels[1]}_correct", "ambiguous", "skip")


def check_verdicts(verdicts: Mapping[str, str], *,
                   labels: Sequence[str] = ("gt", "seg")) -> None:
    """Refuse a verdict nobody defined, naming what was allowed.

    A typo in a hand-edited column would otherwise be silently treated as "not excluded",
    which is the direction that flatters — the reviewer said the reference was wrong, the
    tool scored it against the segmentation anyway, and nothing said so.
    """
    allowed = set(verdict_names(labels)) | {""}
    bad = {key: value for key, value in verdicts.items()
           if str(value).strip() not in allowed}
    if bad:
        shown = ", ".join(f"{k}={v!r}" for k, v in list(bad.items())[:5])
        raise ValueError(
            f"{len(bad)} unrecognised verdict(s): {shown}. Allowed: "
            f"{', '.join(sorted(allowed - {''}))}, or blank for un-reviewed.")


def apply(c: Contingency, verdicts: Mapping[str, str], *,
          exclude: Iterable[str] | None = None,
          labels: Sequence[str] | None = None) -> Contingency:
    """``c`` with the adjudicated pairs' voxels moved out of the denominator.

    ``verdicts`` maps ``pair_key`` (``"<a_id>:<b_id>"``) to a verdict. Pairs whose verdict
    is in ``exclude`` — by default just "the segmentation was right", i.e. the reference is
    defective there — are dropped from the table; their voxels are added to ``n_ignored``,
    so ``n_scored + n_ignored`` still totals the region and ``frac_scored`` reports the
    price of the correction.

    **Excluding is not the same as accepting.** This declines to score where the reference
    is known bad; it does not fold the segmentation's answer in as a corrected reference and
    rescore against that. The latter needs the voxels, not the table, and would produce a
    new reference volume — a worthwhile thing (the verdict table has the same shape as
    ``neu-vol relabel``'s ``--map``, so a corrected annotation is derivable from a review
    pass) but a different operation, deliberately not smuggled in here.

    Unknown ``pair_key`` values are ignored rather than refused: a verdict file outlives the
    table it came from, and re-running with a different ``min_frac`` legitimately changes
    which pairs appear.
    """
    labels = tuple(labels) if labels is not None else c.labels
    exclude = tuple(default_exclude(labels) if exclude is None else exclude)
    check_verdicts(verdicts, labels=labels)

    drop_keys = {key for key, value in verdicts.items()
                 if str(value).strip() in exclude}
    if not drop_keys:
        return c

    keys = np.array([f"{int(a)}:{int(b)}" for a, b in zip(c.a_ids, c.b_ids)], dtype=object)
    keep = np.array([key not in drop_keys for key in keys], dtype=bool)
    return c.keep_pairs(keep)


def summarize(verdicts: Mapping[str, str], *,
              labels: Sequence[str] = ("gt", "seg")) -> dict[str, int]:
    """How many of each verdict, including how many rows are still unreviewed."""
    tally = {name: 0 for name in verdict_names(labels)}
    tally["unreviewed"] = 0
    for value in verdicts.values():
        text = str(value).strip()
        if text == "":
            tally["unreviewed"] += 1
        elif text in tally:
            tally[text] += 1
    return tally
