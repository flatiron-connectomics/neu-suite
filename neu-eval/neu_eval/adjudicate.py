"""Recording that a disagreement was the reference's fault, and rescoring without it.

A manual annotation can be wrong, and scoring against it there charges the segmentation
for being right. So a disagreement table round-trips: it goes out with an empty
``verdict`` column, a reviewer fills it in, and it comes back as the basis for a second,
adjudicated score.

Report both scores, and how much was excluded: an adjudicated score alone cannot be
checked. :func:`apply` therefore moves excluded voxels into ``n_ignored`` rather than
dropping them, and ``frac_scored`` appears in every summary. Verdicts say ``correct``,
not ``right``, so they cannot be confused with left/right.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np

from .overlap import Contingency

#: - ``<reference>_correct``: the annotation was right; the segmentation erred.
#: - ``<other>_correct``: the segmentation was right; the annotation is a defect.
#: - ``ambiguous``: reviewed but unclear. Not excluded by default, since "I could not tell"
#:   is no evidence the reference was wrong.
#: - ``skip``: considered and set aside; a blank means not reviewed.
VERDICTS = ("gt_correct", "seg_correct", "ambiguous", "skip")

#: The verdict column's name in the round-tripped table.
VERDICT_COLUMN = "verdict"

def default_exclude(labels: Sequence[str] = ("gt", "seg")) -> tuple[str, ...]:
    """The verdicts whose voxels leave the denominator: only "the reference is wrong here",
    spelled for the given side names (``--labels v1,v2`` gives ``v2_correct``)."""
    return (f"{labels[1]}_correct",)


def verdict_names(labels: Sequence[str] = ("gt", "seg")) -> tuple[str, ...]:
    """The verdict vocabulary for a pair of side names; :data:`VERDICTS` is the
    ``("gt", "seg")`` case."""
    return (f"{labels[0]}_correct", f"{labels[1]}_correct", "ambiguous", "skip")


def check_verdicts(verdicts: Mapping[str, str], *,
                   labels: Sequence[str] = ("gt", "seg")) -> None:
    """Raise on any verdict outside the vocabulary (blank is allowed). Otherwise a typo
    in a hand-edited column would silently count as "not excluded"."""
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
    is in ``exclude`` (by default, "the segmentation was right") are dropped and their voxels
    added to ``n_ignored``, so ``n_scored + n_ignored`` still totals the region.

    This declines to score where the reference is wrong; it does not substitute the
    segmentation's answer as a corrected reference, which would need the voxels. Unknown
    ``pair_key`` values are ignored, since a verdict file can outlive the table it came from
    (a different ``min_frac`` changes which pairs appear).
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
