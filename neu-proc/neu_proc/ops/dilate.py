"""Label dilation, three ways.

``edt`` has a cupyx implementation and stays on the device; the fastmorph methods have none
and are moved to the host with a warning (:func:`~neu_proc.ops.backend.host_only`).

**``edt`` is UNBOUNDED unless you bound it.** Every background voxel takes the label of the
nearest foreground voxel, so by default the labels grow until they meet and no background is
left — a Voronoi partition of the array rather than a dilation by a radius. Measured on a
16^3 array with two 2^3 seeds: unbounded ``edt`` fills all 4096 voxels, fastmorph 128,
spherical 64. Useful, and not what "dilate" usually implies, so ``max_distance=`` is there to
ask for the bounded version.

``max_distance``'s units follow ``sampling``. With ``sampling=(z, y, x)`` in nm the bound is
in nm; without it, both are counted in voxels — and on an anisotropic volume voxel-counted
growth is lopsided in tissue while looking symmetric in the array. Measured on a 32^3 array
with one 2^3 seed: ``max_distance=5`` (voxels) grows it to 792 voxels, while
``sampling=(40, 8, 8), max_distance=80`` (nm) gives 1280 — the same request, different
answers, which is the whole reason the `Op` layer above declares lengths in nm.

The three methods take **disjoint** parameters — ``edt`` has ``sampling``, ``spherical`` has
``radius``/``anisotropy``, ``fastmorph`` has ``iterations``/``mode`` — so they are less
interchangeable than one shared entry point suggests. Putting a parameter on the wrong method
is refused by name, with a pointer to the one that takes it.
"""

from fastmorph import dilate as fm_dilate, spherical_dilate

from .backend import host_only, like, ndimage_for, stage

#: Method name -> (spellings, the implementation it calls). A table rather than a chain of
#: `startswith` tests so an unknown name can be refused *with the list* — the house pattern
#: (`profiles.get_profile`, `pyramid.REDUCERS`). Before this, a typo fell through every branch
#: and raised `UnboundLocalError: cannot access local variable 'dilated'`.
METHODS = {
    "edt": (("edt", "euclidean"), None),          # resolved per array: scipy or cupyx
    "fastmorph": (("fastmorph", "fm"), fm_dilate),
    "spherical": (("spherical", "spher", "sphere"), spherical_dilate),
}

#: Keyword arguments this function sets itself, so a caller passing one would fight it.
_RESERVED = frozenset({"return_distances", "return_indices", "distances", "indices",
                       "input", "labels"})


def _resolve(method: str) -> str:
    wanted = str(method).lower()
    for name, (spellings, _) in METHODS.items():
        if any(wanted.startswith(s) for s in spellings):
            return name
    raise ValueError(
        f"unknown dilation method {method!r}; known: "
        + ", ".join(f"{n} ({'/'.join(s)})" for n, (s, _) in METHODS.items()))


def accepted(method: str, target=None) -> tuple[str, ...]:
    """The keyword arguments ``method`` accepts, read from the implementation itself.

    Introspected rather than listed, so it cannot drift from the library — all three targets
    turned out to have real signatures, including the C extensions.
    """
    import inspect

    if target is None:
        from scipy import ndimage

        target = METHODS[method][1] or ndimage.distance_transform_edt
    try:
        params = inspect.signature(target).parameters
    except (TypeError, ValueError):                                  # pragma: no cover
        return ()
    return tuple(n for n, p in params.items()
                 if n not in _RESERVED
                 and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
                 and p.default is not p.empty)


def _check_kwargs(method: str, kwargs: dict) -> None:
    """Refuse a keyword the chosen method does not take, saying which one does.

    The underlying libraries already refuse it — loudly, with a TypeError — but the message
    names *their* function and not this one, so `dilate(a, method="edt", radius=3)` reads as
    a bug in scipy. The three methods take genuinely disjoint parameters (edt has
    `sampling=`, spherical has `radius=`/`anisotropy=`, fastmorph has `iterations=`/`mode=`),
    which makes putting one on the wrong method the easy mistake.
    """
    allowed = set(accepted(method))
    for name in kwargs:
        if name in allowed:
            continue
        elsewhere = [m for m in METHODS if name in accepted(m)]
        hint = (f"; it belongs to method={elsewhere[0]!r}" if elsewhere else "")
        raise TypeError(
            f"method={method!r} does not take {name}= (it takes "
            f"{', '.join(sorted(allowed)) or 'no options'}){hint}")


def dilate(arr, method: str = "edt", *, max_distance: float | None = None,
           gpu: bool | None = None, **kwargs):
    """Grow the labels in ``arr``.

    ``max_distance`` bounds the ``edt`` method, which is otherwise **unbounded** — see the
    module docstring. Its units follow ``sampling``: with ``sampling=(40, 8, 8)`` a
    ``max_distance`` of 80 means 80 nm, and without it the bound is counted in voxels. That
    is worth being deliberate about, since the same number means different things.

    ``kwargs`` reach the underlying implementation and are checked against it first, so a
    parameter put on the wrong method says so rather than surfacing as a TypeError from
    scipy or fastmorph.

    **A host array is sent to the device and back for the ``edt`` method**, because measured
    that is faster than staying on the host even counting both transfers: 192^3 takes 811 ms
    on scipy against 16 ms with the round trip, and the margin *grows* with size since the
    EDT is superlinear while a transfer is linear. What you hand in is what you get back, so this returns numpy
    for a numpy input; nothing is left on a GPU you did not ask for. ``gpu=False`` opts out
    (as does ``NEU_PROC_GPU=0``), and for a *chain* move once yourself with
    ``backend.to_device`` so no step transfers at all.
    """
    resolved = _resolve(method)
    if max_distance is not None and resolved != "edt":
        raise TypeError(
            f"max_distance= bounds the 'edt' method; method={method!r} is already bounded — "
            f"use its own radius/iterations parameter ({', '.join(accepted(resolved))})")
    _check_kwargs(resolved, kwargs)

    if resolved == "edt":
        # Enough arithmetic to earn the round trip; `like` puts the answer back where the
        # input lived, so the move is invisible to the caller.
        original, arr = arr, stage(arr, prefer_gpu=True if gpu is None else gpu)
        ndi, arr = ndimage_for(arr, "distance_transform_edt")
        background = (arr == 0)
        # The nearest foreground voxel's INDEX rather than its distance, so the labels come
        # along for free and no second pass is needed. No copy first: fancy-indexing builds a
        # new array anyway, and copying a large one on the device is not free.
        want_distance = max_distance is not None
        result = ndi.distance_transform_edt(
            background,
            return_distances=want_distance,
            return_indices=True,
            **kwargs,
        )
        distance, ixs = result if want_distance else (None, result)
        dilated = arr[tuple(ixs)]
        if want_distance:
            # Only the far BACKGROUND is cleared: a foreground voxel indexes itself, so its
            # distance is 0 and it survives any bound.
            dilated[distance > max_distance] = 0
        return like(dilated, original)

    # No `.copy()`: measured, neither fastmorph function mutates its input — only
    # `spherical_dilate(in_place=True)` does, which is the caller asking for it.
    # fastmorph has no device implementation, so there is nothing to stage toward — and the
    # result comes back where the input was, same contract as the edt path.
    target = METHODS[resolved][1]
    return like(target(host_only(arr, f"fastmorph.{target.__name__}"), **kwargs), arr)
