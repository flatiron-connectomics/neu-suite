"""Label dilation, three ways: ``edt``, ``fastmorph`` and ``spherical``.

``edt`` runs on the GPU when one is available; the fastmorph methods have no GPU version
and move a GPU array to the CPU with a warning (:func:`~neu_proc.ops.backend.cpu_only`).

**``edt`` is unbounded unless given ``max_distance``.** Every background voxel takes the
label of the nearest foreground voxel, so labels grow until they meet and no background is
left: a Voronoi partition, not a dilation by a radius. On a 16^3 array with two 2^3 seeds,
unbounded ``edt`` fills all 4096 voxels, fastmorph 128, spherical 64.

``max_distance`` has the units of ``sampling``: in nm with ``sampling=(z, y, x)`` in nm,
otherwise in voxels, which on an anisotropic volume grows labels lopsidedly in tissue.

The three methods take disjoint parameters (``edt``: ``sampling``; ``spherical``:
``radius``, ``anisotropy``; ``fastmorph``: ``iterations``, ``mode``). A parameter given to
the wrong method is refused with the name of the method that takes it.
"""

from fastmorph import dilate as fm_dilate, spherical_dilate

from .backend import cpu_only, like, ndimage_for, stage

#: Method name -> (accepted spellings, implementation), so an unknown name is refused with
#: the list of known ones.
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
    """The keyword arguments ``method`` accepts, read from the implementation's signature."""
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
    """Refuse a keyword the chosen method does not take, naming the method that does,
    rather than letting scipy or fastmorph raise about their own function."""
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

    ``max_distance`` bounds the ``edt`` method, which is otherwise unbounded; it has the
    units of ``sampling`` (with ``sampling=(40, 8, 8)``, 80 means 80 nm), or voxels without
    it. Other ``kwargs`` go to the implementation, after a check that it takes them.

    With ``edt``, a CPU array is computed on the GPU when one is available, which is faster
    even counting both transfers (see :mod:`~neu_proc.ops.backend`); the result is returned
    as numpy. ``gpu=False`` (or ``NEU_PROC_GPU=0``) keeps it on the CPU.
    """
    resolved = _resolve(method)
    if max_distance is not None and resolved != "edt":
        raise TypeError(
            f"max_distance= bounds the 'edt' method; method={method!r} is already bounded — "
            f"use its own radius/iterations parameter ({', '.join(accepted(resolved))})")
    _check_kwargs(resolved, kwargs)

    if resolved == "edt":
        original, arr = arr, stage(arr, prefer_gpu=True if gpu is None else gpu)
        ndi, arr = ndimage_for(arr, "distance_transform_edt")
        background = (arr == 0)
        # The nearest foreground voxel's index gives each voxel its label in one pass.
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
            # Foreground voxels have distance 0, so only far background is cleared.
            dilated[distance > max_distance] = 0
        return like(dilated, original)

    # fastmorph does not mutate its input (unless given in_place=True) and has no GPU path.
    target = METHODS[resolved][1]
    return like(target(cpu_only(arr, f"fastmorph.{target.__name__}"), **kwargs), arr)
