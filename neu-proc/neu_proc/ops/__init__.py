"""The pure half: array in, array out.

Every function here takes a numpy array and **voxel-native** parameters, and returns a new
array. No store, no dask, no blockrun — which is what makes the whole layer importable and
testable on its own, and is pinned by a test asserting that importing it leaves ``neu_vol``,
``tensorstore`` and ``dask`` out of ``sys.modules``.

Two conventions, both of which cost nothing now and are annoying to retrofit:

- **zyx**, like everything else in the suite. A function that takes a per-axis sequence
  takes it in ``(z, y, x)``.
- **Per-axis lengths, or a scalar that broadcasts.** ``sigma_vox=(1, 4, 4)`` is what a
  future ``Op.bind(frame)`` will hand down on an anisotropic volume — and real pyramids are
  anisotropic — so a signature that only accepts a scalar would have to change. scipy
  already takes a sequence for ``sigma`` and for most footprint arguments.

Use them directly, or through a :class:`~neu_lib.Piece`, which is where the metadata
discipline lives: ``piece.apply`` refuses a shape change without a new ``frame=`` and a
dtype change without a ``kind=``, so a function here can stay dumb about both.

    piece.apply(gaussian, sigma_vox=2)
    piece.apply(threshold, at=0.5, kind="segmentation")
"""

from __future__ import annotations
