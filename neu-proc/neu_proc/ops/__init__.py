"""The array-in, array-out half of neu-proc.

Every function takes an array and voxel parameters, and returns a new array. Nothing here
imports a store, dask or blockrun; a test checks that importing this package leaves
``neu_vol``, ``tensorstore`` and ``dask`` unimported.

- Per-axis arguments are zyx.
- Lengths are per axis, or a scalar applied to all three, since a physical length becomes
  different voxel counts per axis on an anisotropic volume: ``sigma_vox=(1, 4, 4)``.

Use them directly, or through :meth:`neu_lib.Piece.apply`, which refuses a shape change
without a new ``frame=`` and a dtype change without ``kind=``::

    piece.apply(dust, min_voxels=100)
    piece.apply(opening, radius=2)
"""

from __future__ import annotations
