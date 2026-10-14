"""Backend registry.

Importing this package registers all concrete backends with the spec opener, so
``open_backend(spec)`` works for any supported format.
"""

from __future__ import annotations

from .base import ArrayBackend, Region, open_backend, register_backend

# Imported for their self-registration. cloudvolume and dvid import fine without
# `cloud-volume` / `neuclease` and raise only when a volume needs one; `import
# neuclease.dvid` costs ~9 s, which nothing that merely lists backends should pay.
from . import tensorstore, imagestack, hdf5, view, cloudvolume, dvid  # noqa: E402,F401
from . import webknossos  # noqa: E402,F401  (stdlib HTTP only; credentials read at open)
from . import mapped  # noqa: E402,F401

__all__ = [
    "ArrayBackend",
    "Region",
    "open_backend",
    "register_backend",
]
