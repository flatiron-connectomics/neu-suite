"""Caching for ``sources`` reads, which take seconds each from an object store.

A :class:`Cache` is anything with ``in``, ``[]`` and ``[]=``, so a plain ``dict`` works.
:class:`MemoryCache` is the in-memory default and :class:`Yes3Cache` a persistent one.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Hashable, Optional, Protocol, runtime_checkable


@runtime_checkable
class Cache(Protocol):
    """``key in cache``, ``cache[key]``, ``cache[key] = value``."""

    def __contains__(self, key: Hashable) -> bool: ...
    def __getitem__(self, key: Hashable) -> Any: ...
    def __setitem__(self, key: Hashable, value: Any) -> None: ...


class NullCache:
    """Caches nothing. The explicit way to turn caching off."""

    def __contains__(self, key: Hashable) -> bool:
        return False

    def __getitem__(self, key: Hashable) -> Any:
        raise KeyError(key)

    def __setitem__(self, key: Hashable, value: Any) -> None:
        pass

    def __repr__(self) -> str:
        return "NullCache()"


class MemoryCache(dict):
    """A ``dict``, named for readability at the call site."""


_SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}")
_PLAIN = (str, int, float, bool, type(None))


def _plain(key: Any) -> bool:
    return isinstance(key, _PLAIN) or (isinstance(key, tuple) and all(_plain(k) for k in key))


def _normalize_key(key: Hashable) -> str:
    """A file name for ``key``, since yes3 stores each item at ``<location>/<key>.pkl``.

    A str that is already a safe file name is kept. Anything else, e.g.
    ``("mesh", "s3://bucket/volume", 123, 1)``, becomes its safe parts plus a hash of its
    ``repr`` (``mesh__123__1__<12 hex>``), so keys differing only in an unsafe part stay apart.
    Only plain values are accepted: an object's default ``repr`` holds a memory address, so its
    key would differ in every process.
    """
    if not _plain(key):
        raise TypeError(f"a Yes3Cache key must be a str, number, bool, None or a tuple of them, "
                        f"not {type(key).__name__}")
    if isinstance(key, str) and _SAFE.fullmatch(key):
        return key
    parts = key if isinstance(key, tuple) else (key,)
    words = [str(p) for p in parts if _SAFE.fullmatch(str(p)) and len(str(p)) <= 40]
    digest = hashlib.blake2b(repr(key).encode(), digest_size=6).hexdigest()
    return "__".join([*words, digest])


class Yes3Cache:
    """Adapter for a ``yes3`` path-backed cache, so results survive a kernel restart
    (needs the ``cache`` extra). Keys go through :func:`_normalize_key`, since yes3 uses
    them as file names."""

    def __init__(self, location: Any):
        from yes3.caching import setup_cache

        self._cache = setup_cache(location)

    def __contains__(self, key: Hashable) -> bool:
        return _normalize_key(key) in self._cache

    def __getitem__(self, key: Hashable) -> Any:
        return self._cache[_normalize_key(key)]

    def __setitem__(self, key: Hashable, value: Any) -> None:
        self._cache.put(_normalize_key(key), value, update=True)

    def __repr__(self) -> str:
        return f"Yes3Cache({self._cache!r})"


def resolve(cache: Optional[Cache | str | Any], default: Optional[Cache] = None) -> Cache:
    """Normalise the ``cache=`` argument every source function takes.

    ``None`` means ``default`` (or no caching); ``False`` means explicitly none; a path
    means a ``yes3`` cache there; anything with the three methods is used as-is.
    """
    if cache is False:
        return NullCache()
    if cache is None:
        return default if default is not None else NullCache()
    if isinstance(cache, (str,)) or hasattr(cache, "__fspath__"):
        return Yes3Cache(cache)
    if isinstance(cache, Cache):
        return cache
    raise TypeError(
        f"a cache needs __contains__, __getitem__ and __setitem__; "
        f"{type(cache).__name__} has "
        f"{[m for m in ('__contains__', '__getitem__', '__setitem__') if hasattr(cache, m)]}")
