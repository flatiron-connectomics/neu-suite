"""A cache is three methods, so anything can be one.

Reading a body's mesh from S3 takes seconds — measured on sample3, 1.2 s for a skeleton
and 2.2 s for a mesh — so a notebook that redraws a scene wants the second read to be
free. What it does *not* want is a caching library baked into the import graph.

Hence a :class:`Cache` protocol with three dunder methods, an in-memory default, and an
adapter for ``yes3`` when a persistent one is wanted. Anything satisfying the protocol
works: a plain ``dict`` does.
"""

from __future__ import annotations

from typing import Any, Hashable, Optional, Protocol, runtime_checkable


@runtime_checkable
class Cache(Protocol):
    """``key in cache``, ``cache[key]``, ``cache[key] = value``. That is the whole thing."""

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
    """A ``dict``. Named so that passing one reads as an intent rather than an accident."""


class Yes3Cache:
    """Adapter for a ``yes3`` path-backed cache, so results survive the kernel.

    Kept behind an adapter rather than depended on: ``yes3`` is a separate project, and
    the point of the protocol is that this file is the only thing that knows about it.
    """

    def __init__(self, location: Any):
        from yes3.caching import setup_cache

        self._cache = setup_cache(location)

    def __contains__(self, key: Hashable) -> bool:
        return str(key) in self._cache

    def __getitem__(self, key: Hashable) -> Any:
        return self._cache[str(key)]

    def __setitem__(self, key: Hashable, value: Any) -> None:
        self._cache.put(str(key), value, update=True)

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
