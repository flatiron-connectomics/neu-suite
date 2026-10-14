"""Writing and reading ``neuroglancer_uint64_sharded_v1`` indexes.

A sharded index stores many keys in a few ``.shard`` files instead of one object each,
which matters at scale: an annotation source with hundreds of millions of annotations, or a
volume under an inode quota.

Writing goes entirely through tensorstore's ``neuroglancer_uint64_sharded`` kvstore driver
(:func:`write_all`, :func:`read_one`), so the format is not reimplemented for it.
:class:`ShardReader` is the one exception: the sharded multi-resolution mesh format stores
fragment data outside the index, at byte offsets relative to the manifest, and the driver
cannot say where an entry lies. So the addressing (hash, shard index, minishard table) is
reimplemented here, for reading, and a test asserts it agrees with the driver.

Two things a caller has to get right:

- **Keys are 8-byte big-endian uint64** (:func:`key`); everything else in precomputed is
  little-endian.
- **The ``metadata`` block is the sharding spec that also goes into your ``info``.** Pass
  the same dict to both.

``shard_bits=0`` with ``minishard_bits=0`` writes a single ``0.shard``, which suits a
small index.
"""

from __future__ import annotations

import struct
from typing import Any, Iterable, Mapping, Sequence

AT_TYPE = "neuroglancer_uint64_sharded_v1"

#: The only hash the format defines besides ``identity``. The reference uses this for every
#: index; ``identity`` is for keys that are already uniformly distributed.
DEFAULT_HASH = "murmurhash3_x86_128"


def key(chunk_id: int) -> bytes:
    """A uint64 key as the sharded kvstore expects it: 8 bytes, **big**-endian."""
    return struct.pack(">Q", int(chunk_id))


def morton_bits(grid_shape: Sequence[int]) -> list[int]:
    """How many bits the compressed Morton code spends on each dimension.

    ``ceil(log2(size))`` per the spec, so zero for a size-1 dimension.
    """
    out = []
    for size in grid_shape:
        bits = 0
        while (1 << bits) < int(size):
            bits += 1
        out.append(bits)
    return out


def compressed_morton_code(positions, grid_shape: Sequence[int]):
    """Compressed Morton codes for grid cell positions, as precomputed keys them.

    ``positions`` is ``(n, 3)`` or ``(3,)`` in the same axis order as ``grid_shape``, which
    for precomputed is xyz with x varying fastest. Returns an ``int64`` array (or one int).

    Not a row-major flattening: the two agree only while at most one axis is subdivided,
    so a small-grid test cannot tell them apart. Used for hand-built sharded indexes such
    as an annotation spatial index; tensorstore keys volume chunks itself.
    """
    import numpy as np

    array = np.atleast_2d(np.asarray(positions, dtype=np.int64))
    if array.shape[-1] != len(grid_shape):
        raise ValueError(f"positions have {array.shape[-1]} axes, grid_shape has "
                         f"{len(grid_shape)}")
    per_dim = morton_bits(grid_shape)
    code = np.zeros(len(array), dtype=np.int64)
    j = 0
    for i in range(max(per_dim, default=0)):
        for dim, bits in enumerate(per_dim):
            if i < bits:
                code |= ((array[:, dim] >> i) & 1) << j
                j += 1
    return code if np.ndim(positions) > 1 else int(code[0])


def sharding_spec(*, shard_bits: int, minishard_bits: int, preshift_bits: int = 0,
                  hash: str = DEFAULT_HASH, data_encoding: str = "gzip",
                  minishard_index_encoding: str = "gzip") -> dict[str, Any]:
    """The sharding metadata, for both the kvstore and the ``info`` that describes it.

    ``2**shard_bits`` is the number of ``.shard`` files. ``minishard_bits`` splits each shard
    internally, bounding how much index a reader fetches to find one key.
    """
    for name, value in (("shard_bits", shard_bits), ("minishard_bits", minishard_bits),
                        ("preshift_bits", preshift_bits)):
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer; got {value!r}")
    if hash not in (DEFAULT_HASH, "identity"):
        raise ValueError(f"hash must be {DEFAULT_HASH!r} or 'identity'; got {hash!r}")
    for name, value in (("data_encoding", data_encoding),
                        ("minishard_index_encoding", minishard_index_encoding)):
        if value not in ("raw", "gzip"):
            raise ValueError(f"{name} must be 'raw' or 'gzip'; got {value!r}")
    return {"@type": AT_TYPE, "hash": hash, "preshift_bits": preshift_bits,
            "shard_bits": shard_bits, "minishard_bits": minishard_bits,
            "data_encoding": data_encoding,
            "minishard_index_encoding": minishard_index_encoding}


def plan_sharding(n_keys: int, *, per_shard: int = 30_000,
                  max_shard_bits: int = 10) -> dict[str, Any]:
    """Sharding parameters for ``n_keys`` entries, aiming at ``per_shard`` keys per file.

    ``minishard_bits`` then keeps each minishard index small, since a reader fetching one
    key decodes a whole minishard index. A heuristic, not a rule from the spec.
    """
    if n_keys <= 0:
        raise ValueError("n_keys must be positive")
    shard_bits = 0
    while (n_keys >> shard_bits) > per_shard and shard_bits < max_shard_bits:
        shard_bits += 1
    keys_per_shard = max(1, n_keys >> shard_bits)
    # Aim for a few thousand keys per minishard: enough that the index is worth having,
    # small enough that reading one key does not decode a huge index.
    minishard_bits = 0
    while (keys_per_shard >> minishard_bits) > 2_000 and minishard_bits < 9:
        minishard_bits += 1
    return sharding_spec(shard_bits=shard_bits, minishard_bits=minishard_bits)


def open_kvstore(location: str | Mapping[str, Any], metadata: Mapping[str, Any],
                 *parts: str):
    """A sharded kvstore rooted at ``location``/``parts``, local or remote.

    The trailing slash the driver needs is added here.
    """
    import tensorstore as ts

    from .location import ensure_credentials, join, to_kvstore

    base = join(to_kvstore(location), *parts) if parts else dict(to_kvstore(location))
    base = ensure_credentials(base)
    path = str(base.get("path", ""))
    if path and not path.endswith("/"):
        base = {**base, "path": path + "/"}
    return ts.KvStore.open({"driver": "neuroglancer_uint64_sharded",
                            "base": base, "metadata": dict(metadata)}).result()


def clear(location: str | Mapping[str, Any], *parts: str) -> int:
    """Delete the ``.shard`` objects of an index. Returns how many were removed.

    Not atomic: an interrupted rewrite leaves fewer shards than either version had. Only
    ``.shard`` objects are removed, because a sharded mesh or skeleton shares its
    directory with the ``info`` that declares it.
    """
    from .location import _kv, list_keys

    removed = 0
    for name in list_keys(location, *parts):
        if not name.endswith(".shard"):
            continue
        store, resolved = _kv(location, *parts, name)
        store.write(resolved, None).result()
        removed += 1
    return removed


def write_all(location: str | Mapping[str, Any], metadata: Mapping[str, Any],
              entries: Iterable[tuple[int, bytes]], *parts: str,
              replace: bool = True) -> int:
    """Write ``(chunk_id, bytes)`` pairs into a sharded index. Returns how many.

    One transaction for the whole batch, so each shard is written once rather than once
    per key.

    ``replace`` (the default) deletes the existing shards first. Without it, keys already
    present and not rewritten stay, so an index whose key space changed would hold both
    generations behind unchanged object names. Pass ``replace=False`` only to add to an
    index on purpose.
    """
    import tensorstore as ts

    if replace:
        clear(location, *parts)
    kv = open_kvstore(location, metadata, *parts)
    written = 0
    with ts.Transaction() as txn:
        scoped = kv.with_transaction(txn)
        for chunk_id, payload in entries:
            scoped[key(chunk_id)] = bytes(payload)
            written += 1
    return written


def read_one(location: str | Mapping[str, Any], metadata: Mapping[str, Any],
             chunk_id: int, *parts: str) -> bytes | None:
    """One entry by id, or ``None`` if absent. For tests and for spot-checking output."""
    kv = open_kvstore(location, metadata, *parts)
    result = kv.read(key(chunk_id)).result()
    return bytes(result.value) if result.state == "value" else None


# --------------------------------------------------------------------------- #
# reading by byte offset
# --------------------------------------------------------------------------- #

_M32 = 0xFFFFFFFF


def _rotl32(x: int, r: int) -> int:
    return ((x << r) | (x >> (32 - r))) & _M32


def _fmix32(h: int) -> int:
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & _M32
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & _M32
    h ^= h >> 16
    return h


def murmurhash3_x86_128_low64(data: bytes, seed: int = 0) -> int:
    """The low 64 bits of MurmurHash3 x86_128, the hash the sharded format specifies.

    Pure Python: it runs once per key looked up, so a compiled dependency is not worth it.
    ``x86_128``, not ``x64_128``, which gives a different digest and sends every read to the
    wrong shard.
    """
    c1, c2, c3, c4 = 0x239B961B, 0xAB0E9789, 0x38B34AE5, 0xA1E38B93
    h1 = h2 = h3 = h4 = seed & _M32
    length = len(data)
    nblocks = length // 16

    for i in range(nblocks):
        k1, k2, k3, k4 = struct.unpack_from("<IIII", data, i * 16)
        k1 = (_rotl32((k1 * c1) & _M32, 15) * c2) & _M32
        h1 ^= k1
        h1 = ((_rotl32(h1, 19) + h2) * 5 + 0x561CCD1B) & _M32
        k2 = (_rotl32((k2 * c2) & _M32, 16) * c3) & _M32
        h2 ^= k2
        h2 = ((_rotl32(h2, 17) + h3) * 5 + 0x0BCAA747) & _M32
        k3 = (_rotl32((k3 * c3) & _M32, 17) * c4) & _M32
        h3 ^= k3
        h3 = ((_rotl32(h3, 15) + h4) * 5 + 0x96CD1C35) & _M32
        k4 = (_rotl32((k4 * c4) & _M32, 18) * c1) & _M32
        h4 ^= k4
        h4 = ((_rotl32(h4, 13) + h1) * 5 + 0x32AC3B17) & _M32

    tail = data[nblocks * 16:]
    n = len(tail)
    k1 = k2 = k3 = k4 = 0
    if n >= 15:
        k4 ^= tail[14] << 16
    if n >= 14:
        k4 ^= tail[13] << 8
    if n >= 13:
        k4 ^= tail[12]
        h4 ^= (_rotl32((k4 * c4) & _M32, 18) * c1) & _M32
    if n >= 12:
        k3 ^= tail[11] << 24
    if n >= 11:
        k3 ^= tail[10] << 16
    if n >= 10:
        k3 ^= tail[9] << 8
    if n >= 9:
        k3 ^= tail[8]
        h3 ^= (_rotl32((k3 * c3) & _M32, 17) * c4) & _M32
    if n >= 8:
        k2 ^= tail[7] << 24
    if n >= 7:
        k2 ^= tail[6] << 16
    if n >= 6:
        k2 ^= tail[5] << 8
    if n >= 5:
        k2 ^= tail[4]
        h2 ^= (_rotl32((k2 * c2) & _M32, 16) * c3) & _M32
    if n >= 4:
        k1 ^= tail[3] << 24
    if n >= 3:
        k1 ^= tail[2] << 16
    if n >= 2:
        k1 ^= tail[1] << 8
    if n >= 1:
        k1 ^= tail[0]
        h1 ^= (_rotl32((k1 * c1) & _M32, 15) * c2) & _M32

    h1 ^= length
    h2 ^= length
    h3 ^= length
    h4 ^= length
    h1 = (h1 + h2 + h3 + h4) & _M32
    h2 = (h2 + h1) & _M32
    h3 = (h3 + h1) & _M32
    h4 = (h4 + h1) & _M32
    h1, h2, h3, h4 = _fmix32(h1), _fmix32(h2), _fmix32(h3), _fmix32(h4)
    h1 = (h1 + h2 + h3 + h4) & _M32
    h2 = (h2 + h1) & _M32
    return h1 | (h2 << 32)


def shard_location(chunk_id: int, metadata: Mapping[str, Any]) -> tuple[str, int]:
    """``(shard filename, minishard number)`` for a key, per the sharding spec.

    The filename is lowercase hex zero-padded to ``ceil(shard_bits / 4)`` digits plus
    ``.shard``; ``shard_bits=0`` therefore gives ``0.shard``, the single-shard case.
    """
    preshift = int(metadata.get("preshift_bits", 0))
    minishard_bits = int(metadata["minishard_bits"])
    shard_bits = int(metadata["shard_bits"])

    hashed = int(chunk_id) >> preshift
    how = metadata.get("hash", "identity")
    if how == DEFAULT_HASH:
        hashed = murmurhash3_x86_128_low64(struct.pack("<Q", hashed & 0xFFFFFFFFFFFFFFFF))
    elif how != "identity":
        raise ValueError(f"unknown sharded hash {how!r}; expected 'identity' or "
                         f"{DEFAULT_HASH!r}")

    minishard = hashed & ((1 << minishard_bits) - 1)
    shard = (hashed >> minishard_bits) & ((1 << shard_bits) - 1)
    width = -(-shard_bits // 4) or 1
    return f"{shard:0{width}x}.shard", minishard


class ShardReader:
    """Locates and reads entries of a sharded index, by key and by byte offset.

    For the sharded multi-resolution mesh format, whose fragment data sits before the
    manifest at an offset the tensorstore driver cannot report. :func:`read_one` is the
    call for anything that only wants a value.

    The shard and minishard indexes are cached on the instance, so reuse one reader across
    a batch: each shard's indexes are then read once rather than once per body.
    """

    def __init__(self, location: str | Mapping[str, Any], metadata: Mapping[str, Any],
                 *parts: str):
        self.location = location
        self.metadata = dict(metadata)
        self.parts = tuple(parts)
        self._index_length = (1 << int(self.metadata["minishard_bits"])) * 16
        self._shard_index: dict[str, Any] = {}
        self._minishard_index: dict[tuple[str, int], Any] = {}

    def read_range(self, shard: str, start: int, end: int) -> bytes | None:
        """Raw bytes ``[start, end)`` of one shard file, undecoded."""
        from .location import read_range

        if end <= start:
            return b""
        return read_range(self.location, start, end, *self.parts, shard)

    def _shard_table(self, shard: str):
        if shard not in self._shard_index:
            import numpy as np

            raw = self.read_range(shard, 0, self._index_length)
            if raw is None or len(raw) < self._index_length:
                # A missing or short shard file holds no keys to look up.
                self._shard_index[shard] = None
            else:
                self._shard_index[shard] = np.frombuffer(raw, dtype="<u8").reshape(-1, 2)
        return self._shard_index[shard]

    def _minishard_table(self, shard: str, minishard: int):
        if (shard, minishard) in self._minishard_index:
            return self._minishard_index[(shard, minishard)]

        import gzip

        import numpy as np

        table = self._shard_table(shard)
        result = None
        if table is not None and minishard < len(table):
            # Offsets in both index levels are relative to the END of the shard index.
            start = int(table[minishard][0]) + self._index_length
            end = int(table[minishard][1]) + self._index_length
            if end > start:
                raw = self.read_range(shard, start, end)
                if raw:
                    if self.metadata.get("minishard_index_encoding", "raw") == "gzip":
                        raw = gzip.decompress(raw)
                    flat = np.frombuffer(raw, dtype="<u8")
                    n = flat.size // 3
                    # Three parallel arrays, each delta-encoded against a different
                    # baseline: ids against the previous id, starts against the previous
                    # END (hence the sizes term), sizes stored outright.
                    ids = np.cumsum(flat[0:n])
                    sizes = flat[2 * n:3 * n]
                    starts = np.cumsum(flat[n:2 * n])
                    starts[1:] += np.cumsum(sizes[:-1])
                    starts += self._index_length
                    result = np.stack([ids, starts, sizes], axis=1)
        self._minishard_index[(shard, minishard)] = result
        return result

    def locate(self, chunk_id: int) -> tuple[str, int, int] | None:
        """``(shard, offset, size)`` of the stored entry, or ``None`` if the key is absent.

        ``size`` is the encoded length on disk (compressed, with ``data_encoding='gzip'``),
        which is what the mesh format's offsets are relative to.
        """
        import numpy as np

        shard, minishard = shard_location(chunk_id, self.metadata)
        table = self._minishard_table(shard, minishard)
        if table is None:
            return None
        rows = np.nonzero(table[:, 0] == np.uint64(int(chunk_id)))[0]
        if not len(rows):
            return None
        return shard, int(table[rows[0], 1]), int(table[rows[0], 2])

    def read(self, chunk_id: int) -> bytes | None:
        """One entry by id, decoded per ``data_encoding``; ``None`` if absent."""
        import gzip

        found = self.locate(chunk_id)
        if found is None:
            return None
        shard, offset, size = found
        raw = self.read_range(shard, offset, offset + size)
        if raw is None:
            return None
        if self.metadata.get("data_encoding", "raw") == "gzip":
            raw = gzip.decompress(raw)
        return raw
