"""The sharded uint64 index: tensorstore writes it; `ShardReader` reimplements reading so a
caller can learn an entry's byte offset, which the sharded mesh format needs.

The reader is checked against an independent implementation's placements (cloud-volume)
and against `read_one`, i.e. tensorstore reading the same files.
"""

import struct

import pytest

from neu_vol import sharded


def _spec(**kw):
    return sharded.sharding_spec(shard_bits=2, minishard_bits=1, **kw)


# --------------------------------------------------------------------------- #
# the spec block
# --------------------------------------------------------------------------- #
def test_the_spec_matches_a_published_dataset_neuroglancer_reads():
    spec = sharded.sharding_spec(shard_bits=10, minishard_bits=9, preshift_bits=10)
    assert spec == {"@type": "neuroglancer_uint64_sharded_v1",
                    "hash": "murmurhash3_x86_128", "preshift_bits": 10,
                    "shard_bits": 10, "minishard_bits": 9,
                    "data_encoding": "gzip", "minishard_index_encoding": "gzip"}


@pytest.mark.parametrize("kw, match", [
    ({"shard_bits": -1, "minishard_bits": 0}, "non-negative"),
    ({"shard_bits": 0, "minishard_bits": -3}, "non-negative"),
    ({"shard_bits": 0, "minishard_bits": 0, "hash": "sha256"}, "hash must be"),
    ({"shard_bits": 0, "minishard_bits": 0, "data_encoding": "zstd"}, "data_encoding must be"),
])
def test_an_invalid_spec_is_refused(kw, match):
    with pytest.raises(ValueError, match=match):
        sharded.sharding_spec(**kw)


# --------------------------------------------------------------------------- #
# keys
# --------------------------------------------------------------------------- #
def test_keys_are_big_endian_which_is_the_one_place_the_order_flips():
    """Everything else in precomputed is little-endian."""
    assert sharded.key(1) == b"\x00\x00\x00\x00\x00\x00\x00\x01"
    assert sharded.key(0x0102030405060708) == bytes(range(1, 9))
    assert struct.unpack(">Q", sharded.key(2**63 + 5))[0] == 2**63 + 5


# --------------------------------------------------------------------------- #
# writing and reading back
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spec, entries", [
    (_spec(), [(i * 7 + 1, f"payload-{i}".encode()) for i in range(200)]),
    (sharded.sharding_spec(shard_bits=1, minishard_bits=1, data_encoding="raw",
                           minishard_index_encoding="raw"), [(5, b"uncompressed")]),
    # large, hashed-looking ids, as published annotation ids are
    (_spec(), [(i, str(i).encode())
               for i in (48545692435670922, 582569736127172416, 2**64 - 1)]),
])
def test_a_batch_round_trips(tmp_path, spec, entries):
    assert sharded.write_all(str(tmp_path / "idx"), spec, entries) == len(entries)
    for chunk_id, payload in entries[:20]:
        assert sharded.read_one(str(tmp_path / "idx"), spec, chunk_id) == payload


def test_a_missing_key_reads_as_none(tmp_path):
    spec = _spec()
    sharded.write_all(str(tmp_path / "idx"), spec, [(1, b"x")])
    assert sharded.read_one(str(tmp_path / "idx"), spec, 999) is None


def test_the_file_count_is_two_to_the_shard_bits(tmp_path):
    spec = sharded.sharding_spec(shard_bits=2, minishard_bits=0)
    sharded.write_all(str(tmp_path / "idx"), spec, [(i, b"v") for i in range(64)])
    shards = sorted(p.name for p in (tmp_path / "idx").iterdir())
    assert shards == ["0.shard", "1.shard", "2.shard", "3.shard"]


def test_all_zero_bits_writes_a_single_shard(tmp_path):
    """Legal, and what a small (coarse) spatial level uses."""
    spec = sharded.sharding_spec(shard_bits=0, minishard_bits=0)
    sharded.write_all(str(tmp_path / "idx"), spec, [(0, b"only")])
    assert [p.name for p in (tmp_path / "idx").iterdir()] == ["0.shard"]
    assert sharded.read_one(str(tmp_path / "idx"), spec, 0) == b"only"


def test_writing_nothing_is_not_an_error(tmp_path):
    assert sharded.write_all(str(tmp_path / "idx"), _spec(), []) == 0


# --------------------------------------------------------------------------- #
# planning
# --------------------------------------------------------------------------- #
def test_planning_scales_the_shard_count_with_the_key_count():
    small = sharded.plan_sharding(100)
    big = sharded.plan_sharding(1_400_000)
    assert small["shard_bits"] == 0            # one file is plenty
    assert big["shard_bits"] > small["shard_bits"]
    assert 2 ** big["shard_bits"] <= 1024


def test_planning_is_capped_so_a_huge_input_does_not_explode_the_file_count():
    assert sharded.plan_sharding(312_000_000)["shard_bits"] <= 10


def test_planning_produces_a_usable_spec(tmp_path):
    spec = sharded.plan_sharding(5_000)
    sharded.write_all(str(tmp_path / "idx"), spec, [(i, b"v") for i in range(5_000)])
    assert sharded.read_one(str(tmp_path / "idx"), spec, 4_999) == b"v"


def test_a_non_positive_key_count_is_refused():
    with pytest.raises(ValueError, match="must be positive"):
        sharded.plan_sharding(0)


# --------------------------------------------------------------------------- #
# compressed morton code — how a sharded index keys a GRID CELL
# --------------------------------------------------------------------------- #
def test_bits_are_spent_only_where_the_grid_subdivides():
    """ceil(log2(size)) per axis, and zero for a flat axis."""
    assert sharded.morton_bits([1, 1, 1]) == [0, 0, 0]
    assert sharded.morton_bits([2, 1, 1]) == [1, 0, 0]
    assert sharded.morton_bits([16, 8, 16]) == [4, 3, 4]
    assert sharded.morton_bits([3, 5, 9]) == [2, 3, 4]


@pytest.mark.parametrize("pos, grid, code", [
    # x varies fastest: on 2x2x2 the code is x + 2y + 4z (row-major would swap x and z)
    ([1, 0, 0], [2, 2, 2], 1),
    ([0, 1, 0], [2, 2, 2], 2),
    ([0, 0, 1], [2, 2, 2], 4),
    ([1, 1, 1], [2, 2, 2], 7),
    ([2, 1, 0], [4, 4, 1], 6),           # bits interleave x0,y0,x1,y1 = 0,1,1,0
    ([3, 0, 3], [4, 1, 4], 0b1111),      # a flat axis (y) spends no bits
    ([1, 0, 2], [4, 1, 4], 0b1001),
])
def test_compressed_morton_code(pos, grid, code):
    assert sharded.compressed_morton_code(pos, grid) == code


def test_the_code_is_a_bijection_over_the_grid():
    """Every cell gets its own key."""
    import itertools

    import numpy as np

    grid = [4, 2, 8]
    cells = np.array(list(itertools.product(range(4), range(2), range(8))))
    codes = sharded.compressed_morton_code(cells, grid)
    assert len(set(codes.tolist())) == len(cells) == 64
    assert min(codes) == 0 and max(codes) == 63


def test_it_agrees_with_a_row_major_index_exactly_where_that_is_misleading():
    """Degenerate grids agree with row-major order; real multi-axis grids must not."""
    import itertools

    import numpy as np

    def row_major(cells, grid):
        return (cells[:, 0] * grid[1] + cells[:, 1]) * grid[2] + cells[:, 2]

    for grid in ([1, 1, 1], [2, 1, 1], [8, 1, 1]):
        cells = np.array(list(itertools.product(*(range(g) for g in grid))))
        assert np.array_equal(sharded.compressed_morton_code(cells, grid),
                              row_major(cells, grid)), grid

    grid = [16, 8, 16]
    cells = np.array(list(itertools.product(*(range(g) for g in grid))))
    agree = sharded.compressed_morton_code(cells, grid) == row_major(cells, grid)
    assert agree.sum() < 0.05 * len(cells), "a multi-axis grid must NOT agree"


def test_a_position_of_the_wrong_rank_is_refused():
    with pytest.raises(ValueError, match="axes"):
        sharded.compressed_morton_code([1, 2], [4, 4, 4])


def test_rewriting_an_index_replaces_it_rather_than_merging(tmp_path):
    """A merge would keep serving stale keys from the previous key space."""
    dst = str(tmp_path / "idx")
    spec = sharded.plan_sharding(10)
    sharded.write_all(dst, spec, [(i, b"old") for i in range(10)])
    sharded.write_all(dst, spec, [(i, b"new") for i in range(5, 15)])

    assert sharded.read_one(dst, spec, 7) == b"new"
    assert sharded.read_one(dst, spec, 14) == b"new"
    assert sharded.read_one(dst, spec, 0) is None, "a key only the old write used must be gone"


def test_replace_false_keeps_what_was_there(tmp_path):
    dst = str(tmp_path / "idx")
    spec = sharded.plan_sharding(10)
    sharded.write_all(dst, spec, [(i, b"old") for i in range(10)])
    sharded.write_all(dst, spec, [(i, b"new") for i in range(5, 15)], replace=False)
    assert sharded.read_one(dst, spec, 0) == b"old"
    assert sharded.read_one(dst, spec, 7) == b"new"


def test_clear_removes_the_shard_objects(tmp_path):
    dst = str(tmp_path / "idx")
    spec = sharded.plan_sharding(5_000)
    sharded.write_all(dst, spec, [(i, b"v") for i in range(5_000)])
    assert sharded.clear(dst) > 0
    assert sharded.read_one(dst, spec, 1) is None
    assert sharded.clear(dst) == 0        # idempotent


def test_a_cell_written_at_its_morton_key_reads_back_by_position(tmp_path):
    """Read back by a code recomputed from the grid position, as a viewer does."""
    grid = [4, 2, 4]
    spec = sharded.plan_sharding(32)
    entries = [(sharded.compressed_morton_code([x, y, z], grid), f"{x}-{y}-{z}".encode())
               for x in range(4) for y in range(2) for z in range(4)]
    dst = str(tmp_path / "spatial")
    sharded.write_all(dst, spec, entries)
    for x, y, z in ((0, 0, 0), (3, 1, 3), (2, 0, 1)):
        code = sharded.compressed_morton_code([x, y, z], grid)
        assert sharded.read_one(dst, spec, code) == f"{x}-{y}-{z}".encode()


# --------------------------------------------------------------------------- #
# ShardReader: addressing
# --------------------------------------------------------------------------- #

#: (sharding params, [(key, shard filename stem, minishard number)]), from cloud-volume's
#: compute_shard_location. An independent implementation, because a wrong hash still gives a
#: self-consistent reader that finds nothing, which looks like absent data.
SHARD_VECTORS = [
    (dict(preshift_bits=6, minishard_bits=8, shard_bits=10),
     [(0, "0ae", 65), (1, "0ae", 65), (24740, "30e", 43), (255, "3e4", 209),
      (4096, "119", 255), (123456789, "06c", 143), (2**63, "0af", 35),
      (2**64 - 1, "230", 252)]),
    (dict(preshift_bits=0, minishard_bits=0, shard_bits=0),
     [(0, "0", 0), (24740, "0", 0), (2**64 - 1, "0", 0)]),
    (dict(preshift_bits=9, minishard_bits=6, shard_bits=11),
     [(0, "2b9", 1), (24740, "053", 12), (255, "2b9", 1), (4096, "627", 1),
      (123456789, "4c1", 17), (2**63, "1cb", 10), (2**64 - 1, "04d", 34)]),
    (dict(preshift_bits=0, minishard_bits=3, shard_bits=5),
     [(0, "08", 1), (1, "13", 2), (24740, "10", 4), (4096, "07", 5),
      (123456789, "04", 7), (2**63, "15", 0), (2**64 - 1, "03", 2)]),
]


@pytest.mark.parametrize("params,cases", SHARD_VECTORS)
def test_shard_placement_matches_an_independent_implementation(params, cases):
    spec = sharded.sharding_spec(**params)
    for chunk_id, shard, minishard in cases:
        assert sharded.shard_location(chunk_id, spec) == (f"{shard}.shard", minishard), \
            f"key {chunk_id} placed wrong for {params}"


def test_preshift_bits_send_neighbouring_keys_to_one_shard():
    """With 6 preshift bits, ids 0..63 share a shard."""
    spec = sharded.sharding_spec(shard_bits=10, minishard_bits=8, preshift_bits=6)
    first = {sharded.shard_location(i, spec) for i in range(64)}
    assert len(first) == 1
    assert sharded.shard_location(64, spec) != sharded.shard_location(0, spec)


def test_the_identity_hash_is_not_the_murmur_one():
    """The spec's `hash` is read, not assumed."""
    ident = sharded.sharding_spec(shard_bits=5, minishard_bits=3, hash="identity")
    murmur = sharded.sharding_spec(shard_bits=5, minishard_bits=3)
    # 24740 = 0b110000010100100: low 3 bits are the minishard, next 5 the shard.
    assert sharded.shard_location(24740, ident) == ("14.shard", 4)
    assert sharded.shard_location(24740, murmur) != sharded.shard_location(24740, ident)


def test_an_unknown_hash_is_refused_rather_than_silently_treated_as_identity():
    spec = dict(sharded.sharding_spec(shard_bits=2, minishard_bits=1))
    spec["hash"] = "sha256"
    with pytest.raises(ValueError, match="unknown sharded hash"):
        sharded.shard_location(1, spec)


def test_the_murmur_digest_is_the_x86_variant():
    """x86_128, not x64_128: known answer for the empty input."""
    assert sharded.murmurhash3_x86_128_low64(b"") == 0
    assert sharded.murmurhash3_x86_128_low64(b"", seed=1) != 0
    # Stable across calls and sensitive to every input byte.
    digests = {sharded.murmurhash3_x86_128_low64(bytes([b]) + b"\0" * 7) for b in range(32)}
    assert len(digests) == 32
    assert all(0 <= d < 2**64 for d in digests)


# --------------------------------------------------------------------------- #
# ShardReader: values
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("encoding", ["gzip", "raw"])
def test_shard_reader_agrees_with_tensorstore(tmp_path, encoding):
    """`read_one` (tensorstore) and `ShardReader` (our own parsing) return the same bytes."""
    spec = sharded.sharding_spec(shard_bits=3, minishard_bits=2, preshift_bits=0,
                                 data_encoding=encoding, minishard_index_encoding=encoding)
    entries = [(i * 37 + 5, f"payload-{i}-{'x' * i}".encode()) for i in range(150)]
    sharded.write_all(str(tmp_path / "idx"), spec, entries)

    reader = sharded.ShardReader(str(tmp_path / "idx"), spec)
    for chunk_id, payload in entries:
        assert reader.read(chunk_id) == payload
        assert sharded.read_one(str(tmp_path / "idx"), spec, chunk_id) == payload


def test_shard_reader_reports_a_missing_key_as_none(tmp_path):
    spec = sharded.sharding_spec(shard_bits=2, minishard_bits=1)
    sharded.write_all(str(tmp_path / "idx"), spec, [(1, b"a"), (2, b"b")])
    reader = sharded.ShardReader(str(tmp_path / "idx"), spec)
    assert reader.read(999) is None
    assert reader.locate(999) is None


def test_shard_reader_on_an_index_that_does_not_exist_is_none_not_an_error(tmp_path):
    spec = sharded.sharding_spec(shard_bits=2, minishard_bits=1)
    reader = sharded.ShardReader(str(tmp_path / "nothing-here"), spec)
    assert reader.locate(1) is None
    assert reader.read(1) is None


def test_locate_gives_the_byte_range_the_value_actually_occupies(tmp_path):
    """The mesh reader trusts these offsets to find fragment data outside the index."""
    spec = sharded.sharding_spec(shard_bits=1, minishard_bits=1, data_encoding="raw",
                                 minishard_index_encoding="raw")
    entries = [(i, bytes([i]) * (10 + i)) for i in range(1, 20)]
    sharded.write_all(str(tmp_path / "idx"), spec, entries)

    reader = sharded.ShardReader(str(tmp_path / "idx"), spec)
    for chunk_id, payload in entries:
        shard, offset, size = reader.locate(chunk_id)
        assert size == len(payload), "raw encoding stores the value verbatim"
        assert reader.read_range(shard, offset, offset + size) == payload


def test_the_indexes_are_read_once_per_shard_not_once_per_key(tmp_path, monkeypatch):
    """A batch from one shard reads its two indexes once, not once per key."""
    from neu_vol import location

    spec = sharded.sharding_spec(shard_bits=0, minishard_bits=0)
    entries = [(i, f"v{i}".encode()) for i in range(50)]
    sharded.write_all(str(tmp_path / "idx"), spec, entries)

    calls = []
    real = location.read_range
    monkeypatch.setattr(location, "read_range",
                        lambda *a, **k: (calls.append(a[3:]), real(*a, **k))[1])

    reader = sharded.ShardReader(str(tmp_path / "idx"), spec)
    for chunk_id, payload in entries:
        assert reader.read(chunk_id) == payload
    # 50 value reads, plus exactly one shard index and one minishard index.
    assert len(calls) == 52


def test_clear_leaves_a_sibling_info_alone(tmp_path):
    """A sharded mesh or skeleton sits beside its `info`; only `.shard` objects are removed."""
    from neu_vol import location

    spec = sharded.sharding_spec(shard_bits=1, minishard_bits=1)
    dst = str(tmp_path / "skeleton")
    location.write_bytes(dst, b'{"@type": "neuroglancer_skeletons"}', "info")
    sharded.write_all(dst, spec, [(7, b"a-skeleton")])

    assert location.read_bytes(dst, "info") is not None, "info was deleted by the rewrite"
    assert sharded.read_one(dst, spec, 7) == b"a-skeleton"

    # And a rewrite must still replace the shards rather than merging into them.
    sharded.write_all(dst, spec, [(8, b"another")])
    assert sharded.read_one(dst, spec, 7) is None
    assert location.read_bytes(dst, "info") is not None


def test_clear_reports_only_the_shards_it_removed(tmp_path):
    from neu_vol import location

    spec = sharded.sharding_spec(shard_bits=2, minishard_bits=0)
    dst = str(tmp_path / "idx")
    sharded.write_all(dst, spec, [(i, b"v") for i in range(64)])
    location.write_bytes(dst, b"{}", "info")

    n_shards = len([p for p in (tmp_path / "idx").iterdir() if p.name.endswith(".shard")])
    assert sharded.clear(dst) == n_shards
    assert location.read_bytes(dst, "info") == b"{}"
