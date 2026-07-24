from em_blockrun import Manifest


def test_record_load_roundtrip_tuple_keys(tmp_path):
    p = str(tmp_path / "prog.jsonl")
    m = Manifest(p)
    m.reset()
    m.record(0, [((0, 0, 0), "written"), ((0, 0, 1), "empty")])
    m.record(1, [((0, 0, 0), "written")])
    m.close()

    m2 = Manifest(p).load()
    assert m2.is_done(0, (0, 0, 0))
    assert m2.is_done(0, (0, 0, 1))       # empty counts as recorded
    assert m2.is_done(1, (0, 0, 0))
    assert not m2.is_done(0, (9, 9, 9))
    assert m2.done_keys(0) == {(0, 0, 0), (0, 0, 1)}
    assert m2.counts() == {"written": 2, "empty": 1}


def test_scalar_keys_for_per_segment_tasks(tmp_path):
    # e.g. meshing per segment id: key is a bare int (or str), no tuple wrapping
    p = str(tmp_path / "seg.jsonl")
    m = Manifest(p)
    m.reset()
    m.record(0, [(1001, "written"), (1002, "failed"), ("agglo:42", "written")])
    m.close()

    m2 = Manifest(p).load()
    assert m2.is_done(0, 1001)
    assert m2.is_done(0, "agglo:42")
    assert m2.status(0, 1002) == "failed"
    assert m2.done_keys(0) == {1001, 1002, "agglo:42"}


def test_reset_truncates(tmp_path):
    p = str(tmp_path / "prog.jsonl")
    m = Manifest(p)
    m.reset()
    m.record(0, [(5, "written")])
    m.close()
    Manifest(p).reset()
    assert Manifest(p).load().done_keys(0) == set()


def test_tolerates_torn_final_line_and_legacy_fields(tmp_path):
    p = str(tmp_path / "prog.jsonl")
    with open(p, "w") as f:
        f.write('{"group": 0, "key": [0, 0, 0], "status": "written"}\n')
        f.write('{"level": 0, "index": [1, 1, 1], "status": "written"}\n')  # legacy field names
        f.write('{"group": 0, "key": [0, 0, 1], "sta')                      # crash mid-write
    m = Manifest(p).load()
    assert m.is_done(0, (0, 0, 0))
    assert m.is_done(0, (1, 1, 1))        # legacy level/index still read
    assert not m.is_done(0, (0, 0, 1))    # torn line ignored


def test_no_path_is_in_memory_only():
    m = Manifest(None)
    m.reset()
    m.record(0, [(42, "written")])
    assert m.is_done(0, 42)
