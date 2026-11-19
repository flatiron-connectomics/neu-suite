"""Reading a volume's segment properties: one read to lists, and the DataFrame view.

Fixtures are written by hand, as a published source lays them out: tags stored as indices
into a vocabulary, every property paired with `ids` by position.
"""

import json

import numpy as np
import pandas as pd
import pytest

from neu_vol.segment_properties import (SegmentPropertiesProblem, read_segment_properties,
                                        segment_properties_frame)

ROWS = {
    "1": ("KCa_1(L)", ["group:KCa", "side:L", "post:MB(L)", "post:CA(L)"]),
    "2": ("KCa_2(R)", ["group:KCa", "side:R", "post:MB(R)"]),
    "3": ("KCb_1(L)", ["group:KCb", "side:L", "fragment"]),
    "4": ("MBON01(L)", ["side:L"]),
    "140071381": ("PN_GL10(R)", ["group:PN", "truncated", "fragment"]),
}


def _volume(tmp_path, rows=ROWS, *, numbers=None, strings=None, label=True):
    vocabulary = sorted({t for _, tags in rows.values() for t in tags})
    root = tmp_path / "vol"
    (root / "props").mkdir(parents=True, exist_ok=True)
    (root / "info").write_text(json.dumps({"@type": "neuroglancer_multiscale_volume",
                                           "segment_properties": "props", "scales": []}))
    properties = [{"id": "tags", "type": "tags", "tags": vocabulary,
                   "values": [[vocabulary.index(t) for t in v[1]] for v in rows.values()]}]
    if label:
        properties.insert(0, {"id": "instance", "type": "label",
                              "values": [v[0] for v in rows.values()]})
    for name, values in (numbers or {}).items():
        properties.append({"id": name, "type": "number", "data_type": "uint32",
                           "values": values})
    for name, values in (strings or {}).items():
        properties.append({"id": name, "type": "string", "values": values})
    (root / "props" / "info").write_text(json.dumps({
        "@type": "neuroglancer_segment_properties",
        "inline": {"ids": list(rows), "properties": properties}}))
    return str(root)


def test_everything_comes_back_from_ONE_read(tmp_path):
    read = read_segment_properties(_volume(tmp_path, numbers={"pre": [9, 8, 7, 6, 5]},
                                           strings={"parsed_type": ["KCa", "KCa", "KCb",
                                                                    "MBON01", ""]}))
    assert read["ids"] == list(ROWS)
    assert read["labels"][0] == "KCa_1(L)" and read["label_id"] == "instance"
    assert read["numbers"]["pre"] == [9, 8, 7, 6, 5]
    assert read["strings"]["parsed_type"][-1] == ""
    assert read["tags"][2] == ("group:KCb", "side:L", "fragment")


def test_without_a_label_property_segments_are_labelled_by_id(tmp_path):
    read = read_segment_properties(_volume(tmp_path, label=False))
    assert read["labels"] == read["ids"] and read["label_id"] is None


@pytest.mark.parametrize("break_it, match", [
    (lambda info, props: info.pop("segment_properties"), "declares no 'segment_properties'"),
    (lambda info, props: props.update({"@type": "neuroglancer_annotations_v1"}),
     "not 'neuroglancer_segment_properties'"),
    (lambda info, props: props.pop("inline"), "no `inline` block"),
    (lambda info, props: props["inline"]["properties"][0]["values"].pop(),
     "pairs them by position"),
], ids=["undeclared", "wrong-type", "not-inline", "length-mismatch"])
def test_an_unreadable_source_is_refused(tmp_path, break_it, match):
    vol = _volume(tmp_path)
    info = json.loads((tmp_path / "vol" / "info").read_text())
    props = json.loads((tmp_path / "vol" / "props" / "info").read_text())
    break_it(info, props)
    (tmp_path / "vol" / "info").write_text(json.dumps(info))
    (tmp_path / "vol" / "props" / "info").write_text(json.dumps(props))
    with pytest.raises(SegmentPropertiesProblem, match=match):
        read_segment_properties(vol)


def test_the_frame_is_indexed_by_uint64_body(tmp_path):
    df = segment_properties_frame(_volume(tmp_path, numbers={"pre": [9, 8, 7, 6, 5]},
                                          strings={"parsed_type": ["KCa", "KCa", "KCb",
                                                                   "MBON01", ""]}))
    assert df.index.name == "body" and df.index.dtype == np.uint64
    assert df.index[-1] == 140071381
    assert df.loc[np.uint64(1), "instance"] == "KCa_1(L)"
    assert df["pre"].tolist() == [9, 8, 7, 6, 5]
    assert pd.isna(df.loc[np.uint64(140071381), "parsed_type"]), "empty string is missing"


def test_tags_split_into_facet_and_flag_columns(tmp_path):
    df = segment_properties_frame(_volume(tmp_path))
    # single-valued facet: a string column, missing where absent
    assert df["tag:group"].tolist()[:3] == ["KCa", "KCa", "KCb"]
    assert pd.isna(df.loc[np.uint64(4), "tag:group"])
    # a facet some segment carries twice: tuples for everyone
    assert df.loc[np.uint64(1), "tag:post"] == ("MB(L)", "CA(L)")
    assert df.loc[np.uint64(4), "tag:post"] == ()
    # bare tags: boolean flags
    assert df["flag:fragment"].tolist() == [False, False, True, False, True]
    assert df["flag:truncated"].dtype == bool
    assert df.loc[np.uint64(3), "tags"] == ("group:KCb", "side:L", "fragment")


def test_the_frame_accepts_an_earlier_read(tmp_path):
    read = read_segment_properties(_volume(tmp_path))
    assert segment_properties_frame(read).equals(segment_properties_frame(_volume(tmp_path)))
