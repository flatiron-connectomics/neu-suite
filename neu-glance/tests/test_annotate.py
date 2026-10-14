"""`neu-glance annotate`: a local annotation layer from coordinates the caller supplies.

The checks that matter are where an annotation lands (the zyx -> xyz flip, unit
conversion), since a misplaced annotation is still a valid one.
"""

import json

import numpy as np
import pytest

from neu_vol import convert
from neu_vol.backends.tensorstore import TensorStoreBackend
from neu_vol.profiles import zarr3_create_spec

from neu_glance import cli, layers as _layers


@pytest.fixture
def volume(tmp_path):
    """An anisotropic pyramid, 40x8x8 nm, level 1 a (1, 2, 2) downsample: assuming 2**N
    per level would get it wrong."""
    data = np.zeros((16, 32, 32), dtype=np.uint8)
    data[2:6, 4:8, 4:8] = 3
    src = str(tmp_path / "src.zarr")
    be = TensorStoreBackend.create(
        zarr3_create_spec("local", src, data.shape, str(data.dtype),
                          dimension_names=("z", "y", "x"), chunk=(8, 8, 8)),
        delete_existing=True)
    be.write_region(tuple(slice(0, s) for s in data.shape), data)
    dst = str(tmp_path / "vol.zarr")
    convert(src, dst, voxel_size=(40, 8, 8), kind="segmentation", profile="local",
            chunk=(8, 8, 8), factors=[(1, 2, 2)], max_levels=2, min_dim=8,
            delete_existing=True)
    return dst


def _run(capsys, *argv):
    """Run the subcommand and return ``(layer, stderr)``."""
    assert cli.main(["annotate",*argv]) == 0
    out, err = capsys.readouterr()
    return json.loads(out), err


def _csv(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


# --------------------------------------------------------------------------- #
# geometry and the flip
# --------------------------------------------------------------------------- #
def test_points_from_csv_are_written_xyz(tmp_path, volume, capsys):
    """Columns are read as zyx and written as xyz."""
    path = _csv(tmp_path, "p.csv", "z,y,x\n1,2,3\n4,5,6\n")
    layer, _ = _run(capsys, "--volume", volume, "--points", path)

    assert layer["source"]["url"] == "local://annotations"
    assert [a["point"] for a in layer["annotations"]] == [[3, 2, 1], [6, 5, 4]]
    assert [a["type"] for a in layer["annotations"]] == ["point", "point"]


def test_columns_are_found_by_name_not_position(tmp_path, volume, capsys):
    """Extra columns and any column order are fine."""
    path = _csv(tmp_path, "p.csv",
                "body,x,confidence,z,note,y\n99,3,0.8,1,hi,2\n")
    layer, _ = _run(capsys, "--volume", volume, "--points", path)
    assert layer["annotations"][0]["point"] == [3, 2, 1]


def test_box_corners_are_sorted_but_line_direction_is_kept(tmp_path, volume, capsys):
    """A reversed box would render as nothing; a reversed line is just the other way."""
    boxes = _csv(tmp_path, "b.csv", "z0,y0,x0,z1,y1,x1\n8,8,8,2,2,2\n")
    lines = _csv(tmp_path, "l.csv", "z0,y0,x0,z1,y1,x1\n8,8,8,2,2,2\n")
    layer, _ = _run(capsys, "--volume", volume, "--boxes", boxes, "--lines", lines)

    box, line = layer["annotations"]
    assert box["type"] == "axis_aligned_bounding_box"
    assert (box["pointA"], box["pointB"]) == ([2, 2, 2], [8, 8, 8])
    assert (line["pointA"], line["pointB"]) == ([8, 8, 8], [2, 2, 2])


def test_ellipsoids_carry_centre_and_radii(tmp_path, volume, capsys):
    path = _csv(tmp_path, "e.csv", "z,y,x,rz,ry,rx\n4,5,6,1,2,3\n")
    layer, err = _run(capsys, "--volume", volume, "--ellipsoids", path)
    ann = layer["annotations"][0]
    assert ann["type"] == "ellipsoid"
    assert (ann["center"], ann["radii"]) == ([6, 5, 4], [3, 2, 1])
    # radii are an extent, not a place: they must not be bounds-checked as coordinates
    assert "all inside" in err


def test_one_layer_may_mix_kinds(tmp_path, volume, capsys):
    """Local annotation layers may mix types."""
    path = _csv(tmp_path, "p.csv", "z,y,x\n1,2,3\n")
    layer, _ = _run(capsys, "--volume", volume, "--points", path,
                    "--box", "0,0,0,4,4,4", "--line", "1,1,1,2,2,2")
    assert [a["type"] for a in layer["annotations"]] == [
        "point", "axis_aligned_bounding_box", "line"]


# --------------------------------------------------------------------------- #
# ids, descriptions, segments
# --------------------------------------------------------------------------- #
def test_ids_and_descriptions_come_from_the_csv(tmp_path, volume, capsys):
    path = _csv(tmp_path, "p.csv",
                "id,z,y,x,description\nsyn1,1,2,3,pre T-bar\n,4,5,6,\n")
    layer, _ = _run(capsys, "--volume", volume, "--points", path, "--label", "x")
    first, second = layer["annotations"]
    assert (first["id"], first["description"]) == ("syn1", "pre T-bar")
    assert second["id"] == "x001", "rows without an id are numbered by position"
    assert "description" not in second


def test_segments_are_strings_in_a_per_relationship_array(tmp_path, volume, capsys):
    """uint64 ids stay strings, one list per relationship."""
    path = _csv(tmp_path, "p.csv",
                "z,y,x,segments\n1,2,3,22032970 22032971\n4,5,6,10000000000000000001\n")
    layer, err = _run(capsys, "--volume", volume, "--points", path)
    assert layer["annotations"][0]["segments"] == [["22032970", "22032971"]]
    assert layer["annotations"][1]["segments"] == [["10000000000000000001"]]
    assert "2 annotation(s) linked to body ids" in err


# --------------------------------------------------------------------------- #
# relationships bound to segmentation layers (--link)
# --------------------------------------------------------------------------- #
def test_link_binds_each_relationship_to_its_layer(tmp_path, volume, capsys):
    """Each relationship binds to its own layer. `annotationRelationships` must be plain
    names; the viewer rejects (name, layer) pairs with "expected a string"."""
    path = _csv(tmp_path, "l.csv",
                "z0,y0,x0,z1,y1,x1,segments:gt,segments:vi\n"
                "1,2,3,4,5,6,2034 2153,139245\n")
    layer, err = _run(capsys, "--volume", volume, "--lines", path,
                      "--link", "gt=gt_v2_relabel", "--link", "vi=etched")
    assert layer["annotationRelationships"] == ["gt", "vi"]
    assert all(isinstance(r, str) for r in layer["annotationRelationships"])
    assert layer["linkedSegmentationLayer"] == {"gt": "gt_v2_relabel", "vi": "etched"}
    assert layer["annotations"][0]["segments"] == [["2034", "2153"], ["139245"]]
    assert "gt -> gt_v2_relabel, vi -> etched" in err


def test_segment_lists_follow_the_LINK_order_not_the_column_order(tmp_path, volume, capsys):
    """Neuroglancer pairs segment lists with relationships by position, so the lists
    follow the --link order."""
    path = _csv(tmp_path, "p.csv", "z,y,x,segments:vi,segments:gt\n1,2,3,7,9\n")
    layer, _ = _run(capsys, "--volume", volume, "--points", path,
                    "--link", "gt=G", "--link", "vi=V")
    assert layer["annotationRelationships"] == ["gt", "vi"]
    assert layer["annotations"][0]["segments"] == [["9"], ["7"]]


def test_an_empty_relationship_is_an_empty_list_not_a_shifted_one(tmp_path, volume, capsys):
    """Dropping an empty list would shift the next relationship's ids into its slot."""
    path = _csv(tmp_path, "p.csv", "z,y,x,segments:gt,segments:vi\n1,2,3,,5\n")
    layer, _ = _run(capsys, "--volume", volume, "--points", path,
                    "--link", "gt=G", "--link", "vi=V")
    assert layer["annotations"][0]["segments"] == [[], ["5"]]


def test_a_bare_link_binds_the_plain_segments_column(tmp_path, volume, capsys):
    path = _csv(tmp_path, "p.csv", "z,y,x,segments\n1,2,3,42\n")
    layer, _ = _run(capsys, "--volume", volume, "--points", path, "--link", "seg")
    assert layer["annotationRelationships"] == ["segments"]
    assert layer["linkedSegmentationLayer"] == {"segments": "seg"}
    assert layer["annotations"][0]["segments"] == [["42"]]


def test_without_link_the_layer_is_unchanged(tmp_path, volume, capsys):
    path = _csv(tmp_path, "p.csv", "z,y,x,segments\n1,2,3,42\n")
    layer, _ = _run(capsys, "--volume", volume, "--points", path)
    assert "annotationRelationships" not in layer
    assert "linkedSegmentationLayer" not in layer
    assert layer["annotations"][0]["segments"] == [["42"]]


def test_a_linked_layer_may_have_relationships_no_annotation_uses(tmp_path, volume, capsys):
    """A linked relationship no annotation uses gets an empty list."""
    path = _csv(tmp_path, "p.csv", "z,y,x,segments:gt\n1,2,3,5\n")
    layer, _ = _run(capsys, "--volume", volume, "--points", path,
                    "--link", "gt=G", "--link", "vi=V")
    assert layer["annotations"][0]["segments"] == [["5"], []]


@pytest.mark.parametrize("csv,extra,match", [
    ("z,y\n1,2\n", [], r"needs column\(s\) x"),
    # what a spreadsheet does to a body id
    ("z,y,x,segments\n1,2,3,1.23e+18\n", [], "not a whole number"),
    ("id,z,y,x\nsame,1,2,3\nsame,4,5,6\n", [], "duplicate annotation id"),
    ("z,y,x,segments:gt\n1,2,3,42\n", [], "no --link says which layer"),
    ("z,y,x,segments:gt,segments:vi\n1,2,3,1,2\n", ["--link", "gt=G"],
     "vi appear in the input but no --link"),
    ("z,y,x\n1,2,3\n", ["--link", "gt=A", "--link", "gt=B"], "twice"),
], ids=["missing-column", "spreadsheet-id", "duplicate-id", "unlinked-columns",
        "one-unlinked", "linked-twice"])
def test_a_bad_input_is_refused(tmp_path, volume, csv, extra, match):
    path = _csv(tmp_path, "p.csv", csv)
    with pytest.raises(SystemExit, match=match):
        cli.main(["annotate", "--volume", volume, "--points", path, *extra])


# --------------------------------------------------------------------------- #
# units
# --------------------------------------------------------------------------- #
def test_scale_uses_the_real_per_level_voxel_sizes(tmp_path, volume, capsys):
    """With factors (1, 2, 2), a scale-1 voxel is 1 level-0 voxel in z and 2 in y and x."""
    path = _csv(tmp_path, "p.csv", "z,y,x\n3,4,5\n")
    layer, err = _run(capsys, "--volume", volume, "--points", path, "--scale", "1")
    assert layer["annotations"][0]["point"] == [10, 8, 3]      # xyz: x2, x2, x1
    assert "x(1, 2, 2)" in err


def test_nm_divides_by_the_level0_voxel_size(tmp_path, volume, capsys):
    """40x8x8 nm, so z converts differently from y and x."""
    path = _csv(tmp_path, "p.csv", "z,y,x\n400,80,160\n")
    layer, _ = _run(capsys, "--volume", volume, "--points", path, "--nm")
    assert layer["annotations"][0]["point"] == [20, 10, 10]    # xyz


def test_scale_without_a_volume_is_refused(tmp_path):
    path = _csv(tmp_path, "p.csv", "z,y,x\n1,2,3\n")
    with pytest.raises(SystemExit, match="--scale needs --volume"):
        cli.main(["annotate","--points", path, "--scale", "2",
                  "--voxel-size", "8,8,8"])


def test_coordinates_outside_the_volume_are_reported(tmp_path, volume, capsys):
    """The only check against a unit mistake, so it warns and suggests --scale / --nm."""
    path = _csv(tmp_path, "p.csv", "z,y,x\n1,2,3\n99999,2,3\n")
    _, err = _run(capsys, "--volume", volume, "--points", path)
    assert "WARNING: 1 annotation(s) fall outside" in err
    assert "--scale" in err and "--nm" in err


def test_without_a_frame_the_layer_is_unitless_and_says_so(tmp_path, capsys):
    path = _csv(tmp_path, "p.csv", "z,y,x\n1,2,3\n")
    layer, err = _run(capsys, "--points", path)
    assert layer["source"]["transform"]["outputDimensions"]["x"] == [1, ""]
    assert "unitless" in err


def test_the_frame_comes_from_the_volume(tmp_path, volume, capsys):
    path = _csv(tmp_path, "p.csv", "z,y,x\n1,2,3\n")
    layer, _ = _run(capsys, "--volume", volume, "--points", path)
    dims = layer["source"]["transform"]["outputDimensions"]
    assert dims["z"] == [40e-9, "m"] and dims["x"] == [8e-9, "m"]


# --------------------------------------------------------------------------- #
# plumbing
# --------------------------------------------------------------------------- #
def test_inline_flags_need_the_right_count(volume):
    with pytest.raises(SystemExit, match="--box needs 6"):
        cli.main(["annotate","--volume", volume, "--box", "1,2,3"])


def test_nothing_to_annotate_is_an_error(volume):
    with pytest.raises(SystemExit, match="nothing to annotate"):
        cli.main(["annotate","--volume", volume])


def test_out_writes_the_layer(tmp_path, volume, capsys):
    out = tmp_path / "layer.json"
    assert cli.main(["annotate","--volume", volume, "--point", "1,2,3",
                     "--out", str(out), "--name", "syn", "--color", "#ff0044"]) == 0
    layer = json.loads(out.read_text())
    assert layer["name"] == "syn" and layer["annotationColor"] == "#ff0044"
    assert layer["tab"] == "annotations", "so the panel opens on the clickable list"


def test_the_layer_composes_with_gen(tmp_path, volume, capsys):
    """`gen --layer` inlines the emitted layer."""
    from neu_glance.state import parse_url

    out = tmp_path / "layer.json"
    cli.main(["annotate", "--volume", volume, "--point", "8,8,8",
              "--out", str(out)])
    capsys.readouterr()
    assert cli.main(["gen", "--seg", volume, "--layer", str(out)]) == 0
    state = parse_url(capsys.readouterr().out.strip().splitlines()[-1])
    kinds = [ly["type"] for ly in state["layers"]]
    assert kinds == ["segmentation", "annotation"]
    assert state["layers"][1]["annotations"][0]["point"] == [8, 8, 8]


def test_the_cli_column_spec_matches_the_layers_one():
    """cli.py keeps its own copy (importing `layers` would load neu-vol for --help)."""
    assert cli._ANN_CSV_COLUMNS == _layers.CSV_COLUMNS
    assert {k for _p, k in cli._ANN_FLAGS} == set(_layers.KINDS)


def test_occupancy_boxes_still_go_through_the_shared_builder(tmp_path, volume, capsys):
    """`bboxes` builds its boxes through the same local-layer builder."""
    assert cli.main(["bboxes", volume, "--no-tighten"]) == 0
    layer = json.loads(capsys.readouterr().out)
    assert layer["source"]["url"] == "local://annotations"
    ann = layer["annotations"][0]
    assert ann["type"] == "axis_aligned_bounding_box" and ann["id"] == "r00"
    assert ann["pointA"][0] <= ann["pointB"][0]
