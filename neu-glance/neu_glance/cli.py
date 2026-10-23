"""``neu-glance`` — the command line over neu_glance.

Argparse, like the other neu-* commands. Heavy imports stay inside each subcommand, so
``--help`` is fast. The subcommands, by what they produce:

- ``gen`` composes a **state** from volumes, annotation sources and layer files
- ``annotate`` makes a **layer** from coordinates or a CSV
- ``bboxes`` makes a **layer** from a volume's occupancy
- ``serve`` hosts arrays here and **runs a viewer** on them
- ``parse`` reads a URL back into its state JSON
- ``shaders`` lists or prints the built-in shaders

The three producers share one output stage: ``--format {layer,state,url}`` and ``--into``
(merge into an existing state). ``serve`` produces a running server instead, whose sources
are valid only in this process, so it has no output stage.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from . import __version__

_ANN_FLAGS = (("points", "point"), ("boxes", "box"), ("lines", "line"),
              ("ellipsoids", "ellipsoid"))

#: A copy of ``layers.CSV_COLUMNS``, so building the parser imports nothing heavy. A test
#: checks the two agree.
_ANN_CSV_COLUMNS = {
    "point": (("z", "y", "x"),),
    "box": (("z0", "y0", "x0"), ("z1", "y1", "x1")),
    "line": (("z0", "y0", "x0"), ("z1", "y1", "x1")),
    "ellipsoid": (("z", "y", "x"), ("rz", "ry", "rx")),
}

_LAYOUTS = ("4panel", "xy", "yz", "xz", "xy-3d", "yz-3d", "xz-3d", "3d")
_DEFAULT_VIEWER = "https://neuroglancer-demo.appspot.com/"
_SHADER_NAMES = ("synapse",)
#: A copy of the ``shaders.IMAGE_SHADERS`` names, for the same reason. A test checks them.
_IMAGE_SHADER_NAMES = ("grayscale", "probability", "colormap", "rgb", "none")

#: ``gen``'s per-segmentation display flags: (flag, state key, help, range). Neuroglancer
#: ignores an out-of-range value rather than refusing it, so the range is checked here.
_SEG_DISPLAY_FLAGS = (
    ("--silhouette", "meshSilhouetteRendering",
     "3D silhouette (0-10): faces turned towards the viewer fade, leaving the outline; "
     "0 is off, larger is stronger"),
    ("--opacity-2d", "selectedAlpha", "opacity of the selected segments in the 2D panels"),
    ("--opacity-3d", "objectAlpha", "opacity of the meshes in the 3D panel"),
)
_SEG_DISPLAY_RANGE = {"meshSilhouetteRendering": (0.0, 10.0), "selectedAlpha": (0.0, 1.0),
                      "objectAlpha": (0.0, 1.0)}


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _ftriple(value, name):
    """'z,y,x' -> a 3-tuple of floats, or None. Voxel sizes are not always integral."""
    if value is None:
        return None
    parts = tuple(float(v) for v in value.replace("x", ",").split(","))
    if len(parts) != 3:
        raise SystemExit(f"--{name} needs 3 comma-separated values, got {value!r}")
    return parts


def _sextuple(value, name):
    """'z0,y0,x0,z1,y1,x1' -> ``(lo, hi)`` in zyx voxels."""
    parts = tuple(int(v) for v in str(value).replace(" ", "").split(","))
    if len(parts) != 6:
        raise SystemExit(f"--{name} needs 6 comma-separated whole numbers "
                         f"(z0,y0,x0,z1,y1,x1), got {value!r}")
    return parts[:3], parts[3:]


def _write_text(path: str, text: str) -> None:
    """Write to a local path or an object store, uniformly.

    Through ``location`` rather than ``open()`` so ``--out s3://…`` works: the file driver
    creates parent directories and the s3 driver bootstraps credentials.
    """
    from neu_vol.location import write_bytes

    write_bytes(path, text.encode())


def _read_ann_file(path: str) -> str:
    """CSV text from a path, an object-store URL, or ``-`` for stdin."""
    if path == "-":
        return sys.stdin.read()
    from neu_vol.location import read_bytes

    data = read_bytes(path)
    if data is None:
        raise SystemExit(f"could not read {path}")
    return data.decode()


def _inline_records(args) -> list[dict]:
    """Records from the repeatable inline flags, in the order they were given."""
    records = []
    for _plural, kind in _ANN_FLAGS:
        for value in getattr(args, kind) or []:
            n = 3 * len(_ANN_CSV_COLUMNS[kind])
            parts = [p for p in value.replace(" ", "").split(",") if p]
            if len(parts) != n:
                raise SystemExit(
                    f"--{kind} needs {n} comma-separated numbers "
                    f"({', '.join(c for g in _ANN_CSV_COLUMNS[kind] for c in g)}), "
                    f"got {value!r}")
            try:
                flat = [float(p) for p in parts]
            except ValueError:
                raise SystemExit(f"--{kind} {value!r}: not all numbers") from None
            records.append({"kind": kind,
                            "coords": tuple(tuple(flat[i:i + 3])
                                            for i in range(0, n, 3))})
    return records


def _volume_frame(volume: str | None, voxel_size: str | None):
    """``(voxel_size_zyx, units, level-0 spatial shape or None)`` for a layer's frame.

    An annotation layer whose ``outputDimensions`` disagree with the volume loads with every
    annotation misplaced. Reads ``info``-level metadata only (not ``describe``, which opens
    every level).
    """
    voxel = _ftriple(voxel_size, "voxel-size")
    if not volume:
        return voxel, ("nm" if voxel else None), None

    from neu_vol.detect import detect_backend
    from neu_vol.source_metadata import read_source_metadata

    from .sources import volume_extent

    fmt = detect_backend(volume.rstrip("/"))
    if fmt is None:
        raise SystemExit(f"no volume found at {volume}")
    meta = read_source_metadata({"backend": fmt, "path": volume.rstrip("/")}) or {}
    extent = volume_extent(volume.rstrip("/"), fmt)
    return (voxel or (meta.get("voxel_size") and tuple(meta["voxel_size"])),
            "nm" if voxel else meta.get("units"),
            tuple(extent[0]) if extent else None)


def _level0_factor(volume: str, scale: int) -> tuple[int, ...]:
    """How many level-0 voxels one scale-``scale`` voxel spans, per axis.

    From each level's recorded voxel size, never ``2**scale``: pyramids may be anisotropic.
    """
    from neu_vol.detect import detect_backend
    from neu_vol.scales import level_factor
    from neu_vol.source_metadata import location_spec, read_level_voxel_sizes

    fmt = detect_backend(volume.rstrip("/"))
    if fmt is None:
        raise SystemExit(f"--scale {scale} needs the volume's per-level voxel sizes and "
                         f"nothing at {volume} looks like a volume")
    try:
        return level_factor(read_level_voxel_sizes(location_spec(volume.rstrip("/"), fmt)),
                            scale)
    except ValueError as e:
        raise SystemExit(f"--scale {scale}: {volume} {e}") from None


# --------------------------------------------------------------------------- #
# the shared output stage
# --------------------------------------------------------------------------- #
def _add_output_flags(q: argparse.ArgumentParser, *, formats: tuple[str, ...],
                      default: str) -> None:
    """``--format``, ``--out`` and the state-shaping flags every producer shares."""
    q.add_argument("--format", dest="fmt", choices=formats, default=None,
                   help=f"what to emit (default: {default}). 'layer' is a bare layer "
                        f"object for pasting into a state's `layers` array; 'state' is a "
                        f"whole state as JSON, for neuroglancer's {{}} editor; 'url' is a "
                        f"link carrying that state.")
    q.add_argument("--out", action="append", default=None, metavar="PATH_OR_URL",
                   help="write it here instead of stdout (local or s3://…). Repeatable; "
                        "'-' is stdout, so --out link.txt --out - does both")
    q.add_argument("--into", default=None, metavar="URL_OR_JSON",
                   help="add the layers to an EXISTING state rather than building a new "
                        "one — a neuroglancer URL or a state JSON file. The state's own "
                        "dimensions, position and zoom are kept, so your view does not "
                        "move; a layer whose name is already taken is renamed and reported. "
                        "Implies --format url unless --format says otherwise. With nothing "
                        "to add, `gen --into` re-emits the state itself — a view file "
                        "straight to a link. Quote a URL "
                        "with SINGLE quotes: inside double quotes bash expands '!', which "
                        "neuroglancer uses both after '#' and for hidden segments.")
    if "state" in formats or "url" in formats:
        q.add_argument("--layout", default=None, choices=_LAYOUTS,
                       help="neuroglancer panel layout (default: 4panel, or whatever "
                            "--into already had)")
        q.add_argument("--position", default=None, metavar="Z,Y,X",
                       help="where to put the crosshair, in level-0 voxels")
        q.add_argument("--position-order", choices=("zyx", "xyz"), default="zyx",
                       help="axis order of --position. Everything in this package is zyx, "
                            "but neuroglancer DISPLAYS xyz — so pass xyz to use numbers "
                            "copied straight out of the viewer (default: zyx)")
        q.add_argument("--cross-section-scale", type=float, default=None, metavar="S",
                       help="zoom of the 2D panels: nm per screen pixel, smaller is closer")
        q.add_argument("--projection-scale", type=float, default=None, metavar="S",
                       help="zoom of the 3D panel")
        q.add_argument("--hide-slices", action="store_true",
                       help="set showSlices false, hiding the cross-section planes inside "
                            "the 3D panel — the usual thing to want when the link is about "
                            "meshes or skeletons. Omitted from the state unless passed")
        q.add_argument("--hide", action="append", default=None, metavar="LAYER_NAME",
                       help="add this layer switched off, to turn on in the viewer "
                            "(repeatable). For layers that would crowd the view if all were "
                            "shown at once")
        q.add_argument("--select", default=None, metavar="LAYER_NAME",
                       help="open the side panel on this layer")
        q.add_argument("--select-last", action="store_true",
                       help="the same, on whichever layer was added last")
        q.add_argument("--viewer", default=_DEFAULT_VIEWER,
                       help=f"viewer base URL for --format url (default: "
                            f"{_DEFAULT_VIEWER})")
        q.add_argument("--expand", action="store_true",
                       help="with --into a state JSON on s3:// or gs:// and --format url, put "
                            "the whole state in the link. By default such a link only names "
                            "the file (<viewer>#!s3://.../state.json), which the viewer fetches, "
                            "so it stays short and follows later rewrites of the file")
    q.set_defaults(default_format=default)


#: Whether stderr was a terminal when `main` started. Asking later is wrong: inside `main`,
#: `quiet_store_logs` has swapped fd 2 for a pipe, so stderr never looks like a terminal.
_STDERR_IS_TTY: bool | None = None


def _isatty(stream) -> bool:
    return bool(getattr(stream, "isatty", lambda: False)())


def _yellow(text: str, stream=None) -> str:
    """``text`` in yellow when ``stream`` (default stderr) is a terminal, else unchanged.

    Plain for pipes, logs and tests, and whenever ``NO_COLOR`` is set (no-color.org).
    """
    import os

    if stream is not None:
        tty = _isatty(stream)
    else:
        tty = _STDERR_IS_TTY if _STDERR_IS_TTY is not None else _isatty(sys.stderr)
    if os.environ.get("NO_COLOR") or not tty:
        return text
    return f"\033[93m{text}\033[0m"      # 93, bright yellow: plain 33 reads as dark ochre


def _load_into(source: str) -> dict:
    """The ``--into`` state, or exit with the reason — shared by every command taking one.

    The commonest failure is a URL mangled by bash inside double quotes, so the message is
    highlighted.
    """
    from .state import StateProblem, load_state

    try:
        return load_state(source)
    except (StateProblem, ValueError) as e:
        raise SystemExit(_yellow(str(e))) from None


def _is_cloud_state(source: str) -> bool:
    """A state file on an object store, which a viewer can fetch from ``#!<url>``."""
    return source.startswith(("s3://", "gs://")) and "#!" not in source


def _resolve_format(args) -> str:
    """The format actually being emitted.

    ``--format`` defaults to ``None``, so ``--into`` can promote the default ``layer`` to
    ``url`` while still refusing an explicit ``--format layer``. Both :func:`_with_volume` and
    :func:`_emit` use this, so they agree.
    """
    into = getattr(args, "into", None)
    if args.fmt is None:
        return "url" if into else args.default_format
    if into and args.fmt == "layer":
        raise SystemExit(
            "--into merges layers into a state, so --format layer has nothing to emit. "
            "Use --format url (what --into gives by default) or state.")
    return args.fmt


def _emit(args, layers: list[dict], *, voxel=None, units=None, frame=None,
          selected: str | None = None) -> int:
    """Serialize ``layers`` per ``--format``/``--into``/``--out``. Returns an exit code.

    Shared by all three producers.
    """
    from .layers import render
    from .state import LONG_URL, merge_into, state_url

    err = sys.stderr
    into = getattr(args, "into", None)
    fmt = _resolve_format(args)

    if fmt == "layer":
        if len(layers) != 1:
            raise SystemExit(
                f"--format layer emits ONE layer object and there are {len(layers)}. Use "
                f"--format state to get all of them in a state.")
        text = render(layers[0])
    else:
        hide = getattr(args, "hide", None) or []
        unknown = sorted(set(hide) - {lyr["name"] for lyr in layers})
        if unknown:
            raise SystemExit(f"--hide names no layer: {', '.join(unknown)}. The layers are "
                             f"{', '.join(lyr['name'] for lyr in layers)}")
        for lyr in layers:
            if lyr["name"] in hide:
                lyr["visible"] = False
        selected_name = getattr(args, "select", None) or (
            layers[-1]["name"] if getattr(args, "select_last", False) and layers
            else selected)
        position = _ftriple(getattr(args, "position", None), "position")
        if position is not None and getattr(args, "position_order", "zyx") == "xyz":
            position = tuple(reversed(position))
        shaping = dict(
            layout=getattr(args, "layout", None),
            position_zyx=position,
            cross_section_scale=getattr(args, "cross_section_scale", None),
            projection_scale=getattr(args, "projection_scale", None),
            selected=selected_name,
            show_slices=False if getattr(args, "hide_slices", False) else None,
        )
        if into and fmt == "url" and _is_cloud_state(into) and not getattr(args, "expand",
                                                                            False):
            changed = bool(layers or hide) or any(v is not None for v in shaping.values())
            if changed:
                raise SystemExit(
                    f"a link to {into} would show the file as it is, without the layers or view "
                    f"options given here. Write the changed state with --format state --out "
                    f"<file> and link that, or pass --expand to put it in the link.")
            _load_into(into)                      # refuse a file that is not a state
            text = f"{getattr(args, 'viewer', _DEFAULT_VIEWER).rstrip('/')}/#!{into}"
            print(f"  a link to the state file, which the viewer fetches; it must be readable "
                  f"anonymously, with CORS (--expand puts the state in the link)", file=err)
            _deliver(args.out, text)
            return 0
        if into:
            base = _load_into(into)
            state, notes = merge_into(base, layers, **shaping)
            print(f"added {len(layers)} layer(s) to {len(base.get('layers') or [])} "
                  f"already in {into if '#!' not in into else 'the given URL'}"
                  + ("" if layers else " — re-emitting it as is"), file=err)
            print(f"  kept the existing dimensions, position and zoom", file=err)
            for note in notes:
                print(f"  {note}", file=err)
        else:
            # Say where the view opens: with no frame, neuroglancer opens at the origin corner
            # fully zoomed in, which looks like a broken layer.
            if position is not None:
                print(f"  position {tuple(position)} zyx "
                      f"(given as {getattr(args, 'position_order', 'zyx')})", file=err)
            elif frame:
                print(f"  view centred on the {tuple(int(v) for v in frame[0])} zyx frame, "
                      f"zoomed to fit it", file=err)
            else:
                print(f"  no volume or annotation established a frame — neuroglancer will "
                      f"open at the origin, zoomed in. Pass --position to place the view.",
                      file=err)
            from .state import build_state

            state, warning = build_state(
                layers, voxel_size_zyx=voxel, units=units, frame=frame,
                **{k: v for k, v in shaping.items() if k != "layout"},
                layout=shaping["layout"] or "4panel")
            if warning:
                print(f"  WARNING: {warning}", file=err)
        text = (render(state) if fmt == "state"
                else state_url(state, getattr(args, "viewer", _DEFAULT_VIEWER)))
        if fmt == "url" and len(text) > LONG_URL:
            print(f"  note: the URL is {len(text)} characters — long enough that some "
                  f"clients will wrap or truncate it. --format state avoids that.",
                  file=err)

    _deliver(args.out, text)
    return 0


def _deliver(outs: list[str] | None, text: str) -> None:
    """``text`` to each ``--out`` (``-`` is stdout), or to stdout when none is given."""
    if isinstance(outs, str):
        outs = [outs]
    for out in outs or ["-"]:
        if out == "-":
            print(text)
        else:
            _write_text(out, text + "\n")
            print(f"wrote {out}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="neu-glance",
        description="Neuroglancer states, layers and links.\n\n"
                    "Everything a viewer consumes and nothing that produces data: the "
                    "volumes come from neu-vol and the annotation sources from "
                    "neu-mark, and neither of those knows a viewer exists.\n\n"
                    "Neuroglancer keeps its whole state in the URL fragment, so a link IS "
                    "the state. Everything after '#!' stays in the browser and is never "
                    "sent to a server — but the whole state travels in the URL, so a large "
                    "inline annotation layer makes for a long one.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"neu-glance {__version__}")
    p.add_argument("--store-logs", action="store_true",
                   help="do not filter the store's own logging. The S3 driver emits two "
                        "AuthCredentialsProvider lines per prefix at ERROR severity that "
                        "are not failures — they record the providers that missed before "
                        "the environment one succeeded — so they are filtered by default. "
                        "PERMISSION_DENIED and AccessDenied are never filtered")
    sub = p.add_subparsers(dest="command", required=True, metavar="COMMAND")

    # --- gen ----------------------------------------------------------------
    q = sub.add_parser(
        "gen", help="a viewer state from volumes, annotation sources and layer files",
        description="Compose a neuroglancer state and emit it as a link or as JSON.\n\n"
                    "Reads the volumes to get the source scheme and the coordinate space "
                    "right, which is the part that fails silently by hand: a `dimensions` "
                    "block that disagrees with the data loads fine and puts every layer in "
                    "the wrong place.\n\n"
                    "  neu-glance bboxes s3://.../gt_v2 --label gt --out gt.json\n"
                    "  neu-glance gen --image s3://.../em --seg s3://.../gt_v2 \\\n"
                    "      --layer gt.json --segments 1,2,3 --layout xy-3d\n\n"
                    "URL or JSON to stdout, summary to stderr. Reads only.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    q.add_argument("--image", action="append", metavar="VOLUME",
                   help="an image volume (repeatable). Drawn beneath the segmentations")
    q.add_argument("--seg", action="append", metavar="VOLUME",
                   help="a segmentation volume (repeatable)")
    q.add_argument("--segments", action="append", metavar="IDS",
                   help="comma-separated segment ids to select, applied to the --seg "
                        "volumes in order. Repeat for a second segmentation")
    q.add_argument("--all-segments", action="append", default=None, metavar="LAYER_NAME",
                   help="select every segment of this --seg layer that its "
                        "segment_properties lists, so its meshes show in 3D as soon as "
                        "the layer is on (repeatable). For a few named regions, such as "
                        "ROI layers, not a segmentation of thousands of bodies")
    q.add_argument("--annotations", action="append", metavar="SOURCE",
                   help="a precomputed ANNOTATION source — what `neu-mark "
                        "annotation-source` writes (repeatable). Added as its own layer, "
                        "because unlike mesh and skeletons an annotation source is never "
                        "named in a volume's info. Its relationships are bound to the first "
                        "--seg layer, which is what makes 'only this body's synapses' work. "
                        "The synapse shader's show_pre / show_post checkboxes show either end "
                        "alone, and the Annotations tab toggles the filter per relationship")
    q.add_argument("--annotation-shader", default=None, metavar="NAME_OR_PATH",
                   help="GLSL for the annotation layers: a built-in name ("
                        + ", ".join(_SHADER_NAMES) + "), a file, or 'none'. Default picks a "
                        "built-in whose properties the source declares, since a shader "
                        "naming an absent property fails to compile and the layer then "
                        "draws nothing")
    q.add_argument("--no-filter-by-segmentation", dest="filter_by_segmentation",
                   action="store_false",
                   help="show every annotation instead of only those on the selected "
                        "segments. By default the filter is ON, so the annotations follow "
                        "whatever you select — which means a link with no --segments opens "
                        "showing none until you pick a body. Toggle it per relationship in "
                        "the layer's ANNOTATIONS tab. The relationships stay bound either "
                        "way")
    q.add_argument("--layer", action="append", metavar="PATH_OR_URL",
                   help="a JSON file holding a layer, or a state whose layers are taken — "
                        "e.g. the output of `neu-glance bboxes` (repeatable)")
    q.add_argument("--image-opacity", type=float, default=None, metavar="F",
                   help="opacity for the --image layers")
    for flag, key, what in _SEG_DISPLAY_FLAGS:
        q.add_argument(flag, action="append", default=None, metavar="LAYER_NAME=VALUE",
                       help=f"{what} of one --seg layer (repeatable; sets {key})")
    q.add_argument("--color", action="append", default=None, metavar="LAYER_NAME=COLOR",
                   help="colour of every segment of one --seg layer that has no colour of its "
                        "own: a name, #rrggbb (repeatable; sets segmentDefaultColor). Without "
                        "it each segment gets its hash colour")
    q.add_argument("--color-by", action="append", default=None,
                   metavar="LAYER_NAME=PROP[,PROP...]",
                   help="colour each segment of one --seg layer by a segment property, from "
                        "its segment_properties: a tag facet (col, group), a flag tag, or "
                        "label. A segment takes the first property it has a value for, so "
                        "col,group colours by column and falls back to group. Each value gets "
                        "a fixed colour from neuroglancer's segment palette; segments with "
                        "none keep their hash colour. Written as segmentColors, so it is fixed "
                        "when the state is made (repeatable)")
    q.add_argument("--no-pick", action="append", default=None, metavar="LAYER_NAME",
                   help="make clicks go through this --seg layer (repeatable; sets pick "
                        "false), for a shell enclosing what you want to select")
    q.add_argument("--config", default=None, metavar="TOML",
                   help="site display defaults: volume bounds, annotation shader controls, "
                        "and per-layer silhouette, opacities, hide and all_segments by layer "
                        "name (see neu_glance.config). Read only when named; a flag here "
                        "wins over the file")
    q.add_argument("--voxel-size", default=None, metavar="Z,Y,X",
                   help="level-0 voxel size in nm, overriding what the volumes record. "
                        "Required when every layer comes from --layer, since a layer file "
                        "carries its own frame but does not establish the viewer's")
    _add_output_flags(q, formats=("url", "state"), default="url")
    q.set_defaults(func=cmd_gen)

    # --- annotate -----------------------------------------------------------
    q = sub.add_parser(
        "annotate", help="a layer of annotations from coordinates you give",
        description="Build a neuroglancer annotation layer from points, boxes, lines or "
                    "ellipsoids you already have — a synapse table, an ROI list, a few "
                    "coordinates typed by hand.\n\n"
                    "The annotations are LOCAL, carried inline in the state, and that is "
                    "the point: only local annotations appear in the Annotations tab, which "
                    "is what makes them clickable and steppable. For a whole volume's worth "
                    "of synapses, write a precomputed source with `neu-mark "
                    "annotation-source` and add it with `neu-glance gen --annotations`.\n\n"
                    "Coordinates are zyx and in level-0 voxels unless --scale or --nm says "
                    "otherwise.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    q.add_argument("--volume", default=None, metavar="VOLUME",
                   help="a volume to take the frame from, and to bounds-check against. "
                        "Optional, but without it you must pass --voxel-size")
    for plural, kind in _ANN_FLAGS:
        cols = ", ".join(c for g in _ANN_CSV_COLUMNS[kind] for c in g)
        q.add_argument(f"--{plural}", action="append", metavar="CSV",
                       help=f"a CSV of {plural} with columns {cols} (repeatable; a path, a "
                            f"URL, or - for stdin). Optional columns: id, description, "
                            f"segments, and segments:<relationship> for use with --link")
        q.add_argument(f"--{kind}", action="append", metavar="COORDS",
                       help=f"one {kind} inline as {cols} (repeatable)")
    q.add_argument("--scale", type=int, default=None, metavar="N",
                   help="the coordinates are in scale-N voxels; convert using the volume's "
                        "own per-level voxel sizes. Needs --volume")
    q.add_argument("--nm", action="store_true",
                   help="the coordinates are in physical nm; convert using the level-0 "
                        "voxel size")
    q.add_argument("--name", default="annotations", help="layer name")
    q.add_argument("--color", default="#ffee00", help="annotation colour")
    q.add_argument("--label", default="a", metavar="PREFIX",
                   help="prefix for generated annotation ids (default: %(default)s)")
    q.add_argument("--link", action="append", default=[], metavar="[REL=]LAYER",
                   help="bind a relationship to a segmentation LAYER (by its name in the "
                        "state), which is what makes an annotation's segments selectable. "
                        "REL=LAYER binds a segments:REL column; a bare LAYER binds the plain "
                        "segments column. Repeatable, so one annotation can name a body in "
                        "the reference and a segment in each of several other layers")
    q.add_argument("--voxel-size", default=None, metavar="Z,Y,X",
                   help="level-0 voxel size in nm, overriding the volume's")
    _add_output_flags(q, formats=("layer", "state", "url"), default="layer")
    q.set_defaults(func=cmd_annotate)

    # --- bboxes -------------------------------------------------------------
    q = sub.add_parser(
        "bboxes", help="a layer of boxes marking where a sparse volume's data is",
        description="Ask a volume where its data is and get one box per written region.\n\n"
                    "For a volume holding a handful of labeled regions inside a large empty "
                    "frame, finding them is the hard part. The boxes come from listing which "
                    "chunk objects exist — TensorStore never persists an all-fill chunk, so "
                    "the set of present keys IS the occupied footprint — then tightening "
                    "each box to its nonzero voxels at a coarse level.\n\n"
                    "The analysis lives in neu-vol (`ops.occupancy.labeled_regions`); "
                    "this turns its answer into a layer.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    q.add_argument("volume", help="the volume to inspect (path or s3://…)")
    q.add_argument("--level", type=int, default=0,
                   help="which level's chunk grid to list (default: %(default)s)")
    q.add_argument("--tighten-level", type=int, default=None, metavar="N",
                   help="tighten each box to its nonzero voxels at level N. Defaults to "
                        "the greater of 2 and --level; 0 is exact and slowest")
    q.add_argument("--no-tighten", action="store_true",
                   help="skip tightening: the boxes stay chunk-aligned")
    q.add_argument("--kind", choices=("box", "point"), default="box",
                   help="draw each region as a box or as a single centre point")
    q.add_argument("--name", default=None, help="layer name")
    q.add_argument("--label", default=None, metavar="PREFIX",
                   help="prefix for the region ids (default: r)")
    q.add_argument("--color", default="#ffee00", help="annotation colour")
    q.add_argument("--voxel-size", default=None, metavar="Z,Y,X",
                   help="level-0 voxel size in nm, overriding the volume's")
    _add_output_flags(q, formats=("layer", "state", "url"), default="layer")
    q.set_defaults(func=cmd_bboxes)

    # --- parse --------------------------------------------------------------
    # --- serve --------------------------------------------------------------
    # No `_add_output_flags`: this produces a running server, not a serialization.
    q = sub.add_parser(
        "serve", help="host arrays locally and open a viewer on them",
        description="Serve one or more arrays from THIS process and print the viewer "
                    "link.\n\n"
                    "For looking at something that is not published: a ground-truth "
                    "crop, a box out of a volume, a probability map. Each --image / --seg "
                    "/ --prob is a path (an HDF5 file, a volume, a slice stack), "
                    "optionally with a level and a box.\n\n"
                    "  neu-glance serve --seg piece.h5\n"
                    "  neu-glance serve --image vol --seg gt --crop-bbox 0,0,0,64,512,512\n"
                    "  neu-glance serve --image piece.h5:/raw --prob piece.h5:/affinity\n\n"
                    "Runs until Ctrl-C. The link is only valid while it runs: a served "
                    "array is addressed `python://volume/<token>`, scoped to this "
                    "process, so it cannot be shared or saved. Use `gen` for that, "
                    "against a published volume.\n\n"
                    "Needs the optional extra: pip install 'neu-glance[serve]'",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    q.add_argument("--image", action="append", metavar="SRC",
                   help="an image source (repeatable). PATH, or PATH:DATASET for an HDF5 "
                        "container holding more than one array")
    q.add_argument("--seg", action="append", metavar="SRC",
                   help="a segmentation source (repeatable). Served as labels whatever the "
                        "dtype — neuroglancer would read a uint8 label array as an image, "
                        "which loses the colour hashing and the selection UI")
    q.add_argument("--prob", action="append", metavar="SRC",
                   help="a probability / continuous-scalar source (repeatable). One channel "
                        "is drawn in a single colour with the OPACITY carrying the value, so "
                        "it composites over the image under it; a leading 3-channel axis is "
                        "shown as three colours instead")
    q.add_argument("--level", type=int, default=0, metavar="N",
                   help="which level to read from a multiscale --image/--seg/--prob "
                        "(default: 0). Applies to every source")
    q.add_argument("--crop-bbox", default=None, metavar="Z0,Y0,X0,Z1,Y1,X1",
                   help="serve only this box, in --level voxels. Whole volumes are usually "
                        "far too large to hold in memory, so this is required for anything "
                        "but a piece or a small volume")
    q.add_argument("--voxel-size", default=None, metavar="Z,Y,X",
                   help="voxel size in nm, overriding what the sources record. Needed for "
                        "a source that records none (a slice stack, a plain HDF5 file)")
    q.add_argument("--shader", default=None, metavar="NAME_OR_PATH",
                   help="GLSL for the image-like layers: a built-in name ("
                        + ", ".join(_IMAGE_SHADER_NAMES) + "), or 'none' for "
                        "neuroglancer's own default. Default picks by kind and channel "
                        "count")
    q.add_argument("--color", default=None, metavar="COLOR",
                   help="colour for the --prob layers (default: red): a name (red, cyan, "
                        "orange, ...) or #rrggbb. Any colour matplotlib names works too "
                        "where matplotlib is installed. Applies to a shader with a single "
                        "colour, which the one-channel default has; the viewer's own "
                        "control changes it afterwards")
    q.add_argument("--bind", default="127.0.0.1", metavar="ADDRESS",
                   help="the address to serve on (default: 127.0.0.1, reachable only from "
                        "this machine). Use 0.0.0.0 when the browser is elsewhere; the "
                        "printed link then names this host")
    q.add_argument("--port", type=int, default=0, metavar="N",
                   help="port to serve on (default: 0, meaning any free one)")
    q.add_argument("--into", default=None, metavar="PATH_OR_URL",
                   help="start from an existing state, given as a URL or a JSON file, and "
                        "add the served layers to it — keeping its view. Single-quote a "
                        "URL, since bash expands '!' inside double quotes")
    # `--annotate`, not `--annotations`, which in `gen` means a source to load.
    q.add_argument("--annotate", nargs="?", const=True, default=False, metavar="NAME",
                   help="add an empty annotation layer to draw boxes in, read back by "
                        "`Server.boxes()`. Off by default: a viewer that opens with a layer "
                        "nobody asked for reads as a bug. Give a NAME to call it something "
                        "else")
    q.set_defaults(func=cmd_serve)

    q = sub.add_parser(
        "parse", help="a neuroglancer URL back into its state JSON",
        description="Decode the state out of a link, for reading or editing it.\n\n"
                    "Note that `gen --into` accepts a URL directly, so this is not a "
                    "required step in that workflow — it is the inspection tool. The "
                    "reverse, a state JSON back to a link, is `gen --into STATE.json`.")
    q.add_argument("url", help="the neuroglancer URL, in single quotes (or - to read one "
                               "from stdin)")
    q.add_argument("--layers", action="store_true",
                   help="print just the layer names and types, one per line")
    q.add_argument("--out", action="append", default=None, metavar="PATH_OR_URL",
                   help="write the state JSON here instead of stdout. Repeatable; '-' is "
                        "stdout")
    q.set_defaults(func=cmd_parse)

    # --- shaders ------------------------------------------------------------
    q = sub.add_parser(
        "shaders", help="the built-in annotation shaders",
        description="List the built-in shaders, or print one to edit.\n\n"
                    "A shader lives in the viewer state rather than in the source, so a "
                    "link is the only place one can be shipped. Print one, edit it, and "
                    "pass the file back with `--annotation-shader path.glsl`.")
    q.add_argument("name", nargs="?", default=None,
                   help="print this shader's GLSL. Omit to list them")
    q.set_defaults(func=cmd_shaders)

    _add_help_command(sub)
    return p


def _add_help_command(sub) -> None:
    """``neu-glance help [COMMAND ...]`` — the same text ``--help`` prints.

    It re-parses the arguments with ``--help``, so the output cannot differ, and an unknown
    name gets argparse's own error listing the choices.
    """
    q = sub.add_parser(
        "help", help="what `--help` prints, as a subcommand",
        description="Print the help for the tool, or for one of its subcommands.\n\n"
                    "`neu-glance help serve` and `neu-glance serve --help` are the same "
                    "thing.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    q.add_argument("command", nargs="*", metavar="COMMAND",
                   help="the subcommand to describe (default: the whole tool)")
    q.set_defaults(func=cmd_help)


def cmd_help(args) -> int:
    """Hand the arguments back to the parser as ``--help``, and report its exit code."""
    try:
        build_parser().parse_args([*args.command, "--help"])
    except SystemExit as e:                      # what --help always does
        return int(e.code or 0)
    return 0


def _parse_args(argv=None):
    return build_parser().parse_args(argv)


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def _listed_segments(volume: str) -> list[int]:
    """The ids a segmentation's ``segment_properties`` lists; refuses one with none."""
    from neu_vol.location import read_json

    sub = (read_json(volume, "info") or {}).get("segment_properties")
    doc = read_json(volume, sub, "info") if sub else None
    if not doc:
        raise SystemExit(f"--all-segments needs segment_properties, and {volume} has none: "
                         f"there is no list of its segments to select")
    return [int(i) for i in doc["inline"]["ids"]]


def _seg_display(args, layers: list[dict]) -> None:
    """Apply ``--silhouette`` / ``--opacity-2d`` / ``--opacity-3d`` / ``--color`` /
    ``--no-pick`` to the named layers."""
    segs = {lyr["name"]: lyr for lyr in layers if lyr["type"] == "segmentation"}
    for flag, key, _what in _SEG_DISPLAY_FLAGS:
        for value in getattr(args, flag[2:].replace("-", "_")) or []:
            name, sep, number = value.rpartition("=")
            if not sep or not name:
                raise SystemExit(f"{flag} wants LAYER_NAME=VALUE, got {value!r}")
            if name not in segs:
                raise SystemExit(f"{flag} names no --seg layer: {name!r}. The segmentation "
                                 f"layers are {', '.join(segs) or '(none)'}")
            try:
                v = float(number)
            except ValueError:
                raise SystemExit(f"{flag} {value}: {number!r} is not a number") from None
            lo, hi = _SEG_DISPLAY_RANGE[key]
            if not lo <= v <= hi:
                raise SystemExit(f"{flag} {value}: must be between {lo:g} and {hi:g}")
            segs[name][key] = v

    from .shaders import ShaderProblem, as_hex_color

    for value in args.color or []:
        name, sep, color = value.rpartition("=")
        if not sep or not name:
            raise SystemExit(f"--color wants LAYER_NAME=COLOR, got {value!r}")
        if name not in segs:
            raise SystemExit(f"--color names no --seg layer: {name!r}. The segmentation "
                             f"layers are {', '.join(segs) or '(none)'}")
        try:
            segs[name]["segmentDefaultColor"] = as_hex_color(color, f"--color {name}")
        except ShaderProblem as e:
            raise SystemExit(str(e)) from None
    for name in args.no_pick or []:
        if name not in segs:
            raise SystemExit(f"--no-pick names no --seg layer: {name!r}. The segmentation "
                             f"layers are {', '.join(segs) or '(none)'}")
        segs[name]["pick"] = False


def _color_by(args, site: dict, layers: list[dict], volumes: dict[str, str], *,
              file) -> None:
    """Apply ``--color-by`` (or the config's ``color_by``) as each layer's segmentColors."""
    from .sources import SourceProblem, read_segment_properties
    from .state import StateProblem, colors_by_property

    wanted = {n: s["color_by"] for n, s in site["layers"].items()
              if s.get("color_by") and n in volumes}
    for value in args.color_by or []:
        name, sep, keys = value.partition("=")
        if not sep or not name or not keys.strip():
            raise SystemExit(f"--color-by wants LAYER_NAME=PROP[,PROP...], got {value!r}")
        if name not in volumes:
            raise SystemExit(f"--color-by names no --seg layer: {name!r}. The segmentation "
                             f"layers are {', '.join(volumes) or '(none)'}")
        wanted[name] = [k for k in keys.replace(" ", "").split(",") if k]
    by_name = {lyr["name"]: lyr for lyr in layers}
    for name, keys in wanted.items():
        try:
            colors, report = colors_by_property(read_segment_properties(volumes[name]), keys)
        except (SourceProblem, StateProblem) as e:
            raise SystemExit(f"--color-by {name}: {e}") from None
        by_name[name]["segmentColors"] = colors
        counts = ", ".join(f"{k} {n}" for k, n in report["by_key"].items())
        print(f"  {name}: coloured {report['colored']} of {report['segments']} segments by "
              f"{','.join(keys)} ({counts}; {report['values']} values)", file=file)
        if report["several_values"]:
            print(f"    {report['several_values']} had several values for a facet and took "
                  f"the first", file=file)


def cmd_gen(args) -> int:
    """A viewer state from volumes, annotation sources and layer files."""
    from .shaders import ShaderProblem
    from .sources import (SourceProblem, annotation_layer, annotation_source_extent,
                          annotation_source_voxel_size, volume_extent, volume_layer)
    from .state import StateProblem, annotation_extent, load_layer

    from . import config as _config

    err = sys.stderr
    layers: list[dict] = []
    frame_info: dict | None = None
    extents: list[Any] = []
    shader_notes: list[str] = []

    try:
        site = _config.load(args.config) if args.config else _config.parse({})
    except _config.ConfigProblem as e:
        raise SystemExit(str(e)) from None
    bounds = bool(site["volume"].get("bounds", False))
    seg_names = {v.rstrip("/").rsplit("/", 1)[-1] for v in args.seg or []}
    everything = set(args.all_segments or [])
    unknown = sorted(everything - seg_names)
    if unknown:
        raise SystemExit(f"--all-segments names no --seg layer: {', '.join(unknown)}")
    everything |= {n for n, s in site["layers"].items() if s.get("all_segments")} & seg_names

    try:
        # --image before --seg, so the segmentation draws over the image.
        for volume in args.image or []:
            layer, info = volume_layer(volume, kind="image", opacity=args.image_opacity,
                                       bounds=bounds)
            layers.append(layer)
            frame_info = frame_info or info
            extents.append(volume_extent(volume, info["format"]))
        seg_volumes: dict[str, str] = {}
        for i, volume in enumerate(args.seg or []):
            seg_volumes[volume.rstrip("/").rsplit("/", 1)[-1]] = volume
            picked = args.segments[i] if i < len(args.segments or []) else None
            ids = [int(s) for s in picked.replace(",", " ").split()] if picked else []
            if volume.rstrip("/").rsplit("/", 1)[-1] in everything:
                ids += [i for i in _listed_segments(volume) if i not in ids]
            layer, info = volume_layer(volume, kind="segmentation", segments=ids or None,
                                       bounds=bounds)
            layers.append(layer)
            frame_info = frame_info or info
            extents.append(volume_extent(volume, info["format"]))

        # Annotations after the volumes: relationships bind to the first --seg layer by name.
        linked_layer = next((lyr for lyr in layers if lyr["type"] == "segmentation"), None)
        linked = linked_layer["name"] if linked_layer else None
        filtering = bool(args.filter_by_segmentation)
        for source in args.annotations or []:
            layer, info = annotation_layer(
                source, shader=args.annotation_shader, linked_segmentation=linked,
                filter_by_segmentation=filtering)
            layers.append(layer)
            shader_notes.append(f"{layer['name']}: {info['shader'] or 'no shader'}")
            extents.append(annotation_source_extent(info["info"]))
            if frame_info is None:
                voxel = annotation_source_voxel_size(info["info"])
                if voxel:
                    frame_info = {"voxel_size": voxel, "units": "nm",
                                  "format": "annotations"}
        for path in args.layer or []:
            layers.extend(load_layer(path))
        config_notes = _config.apply(site, layers)
    except (SourceProblem, ShaderProblem, StateProblem, ValueError,
            json.JSONDecodeError) as e:
        raise SystemExit(str(e)) from None
    _seg_display(args, layers)
    names = {lyr["name"] for lyr in layers}
    args.hide = list(dict.fromkeys(
        [*(args.hide or []), *(n for n, s in site["layers"].items()
                               if s.get("hide") and n in names)]))

    # With --into and nothing to add, the state itself is the output: a view file straight
    # to a link, with any of the view options (--layout, --hide-slices, …) applied on the way.
    if not layers and not args.into:
        raise SystemExit("nothing to show: pass at least one of --image, --seg, "
                         "--annotations or --layer, or --into to re-emit an existing state")

    voxel = _ftriple(args.voxel_size, "voxel-size")
    units = "nm" if voxel else (frame_info or {}).get("units")
    if voxel is None:
        voxel = (frame_info or {}).get("voxel_size")
    # With --into the existing state supplies `dimensions`, so no voxel size is needed.
    if voxel is None and not args.into:
        raise SystemExit(
            "no voxel size available: --layer files carry their own frame but do not "
            "establish the viewer's, and no --image/--seg/--annotations source recorded "
            "one. Pass --voxel-size Z,Y,X (nm).")

    known = [e for e in extents if e]
    fit = max(known, key=lambda e: max(e[0])) if known else annotation_extent(layers)

    if layers:
        print(f"{len(layers)} layer(s): "
              + ", ".join(f"{lyr['name']} ({lyr['type']})" for lyr in layers), file=err)
    if voxel:
        print(f"  voxel size {tuple(voxel)} zyx, units {units or '?'}", file=err)
    for note in shader_notes:
        print(f"  shader {note}", file=err)
    if args.config:
        print(f"  display defaults from {args.config}", file=err)
    for note in config_notes:
        print(f"  {note}", file=err)
    _color_by(args, site, layers, seg_volumes, file=err)
    _report_bindings(layers, file=err)

    return _emit(args, layers, voxel=voxel, units=units, frame=fit)


def _report_bindings(layers, *, file) -> None:
    """Say which relationships are bound and whether the filter will hide everything."""
    selected_any = any(lyr.get("segments") for lyr in layers)
    for lyr in layers:
        if lyr.get("linkedSegmentationLayer"):
            target = next(iter(lyr["linkedSegmentationLayer"].values()))
            bound = ", ".join(sorted(lyr["linkedSegmentationLayer"]))
            filtered = lyr.get("filterBySegmentation")
            how = (f"; filtered on {', '.join(filtered)}" if filtered
                   else "; showing every annotation")
            print(f"  {lyr['name']}: {bound} bound to {target}{how}", file=file)
            if filtered and not selected_any:
                # Not a warning, but said, since an empty viewport looks like a broken layer.
                print(f"    no segments selected, so this opens EMPTY until you pick a "
                      f"body (or pass --no-filter-by-segmentation)", file=file)
        elif lyr["type"] == "annotation" and isinstance(lyr.get("source"), str):
            print(f"  {lyr['name']}: NOT bound to a segmentation — pass --seg for the "
                  f"relationship index to be usable", file=file)


def _parse_links(values) -> dict[str, str]:
    """``--link`` values as ``{relationship: layer}``, in the order given."""
    from .layers import DEFAULT_RELATIONSHIP

    out: dict[str, str] = {}
    for v in values or []:
        rel, sep, lyr = v.partition("=")
        rel, lyr = (rel.strip(), lyr.strip()) if sep else (DEFAULT_RELATIONSHIP, rel.strip())
        if not rel or not lyr:
            raise SystemExit(f"--link {v!r}: expected REL=LAYER or LAYER")
        if rel in out:
            raise SystemExit(f"--link binds relationship {rel!r} twice "
                             f"({out[rel]!r} and {lyr!r}); each needs exactly one layer")
        out[rel] = lyr
    return out


def cmd_annotate(args) -> int:
    """An annotation layer built from coordinates you supply."""
    from .layers import (DEFAULT_RELATIONSHIP, build_annotation, local_layer,
                         output_dimensions, positions, read_annotation_csv,
                         record_relationships, rescale)

    err = sys.stderr
    records = []
    for plural, kind in _ANN_FLAGS:
        for path in getattr(args, plural) or []:
            try:
                records += read_annotation_csv(_read_ann_file(path), kind, source=path)
            except ValueError as e:
                raise SystemExit(str(e)) from None
    records += _inline_records(args)
    if not records:
        raise SystemExit(
            "nothing to annotate: give at least one of --points/--boxes/--lines/"
            "--ellipsoids (a CSV path, a URL, or - for stdin) or an inline "
            "--point/--box/--line/--ellipsoid")

    voxel, units, shape = _volume_frame(args.volume, args.voxel_size)
    print(f"{args.volume or '(no volume: frame from --voxel-size)'}", file=err)
    if voxel:
        print(f"  frame       {'x'.join(f'{v:g}' for v in voxel)} {units or '?'} "
              f"per level-0 voxel", file=err)

    # Coordinates are stored as level-0 voxels, the frame `outputDimensions` declares.
    if args.scale:
        if not args.volume:
            raise SystemExit("--scale needs --volume: the conversion uses the volume's own "
                             "per-level voxel sizes, which cannot be guessed")
        factor = _level0_factor(args.volume, args.scale)
        records = rescale(records, factor)
        print(f"  scale {args.scale}     x{factor} (zyx) -> level-0 voxels", file=err)
    elif args.nm:
        if not voxel:
            raise SystemExit("--nm needs the level-0 voxel size: pass --voxel-size or a "
                             "--volume that records one")
        records = rescale(records, tuple(1.0 / v for v in voxel))
        print(f"  nm          /{tuple(voxel)} (zyx) -> level-0 voxels", file=err)

    links = _parse_links(args.link)
    named = record_relationships(records)
    if links:
        unbound = [n for n in named if n not in links]
        if unbound:
            raise SystemExit(
                f"relationship(s) {', '.join(unbound)} appear in the input but no --link "
                f"binds them, so their ids would be dropped. Add --link "
                f"{unbound[0]}=<segmentation layer name>.")
    elif any(n != DEFAULT_RELATIONSHIP for n in named):
        extra = [n for n in named if n != DEFAULT_RELATIONSHIP]
        raise SystemExit(
            f"the input has segments:<relationship> columns ({', '.join(extra)}) but no "
            f"--link says which layer each refers to. Add --link {extra[0]}=<layer name> "
            f"for each.")
    order = list(links) if links else None

    ids, annotations = set(), []
    for i, r in enumerate(records):
        ident = str(r.get("id") or f"{args.label}{i:03d}")
        if ident in ids:
            raise SystemExit(f"duplicate annotation id {ident!r}: neuroglancer keys its "
                             f"annotations by id, so duplicates collide. Fix the `id` "
                             f"column, or drop it and let them be numbered.")
        ids.add(ident)
        annotations.append(build_annotation(r, ident, order))

    dims, warning = output_dimensions(voxel, units)
    counts: dict[str, int] = {}
    for r in records:
        counts[r["kind"]] = counts.get(r["kind"], 0) + 1
    print(f"  annotations {len(annotations)}: "
          + ", ".join(f"{n} {k}" for k, n in sorted(counts.items())), file=err)
    n_seg = sum(1 for r in records if r.get("segments") or r.get("relationships"))
    if n_seg:
        print(f"  segments    {n_seg} annotation(s) linked to body ids", file=err)
    if links:
        print("  links       " + ", ".join(f"{k} -> {v}" for k, v in links.items()),
              file=err)

    # A wrong unit gives valid annotations in the wrong place; the volume's extent is the
    # only check available.
    if shape:
        outside = [i for i, r in enumerate(records)
                   if any(not (0 <= c <= shape[a])
                          for g in positions(r) for a, c in enumerate(g))]
        where = f"the level-0 extent {shape} (zyx)"
        if outside:
            print(f"\n  WARNING: {len(outside)} annotation(s) fall outside {where} — first "
                  f"at index {outside[0]}. Check whether the coordinates are in another "
                  f"scale's voxels (--scale N) or in nm (--nm).", file=err)
        else:
            print(f"  bounds      all inside {where}", file=err)
    if warning:
        print(f"\n  WARNING: {warning}", file=err)

    layer = local_layer(annotations, dims, name=args.name, color=args.color,
                        links=links or None)
    layers = _with_volume(args, [layer])
    return _emit(args, layers, voxel=voxel, units=units,
                 frame=(shape, (0, 0, 0)) if shape else None, selected=layer["name"])


def cmd_bboxes(args) -> int:
    """A layer of boxes over a sparse volume's occupied regions."""
    from neu_vol.ops.occupancy import NoOccupancy, labeled_regions

    from .layers import boxes_layer, output_dimensions

    err = sys.stderr
    volume = args.volume.rstrip("/")
    tighten = None if args.no_tighten else (
        max(2, args.level) if args.tighten_level is None else args.tighten_level)
    try:
        regions, ctx = labeled_regions(volume, level=args.level, tighten_level=tighten)
    except (FileNotFoundError, NoOccupancy, ValueError) as e:
        raise SystemExit(str(e)) from None

    voxel, units, shape = _volume_frame(volume, args.voxel_size)
    dims, warning = output_dimensions(voxel, units)

    print(f"{volume}", file=err)
    print(f"  level {args.level}: {len(regions)} region(s)", file=err)
    if not regions:
        print("  no stored chunks — the volume is empty at this level", file=err)
        return 1
    print(f"\n{'#':>3}  {'z':>15} {'y':>15} {'x':>15}  {'extent zyx':>18} "
          f"{'labels':>7}", file=err)
    for i, r in enumerate(regions):
        span = " ".join(f"{r['lo'][a]:6d}-{r['hi'][a]:6d}" for a in range(3))
        ext = "x".join(str(r["hi"][a] - r["lo"][a]) for a in range(3))
        n = "-" if r["n_labels"] is None else f"{r['n_labels']:,}"
        print(f"{i:>3}  {span}  {ext:>18} {n:>7}", file=err)
    if warning:
        print(f"\n  WARNING: {warning}", file=err)

    layer = boxes_layer(regions, dims,
                        name=args.name or f"{volume.rsplit('/', 1)[-1]}-regions",
                        color=args.color, kind=args.kind, label=args.label or "r")
    layers = _with_volume(args, [layer], volume=volume)
    frame = ((shape, (0, 0, 0)) if shape else
             (tuple(regions[0]["hi"][a] - regions[0]["lo"][a] for a in range(3)),
              tuple(regions[0]["lo"])))
    return _emit(args, layers, voxel=voxel, units=units, frame=frame,
                 selected=layer["name"])


def _with_volume(args, layers: list[dict], volume: str | None = None) -> list[dict]:
    """Prepend the volume's own layer when a whole state is being emitted.

    A state without the volume would show annotations floating in nothing. Not with
    ``--into``, whose state already has its volumes.
    """
    if _resolve_format(args) == "layer" or getattr(args, "into", None):
        return layers
    volume = volume or getattr(args, "volume", None)
    if not volume:
        return layers
    from .sources import SourceProblem, volume_layer

    try:
        layer, _info = volume_layer(volume)
    except SourceProblem as e:
        raise SystemExit(str(e)) from None
    return [layer, *layers]


def cmd_parse(args) -> int:
    """A neuroglancer URL back into its state JSON."""
    from .layers import render
    from .state import parse_url

    url = sys.stdin.read().strip() if args.url == "-" else args.url
    try:
        state = parse_url(url)
    except (ValueError, json.JSONDecodeError) as e:
        raise SystemExit(str(e)) from None

    if args.layers:
        for lyr in state.get("layers") or []:
            if isinstance(lyr, dict):
                print(f"{lyr.get('name', '(unnamed)')}\t{lyr.get('type', '?')}")
            else:
                # A state's `layers` may contain non-dict entries (such as comment strings).
                print(f"{lyr}\t(not a layer object)")
        return 0

    _deliver(args.out, render(state))
    return 0


def cmd_serve(args) -> int:
    """Host the given arrays and run a viewer on them until interrupted."""
    from .serving import ServedLayer, ServeProblem, _neuroglancer, serve

    # Check for the extra before reading any data.
    try:
        _neuroglancer()
    except ServeProblem as e:
        raise SystemExit(str(e)) from None

    crop = _sextuple(args.crop_bbox, "crop-bbox") if args.crop_bbox else None
    voxel_size = _ftriple(args.voxel_size, "voxel-size")
    requested = [(src, kind)
                 for kind, flag in (("image", args.image), ("segmentation", args.seg),
                                    ("probability", args.prob))
                 for src in (flag or ())]
    if not requested:
        raise SystemExit("nothing to serve: pass at least one --image, --seg or --prob")
    # Also before reading: a mangled URL is quick to detect.
    into = _load_into(args.into) if args.into else None

    layers = []
    for src, kind in requested:
        # Reading goes through ServedLayer.from_source, as in a notebook.
        try:
            # --color applies to probability layers only, whose default shader has a single
            # colour; a grayscale image would refuse it.
            layer = ServedLayer.from_source(
                src, kind, level=args.level, crop=crop, voxel_size=voxel_size,
                shader=args.shader,
                color=args.color if kind == "probability" else None)
        except ServeProblem as e:
            raise SystemExit(str(e)) from None
        except FileNotFoundError as e:
            raise SystemExit(str(e)) from None
        except KeyError as e:
            # An HDF5 container listing its datasets; the message is the useful part.
            raise SystemExit(str(e.args[0] if e.args else e)) from None
        print(f"  {kind:13} {src}  {layer.array.shape} {layer.array.dtype}  "
              f"voxel {'x'.join(f'{v:g}' for v in layer.frame.voxel_size_nm)} nm  "
              f"origin {tuple(layer.frame.origin_nm)}  as {layer.name!r}",
              file=sys.stderr)
        layers.append(layer)

    try:
        server = serve(layers, bind=args.bind, port=args.port, into=into,
                       annotations=args.annotate or None)
    except ServeProblem as e:
        raise SystemExit(str(e)) from None

    print(server.url)
    print("\nCtrl-C to stop. The link dies with this process — a served array is "
          "addressed\npython://volume/<token>, so it cannot be shared or saved.",
          file=sys.stderr)
    if args.bind == "127.0.0.1":
        print("Serving on 127.0.0.1, reachable only from this machine. Use --bind 0.0.0.0 "
              "if\nthe browser is elsewhere.", file=sys.stderr)
    try:
        import threading

        threading.Event().wait()
    except KeyboardInterrupt:
        print("\nstopping", file=sys.stderr)
        server.stop()
    return 0


def cmd_shaders(args) -> int:
    """List the built-in shaders, or print one.

    Annotation shaders (which need the source's ``prop_`` properties) and image shaders
    (which need a channel count) are listed separately.
    """
    from .shaders import IMAGE_SHADERS, SHADERS, SYNAPSE_COLORINGS, SYNAPSE_FILTERS

    if args.name is None:
        print("annotation shaders  (for `gen --annotation-shader`)")
        for name, entry in SHADERS.items():
            print(f"  {name}\t{entry['doc']}")
            print(f"    reads: {', '.join(entry['properties'])}")
            if name == "synapse":
                for box, props, _glsl in (*SYNAPSE_COLORINGS, *SYNAPSE_FILTERS):
                    print(f"    {box}: if the source has {', '.join(props)}")
        print("\nimage shaders  (for `serve --shader`)")
        for name, entry in IMAGE_SHADERS.items():
            print(f"  {name}\t{entry['doc']}")
            print(f"    channels: {entry['channels']}")
        print("\nA shader naming a property the source does not declare fails to compile "
              "and the layer draws NOTHING, so `gen` checks before applying one; `serve` "
              "checks the channel count for the same reason.")
        return 0
    for registry in (SHADERS, IMAGE_SHADERS):
        if args.name in registry:
            print(registry[args.name]["source"], end="")
            return 0
    raise SystemExit(f"no built-in shader {args.name!r}; known: "
                     + ", ".join([*SHADERS, *IMAGE_SHADERS]))


def main(argv=None) -> int:
    """Dispatch, with store logging filtered unless ``--store-logs`` asks for it.

    Each S3 prefix opened logs two benign ``AuthCredentialsProvider`` lines at ``E``
    severity; they are filtered. A real problem shows as ``PERMISSION_DENIED`` or
    ``AccessDenied``, which is never filtered.
    """
    import logging

    global _STDERR_IS_TTY
    _STDERR_IS_TTY = _isatty(sys.stderr)      # before quiet_store_logs swaps fd 2
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    if getattr(args, "store_logs", False):
        return args.func(args)
    from neu_vol.logs import quiet_store_logs

    with quiet_store_logs(True):
        return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
