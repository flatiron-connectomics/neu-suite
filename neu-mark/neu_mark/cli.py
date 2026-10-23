"""``neu-mark`` — the command line over neu_mark.

Argparse, like `neu-vol` and `neu-morpho`. Heavy imports stay inside the subcommand that
needs them, so `--help` is fast.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from . import __version__

#: The `--src` help, shared by the fetch subcommands. Flowing prose rather than a table:
#: argparse re-wraps `help=` text, and the docs build misreads indented blocks in it.
_SRC_HELP = (
    "DVID source as dvid://SERVER/UUID/INSTANCE, always three segments. "
    "SERVER may carry a port (emdata3:8900); a bare host gets http://, so use "
    "dvid+https://HOST/... for TLS. UUID may be a node uuid (93fdbc, abbreviated is "
    "fine) or a repo:branch ref (93fdbc:main), meaning HEAD of that branch. INSTANCE is "
    "the data instance name. A ':' in the server or the uuid adds no segment — this "
    "splits on '/' alone. Alternatively '@name' looks the instance up in your config "
    "(neu_mark.config), and the resolved URL is printed so the command still says "
    "what it read. "
)


def _add_common(q: argparse.ArgumentParser, *, instance_hint: str,
                bodies_required: bool = True) -> None:
    q.add_argument("--src", required=True, metavar="URL",
                   help=_SRC_HELP + f"For this command INSTANCE is {instance_hint}.")
    q.add_argument("--out", required=True, metavar="DST",
                   help="where the tables go — a local directory or s3://bucket/prefix. "
                        "May contain {uuid}, {uuid:N}, {uuid:full}, {branch} or "
                        "{instance}, expanded from the resolved node.")
    q.add_argument("--bodies", required=bodies_required, metavar="IDS", default=None,
                   help="body ids: inline (123,456), or a path to a .csv/.parquet/"
                        ".feather with a body column, or a text file with one id per "
                        "line." + ("" if bodies_required else " Omit it only with --all.")
                        + (" Required — DVID cannot cheaply enumerate the bodies worth "
                           "asking about." if bodies_required else ""))
    q.add_argument("--body-column", metavar="NAME", default=None,
                   help="which column holds the ids, when --bodies is a table with "
                        "several and none is named recognisably.")
    q.add_argument("--format", dest="fmt", default="parquet",
                   choices=("parquet", "csv", "feather"),
                   help="table format (default: parquet, which is the only one that "
                        "preserves uint64 body ids; csv warns about it).")
    q.add_argument("--dvid-locked", action="store_true",
                   help="read the newest LOCKED ancestor rather than what the ref points "
                        "at now. A branch HEAD is normally an open, still-changing node, "
                        "so a pull from it is not reproducible.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="neu-mark",
        description="Fetch DVID annotations into tables, and publish them to "
                    "neuroglancer.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"neu-mark {__version__}")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    p.add_argument("--store-logs", action="store_true",
                   help="do not filter the store's own logging. The S3 driver emits two "
                        "AuthCredentialsProvider lines per prefix at ERROR severity that "
                        "are not failures — they record the providers that missed before "
                        "the environment one succeeded — so they are filtered by default. "
                        "PERMISSION_DENIED and AccessDenied are never filtered")
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser(
        "points", help="point annotations (synapses) for a set of bodies, or all of them",
        description="Fetch every element of every listed body from a DVID `annotation` "
                    "instance, and write two tables: `points` (one row per element) and "
                    "`relationships` (one row per relationship, with the partner's body "
                    "resolved where it was fetched).\n\n"
                    "--all fetches the WHOLE instance by region instead, and looks up every "
                    "element's body in the synced labelmap at the same node, so every "
                    "partner's body is known. A few million elements fetch in under a "
                    "minute. The body list still chooses what is meshed and named; "
                    "`annotation-source --listed` marks which synapses touch it.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_common(q, instance_hint="an annotation instance, e.g. 'synapses'",
                bodies_required=False)
    q.add_argument("--all", dest="everything", action="store_true",
                   help="every element of the instance, with bodies from the labelmap, "
                        "instead of a body list. Mutually exclusive with --bodies.")
    q.add_argument("--threads", type=int, default=None, metavar="N",
                   help="concurrent requests (default 8; processes for --all's region "
                        "fetch). DVID is a shared service and answers overload with 503; "
                        "raise this deliberately.")
    q.add_argument("--drop-unmatched", action="store_true",
                   help="drop relationships whose partner body was not resolved. Off by "
                        "default: an unresolved partner and a dangling reference look "
                        "identical, so read the reported match rate first.")
    q.add_argument("--connections", action="store_true",
                   help="also write a `connections` table: the oriented, de-duplicated "
                        "(tbar, psd) pairs derived from `relationships`.")
    q.add_argument("--rois", metavar="LIST", default=None,
                   help="label each synapse with the neuropil it falls in, adding a `roi` "
                        "column to the points table. A comma-separated list of DVID roi "
                        "instance names, a path to a file with one per line, '@name' for "
                        "a set from your config, or SPEC.toml:LAYER for the ROIs of a "
                        "roi-layers layer. There is deliberately NO default: the "
                        "combined ROI volume is built by overwriting, so 'all of them' "
                        "would attribute a point to whichever containing ROI came last.")
    q.add_argument("--strict-rois", action="store_true",
                   help="refuse if the chosen ROIs overlap. By default overlap is allowed "
                        "and MEASURED: the later ROI wins, and the report says how many of "
                        "your synapses sit in an intersection and are therefore attributed "
                        "by ROI order alone. Use this if you need a strict partition — the "
                        "dataset has subtracted variants (INP(-ATL)(L), PENP(-AMMC)) "
                        "for that.")
    q.add_argument("--roi-source", metavar="URL", default=None,
                   help="the DVID node to read --rois from, as dvid://SERVER/UUID (default: "
                        "--src's node). For ROIs redrawn on a later node: the points and "
                        "their body ids still come from --src.")
    q.set_defaults(func=cmd_points)

    q = sub.add_parser(
        "bodies", help="per-body annotations (name, status), for a body list or all of them",
        description="Fetch the per-body records from a DVID `keyvalue` instance and write "
                    "a `bodies` table, one row per body. Records are ragged, so the table "
                    "is the union of the fields present.\n\n"
                    "--all reads the WHOLE instance, which is a different population and "
                    "often the one you want: a >=10-synapse selection holds 117 glia while "
                    "the instance holds 1,014, because most glia sit below any synapse "
                    "threshold. It reads a partition of the key space rather than asking "
                    "for the key list, which is unreliable at this size.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_common(q, instance_hint="a keyvalue instance, e.g. 'labels_annotations'",
                bodies_required=False)
    q.add_argument("--all", dest="everything", action="store_true",
                   help="every record in the instance, instead of a body list. Mutually "
                        "exclusive with --bodies.")
    q.set_defaults(func=cmd_bodies)

    q = sub.add_parser(
        "select-bodies",
        help="choose which bodies are worth fetching, by synapse count",
        description="Rank bodies by synapse count using DVID's own `labelsz` index and "
                    "write the list, for feeding to --bodies. Seconds, against ~70 "
                    "minutes to scan every label's voxel size — and synapse count is the "
                    "better criterion anyway: a body with no synapses is a fragment, and "
                    "many of the largest-by-volume bodies are glia.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    q.add_argument("--src", required=True, metavar="URL",
                   help=_SRC_HELP + "For this command INSTANCE is a labelsz instance "
                        "(e.g. 'synapses_labelsz') or the annotation instance it indexes "
                        "(e.g. 'synapses'), in which case its labelsz is found via Syncs.")
    q.add_argument("--out", required=True, metavar="FILE",
                   help="the list's path: .csv (read and edited by hand; every column is "
                        "a non-nullable integer, which csv preserves), .parquet or "
                        ".feather. A provenance record naming the node goes beside it as "
                        "<name>.provenance.json. Accepts the same {uuid}/{branch}/"
                        "{instance} placeholders.")
    q.add_argument("--min-synapses", type=int, default=10, metavar="N",
                   help="keep bodies with at least this many synapses in TOTAL "
                        "(default 10). Total rather than per-type on purpose: sensory "
                        "neurons may have no postsynapses, and a neuron leaving the "
                        "traced volume may have no presynapses.")
    q.add_argument("--min-pre", type=int, default=0, metavar="N",
                   help="additionally require at least N presynapses. Off by default — it "
                        "excludes neurons that only receive within the traced volume.")
    q.add_argument("--min-post", type=int, default=0, metavar="N",
                   help="additionally require at least N postsynapses. Off by default — it "
                        "excludes sensory neurons.")
    q.add_argument("--limit", type=int, default=None, metavar="N",
                   help="keep only the top N after filtering. Results are ranked by total "
                        "synapses, so this is how to ask for 'the biggest N'.")
    q.add_argument("--dvid-locked", action="store_true",
                   help="rank from the newest LOCKED ancestor. Worth doing: body ids "
                        "change with proofreading, so the list and the annotations should "
                        "come from the same node.")
    q.set_defaults(func=cmd_select_bodies)

    q = sub.add_parser(
        "segment-properties",
        help="a neuroglancer segment_properties source: names, tags and counts",
        description="Build a `segment_properties` source from the per-body records and "
                    "write it INTO a precomputed segmentation volume, so one neuroglancer "
                    "layer shows each body's name, its tags and its synapse counts.\n\n"
                    "`label` is the raw `instance` string — which is what lets this run "
                    "before a cell-type parse is settled. Tags carry a facet prefix "
                    "(group:, side:) because the format allows only ONE tags "
                    "property, so every facet has to pool into it.\n\n"
                    "Facets come from rules. The built-in set covers side, curated type "
                    "and the flags; --rules loads a module of your own whose rules "
                    "override the builtins by name, which is where anything "
                    "dataset-specific belongs — optic lobe column, glomerulus, cell type. "
                    "Run --dry-run first to see what each rule fired on.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    q.add_argument("--src", required=True, metavar="URL",
                   help=_SRC_HELP + "For this command INSTANCE is the keyvalue instance "
                        "holding the per-body records, e.g. 'labels_annotations'.")
    q.add_argument("--dst", required=True, metavar="VOLUME",
                   help="the precomputed SEGMENTATION VOLUME to write into (local or "
                        "s3://…). The document lands at <dst>/segment_properties/info, and "
                        "the volume's own info gains the key pointing at it. Accepts the "
                        "same {uuid}/{branch}/{instance} placeholders.")
    which = q.add_mutually_exclusive_group(required=True)
    which.add_argument("--bodies", metavar="IDS",
                       help="body ids: inline, or a .csv/.parquet/.feather with a body "
                            "column, or a text file with one per line.")
    which.add_argument("--meshed-bodies", action="store_true",
                       help="the bodies with a mesh in --dst, e.g. after neu-morpho run. "
                            "Unsharded meshes only.")
    q.add_argument("--body-column", metavar="NAME", default=None,
                   help="which column holds the ids, if the table has several.")
    q.add_argument("--counts", metavar="URL", default=None,
                   help="a labelsz instance (or the annotation instance it indexes) to add "
                        "`pre`, `post` and `syn` number properties. Cheap.")
    q.add_argument("--labelmap", metavar="URL", default=None,
                   help="a labelmap instance to add a `voxels` number property, from "
                        "DVID's own /sizes. ~1 minute for 20k bodies.")
    q.add_argument("--synapses", metavar="DIR", default=None,
                   help="a directory an earlier `points --rois` run wrote. Gives the rules a "
                        "`synapse_neuropils` column: per body, the share of its pre, post "
                        "and all synapses in each ROI, so a rule can tag a cell by where it "
                        "connects. Not itself published.")
    q.add_argument("--roi-spec", metavar="TOML", default=None,
                   help="a roi-layers spec, with --synapses: adds `synapse_neuropils_top`, "
                        "the same shares grouped by the spec's top-level neuropils.")
    q.add_argument("--rules", metavar="PATH", default=None,
                   help="a Python module of @rule functions, or @name for one configured "
                        "under [rule_sets]. Its rules override the built-in ones of the "
                        "same name and any others are added, so a module adding one "
                        "glomerulus rule need not redeclare the rest.")
    q.add_argument("--rules-only", action="store_true",
                   help="use only the module's rules, dropping the built-in ones. For a "
                        "dataset whose annotation strings share nothing with these.")
    q.add_argument("--max-tags", metavar="N", type=float, default=None,
                   help="cap the tag vocabulary. Below 1 this is a fraction of the segments "
                        "written (the default is 0.10, floored at 100 tags so short lists "
                        "are not judged by a ratio); 1 or above is an absolute count, "
                        "respected as given. A rule echoing whole instance strings back "
                        "produces a valid document no viewer can use; raise this when a "
                        "dataset genuinely has that many distinct types.")
    q.add_argument("--dry-run", action="store_true",
                   help="build and report without writing anything. --no-link still writes "
                        "the document; this writes nothing at all.")
    q.add_argument("--drop-glia", action="store_true",
                   help="exclude glia-labelled bodies. By default they are KEPT and carry "
                        "a `glia` tag — not connectome-relevant, but worth seeing.")
    q.add_argument("--no-link", dest="link", action="store_false",
                   help="write the source WITHOUT adding the key to the volume's info. Use "
                        "this to inspect the output before touching a published volume; "
                        "the linking step is what makes a viewer associate the two.")
    q.add_argument("--dvid-locked", action="store_true",
                   help="read the newest LOCKED ancestor rather than the branch HEAD.")
    q.set_defaults(func=cmd_segment_properties, link=True)

    q = sub.add_parser(
        "annotation-source",
        help="a neuroglancer precomputed annotation source: one LINE per connection",
        description="Write a `neuroglancer_annotations_v1` LINE source, one line from each "
                    "presynaptic site to each of its partners. Lines, not points, because a "
                    "connectome is edges — and one endpoint pair is one annotation, so a "
                    "T-bar with five partners contributes five.\n\n"
                    "Input is the tables an earlier `points` run wrote (`--tables`), which "
                    "is the cheap path: the tables are the durable artifact and a rebuild "
                    "with different bounds or sharding need not refetch. Give `--src` and "
                    "`--bodies` instead to fetch first.\n\n"
                    "This is a SEPARATE source from the segmentation — unlike mesh and "
                    "skeletons, an annotation layer cannot be named from a volume's info, so "
                    "a viewer adds it as its own layer.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    q.add_argument("--dst", required=True, metavar="PATH",
                   help="where the source goes (local or s3://…). Accepts the same "
                        "{uuid}/{branch}/{instance} placeholders when --src is given.")
    q.add_argument("--tables", metavar="DIR", default=None,
                   help="a directory an earlier `points` run wrote, holding "
                        "connections.parquet and points.parquet.")
    q.add_argument("--src", metavar="URL", default=None,
                   help=_SRC_HELP + "Only needed to fetch now instead of using --tables.")
    q.add_argument("--bodies", metavar="IDS", default=None,
                   help="body ids to fetch, when --src is given rather than --tables.")
    q.add_argument("--body-column", metavar="NAME", default=None,
                   help="which column holds the ids, if the table has several.")
    q.add_argument("--voxel-size", metavar="X,Y,Z", default="8,8,8",
                   help="nanometres per voxel of the frame the positions are in. Annotation "
                        "coordinates are stored in VOXELS, and this is what puts them in the "
                        "same physical space as the segmentation (default %(default)s).")
    q.add_argument("--bounds", metavar="X0,Y0,Z0,X1,Y1,Z1", default=None,
                   help="the source's lower and upper bound in voxels. Defaults to the "
                        "data's own extent, padded by one voxel.")
    q.add_argument("--drop-partial", dest="include_partial", action="store_false",
                   help="drop lines with only one endpoint's body resolved. By default they "
                        "are KEPT and undifferentiated: the partner is a real synapse whose "
                        "body simply was not in the fetched set, and the line is still where "
                        "the synapse is.")
    q.add_argument("--listed", metavar="IDS", default=None,
                   help="the bodies the segmentation layer shows (meshed and named, e.g. "
                        "the select-bodies list), in any --bodies form: adds `pre_listed` "
                        "and `post_listed`, which the synapse shader's hide_unlisted reads. "
                        "Worth it for tables from `points --all`, where most partners are "
                        "other bodies.")
    q.add_argument("--roi-spec", metavar="TOML", default=None,
                   help="a roi-layers spec: adds `roi` and `neuropil` properties, each "
                        "synapse's ROI and its top-level neuropil by the spec's ids (the ids "
                        "the ROI layers paint, so a synapse coloured by id matches its "
                        "neuropil's mesh). Needs tables from `points --rois`.")
    q.add_argument("--per-cell", type=int, default=4_000, metavar="N",
                   help="annotations a spatial cell aims to hold, which sets how many levels "
                        "the grid needs and how much a zoomed-out view downloads "
                        "(default %(default)s).")
    q.add_argument("--threads", type=int, default=None, metavar="N",
                   help="concurrent DVID requests, when fetching rather than using --tables.")
    q.add_argument("--dvid-locked", action="store_true",
                   help="read the newest LOCKED ancestor rather than the branch HEAD.")
    q.add_argument("--verify", type=int, nargs="?", const=200, default=None, metavar="N",
                   help="after writing, fetch N annotations back THROUGH the source's own "
                        "info and compare them with the table. Everything before the write "
                        "can be checked in memory; only a read-back proves the keys are "
                        "right, and a wrong key leaves a viewer with nothing while every "
                        "byte on the store is correct.")
    q.add_argument("--dry-run", action="store_true",
                   help="report the plan — lines, stride, levels, shards — write nothing.")
    q.set_defaults(func=cmd_annotation_source, include_partial=True)

    q = sub.add_parser(
        "roi-layers",
        help="a node's DVID ROIs as precomputed segmentation layers",
        description="Write a DVID node's ROIs as precomputed segmentations, one per layer "
                    "of a spec, each with segment properties naming its ROIs. A voxel "
                    "holds one label, so ROIs that nest go in different layers, and "
                    "overlaps between ROIs in one layer are settled by the spec's overlap "
                    "policy; ROIs nesting within one layer are refused.\n\n"
                    "The spec (TOML) lists the layers, a neuropil tree that the 'top' and "
                    "'parts' layers cut, ids shared by every layer, descriptions, and the "
                    "ROIs to ignore. Every ROI on the node must be in a layer or ignored, "
                    "so a new one is never dropped unseen. The layers are not meshed here: "
                    "run `neu-morpho run --stages index,mesh` on each.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    q.add_argument("--src", required=True, metavar="URL",
                   help="the DVID node, as dvid://SERVER/UUID: ROIs are a node's roi "
                        "instances, so no instance is named. UUID may be a branch ref, "
                        "resolved once and recorded.")
    q.add_argument("--spec", required=True, metavar="TOML",
                   help="the layer spec (see neu_mark.roi_layers).")
    q.add_argument("--like", required=True, metavar="VOLUME",
                   help="the segmentation the layers go beside. Its level-0 voxel size and "
                        "origin set the grid: an ROI block is 32 of its voxels.")
    q.add_argument("--dst", required=True, metavar="PATH",
                   help="where the layers go, one volume per layer plus rois.json (local or "
                        "s3://…).")
    q.add_argument("--layers", metavar="NAMES", default=None,
                   help="comma-separated layer names to build, instead of all of them.")
    q.add_argument("--overwrite", action="store_true",
                   help="replace layers that already exist. Their meshes go too, so re-run "
                        "the meshing.")
    q.add_argument("--dvid-locked", action="store_true",
                   help="read the newest LOCKED ancestor rather than the branch HEAD.")
    q.add_argument("--dry-run", action="store_true",
                   help="fetch and resolve every layer, report it, write nothing.")
    q.set_defaults(func=cmd_roi_layers)

    q = sub.add_parser(
        "roi-survey",
        help="measure a node's DVID ROIs: sizes, containment tree, overlaps",
        description="Fetch every ROI on a DVID node and report what the blocks say: each "
                    "ROI's size and parent (the smallest larger ROI holding most of it), how "
                    "well each parent's children cover it and how far they stick out, the "
                    "overlaps between ROIs that are not nested, empty and identical ROIs. "
                    "The input to a roi-layers spec, and the check when a node's ROIs "
                    "change.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    q.add_argument("--src", required=True, metavar="URL",
                   help="the DVID node, as dvid://SERVER/UUID. UUID may be a branch ref, "
                        "resolved once and recorded.")
    q.add_argument("--compare", metavar="URL", default=None,
                   help="another node, e.g. the one a spec was written against: report each "
                        "ROI whose blocks changed, and any added or removed.")
    q.add_argument("--spec", metavar="TOML", default=None,
                   help="a roi-layers spec to check against this node: unplaced ROIs, parts "
                        "outside their top-level neuropil, and each layer's overlaps or why "
                        "it would be refused.")
    q.add_argument("--like", metavar="VOLUME", default=None,
                   help="the segmentation the ROIs go beside: report ROIs reaching past its "
                        "level-0 extent.")
    q.add_argument("--contained", type=float, default=None, metavar="FRAC",
                   help="a child this far inside its parent is not reported as sticking out "
                        "(default 0.95).")
    q.add_argument("--max-overlap", type=float, default=None, metavar="FRAC",
                   help="flag siblings overlapping by more than this share of the smaller, "
                        "as roi-layers refuses them (default 0.2).")
    q.add_argument("--out", metavar="JSON", default=None,
                   help="also write the full record, every overlapping pair included.")
    q.add_argument("--dvid-locked", action="store_true",
                   help="read the newest LOCKED ancestor rather than the branch HEAD.")
    q.set_defaults(func=cmd_roi_survey)

    q = sub.add_parser(
        "info", help="what a DVID annotation source is, and which node you would get",
        description="One request per fact: the instance type, what it is synced to, and "
                    "both candidate nodes for the ref (what it points at now, and the "
                    "newest locked one).")
    q.add_argument("--src", required=True, metavar="URL", help=_SRC_HELP)
    q.set_defaults(func=cmd_info)

    _add_help_command(sub)
    return p


def _add_help_command(sub) -> None:
    """``neu-mark help [COMMAND ...]``, which re-parses as ``COMMAND --help``, so it prints
    exactly what ``--help`` does."""
    q = sub.add_parser(
        "help", help="what `--help` prints, as a subcommand",
        description="Print the help for the tool, or for one of its subcommands.\n\n"
                    "`neu-mark help explore` and `neu-mark explore --help` are the same "
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


def _resolve_src(value: str) -> str:
    """An ``@name`` config reference as a URL; anything else unchanged."""
    from . import config as _config

    if isinstance(value, str) and value.startswith(_config.REFERENCE_PREFIX):
        resolved = _config.load().resolve(value)
        print(f"{value}  ->  {resolved}")
        return resolved
    return value


def _src_spec(src: str) -> dict[str, Any]:
    """Resolve ``--src`` to a spec, expanding an ``@name`` config reference.

    A resolved reference is printed, so the command's output shows what the config turned
    it into.
    """
    from neu_vol.dvid import is_url, parse_url

    from . import config as _config

    if isinstance(src, str) and src.startswith(_config.REFERENCE_PREFIX):
        resolved = _config.load().resolve(src)
        print(f"--src {src}  ->  {resolved}")
        src = resolved

    if not is_url(src):
        raise SystemExit(
            f"--src {src!r} is not a DVID URL. This command reads annotations from DVID, "
            f"which is addressed as dvid://server/uuid/instance. A leading "
            f"{_config.REFERENCE_PREFIX!r} looks the name up in your config instead.")
    return {"backend": "dvid", **parse_url(src)}


def _expand_out(out: str, source: dict[str, Any]) -> str:
    """Resolve {uuid}/{branch}/{instance} in the destination.

    Takes the already-resolved source, so ``{uuid}`` is the node actually read (under
    ``--dvid-locked``, not the branch HEAD).
    """
    from neu_vol.ops.naming import expand, has_placeholder

    if not has_placeholder(out):
        return out
    resolved = expand(out, source)
    print(f"--out {out}  ->  {resolved}")
    return resolved


def _load_bodies(args) -> list[int]:
    from . import bodies as _bodies

    ids = _bodies.load(args.bodies, column=args.body_column)
    print(f"bodies: {_bodies.summarise(ids)}")
    return ids


def _meshed_bodies(volume: str) -> list[int]:
    """``--meshed-bodies``: the bodies with a mesh in ``volume``."""
    from neu_vol.readback import UnsupportedSubresource, mesh_body_ids

    from . import bodies as _bodies

    try:
        ids = mesh_body_ids(volume)
    except UnsupportedSubresource as e:
        raise SystemExit(f"--meshed-bodies: {e} Use --bodies.") from None
    if not ids:
        raise SystemExit(f"--meshed-bodies: no meshes in {volume}")
    print(f"bodies: {_bodies.summarise(ids)}, meshed in {volume}")
    return ids


def _roi_list(value: str | None) -> list[str] | None:
    """``--rois`` as a name list: a comma list, a file of names, or an ``@name`` config set."""
    import os

    if not value:
        return None
    from . import notebook as _nb

    spec_path, _, layer = value.rpartition(":")
    if spec_path.endswith(".toml") and os.path.exists(spec_path):
        from . import roi_layers

        resolved = roi_layers.layer_rois(roi_layers.load_spec(spec_path), layer)
        print(f"--rois {value}  ->  {len(resolved)} ROIs")
        return resolved
    if not value.startswith("@") and os.path.exists(value):
        with open(value) as fh:
            return [line.split("#", 1)[0].strip() for line in fh
                    if line.split("#", 1)[0].strip()]
    resolved = _nb.roi_set(value)
    if value.startswith("@"):
        print(f"--rois {value}  ->  {len(resolved)} ROIs")
    return resolved


def _rules_path(value: str | None) -> str | None:
    """``--rules`` as a path: a filename, or an ``@name`` from ``[rule_sets]``.

    Prints what an ``@name`` resolved to.
    """
    from . import config as _config

    if not value:
        return None
    if not value.startswith(_config.REFERENCE_PREFIX):
        return value
    cfg = _config.load()
    resolved = cfg.rule_set(value[len(_config.REFERENCE_PREFIX):])
    print(f"--rules {value}  ->  {resolved}")
    return resolved


def _report_rois(stats: dict | None) -> None:
    """How many synapses landed in a neuropil. A large unlabelled share means the wrong set."""
    if not stats:
        return
    total = stats["labeled"] + stats["unlabeled"]
    if not total:
        return
    print(f"rois: {stats['labeled']} of {total} synapses inside one of "
          f"{len(stats['rois'])} ROIs ({stats['labeled'] / total:.1%}), "
          f"{stats['unlabeled']} outside every one")
    top = sorted(stats["counts"].items(), key=lambda kv: -kv[1])[:5]
    if top:
        print("  " + ", ".join(f"{name} ({n})" for name, n in top))
    if stats.get("overlapping"):
        n = stats.get("ambiguous") or 0
        print(f"  {len(stats['overlapping'])} ROI pairs intersect; {n} synapse(s) "
              f"({n / total:.2%}) sit in an intersection and are attributed by ROI ORDER "
              f"alone" + (f" — {stats.get('ambiguous_pairs')}" if n else ""))
    if stats["unlabeled"] > stats["labeled"]:
        print("  NOTE: most synapses fell outside every ROI given. That is a statement "
              "about the ROI set, not the data — check it covers the traced volume.")


def _report_match(match: dict) -> None:
    """The headline number. A low fraction means the body list is too small."""
    if not match.get("pairs"):
        return
    frac = match["fraction"]
    print(f"connections: {match['pairs']} distinct (tbar, psd) pairs, "
          f"{match['both_ends']} with both bodies resolved ({frac:.1%}), "
          f"{match['one_end']} with one")
    if frac is not None and frac < 0.5:
        print(f"  NOTE: only {frac:.1%} of pairs have both endpoints in the body list. "
              f"A relationship resolves only when BOTH bodies were fetched, so this is "
              f"a statement about the body list's coverage, not about the data. Widen "
              f"--bodies for connectivity work.")


def cmd_points(args) -> int:
    from . import dvid, ops

    everything = bool(getattr(args, "everything", False))
    if everything and args.bodies:
        raise SystemExit("--all fetches every element; pass it or --bodies, not both.")
    if not everything and not args.bodies:
        raise SystemExit("--bodies is required, or pass --all to fetch the whole instance.")

    # Resolve the node first: the destination name and every request derive from it.
    source = dvid.open_points_source(_src_spec(args.src),
                                    prefer_locked=bool(args.dvid_locked))
    out = _expand_out(args.out, source)
    roi_source = None
    if args.roi_source:
        from .roi_layers import open_node

        if not args.rois:
            raise SystemExit("--roi-source says where --rois are read; pass --rois too.")
        try:
            roi_source, _ = open_node(_resolve_src(args.roi_source),
                                      prefer_locked=bool(args.dvid_locked))
        except ValueError as e:
            raise SystemExit(f"--roi-source: {e}") from None
    ids = None if everything else _load_bodies(args)

    kwargs = {} if args.threads is None else {"threads": args.threads}
    result = ops.fetch_points(source, out, ids, fmt=args.fmt,
                              drop_unmatched=bool(args.drop_unmatched),
                              rois=_roi_list(args.rois),
                              on_roi_overlap="error" if args.strict_rois else "warn",
                              roi_source=roi_source,
                              write_connections=bool(args.connections), **kwargs)
    print(f"points: {len(result['points'])} elements, "
          f"{len(result['relationships'])} relationships")
    _report_rois(result.get("rois"))
    _report_match(result["match"])
    if result["failures"]:
        print(f"  {len(result['failures'])} bodies failed and were skipped; "
              f"first few: {list(result['failures'])[:5]}", file=sys.stderr)
    print(f"wrote {', '.join(result['written'])} to {out}")
    return 0


def cmd_bodies(args) -> int:
    from . import dvid, ops

    everything = bool(getattr(args, "everything", False))
    if everything and args.bodies:
        raise SystemExit(
            "--all and --bodies describe different populations; pass one, not both.")
    if not everything and not args.bodies:
        raise SystemExit(
            "--bodies is required, or pass --all to read the whole instance.")

    source = dvid.open_bodies_source(_src_spec(args.src),
                                    prefer_locked=bool(args.dvid_locked))
    out = _expand_out(args.out, source)
    ids = None if everything else _load_bodies(args)

    result = ops.fetch_bodies(source, out, ids, fmt=args.fmt, everything=everything)
    if everything:
        print(f"bodies: {result['found']} records across "
              f"{len(result['ranges'])} key ranges")
    else:
        print(f"bodies: {result['found']} of {result['requested']} had a record")
        if result["missing"]:
            print(f"  {len(result['missing'])} had none (not an error — DVID simply has no "
                  f"annotation for them)")
    print(f"wrote {', '.join(result['written'])} to {out}")
    return 0


def cmd_select_bodies(args) -> int:
    from . import dvid, ops

    source = dvid.open_counts_source(_src_spec(args.src),
                                    prefer_locked=bool(args.dvid_locked))
    out = _expand_out(args.out, source)
    result = ops.write_selected_bodies(source, out, min_synapses=args.min_synapses,
                                       min_pre=args.min_pre, min_post=args.min_post,
                                       limit=args.limit)
    df = result["bodies"]
    print(f"selected {len(df)} bodies (>= {args.min_synapses} synapses"
          + (f", >= {args.min_pre} pre" if args.min_pre else "")
          + (f", >= {args.min_post} post" if args.min_post else "") + ")")
    if len(df):
        print(f"  synapses per body: {int(df['syn'].min())}..{int(df['syn'].max())}, "
              f"{int(df['syn'].sum())} total")
    print(f"wrote {result['path']}")
    print(f"  feed it onward with:  --bodies {result['path']}")
    return 0


def cmd_segment_properties(args) -> int:
    from . import dvid, ops, segprops

    bodies_src = dvid.open_bodies_source(_src_spec(args.src),
                                        prefer_locked=bool(args.dvid_locked))
    counts_src = (dvid.open_counts_source(_src_spec(args.counts),
                                         prefer_locked=bool(args.dvid_locked))
                  if args.counts else None)
    labelmap_src = None
    if args.labelmap:
        from neu_vol.dvid import parse_url
        labelmap_src = {"backend": "dvid", **parse_url(_resolve_src(args.labelmap))}
        labelmap_src["uuid"] = bodies_src["uuid"]      # the SAME node, always

    dst = _expand_out(args.dst, bodies_src)
    ids = _meshed_bodies(dst) if args.meshed_bodies else _load_bodies(args)

    result = ops.segment_properties(
        bodies_src, dst, ids, counts_source=counts_src, labelmap_source=labelmap_src,
        synapses=args.synapses, roi_spec=args.roi_spec, rules_path=_rules_path(args.rules), rules_only=bool(args.rules_only),
        max_tags=args.max_tags,
        keep_glia=not args.drop_glia, link=bool(args.link),
        dry_run=bool(args.dry_run))

    for line in segprops.format_report(result["report"]):
        print(line)
    if result.get("dry_run"):
        print(f"DRY RUN — nothing written. {dst} is untouched.")
        return 0
    print(f"wrote {', '.join(result['written'])} to {dst}")
    if result["linked"]:
        print(f"linked into {dst}/info: {result['linked']}")
    else:
        print("NOT linked — the volume's info is untouched, so a viewer will not associate "
              "these with the labels layer. Re-run without --no-link when you are happy.")
    return 0


def _triple(value: str, name: str, n: int = 3) -> list[float]:
    parts = [p for p in value.replace(" ", "").split(",") if p]
    if len(parts) != n:
        raise SystemExit(f"{name} wants {n} comma-separated numbers, got {len(parts)}: "
                         f"{value!r}")
    return [float(p) for p in parts]


def cmd_annotation_source(args) -> int:
    from . import annsource, dvid, ops

    if bool(args.tables) == bool(args.src):
        raise SystemExit("pass --tables (reuse an earlier fetch) or --src plus --bodies "
                         "(fetch now), not both and not neither.")

    voxel_size = _triple(args.voxel_size, "--voxel-size")
    bounds = None
    if args.bounds:
        b = _triple(args.bounds, "--bounds", 6)
        bounds = (b[:3], b[3:])
    listed = None
    if args.listed:
        from . import bodies as _bodies

        listed = _bodies.load(args.listed, column=args.body_column)
        print(f"listed: {_bodies.summarise(listed)}")

    kwargs: dict[str, Any] = {}
    if args.tables:
        from . import io as _io

        conns, points = _io.read_tables(args.tables, ("connections", "points"))
        dst = args.dst
        kwargs.update(connections=conns, points=points)
        print(f"tables: {len(conns):,} connections, {len(points):,} elements "
              f"from {args.tables}")
    else:
        if not args.bodies:
            raise SystemExit("--src needs --bodies: DVID cannot cheaply enumerate the "
                             "bodies worth asking about.")
        source = dvid.open_points_source(_src_spec(args.src),
                                        prefer_locked=bool(args.dvid_locked))
        dst = _expand_out(args.dst, source)
        kwargs.update(source=source, bodies=_load_bodies(args))
        if args.threads is not None:
            kwargs["threads"] = args.threads

    if args.dry_run:
        # Everything except the write, so the plan reported is the plan that would run.
        import numpy as np

        from . import tables as _t
        if "connections" not in kwargs:
            raise SystemExit("--dry-run needs --tables; there is nothing to plan before "
                             "the fetch has happened.")
        frame, rois = ops.enrich_for_source(kwargs["connections"], kwargs["points"],
                                            args.roi_spec)
        if bounds is None:
            corners = np.vstack([_t.positions_xyz(frame, "pre_"),
                                 _t.positions_xyz(frame, "post_")]).astype(float)
            bounds = (corners.min(axis=0), corners.max(axis=0) + 1.0)
        built = annsource.build(frame, lower_bound=bounds[0], upper_bound=bounds[1],
                                voxel_size_xyz=voxel_size, per_cell=args.per_cell,
                                include_partial=bool(args.include_partial), listed=listed,
                                **rois)
        for line in annsource.format_report(built["report"]):
            print(line)
        print(f"would write to {dst} (nothing written)")
        return 0

    result = ops.annotation_source(dst, voxel_size_xyz=voxel_size, bounds=bounds,
                                   include_partial=bool(args.include_partial),
                                   per_cell=args.per_cell, roi_spec=args.roi_spec,
                                   listed=listed, **kwargs)
    for line in annsource.format_report(result["report"]):
        print(line)
    print(f"wrote {', '.join(result['written'])} to {dst}")

    if args.verify:
        check = annsource.verify(dst, result["table"], sample=args.verify)
        body = check["body_check"]
        print(f"verified {check['sampled']} annotations read back through the source's info")
        if body:
            print(f"  body {body['body']}: {body['found']} lines in the pre index, "
                  f"table says {body['expected']}")
        if check["problems"]:
            print(f"{len(check['problems'])} PROBLEMS — the source is not what the table "
                  f"says:", file=sys.stderr)
            for problem in check["problems"][:10]:
                print(f"  {problem}", file=sys.stderr)
            return 1
        print("  no problems")

    print("add it in neuroglancer as its own layer — an annotation source cannot be named "
          "from the segmentation's info the way mesh and skeletons are.")
    return 0


def cmd_info(args) -> int:
    from neu_vol.dvid import (instance_info, instance_type, node_summary,
                                     synced_instances)

    src = _src_spec(args.src)
    info = instance_info(src)
    kind = instance_type(info)
    synced = synced_instances(info)
    print(f"instance : {src['instance']}  ({kind})")
    print(f"synced to: {', '.join(synced) if synced else '(nothing)'}")
    if kind == "annotation" and not synced:
        print("  WARNING: an unsynced annotation instance cannot be read per body — "
              "DVID's /label endpoint is built from the sync.")

    summary = node_summary(src)
    head = summary["head"]
    print(f"latest      : {head['uuid']}  "
          f"({'locked' if head['locked'] else 'OPEN — still being written'})")
    locked = summary["locked"]
    if locked is None:
        print(f"latest lock : unavailable ({summary['locked_error']})")
    elif locked["uuid"] == head["uuid"]:
        print("latest lock : same node")
    else:
        print(f"latest lock : {locked['uuid']}  "
              f"(walked back {locked['walked']}; --dvid-locked selects this)")
    return 0


def cmd_roi_layers(args) -> int:
    from . import roi_layers

    layers = [s.strip() for s in args.layers.split(",") if s.strip()] if args.layers else None
    record = roi_layers.build(args.src, args.spec, args.like, args.dst, layers=layers,
                              overwrite=args.overwrite, dry_run=args.dry_run,
                              prefer_locked=args.dvid_locked)
    grid = record["grid"]
    print(f"node {record['source']['uuid']}: {grid['shape']} blocks of "
          f"{grid['voxel_size_nm'][0]:g} nm from {grid['origin_nm']} nm")
    for name, layer in record["layers"].items():
        pairs = layer["overlaps"]
        print(f"  {name:20} {layer['segments']:4} segments" +
              (f", {len(pairs)} overlapping pair(s), e.g. {pairs[0]}" if pairs else ""))
    print("dry run: nothing written" if args.dry_run else f"-> {args.dst.rstrip('/')}/")
    return 0


def cmd_roi_survey(args) -> int:
    from . import roi_survey

    kw = {k: v for k, v in (("contained", args.contained),
                            ("max_overlap", args.max_overlap)) if v is not None}
    record = roi_survey.run(args.src, like=args.like, spec=args.spec,
                            compare_to=args.compare, prefer_locked=args.dvid_locked, **kw)
    print(roi_survey.format_report(record))
    if args.out:
        import json

        with open(args.out, "w") as f:
            json.dump(record, f, indent=1)
        print(f"-> {args.out}")
    return 0


def main(argv=None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from neu_vol.logs import quiet_store_logs

    try:
        with quiet_store_logs(not args.store_logs):
            return args.func(args)
    except (ValueError, FileNotFoundError, FileExistsError, ImportError) as exc:
        raise SystemExit(f"{type(exc).__name__}: {exc}") from exc


if __name__ == "__main__":                                       # pragma: no cover
    raise SystemExit(main())
