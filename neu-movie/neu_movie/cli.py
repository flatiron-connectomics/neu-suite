"""neu-movie: a TOML spec in, a rendered movie out.

    neu-movie camera  STATE                 a spec's [view] from a neuroglancer state or link
    neu-movie check   SPEC [--links]        sets, counts, schedule, colour clashes; no Blender
    neu-movie build   SPEC                  meshes and Blender files -> a new build folder
    neu-movie stills  BUILD -f 300 450      render a few frames here, labelled
    neu-movie submit  BUILD                 render on Slurm; the job then labels and encodes
    neu-movie animate SPEC                  build, then submit
    neu-movie relabel BUILD                 rewrite labels from the build's spec.toml ...
    neu-movie finish  BUILD                 ... then encode the movie, and a labelled one

A build is the folder one `build` makes: `renders/<spec name>_<timestamp>/` beside the spec.
Site settings (Blender, neuVid, Slurm) come from `configs/neu-movie.toml`; see
`neu_movie.site`.
"""

import argparse
import sys
from pathlib import Path

from neu_movie import build as build_mod
from neu_movie import compose, links, schedule
from neu_movie import site as site_mod
from neu_movie import spec as spec_mod


def _camera(args):
    from neu_glance.state import load_state
    v = compose.view_from_state(load_state(args.state))

    def numbers(xs):
        return "[" + ", ".join(f"{x:g}" for x in xs) + "]"
    print("[view]")
    print(f"position = {numbers(v['position'])}       # nm")
    print(f"orientation = {numbers(v['orientation'])}")
    print(f"height = {v['height']:g}                  # nm, top to bottom at `position`")
    print()
    print("# in [camera], to orbit about this view's vertical instead of the opening one's:")
    print(f"# up = {numbers(v['up'])}")
    return 0


def _check(args):
    spec = spec_mod.load(args.spec)
    world = build_mod.resolve(spec, site_mod.load(Path(args.spec).parent), limit=args.limit)
    ids, colours = world["ids"], world["colours"]
    print(f"\n{spec['name']}: {spec['mode']}, camera {spec['camera']['style']}, "
          f"{sum(len(v) for v in ids.values())} bodies")
    for i, beat in enumerate(spec["beats"]):
        names = [f"{n} ({len(ids[n])}, {colours[n]})" if n in ids else f"{n} (none)"
                 for n in beat["sets"]]
        print(f"  {schedule.beat_start(spec, i):6.1f} s  {beat['title'] or '-':20s} "
              f"{', '.join(names)}")
    scene = {"unit": world["unit"], "ids": ids, "colours": colours, "neuron_dir": "-",
             "roi_dir": "-", "head": {"dir": "-", "roi": "-"} if spec["head"] else None,
             "neuropil_colours": {r: v["color"] for r, v in world["neuropils"].items()}}
    _, _, info = compose.compose(spec, scene)
    print(f"  {info['seconds']:g} s at {spec['fps']} fps = {info['frames']} frames")
    if args.links:
        for title, url in links.links(spec, world):
            print(f"\n# {title}\n{url}")
    return 0


def build_parser():
    ap = argparse.ArgumentParser(prog="neu-movie", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("camera", help="print a spec's [view] from a neuroglancer state")
    p.add_argument("state", help="a neuroglancer state JSON file or link")

    p = sub.add_parser("check", help="resolve sets and show the schedule; no Blender")
    p.add_argument("spec")
    p.add_argument("--links", action="store_true", help="print a neuroglancer link per beat")
    p.add_argument("--limit", type=int, help="cap every set at this many bodies")

    for name, text in (("build", "make a build: meshes, neuVid input, Blender files"),
                       ("animate", "build, then submit")):
        p = sub.add_parser(name, help=text)
        p.add_argument("spec")
        p.add_argument("--limit", type=int, help="cap every set at this many bodies (tests)")
        p.add_argument("--force", action="store_true",
                       help="re-export meshes already in the cache")
        if name == "animate":
            p.add_argument("--time", help="Slurm time limit (default: the site config's)")
            p.add_argument("--dry-run", action="store_true",
                           help="write the job script but do not submit it")

    p = sub.add_parser("stills", help="render a few frames on this machine")
    p.add_argument("build")
    p.add_argument("-f", "--frames", type=int, nargs="+", required=True)

    p = sub.add_parser("submit", help="render a build on Slurm, then label and encode")
    p.add_argument("build")
    p.add_argument("--time", help="Slurm time limit (default: the site config's)")
    p.add_argument("--frames", help="FIRST-LAST, to render part of it (no movie then)")
    p.add_argument("--dry-run", action="store_true",
                   help="write the job script but do not submit it")

    p = sub.add_parser("relabel", help="rewrite a build's labels from its spec.toml")
    p.add_argument("build")

    p = sub.add_parser("finish", help="encode a build's movie, and a labelled one")
    p.add_argument("build")
    p.add_argument("--processes", type=int, default=4, help="label passes side by side")
    p.add_argument("--labels", action=argparse.BooleanOptionalAction, default=True,
                   help="also label the frames and encode <name>_labeled.mp4 (default: yes)")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "camera":
            return _camera(args)
        if args.command == "check":
            return _check(args)
        if args.command in ("build", "animate"):
            d = build_mod.build(args.spec, limit=args.limit, force=args.force)
            if args.command == "animate":
                build_mod.submit(d, time_limit=args.time, dry_run=args.dry_run)
            else:
                print(d)
            return 0
        if args.command == "stills":
            print(build_mod.stills(args.build, args.frames))
            return 0
        if args.command == "submit":
            first, last = 1, None
            if args.frames:
                first, last = (int(x) for x in args.frames.split("-"))
            build_mod.submit(args.build, time_limit=args.time, first=first, last=last,
                             dry_run=args.dry_run)
            return 0
        if args.command == "relabel":
            build_mod.relabel(args.build)
            return 0
        if args.command == "finish":
            build_mod.finish(args.build, args.processes, labels=args.labels)
            return 0
    except (spec_mod.SpecError, FileNotFoundError, FileExistsError, RuntimeError) as e:
        print(f"neu-movie: {e}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
