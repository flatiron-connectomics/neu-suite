"""Intermediate per-(body, block) fragments (stage 1 -> stage 2 handoff).

Stage 1 writes one fragment per body per block under
``<chunked_dir>/h<bucket>/<body_id>/<iz>_<iy>_<ix>.<fmt>``, ``bucket`` being
``body_id % FANOUT`` in three hex digits. Stage 2 finds bodies by listing the buckets.
Filesystem-backed.

**Buckets keep directories narrow**: one directory per body directly under
``chunked_dir`` made listing a large run hang. They do not reduce the file count. The
``h`` prefix keeps a bucket from parsing as a body id, which is how the old flat layout
is recognized and refused.

Meshes (``.drc``) and skeletons (``.skel``) share the layout, each in its own store.
Skeleton fragments use the precomputed encoding but hold **zyx** global-nm vertices, the
space stage 2 fuses in; the zyx->xyz flip happens once, at output.
"""

from __future__ import annotations

import glob
import os
from typing import Sequence


#: Buckets per fragment store (~250 entries per directory for a million bodies).
FANOUT = 4096


class OldFragmentLayout(RuntimeError):
    """The store holds body directories at its top level, from before bucketing."""


def bucket_name(body_id: int) -> str:
    return f"h{int(body_id) % FANOUT:03x}"


def body_dir(chunked_dir: str, body_id: int) -> str:
    return os.path.join(chunked_dir, bucket_name(body_id), str(int(body_id)))


def _buckets(chunked_dir: str) -> list[str]:
    """The bucket directories, refusing a store in the old flat layout, whose bodies
    stage 2 would not see."""
    # scandir's entry types come with the listing; isdir per entry is a stat each on ceph.
    with os.scandir(chunked_dir) as entries:
        dirs = [e.name for e in entries if e.is_dir()]
    old = next((n for n in dirs if n.isdigit()), None)
    if old is not None:
        raise OldFragmentLayout(
            f"{chunked_dir} holds body directories at its top level (e.g. {old}/), the "
            f"layout before fragment stores were bucketed. Delete the store and rerun "
            f"stage 1 (it is scratch), or finish the run with the neu-morpho that wrote it.")
    return [os.path.join(chunked_dir, n) for n in dirs if n.startswith("h")]


def count_bodies(chunked_dir: str) -> int | None:
    """How many bodies have fragments. ``None`` if the store does not exist.

    On CephFS, from the store's recursive directory counts, which are instant (and may
    lag the filesystem by seconds); elsewhere one listing per bucket, which on ceph takes
    tens of seconds for a 4096-bucket store. Never a walk into the body directories.
    """
    if not os.path.isdir(chunked_dir):
        return None
    buckets = _buckets(chunked_dir)               # also refuses the old flat layout
    try:
        # rsubdirs counts the store itself, its top-level directories and everything
        # below them, and below the top level every directory is a body.
        below = int(os.getxattr(chunked_dir, "ceph.dir.rsubdirs"))
        top = int(os.getxattr(chunked_dir, "ceph.dir.subdirs"))
        return below - 1 - top
    except (AttributeError, OSError, ValueError):
        return sum(sum(1 for n in os.listdir(b) if n.isdigit()) for b in buckets)


def fragment_path(chunked_dir: str, body_id: int, block_index: Sequence[int], fmt: str = "drc") -> str:
    iz, iy, ix = block_index
    return os.path.join(body_dir(chunked_dir, body_id), f"{iz}_{iy}_{ix}.{fmt}")


def write_fragment(chunked_dir: str, body_id: int, block_index: Sequence[int], mesh, fmt: str = "drc") -> None:
    path = fragment_path(chunked_dir, body_id, block_index, fmt)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mesh.serialize(path)                                  # fmt inferred from extension


def list_bodies(chunked_dir: str) -> list[int]:
    if not os.path.isdir(chunked_dir):
        return []
    return sorted(int(d) for b in _buckets(chunked_dir) for d in os.listdir(b) if d.isdigit())


def read_body_fragments(chunked_dir: str, body_id: int, fmt: str = "drc") -> list:
    """Load all of a body's fragment meshes (``vol2mesh.Mesh``)."""
    from vol2mesh import Mesh

    paths = sorted(glob.glob(os.path.join(body_dir(chunked_dir, body_id), f"*.{fmt}")))
    return [Mesh.from_file(p) for p in paths]


# --------------------------------------------------------------------------- #
# Skeleton fragments (zyx vertices, global nm — see the module docstring)
# --------------------------------------------------------------------------- #
def write_skel_fragment(chunked_dir: str, body_id: int, block_index: Sequence[int],
                        skeleton, fmt: str = "skel") -> None:
    path = fragment_path(chunked_dir, body_id, block_index, fmt)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(skeleton.to_precomputed())
    os.replace(tmp, path)          # atomic: a killed worker leaves no half fragment


def read_body_skel_fragments(chunked_dir: str, body_id: int, fmt: str = "skel") -> list:
    """Load all of a body's skeleton fragments (``osteoid.Skeleton``, zyx nm)."""
    from osteoid import Skeleton

    paths = sorted(glob.glob(os.path.join(body_dir(chunked_dir, body_id), f"*.{fmt}")))
    out = []
    for p in paths:
        with open(p, "rb") as f:
            out.append(Skeleton.from_precomputed(f.read(), segid=int(body_id)))
    return out
