"""The site config: where Blender, neuVid and the volume library are, and how to ask Slurm for
a GPU node. Nothing here is a default for any real site; each site writes its own.

Found as ``$NEU_MOVIE_CONFIG``, else the first ``configs/neu-movie.toml`` walking up from the
spec (or build) and then from the working directory. Relative paths in it are relative to
the file. Keys::

    blender = "/path/to/blender"             # the Blender executable
    neuvid = "/path/to/neuVid/neuVid"        # neuVid's script directory (importMeshes.py, ...)
    library = "library"                      # volume library directory, for `library =` names
    meshes = "/path/to/mesh-cache"           # exported OBJ meshes, shared by every build
    ffmpeg_setup = "module load ffmpeg"      # shell run first when ffmpeg is not on PATH

    [slurm]
    options = ["--partition=gpu", "--gpus-per-node=4", "--cpus-per-task=64", "--mem=64G"]
    time = "03:00:00"                        # default; `submit --time` overrides

``blender`` and ``neuvid`` must be visible from the compute nodes, as must the python running
neu-movie, which the render job calls back to label and encode.
"""

import os
import tomllib
from pathlib import Path

ENV_VAR = "NEU_MOVIE_CONFIG"
NAME = Path("configs") / "neu-movie.toml"


class SiteError(RuntimeError):
    pass


def find(*starts):
    if os.environ.get(ENV_VAR):
        return Path(os.environ[ENV_VAR])
    for start in [*starts, Path.cwd()]:
        for d in [Path(start).resolve(), *Path(start).resolve().parents]:
            if (d / NAME).is_file():
                return d / NAME
    return None


class Site:
    def __init__(self, path):
        self.path = Path(path)
        with open(self.path, "rb") as f:
            self.raw = tomllib.load(f)
        self.slurm = dict(self.raw.get("slurm", {}))

    def _path(self, key, required=True):
        value = self.raw.get(key)
        if value is None:
            if required:
                raise SiteError(f"{self.path} sets no `{key}`")
            return None
        return (self.path.parent / os.path.expanduser(value)).resolve()

    @property
    def blender(self):
        return self._path("blender")

    @property
    def neuvid(self):
        return self._path("neuvid")

    @property
    def library(self):
        return self._path("library", required=False)

    @property
    def meshes(self):
        return self._path("meshes")

    @property
    def ffmpeg_setup(self):
        return self.raw.get("ffmpeg_setup", "")


def load(*starts):
    path = find(*starts)
    if path is None:
        raise SiteError(f"no site config: set ${ENV_VAR} or write {NAME} above the spec "
                        f"(see `neu_movie.site`)")
    return Site(path)
