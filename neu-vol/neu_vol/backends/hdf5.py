"""HDF5-backed reader (h5py).

Reads one dataset of an HDF5 file as a volume, in canonical ``(z, y, x)`` /
``(c, z, y, x)`` order. **Reading only**; :meth:`HDF5Backend.write_region` says where HDF5
output is written instead. Each instance holds an open file, reopened from its spec on each
dask worker.

HDF5 is a container, so a path alone is ambiguous: :func:`sole_dataset` picks the dataset
when there is exactly one. The optional ``stored_offset`` / ``stored_voxel_size`` /
``stored_axes`` / ``stored_units`` / ``stored_kind`` readers return the frame a writer
recorded beside the array, which ``source_metadata`` composes into the same metadata a
precomputed ``info`` yields. An axis order the file states but nobody can interpret is an
error, never a guess: reversed, the data lands mirrored through the z=x diagonal.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import Region, register_backend

TAG = "hdf5"

#: Extensions that name an HDF5 container. Also what ``detect.detect_file_backend`` goes
#: by, since HDF5 has no marker object to probe for.
HDF5_EXTENSIONS = (".h5", ".hdf5", ".hdf", ".he5")


def _as_offset(value: Any, where: str) -> tuple[int, ...]:
    """A stored offset as whole voxels. Anything else is an error, not a rounding."""
    arr = np.asarray(value).ravel()
    if arr.size == 0:
        raise ValueError(f"{where} is empty")
    if not np.all(arr == np.rint(arr.astype("float64"))):
        raise ValueError(f"{where} is {arr.tolist()}, which is not whole voxels")
    return tuple(int(v) for v in np.rint(arr.astype("float64")))


def require_local_path(path: str, what: str = "an HDF5 file") -> str:
    """``path`` if it is an ordinary filesystem path; a useful error if it is not.

    h5py has no object-store driver; without this check it tries to create a local file
    called ``s3://bucket/piece.h5`` and reports ``errno = 2``.
    """
    from ..location import is_local, local_path

    if not is_local(path):
        raise ValueError(
            f"{what} must be an ordinary filesystem path, and {path!r} is not: h5py has no "
            f"object-store driver, so nothing here can read or write HDF5 on one. Copy the "
            f"file to local storage first (`rclone copy`, `aws s3 cp`), or write it locally "
            f"and upload it afterwards.")
    # Resolved, not just vetted: h5py does not expand `~`.
    return local_path(path)


def _as_floats(value: Any, where: str) -> tuple[float, ...]:
    """A stored physical vector as floats. Unlike an offset, these are not whole voxels."""
    arr = np.asarray(value).ravel()
    if arr.size == 0:
        raise ValueError(f"{where} is empty")
    return tuple(float(v) for v in arr.astype("float64"))


def axes_string(value: Any) -> str:
    """An axis-order attribute as ``"zyx"``, however h5py handed it over.

    ``"zyx"``, ``b"zyx"``, ``["z","y","x"]`` and ``[b"z",b"y",b"x"]`` are all accepted.
    """
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, str):
        return value.replace(" ", "")
    return "".join(v.decode() if isinstance(v, bytes) else str(v)
                   for v in np.asarray(value).ravel())


def datasets(path: str, *, min_ndim: int = 3) -> list[str]:
    """Every dataset in the file with at least ``min_ndim`` dimensions, by path."""
    import h5py

    found: list[str] = []
    with h5py.File(require_local_path(path), "r") as f:
        f.visititems(lambda name, obj: found.append("/" + name)
                     if isinstance(obj, h5py.Dataset) and obj.ndim >= min_ndim else None)
    return found


#: The attribute names read as a frame, and what each one means. Reported when a file
#: records none, so "no metadata" and "spelled differently" can be told apart.
FRAME_ATTRIBUTES = {
    "voxel_size": "physical size of one voxel, in `units`",
    "voxel_offset": "where this array starts, in whole voxels",
    "offset": "the same origin in physical units",
    "units": "what `voxel_size` is measured in (this suite works in nm)",
    "axes": "the axis order of the three vectors above ('zyx' or 'xyz')",
}

#: What :func:`describe_datasets` reports on top of the frame. ``kind`` is not a frame
#: attribute: a file can state what its voxels mean and still have no physical scale.
DESCRIBED_ATTRIBUTES = {
    **FRAME_ATTRIBUTES,
    "kind": "what the voxels mean: image, probability or segmentation",
}


def describe_datasets(path: str, *, min_ndim: int = 3) -> dict[str, dict]:
    """Every volumetric dataset in the file, with its geometry and recorded frame.

    ``{name: {"shape", "dtype", "chunks", "ndim", <frame attributes>}}``, in file order,
    from one file open, attributes only. What ``neu-vol info`` lists for a container of
    several arrays, such as a file of ground-truth crops.

    Frame values come back **as recorded**, each with its own ``axes``, not reoriented to
    zyx (``source_metadata._read_hdf5`` does that, strictly, for one array). Values fall
    back to the root group's.
    """
    import h5py

    out: dict[str, dict] = {}
    with h5py.File(require_local_path(path), "r") as f:
        root = {k: v for k, v in f.attrs.items() if k in DESCRIBED_ATTRIBUTES}

        def visit(name, obj):
            if not isinstance(obj, h5py.Dataset) or obj.ndim < min_ndim:
                return
            entry = {"shape": tuple(int(s) for s in obj.shape),
                     "dtype": str(obj.dtype),
                     "chunks": tuple(int(c) for c in obj.chunks) if obj.chunks else None,
                     "ndim": int(obj.ndim)}
            for key in DESCRIBED_ATTRIBUTES:
                if key in obj.attrs:
                    value = obj.attrs[key]
                elif key in root:
                    value = root[key]
                else:
                    continue
                if key in ("units", "axes", "kind"):
                    entry[key] = (axes_string(value) if key == "axes"
                                  else (value.decode() if isinstance(value, bytes)
                                        else str(value)))
                else:
                    entry[key] = tuple(float(v) for v in np.asarray(value).ravel())
            out["/" + name] = entry

        f.visititems(visit)
    return out


def sole_dataset(path: str) -> str:
    """The file's one volumetric dataset, for callers that did not name one.

    Answers only when there is exactly one 3D+ dataset; otherwise raises listing them and
    saying how to choose one.
    """
    found = datasets(path)
    if len(found) == 1:
        return found[0]
    if not found:
        raise KeyError(f"{path} contains no 3D+ dataset")
    listed = ", ".join(sorted(found))
    raise KeyError(
        f"{path} contains {len(found)} volumetric datasets ({listed}); say which one — "
        f"`neu-vol info {path}` lists them with their shapes and offsets, then pass "
        f"dataset= (`open_hdf5`, `describe`), "
        f"--dataset (`info`), --dataset (`write`), --src-dataset (`to-hdf5`), or "
        f"'dataset' in a backend spec")


def open_hdf5(path: str, dataset: str | None = None) -> "HDF5Backend":
    """One dataset of an HDF5 file, ready to read regions from.

    The front door for a file you are pointing at by hand::

        be = open_hdf5("piece.h5")                      # one dataset: found
        be = open_hdf5("gt_v1_eval.h5", "/z07901")      # a container: name it
        be.read_region((slice(0, 64), slice(0, 64), slice(0, 64)))

    ``dataset`` is optional only when the file holds exactly one 3D+ array; with several
    it is required, and the error lists them (:func:`sole_dataset`).

    Not a general "open anything" helper: the format is in the name, so nothing is
    detected. To find out what something is, use :func:`~neu_vol.source_metadata.describe`.
    Goes through :func:`~neu_vol.backends.base.open_backend`, so the open file joins the
    per-process cache.
    """
    from ..logs import quiet_reads
    from .base import open_backend

    # A notebook entry point, so it filters store logs itself.
    with quiet_reads():
        path = require_local_path(path)
        return open_backend({"backend": TAG, "path": path,
                             "dataset": dataset if dataset is not None
                             else sole_dataset(path)})


class HDF5Backend:
    """View over a single dataset in an HDF5 file."""

    def __init__(self, spec: Mapping[str, Any]):
        import h5py

        self._spec = dict(spec)
        self._path = require_local_path(str(spec["path"]))
        self._dataset = str(spec.get("dataset", "/data"))
        self._file = h5py.File(self._path, "r")
        if self._dataset not in self._file:
            self._file.close()
            raise KeyError(f"dataset {self._dataset!r} not found in {self._path}")
        self._dset = self._file[self._dataset]

    @classmethod
    def open(cls, spec: Mapping[str, Any]) -> "HDF5Backend":
        return cls(spec)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(s) for s in self._dset.shape)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self._dset.dtype)

    @property
    def chunks(self) -> tuple[int, ...]:
        # HDF5 storage chunks if chunked, else treat the whole array as one chunk.
        return tuple(self._dset.chunks) if self._dset.chunks else self.shape

    def stored_offset(self, name: str = "voxel_offset"):
        """A voxel offset recorded in the file as ``(value, where_it_was_found)``.

        ``None`` if the file records none. Searched in the dataset's attributes, then the
        root group's, then a top-level dataset of that name.

        **The axis order is not knowable from here**: the name is precomputed's, where it
        means xyz. The numbers come back as stored and the caller decides, using
        :meth:`stored_axes` when the file has it.
        """
        import h5py

        for attrs, where in ((self._dset.attrs, f"{self._dataset}.attrs[{name!r}]"),
                             (self._file.attrs, f"/.attrs[{name!r}]")):
            if name in attrs:
                return _as_offset(attrs[name], where), where
        node = self._file.get(name)
        if isinstance(node, h5py.Dataset):
            return _as_offset(node[()], f"/{name}"), f"/{name}"
        return None

    def stored_voxel_size(self, name: str = "voxel_size"):
        """A physical voxel size recorded in the file, as ``(value, where)``, or ``None``.

        Searched as :meth:`stored_offset` is. ``name`` is a parameter because another
        writer may call it something else.
        """
        import h5py

        for attrs, where in ((self._dset.attrs, f"{self._dataset}.attrs[{name!r}]"),
                             (self._file.attrs, f"/.attrs[{name!r}]")):
            if name in attrs:
                return _as_floats(attrs[name], where), where
        node = self._file.get(name)
        if isinstance(node, h5py.Dataset):
            return _as_floats(node[()], f"/{name}"), f"/{name}"
        return None

    def stored_axes(self, name: str = "axes"):
        """The axis order the writer recorded, as ``(order, where)``, or ``None``.

        What makes :meth:`stored_offset`'s numbers unambiguous; ``neu-vol to-hdf5`` writes
        it. Dataset attributes first, then the root group's.
        """
        for attrs, where in ((self._dset.attrs, f"{self._dataset}.attrs[{name!r}]"),
                             (self._file.attrs, f"/.attrs[{name!r}]")):
            if name in attrs:
                return axes_string(attrs[name]), where
        return None

    def stored_units(self, name: str = "units"):
        """The physical unit the writer recorded, as ``(unit, where)``, or ``None``.

        Attributes only, dataset before root. ``neu-vol to-hdf5`` records it beside
        ``voxel_size``; the suite works in nm.
        """
        for attrs, where in ((self._dset.attrs, f"{self._dataset}.attrs[{name!r}]"),
                             (self._file.attrs, f"/.attrs[{name!r}]")):
            if name in attrs:
                value = attrs[name]
                return (value.decode() if isinstance(value, bytes) else str(value)), where
        return None

    def stored_kind(self, name: str = "kind"):
        """What the writer said the voxels MEAN, as ``(kind, where)``, or ``None``.

        Attributes only, dataset before root. Written by ``neu_vol.write_piece`` and
        ``to-hdf5`` (``ops/pack.KIND_ATTR``). The caller trusts only values in
        :data:`neu_lib.KINDS`, since another tool may use the word for something else.
        """
        for attrs, where in ((self._dset.attrs, f"{self._dataset}.attrs[{name!r}]"),
                             (self._file.attrs, f"/.attrs[{name!r}]")):
            if name in attrs:
                value = attrs[name]
                return (value.decode() if isinstance(value, bytes) else str(value)), where
        return None

    def close(self) -> None:
        """Release the open ``h5py.File``.

        Called by :func:`~neu_vol.backends.base.release_backends`: HDF5 refuses to open a
        file for writing while any handle holds it read-only. Idempotent.
        """
        self._file.close()

    def read_region(self, region: Region) -> np.ndarray:
        return np.asarray(self._dset[region])

    def write_region(self, region: Region, data: np.ndarray) -> None:
        """Never: HDF5 output is written whole, by a different path.

        The block engine writes into a volume; one array in a container has no pyramid and
        no chunk objects, and HDF5 has no concurrent writers. A piece is written in one
        call by :func:`neu_vol.write_piece` (an array in memory) or
        :func:`neu_vol.pack_hdf5` (``neu-vol to-hdf5``, from another source).
        """
        raise NotImplementedError(
            "an HDF5 file is written whole, not region by region: use "
            "`neu_vol.write_piece(piece, out)` for an array already in memory, or "
            "`neu_vol.pack_hdf5(src, out)` / `neu-vol to-hdf5` to stream one out of another "
            "source. Both record the frame and voxel_offset that `neu-vol write` needs to "
            "place the result back into a volume.")

    def to_spec(self) -> dict[str, Any]:
        return dict(self._spec)


register_backend(TAG, HDF5Backend.open)
