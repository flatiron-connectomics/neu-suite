"""A layer to host: an array, how to show it, and its translation into neuroglancer terms.

:class:`ServedLayer` is what :func:`neu_glance.serve` and :meth:`neu_glance.Server.add_layer`
take. ``neuroglancer`` is imported only inside functions (the ``serve`` extra).

**The volume type comes from ``kind``, never from the dtype.** neuroglancer would guess that
a uint8 label array is an image, averaging label ids and losing the segment UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from neu_lib import Piece

#: What a served array can be (neu-lib's list). To neuroglancer a probability map is an
#: image with a different default shader.
from neu_lib import KINDS


class ServeProblem(RuntimeError):
    """The arrays or options given cannot be served as asked."""


def _as_served_layer(layer: Any) -> ServedLayer:
    """A :class:`ServedLayer`, :class:`neu_lib.Piece` or mapping, as a :class:`ServedLayer`.
    Shared by :func:`serve` and :meth:`Server.add_layer`, so they accept the same inputs."""
    if isinstance(layer, ServedLayer):
        return layer
    if isinstance(layer, Piece):
        return ServedLayer.from_piece(layer)
    if isinstance(layer, Mapping):
        return ServedLayer(**layer)
    raise ServeProblem(
        f"a layer is a ServedLayer, a neu_lib.Piece, or the dict form of a ServedLayer — "
        f"got {type(layer).__name__}. Build one with ServedLayer.from_array, .from_hdf5, "
        f".from_volume, .from_piece or .from_source.")


def _neuroglancer():
    """The module, or an error naming the extra. Imported here, never at module scope."""
    try:
        import neuroglancer
    except ImportError as e:
        raise ServeProblem(
            "serving needs the `neuroglancer` package, which is an optional extra: "
            "`pip install 'neu-glance[serve]'`. Building a state or a link needs none of "
            "it — only hosting one does."
        ) from e
    return neuroglancer


@dataclass
class ServedLayer:
    """One array to host, and how to show it.

    ``frame`` is a :class:`neu_lib.Frame` (voxel size and origin in nm). Without it a crop
    is placed at nm zero instead of over the volume it came from.

    ``color`` sets the shader's single colour, for shaders that have one, such as the
    ``probability`` default (red, with opacity carrying the value); it accepts anything
    :func:`~neu_glance.shaders.as_hex_color` does. ``opacity`` is the layer's overall
    opacity, multiplied with the shader's per-voxel alpha.

    ``channel_axis`` (a leading, channel-first axis) follows from the array's rank: 4-D has
    one, 3-D does not. Passing it is allowed but checked.
    """
    array: Any
    kind: str = "image"
    name: str | None = None
    frame: Any = None
    shader: str | None = None
    channel_axis: bool | None = None
    opacity: float | None = None
    color: Any = None

    # ------------------------------------------------------------ constructors
    #
    # These take what the source records (frame, name, channel axis) and require what it
    # does not, notably `kind`, which is never guessed from the dtype.

    @classmethod
    def from_array(cls, array: Any, kind: str, *, name: str | None = None,
                   frame: Any = None, voxel_size: Sequence[float] | None = None,
                   origin: Sequence[float] | None = None,
                   channel_axis: bool | None = None, **kwargs) -> "ServedLayer":
        """An in-memory array. ``voxel_size``/``origin`` build the frame for you.

            ServedLayer.from_array(prob, "probability", voxel_size=(40, 8, 8))

        ``channel_axis`` follows from the array's rank (see :class:`ServedLayer`).
        """
        from neu_lib import Frame

        if frame is None and voxel_size is not None:
            frame = Frame(voxel_size_nm=tuple(float(v) for v in voxel_size),
                          origin_nm=tuple(float(o) for o in (origin or (0.0, 0.0, 0.0))))
        elif frame is not None and (voxel_size is not None or origin is not None):
            raise ServeProblem(
                "pass either frame= or voxel_size=/origin=, not both — two frames for one "
                "array is a disagreement nothing here can resolve")
        return cls(array=array, kind=kind, name=name, frame=frame,
                   channel_axis=channel_axis, **kwargs)

    @classmethod
    def _read(cls, src: Any, kind: str | None, name: str | None,
              **read_kwargs) -> "ServedLayer":
        """Read through ``neu_vol.read_piece`` and wrap it. Shared by the three readers."""
        from neu_vol import read_piece

        try:
            piece = read_piece(src, kind, **read_kwargs)
        except ValueError as e:
            # Translated so callers can catch ServeProblem.
            raise ServeProblem(str(e)) from None
        # The name comes from the piece, which read_piece derives from the source.
        return cls.from_piece(piece, name=name)

    @classmethod
    def from_hdf5(cls, path: str, dataset: str | None = None, kind: str | None = None, *,
                  crop: Any = None, name: str | None = None,
                  voxel_size: Sequence[float] | None = None, **kwargs) -> "ServedLayer":
        """One dataset of an HDF5 file, with the frame it records about itself.

            ServedLayer.from_hdf5("piece.h5", kind="segmentation")
            ServedLayer.from_hdf5("gt.h5", "/z07901", "segmentation")

        ``dataset`` may be omitted when the file holds exactly one 3D+ array. The frame comes
        from the file's ``voxel_size`` / ``axes`` / ``voxel_offset`` attributes, as
        ``neu-vol to-hdf5`` writes them; ``voxel_size`` overrides, and is required if the file
        records none. ``kind`` is required, since HDF5 has no standard place to record it.
        The format is forced to HDF5, whatever the extension.
        """
        return cls._read(str(path), kind, name, dataset=dataset, src_format="hdf5",
                         crop=crop, voxel_size=voxel_size, **kwargs)

    @classmethod
    def from_volume(cls, volume: str, kind: str | None = None, *, level: int = 0,
                    crop: Any = None, name: str | None = None,
                    voxel_size: Sequence[float] | None = None,
                    **kwargs) -> "ServedLayer":
        """A box out of a stored volume — zarr, precomputed, an image stack.

            ServedLayer.from_volume("s3://my-bucket/seg_v1", level=1,
                                    crop=((0, 0, 0), (64, 512, 512)))

        ``kind`` defaults to what the volume records (precomputed's ``info["type"]``,
        OME-NGFF's multiscales ``type``) and is required only where it records nothing. A
        whole volume is usually too large for memory, so pass ``crop``.
        """
        return cls._read(volume, kind, name, level=level, crop=crop,
                         voxel_size=voxel_size, **kwargs)

    @classmethod
    def from_piece(cls, piece: Any, kind: str | None = None, *, name: str | None = None,
                   **kwargs) -> "ServedLayer":
        """A :class:`neu_lib.Piece` — an array that already carries its frame and kind.

            piece = neu_vol.read_piece("gt.h5:/vol_03700", "segmentation")
            layer = ServedLayer.from_piece(piece)

        ``kind`` and ``name`` default to the piece's own. If the piece's ``kind`` is
        ``None``, pass one: it is not guessed from the dtype.
        """
        kind = kind or getattr(piece, "kind", None)
        if kind is None:
            raise ServeProblem(
                f"this source records no image/segmentation type, so kind= is required — "
                f"one of {', '.join(KINDS)}. It is not inferred from the dtype: "
                f"neuroglancer's own guess reads a uint8 label array as an image, which "
                f"averages label ids on downsample and loses the colour hashing and the "
                f"selection UI. (A precomputed volume records the type in its `info`, and "
                f"then this is not needed.)")
        return cls(array=piece.array, kind=kind,
                   name=name or getattr(piece, "name", None), frame=piece.frame,
                   **kwargs)

    @classmethod
    def from_source(cls, src: str, kind: str | None = None, *, level: int = 0,
                    crop: Any = None, name: str | None = None,
                    voxel_size: Sequence[float] | None = None,
                    **kwargs) -> "ServedLayer":
        """Anything readable, addressed as ``PATH`` or ``PATH:/DATASET``.

        The form the CLI takes. The format is detected, and a trailing ``:/name`` selects a
        dataset in an HDF5 file (only a colon followed by a slash, so ``s3://`` is safe).
        """
        return cls._read(src, kind, name, level=level, crop=crop,
                         voxel_size=voxel_size, **kwargs)

    # ------------------------------------------------------------ where it is
    @property
    def piece(self) -> Any:
        """This layer's array and frame as a :class:`neu_lib.Piece`, for ``.bbox``,
        ``.bounds_nm`` and ``.crop()``."""
        from neu_lib import Piece

        if self.frame is None:
            raise ServeProblem(
                f"layer {self.name!r} has no frame, so it has no position to report. "
                f"Build it with a voxel_size= or from a source that records one")
        return Piece(array=self.array, frame=self.frame, kind=self.kind)

    @property
    def bbox(self) -> Any:
        """Where this layer sits, in its frame's voxels. See :attr:`piece`."""
        return self.piece.bbox

    @property
    def bounds_nm(self) -> Any:
        """Where this layer sits, in nanometres. Also what ``crop=`` reads off a layer."""
        return self.piece.bounds_nm

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ServeProblem(f"kind must be one of {KINDS}, got {self.kind!r}")
        if not hasattr(self.array, "shape") or not hasattr(self.array, "dtype"):
            raise ServeProblem(
                f"layer {self.name!r}: expected an array with .shape and .dtype, got "
                f"{type(self.array).__name__}")
        rank = len(self.array.shape)
        if rank not in (3, 4):
            raise ServeProblem(
                f"layer {self.name!r}: a served volume is 3 spatial axes, optionally with "
                f"a leading channel axis, so 3-D or 4-D — got {rank}-D "
                f"{tuple(self.array.shape)}")
        derived = rank == 4
        if self.channel_axis is None:
            self.channel_axis = derived
        elif bool(self.channel_axis) != derived:
            raise ServeProblem(
                f"layer {self.name!r}: channel_axis={self.channel_axis} contradicts a "
                f"{rank}-D array {tuple(self.array.shape)}, which has "
                f"{'a' if derived else 'no'} leading channel axis. Leave it out — the rank "
                f"decides it")


def _coordinate_space(ng, frame: Any = None, *, channel_axis: bool = False,
                     name: str | None = None) -> Any:
    """A ``CoordinateSpace`` (in nm) from a ``Frame``, or a unit one if there is none. With
    ``channel_axis``, a leading local ``c^`` axis is added."""
    voxel = (1.0, 1.0, 1.0)
    if frame is not None:
        voxel = tuple(float(v) for v in frame.voxel_size_nm)
        if len(voxel) != 3:
            raise ServeProblem(
                f"layer {name!r}: frame has {len(voxel)} axes, expected 3 (z, y, x)")
    names, scales, units = ["z", "y", "x"], list(voxel), ["nm"] * 3
    if channel_axis:
        # A trailing `^` marks a local dimension, so the shader gets every channel instead
        # of the viewer offering a channel slider.
        names, scales, units = ["c^"] + names, [1.0] + scales, [""] + units
    return ng.CoordinateSpace(names=names, scales=scales, units=units)


def _voxel_offset(frame: Any = None, *, channel_axis: bool = False,
                  name: str | None = None) -> list[int] | None:
    """The origin in whole voxels, or ``None``. Raises if it is not whole, rather than
    rounding and shifting the volume by up to half a voxel."""
    if frame is None:
        return None
    origin = tuple(float(o) for o in frame.origin_nm)
    if not any(origin):
        return None
    voxel = tuple(float(v) for v in frame.voxel_size_nm)
    offset = []
    for o, v in zip(origin, voxel):
        exact = o / v
        if abs(exact - round(exact)) > 1e-6:
            raise ServeProblem(
                f"layer {name!r}: origin {origin} nm is not a whole number of "
                f"{voxel} nm voxels ({exact:g} on one axis), so it cannot be expressed as "
                f"a voxel offset")
        offset.append(int(round(exact)))
    return ([0] + offset) if channel_axis else offset


def _volume_type(layer: ServedLayer) -> str:
    """``segmentation`` or ``image``, from ``kind`` alone — never from the dtype."""
    return "segmentation" if layer.kind == "segmentation" else "image"


def _shader_for(layer: ServedLayer) -> str | None:
    """The GLSL for this layer, or ``None`` to leave neuroglancer's default.

    A segmentation gets none, keeping neuroglancer's label colours and selection UI.
    """
    from .shaders import image_shader

    if layer.kind == "segmentation":
        return None
    return image_shader(layer.shader, kind=layer.kind, channels=_channels(layer))


def _shader_controls(layer: ServedLayer) -> dict[str, Any]:
    """``shaderControls`` for this layer (only ``color`` so far), set in the state so every
    layer shares the built-in shader code."""
    from .shaders import ShaderProblem, as_hex_color, shader_color_control

    if layer.color is None:
        return {}
    if layer.kind == "segmentation":
        raise ShaderProblem(
            "color= is for an image or probability layer; a segmentation is coloured by "
            "neuroglancer's own label hashing, which is what makes two touching bodies "
            "distinguishable. Pick colours in the viewer's segment list instead.")
    control = shader_color_control(layer.shader, kind=layer.kind,
                                   channels=_channels(layer))
    return {control: as_hex_color(layer.color, "color")}


def _channels(layer: ServedLayer) -> int:
    return int(layer.array.shape[0]) if layer.channel_axis else 1
