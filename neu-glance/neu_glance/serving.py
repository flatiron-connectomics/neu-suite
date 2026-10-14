"""Host arrays in this process and open a real neuroglancer on them.

What is hosted is a :class:`~neu_glance.served_layer.ServedLayer`; this module is the
server around it.

Runs the viewer's server inside the calling process, so an array that exists only in a
notebook (a model's probability map, a relabelled crop) can be viewed without writing a
volume. ``neuroglancer`` is imported only inside functions (the ``serve`` extra).

- **A served source is not portable.** Its URL is ``python://volume/<token>``, valid only
  while this process runs, so :meth:`Server.state` is for inspection, not for sharing.
- **The chunk encoding stays ``npz``.** In neuroglancer 2.41.2 the ``raw`` encoder calls
  ``ndarray.tostring``, removed in NumPy 2, and fails on the first chunk requested, after
  the viewer has opened.

:meth:`Server.on_click`, :meth:`Server.boxes`, :meth:`Server.points` and
:meth:`Server.selected_segments` read the browser's state back into the notebook;
:meth:`Server.annotate` adds a layer to draw in.
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Self, Sequence

import numpy as np

from .served_layer import (ServedLayer, ServeProblem, _as_served_layer, _coordinate_space,
                           _neuroglancer, _shader_controls, _shader_for, _volume_type,
                           _voxel_offset)

logger = logging.getLogger(__name__)

#: The chunk encoding; ``raw`` fails under NumPy 2 (see the module docstring).
ENCODING = "npz"

#: Default name of the annotation layer :meth:`Server.boxes` reads back. Opt-in:
#: ``serve(annotations=True)``, ``--annotate`` on the command line, or
#: :meth:`Server.annotate` later.
ANNOTATE_LAYER = "annotations"

#: The annotation tools that can be armed: the two kinds :meth:`Server.boxes` and
#: :meth:`Server.points` read back.
ANNOTATION_TOOLS = {"box": "annotateBoundingBox", "point": "annotatePoint"}


@dataclass
class Click:
    """Where a click landed, and what was under it.

    ``position`` is in the served frame's voxels, ``zyx``, as floats — the cursor is not on
    a voxel centre. ``values`` maps layer name to the value there, which for a segmentation
    is the label id.
    """
    position: tuple[float, ...]
    values: dict[str, Any] = field(default_factory=dict)
    state: Any = None

    @property
    def voxel(self) -> tuple[int, ...]:
        """``position`` floored to whole voxels, which is the voxel actually clicked."""
        import math

        return tuple(int(math.floor(v)) for v in self.position)


def _tool_type(tool: str | None) -> str | None:
    """neuroglancer's name for an annotation tool, or ``None`` to arm nothing."""
    if tool is None:
        return None
    try:
        return ANNOTATION_TOOLS[tool]
    except KeyError:
        raise ServeProblem(
            f"tool must be one of {', '.join(sorted(ANNOTATION_TOOLS))} or None, got "
            f"{tool!r}. Only these two are offered because they are the two `boxes()` and "
            f"`points()` can read back") from None


def _annotation_layer(ng, dimensions: Any, tool: str | None = "box") -> Any:
    """An empty ``local://annotations`` layer in ``dimensions``, with ``tool`` armed.

    ``dimensions`` must be the viewer's own coordinate space; a different one would read
    back every annotation at the wrong coordinates.
    """
    return ng.LocalAnnotationLayer(dimensions=dimensions, tool=_tool_type(tool))


def _faces_box(lo: Sequence[float], hi: Sequence[float]) -> Any:
    """The whole-voxel box containing a shape given by its faces (a box's corners, an
    ellipsoid's extent), rounded outward and at least one voxel wide."""
    from neu_lib import BBox

    low = [math.floor(v) for v in lo]
    return BBox(tuple(low), tuple(max(math.ceil(h), l + 1) for h, l in zip(hi, low)))


def _annotation_bbox(ann: Any) -> Any:
    """The whole-voxel :class:`neu_lib.BBox` containing one annotation, whatever type.

    Handles every neuroglancer annotation type, since the toolbar can place any; an unknown
    type raises. A point lies inside a voxel, so its box is ``floor(p) .. floor(p) + 1``; a
    box corner is a voxel boundary, so it rounds outward to that boundary only (otherwise
    enclosing a box would grow it on every call).
    """
    from neu_lib import BBox

    point = getattr(ann, "point", None)
    if point is not None:                                            # a point: a position
        return BBox.from_points([[float(v) for v in point]])
    a, b = getattr(ann, "point_a", None), getattr(ann, "point_b", None)
    if a is not None and b is not None:
        a = [float(v) for v in a]
        b = [float(v) for v in b]
        if isinstance(ann, _neuroglancer().LineAnnotation):           # two positions
            return BBox.from_points([a, b])
        return _faces_box([min(p, q) for p, q in zip(a, b)],          # a box: two faces
                          [max(p, q) for p, q in zip(a, b)])
    centre, radii = getattr(ann, "center", None), getattr(ann, "radii", None)
    if centre is not None and radii is not None:                     # an ellipsoid: faces
        c = [float(v) for v in centre]
        r = [abs(float(v)) for v in radii]
        return _faces_box([x - y for x, y in zip(c, r)], [x + y for x, y in zip(c, r)])
    points = getattr(ann, "points", None)                            # a polyline
    if points is not None:
        return BBox.from_points([[float(v) for v in p] for p in points])
    raise ServeProblem(
        f"cannot tell where a {type(ann).__name__} is, so it cannot be enclosed — this is "
        f"an annotation type this package does not know about")


def _margin(margin: Any, rank: int) -> tuple[int, ...]:
    """A per-axis margin in whole voxels, from a scalar or a zyx sequence."""
    values = [margin] * rank if isinstance(margin, (int, float)) else list(margin)
    if len(values) != rank:
        raise ServeProblem(
            f"margin has {len(values)} axes but the annotations have {rank} — pass one "
            f"number for every axis, or a single number for all of them")
    return tuple(int(round(float(v))) for v in values)


def _as_bbox(box: Any, what: str) -> Any:
    """A :class:`neu_lib.BBox` from one, or from a ``(lo, hi)`` pair."""
    from neu_lib import BBox

    if isinstance(box, BBox):
        return box
    try:
        lo, hi = box
        return BBox(tuple(lo), tuple(hi))
    except (TypeError, ValueError) as e:
        raise ServeProblem(f"{what} must be a BBox or a (lo, hi) pair: {e}") from None


def _spatial_axes(dimensions: Any) -> list[int]:
    """The indices of the navigable axes (a trailing ``^`` marks a layer-local axis)."""
    return [i for i, n in enumerate(dimensions.names) if not n.endswith("^")]


class Server:
    """A running viewer, and the handle the notebook keeps on it.

    Reads the browser's state back without needing neuroglancer's API. ``.viewer`` is the
    underlying ``neuroglancer.Viewer``, for anything not wrapped.
    """

    def __init__(self, viewer, volumes: dict[str, Any], annotations: str | None) -> None:
        self.viewer = viewer
        self.volumes = volumes
        self._annotations = annotations
        self._actions = 0

    # -------------------------------------------------------------- addressing
    @property
    def url(self) -> str:
        return self.viewer.get_viewer_url()

    def state(self) -> dict:
        """The viewer's current state as JSON.

        For inspection only: its sources are valid only in this process. To share a link,
        use ``neu-glance gen`` on a published volume.
        """
        return self.viewer.state.to_json()

    def _repr_html_(self) -> str:
        return (f'<a href="{self.url}" target="_blank">neuroglancer</a> '
                f'<code>{self.url}</code>')

    def __repr__(self) -> str:
        return f"<Server {self.url} layers={sorted(self.volumes)}>"

    # -------------------------------------------------------------- adding layers
    def _add_layer_in(self, s: Any, layer: ServedLayer,
                      name: str | None = None) -> tuple[str, Any]:
        """Host one array and add its layer in the open transaction ``s``. Returns
        ``(name, volume)``. Used by both :func:`serve` and :meth:`add_layer`; taking the
        transaction lets :func:`serve` push one state with all its layers."""
        ng = _neuroglancer()
        # Layers are keyed by name, so a taken name gets a numeric suffix.
        base = name or layer.name or layer.kind
        resolved, n = base, 1
        while resolved in s.layers:
            resolved, n = f"{base}_{n}", n + 1
        if resolved != base:
            logger.warning("layer name %r was taken; using %r", base, resolved)

        volume = ng.LocalVolume(
            layer.array,
            _coordinate_space(ng, layer.frame, channel_axis=layer.channel_axis,
                              name=resolved),
            volume_type=_volume_type(layer),
            voxel_offset=_voxel_offset(layer.frame, channel_axis=layer.channel_axis,
                                       name=resolved),
            encoding=ENCODING,
        )
        self.volumes[resolved] = volume
        if layer.kind == "segmentation":
            # Raises if color= was given for a segmentation, before anything is built.
            _shader_controls(layer)
            s.layers[resolved] = ng.SegmentationLayer(source=volume)
        else:
            kwargs: dict[str, Any] = {"source": volume}
            shader = _shader_for(layer)
            if shader:
                kwargs["shader"] = shader
            controls = _shader_controls(layer)
            if controls:
                kwargs["shader_controls"] = controls
            if layer.opacity is not None:
                kwargs["opacity"] = float(layer.opacity)
            s.layers[resolved] = ng.ImageLayer(**kwargs)
        return resolved, volume

    def add_layer(self, layer: Any, *, name: str | None = None,
                  color: Any = None) -> Self:
        """Host one more array in this viewer, live. Returns **the server**, so calls chain.

        The new layer appears in the tab already open, with no new URL::

            srv = (serve([em])
                   .add_layer(ServedLayer.from_hdf5("gt.h5", "/vol_03700", "segmentation"))
                   .add_layer(ServedLayer.from_array(pred, "probability",
                                                     voxel_size=(40, 8, 8)),
                              name="prediction"))

        Takes the same inputs as :func:`serve`. A name already in the viewer gets a numeric
        suffix; ``list(srv.volumes)[-1]`` is the name actually used. The view does not move,
        so a layer elsewhere may be off screen (see :meth:`bounds`). ``color`` overrides the
        layer's own, e.g. to add one array twice in two colours::

            srv.add_layer(pred, color="#00ff00")
        """
        _neuroglancer()
        layer = _as_served_layer(layer)
        if color is not None:
            # A copy: the caller may reuse the ServedLayer.
            layer = replace(layer, color=color)
        with self.viewer.txn() as s:
            resolved, _ = self._add_layer_in(s, layer, name)
        logger.info("added layer %r to %s", resolved, self.url)
        return self

    # -------------------------------------------------------------- reading back
    @property
    def position(self) -> tuple[float, ...] | None:
        """Where the viewer is looking, in the served frame's voxels (zyx)."""
        pos = self.viewer.state.position
        return None if pos is None else tuple(float(v) for v in pos)

    def selected_segments(self, layer: str | None = None) -> set[int]:
        """The label ids currently shown in a segmentation layer."""
        name = layer or self._sole_segmentation()
        segments = self.viewer.state.layers[name].segments
        return {int(s) for s in (segments or ())}

    def _sole_segmentation(self) -> str:
        names = [n for n, v in self.volumes.items() if v.volume_type == "segmentation"]
        if len(names) == 1:
            return names[0]
        raise ServeProblem(
            f"say which layer: this viewer has {len(names)} segmentation layers "
            f"({', '.join(sorted(names)) or 'none'})")

    def annotate(self, name: str | None = None, *, tool: str | None = "box",
                 select: bool = True) -> str:
        """Add a layer to draw in, arm a tool, and return the layer's name.

        The layer :meth:`boxes` and :meth:`points` read, added to a viewer already open::

            srv = serve([em, gt])
            srv.annotate()              # draw boxes; ctrl+mousedown0 places one
            srv.boxes()                 # -> [((lo), (hi)), ...] in zyx voxels
            srv.annotate(tool="point")  # same layer, now placing points
            srv.points()

        ``tool`` is what a click places: ``"box"``, ``"point"``, or ``None`` for nothing.
        ``select`` makes it the active layer, without which the armed tool does nothing until
        the layer is picked by hand.

        Calling it again with the same name keeps what is already drawn and re-arms the tool.
        A name taken by a volume layer raises. The name (default: the current annotation
        layer, else :data:`ANNOTATE_LAYER`) becomes the default for :meth:`boxes` and
        :meth:`points`.
        """
        ng = _neuroglancer()
        # Resolved before the transaction: a bad tool must not leave a half-added layer.
        tool_type = _tool_type(tool)
        name = str(name) if name else (self._annotations or ANNOTATE_LAYER)

        existing = (self.viewer.state.layers[name]
                    if name in self.viewer.state.layers else None)
        if existing is not None:
            kind = getattr(existing.layer, "type", None)
            if kind != "annotation":
                raise ServeProblem(
                    f"layer {name!r} is already {'an' if kind == 'image' else 'a'} {kind} "
                    f"layer, and replacing it would take its data with it — pass a name "
                    f"for the annotation layer instead, e.g. annotate({name + '_picks'!r})")

        with self.viewer.txn() as s:
            if existing is None:
                s.layers[name] = _annotation_layer(ng, self._dimensions(), tool)
            else:
                s.layers[name].tool = tool_type
            if select:
                s.selected_layer.layer = name
                s.selected_layer.visible = True

        self._annotations = name
        logger.info("annotation layer %r %s; tool %s", name,
                    "reused" if existing is not None else "added", tool or "not armed")
        return name

    # -------------------------------------------------------------- extents
    def _viewer_scales(self) -> list[float]:
        """The viewer's per-axis scale, in neuroglancer's normalized unit. Used only as a
        ratio against layer scales, so the unit cancels."""
        dims = self._dimensions()
        return [float(dims.scales[i]) for i in _spatial_axes(dims)]

    def _in_viewer_voxels(self, lo: Sequence[float], hi: Sequence[float],
                          scales: Sequence[float]) -> Any:
        """A box in one layer's voxels, as whole voxels of the viewer's own space.

        Rounds inward, since the result is a limit to clip against. A no-op when the layer
        shares the viewer's frame.
        """
        from neu_lib import BBox

        ratio = [s / v for s, v in zip(scales, self._viewer_scales())]
        return BBox(tuple(math.ceil(a * r) for a, r in zip(lo, ratio)),
                    tuple(math.floor(b * r) for b, r in zip(hi, ratio)))

    def _served(self, layer: str | None) -> list[str]:
        if layer is None:
            return sorted(self.volumes)
        if layer not in self.volumes:
            raise ServeProblem(
                f"no served layer named {layer!r} — this viewer serves "
                f"{', '.join(sorted(self.volumes)) or 'nothing'}")
        return [layer]

    def bounds(self, layer: str | None = None) -> Any:
        """Where the served arrays are, as a :class:`neu_lib.BBox` in viewer voxels.

        The extent of the arrays themselves, offset included: one layer by name, or the union
        of all. What ``enclose(clip="volume")`` clips to.
        """
        from functools import reduce

        from neu_lib import BBox

        boxes = []
        for name in self._served(layer):
            volume = self.volumes[name]
            axes = _spatial_axes(volume.dimensions)
            scales = [float(volume.dimensions.scales[i]) for i in axes]
            offset = [int(volume.voxel_offset[i]) for i in axes]
            shape = [int(volume.shape[i]) for i in axes]
            boxes.append(self._in_viewer_voxels(
                offset, [o + s for o, s in zip(offset, shape)], scales))
        return reduce(BBox.union, boxes, BBox.empty(3))

    def data_bounds(self, layer: str | None = None) -> Any:
        """Where the non-zero voxels are, as a :class:`neu_lib.BBox` in viewer voxels.

        Usually much smaller than :meth:`bounds` for a segmentation. One layer by name, or the
        union of all. Reads every voxel (in memory). A layer of all zeros raises.
        """
        from functools import reduce

        from neu_lib import BBox

        boxes = []
        for name in self._served(layer):
            volume = self.volumes[name]
            axes = _spatial_axes(volume.dimensions)
            array = np.asarray(volume.data)
            if array.ndim == 4:
                # Channel-first: a voxel counts if any channel is non-zero.
                array = array.any(axis=0)
            mask = array != 0
            lo, hi = [], []
            for a in range(3):
                seen = np.any(mask, axis=tuple(i for i in range(3) if i != a))
                hits = np.flatnonzero(seen)
                if hits.size == 0:
                    raise ServeProblem(
                        f"layer {name!r} is entirely zero, so it has no data bounds — "
                        f"clip to bounds() instead, or say which layer to use")
                lo.append(int(hits[0]))
                hi.append(int(hits[-1]) + 1)
            scales = [float(volume.dimensions.scales[i]) for i in axes]
            offset = [int(volume.voxel_offset[i]) for i in axes]
            boxes.append(self._in_viewer_voxels([o + v for o, v in zip(offset, lo)],
                                                [o + v for o, v in zip(offset, hi)],
                                                scales))
        return reduce(BBox.union, boxes, BBox.empty(3))

    def _clip_box(self, clip: Any) -> Any:
        """Resolve ``enclose``'s ``clip=`` to the box to intersect with."""
        if clip == "volume":
            return self.bounds()
        if clip == "data":
            return self.data_bounds()
        return _as_bbox(clip, "clip")

    def enclose(self, layer: str | None = None, *, margin: Any = 0,
                replace: bool = False, clip: Any = None,
                description: str | None = None) -> Any:
        """Replace what is drawn with the one box containing it. Returns a `neu_lib.BBox`.

        Place points around a region, then call this to get a box annotation in the viewer
        and the box in whole voxels, zyx, for ``--crop-bbox``, :func:`neu_vol.extract_roi` or
        ``neu-vol write``::

            srv.annotate(tool="point")            # click a point at each extreme
            box = srv.enclose(margin=16, clip="volume", replace=True)
            lo, hi = box                          # a BBox unpacks like the pair it is

        Every annotation in the layer counts, of any type.

        ``margin`` grows the box on every side, in voxels: a scalar, or one value per axis
        (zyx). A negative margin shrinks it; one that empties an axis raises.

        ``clip`` bounds the result:

        - ``"volume"``: the served arrays' extent (:meth:`bounds`);
        - ``"data"``: their non-zero voxels (:meth:`data_bounds`);
        - a :class:`neu_lib.BBox` or ``(lo, hi)`` pair, e.g. ``srv.bounds("seg")``;
        - ``None`` (default): no clipping, with a warning if the box starts below zero.

        A clip that leaves nothing raises. ``replace=True`` deletes the annotations that went
        into the box, leaving only the result. With ``replace=False`` the new box is itself
        an annotation, so calling again with a margin grows it each time.
        """
        ng = _neuroglancer()
        name = layer or self._annotations
        if name is None:
            raise ServeProblem("this viewer has no annotation layer to enclose — call "
                               ".annotate() to add one and draw in it")
        if name not in self.viewer.state.layers:
            raise ServeProblem(f"this viewer has no layer named {name!r}")

        drawn = list(self.viewer.state.layers[name].annotations or ())
        if not drawn:
            raise ServeProblem(
                f"nothing is drawn in layer {name!r}, so there is nothing to enclose — "
                f"place a point or two in the viewer first")

        from functools import reduce

        from neu_lib import BBox

        drawn_boxes = [_annotation_bbox(ann) for ann in drawn]
        ranks = {b.ndim for b in drawn_boxes}
        if len(ranks) > 1:
            raise ServeProblem(
                f"layer {name!r} holds annotations of {sorted(ranks)} dimensions, which "
                f"have no common bounding box")
        found = reduce(BBox.union, drawn_boxes)
        pad = _margin(margin, found.ndim)
        lo_pad = tuple(v - p for v, p in zip(found.lo, pad))
        hi_pad = tuple(v + p for v, p in zip(found.hi, pad))
        # Checked before building the BBox, whose own error would not mention the margin.
        if any(h <= l for l, h in zip(lo_pad, hi_pad)):
            raise ServeProblem(
                f"margin {pad} leaves an empty box {lo_pad} -> {hi_pad}: it shrinks the "
                f"annotations past nothing on at least one axis")
        grown = BBox(lo_pad, hi_pad)

        if clip is None:
            if any(v < 0 for v in grown.lo):
                # Warned rather than clipped, so the result matches what is drawn.
                logger.warning(
                    "enclosing box starts at %s, outside the volume on at least one axis — "
                    "the margin reaches past the data. Pass clip='volume' to trim it",
                    grown.lo)
            box_zyx = grown
        else:
            limit = self._clip_box(clip)
            box_zyx = grown.intersect(limit)
            if box_zyx.is_empty():
                raise ServeProblem(
                    f"clipping {grown.lo} -> {grown.hi} to {limit.lo} -> {limit.hi} leaves "
                    f"nothing: what was drawn lies outside it on at least one axis")
            if box_zyx != grown:
                logger.info("clipped %s -> %s to %s", grown.lo, grown.hi, limit.hi)

        lo, hi = box_zyx
        box = ng.AxisAlignedBoundingBoxAnnotation(
            id=uuid.uuid4().hex, point_a=list(lo), point_b=list(hi),
            description=description)
        with self.viewer.txn() as s:
            keep = list(s.layers[name].annotations or ())
            if replace:
                # Matched by id, so an annotation added meanwhile in the browser survives.
                used = {a.id for a in drawn}
                keep = [a for a in keep if a.id not in used]
            s.layers[name].annotations = keep + [box]

        logger.info("enclosed %d annotation(s) in %s -> %s%s", len(drawn), lo, hi,
                    " (originals deleted)" if replace else "")
        return box_zyx

    def _dimensions(self) -> Any:
        """The viewer's own coordinate space. See :func:`_annotation_layer`."""
        dims = self.viewer.state.dimensions
        if dims is None or not list(dims.names):
            raise ServeProblem(
                "this viewer declares no dimensions, so an annotation layer would have no "
                "coordinate space to record — serve a layer carrying a frame first")
        return dims

    def boxes(self, layer: str | None = None) -> list[tuple[tuple[int, ...], ...]]:
        """Boxes drawn in the viewer, as ``(lo, hi)`` pairs in whole voxels, zyx.

        Ready for ``--crop-bbox``, :func:`neu_vol.extract_roi` or ``neu-vol write``. Corners
        are sorted per axis, since a box drawn up and to the left has them reversed.
        """
        ng = _neuroglancer()
        name = layer or self._annotations
        if name is None:
            raise ServeProblem("this viewer has no annotation layer to read boxes from — "
                               "call .annotate() to add one and draw in it")
        found = []
        for ann in self.viewer.state.layers[name].annotations or ():
            if not isinstance(ann, ng.AxisAlignedBoundingBoxAnnotation):
                continue
            a = [int(round(float(v))) for v in ann.point_a]
            b = [int(round(float(v))) for v in ann.point_b]
            lo = tuple(min(p, q) for p, q in zip(a, b))
            hi = tuple(max(p, q) for p, q in zip(a, b))
            found.append((lo, hi))
        return found

    def points(self, layer: str | None = None) -> list[tuple[int, ...]]:
        """Points drawn in the viewer, in whole voxels, zyx."""
        ng = _neuroglancer()
        name = layer or self._annotations
        if name is None:
            raise ServeProblem("this viewer has no annotation layer to read points from — "
                               "call .annotate(tool='point') to add one and draw in it")
        return [tuple(int(round(float(v))) for v in ann.point)
                for ann in (self.viewer.state.layers[name].annotations or ())
                if isinstance(ann, ng.PointAnnotation)]

    # -------------------------------------------------------------- callbacks
    def on_click(self, callback: Callable[[Click], None], *, button: int = 0,
                 modifiers: str = "control", scope: str = "data_view",
                 name: str | None = None) -> str:
        """Call ``callback(click)`` when the user clicks in the data.

        Returns the action name. The default binding is ``control+mousedown0``, like
        neuroglancer's own tools, since a bare click navigates. The callback is not called
        for a click outside the data. It runs on the server's event-loop thread, and an
        exception in it is logged rather than lost.
        """
        _neuroglancer()
        self._actions += 1
        action = name or f"neu-glance-click-{self._actions}"

        def handler(action_state) -> None:
            try:
                position = getattr(action_state, "mouse_voxel_coordinates", None)
                if position is None:
                    return
                values = {}
                for layer_name in self.volumes:
                    try:
                        selected = action_state.selected_values[layer_name]
                        values[layer_name] = getattr(selected, "value", selected)
                    except Exception:                                     # noqa: BLE001
                        pass
                callback(Click(position=tuple(float(v) for v in position),
                               values=values, state=action_state))
            except Exception:                                             # noqa: BLE001
                logger.exception("click handler %s failed", action)

        self.viewer.actions.add(action, handler)
        binding = f"{modifiers}+mousedown{button}" if modifiers else f"mousedown{button}"
        with self.viewer.config_state.txn() as s:
            getattr(s.input_event_bindings, scope)[binding] = action
        logger.info("bound %s to %s (%s)", binding, action, scope)
        return action

    def on_change(self, callback: Callable[[], None]) -> None:
        """Call ``callback()`` whenever the browser changes the state.

        The callback takes no arguments, as in neuroglancer; it should re-read what it needs.
        """
        self.viewer.shared_state.add_changed_callback(callback)

    def screenshot(self, path: str | None = None):
        """A PNG of what the browser is showing. Blocks until it replies.

        **With no browser connected this blocks forever**: neuroglancer waits for the reply
        with no timeout. :func:`neu_glance.movie.rendering.wait_for_browser` is the bounded form.
        """
        reply = self.viewer.screenshot().screenshot
        if path is None:
            return reply.image_pixels
        with open(path, "wb") as f:
            f.write(reply.image)
        return path

    def stop(self) -> None:
        """Stop the server, invalidating every viewer URL in this process.

        There is one server per process, shared by every viewer, so this stops them all
        (like :func:`stop_serving`).
        """
        stop_serving()


#: The fraction of the window (per direction) each cross-section panel gets, per layout,
#: so the opening view fits the data in each panel rather than in the whole window.
_PANEL_FRACTION = {
    "4panel": 0.5, "xy-3d": 0.5, "yz-3d": 0.5, "xz-3d": 0.5,
    "xy": 1.0, "yz": 1.0, "xz": 1.0, "3d": 1.0,
}


def stop_serving() -> bool:
    """Stop the viewer server, whatever started it. ``True`` if one was running.

    For when the :class:`Server` handle is lost (a re-run cell, an unassigned result). There
    is one server per process, a daemon thread, and every viewer URL dies with it. The
    library never blocks a notebook; a stuck cell is running the ``neu-glance serve``
    command, which runs until interrupted.
    """
    ng = _neuroglancer()
    if not ng.server.is_server_running():
        return False
    ng.server.stop()
    logger.info("viewer server stopped; every link from this process is now dead")
    return True


def _opening_view(layers: Sequence[ServedLayer], *, layout: str = "4panel",
                  fit: float = 1.0):
    """``(position, cross_section_scale, projection_scale)`` framing every served layer.

    In voxels of the first layer's frame (the viewer's units), framing the union of the
    layers' nm bounds, scaled by the layout's panel fraction and by ``fit``. ``None`` when no
    layer has a frame.
    """
    from .state import default_view

    framed = [ln for ln in layers if ln.frame is not None]
    if not framed:
        return None
    los, his = zip(*(ln.bounds_nm for ln in framed))
    lo = tuple(min(v[a] for v in los) for a in range(3))
    hi = tuple(max(v[a] for v in his) for a in range(3))
    voxel = framed[0].frame.voxel_size_nm
    extent = tuple((b - a) / v for a, b, v in zip(lo, hi, voxel))
    offset = tuple(a / v for a, v in zip(lo, voxel))
    centre, cross, projection = default_view(extent, offset)
    scale = fit / _PANEL_FRACTION.get(layout, 0.5)
    return centre, cross * scale, projection * scale


def serve(layers: Sequence[ServedLayer], *, bind: str | None = None, port: int = 0,
          into: dict | None = None, annotations: str | bool | None = None,
          position: Sequence[float] | None = None, layout: str = "4panel",
          fit: float = 1.0) -> Server:
    """Host ``layers`` and return a :class:`Server` carrying the viewer URL.

    Each entry is a :class:`ServedLayer`, a :class:`neu_lib.Piece` (which arrives already
    named and knowing its kind), or the ``dict`` form of a ServedLayer.

    ``bind`` / ``port`` take effect only before the first viewer in the process
    (neuroglancer reads them at server start); a later different address is ignored with a
    warning. ``bind="0.0.0.0"`` puts the host's name in the URL, for a browser elsewhere.

    ``into`` merges the layers into an existing state (from :func:`neu_glance.load_state` or
    :func:`neu_glance.parse_url`), keeping its view. ``annotations`` (off by default) adds an
    empty annotation layer for :meth:`Server.boxes`: ``True`` for the default name, or a
    name. :meth:`Server.annotate` does the same later.

    By default the view opens centred on the data and zoomed to fit it in each panel of
    ``layout``; ``position`` overrides the centre, and ``fit`` above 1 shows more around it.
    """
    ng = _neuroglancer()
    layers = [_as_served_layer(ln) for ln in layers]
    if not layers:
        raise ServeProblem("nothing to serve: pass at least one layer")

    if bind is not None or port:
        if ng.server.is_server_running():
            logger.warning(
                "a viewer server is already running in this process, so bind=%r port=%r "
                "are ignored — neuroglancer reads them once, at server start", bind, port)
        else:
            ng.set_server_bind_address(bind or "127.0.0.1", port)

    viewer = ng.Viewer()
    # The viewer's dimensions: the first layer's frame, spatial axes only. Set explicitly,
    # since mismatched dimensions load fine and misplace every layer.
    spatial = _coordinate_space(ng, layers[0].frame, name=layers[0].name)
    annotate_name = ((ANNOTATE_LAYER if annotations is True else str(annotations))
                     if annotations else None)
    # Built first so the layers go through Server._add_layer_in, as add_layer's do.
    server = Server(viewer, {}, annotate_name)

    with viewer.txn() as s:
        if not (into and into.get("dimensions")):
            s.dimensions = spatial
        if into:
            # Keep the incoming view, as state.merge_into does.
            for key, value in into.items():
                if key != "layers":
                    try:
                        setattr(s, key, value)
                    except Exception:                                     # noqa: BLE001
                        logger.debug("state key %r not settable, skipped", key)
            for existing in into.get("layers", ()):
                name = existing.get("name")
                if name:
                    s.layers[name] = existing

        for i, layer in enumerate(layers):
            # Default names image0, image1, ... when there are several layers.
            server._add_layer_in(
                s, layer, layer.name or f"{layer.kind}{'' if len(layers) == 1 else i}")

        if annotate_name:
            s.layers[annotate_name] = _annotation_layer(ng, spatial)
        # Centre on the data and fit it; otherwise neuroglancer opens at the origin corner.
        if position is not None:
            s.position = [float(v) for v in position]
        else:
            view = _opening_view(layers, layout=layout, fit=fit)
            if view is not None:
                s.position, s.cross_section_scale, s.projection_scale = view
        if layout and not (into and into.get("layout")):
            s.layout = layout

    logger.info("serving %d layer(s) at %s — .stop() or neu_glance.stop_serving() to end it",
                len(server.volumes), server.url)
    return server
