"""Keyframe animation over neuroglancer states — a timeline, and the interpolation under it.

A :class:`Timeline` wraps one **base state** (a link tuned in the browser) and a list of
tweens (:class:`Tween`), each moving one property over a window of time with an easing.
Anything no tween mentions keeps its base value, so the base can be edited without
touching the tweens.

Pure: plain dicts in and out, stdlib only, no neuroglancer. :mod:`neu_glance.movie.rendering`
is the half that needs a viewer.

neuroglancer's own ``ViewerState.interpolate`` is not used: in 2.41.2 it raises for any
state with a layer (it reads a nonexistent ``layer_position``), it covers few layer
properties, and it adds orientations a state did not have. Its zoom and quaternion
formulas are ported here (:func:`interpolate_zoom`, :func:`interpolate_quaternion`), and a
test checks them against neuroglancer's.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterator, Mapping, Sequence

from ..shaders import as_hex_color
from ..state import (FIT_MARGIN, StateProblem, default_view, parse_url,
                    split_segment_layer, subset_layers)


class AnimateProblem(RuntimeError):
    """A timeline could not be built, or would not animate what it claims to."""


#: Where a fade starts, just above zero: neuroglancer skips drawing a layer at
#: ``objectAlpha <= 0``, and a layer that is not drawn may not have its mesh loaded when it
#: fades in. 0.001 looks the same as zero on screen.
FADE_FROM = 0.001

#: The top-level keys :meth:`Timeline.view` takes from a state: the camera, not the layers
#: or layout.
CAMERA_KEYS = ("position", "projectionOrientation", "projectionScale", "projectionDepth",
               "crossSectionOrientation", "crossSectionScale")

#: Neuroglancer's layer defaults (from ``viewer_state.py``), used as a tween's start when
#: the state omits the property. An absent property is not zero: an absent ``objectAlpha``
#: is fully opaque.
LAYER_DEFAULTS: dict[str, Any] = {
    "objectAlpha": 1.0,
    "selectedAlpha": 0.5,
    "notSelectedAlpha": 0.0,
    "saturation": 1.0,
    "meshSilhouetteRendering": 0.0,
    "meshRenderScale": 10.0,
    "crossSectionRenderScale": 1.0,
    "opacity": 0.5,
    "colorSeed": 0,
    "hideSegmentZero": True,
    "visible": True,
    "archived": False,
    "pick": True,
}

#: The same for top-level view keys. The zoom scales have no default (neuroglancer fits the
#: data), so a zoom tween needs them in the state or an explicit ``start``.
VIEW_DEFAULTS: dict[str, Any] = {
    "projectionOrientation": [0.0, 0.0, 0.0, 1.0],
    "crossSectionOrientation": [0.0, 0.0, 0.0, 1.0],
    "showSlices": True,
    "showAxisLines": True,
    "showDefaultAnnotations": True,
    "showScaleBar": True,
    "layout": "4panel",
}


# --------------------------------------------------------------------------- #
# easing
# --------------------------------------------------------------------------- #
def _smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def _smootherstep(t: float) -> float:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


#: Easing curves, applied to the time fraction before interpolating, so an eased zoom is
#: still exponential and an eased slerp still a great-circle arc. Each must satisfy
#: ``e(0) == 0`` and ``e(1) == 1`` and be monotone: an overshooting curve would push values
#: out of range, which neuroglancer silently clamps.
EASINGS: dict[str, Callable[[float], float]] = {
    "linear": lambda t: t,
    "in": lambda t: t * t,
    "out": lambda t: 1.0 - (1.0 - t) * (1.0 - t),
    "in-out": _smoothstep,
    "in-out-cubic": _smootherstep,
    "step": lambda t: 0.0 if t < 1.0 else 1.0,
}

#: The default: smoothstep, with zero velocity at both ends. A linear camera move starts and
#: stops with a visible jerk.
DEFAULT_EASE = "in-out"


def ease(name: str, t: float) -> float:
    """``EASINGS[name]`` applied to ``t``, clamped to ``[0, 1]``."""
    try:
        curve = EASINGS[name]
    except KeyError:
        raise AnimateProblem(
            f"unknown easing {name!r}; known: {', '.join(sorted(EASINGS))}") from None
    return curve(min(1.0, max(0.0, float(t))))


# --------------------------------------------------------------------------- #
# interpolators
# --------------------------------------------------------------------------- #
def interpolate_linear(a: Any, b: Any, t: float) -> Any:
    """Straight lerp, element-wise for equal-length sequences.

    Sequences of different lengths hold ``a``, as neuroglancer does.
    """
    if a is None or b is None:
        return a if b is None else b
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if not (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple))
                and len(a) == len(b)):
            return a
        return [float(x) * (1.0 - t) + float(y) * t for x, y in zip(a, b)]
    return float(a) * (1.0 - t) + float(b) * t


def interpolate_zoom(a: Any, b: Any, t: float) -> Any:
    """Exponential interpolation, for the two zoom scales. Ported from ``viewer_state.py:107``.

    Zoom is perceived multiplicatively, so a linear zoom visibly accelerates towards the
    close end. A non-positive endpoint falls back to a linear lerp rather than raising
    mid-render.
    """
    if a is None or b is None:
        return a if b is None else b
    a, b = float(a), float(b)
    if a <= 0.0 or b <= 0.0:
        return interpolate_linear(a, b, t)
    return a * math.exp(math.log(b / a) * t)


def interpolate_quaternion(a: Any, b: Any, t: float) -> list[float]:
    """Spherical linear interpolation. Ported from ``viewer_state.py:75-104`` (via gl-matrix).

    ``q`` and ``-q`` are the same rotation, so ``b`` is negated when needed to take the short
    way round (otherwise two views 20 degrees apart can turn 340 degrees). Inputs are assumed
    to be unit quaternions, as neuroglancer writes them.
    """
    a = list(VIEW_DEFAULTS["projectionOrientation"]) if a is None else [float(v) for v in a]
    b = list(VIEW_DEFAULTS["projectionOrientation"]) if b is None else [float(v) for v in b]
    cosom = sum(x * y for x, y in zip(a, b))
    if cosom < 0.0:
        cosom = -cosom
        b = [-y for y in b]
    if (1.0 - cosom) > 0.000001:
        omega = math.acos(max(-1.0, min(1.0, cosom)))
        sinom = math.sin(omega)
        scale0 = math.sin((1.0 - t) * omega) / sinom
        scale1 = math.sin(t * omega) / sinom
    else:
        scale0, scale1 = 1.0 - t, t
    return [scale0 * x + scale1 * y for x, y in zip(a, b)]


#: Named rotation axes for :meth:`Timeline.orbit`. ``up``, ``right`` and ``forward`` are the
#: screen's axes, the same from any starting view; ``x``, ``y`` and ``z`` are the volume's,
#: reproducible across scenes.
AXES: dict[str, tuple[float, float, float]] = {
    "up": (0.0, 1.0, 0.0), "right": (1.0, 0.0, 0.0), "forward": (0.0, 0.0, 1.0),
    "x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0),
}

#: Which axes are read in the camera's frame rather than the volume's.
SCREEN_AXES = frozenset({"up", "right", "forward"})


def _quaternion_multiply(a: Sequence[float], b: Sequence[float]) -> list[float]:
    """Hamilton product, ``xyzw`` order (as in neuroglancer and gl-matrix)."""
    ax, ay, az, aw = (float(v) for v in a)
    bx, by, bz, bw = (float(v) for v in b)
    return [aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz]


def axis_quaternion(axis: Any, degrees: float) -> list[float]:
    """A rotation of ``degrees`` about ``axis`` as a unit quaternion, ``xyzw``."""
    vector = AXES.get(axis) if isinstance(axis, str) else axis
    if vector is None:
        raise AnimateProblem(
            f"unknown axis {axis!r}; use a 3-sequence or one of: {', '.join(sorted(AXES))}")
    length = math.sqrt(sum(float(v) ** 2 for v in vector))
    if length == 0.0:
        raise AnimateProblem("an orbit axis cannot be the zero vector")
    half = math.radians(float(degrees)) / 2.0
    scale = math.sin(half) / length
    return [float(vector[0]) * scale, float(vector[1]) * scale, float(vector[2]) * scale,
            math.cos(half)]


def interpolate_orbit(a: Any, b: Any, t: float) -> list[float]:
    """Rotate ``a`` progressively about an axis. ``b`` is ``{"axis", "degrees", "screen"}``.

    Slerp between two views always takes the shortest arc, so a full turn would be a no-op
    and 200 degrees would become 160 the other way. Here the angle varies, so any rotation
    works, including several turns. ``screen`` reads the axis in the camera's frame
    (post-multiply) or, if false, the volume's (pre-multiply).
    """
    start = list(VIEW_DEFAULTS["projectionOrientation"]) if a is None else [float(v) for v in a]
    spin = axis_quaternion(b["axis"], float(b["degrees"]) * t)
    return (_quaternion_multiply(start, spin) if b.get("screen", True)
            else _quaternion_multiply(spin, start))


def interpolate_zoom_by(a: Any, b: Any, t: float) -> Any:
    """Scale ``a`` by a factor, geometrically. ``b`` is the factor at the end of the window.

    Relative and exponential, like :func:`interpolate_zoom`: a factor of 0.5 is twice as
    close, from wherever the camera is.
    """
    if a is None:
        return a
    return float(a) * (float(b) ** t)


def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _linear_to_srgb(c: float) -> float:
    return c * 12.92 if c <= 0.0031308 else 1.055 * (c ** (1.0 / 2.4)) - 0.055


def interpolate_hex_color(a: Any, b: Any, t: float, *, space: str = "linear") -> str:
    """A colour ramp, in **linear light** by default. Returns ``#rrggbb``.

    Endpoints are anything :func:`neu_glance.shaders.as_hex_color` accepts. Hex channels are
    gamma-encoded, so averaging them directly makes black-to-red pass through a dark maroon;
    mixing in linear light gives a midpoint that looks halfway. ``space="srgb"`` mixes the
    encoded values, to match ramps made elsewhere.
    """
    lo = as_hex_color(a, "colour")
    hi = as_hex_color(b, "colour")
    pairs = [(int(lo[1 + 2 * i:3 + 2 * i], 16) / 255.0,
              int(hi[1 + 2 * i:3 + 2 * i], 16) / 255.0) for i in range(3)]
    if space == "linear":
        mixed = [_linear_to_srgb(_srgb_to_linear(x) * (1.0 - t) + _srgb_to_linear(y) * t)
                 for x, y in pairs]
    elif space == "srgb":
        mixed = [x * (1.0 - t) + y * t for x, y in pairs]
    else:
        raise AnimateProblem(f"unknown colour space {space!r}; use 'linear' or 'srgb'")
    return "#" + "".join(f"{int(round(min(1.0, max(0.0, c)) * 255)):02x}" for c in mixed)


def interpolate_color_map(a: Any, b: Any, t: float, *, default: str | None = None) -> dict:
    """``segmentColors``, interpolated per segment id.

    A key on both sides ramps. A key on one side only ramps to or from ``default`` (the
    layer's ``segmentDefaultColor``) if given, and otherwise **steps** at the end of the
    window.
    """
    a = {str(k): v for k, v in (a or {}).items()}
    b = {str(k): v for k, v in (b or {}).items()}
    out: dict[str, str] = {}
    for key in list(a) + [k for k in b if k not in a]:
        if key in a and key in b:
            out[key] = interpolate_hex_color(a[key], b[key], t)
        elif key in a:                                  # dropped: ramp out towards the default
            out[key] = (interpolate_hex_color(a[key], default, t) if default is not None
                        else interpolate_step(a[key], None, t))
        else:                                           # added: ramp in from the default
            out[key] = (interpolate_hex_color(default, b[key], t) if default is not None
                        else interpolate_step(None, b[key], t))
        if out[key] is None:
            del out[key]
    return out


def interpolate_step(a: Any, b: Any, t: float) -> Any:
    """Hold ``a``, take ``b`` at the **end** of the transition.

    Not at the midpoint, where a flip mid-move looks like an artefact. For a change at
    another time, use a zero-length tween there.
    """
    return b if t >= 1.0 else a


#: Interpolator by name.
INTERPOLATORS: dict[str, Callable[[Any, Any, float], Any]] = {
    "linear": interpolate_linear,
    "zoom": interpolate_zoom,
    "slerp": interpolate_quaternion,
    "color": interpolate_hex_color,
    "colors": interpolate_color_map,
    "step": interpolate_step,
    # Relative moves: `end` is a delta, so a tween must name these explicitly
    # (Timeline.orbit and Timeline.zoom do).
    "orbit": interpolate_orbit,
    "zoom_by": interpolate_zoom_by,
}

#: Which interpolator a property gets. **Anything not named here steps**, so an unlisted
#: property never gets an invented midpoint.
PROPERTY_KINDS: dict[str, str] = {
    # per-layer, continuous
    "objectAlpha": "linear",
    "opacity": "linear",
    "selectedAlpha": "linear",
    "notSelectedAlpha": "linear",
    "saturation": "linear",
    "meshSilhouetteRendering": "linear",
    "meshRenderScale": "linear",
    "crossSectionRenderScale": "linear",
    # top-level view
    "position": "linear",
    "crossSectionDepth": "linear",
    "projectionDepth": "linear",
    "projectionScale": "zoom",
    "crossSectionScale": "zoom",
    "projectionOrientation": "slerp",
    "crossSectionOrientation": "slerp",
    # colour
    "segmentDefaultColor": "color",
    "annotationColor": "color",
    "segmentColors": "colors",
    # stepped, listed explicitly (fade individual segments by splitting the layer instead)
    "segments": "step",
    "visible": "step",
    "archived": "step",
    "layout": "step",
    "shader": "step",
    "source": "step",
}


def kind_for(prop: str) -> str:
    """The interpolator name for ``prop``. Unknown properties step; see :data:`PROPERTY_KINDS`."""
    return PROPERTY_KINDS.get(prop, "step")


# --------------------------------------------------------------------------- #
# addressing
# --------------------------------------------------------------------------- #
# A path is ("<view property>",) or ("layers", "<layer name>", "<layer property>"). A tuple,
# not a dotted string, because layer names often contain dots and spaces.
def _layer_in(state: Mapping[str, Any], name: str) -> dict | None:
    for lyr in state.get("layers") or []:
        if isinstance(lyr, dict) and lyr.get("name") == name:
            return lyr
    return None


def _get_at(state: Mapping[str, Any], path: Sequence[str]) -> Any:
    if len(path) == 1:
        return state.get(path[0])
    layer = _layer_in(state, path[1])
    return None if layer is None else layer.get(path[2])


def _set_at(state: dict, path: Sequence[str], value: Any) -> None:
    if len(path) == 1:
        state[path[0]] = value
        return
    layer = _layer_in(state, path[1])
    if layer is not None:
        layer[path[2]] = value


def _default_for(path: Sequence[str]) -> Any:
    table = VIEW_DEFAULTS if len(path) == 1 else LAYER_DEFAULTS
    return table.get(path[-1])


def _describe(path: Sequence[str]) -> str:
    return path[0] if len(path) == 1 else f"{path[2]} on layer {path[1]!r}"


# --------------------------------------------------------------------------- #
# tweens
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Tween:
    """One property, moving from one value to another over one window of time."""

    path: tuple[str, ...]
    end: Any
    start: Any = None
    at: float = 0.0
    seconds: float = 0.0
    ease: str = DEFAULT_EASE
    kind: str | None = None

    @property
    def stop(self) -> float:
        return self.at + self.seconds

    def value_at(self, t: float, start: Any) -> Any:
        """The value at ``t``, or :data:`_NOTHING` before the tween begins (so the earlier
        value shows). After the window, the end value holds."""
        interpolate = INTERPOLATORS[self.kind or kind_for(self.path[-1])]
        # The held value is the interpolator at t=1, not `end`: for a relative kind (orbit,
        # zoom_by), `end` is a delta.
        if self.seconds <= 0.0:
            return interpolate(start, self.end, 1.0) if t >= self.at else _NOTHING
        if t >= self.stop:
            return interpolate(start, self.end, 1.0)
        if t <= self.at:
            return _NOTHING
        return interpolate(start, self.end, ease(self.ease, (t - self.at) / self.seconds))

    def to_json(self) -> dict:
        obj = {"path": list(self.path), "end": self.end, "at": self.at,
               "seconds": self.seconds, "ease": self.ease}
        if self.start is not None:
            obj["start"] = self.start
        if self.kind is not None:
            obj["kind"] = self.kind
        return obj

    @classmethod
    def from_json(cls, obj: Mapping[str, Any]) -> "Tween":
        return cls(path=tuple(obj["path"]), end=obj["end"], start=obj.get("start"),
                   at=float(obj.get("at", 0.0)), seconds=float(obj.get("seconds", 0.0)),
                   ease=obj.get("ease", DEFAULT_EASE), kind=obj.get("kind"))


class _Nothing:
    def __repr__(self) -> str:                                          # pragma: no cover
        return "<no contribution>"


#: Distinct from ``None``, which is a legitimate property value.
_NOTHING = _Nothing()

#: Tolerance for two window edges to count as touching rather than overlapping: far below a
#: frame, far above float rounding.
_TIME_EPSILON = 1e-9


# --------------------------------------------------------------------------- #
# the timeline
# --------------------------------------------------------------------------- #
@dataclass
class Timeline:
    """A base state plus tweens over it. Every frame it emits is a plain state dict.

    Frame *i* is at ``i / fps``; ``fps`` is kept here so a saved timeline fully describes its
    frames.
    """

    base: dict
    fps: float = 30.0
    tweens: list[Tween] = field(default_factory=list)
    groups: dict[str, tuple[str, ...]] = field(default_factory=dict)
    cursor: float = 0.0
    notes: list[str] = field(default_factory=list)
    _duration: float | None = None

    def __post_init__(self) -> None:
        self.base = json.loads(json.dumps(dict(self.base)))   # never alias the caller's state

    # -- building ---------------------------------------------------------- #
    def split(self, layer: str, *, prefix: str | None = None,
              names: Mapping[str, str] | None = None,
              segments: Sequence[Any] | None = None) -> dict[str, str]:
        """One layer per segment of ``layer``. Returns ``{segment_id: layer_name}``.

        Needed to fade objects individually (:func:`neu_glance.split_segment_layer`). Build
        groups from the returned layer names.
        """
        template = f"{prefix} {{segment}}" if prefix else "{layer} {segment}"
        self.base, created, notes = split_segment_layer(
            self.base, layer, names=names, segments=segments, name_template=template)
        self.notes.extend(notes)
        return created

    def sets(self, layer: str, subsets: Mapping[str, Sequence[Any]], *,
             colors: Mapping[str, str] | None = None,
             object_alpha: float | None = None) -> dict[str, str]:
        """One layer per named SET of segments. Returns ``{set: layer_name}``.

        For populations that fade together (:func:`neu_glance.subset_layers`); use
        :meth:`split` for individual objects.
        """
        self.base, created, notes = subset_layers(
            self.base, layer, subsets, colors=colors, object_alpha=object_alpha)
        self.notes.extend(notes)
        return created

    def set(self, *, layer: str | None = None, **props: Any) -> "Timeline":
        """An opening value at ``t=0``, held until a later tween takes the property over."""
        for prop, value in props.items():
            self._add(Tween(path=self._path(layer, prop), end=value, at=0.0, seconds=0.0))
        return self

    def tween(self, *, layer: str | None = None, at: float | None = None,
              seconds: float, ease: str = DEFAULT_EASE, start: Any = None,
              kind: str | None = None, **props: Any) -> "Timeline":
        """Move each named property to its given value over ``seconds``, starting at ``at``.

        ``at=None`` uses the cursor, which only :meth:`hold` moves, so consecutive calls run
        at the same time unless the cursor is advanced.

        ``start=None`` takes the value the property last held: an earlier tween's end, else
        the base state, else :data:`LAYER_DEFAULTS` / :data:`VIEW_DEFAULTS` (not zero).
        """
        when = self.cursor if at is None else float(at)
        for prop, value in props.items():
            self._add(Tween(path=self._path(layer, prop), end=value, start=start,
                            at=when, seconds=float(seconds), ease=ease, kind=kind))
        return self

    def group(self, name: str, layers: Sequence[str]) -> "Timeline":
        """Name a set of layers that appear together — a neuron and the neuropils it innervates."""
        missing = [n for n in layers if _layer_in(self.base, n) is None]
        if missing:
            raise AnimateProblem(
                f"group {name!r} names layer(s) not in this state: {', '.join(missing)}. "
                f"Split a segmentation layer first, and build groups from what split() returns")
        self.groups[name] = tuple(layers)
        return self

    def sequence(self, names: Sequence[str], *, start: float | None = None,
                 seconds: float = 1.2, stagger: float | None = None,
                 ease: str = DEFAULT_EASE, fade_from: float = FADE_FROM) -> "Timeline":
        """Fade in each group in turn, ``stagger`` seconds apart. Returns ``self``.

        ``stagger=None`` means back to back; a shorter stagger overlaps the fades. Each layer
        also starts at ``objectAlpha = fade_from`` from ``t=0``, so the scene starts empty.
        """
        begin = self.cursor if start is None else float(start)
        step = float(seconds) if stagger is None else float(stagger)
        for i, name in enumerate(names):
            try:
                layers = self.groups[name]
            except KeyError:
                raise AnimateProblem(
                    f"no group named {name!r}; defined: {', '.join(sorted(self.groups)) or '(none)'}"
                ) from None
            for layer in layers:
                self.set(layer=layer, objectAlpha=fade_from)
                self.tween(layer=layer, at=begin + i * step, seconds=seconds, ease=ease,
                           objectAlpha=1.0)
        return self

    def view(self, target: Mapping[str, Any] | str, *, at: float | None = None,
             seconds: float, ease: str = DEFAULT_EASE) -> "Timeline":
        """Move the camera to the view held by another state. Returns ``self``.

        ``target`` is a state dict or a neuroglancer URL: fly the camera in the browser, copy
        the link, paste it here. Only :data:`CAMERA_KEYS` are taken; the target's layers and
        layout are ignored.
        """
        if isinstance(target, str):
            try:
                target = parse_url(target)
            except (ValueError, StateProblem) as e:
                raise AnimateProblem(
                    f"view() takes a state dict or a neuroglancer URL: {e}. For a state on "
                    f"disk, read it with neu_glance.load_state() first") from None
        present = {k: target[k] for k in CAMERA_KEYS if k in target}
        if not present:
            raise AnimateProblem(
                f"that state carries none of the camera keys ({', '.join(CAMERA_KEYS)}), so "
                f"there is no view in it to move to")
        return self.tween(at=at, seconds=seconds, ease=ease, **present)

    def orbit(self, degrees: float, *, axis: Any = "up", at: float | None = None,
              seconds: float, ease: str = DEFAULT_EASE) -> "Timeline":
        """Rotate the camera ``degrees`` about ``axis``. Relative — no keyframe needed.

        ``axis`` defaults to ``"up"``, the screen's vertical: a turntable from any starting
        view. A volume axis (``"z"``, or a 3-sequence) is reproducible across scenes. A full
        turn is ``orbit(360)`` (see :func:`interpolate_orbit`).
        """
        axis_quaternion(axis, 0.0)      # validate now rather than mid-render
        return self.tween(at=at, seconds=seconds, ease=ease, kind="orbit",
                          projectionOrientation={"axis": axis, "degrees": float(degrees),
                                                 "screen": (isinstance(axis, str)
                                                            and axis in SCREEN_AXES)})

    def zoom(self, factor: float, *, at: float | None = None, seconds: float,
             ease: str = DEFAULT_EASE) -> "Timeline":
        """Scale the 3D zoom by ``factor``. Relative: ``0.5`` is twice as close.

        Needs no knowledge of the current ``projectionScale``.
        """
        if factor <= 0:
            raise AnimateProblem(f"a zoom factor must be positive, got {factor!r}")
        return self.tween(at=at, seconds=seconds, ease=ease, kind="zoom_by",
                          projectionScale=float(factor))

    def voxel_size_nm(self) -> tuple[float, float, float]:
        """The state's own voxel size in nm, zyx, from its ``dimensions``."""
        dims = self.base.get("dimensions") or {}
        try:
            return tuple(float(dims[axis][0]) * 1e9 for axis in ("z", "y", "x"))
        except (KeyError, TypeError, IndexError):
            raise AnimateProblem(
                "this state has no usable `dimensions`, so a box in nanometres cannot be "
                "converted to the coordinates its `position` is written in. Pass "
                "units='voxels' if the box is already in those.") from None

    def frame_on(self, box: Any, *, at: float | None = None, seconds: float,
                 ease: str = DEFAULT_EASE, margin: float = 1.15, units: str = "nm",
                 zoom: bool = True, min_scale: float | None = None,
                 rotation_safe: bool = False) -> "Timeline":
        """Pan (and by default zoom) so ``box`` fills the view.

        ``box`` is ``(lo_zyx, hi_zyx)`` or anything with ``.lo`` / ``.hi`` (a
        :class:`neu_lib.BBox`), e.g. from :func:`neu_glance.sources.segment_boxes`.

        ``units="nm"`` (the default, what ``segment_boxes`` returns) is converted to the
        state's voxels with its ``dimensions``, since neuroglancer's ``position`` is in voxels;
        pass ``units="voxels"`` for a box already in viewer coordinates. A wrong choice does
        not raise; it misplaces the camera by the voxel size.

        Framing matches :func:`neu_glance.state.default_view`, with ``margin`` as the room
        around the box; ``zoom=False`` only pans. ``rotation_safe=True`` fits the box's
        diagonal, for a shot that turns: ``projectionScale`` fixes the vertical extent, so a box
        framed at one angle can overflow at another. ``min_scale`` caps how close the camera
        goes. An orbit pivots about ``position``, which this also centres.
        """
        lo, hi = (box.lo, box.hi) if hasattr(box, "lo") else box
        lo, hi = [float(v) for v in lo], [float(v) for v in hi]
        if units == "nm":
            voxel = self.voxel_size_nm()
            lo = [v / s for v, s in zip(lo, voxel)]
            hi = [v / s for v, s in zip(hi, voxel)]
        elif units != "voxels":
            raise AnimateProblem(f"units must be 'nm' or 'voxels', got {units!r}")
        extent = [max(1e-9, b - a) for a, b in zip(lo, hi)]
        centre, _cross, projection = default_view(extent, lo)
        moves: dict[str, Any] = {"position": [float(v) for v in centre[::-1]]}
        if zoom:
            scale = float(projection) * float(margin) / FIT_MARGIN
            if rotation_safe:
                # Fit the diagonal (the bounding sphere), which no rotation can grow.
                scale *= math.sqrt(sum(e * e for e in extent)) / max(extent)
            # A floor on how close the camera goes, so a sequence over objects of very
            # different sizes does not lurch in and out.
            moves["projectionScale"] = max(scale, float(min_scale)) if min_scale else scale
        return self.tween(at=at, seconds=seconds, ease=ease, **moves)

    def hold(self, seconds: float) -> "Timeline":
        """Advance the cursor. The only thing that moves it."""
        self.cursor += float(seconds)
        return self

    def _path(self, layer: str | None, prop: str) -> tuple[str, ...]:
        if layer is None:
            return (prop,)
        if _layer_in(self.base, layer) is None:
            known = ", ".join(repr(lyr.get("name")) for lyr in self.base.get("layers") or []
                              if isinstance(lyr, dict))
            raise AnimateProblem(
                f"no layer named {layer!r} in this state. It has: {known}")
        return ("layers", layer, prop)

    def _add(self, tween: Tween) -> None:
        for other in self.tweens:
            if other.path != tween.path:
                continue
            # Windows may touch (one tween handing off to the next) but not overlap. The
            # tolerance absorbs float rounding: 4.0 + 7*0.8 + 0.8 != 4.0 + 8*0.8.
            if tween.at < other.stop - _TIME_EPSILON and other.at < tween.stop - _TIME_EPSILON:
                raise AnimateProblem(
                    f"two tweens on {_describe(tween.path)} overlap: "
                    f"[{other.at:g}, {other.stop:g}] and [{tween.at:g}, {tween.stop:g}]. "
                    f"Split them into non-overlapping windows, or set the value once")
            if tween.seconds == 0.0 and other.seconds == 0.0 and tween.at == other.at:
                raise AnimateProblem(
                    f"{_describe(tween.path)} is set twice at t={tween.at:g}")
        self.tweens.append(tween)

    # -- reading out -------------------------------------------------------- #
    @property
    def duration(self) -> float:
        """Seconds. The furthest tween end, or the cursor, whichever is later."""
        if self._duration is not None:
            return self._duration
        return max([self.cursor] + [t.stop for t in self.tweens])

    @duration.setter
    def duration(self, seconds: float | None) -> None:
        self._duration = None if seconds is None else float(seconds)

    @property
    def frame_count(self) -> int:
        """Frames, **including** the last one — a 2 s timeline at 30 fps is 61 frames."""
        return int(round(self.duration * self.fps)) + 1

    def resolved(self) -> list[Tween]:
        """The tweens in application order, each with its ``start`` filled in.

        On each property, the first tween starts from the base state and each later one from
        where the previous ended.
        """
        out: list[Tween] = []
        previous: dict[tuple[str, ...], Any] = {}
        for tween in sorted(self.tweens, key=lambda t: (t.path, t.at, t.seconds)):
            start = tween.start
            if start is None:
                if tween.path in previous:
                    start = previous[tween.path]
                else:
                    start = _get_at(self.base, tween.path)
                    if start is None:
                        start = _default_for(tween.path)
            if start is None and (tween.kind or kind_for(tween.path[-1])) != "step":
                raise AnimateProblem(
                    f"nothing to start from for {_describe(tween.path)}: the state does not "
                    f"carry it and there is no known default. Pass start= explicitly")
            out.append(replace(tween, start=start))
            # The next tween starts where this one ends: the interpolator at t=1, since a
            # relative move's `end` is a delta.
            previous[tween.path] = INTERPOLATORS[tween.kind or kind_for(tween.path[-1])](
                start, tween.end, 1.0)
        return sorted(out, key=lambda t: (t.at, t.seconds, t.path))

    def at(self, seconds: float) -> dict:
        """The state at ``seconds``. A plain dict; the caller's base is never mutated."""
        state = copy.deepcopy(self.base)
        for tween in self.resolved():
            value = tween.value_at(float(seconds), tween.start)
            if value is not _NOTHING:
                _set_at(state, tween.path, value)
        return state

    def frames(self, *, start_frame: int = 0,
               end_frame: int | None = None) -> Iterator[tuple[int, float, dict]]:
        """``(index, seconds, state)`` per frame.

        Times are ``index / fps``, computed rather than accumulated, so they do not drift.
        """
        stop = self.frame_count if end_frame is None else min(end_frame, self.frame_count)
        for i in range(max(0, start_frame), stop):
            t = i / self.fps
            yield i, t, self.at(t)

    def check(self) -> list[str]:
        """Raise on anything unworkable; return notes worth printing. Called by the renderer."""
        notes = list(self.notes)
        self.resolved()
        if self.duration <= 0:
            notes.append("this timeline has zero duration, so it renders a single frame")
        for tween in self.tweens:
            if len(tween.path) == 3 and _layer_in(self.base, tween.path[1]) is None:
                raise AnimateProblem(f"tween names layer {tween.path[1]!r}, which is not in "
                                     f"the state any more")
            if (tween.kind or kind_for(tween.path[-1])) == "step" and tween.seconds > 0:
                notes.append(f"{_describe(tween.path)} does not interpolate, so its "
                             f"{tween.seconds:g}s window is a step at the end of it")
        return notes

    # -- serialization ------------------------------------------------------ #
    def to_json(self) -> dict:
        """The whole timeline, base state included, for writing beside the frames."""
        return {"fps": self.fps, "duration": self.duration, "base": self.base,
                "groups": {k: list(v) for k, v in self.groups.items()},
                "tweens": [t.to_json() for t in self.tweens]}

    @classmethod
    def from_json(cls, obj: Mapping[str, Any]) -> "Timeline":
        """The inverse of :meth:`to_json`, so a rendered timeline can be edited and re-run."""
        tl = cls(base=obj["base"], fps=float(obj.get("fps", 30.0)))
        tl.tweens = [Tween.from_json(t) for t in obj.get("tweens", ())]
        tl.groups = {k: tuple(v) for k, v in (obj.get("groups") or {}).items()}
        if obj.get("duration") is not None:
            tl.duration = float(obj["duration"])
        return tl
