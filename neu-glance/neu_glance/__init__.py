"""neu-glance — neuroglancer states, layers and links.

Everything a viewer consumes, and no data output. neu-vol and neu-mark write volumes and
annotation sources without knowing a viewer exists; viewer knowledge (shaders, states,
links) lives here.

What it makes:

- a **state**, as a URL or JSON (:mod:`neu_glance.state`);
- a **layer** for a volume or precomputed annotation source on a store
  (:mod:`neu_glance.sources`);
- a **layer** of local annotations, from coordinates or occupancy boxes
  (:mod:`neu_glance.layers`).

What it runs, both needing a browser: a **viewer** over arrays held in this process
(:mod:`neu_glance.serving`), and an **animation**, a timeline of tweens rendered frame by
frame through a real viewer (:mod:`neu_glance.animate`, :mod:`neu_glance.rendering`).

Only :mod:`neu_glance.sources` reads a store; ``layers``, ``state`` and ``animate`` are
pure. Top-level names resolve lazily (PEP 562), so ``neu-glance --help`` does not import
neu-vol.
"""

from __future__ import annotations

__version__ = "0.1.0"

#: name -> module it lives in, explicit so a light name cannot pull in a heavy module.
_EXPORTS = {
    "build_state": "state",
    "state_url": "state",
    "parse_url": "state",
    "load_layer": "state",
    "load_state": "state",
    "merge_into": "state",
    "default_view": "state",
    "annotation_extent": "state",
    "split_segment_layer": "state",
    "subset_layers": "state",
    "select_segments": "sources",
    "segment_labels": "sources",
    "segment_tags": "sources",
    "segment_boxes": "sources",
    "read_segment_properties": "sources",
    # Animation: `animate` is stdlib only; `rendering` needs the `serve` extra.
    "Timeline": "animate",
    "Tween": "animate",
    "EASINGS": "animate",
    "AnimateProblem": "animate",
    "record": "rendering",
    "render_state": "rendering",
    "ffmpeg_command": "rendering",
    "RenderProblem": "rendering",
    "volume_layer": "sources",
    "volume_extent": "sources",
    "annotation_layer": "sources",
    "annotation_layer_pair": "sources",
    "read_annotation_info": "sources",
    "annotation_source_extent": "sources",
    "annotation_source_voxel_size": "sources",
    "local_layer": "layers",
    "boxes_layer": "layers",
    "build_annotation": "layers",
    "read_annotation_csv": "layers",
    "output_dimensions": "layers",
    "rescale": "layers",
    "render": "layers",
    "SHADERS": "shaders",
    "IMAGE_SHADERS": "shaders",
    "pick_shader": "shaders",
    "image_shader": "shaders",
    # Hosting arrays in this process: `serve` is the function, `serving` the module.
    "serve": "serving",
    "stop_serving": "serving",
    "ServedLayer": "serving",
    "Server": "serving",
    "ServeProblem": "serving",
}

__all__ = ["__version__", *sorted(_EXPORTS)]


def __getattr__(name: str):
    """Resolve a top-level export on first use."""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(f".{module}", __name__), name)


def __dir__():
    return list(__all__)
