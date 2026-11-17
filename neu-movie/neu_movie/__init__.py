"""Animations of EM neurons and neuropils: a TOML spec in, a rendered movie out.

A spec names the volumes (by volume-library name or URL), the opening camera, the sets of
cells to show and in what order (beats), and how the camera moves. ``neu-movie build`` turns it into a
BUILD: a folder holding everything one invocation made — the spec it was made from, the
resolved body ids, the neuVid input and Blender files, then frames and the movie. Rendering
is neuVid's, in Blender, usually on a Slurm GPU node (``neu-movie submit``).
"""

__version__ = "0.1.0"
