"""Block-map engine.

A job is a map over **output chunks**: tile the output grid into blocks (:class:`Block`),
and run a per-block function that reads the input it needs, transforms it, and writes its
chunk. Tasks should be idempotent, so a relaunch resumes, and should return small status
values rather than arrays: outputs go to the store, not through the scheduler.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Iterator, Sequence

# A per-axis tuple of slices bounding a block in a volume's index space.
Region = tuple[slice, ...]


@dataclass(frozen=True)
class Block:
    """One output chunk: its grid index and its region in the output volume."""

    index: tuple[int, ...]   # position in the chunk grid, per axis
    region: Region           # slices into the output volume

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(s.stop - s.start for s in self.region)


def iter_blocks(shape: Sequence[int], chunks: Sequence[int]) -> Iterator[Block]:
    """Tile ``shape`` into blocks of ``chunks`` (edge blocks are clipped).

    Axis order is whatever ``shape``/``chunks`` use (canonical ``(c,) z, y, x``).
    """
    if len(shape) != len(chunks):
        raise ValueError(f"shape {tuple(shape)} and chunks {tuple(chunks)} differ in rank")
    if any(c <= 0 for c in chunks):
        raise ValueError(f"chunks must be positive, got {tuple(chunks)}")
    grid = [-(-s // c) for s, c in zip(shape, chunks)]  # ceil division per axis
    for index in product(*(range(g) for g in grid)):
        region = tuple(
            slice(i * c, min((i + 1) * c, s))
            for i, c, s in zip(index, chunks, shape)
        )
        yield Block(index=index, region=region)


def _apply_batch(batch: Sequence[Block], fn: Callable[[Block], Any]) -> list[Any]:
    """Run ``fn`` over a batch of blocks (one dask task per batch)."""
    return [fn(b) for b in batch]


def block_map(
    blocks: Sequence[Block],
    fn: Callable[[Block], Any],
    *,
    client: Any | None = None,
    npartitions: int | None = None,
    on_result: Callable[[list[Any]], None] | None = None,
) -> list[Any]:
    """Apply ``fn`` to each block, returning the per-block results.

    ``fn`` must be top-level / picklable and should return a *small* status value
    (write big outputs to the store, not back through the scheduler).

    ``on_result``, if given, is called with each batch's results as they arrive, e.g.
    to record them in a :class:`~blockrun.manifest.Manifest` so a run that dies midway
    leaves an accurate record.

    With ``client=None`` it runs serially in-process. Otherwise blocks are split
    round-robin into ``npartitions`` batches (default ``min(len(blocks), 512)``), one
    task each, and results stream back through ``as_completed``.
    """
    blocks = list(blocks)
    if not blocks:
        return []
    if client is None:
        results = []
        for b in blocks:
            r = fn(b)
            results.append(r)
            if on_result is not None:
                on_result([r])
        return results

    from dask.distributed import as_completed

    n_batches = npartitions or min(len(blocks), 512)
    n_batches = max(1, min(n_batches, len(blocks)))
    # round-robin split balances cost across batches (adjacent blocks are similar)
    batches = [b for b in (blocks[i::n_batches] for i in range(n_batches)) if b]
    futures = client.map(_apply_batch, batches, fn=fn, pure=False)

    results: list[Any] = []
    for fut in as_completed(futures):
        res = fut.result()
        if on_result is not None:
            on_result(res)
        results.extend(res)
        fut.release()
    return results


def idempotent(
    fn: Callable[[Block], Any],
    is_done: Callable[[Block], bool],
    *,
    skipped: Any = "skipped",
) -> Callable[[Block], Any]:
    """Wrap ``fn`` so blocks that are already done are skipped.

    ``is_done`` runs on the worker, so keep it cheap (a store existence or metadata
    check).
    """

    def wrapped(block: Block) -> Any:
        if is_done(block):
            return skipped
        return fn(block)

    return wrapped
