"""Dask cluster startup for local or SLURM execution.

The config's first ``jobqueue`` key selects the cluster type: ``local`` ->
LocalCluster, ``slurm`` -> SLURMCluster (``scale`` submits sbatch). The client
process must stay alive for the whole run; on exit the cluster (and its SLURM jobs)
are torn down.

Config loading — bundled templates, layered overrides, key validation — lives in
:mod:`blockrun.dask_config`.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Iterator, Sequence

import dask
from dask.distributed import Client

from .dask_config import deep_merge, load_config, write_effective_config

# Pin every worker to a single thread for numerical libs, so N single-threaded
# dask workers on a node don't each spawn a full BLAS thread-pool and oversubscribe.
_THREAD_PIN = [
    f"export {v}=1"
    for v in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
]


@contextlib.contextmanager
def start_dask(
    num_workers: int,
    config_path: str | Sequence[str] = "dask-local",
    label: str = "job",
    logger: logging.Logger | None = None,
    wait_timeout: int = 600,
    effective_config_path: str | None = None,
) -> Iterator[Client]:
    """Start a Dask cluster from one or more YAML configs; yield a connected Client.

    ``config_path`` is a bundled name (see :func:`bundled_configs`) or a path, or a
    **sequence of either**, deep-merged left to right — so a site config carries only
    the keys that differ from a template. Unrecognised keys raise rather than merging
    silently; see :mod:`blockrun.dask_config`.

    The default is the bundled ``dask-local``, so ``start_dask(4)`` works from any
    directory. It used to be the relative path ``"dask-config.yaml"``, which resolved
    only from whichever directory happened to contain such a file.

    The config's first ``jobqueue`` key selects the cluster type:
      - ``"local"``: a LocalCluster with ``num_workers`` single-threaded workers.
      - ``"slurm"``: a SLURMCluster; ``scale(num_workers)`` submits sbatch jobs.

    Tears the cluster (and SLURM jobs) down on exit.

    ``effective_config_path`` (a directory or file) records the merged result. Worth
    passing from a run directory: with layered overrides, "what config did this run
    actually use" is otherwise only answerable by replaying the merge by hand.
    """
    logger = logger or logging.getLogger(__name__)
    config, sources = load_config(config_path)

    if not config.get("jobqueue"):
        raise ValueError(
            f"no 'jobqueue' section in {' + '.join(sources)}; it selects the cluster "
            f"type ('local' or 'slurm')")
    cluster_type = next(iter(config["jobqueue"]))

    # Default the job name to the caller's label, so `squeue` and the per-job worker
    # logs say WHICH command submitted them. dask-jobqueue's own default is the
    # uninformative "dask-worker", and `name` drives two things: `#SBATCH -J` and the
    # `<log-directory>/<name>-<jobid>.{err,out}` filenames. Deriving it from the
    # command means it does not have to be edited into every site config — and a
    # config that DOES set it still wins.
    section = config["jobqueue"].get(cluster_type) or {}
    if "name" not in section:
        config = deep_merge(config, {"jobqueue": {cluster_type: {"name": label}}})

    # Dump AFTER the injection, so the record matches what dask actually received.
    if effective_config_path:
        written = write_effective_config(effective_config_path, config, sources)
        logger.info("dask config: %s -> %s", " + ".join(sources), written)
    else:
        logger.info("dask config: %s", " + ".join(sources))

    dask.config.update(dask.config.config, config)  # load YAML into dask's global config

    if cluster_type == "local":
        from dask.distributed import LocalCluster

        cluster = LocalCluster(n_workers=num_workers, threads_per_worker=1)
    elif cluster_type == "slurm":
        from dask_jobqueue import SLURMCluster

        cluster = SLURMCluster(job_script_prologue=_THREAD_PIN)  # reads jobqueue.slurm.* from config
        cluster.scale(num_workers)                               # <-- submits sbatch jobs
    else:
        raise ValueError(f"unsupported cluster type: {cluster_type!r}")

    client = Client(cluster)
    try:
        try:
            client.wait_for_workers(num_workers, timeout=wait_timeout)
        except TimeoutError:
            logger.warning(
                "Not all workers up within %ss; proceeding with whatever is up "
                "(more will join as SLURM schedules them).",
                wait_timeout,
            )
        logger.info(
            "[%s] %d workers requested. Dashboard: %s",
            label,
            num_workers,
            client.dashboard_link,
        )
        yield client
    finally:
        client.shutdown()
        client.close()
