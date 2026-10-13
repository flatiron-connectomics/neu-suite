# Parallel maps with Dask + SLURM

How to run an **embarrassingly-parallel "apply `fn` to each of N independent
items"** workload on a SLURM cluster, using `dask-jobqueue` to launch the workers
as SLURM jobs. Nothing here is specific to one site: the cluster's partitions,
sizing and quotas are values you look up once (§10) and put in a config, not
things this document knows.

> **Scope.** This is the orchestration background — the mental model, the
> operational ladder, and the failure modes. It is *not* a copy-paste toolkit:
> `blockrun` already provides `start_dask`, `block_map` and `Manifest`, and the
> config templates ship with it. Read **Gotchas** before launching; several are
> silent failure modes.

---

## 1. When to use this (and when not to)

Use it when: the work is **many independent tasks** (one per file / object /
chunk), each doing nontrivial CPU work, and you want them spread across SLURM
workers with results collected back in the driver.

**Don't reach for dask if** the tasks are pure fire-and-forget with no in-Python
result passing — a plain `sbatch` array, or
[disBatch](https://github.com/flatironinstitute/disBatch), is simpler and lighter
on the scheduler (disBatch runs *inside* an existing allocation rather than
creating its own). Dask earns its keep when you want: in-memory result passing,
multi-stage task graphs with dependencies, futures/adaptive scaling, **or the
same code running locally and on SLURM** by swapping one config.

---

## 2. Mental model

- **Scheduler** runs in *your driver process*. It holds the task graph and hands
  tasks to workers.
- **Workers** are `dask-worker` processes. `dask_jobqueue.SLURMCluster.scale(N)`
  **generates an sbatch script and submits it** — the launched workers connect
  back to the scheduler over TCP. You never call `sbatch` yourself; dask does. On
  driver exit, the jobs are `scancel`led automatically.
- The driver typically runs on a **workstation or login node**, workers on
  **compute nodes**, connected over **ethernet — no InfiniBand needed** for
  loosely-coupled maps. **The driver must stay alive for the whole run** (use
  `tmux`/`nohup`/a background process); if it dies, the cluster and its SLURM
  jobs are torn down.

---

## 3. Dependencies

Install `blockrun`; it pulls `dask`, `distributed`, `dask-jobqueue` and `pyyaml`,
and nothing else. Plus whatever your `fn` needs.

---

## 4. Starting a cluster

Use `blockrun.start_dask` rather than writing your own — it is the same helper
every tool in the suite launches through, so a fix reaches all of them:

```python
from blockrun import start_dask, block_map

with start_dask(48, ["dask-slurm-example", "my-site.yaml"], label="my-job") as client:
    results = block_map(items, work_fn, client=client)
```

What it does for you, and why each part exists:

- **Picks the cluster type from the config's first `jobqueue` key** — `local` for
  a `LocalCluster`, `slurm` for a `SLURMCluster` it then `scale()`s. The same call
  serves dev and production; only the config changes.
- **Pins numerical-library threading** in the job prologue
  (`OMP_NUM_THREADS=1` and friends), so N single-threaded dask workers on a node
  don't each spawn a full BLAS thread-pool and oversubscribe it.
- **Names the SLURM job after the calling command**, so `squeue` says which tool
  submitted it.
- **Tears the cluster down** — and `scancel`s its jobs — on exit, including on
  exception.
- `effective_config_path=<run dir>` records the **merged** config beside the run.
  With layered overrides, "what settings did this run actually use" is otherwise
  only answerable by replaying the merge by hand.

It does not block forever waiting for a full cluster: past `wait_timeout` it
proceeds with the workers that are up and lets the rest join as SLURM schedules
them.

---

## 5. The config

Two templates ship with `blockrun`: **`dask-local`** (the default) and
**`dask-slurm-example`**. `config_path` takes a bundled name or a path, or a
sequence of either, **deep-merged left to right** — so your site config is the
handful of keys that actually differ, not a fork of the template that silently
drifts from it:

```yaml
# my-site.yaml — everything else comes from dask-slurm-example
jobqueue:
  slurm:
    queue: my-partition
    log-directory: /path/with/room
```

Dicts merge; scalars and lists **replace**. **Unrecognised keys raise** rather
than deep-merging silently into nothing — a typo would otherwise give you a run
using different settings than you wrote, with no error.

**Knobs that matter**, all in `jobqueue.slurm` and mapping ~1:1 onto `#SBATCH`
flags: `cores` = CPUs per job; `processes` = dask workers per job;
`memory / processes` = RAM per worker; `cores / processes` = threads per worker
(keep at 1, paired with the thread pinning above). To give each worker more RAM,
**lower `processes`** — fewer, fatter workers. `start_dask(num_workers=48)`
against `processes: 24` means 2 jobs.

Size `cores` and `memory` to the **smallest node you'd accept**, and prefer *not*
to constrain the node type: a partition usually holds several, and staying
eligible on all of them starts sooner than waiting for the biggest. The template
carries the memory-manager block at the bottom, which is not site-specific and
should be carried over verbatim — see Gotcha #1.

---

## 6. The application pattern

Express the work as a map over a list. `fn` must be **top-level / importable**
(picklable) and ideally **idempotent** (Gotcha #6). `blockrun.block_map` batches
the submission and streams results back through `as_completed`, with an
`on_result` callback for durable progress; `blockrun.iter_blocks` tiles an N-d
shape if your items are volume blocks, and `blockrun.Manifest` is a
single-writer JSONL completion log for resume.

Keep the **driver lightweight**: compute the work-list (and any shared read-only
context) in the driver, pass each worker only what it needs, and have workers
**write their outputs to shared storage** and return small status tuples — don't
ship large arrays back through the scheduler.

---

## 7. Smoke test, then scale (do this every time)

1. **Local, tiny:** the `dask-local` config, 2 workers, 3–5 items. Confirm `fn`
   runs, outputs land, results return. This catches pickling errors, path bugs
   and logic bugs without touching SLURM.
2. **SLURM, small:** your site config, `num_workers` = one node's worth, full but
   modest item list. Confirm jobs start (`squeue`) and throughput is sane.
3. **SLURM, full:** scale to your CPU budget (Gotcha #4).

---

## 8. Launch and monitor

```bash
# Launch in a session that will outlive your terminal:
nohup python -u run.py > run.log 2>&1 &        # or run inside tmux
# Dashboard URL is printed by start_dask (http://<host>:8787/status).
```

Monitoring — read-only, and **don't poll in tight loops**, since the SLURM
service is shared:

```bash
squeue -u "$USER" -o "%.18i %.22j %.8T %.11M %R"                     # your jobs + state
sacct -u "$USER" --starttime now-1day -o JobID,State,MaxRSS,Elapsed  # post-hoc / OOM check
# progress = count of outputs produced so far, not by hammering squeue
```

---

## 9. Gotchas (symptom → fix) — read before launching

1. **Workers hang, never run; logs say "Worker is at 0% memory usage. Pausing
   worker."** — `distributed.worker.memory` thresholds were set to **`0.0`**,
   which means "trigger at 0% usage" → pause immediately → deadlock. **Use
   `false`** to disable memory management, not `0.0`.

2. **Each worker spawns many threads / the node is oversubscribed / it's slower
   than expected.** — numpy/BLAS multithreading. Ensure the thread-pin prologue
   is applied (`start_dask` does it) and `cores == processes`, i.e. one thread
   per worker.

3. **`KilledWorker` kills the whole run after a few retries.** — A worker was
   **OS-OOM-killed**, or hit the SLURM `--mem` cgroup limit. This is
   **uncatchable in-process**: dask retries the task on other workers, they die
   too, and the phase fails. dask's *memory management* (target/spill) does
   **not** help — it governs dask-tracked managed memory, not arbitrary
   allocations such as a big array read. Fixes, in order of preference:
   - **Bound per-task memory**: cap the input each task loads; for oversized
     items, fall back to a coarser resolution, or chunk and merge.
   - **Run the heavy part in a forked child** with `resource.setrlimit(RLIMIT_AS,
     cap)`, so a runaway dies *catchably* in the child and you skip that item.
   - Give workers more RAM (lower `processes`).

   Keep `per_worker_RAM × tasks_in_flight` under the node limit.

4. **Jobs sit PENDING forever, or only some start.** — You exceeded your
   **per-user QOS CPU cap**. `scale()` past it and the extra jobs simply pend.
   Size `num_workers` so `jobs × cores ≤ cap`; §10 is how to find the number.

5. **A node crashes mid-run and you lose everything.** — All workers were on one
   node. **Spread across ≥2 nodes** for `NODE_FAIL` resilience: size `cores` to a
   whole node and request 2+ jobs.

6. **A run dies or times out partway and you have to redo it all.** — Make tasks
   **idempotent**: at the top of `fn`, skip items whose output already exists; in
   the driver, filter the work-list to not-yet-done items (this is what
   `Manifest` is for). Then a failed run *resumes* by re-launching the same
   command, which also lets you grow the walltime budget across restarts.

7. **The run exceeds `walltime` and SLURM kills the jobs.** — `walltime` is **per
   job**. Estimate `N_items × per_item_time / num_workers`; if it is near the
   limit, raise `walltime` and/or rely on idempotent resume from #6.

8. **Transient read errors or timeouts under many parallel workers.** — Dozens
   of workers hammering shared storage can momentarily time out. Add
   **retry-with-backoff** around the read in `fn`, and classify: retry the
   transient errors, fail fast on the permanent ones.

9. **`scale()` is real job submission.** It issues `sbatch` and counts against
   your QOS. If your site expects allocation to be a manual checkpoint, treat
   launching the driver as that step. Monitor only with read-only
   `squeue`/`sacct`/`seff`/`scontrol show`.

10. **Too many or too few partitions.** One task per tiny item → scheduler
    overhead dominates; one giant partition → no parallelism. Aim for a few ×
    `num_workers` partitions, or chunk items so each task is seconds-to-minutes
    of work.

---

## 10. Know your site (what to look up once)

None of these have portable values — get them from your cluster's documentation
or its admins, then write them into a site config that you keep **outside** this
repo:

- **Partitions**, and the node types in each: core count, RAM per node, and
  therefore GB/core. This sets `cores`, `processes` and `memory`. Note whether a
  partition is heterogeneous, which is the argument against constraining node
  type.
- **Your per-user QOS cap** on concurrent CPUs (and jobs). This bounds
  `num_workers` — see Gotcha #4. `sacctmgr show user $USER withassoc` and
  `sinfo` are the read-only ways to ask.
- **Whether you need `account`** (`-A`). With a single account association SLURM
  applies your default and `dask-jobqueue` omits the directive entirely; set it
  only if you belong to several. An account and a partition may share a name, so
  check which one you mean.
- **Which filesystem to write to.** Worker logs and large outputs want capacity
  and inodes, not a backed-up home directory — `log-directory` in particular
  should not be `$HOME`. Bulk storage is also usually what compute nodes can see.
- **Whether the environment is modules or conda**, and don't mix the two: they
  are built against different libraries, so mixing gives ABI mismatches at
  runtime. Build the environment one way.
- **Local etiquette.** The scheduler is shared: scope queries (`squeue -u
  $USER`), never poll in loops, and don't submit beyond your QOS.

---

## 11. Minimal end-to-end skeleton

```
myproject/
├── my-site.yaml     # §5: the few keys that differ from dask-slurm-example
└── run.py           # §6: build items, define work_fn, main()
```

```bash
# 1. smoke test against the local config and a few items, then:
python run.py
# 2. full run, surviving logout:
nohup python -u run.py > run.log 2>&1 &
squeue -u "$USER"
```

The per-application work is `work_fn` and the item list; the orchestration is
`start_dask` + `block_map` + a config.
