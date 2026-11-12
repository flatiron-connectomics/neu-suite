"""Run a recipe's steps, record the run, and report on it.

The first step to exit non-zero stops the run. neu-morpho and neu-skel exit 1 when some
bodies failed, unless given ``--allow-body-failures``; their failures files say which,
and ``--from`` that step retries them. Stopping the runner (Ctrl-C, or ``kill`` on its
pid) stops the step it is running too, and ``status`` then reports the run as stopped.

Each run (not a dry run) writes ``runs/<recipe>_<YYYYmmddTHHMM>/`` beside the recipe::

    recipe.toml      the recipe as run
    record.json      resolved volumes, vars and commands; the git commit and dirty files
                     of each repository involved; per-step start, end and exit code.
                     Rewritten after every step, so a killed run keeps its record.
    env-explicit.txt ``mamba list --explicit`` (or conda's), the environment's exact packages
    pip-freeze.txt   pip's view, which includes editable installs
    steps/<n>-<name>.log   each step's combined stdout and stderr, also echoed here;
                     written line by line, or every N seconds with --log-flush N
    configs/         a copy of every file a step passed as --config

Steps run with ``$NEU_PROGRESS_DIR`` set to ``runs/progress/``, shared by every run, where
neu-vol keeps the manifest of a remote destination; a later run resumes from it.

The repositories recorded are the one holding the recipe and every one holding an
editable install in the running environment, which together are the code a step can
run. Nothing here submits a Slurm job: a step's ``--config`` decides where its workers run.

``status`` reads the latest run: each step's state and time, and for the step running (or
the one that stopped the run) the end of its log and the output of its ``progress``
command. It reads files and runs only those read-only commands, so it is safe to repeat
while a run is in flight.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from importlib import metadata
from pathlib import Path
from urllib.parse import unquote, urlparse

#: What neu-morpho and neu-skel log when --allow-body-failures turns failures into exit 0.
_ALLOWED_FAILURES = re.compile(r"(\d+) bodies failed; exiting 0")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _git(path: Path, *args: str) -> str | None:
    r = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def git_state(path: Path) -> dict:
    """The commit, branch and modified tracked files of the repository at ``path``."""
    commit = _git(path, "rev-parse", "HEAD")
    if commit is None:
        return {"path": str(path), "commit": None}
    # Porcelain lines are "XY path"; X may be a space, so the output is not stripped.
    dirty = _git(path, "status", "--porcelain", "--untracked-files=no") or ""
    return {"path": str(path), "commit": commit.strip(),
            "branch": (_git(path, "branch", "--show-current") or "").strip(),
            "dirty": [line[3:] for line in dirty.splitlines() if line]}


def editable_sources() -> dict[str, Path]:
    """Distribution name -> source directory, for every editable install."""
    out = {}
    for dist in metadata.distributions():
        text = dist.read_text("direct_url.json")
        if not text:
            continue
        try:
            info = json.loads(text)
        except ValueError:
            continue
        url = info.get("url", "")
        if info.get("dir_info", {}).get("editable") and url.startswith("file://"):
            out[dist.metadata["Name"]] = Path(unquote(urlparse(url).path))
    return out


def repositories(recipe: Path) -> dict[str, dict]:
    """The git state of the recipe's repository and of each editable install's, keyed by
    repository directory name; each lists the installed packages it holds."""
    found: dict[Path, list[str]] = {}
    for anchor, package in [(recipe.parent, None),
                            *((p, n) for n, p in sorted(editable_sources().items()))]:
        top = _git(anchor, "rev-parse", "--show-toplevel")
        if top is None:
            continue
        packages = found.setdefault(Path(top.strip()), [])
        if package:
            packages.append(package)
    out: dict[str, dict] = {}
    for top, packages in found.items():
        key, n = top.name, 1
        while key in out:
            n += 1
            key = f"{top.name}-{n}"
        out[key] = {**git_state(top), "packages": packages}
    return out


def _environment(run_dir: Path) -> None:
    tool = shutil.which("mamba") or shutil.which("conda")
    if tool:
        with open(run_dir / "env-explicit.txt", "w") as f:
            subprocess.run([tool, "list", "--explicit"], stdout=f, stderr=subprocess.DEVNULL)
    with open(run_dir / "pip-freeze.txt", "w") as f:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=f,
                       stderr=subprocess.DEVNULL)


def _copy_configs(argv: list[str], cwd: Path, dest: Path, step: str) -> None:
    for flag, value in zip(argv, argv[1:]):
        if flag == "--config":
            src = Path(value) if Path(value).is_absolute() else cwd / value
            if src.is_file():                      # a bundled template name is not a file
                dest.mkdir(exist_ok=True)
                shutil.copy2(src, dest / f"{step}__{src.name}")


def runs_dir(recipe: Path) -> Path:
    return recipe.parent / "runs"


#: Set for every step and ``progress`` command: where neu-vol keeps the manifest of a
#: remote destination, so it outlives the run that wrote it and the next run resumes.
PROGRESS_DIR_ENV = "NEU_PROGRESS_DIR"


def progress_dir(recipe: Path) -> Path:
    return runs_dir(recipe) / "progress"


def step_env(recipe: Path) -> dict[str, str]:
    """The environment a step or ``progress`` command runs in."""
    return {**os.environ, "PYTHONUNBUFFERED": "1",
            PROGRESS_DIR_ENV: str(progress_dir(recipe).resolve())}


def _new_run_dir(recipe: Path) -> Path:
    base = runs_dir(recipe) / f"{recipe.stem}_{datetime.now():%Y%m%dT%H%M}"
    run_dir, n = base, 1
    while run_dir.exists():
        n += 1
        run_dir = base.with_name(f"{base.name}-{n}")
    (run_dir / "steps").mkdir(parents=True)
    return run_dir


class _StepLog:
    """A step's log: each line written as it arrives, or with ``every`` set, buffered and
    flushed every ``every`` seconds by a timer, so a quiet step's last lines still land."""

    def __init__(self, path: Path, every: float = 0.0):
        self._file = open(path, "w", buffering=-1 if every else 1)
        self._lock = threading.Lock()
        self._done = threading.Event()
        if every:
            threading.Thread(target=self._flush_every, args=(every,), daemon=True).start()

    def write(self, text: str) -> None:
        with self._lock:
            self._file.write(text)

    def _flush_every(self, every: float) -> None:
        while not self._done.wait(every):
            with self._lock:
                self._file.flush()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._done.set()
        with self._lock:
            self._file.close()


def stream(argv: list[str], log_path: Path, *, cwd: Path, env: dict[str, str] | None = None,
           flush_every: float = 0.0) -> int:
    """Run one command from ``cwd``, teeing its output to the terminal and a log."""
    env = env or {**os.environ, "PYTHONUNBUFFERED": "1"}
    with _StepLog(log_path, flush_every) as log:
        log.write(f"$ {shlex.join(argv)}\n")
        try:
            proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
        except FileNotFoundError as e:
            log.write(f"{e}\n")
            print(e, flush=True)
            return 127
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
            return proc.wait()
        finally:
            # A runner stopped mid-step (a signal, Ctrl-C) takes its step with it, rather
            # than leaving a driver and its cluster running unrecorded.
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    proc.kill()


def run(recipe: Path, resolved: dict, steps: list[dict], argv: list[str], *,
        log_flush: float = 0.0) -> int:
    """Run ``steps`` (a selection of ``resolved["steps"]``) and record the run."""
    # `kill <pid>` sends SIGTERM, which would otherwise end the runner with no cleanup.
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    cwd = resolved["cwd"]
    run_dir = _new_run_dir(recipe)
    shutil.copy2(recipe, run_dir / "recipe.toml")
    _environment(run_dir)
    record = {
        "recipe": str(recipe), "runner_argv": argv, "host": socket.gethostname(),
        "pid": os.getpid(), "started": _now(), "cwd": str(cwd),
        "progress_dir": str(progress_dir(recipe).resolve()),
        "library": None if resolved["library"] is None else str(resolved["library"]),
        "git": repositories(recipe),
        "volumes": resolved["volumes"], "vars": resolved["vars"],
        # every step of the recipe, for `status`, which then needs neither the recipe
        # nor the library as they are now
        "recipe_steps": [{"index": s["index"], "name": s["name"],
                          "progress": s["progress"]} for s in resolved["steps"]],
        "steps": [{"name": s["name"], "command": shlex.join(s["argv"])} for s in steps],
    }

    def save():
        (run_dir / "record.json").write_text(json.dumps(record, indent=2) + "\n")

    save()
    print(f"run record: {run_dir}", flush=True)
    status = 0
    for step, entry in zip(steps, record["steps"]):
        _copy_configs(step["argv"], cwd, run_dir / "configs", step["name"])
        log_path = run_dir / "steps" / f"{step['index']}-{step['name']}.log"
        print(f"\n=== step {step['index']}: {step['name']}\n$ {entry['command']}",
              flush=True)
        entry["started"] = _now()
        save()
        t = time.time()
        code = stream(step["argv"], log_path, cwd=cwd, env=step_env(recipe),
                      flush_every=log_flush)
        entry.update(finished=_now(), minutes=round((time.time() - t) / 60, 2),
                     exit_code=code, log=str(log_path.relative_to(run_dir)))
        save()
        if code != 0:
            print(f"\n=== step {step['name']} exited {code}; stopping. Log: {log_path}\n"
                  f"    resume with: --from {step['name']}", flush=True)
            status = code
            break
    record["finished"] = _now()
    save()
    if status == 0:
        print(f"\n=== all {len(steps)} step(s) done. Record: {run_dir}", flush=True)
    return status


def runs(recipe: Path) -> list[Path]:
    """The recipe's run directories, oldest first."""
    return sorted(p for p in runs_dir(recipe).glob(f"{recipe.stem}_*")
                  if (p / "record.json").is_file())


def _runner_alive(record: dict) -> bool | None:
    """Is the run's runner still running? ``None`` when it ran on another host."""
    if record.get("host") != socket.gethostname() or "pid" not in record:
        return None
    try:
        with open(f"/proc/{record['pid']}/cmdline", "rb") as f:
            argv = [os.fsdecode(a) for a in f.read().split(b"\0")]
    except OSError:
        return False
    # The pid may since belong to something else: require this runner as the program,
    # as the script an interpreter runs, or as the module `-m` runs.
    return (any(Path(a).name == "neu-recipe" for a in argv[:2])
            or any(a == "-m" and b.split(".")[0] == "neu_recipe"
                   for a, b in zip(argv, argv[1:])))


def _since(iso: str) -> float:
    return (datetime.now().astimezone() - datetime.fromisoformat(iso)).total_seconds() / 60


def _tail(path: Path, n: int = 5) -> list[str]:
    try:
        with open(path, errors="replace") as f:
            return f.read().splitlines()[-n:]
    except OSError:
        return []


def status(recipe: Path) -> int:
    """Print the state of the recipe's latest run."""
    found = runs(recipe)
    if not found:
        print(f"no runs of {recipe.name} yet (looked in {runs_dir(recipe)})")
        return 0
    run_dir = found[-1]
    record = json.loads((run_dir / "record.json").read_text())
    steps = record.get("recipe_steps") or [
        {"index": i, "name": s["name"], "progress": None}
        for i, s in enumerate(record["steps"], 1)]
    cwd = Path(record.get("cwd") or recipe.parent)

    alive = _runner_alive(record)
    if "finished" in record:
        state = f"finished {record['finished']}"
    elif alive is None:
        state = f"no finish recorded; it ran on {record.get('host')}, so cannot tell more"
    elif alive:
        state = f"running (pid {record['pid']}), {_since(record['started']):.0f} min so far"
    else:
        state = "STOPPED: no finish recorded and the runner is gone (killed or crashed)"
    print(f"{run_dir}\n  started {record['started']}  {state}")
    if len(found) > 1:
        print(f"  ({len(found) - 1} earlier run(s) in {run_dir.parent})")

    entries = {e["name"]: e for e in record["steps"]}
    focus = None

    def log_of(step):
        return run_dir / "steps" / f"{step['index']}-{step['name']}.log"

    print()
    for step in steps:
        e = entries.get(step["name"])
        if e is None:
            line = "not in this run"
        elif "exit_code" in e:
            line = (f"done, {e['minutes']} min" if e["exit_code"] == 0
                    else f"FAILED (exit {e['exit_code']}), {e['minutes']} min")
            allowed = [m for x in _tail(log_of(step), 20) for m in _ALLOWED_FAILURES.findall(x)]
            if allowed:
                line += f", {allowed[-1]} bodies failed (allowed)"
            if e["exit_code"] != 0:
                focus = step
        elif "started" in e:
            running = alive is not False and "finished" not in record
            line = (f"running, {_since(e['started']):.0f} min so far" if running
                    else "interrupted")
            focus = step
        else:
            line = "pending"
        print(f"  [{step['index']}] {step['name']:16} {line}")

    if focus is not None:
        log_path = log_of(focus)
        print(f"\n--- end of {log_path.relative_to(run_dir)}")
        for line in _tail(log_path):
            print(f"  {line}")
        if focus.get("progress"):
            print(f"\n--- $ {shlex.join(focus['progress'])}")
            try:
                r = subprocess.run(focus["progress"], cwd=cwd, env=step_env(recipe),
                                   capture_output=True, text=True)
                text = r.stdout + r.stderr
            except FileNotFoundError as e:
                text = str(e)
            for line in text.splitlines():
                print(f"  {line}")
    return 0
