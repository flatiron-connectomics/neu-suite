"""Running a recipe: what a run records, where steps run, stopping, and `status`."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from neu_recipe import runner
from neu_recipe.cli import main

PY = sys.executable


def write_recipe(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "pipelines" / "demo.toml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(text)
    return path


def git(path, *args):
    subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True)


def commit_all(path, *names):
    git(path, "add", *names)
    git(path, "-c", "user.name=t", "-c", "user.email=t@example.org", "commit", "-q",
        "-m", "x")


@pytest.fixture
def quick(monkeypatch):
    """No environment dump and no scan of the real environment's editable installs."""
    monkeypatch.setattr(runner, "_environment", lambda run_dir: None)
    monkeypatch.setattr(runner, "editable_sources", lambda: {})


def run_recipe(tmp_path, text, *extra):
    recipe = write_recipe(tmp_path, text)
    code = main(["run", str(recipe), *extra])
    found = sorted((recipe.parent / "runs").glob("demo_*")) if (
        recipe.parent / "runs").exists() else []
    return code, found


def test_a_run_records_each_step_and_stops_at_the_first_failure(tmp_path, quick, capsys):
    config = tmp_path / "site.yaml"
    config.write_text("jobqueue: {}\n")
    code, found = run_recipe(tmp_path, f'''
[vars]
cfg = "{config}"
[[step]]
name = "ok"
run = "{PY} -c 'print(\\"hello\\")' --config {{cfg}}"
[[step]]
name = "fails"
run = "{PY} -c 'import sys; print(\\"bad\\"); sys.exit(3)'"
[[step]]
name = "never"
run = "{PY} -c 'print(1)'"
''')
    assert code == 3 and len(found) == 1
    run_dir = found[0]
    record = json.loads((run_dir / "record.json").read_text())
    ok, fails, never = record["steps"]
    assert ok["exit_code"] == 0 and fails["exit_code"] == 3 and "exit_code" not in never
    assert [s["name"] for s in record["recipe_steps"]] == ["ok", "fails", "never"]
    assert (run_dir / "steps" / "1-ok.log").read_text().splitlines()[-1] == "hello"
    assert "bad" in (run_dir / "steps" / "2-fails.log").read_text()
    assert (run_dir / "configs" / "ok__site.yaml").read_text() == "jobqueue: {}\n"
    assert (run_dir / "recipe.toml").exists() and record["finished"]
    assert record["pid"] == os.getpid()
    out = capsys.readouterr().out
    assert "hello" in out and "resume with: --from fails" in out


def test_commands_run_from_the_recipes_cwd(tmp_path, quick):
    where = f'''
[[step]]
name = "where"
run = "{PY} -c 'import os; print(os.getcwd())'"
'''
    code, found = run_recipe(tmp_path, where)
    log = found[0] / "steps" / "1-where.log"
    assert code == 0 and log.read_text().splitlines()[-1] == str(
        (tmp_path / "pipelines").resolve())

    code, found = run_recipe(tmp_path, '[recipe]\ncwd = ".."\n' + where)
    log = found[-1] / "steps" / "1-where.log"
    assert code == 0 and log.read_text().splitlines()[-1] == str(tmp_path.resolve())


def test_every_run_shares_one_progress_dir(tmp_path, quick, monkeypatch):
    monkeypatch.delenv(runner.PROGRESS_DIR_ENV, raising=False)
    show = f'''
[[step]]
name = "show"
run = "{PY} -c 'import os; print(os.environ[\\"NEU_PROGRESS_DIR\\"])'"
'''
    run_recipe(tmp_path, show)
    code, found = run_recipe(tmp_path, show)
    want = str((tmp_path / "pipelines" / "runs" / "progress").resolve())
    assert code == 0 and len(found) == 2
    for run_dir in found:
        assert (run_dir / "steps" / "1-show.log").read_text().splitlines()[-1] == want
        assert json.loads((run_dir / "record.json").read_text())["progress_dir"] == want


def test_a_dry_run_runs_and_writes_nothing(tmp_path, quick, capsys):
    marker = tmp_path / "ran"
    code, found = run_recipe(tmp_path, f'''
[[step]]
name = "touch"
run = "{PY} -c 'open(\\"{marker}\\", \\"w\\")'"
''', "--dry-run")
    assert code == 0 and found == [] and not marker.exists()
    assert "[1] touch" in capsys.readouterr().out


def test_a_missing_program_is_a_failed_step_not_a_crash(tmp_path, quick):
    code, found = run_recipe(tmp_path, '''
[[step]]
name = "missing"
run = "definitely-not-a-program-xyz --help"
''')
    assert code == 127
    assert json.loads((found[0] / "record.json").read_text())["steps"][0]["exit_code"] == 127


def test_a_negative_log_flush_is_refused(tmp_path):
    recipe = write_recipe(tmp_path, '[[step]]\nname = "s"\nrun = "t"\n')
    with pytest.raises(SystemExit):
        main(["run", str(recipe), "--log-flush", "-1"])


# --------------------------------------------------------------------------- #
# what the record says about the code
# --------------------------------------------------------------------------- #

def test_the_git_state_names_dirty_files_exactly(tmp_path):
    git(tmp_path, "init", "-q")
    for name in (".hidden", "plain"):
        (tmp_path / name).write_text("a")
    commit_all(tmp_path, ".hidden", "plain")
    (tmp_path / ".hidden").write_text("b")
    (tmp_path / "plain").write_text("b")
    state = runner.git_state(tmp_path)
    assert len(state["commit"]) == 40
    assert state["dirty"] == [".hidden", "plain"]
    assert runner.git_state(tmp_path / "not-a-repo")["commit"] is None


def test_the_record_names_the_recipes_repository_and_each_editable_installs(
        tmp_path, monkeypatch):
    recipes, code = tmp_path / "recipes", tmp_path / "code"
    for repo in (recipes, code):
        (repo / "pkg_a").mkdir(parents=True)
        git(repo, "init", "-q")
        (repo / "README").write_text("x")
        commit_all(repo, "README")
    monkeypatch.setattr(runner, "editable_sources", lambda: {
        "pkg-a": code / "pkg_a", "pkg-b": code, "elsewhere": tmp_path / "not-a-repo"})

    found = runner.repositories(recipes / "demo.toml")
    assert set(found) == {"recipes", "code"}
    assert found["recipes"]["packages"] == []
    assert found["code"]["packages"] == ["pkg-a", "pkg-b"]
    assert len(found["code"]["commit"]) == 40


def test_editable_installs_are_read_from_direct_url(monkeypatch, tmp_path):
    class Dist:
        def __init__(self, name, direct_url):
            self.metadata, self._text = {"Name": name}, direct_url

        def read_text(self, filename):
            return self._text if filename == "direct_url.json" else None

    editable = json.dumps({"url": f"file://{tmp_path}/my%20pkg",
                           "dir_info": {"editable": True}})
    wheel = json.dumps({"url": "https://example.org/x.whl", "archive_info": {}})
    monkeypatch.setattr(runner.metadata, "distributions", lambda: [
        Dist("mine", editable), Dist("theirs", wheel), Dist("plain", None)])
    assert runner.editable_sources() == {"mine": tmp_path / "my pkg"}


# --------------------------------------------------------------------------- #
# stopping
# --------------------------------------------------------------------------- #

def test_a_runner_is_alive_only_if_its_pid_runs_this_runner():
    import socket

    here = socket.gethostname()
    # this pid is pytest, not neu-recipe
    assert runner._runner_alive({"host": here, "pid": os.getpid()}) is False
    assert runner._runner_alive({"host": here, "pid": 2 ** 22 + 7}) is False
    assert runner._runner_alive({"host": "elsewhere", "pid": 1}) is None


def test_terminating_the_runner_stops_its_step(tmp_path):
    """Otherwise a killed run leaves its step (a driver and its cluster) running."""
    import time

    pid_file = tmp_path / "child.pid"
    recipe = write_recipe(tmp_path, f'''
[[step]]
name = "slow"
run = "{PY} -c 'import os, time; open(\\"{pid_file}\\", \\"w\\").write(str(os.getpid())); time.sleep(120)'"
''')
    proc = subprocess.Popen([PY, "-m", "neu_recipe", "run", str(recipe)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    deadline = time.time() + 30
    while not pid_file.exists() or not pid_file.read_text():
        assert time.time() < deadline, "the step never started"
        time.sleep(0.1)
    child = int(pid_file.read_text())
    record = json.loads(next((recipe.parent / "runs").glob("demo_*/record.json")).read_text())
    assert runner._runner_alive(record) is True
    proc.terminate()
    assert proc.wait(timeout=60) == 143
    deadline = time.time() + 10
    while os.path.exists(f"/proc/{child}") and time.time() < deadline:
        time.sleep(0.1)
    assert not os.path.exists(f"/proc/{child}")


@pytest.mark.parametrize("flush_every", [0.0, 0.3])
def test_a_quiet_steps_last_line_reaches_its_log_while_it_runs(tmp_path, flush_every):
    """Line by line, or buffered with a timer: either way `status` sees the latest line
    while the step is still silent, not only when it next prints or ends."""
    import threading
    import time

    log = tmp_path / "step.log"
    argv = [PY, "-c", "import time; print('first', flush=True); time.sleep(4); print('second')"]
    worker = threading.Thread(target=runner.stream, args=(argv, log),
                              kwargs={"cwd": tmp_path, "flush_every": flush_every})
    worker.start()
    deadline = time.time() + 3
    output: list[str] = []                    # the lines below the command's own echo
    while time.time() < deadline and "first" not in output:
        time.sleep(0.1)
        output = log.read_text().splitlines()[1:] if log.exists() else []
    assert output == ["first"]
    worker.join()
    assert log.read_text().splitlines()[-1] == "second"


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #

T0 = "2026-01-01T00:00:00+00:00"
RECIPE_STEPS = [{"index": 1, "name": "first", "progress": None},
                {"index": 2, "name": "second",
                 "progress": [PY, "-c", "print('halfway')"]},
                {"index": 3, "name": "third", "progress": None}]


def fake_run(tmp_path, steps, *, finished=False, logs=None):
    """A run directory as the runner leaves it, without running anything."""
    recipe = write_recipe(tmp_path, "# not read by status\n")
    run_dir = recipe.parent / "runs" / "demo_20260101T0000"
    (run_dir / "steps").mkdir(parents=True)
    record = {"started": T0, "host": "here", "pid": 1, "cwd": str(tmp_path),
              "recipe_steps": RECIPE_STEPS, "steps": steps}
    if finished:
        record["finished"] = "2026-01-01T01:00:00+00:00"
    (run_dir / "record.json").write_text(json.dumps(record))
    for name, text in (logs or {}).items():
        (run_dir / "steps" / name).write_text(text)
    return recipe


def status_of(recipe, monkeypatch, capsys, alive):
    monkeypatch.setattr(runner, "_runner_alive", lambda record: alive)
    assert main(["status", str(recipe)]) == 0
    return capsys.readouterr().out


def test_status_shows_the_running_step_its_log_and_its_progress(tmp_path, monkeypatch,
                                                                 capsys):
    recipe = fake_run(tmp_path, [
        {"name": "first", "started": T0, "exit_code": 0, "minutes": 1.5},
        {"name": "second", "started": T0},
        {"name": "third"}], logs={"2-second.log": "a\nb\nlast line\n"})
    out = status_of(recipe, monkeypatch, capsys, alive=True)
    assert "[1] first            done, 1.5 min" in out
    assert "[2] second           running" in out and "[3] third            pending" in out
    assert "last line" in out and "halfway" in out


def test_a_progress_command_reads_the_runs_progress_dir(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv(runner.PROGRESS_DIR_ENV, raising=False)
    monkeypatch.setattr(runner, "_runner_alive", lambda record: True)
    recipe = fake_run(tmp_path, [{"name": "first", "started": T0}])
    record_path = next((recipe.parent / "runs").glob("demo_*")) / "record.json"
    record = json.loads(record_path.read_text())
    record["recipe_steps"][0]["progress"] = [
        PY, "-c", "import os; print('dir', os.environ['NEU_PROGRESS_DIR'])"]
    record_path.write_text(json.dumps(record))
    assert main(["status", str(recipe)]) == 0
    want = (recipe.parent / "runs" / "progress").resolve()
    assert f"dir {want}" in capsys.readouterr().out


def test_status_says_when_the_runner_died_mid_step(tmp_path, monkeypatch, capsys):
    recipe = fake_run(tmp_path, [{"name": "first", "started": T0}])
    out = status_of(recipe, monkeypatch, capsys, alive=False)
    assert "STOPPED" in out and "[1] first            interrupted" in out
    assert "[2] second           not in this run" in out


def test_status_of_a_finished_run_reports_allowed_failures(tmp_path, monkeypatch, capsys):
    recipe = fake_run(tmp_path, [
        {"name": n, "started": T0, "exit_code": 0, "minutes": 1.0}
        for n in ("first", "second", "third")], finished=True,
        logs={"2-second.log": "... WARNING 9 bodies failed; exiting 0 "
                              "(--allow-body-failures)\n"})
    out = status_of(recipe, monkeypatch, capsys, alive=False)
    assert "finished 2026-01-01T01:00:00" in out
    assert "[2] second           done, 1.0 min, 9 bodies failed (allowed)" in out
    assert "halfway" not in out                         # nothing in flight to ask about


def test_status_shows_the_failed_step(tmp_path, monkeypatch, capsys):
    recipe = fake_run(tmp_path, [
        {"name": "first", "started": T0, "exit_code": 0, "minutes": 1.0},
        {"name": "second", "started": T0, "exit_code": 2, "minutes": 0.1}],
        finished=True, logs={"2-second.log": "Traceback ...\nValueError: boom\n"})
    out = status_of(recipe, monkeypatch, capsys, alive=False)
    assert "FAILED (exit 2)" in out and "ValueError: boom" in out and "halfway" in out


def test_status_with_no_runs_says_so(tmp_path, capsys):
    recipe = write_recipe(tmp_path, '[[step]]\nname = "s"\nrun = "t"\n')
    assert main(["status", str(recipe)]) == 0
    assert "no runs of demo.toml yet" in capsys.readouterr().out
