"""Start, pause, resume, stop, extend, copy and queue runs.

Each run trains in its own background process (runs/worker.py), detached from the GUI and the
terminal: closing the browser or the app does not stop training. Only one run uses the GPU at a time;
starting another while one is active puts it in the queue, and the worker starts the next queued run
when it ends.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..util import read_json, write_json
from ..workspace import Project, Workspace
from .config import RunConfig
from .store import ACTIVE, RESUMABLE, Run

log = logging.getLogger("model_trainer.manager")


class Manager:
    def __init__(self, workspace: Workspace) -> None:
        self.ws = workspace

    # ---------------------------------------------------------------------- queries
    def active(self) -> list[Run]:
        return [Run(p) for p in self.ws.all_runs() if Run(p).state().get("state") in ACTIVE]

    def interrupted(self) -> list[Run]:
        return [Run(p) for p in self.ws.all_runs() if Run(p).state().get("state") == "interrupted"]

    def queue(self) -> list[dict[str, Any]]:
        return read_json(self.ws.queue_path, [])

    # --------------------------------------------------------------------- creation
    def create(self, project: Project, cfg: RunConfig) -> Run:
        problems = cfg.problems()
        if problems:
            raise ValueError("; ".join(problems))
        return Run.create(project, cfg)

    def copy(self, run: Run, name: str | None = None) -> Run:
        """A new run with the same settings (to change something and train again)."""
        cfg = run.config
        cfg.name = name or f"{cfg.name}-copy"
        return Run.create(Project(run.path.parents[1]), cfg)

    # ----------------------------------------------------------------------- control
    def start(self, run: Run, mode: str = "start", queue_if_busy: bool = True) -> str:
        """Launch the worker (or queue it). Returns 'started' or 'queued'."""
        state = run.state().get("state")
        if state in ACTIVE:
            raise RuntimeError(f"{run.id} is already {state}")
        if mode == "start" and state in ("paused", "interrupted") and run.resume_ckpt.is_file():
            mode = "resume"
        busy = [r for r in self.active() if r.path != run.path]
        if busy and queue_if_busy:
            self._enqueue(run, mode)
            run.set_state("queued", queued_mode=mode)
            return "queued"
        self._launch(run, mode)
        return "started"

    def resume(self, run: Run) -> str:
        state = run.state().get("state")
        if state not in RESUMABLE:
            raise RuntimeError(f"cannot resume a run that is {state}")
        if not run.resume_ckpt.is_file():
            return self.start(run, "start")  # never finished an epoch: start over
        return self.start(run, "resume")

    def extend(self, run: Run, epochs: int) -> str:
        return self.start(run, f"extend:{int(epochs)}")

    def pause(self, run: Run) -> None:
        self._command(run, "pause")

    def stop(self, run: Run, now: bool = False) -> None:
        if run.state().get("state") == "queued":
            self._dequeue(run)
            run.set_state("created")
            return
        self._command(run, "stop_now" if now else "stop")

    def _command(self, run: Run, command: str) -> None:
        run.request(command)
        if run.state().get("remote"):
            from ..remote import send_command

            send_command(self.ws, run, command)

    def delete(self, run: Run) -> None:
        if run.state().get("state") in ACTIVE:
            raise RuntimeError("stop the run before deleting it")
        self._dequeue(run)
        shutil.rmtree(run.path, ignore_errors=True)

    # ------------------------------------------------------------------------- queue
    def _enqueue(self, run: Run, mode: str) -> None:
        q = [e for e in self.queue() if e["run"] != str(run.path)]
        q.append({"run": str(run.path), "mode": mode})
        write_json(self.ws.queue_path, q)

    def _dequeue(self, run: Run) -> None:
        write_json(self.ws.queue_path, [e for e in self.queue() if e["run"] != str(run.path)])

    def move_in_queue(self, run: Run, delta: int) -> None:
        q = self.queue()
        i = next((k for k, e in enumerate(q) if e["run"] == str(run.path)), None)
        if i is None:
            return
        j = max(0, min(len(q) - 1, i + delta))
        q.insert(j, q.pop(i))
        write_json(self.ws.queue_path, q)

    def start_next_queued(self) -> Run | None:
        if self.active():
            return None
        q = self.queue()
        while q:
            e = q.pop(0)
            write_json(self.ws.queue_path, q)
            run = Run(Path(e["run"]))
            if not run.config_path.is_file() or run.state().get("state") != "queued":
                continue
            sweep = run.state().get("sweep")
            if sweep:
                from ..sweeps import over_budget

                if over_budget(self.ws, sweep):
                    run.set_state("created", note="skipped: the sweep's time budget was used up")
                    continue
            self._launch(run, e.get("mode", "start"))
            return run
        return None

    # ------------------------------------------------------------------------ launch
    def _launch(self, run: Run, mode: str) -> None:
        if run.config.options.executor.startswith("ssh:"):
            from ..remote import launch_remote

            launch_remote(self.ws, run, mode)
            return
        self.start_process(run, mode)

    def start_process(self, run: Run, mode: str, after: int | None = None) -> None:
        """The worker process on this machine; `after`: wait for that process to exit first."""
        weights_cache = (
            self.ws.cache_dir / "weights"
        )  # Ultralytics downloads base models into the cwd
        weights_cache.mkdir(parents=True, exist_ok=True)
        run.clear_command()
        run.set_state(
            "preparing" if not (run.data_dir / "manifest.json").is_file() else "running", mode=mode
        )
        log_fh = run.log_path.open("ab")
        cmd = [sys.executable, "-m", "model_trainer.runs.worker", str(run.path), mode]
        if after:
            cmd.append(f"--after={after}")
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
        kw: dict[str, Any] = {
            "cwd": weights_cache,
            "stdout": log_fh,
            "stderr": subprocess.STDOUT,
            "env": env,
            "stdin": subprocess.DEVNULL,
        }
        if sys.platform == "win32":
            flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
            try:
                proc = subprocess.Popen(
                    cmd, creationflags=flags | 0x01000000, **kw
                )  # CREATE_BREAKAWAY_FROM_JOB
            except OSError:
                proc = subprocess.Popen(cmd, creationflags=flags, **kw)
        else:
            proc = subprocess.Popen(cmd, start_new_session=True, **kw)
        log_fh.close()
        run.heartbeat(proc.pid)
        run.update_state(pid=proc.pid)
        log.info("started %s (%s), pid %d", run.id, mode, proc.pid)
