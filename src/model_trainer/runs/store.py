"""A run folder and its state.

<run>/run.yaml              settings (RunConfig)
<run>/state.json            state machine (see STATES), written atomically
<run>/heartbeat.json        the worker's "I am alive" (every few seconds)
<run>/command.json          requests from the GUI/CLI: pause | stop | stop_now
<run>/metrics.jsonl         one line per epoch: losses, metrics, learning rate, time, GPU
<run>/events.jsonl          warnings and milestones
<run>/progress.json         progress inside the current epoch
<run>/log.txt               the worker's console output
<run>/data/                 the frozen training dataset (datasets/prepare.py)
<run>/ultra/train/          Ultralytics' own output (weights/best.pt, plots)
<run>/checkpoints/resume.pt full checkpoint after the last finished epoch (for resuming)
<run>/snapshots/epoch_NNN.pt periodic copies of the weights
<run>/watch/epoch_NNN/      predictions on fixed validation images after each epoch
<run>/results/              evaluation (results.json, confusion, errors)
<run>/package/              the model package (weights, exports, model_card.json)
<run>/env.json              versions, GPU, driver, command line
"""

from __future__ import annotations

import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from ..util import now, pid_alive, read_json, read_jsonl, write_json
from ..workspace import Project, safe_name
from .config import RunConfig

STATES = (
    "created",  # saved, never started
    "queued",  # waiting in the queue
    "preparing",  # building the run dataset
    "running",
    "pausing",  # finishing the current epoch, then pausing
    "paused",  # can be resumed exactly where it stopped
    "stopping",
    "evaluating",  # training ended; scoring the best weights and building the model package
    "stopped",  # stopped by the user; best weights kept, can be extended
    "done",  # finished all epochs (or early stopping)
    "failed",
    "interrupted",  # crashed / power loss: can be resumed from the last epoch
)
ACTIVE = ("preparing", "running", "pausing", "stopping", "evaluating")
RESUMABLE = ("paused", "interrupted", "failed", "stopped", "done")
HEARTBEAT_S = 5
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
STALE_S = 90


class Run:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    # ---------------------------------------------------------------- creation / config
    @classmethod
    def create(cls, project: Project, cfg: RunConfig) -> Run:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = safe_name(f"{stamp}_{cfg.name}")
        path = project.runs_dir / base
        k = 1
        while path.exists():
            path = project.runs_dir / f"{base}_{k}"
            k += 1
        path.mkdir(parents=True)
        run = cls(path)
        cfg.project = project.name
        cfg.save(run.config_path)
        run.set_state("created")
        return run

    @property
    def id(self) -> str:
        return self.path.name

    @property
    def config_path(self) -> Path:
        return self.path / "run.yaml"

    @property
    def config(self) -> RunConfig:
        return RunConfig.load(self.config_path)

    def save_config(self, cfg: RunConfig) -> None:
        cfg.save(self.config_path)

    # -------------------------------------------------------------------------- paths
    data_dir = property(lambda self: self.path / "data")
    ultra_dir = property(lambda self: self.path / "ultra" / "train")
    weights_dir = property(lambda self: self.ultra_dir / "weights")
    resume_ckpt = property(lambda self: self.path / "checkpoints" / "resume.pt")
    snapshots_dir = property(lambda self: self.path / "snapshots")
    watch_dir = property(lambda self: self.path / "watch")
    results_dir = property(lambda self: self.path / "results")
    package_dir = property(lambda self: self.path / "package")
    log_path = property(lambda self: self.path / "log.txt")

    @property
    def best_weights(self) -> Path | None:
        for p in (
            self.package_dir / "best.pt",
            self.weights_dir / "best.pt",
            self.weights_dir / "last.pt",
        ):
            if p.is_file():
                return p
        return None

    # ---------------------------------------------------------------------------- state
    def state(self) -> dict[str, Any]:
        s = read_json(self.path / "state.json", {"state": "created"})
        if s.get("state") in ACTIVE and not self.alive():
            s["state"] = "interrupted"
            s["interrupted_from"] = s.get("state")
        return s

    def set_state(self, state: str, **extra: Any) -> dict[str, Any]:
        assert state in STATES, state
        s = read_json(self.path / "state.json", {})
        s.update(extra)
        s["state"] = state
        s.setdefault("history", []).append([now(), state])
        s["history"] = s["history"][-100:]
        s["updated"] = now()
        write_json(self.path / "state.json", s)
        return s

    def update_state(self, **extra: Any) -> None:
        s = read_json(self.path / "state.json", {})
        s.update(extra)
        write_json(self.path / "state.json", s)

    def heartbeat(self, pid: int) -> None:
        write_json(self.path / "heartbeat.json", {"pid": pid, "time": time.time()})

    def alive(self) -> bool:
        hb = read_json(self.path / "heartbeat.json", {})
        if not hb:
            return False
        fresh = time.time() - hb.get("time", 0) < STALE_S
        if read_json(self.path / "state.json", {}).get("remote"):
            return fresh  # the process lives on another machine: only the synced heartbeat tells
        return fresh and pid_alive(hb.get("pid"))

    # ------------------------------------------------------------------------- commands
    def request(self, command: str) -> None:
        assert command in ("pause", "stop", "stop_now"), command
        write_json(self.path / "command.json", {"command": command, "time": now()})

    def pending_command(self) -> str | None:
        return read_json(self.path / "command.json", {}).get("command")

    def clear_command(self) -> None:
        (self.path / "command.json").unlink(missing_ok=True)

    # -------------------------------------------------------------------------- records
    def metrics(self) -> list[dict[str, Any]]:
        return read_jsonl(self.path / "metrics.jsonl")

    def events(self) -> list[dict[str, Any]]:
        return read_jsonl(self.path / "events.jsonl")

    def progress(self) -> dict[str, Any]:
        return read_json(self.path / "progress.json", {})

    def results(self) -> dict[str, Any]:
        return read_json(self.results_dir / "results.json", {})

    def summary(self) -> dict[str, Any]:
        """One line for run lists."""
        cfg = self.config
        st = self.state()
        m = self.metrics()
        last = m[-1] if m else {}
        res = self.results()
        return {
            "id": self.id,
            "name": cfg.name,
            "project": cfg.project,
            "dataset": cfg.dataset,
            "task": cfg.task,
            "model": f"{cfg.model.family}{cfg.model.size}",
            "state": st.get("state"),
            "epoch": last.get("epoch"),
            "epochs": st.get("epochs") or cfg.train_args().get("epochs"),
            "fitness": max((r.get("fitness", 0) or 0 for r in m), default=None),
            "main_metric": res.get("main_metric"),
            "main_value": res.get("main_value"),
            "updated": st.get("updated"),
            "notes": cfg.notes,
        }

    def tail_log(self, n: int = 200) -> str:
        if not self.log_path.is_file():
            return ""
        with self.log_path.open("rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 200_000))
            text = fh.read().decode("utf-8", errors="replace")
        # progress bars rewrite one line with carriage returns: keep the last state of each line
        text = _ANSI.sub("", text)
        lines = [ln.split("\r")[-1] for ln in text.replace("\r\n", "\n").split("\n")]
        return "\n".join(lines[-n:])
