"""Where everything lives on disk.

<workspace>/                        MTRAIN_HOME, --workspace, or ~/model-trainer-workspace
  settings.json                     notifications, remote machines, last project
  queue.json                        runs waiting to start (one at a time per machine)
  cache/thumbs/                     gallery thumbnails
  projects/<project>/
    datasets/<dataset>/dataset.json imported datasets (a manifest pointing at the files)
    runs/<run>/                     training runs (see runs/store.py)
    templates/<name>.yaml           saved wizard settings
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .util import read_json, write_json

DEFAULT_SETTINGS: dict[str, Any] = {
    "project": "default",
    "notify": {
        "enabled": False,
        "server": "https://ntfy.sh",
        "topic": "",  # a private, hard-to-guess name; subscribe to it in the ntfy phone app
        "events": ["finished", "failed", "warning", "paused"],
    },
    "remotes": [],  # [{name, host, user, workdir, python}] machines reachable over SSH
    "gpu_tflops": None,  # measured training throughput, refines the time estimates
}

_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_name(name: str) -> str:
    """File-system safe name (letters, digits, _ . -)."""
    s = _NAME.sub("_", name.strip()).strip("._")
    return s or "unnamed"


def default_root() -> Path:
    env = os.environ.get("MTRAIN_HOME")
    return Path(env).expanduser() if env else Path.home() / "model-trainer-workspace"


@dataclass
class Project:
    root: Path

    @property
    def name(self) -> str:
        return self.root.name

    @property
    def datasets_dir(self) -> Path:
        return self.root / "datasets"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def templates_dir(self) -> Path:
        return self.root / "templates"

    def ensure(self) -> Project:
        for d in (self.datasets_dir, self.runs_dir, self.templates_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self

    def datasets(self) -> list[str]:
        if not self.datasets_dir.is_dir():
            return []
        return sorted(p.name for p in self.datasets_dir.iterdir() if (p / "dataset.json").is_file())

    def runs(self) -> list[Path]:
        if not self.runs_dir.is_dir():
            return []
        runs = [p for p in self.runs_dir.iterdir() if (p / "run.yaml").is_file()]
        return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)

    def templates(self) -> list[str]:
        if not self.templates_dir.is_dir():
            return []
        return sorted(p.stem for p in self.templates_dir.glob("*.yaml"))


class Workspace:
    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        self.root = Path(root).expanduser() if root else default_root()
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def settings_path(self) -> Path:
        return self.root / "settings.json"

    @property
    def queue_path(self) -> Path:
        return self.root / "queue.json"

    @property
    def cache_dir(self) -> Path:
        d = self.root / "cache"
        d.mkdir(exist_ok=True)
        return d

    def settings(self) -> dict[str, Any]:
        s = read_json(self.settings_path, {})
        out = {**DEFAULT_SETTINGS, **s}
        out["notify"] = {**DEFAULT_SETTINGS["notify"], **(s.get("notify") or {})}
        return out

    def save_settings(self, settings: dict[str, Any]) -> None:
        write_json(self.settings_path, settings)

    def projects(self) -> list[str]:
        d = self.root / "projects"
        if not d.is_dir():
            return []
        return sorted(p.name for p in d.iterdir() if p.is_dir())

    def project(self, name: str | None = None) -> Project:
        name = safe_name(name or self.settings().get("project") or "default")
        return Project(self.root / "projects" / name).ensure()

    def find_run(self, run_id: str) -> Path | None:
        """A run folder by its id (folder name), in any project."""
        for p in (self.root / "projects").glob(f"*/runs/{run_id}"):
            if (p / "run.yaml").is_file():
                return p
        return None

    def all_runs(self) -> list[Path]:
        runs = [p for p in (self.root / "projects").glob("*/runs/*") if (p / "run.yaml").is_file()]
        return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)
