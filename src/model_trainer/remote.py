"""Train on another machine over SSH (experimental), and benchmark models on a Raspberry Pi.

A remote is set up once in Settings: host, user, port, the folder to work in, and the Python that
has model-trainer installed (`pip install model-trainer` or a clone + `pip install -e .`). The local
OpenSSH client (`ssh`, `scp`, built into Windows 10+) must be able to log in without a password
(SSH key).

How a remote run works:
1. the run dataset is prepared locally (so the remote needs nothing but the run folder);
2. the run folder is copied to <workdir>/projects/<project>/runs/<run>;
3. the worker starts there with nohup; pause/stop write command.json remotely;
4. `sync()` pulls state, metrics, events and (at the end) results and the model package back.
"""

from __future__ import annotations

import logging
import shlex
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .runs.store import Run
from .util import write_json
from .workspace import Workspace

log = logging.getLogger("model_trainer.remote")

Runner = Callable[[Sequence[str]], subprocess.CompletedProcess]
SYNC_FILES = (
    "state.json",
    "heartbeat.json",
    "metrics.jsonl",
    "events.jsonl",
    "progress.json",
    "log.txt",
    "env.json",
)


def _run(cmd: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(cmd),
        capture_output=True,
        text=True,
        timeout=3600,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


class Remote:
    def __init__(self, spec: Mapping[str, Any], runner: Runner | None = None) -> None:
        self.name = spec["name"]
        self.host = spec["host"]
        self.user = spec.get("user") or ""
        self.port = int(spec.get("port") or 22)
        self.workdir = spec.get("workdir") or "~/model-trainer-workspace"
        self.python = spec.get("python") or "python3"
        self.runner = runner or _run

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}" if self.user else self.host

    def ssh(self, command: str) -> subprocess.CompletedProcess:
        return self.runner(
            ["ssh", "-p", str(self.port), "-o", "BatchMode=yes", self.target, command]
        )

    def scp_to(
        self, local: Path, remote: str, recursive: bool = False
    ) -> subprocess.CompletedProcess:
        cmd = (
            ["scp", "-P", str(self.port), "-o", "BatchMode=yes"]
            + (["-r"] if recursive else [])
            + [str(local), f"{self.target}:{remote}"]
        )
        return self.runner(cmd)

    def scp_from(
        self, remote: str, local: Path, recursive: bool = False
    ) -> subprocess.CompletedProcess:
        cmd = (
            ["scp", "-P", str(self.port), "-o", "BatchMode=yes"]
            + (["-r"] if recursive else [])
            + [f"{self.target}:{remote}", str(local)]
        )
        return self.runner(cmd)

    def check(self) -> tuple[bool, str]:
        r = self.ssh(
            f'{self.python} -c "import model_trainer, ultralytics, torch; print(torch.cuda.is_available())"'
        )
        if r.returncode != 0:
            return False, (r.stderr or r.stdout).strip()[:300]
        return (
            True,
            "connected; CUDA available" if "True" in r.stdout else "connected; no GPU (CPU only)",
        )

    def run_dir(self, run: Run) -> str:
        return f"{self.workdir}/projects/{run.config.project}/runs/{run.id}"


def get_remote(ws: Workspace, name: str, runner: Runner | None = None) -> Remote:
    for spec in ws.settings().get("remotes") or []:
        if spec.get("name") == name:
            return Remote(spec, runner)
    raise KeyError(f"no remote called {name!r} in Settings")


def launch_remote(ws: Workspace, run: Run, mode: str, runner: Runner | None = None) -> None:
    from .runs.worker import Worker

    remote = get_remote(ws, run.config.options.executor.split(":", 1)[1], runner)
    Worker(run).prepare_data()
    rdir = remote.run_dir(run)
    parent = rdir.rsplit("/", 1)[0]
    r = remote.ssh(f"mkdir -p {shlex.quote(parent)}")
    if r.returncode != 0:
        raise RuntimeError(f"cannot reach {remote.name}: {r.stderr.strip()[:300]}")
    r = remote.scp_to(run.path, parent + "/", recursive=True)
    if r.returncode != 0:
        raise RuntimeError(f"copying the run to {remote.name} failed: {r.stderr.strip()[:300]}")
    cmd = (
        f"cd {shlex.quote(rdir)} && nohup {remote.python} -m model_trainer.runs.worker {shlex.quote(rdir)} {mode} "
        f">> log.txt 2>&1 < /dev/null &"
    )
    r = remote.ssh(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"starting the worker on {remote.name} failed: {r.stderr.strip()[:300]}")
    run.set_state("running", remote=remote.name, mode=mode)


def send_command(ws: Workspace, run: Run, command: str, runner: Runner | None = None) -> None:
    remote = get_remote(ws, run.config.options.executor.split(":", 1)[1], runner)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "command.json"
        write_json(p, {"command": command})
        remote.scp_to(p, remote.run_dir(run) + "/command.json")


def sync(ws: Workspace, run: Run, final: bool = False, runner: Runner | None = None) -> None:
    """Pull the run's progress (and at the end its results and package) from the remote."""
    remote = get_remote(ws, run.config.options.executor.split(":", 1)[1], runner)
    rdir = remote.run_dir(run)
    for f in SYNC_FILES:
        remote.scp_from(f"{rdir}/{f}", run.path / f)
    if final:
        for d in ("results", "package", "watch", "checkpoints", "ultra"):
            remote.scp_from(f"{rdir}/{d}", run.path, recursive=True)


BENCH_SCRIPT = """
import sys, time
from ultralytics import YOLO
import numpy as np
m = YOLO(sys.argv[1], task=sys.argv[3])
img = (np.random.rand(int(sys.argv[2]), int(sys.argv[2]), 3) * 255).astype('uint8')
for _ in range(3): m.predict(img, imgsz=int(sys.argv[2]), verbose=False)
t = []
for _ in range(int(sys.argv[4])):
    s = time.perf_counter(); m.predict(img, imgsz=int(sys.argv[2]), verbose=False); t.append((time.perf_counter() - s) * 1000)
t.sort(); print('MS', t[len(t) // 2])
"""


def benchmark_remote(
    ws: Workspace,
    remote_name: str,
    model_path: Path,
    imgsz: int,
    task: str,
    n: int = 20,
    runner: Runner | None = None,
) -> float:
    """Median ms per image of an exported model on a remote device (e.g. the Raspberry Pi)."""
    remote = get_remote(ws, remote_name, runner)
    rtmp = f"{remote.workdir}/bench"
    remote.ssh(f"mkdir -p {shlex.quote(rtmp)}")
    remote.scp_to(model_path, rtmp + "/", recursive=model_path.is_dir())
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "mt_bench.py"
        script.write_text(BENCH_SCRIPT, encoding="utf-8")
        remote.scp_to(script, rtmp + "/mt_bench.py")
    r = remote.ssh(
        f"cd {shlex.quote(rtmp)} && {remote.python} mt_bench.py {shlex.quote(model_path.name)} {imgsz} {task} {n}"
    )
    for line in r.stdout.splitlines():
        if line.startswith("MS "):
            return float(line.split()[1])
    raise RuntimeError(
        f"benchmark on {remote.name} failed: {(r.stderr or r.stdout).strip()[-400:]}"
    )
