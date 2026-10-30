"""Hyperparameter sweeps: many runs from one base run, varying a few settings, within a budget.

    space:
      lr0: [0.001, 0.01]                      grid: every combination
      imgsz: [640, 960]
      model.size: [n, s]                      dotted keys reach model / top-level settings
    or, for random search:
      lr0: {min: 0.0005, max: 0.02, log: true}
      mosaic: {min: 0.5, max: 1.0}

The runs are queued and train one after another. When the time budget is spent, the remaining
queued runs of the sweep are skipped.
"""

from __future__ import annotations

import copy
import itertools
import math
import random
import time
from collections.abc import Mapping
from typing import Any

from .runs.config import RunConfig
from .runs.manager import Manager
from .runs.store import Run
from .util import now, read_json, write_json
from .workspace import Project, Workspace, safe_name


def expand(
    space: Mapping[str, Any], mode: str = "grid", n: int = 10, seed: int = 0
) -> list[dict[str, Any]]:
    keys = list(space)
    if mode == "grid":
        lists = [v if isinstance(v, list) else [v] for v in space.values()]
        return (
            [dict(zip(keys, combo)) for combo in itertools.product(*lists)][:n]
            if n
            else [dict(zip(keys, combo)) for combo in itertools.product(*lists)]
        )
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        point = {}
        for k, v in space.items():
            if isinstance(v, list):
                point[k] = rng.choice(v)
            elif isinstance(v, Mapping):
                lo, hi = float(v["min"]), float(v["max"])
                if v.get("log"):
                    x = math.exp(rng.uniform(math.log(lo), math.log(hi)))
                else:
                    x = rng.uniform(lo, hi)
                point[k] = int(round(x)) if v.get("int") else round(x, 6)
            else:
                point[k] = v
        out.append(point)
    return out


def apply(cfg: RunConfig, point: Mapping[str, Any]) -> RunConfig:
    c = copy.deepcopy(cfg)
    for k, v in point.items():
        if k.startswith("model."):
            setattr(c.model, k.split(".", 1)[1], v)
        elif k in ("subset", "balance", "aug_preset"):
            setattr(c, k, v)
        else:
            c.train[k] = v
    return c


def label(point: Mapping[str, Any]) -> str:
    return "_".join(f"{k.split('.')[-1]}{v}" for k, v in point.items())


def create_sweep(
    ws: Workspace,
    project: Project,
    base: RunConfig,
    space: Mapping[str, Any],
    mode: str = "grid",
    max_runs: int = 12,
    budget_hours: float | None = None,
    seed: int = 0,
    name: str | None = None,
) -> dict[str, Any]:
    points = expand(space, mode, max_runs, seed)
    sid = safe_name(name or f"sweep-{time.strftime('%Y%m%d-%H%M%S')}")
    mgr = Manager(ws)
    runs = []
    for p in points:
        cfg = apply(base, p)
        cfg.name = f"{base.name}-{label(p)}"[:80]
        cfg.notes = f"sweep {sid}: {dict(p)}"
        run = mgr.create(project, cfg)
        run.update_state(sweep=sid, sweep_point=dict(p))
        runs.append(run)
    sweep = {
        "id": sid,
        "project": project.name,
        "space": dict(space),
        "mode": mode,
        "points": points,
        "runs": [r.id for r in runs],
        "budget_hours": budget_hours,
        "created": now(),
        "started": None,
    }
    write_json(ws.root / "sweeps" / f"{sid}.json", sweep)
    return sweep


def start_sweep(ws: Workspace, sid: str) -> list[str]:
    path = ws.root / "sweeps" / f"{sid}.json"
    sweep = read_json(path)
    sweep["started"] = time.time()
    write_json(path, sweep)
    mgr = Manager(ws)
    out = []
    for rid in sweep["runs"]:
        p = ws.find_run(rid)
        if p:
            out.append(mgr.start(Run(p)))
    return out


def over_budget(ws: Workspace, sid: str) -> bool:
    sweep = read_json(ws.root / "sweeps" / f"{sid}.json", {})
    b, t0 = sweep.get("budget_hours"), sweep.get("started")
    return bool(b and t0 and time.time() - t0 > float(b) * 3600)


def sweeps(ws: Workspace) -> list[dict[str, Any]]:
    d = ws.root / "sweeps"
    return [read_json(p) for p in sorted(d.glob("*.json"))] if d.is_dir() else []


def results(ws: Workspace, sid: str) -> list[dict[str, Any]]:
    sweep = read_json(ws.root / "sweeps" / f"{sid}.json")
    rows = []
    for rid, point in zip(sweep["runs"], sweep["points"]):
        p = ws.find_run(rid)
        if not p:
            continue
        r = Run(p)
        res = r.results()
        rows.append(
            {"run": rid, **point, "state": r.state().get("state"), "score": res.get("main_value")}
        )
    return sorted(rows, key=lambda r: -(r["score"] if r["score"] is not None else -1))
