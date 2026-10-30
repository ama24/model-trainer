"""Compare runs: what was different (settings diff) and what it did (curves, final scores)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .runs.store import Run


def flat_settings(run: Run) -> dict[str, Any]:
    cfg = run.config
    out: dict[str, Any] = {
        "dataset": cfg.dataset,
        "task": cfg.task,
        "model": f"{cfg.model.family}{cfg.model.size}",
        "start": cfg.model.start + (f" ({cfg.model.source})" if cfg.model.source else ""),
        "augmentation preset": cfg.aug_preset,
        "subset": cfg.subset,
        "balance": cfg.balance,
        "split": cfg.split.get("mode"),
        "dropped classes": ", ".join(cfg.classes.get("drop") or []) or "-",
        "merged classes": ", ".join(
            f"{k}->{v}" for k, v in (cfg.classes.get("merge") or {}).items()
        )
        or "-",
    }
    out.update(cfg.train_args())
    return out


def settings_diff(runs: Sequence[Run]) -> list[dict[str, Any]]:
    """Rows of settings that differ between the runs: {setting, <run id>: value, ...}."""
    flats = [flat_settings(r) for r in runs]
    keys = sorted({k for f in flats for k in f})
    rows = []
    for k in keys:
        vals = [f.get(k) for f in flats]
        if len({repr(v) for v in vals}) > 1:
            rows.append({"setting": k, **{r.id: v for r, v in zip(runs, vals)}})
    return rows


def results_table(runs: Sequence[Run]) -> list[dict[str, Any]]:
    rows = []
    for r in runs:
        s = r.summary()
        res = r.results()
        m = r.metrics()
        rows.append(
            {
                "run": r.id,
                "name": s["name"],
                "model": s["model"],
                "state": s["state"],
                "epochs": m[-1]["epoch"] if m else 0,
                "best epoch": res.get("best_epoch"),
                "main metric": res.get("main_metric"),
                "score": res.get("main_value"),
                "best fitness": max((x.get("fitness") or 0 for x in m), default=None),
                "minutes": round(sum(x.get("epoch_s", 0) for x in m) / 60, 1),
            }
        )
    return rows


def curves(runs: Sequence[Run], metric: str) -> dict[str, list[tuple[int, float]]]:
    out = {}
    for r in runs:
        out[r.id] = [
            (x["epoch"], x[metric]) for x in r.metrics() if metric in x and x[metric] is not None
        ]
    return out


def common_metrics(runs: Sequence[Run]) -> list[str]:
    keys: set[str] | None = None
    for r in runs:
        m = r.metrics()
        k = set().union(*(x.keys() for x in m)) if m else set()
        keys = k if keys is None else keys & k
    skip = {"epoch", "time", "batch", "epoch_s"}
    return sorted(k for k in (keys or set()) if k not in skip)


def best(runs: Sequence[Run], key: str = "score") -> Mapping[str, Any] | None:
    rows = [r for r in results_table(runs) if r.get(key) is not None]
    return max(rows, key=lambda r: r[key]) if rows else None
