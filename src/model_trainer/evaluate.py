"""Evaluate a trained model and package it.

- Ultralytics validation on the run's validation split: mean and per-class scores, confusion matrix,
  precision-recall / F1 curves (-> recommended confidence threshold).
- Our own error analysis: every validation image is predicted and matched to its labels, sorting the
  mistakes into missed objects, false detections, wrong class and poor localisation, plus recall by
  object size. The GUI's error gallery reads results/errors.json.
- The model package (package/): best.pt, exports and model_card.json, which other projects read.
"""

from __future__ import annotations

import contextlib
import logging
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .datasets.formats import parse_yolo_line
from .datasets.health import LARGE_PX, SMALL_PX, iou
from .runs.store import Run
from .util import atomic_copy, now, read_json, read_yaml, sha1_file, write_json

log = logging.getLogger("model_trainer.evaluate")

MAIN_METRIC = {
    "detect": ("metrics/mAP50-95(B)", "mAP50-95"),
    "obb": ("metrics/mAP50-95(B)", "mAP50-95"),
    "segment": ("metrics/mAP50-95(M)", "mask mAP50-95"),
    "pose": ("metrics/mAP50-95(P)", "pose mAP50-95"),
    "classify": ("metrics/accuracy_top1", "top-1 accuracy"),
}


def _data_arg(run: Run) -> str:
    return str(run.data_dir) if run.config.task == "classify" else str(run.data_dir / "data.yaml")


def validate(
    weights: Path,
    data: str,
    imgsz: int,
    out_dir: Path,
    split: str = "val",
    device: str | None = None,
) -> Any:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    return model.val(
        data=data,
        split=split,
        imgsz=imgsz,
        plots=True,
        project=str(out_dir.parent),
        name=out_dir.name,
        exist_ok=True,
        device=device or None,
        verbose=False,
        batch=8,
    )


def metrics_dict(m: Any, task: str) -> dict[str, Any]:
    """Plain numbers from an Ultralytics metrics object."""
    out: dict[str, Any] = {"mean": {k: float(v) for k, v in m.results_dict.items()}}
    key, label = MAIN_METRIC[task]
    out["main_metric"], out["main_value"] = label, float(m.results_dict.get(key, 0.0))
    with contextlib.suppress(Exception):
        out["per_class"] = [
            {
                k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                for k, v in row.items()
            }
            for row in m.summary()
        ]
    with contextlib.suppress(Exception):
        cm = m.confusion_matrix
        out["confusion"] = {
            "matrix": cm.matrix.astype(float).tolist(),
            "names": list(m.names.values()) if hasattr(m, "names") else None,
        }
    if task != "classify":
        with contextlib.suppress(Exception):
            curves = {}
            for name, (x, y, xl, yl) in zip(m.curves, m.curves_results):
                y = np.asarray(y)
                ym = y.mean(0) if y.ndim == 2 else y
                idx = np.linspace(0, len(x) - 1, 101).astype(int)
                curves[name] = {
                    "x": np.asarray(x)[idx].round(4).tolist(),
                    "y": ym[idx].round(4).tolist(),
                    "per_class": (y[:, idx].round(4).tolist() if y.ndim == 2 else None),
                    "xlabel": xl,
                    "ylabel": yl,
                }
            out["curves"] = curves
            f1 = next((v for k, v in curves.items() if k.startswith("F1")), None)
            if f1:
                i = int(np.argmax(f1["y"]))
                out["recommended_conf"] = round(float(f1["x"][i]), 3)
                out["best_f1"] = round(float(f1["y"][i]), 4)
    with contextlib.suppress(Exception):
        out["speed_ms"] = {k: float(v) for k, v in m.speed.items()}
    return out


def plain_summary(res: dict[str, Any], task: str) -> str:
    mean = res.get("mean", {})
    if task == "classify":
        return f"The model names the right class for {mean.get('metrics/accuracy_top1', 0):.0%} of the validation images (top-5: {mean.get('metrics/accuracy_top5', 0):.0%})."
    suffix = {"segment": "(M)", "pose": "(P)"}.get(task, "(B)")
    p = mean.get(f"metrics/precision{suffix}", mean.get("metrics/precision(B)", 0))
    r = mean.get(f"metrics/recall{suffix}", mean.get("metrics/recall(B)", 0))
    text = (
        f"Of every 100 labelled objects in the validation images the model finds about {r * 100:.0f}; "
        f"of every 100 detections it reports, about {p * 100:.0f} are right. "
        f"{res['main_metric']} = {res['main_value']:.3f} (1.0 would be perfect)."
    )
    if res.get("recommended_conf"):
        text += f" Best confidence threshold: {res['recommended_conf']:.2f}."
    return text


# ------------------------------------------------------------------------- error analysis


def _read_labels(
    label_file: Path, kpt_shape: Sequence[int] | None
) -> list[tuple[int, tuple[float, float, float, float]]]:
    out = []
    if not label_file.is_file():
        return out
    obb = False
    for line in label_file.read_text(encoding="utf-8").splitlines():
        o, _ = parse_yolo_line(line, kpt_shape, force_obb=obb or len(line.split()) == 9)
        if o is not None and o.box:
            out.append((o.cls, o.box))
    return out


def _pred_boxes(r: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if getattr(r, "obb", None) is not None and r.obb is not None:
        b = r.obb
        return (
            b.xywhr.cpu().numpy()[:, :4] / np.array([*r.orig_shape[::-1]] * 2),
            b.cls.cpu().numpy().astype(int),
            b.conf.cpu().numpy(),
        )
    b = r.boxes
    if b is None:
        return np.zeros((0, 4)), np.zeros(0, int), np.zeros(0)
    return b.xywhn.cpu().numpy(), b.cls.cpu().numpy().astype(int), b.conf.cpu().numpy()


def match_image(
    gt: Sequence[tuple[int, Sequence[float]]],
    pred: Sequence[tuple[int, Sequence[float], float]],
    iou_ok: float = 0.5,
    iou_loc: float = 0.1,
) -> dict[str, list[dict[str, Any]]]:
    """Greedy matching by confidence. Returns lists of hits and the four error types."""
    out: dict[str, list[dict[str, Any]]] = {
        "hit": [],
        "missed": [],
        "false": [],
        "wrong_class": [],
        "localisation": [],
    }
    used = set()
    for pc, pb, conf in sorted(pred, key=lambda p: -p[2]):
        best, best_iou, best_any, best_any_iou = None, 0.0, None, 0.0
        for j, (gc, gb) in enumerate(gt):
            if j in used:
                continue
            v = iou(pb, gb)
            if gc == pc and v > best_iou:
                best, best_iou = j, v
            if v > best_any_iou:
                best_any, best_any_iou = j, v
        rec = {"cls": int(pc), "box": [float(x) for x in pb], "conf": float(conf)}
        if best is not None and best_iou >= iou_ok:
            used.add(best)
            out["hit"].append({**rec, "iou": best_iou})
        elif best_any is not None and best_any_iou >= iou_ok:
            used.add(best_any)
            out["wrong_class"].append(
                {**rec, "true_cls": int(gt[best_any][0]), "iou": best_any_iou}
            )
        elif best is not None and best_iou >= iou_loc:
            used.add(best)
            out["localisation"].append({**rec, "iou": best_iou})
        else:
            out["false"].append(rec)
    for j, (gc, gb) in enumerate(gt):
        if j not in used:
            out["missed"].append({"cls": int(gc), "box": [float(x) for x in gb]})
    return out


def error_analysis(
    weights: Path,
    data_dir: Path,
    imgsz: int,
    conf: float,
    device: str | None = None,
    limit: int = 2000,
) -> dict[str, Any]:
    from ultralytics import YOLO

    data = read_yaml(data_dir / "data.yaml")
    kpt = data.get("kpt_shape")
    images = sorted((data_dir / "images" / "val").glob("*"))[:limit]
    model = YOLO(str(weights))
    per_image = []
    totals = Counter()
    size_gt = Counter()
    size_hit = Counter()
    for img in images:
        gt = _read_labels(data_dir / "labels" / "val" / (img.stem + ".txt"), kpt)
        r = model.predict(str(img), imgsz=imgsz, conf=conf, verbose=False, device=device or None)[0]
        boxes, cls, confs = _pred_boxes(r)
        m = match_image(gt, [(int(c), b, float(s)) for b, c, s in zip(boxes, cls, confs)])
        h, w = r.orig_shape
        scale = imgsz / max(w, h)
        for _, gb in gt:
            s = min(gb[2] * w, gb[3] * h) * scale
            size_gt[_bucket(s)] += 1
        for hit in m["hit"]:
            s = min(hit["box"][2] * w, hit["box"][3] * h) * scale
            size_hit[_bucket(s)] += 1
        counts = {k: len(v) for k, v in m.items()}
        totals.update(counts)
        errors = sum(counts[k] for k in ("missed", "false", "wrong_class", "localisation"))
        per_image.append(
            {
                "image": str(img),
                "errors": errors,
                **counts,
                "detail": {k: v for k, v in m.items() if k != "hit"},
            }
        )
    per_image.sort(key=lambda r: -r["errors"])
    return {
        "conf": conf,
        "images": len(images),
        "totals": dict(totals),
        "recall_by_size": {
            b: {
                "objects": size_gt[b],
                "found": size_hit[b],
                "recall": size_hit[b] / size_gt[b] if size_gt[b] else None,
            }
            for b in ("small", "medium", "large")
        },
        "per_image": per_image,
    }


def _bucket(s: float) -> str:
    return "small" if s < SMALL_PX else "large" if s > LARGE_PX else "medium"


# ------------------------------------------------------------------------------ run level


def evaluate_run(run: Run, device: str | None = None) -> dict[str, Any]:
    cfg = run.config
    weights = run.weights_dir / "best.pt"
    if not weights.is_file():
        weights = run.weights_dir / "last.pt"
    if not weights.is_file():
        raise FileNotFoundError("no trained weights")
    imgsz = int(cfg.train_args().get("imgsz", 640))
    m = validate(
        weights, _data_arg(run), imgsz, run.results_dir / "val", device=device or cfg.options.device
    )
    res = metrics_dict(m, cfg.task)
    res["summary"] = plain_summary(res, cfg.task)
    res["weights"] = str(weights)
    res["created"] = now()
    res["task"] = cfg.task
    hist = run.metrics()
    if hist:
        best = max(hist, key=lambda r: r.get("fitness") or 0)
        res["best_epoch"] = best["epoch"]
        res["epochs_trained"] = hist[-1]["epoch"]
    if cfg.task != "classify":
        with contextlib.suppress(Exception):
            errs = error_analysis(
                weights,
                run.data_dir,
                imgsz,
                res.get("recommended_conf") or 0.25,
                cfg.options.device,
            )
            write_json(run.results_dir / "errors.json", errs)
            res["errors"] = {"totals": errs["totals"], "recall_by_size": errs["recall_by_size"]}
    write_json(run.results_dir / "results.json", res)
    return res


def evaluate_external(
    run: Run, dataset_yaml: Path, out_name: str, device: str | None = None
) -> dict[str, Any]:
    """Score the run's model on another labelled dataset (same class names) - the 'final check'."""
    cfg = run.config
    weights = run.best_weights
    if weights is None:
        raise FileNotFoundError("no trained weights")
    imgsz = int(cfg.train_args().get("imgsz", 640))
    out = run.results_dir / "external" / out_name
    m = validate(weights, str(dataset_yaml), imgsz, out, device=device)
    res = metrics_dict(m, cfg.task)
    res["summary"] = plain_summary(res, cfg.task)
    res["dataset"] = str(dataset_yaml)
    res["created"] = now()
    write_json(out / "results.json", res)
    return res


def package(run: Run, res: dict[str, Any] | None = None) -> Path:
    """package/: best.pt + model_card.json (exports are added by export.py)."""
    cfg = run.config
    res = res or run.results()
    src = run.weights_dir / "best.pt"
    if not src.is_file():
        src = run.weights_dir / "last.pt"
    d = run.package_dir
    atomic_copy(src, d / "best.pt")
    manifest = read_json(run.data_dir / "manifest.json", {})
    card = read_json(d / "model_card.json", {})
    card.update(
        {
            "format": "model-trainer/1",
            "name": cfg.name,
            "run": run.id,
            "project": cfg.project,
            "task": cfg.task,
            "classes": [
                v
                for _, v in sorted(
                    (manifest.get("classes") or {}).items(), key=lambda kv: int(kv[0])
                )
            ],
            "imgsz": int(cfg.train_args().get("imgsz", 640)),
            "base_model": f"{cfg.model.family}{cfg.model.size} ({cfg.model.start})",
            "metrics": {
                "main_metric": res.get("main_metric"),
                "main_value": res.get("main_value"),
                **res.get("mean", {}),
            },
            "recommended_conf": res.get("recommended_conf"),
            "dataset": {
                "name": cfg.dataset,
                "fingerprint": manifest.get("dataset_fingerprint"),
                "images": manifest.get("images"),
                "class_mapping": manifest.get("mapping"),
            },
            "training": {
                "args": cfg.train_args(),
                "augmentation": cfg.aug_preset,
                "epochs_trained": res.get("epochs_trained"),
                "best_epoch": res.get("best_epoch"),
            },
            "weights": {"best.pt": {"sha1": sha1_file(d / "best.pt")}},
            "environment": read_json(run.path / "env.json", {}),
            "created": now(),
        }
    )
    data_yaml = run.data_dir / "data.yaml"
    data = (read_yaml(data_yaml) or {}) if data_yaml.is_file() else {}
    if data.get("kpt_shape"):
        card["kpt_shape"] = data["kpt_shape"]
    write_json(d / "model_card.json", card)
    return d
