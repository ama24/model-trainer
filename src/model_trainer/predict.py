"""Playground: run a model on your own images or videos."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

log = logging.getLogger("model_trainer.predict")

_MODELS: dict[str, Any] = {}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def load_model(weights: str | Path) -> Any:
    from ultralytics import YOLO

    key = str(weights)
    if key not in _MODELS:
        _MODELS[key] = YOLO(key)
    return _MODELS[key]


@dataclass
class Prediction:
    image: str
    plotted: Image.Image
    objects: list[dict[str, Any]] = field(default_factory=list)
    ms: float = 0.0
    top: list[tuple[str, float]] = field(default_factory=list)  # classification: best classes


def predict_images(
    weights: str | Path,
    images: Sequence[str | Path],
    conf: float = 0.25,
    iou: float = 0.7,
    imgsz: int | None = None,
    device: str | None = None,
) -> list[Prediction]:
    model = load_model(weights)
    out = []
    kw: dict[str, Any] = {"conf": conf, "iou": iou, "verbose": False}
    if imgsz:
        kw["imgsz"] = imgsz
    if device:
        kw["device"] = device
    for img in images:
        r = model.predict(str(img), **kw)[0]
        plotted = Image.fromarray(r.plot()[..., ::-1])
        p = Prediction(str(img), plotted, ms=float(sum(r.speed.values())))
        names = r.names
        if r.probs is not None:
            top5 = r.probs.top5
            p.top = [(names[int(i)], float(r.probs.data[int(i)])) for i in top5]
        if r.boxes is not None:
            for b, c, s in zip(
                r.boxes.xywhn.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()
            ):
                p.objects.append(
                    {"class": names[int(c)], "conf": float(s), "box": [float(v) for v in b]}
                )
        if getattr(r, "obb", None) is not None:
            for c, s in zip(r.obb.cls.cpu().numpy(), r.obb.conf.cpu().numpy()):
                p.objects.append({"class": names[int(c)], "conf": float(s)})
        out.append(p)
    return out


def predict_video(
    weights: str | Path,
    video: str | Path,
    out_path: str | Path,
    conf: float = 0.25,
    iou: float = 0.7,
    imgsz: int | None = None,
    max_frames: int = 3000,
) -> dict[str, Any]:
    import cv2

    model = load_model(weights)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n, counts = 0, 0
    try:
        while n < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            r = model.predict(
                frame, conf=conf, iou=iou, verbose=False, **({"imgsz": imgsz} if imgsz else {})
            )[0]
            writer.write(r.plot())
            counts += 0 if r.boxes is None else len(r.boxes)
            n += 1
    finally:
        cap.release()
        writer.release()
    return {"frames": n, "detections": counts, "output": str(out_path)}


def save_as_dataset(preds: Sequence[Prediction], weights: str | Path, out: Path) -> Path:
    """Write images + predicted labels as a YOLO detection dataset, to correct and import (active learning)."""
    import shutil

    from .util import write_yaml

    model = load_model(weights)
    names = dict(model.names)
    (out / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "train").mkdir(parents=True, exist_ok=True)
    inv = {v: k for k, v in names.items()}
    for p in preds:
        src = Path(p.image)
        shutil.copy2(src, out / "images" / "train" / src.name)
        lines = [
            f"{inv[o['class']]} " + " ".join(f"{v:.6f}" for v in o["box"])
            for o in p.objects
            if "box" in o
        ]
        (out / "labels" / "train" / f"{src.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
    write_yaml(
        out / "data.yaml",
        {"path": str(out), "train": "images/train", "val": "images/train", "names": names},
    )
    return out
