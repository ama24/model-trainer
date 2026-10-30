"""'What the model sees': training samples after resizing and augmentation, drawn with their labels.

Uses Ultralytics' own training data pipeline, so the preview is exactly what training will do (mosaic,
flips, colour changes, ...). A small temporary copy of the dataset is prepared in the cache.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .datasets import prepare as prep
from .datasets.formats import Dataset, Obj
from .datasets.render import draw
from .util import sha1_text

log = logging.getLogger("model_trainer.augpreview")


def preview_data(
    ds: Dataset, task: str, cache: Path, classes: Mapping[str, Any] | None = None
) -> Path:
    """A small prepared dataset (about 60 training images) for previews, cached by settings."""
    key = sha1_text(f"{ds.source}|{task}|{sorted((classes or {}).items())}|{len(ds.items)}")[:16]
    out = cache / "preview" / key
    if (out / "manifest.json").is_file():
        return out
    n_train = sum(1 for it in ds.items if it.split == "train") or len(ds.items)
    split = (
        prep.SplitPlan(mode="keep")
        if any(it.split == "val" for it in ds.items)
        else prep.SplitPlan(mode="resplit")
    )
    prep.prepare(
        ds,
        out,
        task,
        prep.ClassPlan.from_dict(classes),
        split,
        subset=min(1.0, 60 / max(n_train, 1)),
        hash_files=False,
    )
    return out


def samples(
    data_dir: Path, task: str, args: Mapping[str, Any], n: int = 9, seed: int | None = None
) -> list[Image.Image]:
    """`n` augmented training samples with their labels drawn on."""
    from ultralytics.cfg import get_cfg

    rng = random.Random(seed)
    overrides = {k: v for k, v in args.items() if k in _AUG_AND_SIZE}
    overrides.update(task=task, mode="train")
    cfg = get_cfg(overrides=overrides)
    if task == "classify":
        return _classify_samples(data_dir, cfg, n, rng)
    from ultralytics.data import build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    data = check_det_dataset(str(data_dir / "data.yaml"))
    dataset = build_yolo_dataset(
        cfg,
        str(data_dir / "images" / "train"),
        batch=1,
        data=data,
        mode="train",
        rect=False,
        stride=32,
    )
    names = {int(k): v for k, v in data["names"].items()}
    out = []
    random.seed(rng.random())  # Ultralytics' augmentations use the global generators
    np.random.seed(rng.randrange(2**31))
    for _ in range(n):
        s = dataset[rng.randrange(len(dataset))]
        img = s["img"].numpy().transpose(1, 2, 0)
        pil = Image.fromarray(np.ascontiguousarray(img))
        objs = []
        for c, b in zip(s["cls"].numpy().reshape(-1), s["bboxes"].numpy()):
            if task == "obb":
                objs.append(Obj(int(c), box=None, obb=_xywhr_to_corners(b, pil.size)))
            else:
                objs.append(Obj(int(c), box=tuple(float(v) for v in b[:4])))
        if "keypoints" in s and len(s["keypoints"]):
            kp = s["keypoints"].numpy()
            for o, k in zip(objs, kp):
                o.keypoints = [
                    (float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else 2.0) for p in k
                ]
        if task == "segment" and "masks" in s:
            for o, poly in zip(objs, _mask_polygons(s["masks"].numpy(), len(objs))):
                o.polygon = poly
        out.append(draw(pil, objs, names, ("boxes", "masks", "keypoints", "labels")))
    return out


def _mask_polygons(masks: np.ndarray, n: int) -> list[list[float] | None]:
    """Outline of each object's training mask, normalised (the augmented masks the model learns).

    Ultralytics stores either one mask per object (N, h, w) or, with overlap_mask (the default), a
    single (1, h, w) image where object k has the value k + 1 (a 0/1 mask when there is one object).
    """
    import cv2

    h, w = masks.shape[-2:]
    out: list[list[float] | None] = []
    for k in range(n):
        m = masks[0] == k + 1 if masks.shape[0] == 1 else masks[k] > 0
        contours, _ = cv2.findContours(
            m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            out.append(None)
            continue
        c = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)
        out.append(((c + 0.5) / np.array([w, h])).flatten().tolist())
    return out


def _xywhr_to_corners(b: np.ndarray, size: tuple[int, int]) -> list[float]:
    import cv2

    w, h = size
    cx, cy, bw, bh, r = (float(v) for v in b[:5])
    pts = cv2.boxPoints(((cx * w, cy * h), (bw * w, bh * h), np.degrees(r)))
    return (pts / np.array([w, h])).flatten().tolist()


def _classify_samples(data_dir: Path, cfg: Any, n: int, rng: random.Random) -> list[Image.Image]:
    from ultralytics.data import ClassificationDataset

    ds = ClassificationDataset(str(data_dir / "train"), cfg, augment=True, prefix="")
    out = []
    for _ in range(n):
        s = ds[rng.randrange(len(ds))]
        img = (s["img"].numpy().transpose(1, 2, 0).clip(0, 1) * 255).astype(np.uint8)
        pil = Image.fromarray(img)
        from PIL import ImageDraw

        ImageDraw.Draw(pil).text((4, 4), ds.base.classes[int(s["cls"])], fill=(255, 255, 0))
        out.append(pil)
    return out


_AUG_AND_SIZE = {
    "imgsz",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "degrees",
    "translate",
    "scale",
    "shear",
    "perspective",
    "flipud",
    "fliplr",
    "mosaic",
    "mixup",
    "cutmix",
    "copy_paste",
    "copy_paste_mode",
    "erasing",
    "auto_augment",
    "bgr",
}
