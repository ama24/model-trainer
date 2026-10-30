"""Tiny synthetic datasets (coloured shapes) in every supported format."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

NAMES = {0: "square", 1: "circle", 2: "triangle"}
COLOURS = [(220, 60, 60), (60, 200, 80), (70, 90, 230)]


def _shapes(rng: random.Random, w: int, h: int, n: int) -> list[tuple[int, list[float]]]:
    out = []
    for _ in range(n):
        cls = rng.choice([0, 0, 0, 1, 1, 2])  # imbalanced on purpose
        s = rng.uniform(0.12, 0.3) * min(w, h)
        cx, cy = rng.uniform(s, w - s), rng.uniform(s, h - s)
        if cls == 0:
            poly = [
                cx - s / 2,
                cy - s / 2,
                cx + s / 2,
                cy - s / 2,
                cx + s / 2,
                cy + s / 2,
                cx - s / 2,
                cy + s / 2,
            ]
        elif cls == 1:
            import math

            poly = []
            for k in range(16):
                a = 2 * math.pi * k / 16
                poly += [cx + s / 2 * math.cos(a), cy + s / 2 * math.sin(a)]
        else:
            poly = [cx, cy - s / 2, cx + s / 2, cy + s / 2, cx - s / 2, cy + s / 2]
        out.append((cls, poly))
    return out


def make_yolo(
    root: Path,
    n_train: int = 24,
    n_val: int = 8,
    seed: int = 0,
    w: int = 160,
    h: int = 120,
    groups: bool = True,
    kind: str = "seg",
) -> Path:
    rng = random.Random(seed)
    rows = ["image,split,group"]
    for split, n in (("train", n_train), ("val", n_val)):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img = Image.new("RGB", (w, h), (rng.randint(80, 160),) * 3)
            d = ImageDraw.Draw(img)
            lines = []
            shapes = [] if i == 0 else _shapes(rng, w, h, rng.randint(1, 3))  # image 0: background
            for cls, poly in shapes:
                d.polygon(
                    [(poly[k], poly[k + 1]) for k in range(0, len(poly), 2)], fill=COLOURS[cls]
                )
                xs, ys = poly[0::2], poly[1::2]
                if kind == "seg":
                    lines.append(
                        f"{cls} "
                        + " ".join(
                            f"{v / (w if k % 2 == 0 else h):.5f}" for k, v in enumerate(poly)
                        )
                    )
                else:
                    cx, cy = (min(xs) + max(xs)) / 2 / w, (min(ys) + max(ys)) / 2 / h
                    bw, bh = (max(xs) - min(xs)) / w, (max(ys) - min(ys)) / h
                    lines.append(f"{cls} {cx:.5f} {cy:.5f} {bw:.5f} {bh:.5f}")
            name = f"{split}_{i:03d}"
            img.save(root / "images" / split / f"{name}.jpg", quality=95)
            (root / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + "\n")
            rows.append(f"{name}.jpg,{split},g{i // 2}_{split}")
    (root / "data.yaml").write_text(
        f"path: {root.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {k}: {v}\n" for k, v in NAMES.items())
    )
    if groups:
        (root / "groups.csv").write_text("\n".join(rows) + "\n")
    return root


def make_coco(root: Path) -> Path:
    yolo = make_yolo(root / "_src", 6, 3, groups=False)
    for split in ("train", "val"):
        images, anns = [], []
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(sorted((yolo / "images" / split).glob("*.jpg"))):
            Image.open(p).save(root / "images" / split / p.name)
            images.append({"id": i + 1, "file_name": p.name, "width": 160, "height": 120})
            for line in (yolo / "labels" / split / (p.stem + ".txt")).read_text().split("\n"):
                if not line.strip():
                    continue
                v = line.split()
                poly = [float(x) * (160 if k % 2 == 0 else 120) for k, x in enumerate(v[1:])]
                xs, ys = poly[0::2], poly[1::2]
                anns.append(
                    {
                        "id": len(anns) + 1,
                        "image_id": i + 1,
                        "category_id": int(v[0]) + 1,
                        "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
                        "segmentation": [poly],
                        "area": 1,
                        "iscrowd": 0,
                    }
                )
        cats = [{"id": k + 1, "name": n} for k, n in NAMES.items()]
        (root / "annotations").mkdir(exist_ok=True)
        (root / "annotations" / f"{split}.json").write_text(
            json.dumps({"images": images, "annotations": anns, "categories": cats})
        )
    import shutil

    shutil.rmtree(yolo)
    return root


def make_folders(root: Path) -> Path:
    rng = random.Random(1)
    for split, n in (("train", 8), ("val", 3)):
        for cls, name in NAMES.items():
            d = root / split / name
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                img = Image.new("RGB", (64, 64), (rng.randint(0, 255),) * 3)
                ImageDraw.Draw(img).ellipse([10, 10, 54, 54], fill=COLOURS[cls])
                img.save(d / f"{name}_{i}.png")
    return root


@pytest.fixture
def yolo_seg(tmp_path) -> Path:
    return make_yolo(tmp_path / "shapes")


@pytest.fixture
def yolo_det(tmp_path) -> Path:
    return make_yolo(tmp_path / "shapes_det", kind="det")


@pytest.fixture
def coco_ds(tmp_path) -> Path:
    return make_coco(tmp_path / "coco")


@pytest.fixture
def folders_ds(tmp_path) -> Path:
    return make_folders(tmp_path / "cls")


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    from model_trainer.workspace import Workspace

    monkeypatch.setenv("MTRAIN_HOME", str(tmp_path / "ws"))
    return Workspace(tmp_path / "ws")
