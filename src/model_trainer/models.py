"""Base models: families x sizes, published numbers, and rough speed/memory estimates.

The numbers are Ultralytics' published COCO detection results at 640 px (approximate; segmentation,
pose and OBB variants are a little larger). They are for choosing, not for promises.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SIZES = ("n", "s", "m", "l", "x")
SIZE_LABELS = {"n": "nano", "s": "small", "m": "medium", "l": "large", "x": "extra large"}
TASK_SUFFIX = {"detect": "", "segment": "-seg", "pose": "-pose", "obb": "-obb", "classify": "-cls"}


@dataclass(frozen=True)
class Family:
    key: str
    label: str
    prefix: str
    tasks: tuple[str, ...]
    sizes: tuple[str, ...]
    note: str
    # size -> (params M, GFLOPs, COCO mAP50-95)
    numbers: dict[str, tuple[float, float, float]]


FAMILIES: dict[str, Family] = {
    f.key: f
    for f in (
        Family(
            "yolo26",
            "YOLO26 (newest)",
            "yolo26",
            ("detect", "segment", "pose", "obb", "classify"),
            SIZES,
            "Newest Ultralytics family: no NMS step, faster on CPUs and edge devices. Recommended default.",
            {
                "n": (2.4, 5.4, 40.9),
                "s": (9.5, 20.7, 48.6),
                "m": (20.4, 68.2, 53.1),
                "l": (24.8, 86.4, 55.0),
                "x": (55.7, 193.9, 57.5),
            },
        ),
        Family(
            "yolo11",
            "YOLO11",
            "yolo11",
            ("detect", "segment", "pose", "obb", "classify"),
            SIZES,
            "Widely used, very well tested. A safe choice when you need compatibility with other tools.",
            {
                "n": (2.6, 6.5, 39.5),
                "s": (9.4, 21.5, 47.0),
                "m": (20.1, 68.0, 51.5),
                "l": (25.3, 86.9, 53.4),
                "x": (56.9, 194.9, 54.7),
            },
        ),
        Family(
            "yolov8",
            "YOLOv8",
            "yolov8",
            ("detect", "segment", "pose", "obb", "classify"),
            SIZES,
            "Older but extremely common; most tutorials and deployment tools support it.",
            {
                "n": (3.2, 8.7, 37.3),
                "s": (11.2, 28.6, 44.9),
                "m": (25.9, 78.9, 50.2),
                "l": (43.7, 165.2, 52.9),
                "x": (68.2, 257.8, 53.9),
            },
        ),
        Family(
            "rtdetr",
            "RT-DETR (transformer)",
            "rtdetr",
            ("detect",),
            ("l", "x"),
            "Transformer detector: strong on crowded scenes, but heavier and slower to train. Detection only.",
            {"l": (32.0, 110.0, 53.0), "x": (67.0, 234.0, 54.8)},
        ),
    )
}


def weights_name(family: str, size: str, task: str) -> str:
    f = FAMILIES[family]
    if task not in f.tasks:
        raise ValueError(f"{f.label} does not support {task}")
    if family == "rtdetr":
        return f"rtdetr-{size}.pt"
    return f"{f.prefix}{size}{TASK_SUFFIX[task]}.pt"


def options(task: str) -> list[dict[str, Any]]:
    """Every (family, size) that supports the task, with its numbers, for the model table."""
    out = []
    for f in FAMILIES.values():
        if task not in f.tasks:
            continue
        for s in f.sizes:
            p, g, m = f.numbers[s]
            out.append(
                {
                    "family": f.key,
                    "family_label": f.label,
                    "size": s,
                    "size_label": SIZE_LABELS[s],
                    "weights": weights_name(f.key, s, task),
                    "params_m": p,
                    "gflops": g * (1.25 if task in ("segment", "pose") else 1.0),
                    "coco_map": m,
                    "note": f.note,
                }
            )
    return out


def gflops(family: str, size: str, task: str) -> float:
    g = FAMILIES[family].numbers[size][1]
    return g * (1.25 if task in ("segment", "pose") else 0.3 if task == "classify" else 1.0)


def estimate_vram_gb(family: str, size: str, task: str, imgsz: int, batch: int) -> float:
    """Rough training memory: framework overhead + per-image activations (grows with size^2)."""
    g = gflops(family, size, task)
    per_image = (0.09 + 0.0035 * g) * (imgsz / 640) ** 2
    if family == "rtdetr":
        per_image *= 1.6
    return 0.7 + batch * per_image


def max_batch(
    family: str, size: str, task: str, imgsz: int, vram_gb: float, headroom: float = 0.85
) -> int:
    """Largest power-of-two-ish batch that fits the memory estimate."""
    best = 1
    for b in (1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64):
        if estimate_vram_gb(family, size, task, imgsz, b) <= vram_gb * headroom:
            best = b
    return best


def estimate_epoch_seconds(
    family: str, size: str, task: str, imgsz: int, n_images: int, tflops: float | None = None
) -> float:
    """Rough time per epoch: training costs ~3x the forward pass; data loading adds a floor."""
    tflops = tflops or 4.0  # effective training throughput of a mid-range GPU (RTX 2070-class)
    g = gflops(family, size, task) * (imgsz / 640) ** 2
    compute = 3 * g * n_images / (tflops * 1000)
    loading = n_images * 0.004  # decoding + augmentation
    return max(compute, loading) + 0.15 * n_images * 0.01 + 5  # + validation, start-up


def recommend(n_train: int, task: str, vram_gb: float | None, imgsz: int) -> tuple[str, str, str]:
    """(family, size, why) from the dataset size and GPU memory."""
    if n_train < 300:
        size, why = "n", "Few images: a small model learns faster and overfits less."
    elif n_train < 3000:
        size, why = (
            "s",
            "A medium-sized dataset: 'small' is the usual sweet spot of speed and accuracy.",
        )
    else:
        size, why = "m", "A large dataset can feed a bigger model."
    if vram_gb is not None and imgsz >= 1024 and size == "m" and vram_gb < 10:
        size, why = (
            "s",
            why
            + f" At {imgsz} px on {vram_gb:.0f} GB, 'small' leaves room for a useful batch size.",
        )
    return "yolo26", size, why


def gpu_memory_gb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1e9
    except Exception:  # noqa: BLE001 - torch missing or broken: no GPU information
        return None
    return None
