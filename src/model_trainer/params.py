"""Every training setting the app exposes, with defaults, limits and plain-language help.

Each parameter maps 1:1 to an Ultralytics training argument (`key`). `level` decides whether it shows
in Simple mode. The help has four parts: what it does, a typical value, and what raising or lowering
it does.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Param:
    key: str
    label: str
    group: str  # data | augment | train | loss | system
    kind: str  # int | float | bool | choice | str
    default: Any
    what: str
    typical: str = ""
    up: str = ""  # effect of raising it
    down: str = ""  # effect of lowering it
    level: str = "advanced"  # simple | advanced
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: tuple[Any, ...] = ()
    tasks: tuple[str, ...] = ()  # empty = all tasks
    warn_colour: bool = False  # changes colours (bad when colour defines a class)
    extra: dict[str, Any] = field(default_factory=dict)

    def help(self) -> str:
        parts = [self.what]
        if self.typical:
            parts.append(f"Typical: {self.typical}.")
        if self.up:
            parts.append(f"Higher: {self.up}")
        if self.down:
            parts.append(f"Lower: {self.down}")
        return "\n".join(parts)


DET_TASKS = ("detect", "segment", "pose", "obb")

PARAMS: list[Param] = [
    # ----------------------------------------------------------------- data preparation
    Param(
        "imgsz",
        "Image size (px)",
        "data",
        "int",
        640,
        "Images are resized so their long side has this many pixels before the model sees them.",
        "640; 1024-1280 for small objects",
        "small objects become visible, but training is slower and needs more GPU memory.",
        "faster, but small objects may disappear.",
        level="simple",
        min=32,
        max=2048,
        step=32,
    ),
    Param(
        "rect",
        "Rectangular batches",
        "data",
        "bool",
        False,
        "Keeps the images' aspect ratio inside a batch instead of padding them to squares.",
        "off; on for very wide/tall images",
        "less wasted computation on non-square images (but disables shuffling, can lower accuracy).",
        tasks=DET_TASKS,
    ),
    Param(
        "cache",
        "Cache images",
        "data",
        "choice",
        "auto",
        "Keep decoded images in RAM or on disk so they are not re-read every epoch. 'auto' keeps "
        "them in RAM when they fit in the free memory at the start of training, otherwise off.",
        "auto",
        "'ram' is fastest but uses a lot of memory; 'disk' saves decoded copies next to the images.",
        choices=("auto", False, "ram", "disk"),
    ),
    Param(
        "workers",
        "Data loader workers",
        "data",
        "int",
        8,
        "Parallel processes preparing images while the GPU trains.",
        "4-8 on Windows",
        "faster feeding of the GPU, more RAM.",
        "less RAM; the GPU may wait for data.",
        min=0,
        max=32,
        step=1,
    ),
    Param(
        "fraction",
        "Use a fraction of the training set",
        "data",
        "float",
        1.0,
        "Train on only part of the training images (Ultralytics' own sampling).",
        "1.0; the wizard's 'quick test' subset is usually better",
        min=0.01,
        max=1.0,
        step=0.01,
    ),
    # ------------------------------------------------------------------- training
    Param(
        "epochs",
        "Epochs",
        "train",
        "int",
        100,
        "One epoch = the model sees every training image once.",
        "100 (50 for large datasets, 200-300 for small ones)",
        "longer training, usually better until it stops improving (early stopping ends it).",
        "quicker, may stop before the model is good.",
        level="simple",
        min=1,
        max=2000,
        step=1,
    ),
    Param(
        "patience",
        "Early stopping patience",
        "train",
        "int",
        50,
        "Stop when the validation score has not improved for this many epochs.",
        "30-50",
        "keeps going through plateaus.",
        "stops sooner, saves time, may stop too early.",
        level="simple",
        min=0,
        max=1000,
        step=1,
    ),
    Param(
        "batch",
        "Batch size",
        "train",
        "int",
        -1,
        "Images processed together in one step. -1 = automatic (fills ~60 % of the GPU memory).",
        "-1 (auto) or 8-32",
        "smoother learning, needs more GPU memory.",
        "fits on smaller GPUs; noisier learning.",
        level="simple",
        min=-1,
        max=512,
        step=1,
    ),
    Param(
        "optimizer",
        "Optimizer",
        "train",
        "choice",
        "auto",
        "The algorithm that updates the weights. 'auto' picks AdamW for short runs and MuSGD for long ones.",
        "auto",
        choices=("auto", "SGD", "MuSGD", "AdamW", "Adam", "NAdam", "RAdam", "RMSprop"),
    ),
    Param(
        "lr0",
        "Initial learning rate",
        "train",
        "float",
        0.01,
        "Step size of each weight update at the start of training (ignored when optimizer is 'auto').",
        "0.01 for SGD, 0.001 for AdamW",
        "learns faster but can become unstable.",
        "more stable, slower.",
        min=1e-6,
        max=0.5,
        step=0.0005,
    ),
    Param(
        "lrf",
        "Final learning rate (fraction)",
        "train",
        "float",
        0.01,
        "The learning rate at the end, as a fraction of the initial one.",
        "0.01",
        min=1e-4,
        max=1.0,
        step=0.005,
    ),
    Param(
        "cos_lr",
        "Cosine learning-rate schedule",
        "train",
        "bool",
        False,
        "Lower the learning rate along a cosine curve instead of a straight line.",
        "off",
        "smoother end of training.",
    ),
    Param(
        "warmup_epochs",
        "Warm-up epochs",
        "train",
        "float",
        3.0,
        "Start with a tiny learning rate and ramp it up over these epochs.",
        "3",
        "safer start for big learning rates.",
        min=0,
        max=20,
        step=0.5,
    ),
    Param(
        "momentum",
        "Momentum",
        "train",
        "float",
        0.937,
        "How much each update keeps of the previous direction (SGD momentum / Adam beta1).",
        "0.937",
        min=0.0,
        max=0.999,
        step=0.001,
    ),
    Param(
        "weight_decay",
        "Weight decay",
        "train",
        "float",
        0.0005,
        "A small pull of the weights towards zero to fight overfitting.",
        "0.0005",
        "stronger regularisation.",
        min=0.0,
        max=0.01,
        step=0.0001,
    ),
    Param(
        "freeze",
        "Freeze first layers",
        "train",
        "int",
        0,
        "Do not train the first N layers (keep what the pretrained model already knows).",
        "0; 10 when fine-tuning on very little data",
        "faster, keeps general features, adapts less.",
        min=0,
        max=30,
        step=1,
    ),
    Param(
        "amp",
        "Mixed precision (AMP)",
        "train",
        "bool",
        True,
        "Compute mostly in 16-bit floats: ~2x faster and half the memory on modern GPUs.",
        "on",
        down="turn off only if training becomes unstable (NaN losses).",
    ),
    Param(
        "dropout",
        "Dropout (classification)",
        "train",
        "float",
        0.0,
        "Randomly switch off connections in the classifier head during training.",
        "0-0.2",
        "fights overfitting on small datasets.",
        min=0,
        max=0.8,
        step=0.05,
        tasks=("classify",),
    ),
    Param(
        "multi_scale",
        "Multi-scale training",
        "train",
        "float",
        0.0,
        "Randomly vary the image size per batch by up to this fraction.",
        "0 (0.25-0.5 for robustness)",
        "more robust to object size, slower.",
        min=0,
        max=0.5,
        step=0.05,
        tasks=DET_TASKS,
    ),
    Param(
        "cls_pw",
        "Class weighting (imbalance)",
        "train",
        "float",
        0.0,
        "Weight rare classes more in the loss: 0 = off, 1 = full inverse frequency.",
        "0; 0.3-0.5 for imbalanced data",
        "rare classes count more (may raise false positives for them).",
        min=0,
        max=1,
        step=0.05,
        tasks=DET_TASKS,
    ),
    Param(
        "max_det",
        "Max detections per image",
        "train",
        "int",
        300,
        "Upper limit of objects the model reports per image.",
        "300; raise for very crowded images",
        min=1,
        max=3000,
        step=10,
        tasks=DET_TASKS,
    ),
    Param(
        "seed",
        "Random seed",
        "train",
        "int",
        0,
        "Makes shuffling and augmentation repeatable.",
        "0",
        min=0,
        max=2**31 - 1,
        step=1,
    ),
    Param(
        "deterministic",
        "Deterministic",
        "train",
        "bool",
        True,
        "Use repeatable (sometimes slower) GPU algorithms.",
        "on",
    ),
    Param(
        "close_mosaic",
        "Turn off mosaic for the last N epochs",
        "train",
        "int",
        10,
        "Train the last epochs on normal (non-mosaic) images so the model ends on realistic inputs.",
        "10",
        min=0,
        max=100,
        step=1,
        tasks=DET_TASKS,
    ),
    # ----------------------------------------------------------------------- loss
    Param(
        "box",
        "Box loss weight",
        "loss",
        "float",
        7.5,
        "How much box position errors count.",
        "7.5",
        min=0,
        max=50,
        step=0.5,
        tasks=DET_TASKS,
    ),
    Param(
        "cls",
        "Class loss weight",
        "loss",
        "float",
        0.5,
        "How much classification errors count.",
        "0.5",
        min=0,
        max=10,
        step=0.1,
        tasks=DET_TASKS,
    ),
    Param(
        "dfl",
        "Distribution focal loss weight",
        "loss",
        "float",
        1.5,
        "Fine box-edge accuracy term.",
        "1.5",
        min=0,
        max=10,
        step=0.1,
        tasks=DET_TASKS,
    ),
    Param(
        "pose",
        "Keypoint loss weight",
        "loss",
        "float",
        12.0,
        "How much keypoint position errors count.",
        "12",
        min=0,
        max=50,
        step=0.5,
        tasks=("pose",),
    ),
    Param(
        "kobj",
        "Keypoint visibility loss weight",
        "loss",
        "float",
        1.0,
        "How much keypoint visibility errors count.",
        "1",
        min=0,
        max=10,
        step=0.1,
        tasks=("pose",),
    ),
    # -------------------------------------------------------------------- augmentation
    Param(
        "hsv_h",
        "Hue shift",
        "augment",
        "float",
        0.015,
        "Randomly shifts colours around the colour wheel.",
        "0.015",
        "more colour variety; BAD if colour tells classes apart.",
        min=0,
        max=0.5,
        step=0.005,
        warn_colour=True,
    ),
    Param(
        "hsv_s",
        "Saturation change",
        "augment",
        "float",
        0.7,
        "Randomly makes colours more or less vivid.",
        "0.7",
        min=0,
        max=1,
        step=0.05,
        warn_colour=True,
    ),
    Param(
        "hsv_v",
        "Brightness change",
        "augment",
        "float",
        0.4,
        "Randomly makes images brighter or darker.",
        "0.4",
        "robust to lighting.",
        min=0,
        max=1,
        step=0.05,
    ),
    Param(
        "degrees",
        "Rotation (deg)",
        "augment",
        "float",
        0.0,
        "Random rotation up to +- this angle.",
        "0; 10-180 for top-down views",
        min=0,
        max=180,
        step=1,
    ),
    Param(
        "translate",
        "Translation",
        "augment",
        "float",
        0.1,
        "Random shift as a fraction of the image size.",
        "0.1",
        min=0,
        max=0.9,
        step=0.05,
    ),
    Param(
        "scale",
        "Scale",
        "augment",
        "float",
        0.5,
        "Random zoom in/out by up to this fraction.",
        "0.5",
        "robust to object size; too much when the camera distance is fixed.",
        min=0,
        max=0.95,
        step=0.05,
    ),
    Param(
        "shear",
        "Shear (deg)",
        "augment",
        "float",
        0.0,
        "Random slanting.",
        "0",
        min=0,
        max=45,
        step=1,
    ),
    Param(
        "perspective",
        "Perspective",
        "augment",
        "float",
        0.0,
        "Random perspective warp.",
        "0 (0.0001-0.001 if the camera angle varies)",
        min=0,
        max=0.001,
        step=0.0001,
    ),
    Param(
        "fliplr",
        "Horizontal flip",
        "augment",
        "float",
        0.5,
        "Probability of mirroring left-right.",
        "0.5; 0 if left/right matters (text, chirality)",
        min=0,
        max=1,
        step=0.05,
    ),
    Param(
        "flipud",
        "Vertical flip",
        "augment",
        "float",
        0.0,
        "Probability of flipping upside-down.",
        "0; 0.5 for top-down photos",
        min=0,
        max=1,
        step=0.05,
    ),
    Param(
        "mosaic",
        "Mosaic",
        "augment",
        "float",
        1.0,
        "Glue 4 training images into one: more objects and contexts per image.",
        "1.0",
        "strong help for small datasets.",
        min=0,
        max=1,
        step=0.05,
        tasks=DET_TASKS,
    ),
    Param(
        "mixup",
        "MixUp",
        "augment",
        "float",
        0.0,
        "Blend two images (ghost-like overlay).",
        "0-0.15",
        min=0,
        max=1,
        step=0.05,
        tasks=DET_TASKS,
    ),
    Param(
        "cutmix",
        "CutMix",
        "augment",
        "float",
        0.0,
        "Paste a rectangle from another image.",
        "0",
        min=0,
        max=1,
        step=0.05,
        tasks=DET_TASKS,
    ),
    Param(
        "copy_paste",
        "Copy-paste",
        "augment",
        "float",
        0.0,
        "Copy objects (by their masks) from one image into another.",
        "0-0.3 for segmentation",
        min=0,
        max=1,
        step=0.05,
        tasks=("segment",),
    ),
    Param(
        "erasing",
        "Random erasing (classification)",
        "augment",
        "float",
        0.4,
        "Blank out a random patch of the image.",
        "0.4",
        min=0,
        max=0.9,
        step=0.05,
        tasks=("classify",),
    ),
    Param(
        "auto_augment",
        "Auto-augment policy (classification)",
        "augment",
        "choice",
        "randaugment",
        "A ready-made set of random image changes for classification.",
        "randaugment",
        choices=("randaugment", "autoaugment", "augmix", None),
        tasks=("classify",),
    ),
]

BY_KEY = {p.key: p for p in PARAMS}
GROUPS = {
    "data": "Data preparation",
    "train": "Training",
    "augment": "Augmentation",
    "loss": "Loss weights",
}

AUG_KEYS = [p.key for p in PARAMS if p.group == "augment"]

AUG_PRESETS: dict[str, dict[str, Any]] = {
    "none": {
        "hsv_h": 0,
        "hsv_s": 0,
        "hsv_v": 0,
        "degrees": 0,
        "translate": 0,
        "scale": 0,
        "shear": 0,
        "perspective": 0,
        "fliplr": 0,
        "flipud": 0,
        "mosaic": 0,
        "mixup": 0,
        "cutmix": 0,
        "copy_paste": 0,
        "erasing": 0,
    },
    "light": {
        "hsv_h": 0.01,
        "hsv_s": 0.4,
        "hsv_v": 0.3,
        "degrees": 0,
        "translate": 0.05,
        "scale": 0.25,
        "shear": 0,
        "perspective": 0,
        "fliplr": 0.5,
        "flipud": 0,
        "mosaic": 0.5,
        "mixup": 0,
        "cutmix": 0,
        "copy_paste": 0,
        "erasing": 0.2,
    },
    "standard": {k: BY_KEY[k].default for k in AUG_KEYS},
    "strong": {
        "hsv_h": 0.02,
        "hsv_s": 0.8,
        "hsv_v": 0.5,
        "degrees": 15,
        "translate": 0.2,
        "scale": 0.7,
        "shear": 2,
        "perspective": 0.0005,
        "fliplr": 0.5,
        "flipud": 0.2,
        "mosaic": 1.0,
        "mixup": 0.15,
        "cutmix": 0.1,
        "copy_paste": 0.3,
        "erasing": 0.5,
    },
}
AUG_PRESET_HELP = {
    "none": "No changes at all. Only for debugging or when images are already very varied.",
    "light": "Gentle changes. Good when the camera, lighting and distance are fixed (a rig, a production line).",
    "standard": "Ultralytics' defaults. A good start for most photos.",
    "strong": "Heavy changes. For small datasets or very varied real-world conditions; needs more epochs.",
}


def applies(p: Param, task: str) -> bool:
    return not p.tasks or task in p.tasks


def defaults(task: str) -> dict[str, Any]:
    return {p.key: p.default for p in PARAMS if applies(p, task)}


def validate(values: Mapping[str, Any], task: str) -> tuple[dict[str, Any], list[str]]:
    """Clean values (known keys, right types, within limits) and a list of problems."""
    out: dict[str, Any] = {}
    problems: list[str] = []
    for k, v in values.items():
        p = BY_KEY.get(k)
        if p is None:
            out[k] = v  # raw Ultralytics argument from the YAML tab: passed through
            continue
        if not applies(p, task):
            continue
        try:
            if p.kind == "int":
                v = int(v)
            elif p.kind == "float":
                v = float(v)
            elif p.kind == "bool":
                v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")
        except (TypeError, ValueError):
            problems.append(f"{p.label}: {v!r} is not a valid {p.kind}")
            continue
        if p.kind in ("int", "float"):
            if p.min is not None and v < p.min:
                problems.append(f"{p.label}: {v} is below the minimum {p.min}")
                v = type(v)(p.min)
            if p.max is not None and v > p.max:
                problems.append(f"{p.label}: {v} is above the maximum {p.max}")
                v = type(v)(p.max)
        if p.kind == "choice" and p.choices and v not in p.choices:
            problems.append(f"{p.label}: {v!r} is not one of {p.choices}")
            continue
        out[k] = v
    return out, problems


def changed(values: Mapping[str, Any], task: str) -> dict[str, Any]:
    """Only the values that differ from the default (what the GUI highlights)."""
    d = defaults(task)
    return {k: v for k, v in values.items() if k in d and d[k] != v}
