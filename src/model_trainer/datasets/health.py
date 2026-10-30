"""Dataset health report: statistics plus checks, each explained in plain language.

Every check returns a level (ok / info / warn / bad), the measured value, and three short texts:
what it means, why it matters, and what to do. The GUI shows them as cards; `mtrain dataset check`
prints them.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .formats import SPLITS, Dataset, Item, Obj

TINY_PX = 8  # objects smaller than this (after resizing to the training size) are hard to detect
SMALL_PX = 32  # COCO "small": below 32 x 32 px
LARGE_PX = 96  # COCO "large": above 96 x 96 px
HEATMAP_CELLS = 16


@dataclass
class Check:
    id: str
    title: str
    level: str  # ok | info | warn | bad
    value: str
    meaning: str
    why: str
    action: str
    details: list[str] = field(default_factory=list)
    filter: str | None = None  # gallery filter that shows the affected images


@dataclass
class Report:
    stats: dict[str, Any]
    checks: list[Check]

    def to_dict(self) -> dict[str, Any]:
        return {"stats": self.stats, "checks": [asdict(c) for c in self.checks]}

    @property
    def worst(self) -> str:
        order = ["ok", "info", "warn", "bad"]
        return max((c.level for c in self.checks), key=order.index, default="ok")


# ----------------------------------------------------------------------------- geometry


def obj_size_px(o: Obj, it: Item, imgsz: int) -> tuple[float, float]:
    """Object width/height in pixels once the image is resized so its long side is `imgsz`."""
    if o.box is None or not it.width or not it.height:
        return (0.0, 0.0)
    scale = imgsz / max(it.width, it.height)
    return (o.box[2] * it.width * scale, o.box[3] * it.height * scale)


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU of two cx, cy, w, h boxes."""
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0.0, min(ax1, bx1) - max(ax0, bx0)), max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def label_errors(it: Item, n_classes: int) -> list[str]:
    errs = list(it.problems)
    for o in it.objects:
        if o.cls < 0 or o.cls >= n_classes:
            errs.append(f"class id {o.cls} is not in the class list")
        if o.box is not None:
            cx, cy, w, h = o.box
            if w <= 0 or h <= 0:
                errs.append("zero-size box")
            elif cx - w / 2 < -0.01 or cy - h / 2 < -0.01 or cx + w / 2 > 1.01 or cy + h / 2 > 1.01:
                errs.append("box outside the image")
        if o.polygon is not None and len(o.polygon) < 6:
            errs.append("polygon with fewer than 3 points")
        pts = o.polygon or o.obb
        if pts and any(v < -0.01 or v > 1.01 for v in pts):
            errs.append("polygon outside the image")
    boxes = [o for o in it.objects if o.box is not None]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i].cls == boxes[j].cls and iou(boxes[i].box, boxes[j].box) > 0.95:  # type: ignore[arg-type]
                errs.append("duplicate label (two identical boxes)")
                break
    return errs


# ------------------------------------------------------------------------------ hashing


def dhash(path: str, size: int = 8) -> int | None:
    """64-bit difference hash: near-identical images have hashes a few bits apart."""
    from PIL import Image

    try:
        with Image.open(path) as im:
            g = np.asarray(im.convert("L").resize((size + 1, size)), dtype=np.int16)
    except (OSError, ValueError):
        return None
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def near_duplicates(hashes: dict[str, int], max_bits: int = 4) -> list[tuple[str, str, int]]:
    """Pairs of images whose hashes differ by at most `max_bits` (bucketed, so it scales)."""
    buckets: dict[tuple[int, int], list[str]] = defaultdict(list)
    for k, h in hashes.items():
        for band in range(4):  # 4 bands of 16 bits: near pairs share at least one band exactly
            buckets[(band, (h >> (16 * band)) & 0xFFFF)].append(k)
    pairs: dict[tuple[str, str], int] = {}
    for keys in buckets.values():
        if len(keys) < 2 or len(keys) > 200:
            continue
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = sorted((keys[i], keys[j]))
                if (a, b) in pairs:
                    continue
                d = bin(hashes[a] ^ hashes[b]).count("1")
                if d <= max_bits:
                    pairs[(a, b)] = d
    return [(a, b, d) for (a, b), d in sorted(pairs.items())]


# -------------------------------------------------------------------------------- report


def analyse(ds: Dataset, imgsz: int = 640, hash_images: bool = True) -> Report:
    names = ds.names
    nc = len(names)
    splits = [s for s in SPLITS if any(it.split == s for it in ds.items)]
    per_split = Counter(it.split for it in ds.items)
    inst = {s: Counter() for s in splits}
    imgs = {s: Counter() for s in splits}
    sizes_short: list[float] = []
    sizes_by_class: dict[int, list[float]] = defaultdict(list)
    per_image = Counter()
    heat = np.zeros((HEATMAP_CELLS, HEATMAP_CELLS), dtype=int)
    cooc = np.zeros((nc, nc), dtype=int)
    img_sizes = Counter()
    aspect = []
    empty = Counter()
    errors: dict[str, list[str]] = {}
    for it in ds.items:
        img_sizes[f"{it.width}x{it.height}"] += 1
        if it.width and it.height:
            aspect.append(it.width / it.height)
        per_image[min(len(it.objects), 100)] += 1
        if not it.objects:
            empty[it.split] += 1
        present = set()
        for o in it.objects:
            inst[it.split][o.cls] += 1
            present.add(o.cls)
            if o.box is not None and ds.label_kind != "classes":
                w, h = obj_size_px(o, it, imgsz)
                s = min(w, h)
                sizes_short.append(s)
                sizes_by_class[o.cls].append(s)
                cx = min(HEATMAP_CELLS - 1, max(0, int(o.box[0] * HEATMAP_CELLS)))
                cy = min(HEATMAP_CELLS - 1, max(0, int(o.box[1] * HEATMAP_CELLS)))
                heat[cy, cx] += 1
        for c in present:
            imgs[it.split][c] += 1
        valid = sorted(c for c in present if 0 <= c < nc)
        for i, a in enumerate(valid):
            for b in valid[i + 1 :]:
                cooc[a, b] += 1
                cooc[b, a] += 1
        errs = label_errors(it, nc)
        if errs:
            errors[it.key] = errs

    total_inst = Counter()
    total_imgs = Counter()
    for s in splits:
        total_inst.update(inst[s])
        total_imgs.update(imgs[s])
    stats: dict[str, Any] = {
        "name": ds.name,
        "format": ds.format,
        "label_kind": ds.label_kind,
        "imgsz": imgsz,
        "classes": {int(k): v for k, v in names.items()},
        "images": {s: per_split[s] for s in splits},
        "images_total": len(ds.items),
        "instances": {s: {int(c): inst[s][c] for c in names} for s in splits},
        "images_per_class": {s: {int(c): imgs[s][c] for c in names} for s in splits},
        "instances_total": {int(c): total_inst[c] for c in names},
        "background_images": {s: empty[s] for s in splits},
        "objects_per_image": {int(k): v for k, v in sorted(per_image.items())},
        "image_sizes": dict(img_sizes.most_common(10)),
        "aspect_ratio": _summary(aspect),
        "object_size_px": _summary(sizes_short),
        "object_size_hist": _hist(sizes_short),
        "object_size_by_class": {int(c): _summary(v) for c, v in sizes_by_class.items()},
        "heatmap": heat.tolist(),
        "cooccurrence": cooc.tolist(),
        "label_errors": errors,
    }
    checks: list[Check] = []
    checks += _check_splits(stats, splits)
    if ds.label_kind != "classes":
        checks.append(_check_background(stats, len(ds.items)))
    checks += _check_balance(stats, names, ds.label_kind)
    if ds.label_kind != "classes" and sizes_short:
        checks.append(_check_size(stats, sizes_short, imgsz))
        checks.append(_check_crowding(stats, ds.items))
        checks.append(_check_position(heat))
    checks.append(_check_errors(errors))
    checks.append(_check_image_sizes(img_sizes, aspect))
    if hash_images:
        checks += _check_duplicates(ds, stats)
    checks.append(_check_groups(ds))
    return Report(stats, [c for c in checks if c is not None])


def _summary(v: Sequence[float]) -> dict[str, float]:
    if not v:
        return {}
    a = np.asarray(v, dtype=float)
    return {
        "min": float(a.min()),
        "p10": float(np.percentile(a, 10)),
        "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)),
        "max": float(a.max()),
        "mean": float(a.mean()),
    }


def _hist(v: Sequence[float]) -> dict[str, int]:
    edges = [0, 4, 8, 16, 32, 64, 96, 128, 256, 512, 100000]
    out = {}
    a = np.asarray(v, dtype=float)
    for lo, hi in zip(edges[:-1], edges[1:]):
        label = f"{lo}-{hi}" if hi < 100000 else f"{lo}+"
        out[label] = int(((a >= lo) & (a < hi)).sum())
    return out


def _check_splits(stats: dict[str, Any], splits: list[str]) -> list[Check]:
    n = stats["images"]
    val = n.get("val", 0)
    total = stats["images_total"]
    meaning = (
        "Training images teach the model; validation images are kept aside to measure it honestly."
    )
    why = "Without a validation set you cannot tell a good model from one that memorised its training images."
    if total == 0:
        return [
            Check(
                "splits",
                "Images per split",
                "bad",
                "no images",
                meaning,
                why,
                "Check the dataset path and the image folders.",
            )
        ]
    if val == 0:
        return [
            Check(
                "splits",
                "Images per split",
                "warn",
                ", ".join(f"{s}: {n[s]}" for s in splits),
                meaning,
                why,
                "No validation images: re-split in the wizard (step 1), e.g. 85 % train / 15 % val.",
            )
        ]
    share = val / total
    level = "ok" if 0.08 <= share <= 0.35 else "info"
    action = (
        "Fine."
        if level == "ok"
        else "Unusual share of validation images. 10-20 % is typical; re-split in the wizard if needed."
    )
    if val < 30:
        level, action = (
            "warn",
            "Fewer than 30 validation images: the scores will jump around. Add more if you can.",
        )
    return [
        Check(
            "splits",
            "Images per split",
            level,
            ", ".join(f"{s}: {n[s]}" for s in splits) + f" ({share:.0%} validation)",
            meaning,
            why,
            action,
        )
    ]


def _check_background(stats: dict[str, Any], total: int) -> Check:
    bg = sum(stats["background_images"].values())
    share = bg / max(total, 1)
    level = "ok" if share <= 0.15 else "warn" if share <= 0.5 else "bad"
    if bg == 0:
        level = "info"
    return Check(
        "background",
        "Background images (no objects)",
        level,
        f"{bg} of {total} ({share:.0%})",
        "Images without any labelled object.",
        "A few (up to ~10 %) teach the model what is NOT an object and reduce false detections. Too many waste "
        "training time and can make the model shy.",
        {
            "ok": "Good amount.",
            "info": "None at all: consider adding a few empty scenes (empty table, background only).",
            "warn": "Many empty images: check they are really empty and not just missing labels.",
            "bad": "Most images have no labels. Are the label files in the right place (labels/ next to images/)?",
        }[level],
        filter="background",
    )


def _check_balance(stats: dict[str, Any], names: dict[int, str], kind: str) -> list[Check]:
    tot = stats["instances_total"]
    counts = {c: tot.get(c, 0) for c in names}
    nonzero = [v for v in counts.values() if v > 0]
    checks = []
    if not nonzero:
        return [
            Check(
                "balance",
                "Class balance",
                "bad",
                "no labels",
                "How many examples each class has.",
                "Nothing to learn from.",
                "Check the label files.",
            )
        ]
    ratio = max(nonzero) / min(nonzero)
    level = "ok" if ratio <= 3 else "info" if ratio <= 10 else "warn" if ratio <= 50 else "bad"
    biggest = max(counts, key=counts.get)
    smallest = min((c for c in counts if counts[c] > 0), key=counts.get)
    unit = "images" if kind == "classes" else "objects"
    checks.append(
        Check(
            "balance",
            "Class balance (imbalance ratio)",
            level,
            f"{ratio:.1f}x  (largest: {names[biggest]} {counts[biggest]}, smallest: {names[smallest]} {counts[smallest]})",
            "The imbalance ratio is the largest class count divided by the smallest.",
            "Models learn the common classes well and the rare ones badly; rare classes are also the ones "
            "the scores hide, because the average is dominated by the big classes.",
            {
                "ok": "Well balanced.",
                "info": "Mild imbalance; usually fine. Watch the per-class scores of the small classes.",
                "warn": "Strong imbalance: turn on 'oversample rare classes' in the wizard, or add data for the small classes.",
                "bad": "Extreme imbalance: add data for the rare classes, merge them, or drop them.",
            }[level],
        )
    )
    rare = [names[c] for c, v in counts.items() if 0 < v < 50]
    missing = [names[c] for c, v in counts.items() if v == 0]
    if rare or missing:
        details = [
            f"{names[c]}: {v} {unit}"
            for c, v in sorted(counts.items(), key=lambda kv: kv[1])
            if v < 50
        ]
        checks.append(
            Check(
                "rare",
                "Classes too rare to learn",
                "bad" if missing else "warn",
                f"{len(rare)} rare, {len(missing)} without any example",
                f"Classes with fewer than 50 {unit} (or none at all).",
                "As a rule of thumb a class needs ~100+ examples (more if it looks like another class) to be "
                "learned reliably. A class without examples can never be predicted.",
                "Add examples, merge the class with a similar one, or drop it in the wizard (step 1).",
                details,
            )
        )
    return checks


def _check_size(stats: dict[str, Any], sizes: list[float], imgsz: int) -> Check:
    a = np.asarray(sizes)
    tiny = float((a < TINY_PX).mean())
    small = float((a < SMALL_PX).mean())
    level = "ok" if tiny < 0.02 else "warn" if tiny < 0.15 else "bad"
    action = {
        "ok": f"Objects are big enough at {imgsz} px.",
        "warn": f"Some objects become tiny at {imgsz} px. Try a larger image size (e.g. {min(imgsz * 2, 1920)}).",
        "bad": f"Many objects vanish at {imgsz} px. Use a larger image size (e.g. {min(imgsz * 2, 1920)}) "
        "or cut the images into tiles.",
    }[level]
    return Check(
        "object_size",
        f"Object size at the training size ({imgsz} px)",
        level,
        f"median {np.median(a):.0f} px, {tiny:.0%} below {TINY_PX} px, {small:.0%} below {SMALL_PX} px",
        f"Images are shrunk so their long side is {imgsz} px before training. This is the size of the "
        "objects' short side after that shrinking.",
        f"Below about {TINY_PX} px an object is only a few dots and the model misses it; below {SMALL_PX} px "
        "(COCO's 'small') accuracy drops noticeably. This is the most common hidden reason for poor results.",
        action,
        filter="tiny",
    )


def _check_crowding(stats: dict[str, Any], items: Sequence[Item]) -> Check:
    counts = [len(it.objects) for it in items if it.objects]
    mx = max(counts, default=0)
    level = "ok" if mx <= 300 else "warn"
    return Check(
        "crowding",
        "Objects per image",
        level,
        f"median {np.median(counts) if counts else 0:.0f}, max {mx}",
        "How many labelled objects an image holds.",
        "The model keeps at most `max_det` detections per image (default 300). Very crowded images also need "
        "a larger image size so objects do not merge.",
        "Fine." if level == "ok" else "Raise max_det above the maximum in Advanced > Training.",
        filter="crowded",
    )


def _check_position(heat: np.ndarray) -> Check:
    total = heat.sum()
    if total == 0:
        return None  # type: ignore[return-value]
    # share of the image area that holds 80 % of the object centres
    flat = np.sort(heat.flatten())[::-1]
    cells = int(np.searchsorted(np.cumsum(flat), 0.8 * total) + 1)
    share = cells / heat.size
    level = "ok" if share >= 0.25 else "info"
    return Check(
        "position",
        "Where objects are in the image",
        level,
        f"80 % of objects sit in {share:.0%} of the image area",
        "A heatmap of the object centres.",
        "If objects only ever appear in one region (e.g. the centre), the model may fail when they appear "
        "elsewhere. Flips, translation and mosaic augmentation help.",
        "Fine."
        if level == "ok"
        else "Objects are concentrated: keep translate/mosaic augmentation on.",
    )


def _check_errors(errors: dict[str, list[str]]) -> Check:
    n = len(errors)
    kinds = Counter(e.split(":")[0] for errs in errors.values() for e in errs)
    return Check(
        "label_errors",
        "Label problems",
        "ok" if n == 0 else "bad" if n > 20 else "warn",
        "none"
        if n == 0
        else f"{n} image(s): " + ", ".join(f"{k} ({v})" for k, v in kinds.most_common(5)),
        "Labels that cannot be right: boxes outside the image, zero size, duplicates, unknown class ids, "
        "unreadable files.",
        "Wrong labels teach the model wrong things and lower every score.",
        "Fix them in your labelling tool (the gallery filter 'label problems' lists them)."
        if n
        else "Nothing to fix.",
        [f"{k}: {'; '.join(v[:3])}" for k, v in list(errors.items())[:30]],
        filter="errors",
    )


def _check_image_sizes(sizes: Counter, aspect: list[float]) -> Check:
    a = np.asarray(aspect) if aspect else np.asarray([1.0])
    non_square = float((np.abs(a - 1) > 0.15).mean())
    return Check(
        "image_sizes",
        "Image sizes",
        "info" if len(sizes) > 1 else "ok",
        ", ".join(f"{k} ({v})" for k, v in sizes.most_common(4)),
        "The resolutions and shapes of the images.",
        "Mixed sizes are fine (everything is resized). For non-square images, 'rectangular batches' wastes "
        "less computation on padding.",
        "Consider 'rectangular batches' (Advanced > Data)." if non_square > 0.5 else "Fine.",
    )


def _check_duplicates(ds: Dataset, stats: dict[str, Any]) -> list[Check]:
    hashes = {}
    for it in ds.items:
        h = dhash(str(it.image))
        if h is not None:
            hashes[it.key] = h
    pairs = near_duplicates(hashes)
    split_of = {it.key: it.split for it in ds.items}
    cross = [(a, b, d) for a, b, d in pairs if split_of[a] != split_of[b]]
    stats["duplicates"] = [[a, b, d] for a, b, d in pairs[:500]]
    checks = [
        Check(
            "leakage_duplicates",
            "Duplicates across train and validation (leakage)",
            "ok" if not cross else "bad" if len(cross) > 0.02 * max(len(ds.items), 1) else "warn",
            f"{len(cross)} near-identical pair(s) split between sets",
            "The same (or almost the same) photo is in both the training and the validation set.",
            "The model has already seen those validation images, so the scores look better than reality.",
            "Re-split by group in the wizard, or remove the duplicates."
            if cross
            else "No leakage found.",
            [f"{a}  ~  {b}  ({d} bits)" for a, b, d in cross[:30]],
            filter="duplicates",
        )
    ]
    same = len(pairs) - len(cross)
    if same:
        checks.append(
            Check(
                "duplicates",
                "Duplicate images within a split",
                "info",
                f"{same} near-identical pair(s)",
                "Almost identical images in the same set.",
                "Harmless but they add little; many duplicates inflate the dataset size without adding variety.",
                "Optional: remove them to save training time.",
                filter="duplicates",
            )
        )
    return checks


def _check_groups(ds: Dataset) -> Check:
    groups: dict[str, set[str]] = defaultdict(set)
    for it in ds.items:
        if it.group is not None:
            groups[it.group].add(it.split)
    if not groups:
        return Check(
            "groups",
            "Groups (images that belong together)",
            "info",
            "no groups.csv",
            "A groups.csv file can say which images show the same scene (e.g. several photos of one layout).",
            "Images of one scene must stay in the same split, otherwise validation measures memory, not skill.",
            "Optional. Add groups.csv (image,group) next to the dataset yaml if your images come in groups.",
        )
    split_groups = [g for g, s in groups.items() if len(s) > 1]
    return Check(
        "groups",
        "Groups split across train and validation (leakage)",
        "ok" if not split_groups else "bad",
        f"{len(groups)} groups, {len(split_groups)} split across sets",
        "Images of one group (scene) found in more than one split.",
        "Those validation images were partly seen during training, so the scores look better than reality.",
        "Fine." if not split_groups else "Re-split by group in the wizard (step 1).",
        split_groups[:30],
    )


def recommend_imgsz(
    stats: dict[str, Any], candidates: Sequence[int] = (320, 480, 640, 800, 960, 1024, 1280)
) -> int:
    """Smallest standard size at which 95 % of objects are at least 10 px (short side)."""
    sizes = stats.get("object_size_px") or {}
    p10 = sizes.get("p10")
    imgsz = stats.get("imgsz", 640)
    if not p10:
        return 640
    for c in candidates:
        if p10 * c / imgsz >= 10:
            return c
    return candidates[-1]


def gallery_filter(ds: Dataset, report: dict[str, Any], name: str, imgsz: int = 640) -> list[Item]:
    """Images matching a gallery filter: all | background | errors | tiny | crowded | duplicates | class:<id>."""
    if name == "background":
        return [it for it in ds.items if not it.objects]
    if name == "errors":
        bad = set(report["stats"]["label_errors"])
        return [it for it in ds.items if it.key in bad]
    if name == "tiny":
        return [
            it
            for it in ds.items
            if any(0 < min(obj_size_px(o, it, imgsz)) < TINY_PX for o in it.objects)
        ]
    if name == "crowded":
        return sorted(ds.items, key=lambda it: -len(it.objects))[:200]
    if name == "duplicates":
        keys = {k for a, b, _ in report["stats"].get("duplicates", []) for k in (a, b)}
        return [it for it in ds.items if it.key in keys]
    if name.startswith("class:"):
        c = int(name.split(":", 1)[1])
        return [it for it in ds.items if any(o.cls == c for o in it.objects)]
    if name.startswith("split:"):
        s = name.split(":", 1)[1]
        return [it for it in ds.items if it.split == s]
    return list(ds.items)


def smallest_first(items: Sequence[Item], imgsz: int = 640) -> list[Item]:
    def key(it: Item) -> float:
        s = [min(obj_size_px(o, it, imgsz)) for o in it.objects if o.box is not None]
        return min(s) if s else math.inf

    return sorted(items, key=key)
