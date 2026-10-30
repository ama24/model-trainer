"""Build the exact dataset a run trains on, frozen inside the run folder.

    <run>/data/images/<split>/*.jpg     hard links to the original images (no extra space)
    <run>/data/labels/<split>/*.txt     labels converted to the run's task and class list
    <run>/data/train.txt                training image list (rare-class images repeated)
    <run>/data/data.yaml                what Ultralytics reads
    <run>/data/manifest.json            every file with size and SHA-1, the class mapping, the splits

Classification runs get the folder layout Ultralytics expects (`<split>/<class>/<image>`), cut
from the boxes when the source dataset has boxes.
"""

from __future__ import annotations

import logging
import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..util import link_or_copy, now, read_json, sha1_file, write_json, write_text, write_yaml
from .formats import SPLITS, Dataset, Item, Obj

log = logging.getLogger("model_trainer.prepare")


@dataclass
class ClassPlan:
    """What happens to each original class: dropped, merged into another, renamed."""

    drop: list[str] = field(default_factory=list)
    merge: dict[str, str] = field(default_factory=dict)  # old name -> name it joins
    rename: dict[str, str] = field(default_factory=dict)  # name -> new name
    dropped_images: str = "background"  # background: keep the image without those labels | remove

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> ClassPlan:
        d = dict(d or {})
        return cls(
            drop=list(d.get("drop") or []),
            merge=dict(d.get("merge") or {}),
            rename=dict(d.get("rename") or {}),
            dropped_images=d.get("dropped_images", "background"),
        )

    def mapping(self, names: Mapping[int, str]) -> tuple[dict[int, int | None], dict[int, str]]:
        """old class id -> new id (None = dropped), and the new id -> name list."""
        final: dict[int, str | None] = {}
        for i, n in names.items():
            if n in self.drop:
                final[i] = None
                continue
            target = n
            seen = set()
            while target in self.merge and target not in seen:  # follow merge chains
                seen.add(target)
                target = self.merge[target]
            final[i] = self.rename.get(target, target)
        new_names: dict[int, str] = {}
        index: dict[str, int] = {}
        for i in sorted(names):
            n = final[i]
            if n is not None and n not in index:
                index[n] = len(index)
                new_names[index[n]] = n
        return {i: (None if n is None else index[n]) for i, n in final.items()}, new_names


@dataclass
class SplitPlan:
    mode: str = "keep"  # keep the dataset's splits | resplit
    ratios: dict[str, float] = field(default_factory=lambda: {"train": 0.85, "val": 0.15})
    stratify: bool = True  # keep every class's share similar in each split
    by_group: bool = True  # images of one group stay together
    seed: int = 0

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> SplitPlan:
        d = dict(d or {})
        out = cls()
        for k in ("mode", "stratify", "by_group", "seed"):
            if k in d:
                setattr(out, k, d[k])
        if d.get("ratios"):
            out.ratios = {k: float(v) for k, v in d["ratios"].items() if float(v) > 0}
        return out


# ---------------------------------------------------------------------------- splitting


def resplit(items: Sequence[Item], plan: SplitPlan) -> dict[str, str]:
    """item key -> new split. Iterative stratification over groups (rarest classes first)."""
    rng = random.Random(plan.seed)
    units: dict[str, list[Item]] = defaultdict(list)
    for it in items:
        g = it.group if (plan.by_group and it.group is not None) else f"__{it.key}"
        units[g].append(it)
    unit_ids = list(units)
    rng.shuffle(unit_ids)
    total = sum(plan.ratios.values())
    ratios = {s: r / total for s, r in plan.ratios.items()}
    counts = {u: Counter(o.cls for it in units[u] for o in it.objects) for u in unit_ids}
    class_total = Counter()
    for c in counts.values():
        class_total.update(c)
    if plan.stratify and class_total:
        # units holding the rarest classes are placed first, while every split still has room
        unit_ids.sort(key=lambda u: min((class_total[c] for c in counts[u]), default=math.inf))
    n_images = sum(len(v) for v in units.values())
    assigned_imgs = Counter()
    assigned_cls: dict[str, Counter] = {s: Counter() for s in ratios}
    out: dict[str, str] = {}
    for u in unit_ids:

        def deficit(s: str, u: str = u) -> float:
            want_img = ratios[s] * n_images - assigned_imgs[s]
            score = want_img / max(n_images, 1)
            if plan.stratify and counts[u]:
                score += sum(
                    (ratios[s] * class_total[c] - assigned_cls[s][c]) / max(class_total[c], 1)
                    for c in counts[u]
                ) / len(counts[u])
            return score

        best = max(ratios, key=lambda s: (deficit(s), ratios[s]))
        for it in units[u]:
            out[it.key] = best
        assigned_imgs[best] += len(units[u])
        assigned_cls[best].update(counts[u])
    return out


def subset_keys(
    items: Sequence[Item], fraction: float, seed: int, by_group: bool = True
) -> set[str]:
    """Keep `fraction` of the training images (whole groups), for quick test runs."""
    if fraction >= 1:
        return {it.key for it in items}
    units: dict[str, list[str]] = defaultdict(list)
    for it in items:
        units[it.group if (by_group and it.group) else it.key].append(it.key)
    ids = sorted(units)
    random.Random(seed).shuffle(ids)
    keep = ids[: max(1, round(len(ids) * fraction))]
    return {k for u in keep for k in units[u]}


def repeat_factors(items: Sequence[Item], max_repeat: float = 4.0, seed: int = 0) -> dict[str, int]:
    """How often each training image is listed, so rare classes are seen more often.

    Class factor r(c) = sqrt(median image share / image share of c), capped at `max_repeat`
    (repeat-factor sampling, as used for long-tailed datasets). An image gets the factor of its
    rarest class; fractions are rounded randomly with the seed.
    """
    n = len(items)
    freq = Counter(c for it in items for c in {o.cls for o in it.objects})
    if not freq or n == 0:
        return {it.key: 1 for it in items}
    shares = {c: v / n for c, v in freq.items()}
    median = float(np.median(list(shares.values())))
    r = {c: min(max_repeat, max(1.0, math.sqrt(median / s))) for c, s in shares.items()}
    rng = random.Random(seed)
    out = {}
    for it in items:
        f = max((r[o.cls] for o in it.objects), default=1.0)
        whole = int(f)
        out[it.key] = whole + (1 if rng.random() < f - whole else 0)
    return out


# ---------------------------------------------------------------------------- label lines


def _fmt(v: Sequence[float]) -> str:
    return " ".join(f"{min(max(x, 0.0), 1.0):.6f}" for x in v)


def obb_from_polygon(poly: Sequence[float], w: int, h: int) -> list[float]:
    import cv2

    pts = np.array(poly, dtype=np.float32).reshape(-1, 2) * np.array([w, h], dtype=np.float32)
    box = cv2.boxPoints(cv2.minAreaRect(pts))
    return (box / np.array([w, h], dtype=np.float32)).flatten().tolist()


def label_line(o: Obj, new_cls: int, task: str, it: Item, kpt_ndim: int = 3) -> str | None:
    if task == "detect" and o.box:
        return f"{new_cls} {_fmt(o.box)}"
    if task == "segment" and o.polygon:
        return f"{new_cls} {_fmt(o.polygon)}"
    if task == "pose" and o.box and o.keypoints is not None:
        parts = []
        for x, y, vis in o.keypoints:
            parts += [f"{x:.6f}", f"{y:.6f}"] + ([f"{int(vis)}"] if kpt_ndim == 3 else [])
        return f"{new_cls} {_fmt(o.box)} {' '.join(parts)}"
    if task == "obb":
        corners = o.obb or (obb_from_polygon(o.polygon, it.width, it.height) if o.polygon else None)
        if corners is None and o.box:
            cx, cy, bw, bh = o.box
            corners = [
                cx - bw / 2,
                cy - bh / 2,
                cx + bw / 2,
                cy - bh / 2,
                cx + bw / 2,
                cy + bh / 2,
                cx - bw / 2,
                cy + bh / 2,
            ]
        if corners:
            return f"{new_cls} {_fmt(corners)}"
    return None


# ---------------------------------------------------------------------------- the builder


def prepare(
    ds: Dataset,
    out: Path,
    task: str,
    classes: ClassPlan | None = None,
    split: SplitPlan | None = None,
    subset: float = 1.0,
    balance: str = "none",  # none | oversample
    max_repeat: float = 4.0,
    crop_padding: float = 0.1,
    hash_files: bool = True,  # SHA-1 of every file in the manifest (off for throw-away previews)
) -> dict[str, Any]:
    """Write the run dataset under `out` and return the manifest."""
    classes = classes or ClassPlan()
    split = split or SplitPlan()
    mapping, new_names = classes.mapping(ds.names)
    if not new_names:
        raise ValueError("every class is dropped: nothing left to train")
    if split.mode == "resplit":
        new_split = resplit(ds.items, split)
    else:
        new_split = {it.key: it.split for it in ds.items}
        if not any(s == "val" for s in new_split.values()):
            raise ValueError(
                "the dataset has no validation images: choose 're-split' in step 1 of the wizard"
            )
    items: list[tuple[Item, str]] = []
    for it in ds.items:
        if "image missing or unreadable" in it.problems:
            continue
        if classes.dropped_images == "remove" and any(
            mapping.get(o.cls) is None for o in it.objects
        ):
            continue
        items.append((it, new_split[it.key]))
    train_items = [it for it, s in items if s == "train"]
    keep = subset_keys(train_items, subset, split.seed, split.by_group)
    items = [(it, s) for it, s in items if s != "train" or it.key in keep]

    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}
    used_names: set[str] = set()
    counts = {s: Counter() for s in SPLITS}
    n_images = Counter()
    skipped = Counter()
    train_list: list[tuple[Item, str]] = []
    groups_rows: list[tuple[str, str, str]] = []
    for it, s in items:
        stem = it.image.stem
        name = stem
        k = 1
        while f"{s}/{name}" in used_names:
            name = f"{stem}_{k}"
            k += 1
        used_names.add(f"{s}/{name}")
        objs = [(o, mapping.get(o.cls)) for o in it.objects]
        objs = [(o, c) for o, c in objs if c is not None]
        if task == "classify":
            n = _write_classify(
                it, s, name, objs, new_names, out, files, ds.label_kind, crop_padding
            )
            n_images[s] += n
            for _, c in objs:
                counts[s][c] += 1
            continue
        rel_img = f"images/{s}/{name}{it.image.suffix.lower()}"
        link_or_copy(it.image, out / rel_img)
        lines = []
        kpt_ndim = int(ds.kpt_shape[1]) if ds.kpt_shape else 3
        for o, c in objs:
            line = label_line(o, c, task, it, kpt_ndim)
            if line is None:
                skipped[f"no {task} label"] += 1
                continue
            lines.append(line)
            counts[s][c] += 1
        write_text(out / f"labels/{s}/{name}.txt", "\n".join(lines) + ("\n" if lines else ""))
        files[rel_img] = {"size": (out / rel_img).stat().st_size, "source": str(it.image)}
        n_images[s] += 1
        groups_rows.append((Path(rel_img).name, s, it.group or ""))
        if s == "train":
            train_list.append((it, rel_img))

    repeats = {}
    if task != "classify":
        if balance == "oversample":
            repeats = repeat_factors([it for it, _ in train_list], max_repeat, split.seed)
        lines = []
        for it, rel in train_list:
            lines += [f"./{rel}"] * repeats.get(it.key, 1)
        write_text(out / "train.txt", "\n".join(lines) + "\n")
        _write_groups(out / "groups.csv", groups_rows)
    if hash_files:
        for rel, info in files.items():
            info["sha1"] = sha1_file(out / rel)
    yaml_path = out / "data.yaml"
    if task == "classify":
        data = {"path": str(out), "train": "train", "val": "val", "names": new_names}
        if n_images.get("test"):
            data["test"] = "test"
    else:
        data = {"path": str(out), "train": "train.txt", "val": "images/val", "names": new_names}
        if n_images.get("test"):
            data["test"] = "images/test"
        if task == "pose":
            data["kpt_shape"] = ds.kpt_shape or [3, 3]
            if ds.flip_idx:
                data["flip_idx"] = ds.flip_idx
    write_yaml(yaml_path, data)
    manifest = {
        "created": now(),
        "source_dataset": ds.name,
        "source": ds.source,
        "task": task,
        "classes": new_names,
        "mapping": {ds.names[i]: (None if c is None else new_names[c]) for i, c in mapping.items()},
        "images": dict(n_images),
        "instances": {
            s: {new_names[c]: v for c, v in counts[s].items()} for s in SPLITS if counts[s]
        },
        "train_list_length": sum(repeats.get(it.key, 1) for it, _ in train_list)
        if task != "classify"
        else None,
        "skipped": dict(skipped),
        "settings": {
            "classes": vars(classes),
            "split": vars(split),
            "subset": subset,
            "balance": balance,
            "max_repeat": max_repeat,
        },
        "files": files,
    }
    write_json(out / "manifest.json", manifest)
    if skipped:
        log.warning("objects without a %s label were skipped: %s", task, dict(skipped))
    return manifest


def _write_groups(path: Path, rows: Sequence[tuple[str, str, str]]) -> None:
    import csv
    import io

    buf = io.StringIO()
    wr = csv.writer(buf, lineterminator="\n")
    wr.writerow(["image", "split", "group"])
    wr.writerows(rows)
    write_text(path, buf.getvalue())


def _write_classify(
    it: Item,
    split: str,
    name: str,
    objs: Sequence[tuple[Obj, int]],
    names: Mapping[int, str],
    out: Path,
    files: dict[str, dict[str, Any]],
    kind: str,
    padding: float,
) -> int:
    from PIL import Image

    if kind == "classes":
        if not objs:
            return 0
        c = objs[0][1]
        rel = f"{split}/{names[c]}/{name}{it.image.suffix.lower()}"
        link_or_copy(it.image, out / rel)
        files[rel] = {"size": (out / rel).stat().st_size, "source": str(it.image)}
        return 1
    n = 0
    with Image.open(it.image) as im:
        img = im.convert("RGB")
    w, h = img.size
    for k, (o, c) in enumerate(objs):
        if not o.box:
            continue
        cx, cy, bw, bh = o.box
        side = max(bw * w, bh * h) * (1 + 2 * padding)
        x0, y0 = int(cx * w - side / 2), int(cy * h - side / 2)
        crop = img.crop((x0, y0, x0 + int(side), y0 + int(side)))
        rel = f"{split}/{names[c]}/{name}_{k}.jpg"
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        crop.save(out / rel, quality=95)
        files[rel] = {"size": (out / rel).stat().st_size, "source": str(it.image), "crop": k}
        n += 1
    return n


def verify(out: Path, full: bool = False) -> list[str]:
    """Problems with a prepared run dataset (missing or changed files). Empty = intact."""
    m = read_json(out / "manifest.json")
    problems = []
    for rel, info in m["files"].items():
        p = out / rel
        if not p.is_file():
            problems.append(f"missing: {rel}")
        elif p.stat().st_size != info["size"]:
            problems.append(f"changed size: {rel}")
        elif full and sha1_file(p) != info.get("sha1"):
            problems.append(f"changed content: {rel}")
        if len(problems) > 50:
            problems.append("... (more)")
            break
    return problems
