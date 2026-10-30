"""Read datasets in the common formats into one in-memory model.

Supported:
- **YOLO** (Ultralytics): a dataset yaml (`path`, `train`, `val`, `test`, `names`, optional `kpt_shape`)
  with `images/<split>` and `labels/<split>` folders (or txt image lists). Label lines hold boxes,
  polygons (segmentation), keypoints (pose) or 4 corners (rotated boxes, OBB).
- **COCO**: one json per split (`train.json`, `instances_val2017.json`, ...) with bboxes, polygon
  segmentations and keypoints.
- **Classification folders**: `<root>/<split>/<class>/*.jpg` or `<root>/<class>/*.jpg`.

An optional `groups.csv` (image, [split,] group) says which images belong together (e.g. photos of
one scene) so they are never split between training and validation.
"""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from ..util import IMAGE_EXTS, read_yaml

log = logging.getLogger("model_trainer.datasets")

SPLITS = ("train", "val", "test")
TASKS = ("detect", "segment", "pose", "obb", "classify")
TASK_LABELS = {
    "detect": "Detection (boxes)",
    "segment": "Segmentation (masks)",
    "pose": "Pose (keypoints)",
    "obb": "Rotated boxes (OBB)",
    "classify": "Classification (whole image)",
}


@dataclass
class Obj:
    cls: int
    box: tuple[float, float, float, float] | None = None  # cx, cy, w, h normalised 0..1
    polygon: list[float] | None = None  # x1, y1, x2, y2, ... normalised
    keypoints: list[tuple[float, float, float]] | None = None  # x, y, visibility (2 = visible)
    obb: list[float] | None = None  # 4 corners x1, y1, ..., x4, y4 normalised


@dataclass
class Item:
    image: Path
    split: str
    objects: list[Obj] = field(default_factory=list)
    width: int = 0
    height: int = 0
    group: str | None = None
    label_file: Path | None = None
    problems: list[str] = field(default_factory=list)  # label errors found while reading

    @property
    def key(self) -> str:
        return f"{self.split}/{self.image.name}"


@dataclass
class Dataset:
    name: str
    root: Path
    format: str  # yolo | coco | folders
    names: dict[int, str]
    items: list[Item]
    label_kind: str  # what the labels contain: boxes | polygons | keypoints | obb | classes
    kpt_shape: list[int] | None = None
    flip_idx: list[int] | None = None
    source: str = ""  # the yaml / json / folder it was read from

    def tasks(self) -> dict[str, str]:
        """Tasks the labels can support -> '' if fine, else the reason it is not possible."""
        k = self.label_kind
        why_not = {
            "detect": ""
            if k in ("boxes", "polygons", "keypoints", "obb")
            else "needs boxes or masks",
            "segment": "" if k == "polygons" else "needs polygon / mask labels",
            "pose": "" if k == "keypoints" else "needs keypoint labels",
            "obb": "" if k in ("obb", "polygons") else "needs rotated boxes or polygons",
            "classify": "",  # folders directly; boxes/masks via object crops
        }
        return why_not

    def default_task(self) -> str:
        return {
            "boxes": "detect",
            "polygons": "segment",
            "keypoints": "pose",
            "obb": "obb",
            "classes": "classify",
        }[self.label_kind]

    def split_counts(self) -> dict[str, int]:
        out = {s: 0 for s in SPLITS}
        for it in self.items:
            out[it.split] = out.get(it.split, 0) + 1
        return out


# --------------------------------------------------------------------------- detection


def detect_format(path: Path) -> tuple[str, Path]:
    """(format, the file or folder to read) for a dataset folder, yaml or json."""
    path = Path(path)
    if path.is_file():
        if path.suffix in (".yaml", ".yml"):
            return "yolo", path
        if path.suffix == ".json":
            return "coco", path
        raise ValueError(f"{path}: expected a dataset .yaml (YOLO) or .json (COCO)")
    if not path.is_dir():
        raise FileNotFoundError(path)
    yamls = [p for p in sorted(path.glob("*.y*ml")) if _is_yolo_yaml(p)]
    if yamls:
        # prefer data.yaml / dataset.yaml, then the first one
        yamls.sort(key=lambda p: (p.stem not in ("data", "dataset"), p.name))
        return "yolo", yamls[0]
    jsons = [p for p in sorted(path.rglob("*.json")) if _is_coco_json(p)]
    if jsons:
        return "coco", path
    if (path / "images").is_dir() and (path / "labels").is_dir():
        return "yolo", path  # images/labels folders without a yaml
    if _class_folders(path):
        return "folders", path
    raise ValueError(
        f"{path}: no dataset found. Expected a YOLO dataset yaml, COCO json files, or one folder per class."
    )


def _is_yolo_yaml(p: Path) -> bool:
    try:
        d = read_yaml(p)
    except Exception:  # noqa: BLE001 - any unreadable yaml is simply not a dataset yaml
        return False
    return isinstance(d, dict) and "names" in d and ("train" in d or "val" in d)


def _is_coco_json(p: Path) -> bool:
    if p.stat().st_size > 2_000_000_000:
        return False
    try:
        with p.open(encoding="utf-8") as fh:
            head = fh.read(4096)
    except (OSError, UnicodeDecodeError):
        return False
    return '"images"' in head or '"annotations"' in head or '"categories"' in head


def _class_folders(path: Path) -> list[Path]:
    """Folders that directly hold images (one per class), at depth 1 or 2 (split/class)."""
    out = []
    for d in sorted(path.iterdir()):
        if not d.is_dir():
            continue
        if any(f.suffix.lower() in IMAGE_EXTS for f in d.iterdir() if f.is_file()):
            out.append(d)
        else:
            out += [
                c
                for c in sorted(d.iterdir())
                if c.is_dir() and any(f.suffix.lower() in IMAGE_EXTS for f in c.iterdir())
            ]
    return out


def load(path: str | Path, name: str | None = None, label_kind: str | None = None) -> Dataset:
    """Read any supported dataset. `label_kind="obb"` forces 8-number lines to be read as OBB."""
    fmt, src = detect_format(Path(path))
    if fmt == "yolo":
        ds = load_yolo(src, label_kind)
    elif fmt == "coco":
        ds = load_coco(src)
    else:
        ds = load_folders(src)
    if name:
        ds.name = name
    apply_groups(ds)
    read_sizes(ds.items)
    return ds


# -------------------------------------------------------------------------------- YOLO


def _list_images(spec: Any, base: Path) -> list[Path]:
    if spec is None:
        return []
    if isinstance(spec, (list, tuple)):
        return [p for s in spec for p in _list_images(s, base)]
    p = Path(str(spec))
    if not p.is_absolute():
        p = base / p
    if p.is_dir():
        return sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTS)
    if p.is_file() and p.suffix == ".txt":
        out = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                q = Path(line)
                out.append(q if q.is_absolute() else (p.parent / q).resolve())
        return out
    log.warning("%s does not exist", p)
    return []


def label_path_for(image: Path) -> Path:
    """Ultralytics convention: .../images/<split>/x.jpg -> .../labels/<split>/x.txt"""
    parts = list(image.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image.with_suffix(".txt")


def parse_yolo_line(
    line: str, kpt_shape: Sequence[int] | None, force_obb: bool = False
) -> tuple[Obj | None, str | None]:
    """One label line -> (object, problem)."""
    vals = line.split()
    if not vals:
        return None, None
    try:
        cls = int(float(vals[0]))
        nums = [float(v) for v in vals[1:]]
    except ValueError:
        return None, f"unreadable line: {line[:60]!r}"
    if kpt_shape:
        nk, ndim = int(kpt_shape[0]), int(kpt_shape[1])
        if len(nums) != 4 + nk * ndim:
            return None, f"pose line has {len(nums)} numbers, expected {4 + nk * ndim}"
        kps = []
        for i in range(nk):
            x, y = nums[4 + i * ndim], nums[5 + i * ndim]
            v = nums[6 + i * ndim] if ndim == 3 else 2.0
            kps.append((x, y, v))
        return Obj(cls, box=tuple(nums[:4]), keypoints=kps), None  # type: ignore[arg-type]
    if len(nums) == 4:
        return Obj(cls, box=tuple(nums)), None  # type: ignore[arg-type]
    if len(nums) == 8 and force_obb:
        return Obj(cls, obb=nums, box=_poly_box(nums)), None
    if len(nums) >= 6 and len(nums) % 2 == 0:
        return Obj(cls, polygon=nums, box=_poly_box(nums)), None
    if len(nums) < 6:
        return None, f"line with {len(nums)} numbers (a box needs 4, a polygon at least 6)"
    return None, f"odd number of polygon coordinates ({len(nums)})"


def _poly_box(p: Sequence[float]) -> tuple[float, float, float, float]:
    xs, ys = p[0::2], p[1::2]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    return ((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)


def load_yolo(src: Path, label_kind: str | None = None) -> Dataset:
    if src.is_dir():  # images/ + labels/ without a yaml: splits from the folder names
        cfg: dict[str, Any] = {"path": str(src), "names": {}}
        for s in SPLITS:
            if (src / "images" / s).is_dir():
                cfg[s] = f"images/{s}"
        if not any(s in cfg for s in SPLITS):
            cfg["train"] = "images"
        yaml_path = None
    else:
        cfg = read_yaml(src)
        yaml_path = src
    base = Path(cfg.get("path") or (yaml_path.parent if yaml_path else src))
    if not base.is_absolute():
        base = ((yaml_path.parent if yaml_path else Path.cwd()) / base).resolve()
    if not base.exists() and yaml_path is not None:
        base = yaml_path.parent  # moved dataset: the yaml's own folder
    names = cfg.get("names") or {}
    if isinstance(names, list):
        names = dict(enumerate(names))
    names = {int(k): str(v) for k, v in names.items()}
    kpt_shape = cfg.get("kpt_shape")
    force_obb = label_kind == "obb" or (yaml_path is not None and "obb" in yaml_path.stem.lower())
    items: list[Item] = []
    seen: set[Path] = set()
    for split in SPLITS:
        for img in _list_images(cfg.get(split), base):
            if img in seen:
                continue
            seen.add(img)
            it = Item(img, split, label_file=label_path_for(img))
            if it.label_file and it.label_file.is_file():
                for line in it.label_file.read_text(encoding="utf-8").splitlines():
                    obj, problem = parse_yolo_line(line, kpt_shape, force_obb)
                    if problem:
                        it.problems.append(problem)
                    if obj is not None:
                        it.objects.append(obj)
            items.append(it)
    max_cls = max((o.cls for it in items for o in it.objects), default=-1)
    for i in range(max_cls + 1):
        names.setdefault(i, f"class_{i}")
    if kpt_shape:
        kind = "keypoints"
    elif force_obb:
        kind = "obb"
    elif any(o.polygon for it in items for o in it.objects):
        kind = "polygons"
    else:
        kind = "boxes"
    return Dataset(
        name=(yaml_path.parent.name if yaml_path else src.name),
        root=base,
        format="yolo",
        names=dict(sorted(names.items())),
        items=items,
        label_kind=kind,
        kpt_shape=list(kpt_shape) if kpt_shape else None,
        flip_idx=cfg.get("flip_idx"),
        source=str(yaml_path or src),
    )


# -------------------------------------------------------------------------------- COCO


def _split_of(name: str) -> str:
    n = name.lower()
    if "val" in n:
        return "val"
    if "test" in n:
        return "test"
    return "train"


def load_coco(src: Path) -> Dataset:
    files = [src] if src.is_file() else [p for p in sorted(src.rglob("*.json")) if _is_coco_json(p)]
    root = src.parent if src.is_file() else src
    names: dict[int, str] = {}
    cat_index: dict[int, int] = {}
    items: list[Item] = []
    kpt_shape = None
    kinds: set[str] = set()
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        split = _split_of(f.stem)
        for c in sorted(data.get("categories", []), key=lambda c: c["id"]):
            if c["id"] not in cat_index:
                cat_index[c["id"]] = len(cat_index)
                names[cat_index[c["id"]]] = c["name"]
            if c.get("keypoints"):
                kpt_shape = [len(c["keypoints"]), 3]
        by_img: dict[int, Item] = {}
        for im in data.get("images", []):
            p = _find_image(im["file_name"], f.parent, root, split)
            it = Item(p, split, width=int(im.get("width", 0)), height=int(im.get("height", 0)))
            by_img[im["id"]] = it
            items.append(it)
        for a in data.get("annotations", []):
            it = by_img.get(a["image_id"])
            if it is None:
                continue
            if a["category_id"] not in cat_index:
                it.problems.append(f"unknown category id {a['category_id']}")
                continue
            w, h = max(it.width, 1), max(it.height, 1)
            x, y, bw, bh = a.get("bbox", [0, 0, 0, 0])
            obj = Obj(
                cat_index[a["category_id"]],
                box=((x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h),
            )
            seg = a.get("segmentation")
            if isinstance(seg, list) and seg and isinstance(seg[0], list):
                largest = max(seg, key=len)  # multi-part masks: keep the largest part
                obj.polygon = [v / (w if i % 2 == 0 else h) for i, v in enumerate(largest)]
                kinds.add("polygons")
            elif isinstance(seg, dict):
                it.problems.append("RLE mask not supported (box kept)")
            if a.get("keypoints"):
                k = a["keypoints"]
                obj.keypoints = [(k[i] / w, k[i + 1] / h, k[i + 2]) for i in range(0, len(k), 3)]
                kinds.add("keypoints")
            if obj.box and obj.box[2] > 0:
                kinds.add("boxes")
            it.objects.append(obj)
    kind = "keypoints" if "keypoints" in kinds else "polygons" if "polygons" in kinds else "boxes"
    return Dataset(
        name=root.name,
        root=root,
        format="coco",
        names=names,
        items=items,
        label_kind=kind,
        kpt_shape=kpt_shape if kind == "keypoints" else None,
        source=str(src),
    )


def _find_image(file_name: str, json_dir: Path, root: Path, split: str) -> Path:
    candidates = [
        json_dir / file_name,
        json_dir / "images" / file_name,
        root / file_name,
        root / "images" / file_name,
        root / "images" / split / file_name,
        root / split / file_name,
        json_dir.parent / "images" / file_name,
        json_dir.parent / "images" / split / file_name,
    ]
    for c in candidates:
        if c.is_file():
            return c
    return candidates[0]


# ----------------------------------------------------------------------------- folders


def load_folders(src: Path) -> Dataset:
    class_dirs = _class_folders(src)
    names_order = sorted({d.name for d in class_dirs})
    idx = {n: i for i, n in enumerate(names_order)}
    items = []
    for d in class_dirs:
        split = d.parent.name if d.parent != src and d.parent.name in SPLITS else "train"
        if d.parent != src and d.parent.name not in SPLITS:
            split = _split_of(d.parent.name)
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in IMAGE_EXTS:
                items.append(Item(f, split, objects=[Obj(idx[d.name])]))
    return Dataset(
        name=src.name,
        root=src,
        format="folders",
        names=dict(enumerate(names_order)),
        items=items,
        label_kind="classes",
        source=str(src),
    )


# ------------------------------------------------------------------------------ groups


def apply_groups(ds: Dataset) -> None:
    """Read groups.csv (image, [split,] group) next to the dataset, if there is one."""
    candidates = [ds.root / "groups.csv", Path(ds.source).parent / "groups.csv"]
    path = next((c for c in candidates if c.is_file()), None)
    if path is None:
        return
    by_name: dict[str, str] = {}
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("image") and row.get("group"):
                by_name[Path(row["image"]).name] = row["group"]
    for it in ds.items:
        it.group = by_name.get(it.image.name)


def read_sizes(items: Iterable[Item]) -> None:
    """Image width/height from the file header (fast, no decoding)."""
    for it in items:
        if it.width and it.height:
            continue
        try:
            with Image.open(it.image) as im:
                it.width, it.height = im.size
        except (OSError, ValueError):
            it.problems.append("image missing or unreadable")
