"""Imported datasets: a manifest in the project, a fingerprint, a cached health report.

Importing does not copy the images by default; it records where they are and a fingerprint of
every file (name, size, modification time, and the label contents). When the files change
afterwards the dataset is flagged as changed. Each training run then freezes its own copy
(hard links, so no extra disk space) of exactly the images it uses.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path
from typing import Any

from ..util import now, read_json, write_json
from ..workspace import Project, safe_name
from . import formats, health
from .formats import Dataset

log = logging.getLogger("model_trainer.datasets")

_CACHE: dict[tuple[str, str], Dataset] = {}


def fingerprint(ds: Dataset) -> str:
    h = hashlib.sha1()
    for it in sorted(ds.items, key=lambda i: str(i.image)):
        try:
            st = it.image.stat()
            h.update(f"{it.split}|{it.image.name}|{st.st_size}|{int(st.st_mtime)}".encode())
        except OSError:
            h.update(f"{it.split}|{it.image.name}|missing".encode())
        for o in it.objects:
            h.update(f"{o.cls}{o.box}{o.polygon}{o.keypoints}{o.obb}".encode())
    h.update(repr(sorted(ds.names.items())).encode())
    return h.hexdigest()


def dataset_dir(project: Project, name: str) -> Path:
    return project.datasets_dir / safe_name(name)


def import_dataset(
    project: Project,
    path: str | Path,
    name: str | None = None,
    label_kind: str | None = None,
    copy: bool = False,
) -> dict[str, Any]:
    """Register a dataset folder (or yaml/json) in the project. Returns the manifest."""
    path = Path(path).expanduser().resolve()
    ds = formats.load(path, label_kind=label_kind)
    name = safe_name(name or ds.name)
    d = dataset_dir(project, name)
    if (d / "dataset.json").is_file():
        raise FileExistsError(f"a dataset called {name!r} already exists in project {project.name}")
    d.mkdir(parents=True, exist_ok=True)
    source = path
    if copy:
        src_root = path if path.is_dir() else path.parent
        dst = d / "files"
        shutil.copytree(src_root, dst, copy_function=_link_copy, dirs_exist_ok=True)
        source = dst / path.name if path.is_file() else dst
        ds = formats.load(source, label_kind=label_kind)
    manifest = {
        "name": name,
        "source": str(source),
        "label_kind": ds.label_kind,
        "forced_kind": label_kind,
        "format": ds.format,
        "copied": copy,
        "imported": now(),
        "fingerprint": fingerprint(ds),
        "classes": {int(k): v for k, v in ds.names.items()},
        "images": ds.split_counts(),
        "tasks": ds.tasks(),
        "default_task": ds.default_task(),
        "kpt_shape": ds.kpt_shape,
    }
    write_json(d / "dataset.json", manifest)
    _CACHE[(str(d), manifest["fingerprint"])] = ds
    log.info(
        "imported %s: %s images, %d classes", name, sum(manifest["images"].values()), len(ds.names)
    )
    return manifest


def _link_copy(src: str, dst: str) -> None:
    from ..util import link_or_copy

    link_or_copy(src, dst)


def manifest(project: Project, name: str) -> dict[str, Any]:
    return read_json(dataset_dir(project, name) / "dataset.json")


def source_missing(m: dict[str, Any]) -> bool:
    """The files the dataset was imported from are gone (moved, deleted, or being rebuilt)."""
    return not Path(m["source"]).exists()


def open_dataset(project: Project, name: str) -> tuple[Dataset, dict[str, Any]]:
    """The dataset (cached in memory) and its manifest; manifest['changed'] if files moved on."""
    d = dataset_dir(project, name)
    m = read_json(d / "dataset.json")
    key = (str(d), m["fingerprint"])
    ds = _CACHE.get(key)
    if ds is None:
        ds = formats.load(m["source"], name=name, label_kind=m.get("forced_kind"))
        fp = fingerprint(ds)
        m["changed"] = fp != m["fingerprint"]
        if not m["changed"]:
            _CACHE[key] = ds
    else:
        m["changed"] = False
    ds.name = name
    return ds, m


def refresh(project: Project, name: str) -> dict[str, Any]:
    """Accept the current files as the dataset (after editing labels): new fingerprint."""
    d = dataset_dir(project, name)
    m = read_json(d / "dataset.json")
    ds = formats.load(m["source"], name=name, label_kind=m.get("forced_kind"))
    m.update(
        fingerprint=fingerprint(ds),
        classes={int(k): v for k, v in ds.names.items()},
        images=ds.split_counts(),
        refreshed=now(),
    )
    write_json(d / "dataset.json", m)
    for f in d.glob("health_*.json"):
        f.unlink()
    return m


def health_report(
    project: Project, name: str, imgsz: int = 640, force: bool = False
) -> dict[str, Any]:
    """Health report for one training size, cached until the dataset changes."""
    ds, m = open_dataset(project, name)
    path = dataset_dir(project, name) / f"health_{imgsz}.json"
    cached = read_json(path, {}) if path.is_file() and not force else {}
    if cached.get("fingerprint") == m["fingerprint"] and not m.get("changed"):
        return cached
    rep = health.analyse(ds, imgsz).to_dict()
    rep["fingerprint"] = m["fingerprint"]
    rep["created"] = now()
    rep["recommended_imgsz"] = health.recommend_imgsz(rep["stats"])
    write_json(path, rep)
    return rep


def delete_dataset(project: Project, name: str) -> None:
    shutil.rmtree(dataset_dir(project, name), ignore_errors=True)
