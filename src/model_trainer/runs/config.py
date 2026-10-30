"""A run's settings (run.yaml): everything needed to reproduce it."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .. import params as P
from ..util import read_yaml, write_yaml


@dataclass
class ModelChoice:
    family: str = "yolo26"
    size: str = "n"
    start: str = "pretrained"  # pretrained | scratch | run | file
    source: str = ""  # run id (start=run) or a weights path (start=file)


@dataclass
class RunOptions:
    snapshot_every: int = 10  # also keep a copy of the weights every N epochs (0 = off)
    keep_snapshots: int = 5
    device: str = ""  # "" = first GPU if there is one, else CPU; "cpu"; "0"; "0,1"
    watch_images: int = 8  # validation images shown with predictions after each epoch
    notify: bool = True
    executor: str = "local"  # local | ssh:<remote name>


@dataclass
class RunConfig:
    name: str
    project: str
    dataset: str
    task: str
    model: ModelChoice = field(default_factory=ModelChoice)
    classes: dict[str, Any] = field(default_factory=dict)  # ClassPlan: drop / merge / rename
    split: dict[str, Any] = field(default_factory=lambda: {"mode": "keep"})  # SplitPlan
    subset: float = 1.0
    balance: str = "none"  # none | oversample
    aug_preset: str = "standard"
    train: dict[str, Any] = field(default_factory=dict)  # Ultralytics arguments (see params.py)
    options: RunOptions = field(default_factory=RunOptions)
    notes: str = ""

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> RunConfig:
        d = dict(d)
        model = ModelChoice(**(d.pop("model", None) or {}))
        options = RunOptions(**(d.pop("options", None) or {}))
        known = set(cls.__dataclass_fields__)
        return cls(model=model, options=options, **{k: v for k, v in d.items() if k in known})

    @classmethod
    def load(cls, path: Path) -> RunConfig:
        return cls.from_dict(read_yaml(path))

    def save(self, path: Path) -> None:
        write_yaml(path, self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def train_args(self) -> dict[str, Any]:
        """Ultralytics arguments: defaults < augmentation preset < explicit values."""
        args = P.defaults(self.task)
        args.update({k: v for k, v in P.AUG_PRESETS.get(self.aug_preset, {}).items() if k in args})
        clean, _ = P.validate(self.train, self.task)
        args.update(clean)
        return args

    def problems(self) -> list[str]:
        from ..models import FAMILIES

        out = []
        if self.model.family not in FAMILIES:
            out.append(f"unknown model family {self.model.family!r}")
        elif self.task not in FAMILIES[self.model.family].tasks:
            out.append(f"{FAMILIES[self.model.family].label} cannot do {self.task}")
        elif self.model.size not in FAMILIES[self.model.family].sizes:
            out.append(f"{FAMILIES[self.model.family].label} has no size {self.model.size!r}")
        if self.model.start in ("run", "file") and not self.model.source:
            out.append("choose the run or weights file to start from")
        _, p = P.validate(self.train, self.task)
        return out + p
