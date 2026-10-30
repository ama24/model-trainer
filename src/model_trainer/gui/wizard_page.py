"""New run wizard: dataset & classes -> base model -> data -> augmentation -> training -> review."""

from __future__ import annotations

import copy
from typing import Any

import yaml
from nicegui import run, ui

from .. import models as M
from .. import params as P
from ..datasets import store
from ..datasets.formats import TASK_LABELS
from ..runs.config import RunConfig
from ..runs.manager import Manager
from ..runs.store import Run
from ..util import free_disk_gb, human_time, read_yaml, write_yaml
from .common import empty, frame, help_icon, param_input, project, simple_mode, ws

TASK_HELP = {
    "detect": "Find objects and draw a box around each one.",
    "segment": "Find objects and outline their exact shape (a mask per object).",
    "pose": "Find objects and locate their keypoints (e.g. joints, corners, tips).",
    "obb": "Find objects with rotated boxes that hug long or tilted objects.",
    "classify": "Say what the whole image shows (one label per image). With box labels, each object is cropped out first.",
}
START_HELP = {
    "pretrained": "Start from weights trained on millions of images (COCO / ImageNet). Almost always the best choice.",
    "scratch": "Random weights. Only for very large datasets or very unusual images; needs many more epochs.",
    "run": "Continue from one of your earlier runs (fine-tuning): e.g. synthetic first, then real photos.",
    "file": "Start from a .pt weights file on disk.",
}


def wizard_page(
    dataset: str | None = None, from_run: str | None = None, template: str | None = None
) -> None:
    with frame("New run"):
        proj = project()
        # datasets whose files are gone cannot be trained on (the datasets page explains why)
        names = [n for n in proj.datasets() if not store.source_missing(store.manifest(proj, n))]
        if not names:
            empty("Import a dataset first.", ("Go to datasets", "/datasets"))
            return
        Wizard(names, dataset, from_run, template).build()


class Wizard:
    def __init__(
        self, datasets: list[str], dataset: str | None, from_run: str | None, template: str | None
    ) -> None:
        self.datasets = datasets
        self.simple = simple_mode()
        cfg: RunConfig | None = None
        if from_run and (p := ws().find_run(from_run)):
            cfg = Run(p).config
            cfg.name = f"{cfg.name}-2"
        elif template:
            path = project().templates_dir / f"{template}.yaml"
            if path.is_file():
                cfg = RunConfig.from_dict(read_yaml(path))
        if cfg is None:
            ds = dataset if dataset in datasets else datasets[0]
            m = store.manifest(project(), ds)
            cfg = RunConfig(name=ds, project=project().name, dataset=ds, task=m["default_task"])
        self.cfg = cfg
        self.vram = M.gpu_memory_gb()
        self.load_dataset_info()

    # ------------------------------------------------------------------ data helpers
    def load_dataset_info(self) -> None:
        self.manifest = store.manifest(project(), self.cfg.dataset)
        self.classes: dict[int, str] = {int(k): v for k, v in self.manifest["classes"].items()}
        imgsz = int(self.cfg.train.get("imgsz", 640))
        path = store.dataset_dir(project(), self.cfg.dataset) / f"health_{imgsz}.json"
        self.health = read_yaml(path) if path.is_file() else None  # json is valid yaml

    @property
    def args(self) -> dict[str, Any]:
        return self.cfg.train_args()

    def set_train(self, key: str, value: Any) -> None:
        self.cfg.train[key] = value
        self.refresh_estimates()

    # ------------------------------------------------------------------------- layout
    def build(self) -> None:
        with ui.row().classes("items-center w-full"):
            ui.label("New training run").classes("text-2xl font-bold")
            ui.space()
            tpls = project().templates()
            if tpls:
                ui.select(
                    tpls,
                    label="Load template",
                    on_change=lambda e: ui.navigate.to(f"/new?template={e.value}"),
                ).classes("w-56")
        if self.simple:
            ui.label(
                "Simple mode: only the essential settings are shown, with recommended values. "
                "Switch on 'Advanced' (top right) to see everything."
            ).classes("text-sm text-gray-600")
        with ui.stepper().props("vertical animated header-nav").classes("w-full") as self.stepper:
            with ui.step("Dataset & classes", icon="photo_library"):
                self.step_dataset()
                self.nav(first=True)
            with ui.step("Base model", icon="hub"):
                self.step_model()
                self.nav()
            with ui.step("Data preparation", icon="aspect_ratio"):
                self.step_data()
                self.nav()
            with ui.step("Augmentation", icon="auto_fix_high"):
                self.step_augment()
                self.nav()
            with ui.step("Training", icon="fitness_center"):
                self.step_train()
                self.nav()
            with ui.step("Review & start", icon="rocket_launch"):
                self.step_review()

    def nav(self, first: bool = False) -> None:
        with ui.stepper_navigation():
            ui.button("Next", on_click=self.next_step).mark("next")
            if not first:
                ui.button("Back", on_click=self.stepper.previous).props("flat")

    def next_step(self) -> None:
        self.stepper.next()
        self.refresh_review()

    # ----------------------------------------------------------------- 1 dataset
    def step_dataset(self) -> None:
        with ui.row().classes("items-center"):
            ui.select(
                self.datasets,
                value=self.cfg.dataset,
                label="Dataset",
                on_change=lambda e: self.change_dataset(e.value),
            ).classes("w-64").mark("dataset")
            ui.button(
                "Open health report",
                icon="health_and_safety",
                on_click=lambda: ui.navigate.to(f"/dataset/{self.cfg.dataset}", new_tab=True),
            ).props("flat")
        self.task_box = ui.column()
        self.class_box = ui.column().classes("w-full")
        self.render_task()
        self.render_classes()
        with ui.expansion(
            "Balance, splits and quick tests", icon="balance", value=not self.simple
        ).classes("w-full"):
            with ui.row().classes("items-center"):
                ui.radio(
                    {"none": "Use images as they are", "oversample": "Oversample rare classes"},
                    value=self.cfg.balance,
                    on_change=lambda e: setattr(self.cfg, "balance", e.value),
                ).props("inline")
                help_icon(
                    "Oversampling lists training images of rare classes several times per epoch (up to 4x), so the model sees them "
                    "more often. The validation set is never changed. Alternative (Advanced > Training): class weighting."
                )
            split = self.cfg.split
            with ui.row().classes("items-center"):
                ui.radio(
                    {"keep": "Keep the dataset's train/val split", "resplit": "Make a new split"},
                    value=split.get("mode", "keep"),
                    on_change=lambda e: (
                        split.update(mode=e.value),
                        self.split_box.set_visibility(e.value == "resplit"),
                    ),
                ).props("inline")
                help_icon(
                    "A new split is stratified (every class keeps a similar share in each part) and keeps images of one group "
                    "(groups.csv) together, so validation never contains near-copies of training images."
                )
            with ui.row().classes("items-center") as self.split_box:
                ratios = split.setdefault("ratios", {"train": 0.85, "val": 0.15})
                ui.number(
                    "Validation share",
                    value=ratios.get("val", 0.15),
                    min=0.05,
                    max=0.5,
                    step=0.05,
                    on_change=lambda e: ratios.update(
                        val=float(e.value or 0.15),
                        train=round(1 - float(e.value or 0.15) - ratios.get("test", 0), 3),
                    ),
                ).classes("w-40")
                ui.number(
                    "Test share",
                    value=ratios.get("test", 0.0),
                    min=0,
                    max=0.3,
                    step=0.05,
                    on_change=lambda e: ratios.update(
                        test=float(e.value or 0),
                        train=round(1 - ratios.get("val", 0.15) - float(e.value or 0), 3),
                    ),
                ).classes("w-40")
                ui.number(
                    "Seed",
                    value=split.get("seed", 0),
                    step=1,
                    on_change=lambda e: split.update(seed=int(e.value or 0)),
                ).classes("w-28")
            self.split_box.set_visibility(split.get("mode") == "resplit")
            with ui.row().classes("items-center"):
                ui.label("Use this share of the training images:")
                ui.slider(
                    min=0.05,
                    max=1.0,
                    step=0.05,
                    value=self.cfg.subset,
                    on_change=lambda e: (
                        setattr(self.cfg, "subset", float(e.value)),
                        self.refresh_estimates(),
                    ),
                ).classes("w-64").props("label-always")
                help_icon(
                    "Quick test: train on a fraction (whole groups) to check the settings in minutes before the full run."
                )

    def change_dataset(self, name: str) -> None:
        self.cfg.dataset = name
        self.cfg.classes = {}
        self.load_dataset_info()
        if self.cfg.task not in self.manifest["tasks"] or self.manifest["tasks"][self.cfg.task]:
            self.cfg.task = self.manifest["default_task"]
        if self.cfg.name in self.datasets:
            self.cfg.name = name
        self.render_task()
        self.render_classes()
        self.refresh_model_table()

    def render_task(self) -> None:
        self.task_box.clear()
        with self.task_box:
            ui.label("What should the model do?").classes("font-medium")
            opts = {}
            for t, why in self.manifest["tasks"].items():
                opts[t] = TASK_LABELS[t] + (f" - not possible: {why}" if why else "")
            ui.radio(opts, value=self.cfg.task, on_change=lambda e: self.change_task(e.value)).mark(
                "task"
            )
            self.task_help = ui.label(TASK_HELP[self.cfg.task]).classes("text-sm text-gray-600")

    def change_task(self, task: str) -> None:
        why = self.manifest["tasks"].get(task)
        if why:
            ui.notify(
                f"{TASK_LABELS[task]} is not possible with this dataset: {why}", type="warning"
            )
            self.render_task()
            return
        self.cfg.task = task
        self.task_help.text = TASK_HELP[task]
        fam = M.FAMILIES[self.cfg.model.family]
        if task not in fam.tasks:
            self.cfg.model.family = "yolo26"
        self.refresh_model_table()

    def class_counts(self) -> dict[int, tuple[int, int]]:
        """class id -> (objects, images), counted over all splits."""
        ds, _ = store.open_dataset(project(), self.cfg.dataset)
        objs: dict[int, int] = {}
        imgs: dict[int, int] = {}
        for it in ds.items:
            for c in {o.cls for o in it.objects}:
                imgs[c] = imgs.get(c, 0) + 1
            for o in it.objects:
                objs[o.cls] = objs.get(o.cls, 0) + 1
        return {i: (objs.get(i, 0), imgs.get(i, 0)) for i in self.classes}

    def render_classes(self) -> None:
        self.class_box.clear()
        self.counts = self.class_counts()
        with self.class_box:
            with ui.row().classes("items-center w-full"):
                ui.label(f"Classes ({len(self.classes)})").classes("font-medium")
                help_icon(
                    "Keep, drop, merge or rename each class.\n"
                    "- drop: its objects become background (or their images are removed, see below)\n"
                    "- merge into: joins this class with another one (e.g. all sizes of one part)\n"
                    "- rename: gives the class a new name\n"
                    "The mapping is saved with the model."
                )
                ui.space()
                self.class_summary = ui.label().classes("text-sm text-gray-600")
            with ui.row().classes("items-center w-full gap-2"):
                ui.input(
                    placeholder="Search classes", on_change=lambda e: self.filter_classes(e.value)
                ).props("dense clearable").classes("w-56")
                ui.button("Keep all", on_click=self.keep_all).props("flat dense")
                min_n = ui.number(value=50, min=1, step=10).props("dense").classes("w-20")
                ui.button(
                    "Drop classes with fewer objects",
                    on_click=lambda: self.drop_rare(int(min_n.value or 1)),
                ).props("flat dense").tooltip(
                    "Drop every class with fewer labelled objects than this"
                )
            header = "text-xs font-bold text-gray-600"
            with ui.row().classes("w-full no-wrap items-center gap-2 px-2"):
                ui.label("Class").classes(f"{header} w-56")
                ui.label("Objects").classes(f"{header} w-20 text-right")
                ui.label("Images").classes(f"{header} w-20 text-right")
                ui.label("Action").classes(f"{header} w-40")
                ui.label("Merge into / new name").classes(f"{header} w-56")
            self.class_rows: dict[str, ui.row] = {}
            with (
                ui.scroll_area()
                .classes("w-full border rounded")
                .style(f"height: {min(420, 44 * len(self.classes) + 10)}px")
            ):
                for i, n in self.classes.items():
                    self.class_row(i, n)
            if not self.simple:
                plan = self.cfg.classes
                ui.radio(
                    {
                        "background": "Dropped objects become background",
                        "remove": "Remove images that contain dropped classes",
                    },
                    value=plan.get("dropped_images", "background"),
                    on_change=lambda e: plan.update(dropped_images=e.value),
                ).props("inline")
        self.update_class_summary()

    def class_row(self, i: int, n: str) -> None:
        plan = self.cfg.classes
        merge, rename = dict(plan.get("merge") or {}), dict(plan.get("rename") or {})
        action = (
            "drop"
            if n in (plan.get("drop") or [])
            else "merge"
            if n in merge
            else "rename"
            if n in rename
            else "keep"
        )
        objs, imgs = self.counts.get(i, (0, 0))
        colour = "text-red-700" if objs == 0 else "text-orange-700" if objs < 50 else ""
        others = [m for m in self.classes.values() if m != n]
        with ui.row().classes("w-full no-wrap items-center gap-2 px-2 py-1 border-b") as row:
            ui.label(n).classes("w-56 truncate").tooltip(n)
            ui.label(str(objs)).classes(f"w-20 text-right {colour}")
            ui.label(str(imgs)).classes("w-20 text-right")
            sel = (
                ui.select(
                    {
                        "keep": "keep",
                        "drop": "drop",
                        "merge": "merge into...",
                        "rename": "rename to...",
                    },
                    value=action,
                )
                .props("dense outlined")
                .classes("w-40")
            )
            target = (
                ui.select(
                    others,
                    value=merge.get(n) or rename.get(n),
                    new_value_mode="add-unique",
                    with_input=True,
                )
                .props("dense outlined")
                .classes("w-56")
            )  # an existing class, or type a new name
            target.set_enabled(action in ("merge", "rename"))

            def on_action(e: Any, n: str = n, t: Any = target) -> None:
                t.set_enabled(e.value in ("merge", "rename"))
                self.class_action(n, e.value, t.value or "")

            sel.on_value_change(on_action)
            target.on_value_change(
                lambda e, n=n, s=sel: self.class_action(n, s.value, e.value or "")
            )
        self.class_rows[n] = row

    def filter_classes(self, text: str | None) -> None:
        q = (text or "").lower()
        for n, row in self.class_rows.items():
            row.set_visibility(q in n.lower())

    def keep_all(self) -> None:
        self.cfg.classes.update(drop=[], merge={}, rename={})
        self.render_classes()

    def drop_rare(self, min_objects: int) -> None:
        rare = [n for i, n in self.classes.items() if self.counts.get(i, (0, 0))[0] < min_objects]
        plan = self.cfg.classes
        plan["drop"] = sorted(set(plan.get("drop") or []) | set(rare))
        ui.notify(
            f"{len(rare)} class(es) with fewer than {min_objects} objects dropped", type="info"
        )
        self.render_classes()

    def update_class_summary(self) -> None:
        if hasattr(self, "class_summary"):
            self.class_summary.text = self._class_summary()

    def class_action(self, name: str, action: str, target: str) -> None:
        plan = self.cfg.classes
        drop = [n for n in plan.get("drop") or [] if n != name]
        merge = {k: v for k, v in (plan.get("merge") or {}).items() if k != name}
        rename = {k: v for k, v in (plan.get("rename") or {}).items() if k != name}
        if action == "drop":
            drop.append(name)
        elif action == "merge" and target:
            merge[name] = target
        elif action == "rename" and target:
            rename[name] = target
        plan.update(drop=drop, merge=merge, rename=rename)
        self.update_class_summary()

    # ------------------------------------------------------------------- 2 model
    def step_model(self) -> None:
        m = self.cfg.model
        n_train = self.manifest["images"].get("train", 0)
        fam, size, why = M.recommend(n_train, self.cfg.task, self.vram, int(self.args["imgsz"]))
        with ui.row().classes("items-center"):
            ui.icon("lightbulb", color="amber").classes("text-2xl")
            ui.label(f"Recommended: {M.FAMILIES[fam].label} {M.SIZE_LABELS[size]}. {why}").classes(
                "text-sm"
            )
            ui.button("Use it", on_click=lambda: self.pick_model(fam, size)).props("flat dense")
        with ui.row().classes("items-center"):
            self.fam_sel = (
                ui.select(
                    {k: f.label for k, f in M.FAMILIES.items()},
                    value=m.family,
                    label="Family",
                    on_change=lambda e: self.pick_model(e.value, None),
                )
                .classes("w-56")
                .mark("family")
            )
            self.size_sel = ui.toggle(
                {s: M.SIZE_LABELS[s] for s in M.FAMILIES[m.family].sizes},
                value=m.size,
                on_change=lambda e: self.pick_model(None, e.value),
            ).mark("size")
        self.fam_note = ui.label(M.FAMILIES[m.family].note).classes("text-sm text-gray-600")
        self.model_table = (
            ui.table(
                columns=[
                    {"name": "model", "label": "Model", "field": "model", "align": "left"},
                    {"name": "params", "label": "Parameters", "field": "params"},
                    {"name": "coco", "label": "COCO mAP", "field": "coco"},
                    {"name": "vram", "label": "GPU memory (est.)", "field": "vram"},
                    {"name": "fits", "label": "Fits?", "field": "fits"},
                    {"name": "epoch", "label": "Time / epoch (est.)", "field": "epoch"},
                ],
                rows=[],
                row_key="model",
            )
            .props("dense flat bordered")
            .classes("w-full")
        )
        ui.label(
            "Estimates are rough: they assume the batch size shown in step 5 (auto = 16 for the estimate) and your GPU. "
            "COCO mAP is the published score on the COCO benchmark (80 everyday classes), a guide to the model's capacity."
        ).classes("text-xs text-gray-500")
        ui.label("Start from").classes("font-medium mt-2")
        ui.radio(
            {k: k.capitalize() if k != "run" else "One of my runs" for k in START_HELP},
            value=m.start,
            on_change=lambda e: (
                setattr(m, "start", e.value),
                self.start_help.set_text(START_HELP[e.value]),
                self.render_source(),
            ),
        ).props("inline")
        self.start_help = ui.label(START_HELP[m.start]).classes("text-sm text-gray-600")
        self.source_box = ui.row()
        self.render_source()
        self.refresh_model_table()

    def render_source(self) -> None:
        self.source_box.clear()
        m = self.cfg.model
        with self.source_box:
            if m.start == "run":
                runs = {p.name: f"{p.name}" for p in project().runs() if Run(p).best_weights}
                ui.select(
                    runs,
                    value=m.source if m.source in runs else None,
                    label="Run",
                    on_change=lambda e: setattr(m, "source", e.value),
                ).classes("w-96")
            elif m.start == "file":
                ui.input(
                    "Weights file (.pt)",
                    value=m.source,
                    on_change=lambda e: setattr(m, "source", e.value.strip().strip('"')),
                ).classes("w-96")

    def pick_model(self, family: str | None, size: str | None) -> None:
        m = self.cfg.model
        if family:
            if self.cfg.task not in M.FAMILIES[family].tasks:
                ui.notify(
                    f"{M.FAMILIES[family].label} cannot do {TASK_LABELS[self.cfg.task]}",
                    type="warning",
                )
                self.fam_sel.value = m.family
                return
            m.family = family
            sizes = M.FAMILIES[family].sizes
            self.size_sel.options = {s: M.SIZE_LABELS[s] for s in sizes}
            if m.size not in sizes:
                m.size = sizes[0]
            self.size_sel.value = m.size
            self.size_sel.update()
            self.fam_sel.value = family
            self.fam_note.text = M.FAMILIES[family].note
        if size:
            m.size = size
            self.size_sel.value = size
        self.refresh_model_table()

    def refresh_model_table(self) -> None:
        if not hasattr(self, "model_table"):
            return
        a = self.args
        imgsz, batch = int(a["imgsz"]), int(a["batch"]) if int(a["batch"]) > 0 else 16
        n = int(self.manifest["images"].get("train", 0) * self.cfg.subset)
        rows = []
        for o in M.options(self.cfg.task):
            if o["family"] != self.cfg.model.family:
                continue
            v = M.estimate_vram_gb(o["family"], o["size"], self.cfg.task, imgsz, batch)
            fits = (
                "?"
                if self.vram is None
                else (
                    "yes"
                    if v <= self.vram * 0.9
                    else f"no - try batch {M.max_batch(o['family'], o['size'], self.cfg.task, imgsz, self.vram)}"
                )
            )
            rows.append(
                {
                    "model": f"{o['weights'].removesuffix('.pt')}"
                    + ("  <- selected" if o["size"] == self.cfg.model.size else ""),
                    "params": f"{o['params_m']:.1f} M",
                    "coco": f"{o['coco_map']:.1f}",
                    "vram": f"{v:.1f} GB",
                    "fits": fits,
                    "epoch": human_time(
                        M.estimate_epoch_seconds(
                            o["family"],
                            o["size"],
                            self.cfg.task,
                            imgsz,
                            n,
                            ws().settings().get("gpu_tflops"),
                        )
                    ),
                }
            )
        self.model_table.rows = rows
        self.model_table.update()

    # -------------------------------------------------------------------- 3 data
    def step_data(self) -> None:
        rec = self.health.get("recommended_imgsz") if self.health else None
        if rec:
            with ui.row().classes("items-center"):
                ui.icon("lightbulb", color="amber").classes("text-2xl")
                ui.label(
                    f"The health report recommends {rec} px so that small objects stay visible."
                ).classes("text-sm")
                ui.button(f"Use {rec}", on_click=lambda: self.imgsz_input.set_value(rec)).props(
                    "flat dense"
                )
        p = P.BY_KEY["imgsz"]
        with ui.row().classes("items-center"):
            self.imgsz_input = (
                ui.number(
                    p.label,
                    value=self.args["imgsz"],
                    min=32,
                    max=2048,
                    step=32,
                    on_change=lambda e: self.set_train("imgsz", int(e.value or 640)),
                )
                .classes("w-48")
                .mark("imgsz")
            )
            help_icon(p.help())
        self.size_warn = ui.label().classes("text-sm text-warning")
        if not self.simple:
            for key in ("rect", "cache", "workers", "fraction"):
                q = P.BY_KEY[key]
                if P.applies(q, self.cfg.task):
                    param_input(q, self.args[key], self.set_train)

    # ----------------------------------------------------------------- 4 augment
    def step_augment(self) -> None:
        ui.markdown(
            "Augmentation shows the model slightly changed copies of the training images (flipped, rotated, recoloured, "
            "zoomed...) so it learns what matters and ignores what does not. Preview the result on the dataset page "
            "(**What the model sees**)."
        )
        with ui.row().classes("items-center"):
            ui.toggle(
                list(P.AUG_PRESETS),
                value=self.cfg.aug_preset,
                on_change=lambda e: self.pick_preset(e.value),
            ).mark("aug")
            self.preset_help = ui.label(P.AUG_PRESET_HELP[self.cfg.aug_preset]).classes(
                "text-sm text-gray-600"
            )
        ui.button(
            "Preview on the dataset",
            icon="visibility",
            on_click=lambda: ui.navigate.to(f"/dataset/{self.cfg.dataset}", new_tab=True),
        ).props("flat")
        self.aug_box = ui.column().classes("w-full")
        self.render_aug()

    def pick_preset(self, name: str) -> None:
        self.cfg.aug_preset = name
        for k in P.AUG_KEYS:
            self.cfg.train.pop(k, None)
        self.preset_help.text = P.AUG_PRESET_HELP[name]
        self.render_aug()

    def render_aug(self) -> None:
        self.aug_box.clear()
        a = self.args
        preset = P.AUG_PRESETS.get(self.cfg.aug_preset, {})
        keys = [k for k in P.AUG_KEYS if P.applies(P.BY_KEY[k], self.cfg.task)]
        if self.simple:
            keys = [k for k in keys if k in ("fliplr", "flipud", "degrees", "hsv_h")]
        with self.aug_box:
            if self.simple:
                ui.label("Most useful switches (all others follow the preset):").classes("text-sm")
            with ui.grid(columns=2).classes("w-full"):
                for k in keys:
                    param_input(
                        P.BY_KEY[k],
                        a[k],
                        self.set_train,
                        default=preset.get(k, P.BY_KEY[k].default),
                    )
            ui.label(
                "Hue and saturation change colours: set them to 0 if colour is what tells your classes apart."
            ).classes("text-xs text-gray-500")

    # ----------------------------------------------------------------- 5 training
    def step_train(self) -> None:
        a = self.args
        simple_keys = ["epochs", "patience", "batch"]
        for k in simple_keys:
            param_input(P.BY_KEY[k], a[k], self.set_train)
        if self.simple:
            return
        for group in ("train", "loss"):
            with ui.expansion(P.GROUPS[group], icon="tune", value=group == "train").classes(
                "w-full"
            ):
                with ui.grid(columns=2).classes("w-full"):
                    for p in P.PARAMS:
                        if (
                            p.group == group
                            and p.key not in simple_keys
                            and P.applies(p, self.cfg.task)
                        ):
                            param_input(p, a[p.key], self.set_train)
        with ui.expansion("Raw settings (YAML)", icon="code").classes("w-full"):
            ui.label(
                "Every Ultralytics training argument can be set here; it overrides the fields above."
            ).classes("text-sm")
            raw = ui.codemirror(
                yaml.safe_dump(dict(self.cfg.train), sort_keys=True), language="YAML"
            ).classes("w-full h-48")

            def apply_raw() -> None:
                try:
                    data = yaml.safe_load(raw.value) or {}
                    assert isinstance(data, dict)
                except Exception as exc:  # noqa: BLE001
                    ui.notify(f"Not valid YAML: {exc}", type="negative")
                    return
                self.cfg.train = data
                ui.notify("Applied", type="positive")
                self.refresh_estimates()

            ui.button("Apply YAML", on_click=apply_raw).props("flat")

    # -------------------------------------------------------------------- 6 review
    def step_review(self) -> None:
        with ui.row().classes("items-center w-full"):
            ui.input(
                "Run name",
                value=self.cfg.name,
                on_change=lambda e: setattr(self.cfg, "name", e.value),
            ).classes("w-72").mark("name")
            ui.input(
                "Notes (what are you trying?)",
                value=self.cfg.notes,
                on_change=lambda e: setattr(self.cfg, "notes", e.value),
            ).classes("w-96")
        self.review = ui.column().classes("w-full")
        o = self.cfg.options
        with ui.expansion("Run options", icon="settings", value=not self.simple).classes("w-full"):
            with ui.grid(columns=2).classes("w-full"):
                with ui.row().classes("items-center"):
                    ui.number(
                        "Save a copy of the weights every N epochs",
                        value=o.snapshot_every,
                        min=0,
                        step=1,
                        on_change=lambda e: setattr(o, "snapshot_every", int(e.value or 0)),
                    ).classes("w-72")
                    help_icon(
                        "Snapshots let you go back to an earlier epoch if training goes wrong late. 0 = off. "
                        "(The latest epoch is always saved for resuming, and the best epoch is always kept.)"
                    )
                ui.number(
                    "Keep the last N snapshots",
                    value=o.keep_snapshots,
                    min=1,
                    step=1,
                    on_change=lambda e: setattr(o, "keep_snapshots", int(e.value or 5)),
                ).classes("w-72")
                with ui.row().classes("items-center"):
                    ui.number(
                        "'Watch it learn' images",
                        value=o.watch_images,
                        min=0,
                        max=24,
                        step=1,
                        on_change=lambda e: setattr(o, "watch_images", int(e.value or 0)),
                    ).classes("w-72")
                    help_icon(
                        "After every epoch the same validation images are predicted, so you can watch the model improve. 0 = off."
                    )
                ui.switch(
                    "Phone notifications for this run",
                    value=o.notify,
                    on_change=lambda e: setattr(o, "notify", e.value),
                )
                with ui.row().classes("items-center"):
                    ui.input(
                        "Device",
                        value=o.device,
                        placeholder="auto",
                        on_change=lambda e: setattr(o, "device", e.value.strip()),
                    ).classes("w-40")
                    help_icon(
                        "Empty = the first GPU (or the CPU without one). 'cpu', '0', '1', or '0,1' for two GPUs."
                    )
                remotes = {"local": "This computer"} | {
                    f"ssh:{r['name']}": f"{r['name']} (SSH)"
                    for r in ws().settings().get("remotes") or []
                }
                ui.select(
                    remotes,
                    value=o.executor if o.executor in remotes else "local",
                    label="Train on",
                    on_change=lambda e: setattr(o, "executor", e.value),
                ).classes("w-56")
        with ui.row():
            ui.button(
                "Start now", icon="play_arrow", on_click=lambda: self.create(start=True)
            ).props("color=positive").mark("start")
            ui.button("Add to queue", icon="queue", on_click=lambda: self.create(start=True)).props(
                "outline"
            ).tooltip("Runs train one after another. If nothing is training, it starts right away.")
            ui.button(
                "Save without starting", icon="save", on_click=lambda: self.create(start=False)
            ).props("flat")
            ui.button("Save as template", icon="bookmark", on_click=self.save_template).props(
                "flat"
            )
        self.refresh_review()

    def _cache_summary(self) -> str:
        cache = self.args.get("cache")
        if cache != "auto":
            return {False: "off", "ram": "in RAM", "disk": "on disk"}.get(cache, str(cache))
        n = int(sum(self.manifest["images"].values()) * self.cfg.subset)
        imgsz = int(self.args["imgsz"])
        need, free = M.cache_ram_gb(n, imgsz), M.free_ram_gb()
        where = "in RAM" if M.ram_cache_fits(n, imgsz, free) else "off (images read from disk)"
        free_text = f", {free:.1f} GB free now" if free is not None else ""
        return f"auto: {where} - about {need:.1f} GB{free_text}; decided again when training starts"

    def estimates(self) -> dict[str, Any]:
        a = self.args
        m = self.cfg.model
        n = int(self.manifest["images"].get("train", 0) * self.cfg.subset)
        if self.cfg.balance == "oversample":
            n = int(n * 1.3)
        imgsz = int(a["imgsz"])
        batch = (
            int(a["batch"])
            if int(a["batch"]) > 0
            else (
                M.max_batch(m.family, m.size, self.cfg.task, imgsz, self.vram * 0.6 / 0.85)
                if self.vram
                else 16
            )
        )
        per_epoch = M.estimate_epoch_seconds(
            m.family, m.size, self.cfg.task, imgsz, n, ws().settings().get("gpu_tflops")
        )
        params = M.FAMILIES[m.family].numbers[m.size][0]
        disk = params * 1e6 * 4 * 3 * (3 + self.cfg.options.keep_snapshots) / 1e9
        return {
            "train_images": n,
            "batch": batch,
            "vram": M.estimate_vram_gb(m.family, m.size, self.cfg.task, imgsz, batch),
            "epoch_s": per_epoch,
            "total_s": per_epoch * int(a["epochs"]),
            "disk_gb": disk,
            "free_gb": free_disk_gb(ws().root),
        }

    def refresh_estimates(self) -> None:
        self.refresh_model_table()
        if hasattr(self, "size_warn") and self.health:
            sizes = self.health["stats"].get("object_size_px") or {}
            p10 = sizes.get("p10")
            if p10:
                at = p10 * int(self.args["imgsz"]) / self.health["stats"]["imgsz"]
                self.size_warn.text = (
                    f"At {self.args['imgsz']} px the smallest 10 % of objects are about {at:.0f} px: "
                    + ("too small to detect reliably - use a larger size." if at < 8 else "fine.")
                )
        self.refresh_review()

    def problems(self) -> list[tuple[str, str]]:
        out = [("bad", p) for p in self.cfg.problems()]
        e = self.estimates()
        if self.vram and e["vram"] > self.vram * 0.95:
            out.append(
                (
                    "warn",
                    f"estimated {e['vram']:.1f} GB of GPU memory, the GPU has {self.vram:.1f} GB: lower the batch size or image size "
                    "(training would retry with half the batch automatically)",
                )
            )
        if e["free_gb"] < e["disk_gb"] + 2:
            out.append(("bad", f"only {e['free_gb']:.1f} GB free on the workspace disk"))
        if self.manifest["images"].get("val", 0) == 0 and self.cfg.split.get("mode") != "resplit":
            out.append(
                ("bad", "the dataset has no validation images: choose 'Make a new split' in step 1")
            )
        if self.health:
            for c in self.health["checks"]:
                if c["level"] == "bad":
                    out.append(("warn", f"dataset: {c['title']} - {c['value']}"))
        cols = [k for k in ("hsv_h", "hsv_s") if float(self.args.get(k, 0) or 0) > 0]
        if cols and self.cfg.task != "classify" and not self.simple:
            out.append(
                (
                    "info",
                    "colour augmentation is on: fine unless colour is what separates your classes",
                )
            )
        if self.cfg.model.start == "scratch" and int(self.args["epochs"]) < 200:
            out.append(("info", "training from scratch usually needs 300+ epochs"))
        return out

    def refresh_review(self) -> None:
        if not hasattr(self, "review"):
            return
        self.review.clear()
        a = self.args
        e = self.estimates()
        m = self.cfg.model
        with self.review:
            with ui.card().classes("w-full").props("flat bordered"):
                rows = [
                    ("Dataset", f"{self.cfg.dataset} - {TASK_LABELS[self.cfg.task]}"),
                    ("Classes", self._class_summary()),
                    (
                        "Model",
                        f"{M.FAMILIES[m.family].label} {M.SIZE_LABELS[m.size]} ({m.start}{': ' + m.source if m.source else ''})",
                    ),
                    (
                        "Images",
                        f"{e['train_images']} for training per epoch"
                        + (" (rare classes repeated)" if self.cfg.balance == "oversample" else ""),
                    ),
                    (
                        "Image size / batch",
                        f"{a['imgsz']} px / {a['batch'] if int(a['batch']) > 0 else 'auto (~' + str(e['batch']) + ')'}",
                    ),
                    ("Image cache", self._cache_summary()),
                    (
                        "Epochs",
                        f"{a['epochs']} (stops early after {a['patience']} epochs without improvement)",
                    ),
                    (
                        "Augmentation",
                        self.cfg.aug_preset
                        + (
                            f" + {len(P.changed({k: a[k] for k in P.AUG_KEYS if k in a}, self.cfg.task))} changes"
                            if self.cfg.train
                            else ""
                        ),
                    ),
                    (
                        "Estimated time",
                        f"{human_time(e['epoch_s'])} per epoch, up to {human_time(e['total_s'])} in total",
                    ),
                    (
                        "GPU memory",
                        f"~{e['vram']:.1f} GB" + (f" of {self.vram:.1f} GB" if self.vram else ""),
                    ),
                    (
                        "Disk",
                        f"~{e['disk_gb']:.1f} GB for checkpoints ({e['free_gb']:.0f} GB free)",
                    ),
                ]
                changed = P.changed(a, self.cfg.task)
                if changed:
                    rows.append(
                        ("Changed settings", ", ".join(f"{k}={v}" for k, v in changed.items()))
                    )
                for k, v in rows:
                    with ui.row().classes("no-wrap"):
                        ui.label(k).classes("w-44 font-medium text-gray-700")
                        ui.label(str(v))
            for level, text in self.problems():
                colour = {"bad": "negative", "warn": "warning", "info": "info"}[level]
                with ui.row().classes("items-center no-wrap"):
                    ui.icon(
                        {"bad": "error", "warn": "warning", "info": "info"}[level], color=colour
                    )
                    ui.label(text).classes("text-sm")

    def _class_summary(self) -> str:
        from ..datasets.prepare import ClassPlan

        _, new = ClassPlan.from_dict(self.cfg.classes).mapping(self.classes)
        extra = []
        if self.cfg.classes.get("drop"):
            extra.append(f"dropped {', '.join(self.cfg.classes['drop'])}")
        if self.cfg.classes.get("merge"):
            extra.append(
                "merged " + ", ".join(f"{k}->{v}" for k, v in self.cfg.classes["merge"].items())
            )
        return f"{len(new)} classes" + (f" ({'; '.join(extra)})" if extra else "")

    async def create(self, start: bool) -> None:
        bad = [t for lvl, t in self.problems() if lvl == "bad"]
        if bad:
            ui.notify("Fix first: " + "; ".join(bad), type="negative", multi_line=True)
            return
        mgr = Manager(ws())
        cfg = copy.deepcopy(self.cfg)
        cfg.project = project().name
        try:
            r = mgr.create(project(), cfg)
            if start:
                how = await run.io_bound(mgr.start, r)
                ui.notify(
                    "Training started"
                    if how == "started"
                    else "Another run is training: added to the queue",
                    type="positive",
                )
        except Exception as exc:  # noqa: BLE001 - shown to the user
            ui.notify(f"Could not create the run: {exc}", type="negative", multi_line=True)
            return
        ui.navigate.to(f"/run/{r.id}")

    def save_template(self) -> None:
        with ui.dialog() as d, ui.card():
            name = ui.input("Template name", value=self.cfg.name)

            def save() -> None:
                from ..workspace import safe_name

                write_yaml(
                    project().templates_dir / f"{safe_name(name.value)}.yaml", self.cfg.to_dict()
                )
                ui.notify("Template saved", type="positive")
                d.close()

            ui.button("Save", on_click=save)
        d.open()
