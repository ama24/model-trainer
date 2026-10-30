"""Datasets: import, health report, gallery, 'what the model sees'."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

from nicegui import run, ui

from .. import params as P
from ..datasets import health, render, store
from ..datasets.formats import TASK_LABELS
from .common import (
    LEVEL_COLOUR,
    LEVEL_ICON,
    bar_chart,
    empty,
    frame,
    help_icon,
    project,
    simple_mode,
    ws,
)

FORMAT_HELP = (
    "Supported formats:\n"
    "- YOLO: a folder with data.yaml (or images/ and labels/ folders)\n"
    "- COCO: json files (train.json, val.json) next to the images\n"
    "- Classification: one folder per class (optionally inside train/ and val/)\n"
    "Optional groups.csv (image,group) keeps photos of one scene in the same split."
)


def datasets_page() -> None:
    with frame("Datasets"):
        proj = project()
        with ui.row().classes("items-center"):
            ui.label("Datasets").classes("text-2xl font-bold")
            ui.button("Import dataset", icon="add", on_click=lambda: import_dialog().open()).mark(
                "import"
            )
            help_icon(FORMAT_HELP)
        names = proj.datasets()
        if not names:
            empty("No datasets yet. Import a folder in YOLO, COCO or class-folder format to start.")
            return
        rows = []
        for n in names:
            m = store.manifest(proj, n)
            rows.append(
                {
                    "name": n,
                    "format": m["format"],
                    "labels": m["label_kind"],
                    "classes": len(m["classes"]),
                    "train": m["images"].get("train", 0),
                    "val": m["images"].get("val", 0),
                    "test": m["images"].get("test", 0),
                    "imported": m["imported"][:16].replace("T", " "),
                }
            )
        cols = [
            {"name": k, "label": k.capitalize(), "field": k, "sortable": True, "align": "left"}
            for k in rows[0]
        ]
        t = (
            ui.table(columns=cols, rows=rows, row_key="name")
            .classes("w-full")
            .props("flat bordered")
        )
        t.on("rowClick", lambda e: ui.navigate.to(f"/dataset/{e.args[1]['name']}"))
        ui.label("Click a dataset to see its health report and images.").classes(
            "text-sm text-gray-500"
        )


def import_dialog() -> ui.dialog:
    with ui.dialog() as dlg, ui.card().classes("w-[640px]"):
        ui.label("Import a dataset").classes("text-xl font-bold")
        ui.markdown(
            "Point to a folder on this computer (or a dataset `.yaml` / COCO `.json`), or upload a **zip**. "
            "Nothing is copied unless you ask: the app remembers where the files are and notices if they change."
        )
        path = (
            ui.input("Folder, data.yaml or COCO json", placeholder=r"C:\data\my_dataset")
            .classes("w-full")
            .mark("path")
        )
        name = ui.input("Name (optional)").classes("w-full")
        with ui.row():
            copy = ui.checkbox(
                "Copy into the workspace (hard links: no extra space on the same disk)"
            )
        obb = ui.checkbox("8-number label lines are rotated boxes (OBB), not 4-point polygons")
        status = ui.label().classes("text-sm")

        async def do_import(p: str | None = None) -> None:
            src = (p or path.value or "").strip().strip('"')
            if not src:
                status.text = "Enter a path first."
                return
            status.text = "Reading the dataset..."
            try:
                m = await run.io_bound(
                    store.import_dataset,
                    project(),
                    src,
                    name.value or None,
                    "obb" if obb.value else None,
                    copy.value,
                )
            except Exception as exc:  # noqa: BLE001 - shown to the user
                status.text = f"Could not import: {exc}"
                return
            ui.notify(f"Imported {m['name']}", type="positive")
            dlg.close()
            ui.navigate.to(f"/dataset/{m['name']}")

        async def uploaded(e: Any) -> None:
            dest = ws().root / "uploads" / Path(e.file.name).stem
            dest.mkdir(parents=True, exist_ok=True)
            zpath = dest.with_suffix(".zip")
            await e.file.save(zpath)
            status.text = "Unpacking..."
            await run.io_bound(_unzip, zpath, dest)
            await do_import(str(_dataset_root(dest)))

        with ui.row():
            ui.button("Import", icon="download", on_click=lambda: do_import()).mark("do-import")
            ui.button("Cancel", on_click=dlg.close).props("flat")
        ui.separator()
        ui.upload(label="... or upload a .zip", auto_upload=True, on_upload=uploaded).props(
            "accept=.zip"
        ).classes("w-full")
    return dlg


def _unzip(zpath: Path, dest: Path) -> None:
    with zipfile.ZipFile(zpath) as z:
        for member in z.namelist():  # refuse paths that would escape the folder
            target = (dest / member).resolve()
            if not str(target).startswith(str(dest.resolve())):
                raise ValueError(f"unsafe path in zip: {member}")
        z.extractall(dest)
    zpath.unlink(missing_ok=True)


def _dataset_root(d: Path) -> Path:
    entries = [p for p in d.iterdir() if not p.name.startswith(("__MACOSX", "."))]
    return entries[0] if len(entries) == 1 and entries[0].is_dir() else d


# --------------------------------------------------------------------------- detail page


def dataset_page(name: str) -> None:
    with frame(f"Dataset {name}"):
        proj = project()
        if name not in proj.datasets():
            empty(f"No dataset {name!r} in project {proj.name}.", ("Back to datasets", "/datasets"))
            return
        m = store.manifest(proj, name)
        with ui.row().classes("items-center w-full"):
            ui.label(name).classes("text-2xl font-bold")
            ui.badge(m["format"]).props("outline")
            ui.badge(m["label_kind"]).props("outline")
            ui.space()
            ui.button(
                "Train a model on it",
                icon="play_arrow",
                on_click=lambda: ui.navigate.to(f"/new?dataset={name}"),
            ).props("color=primary")
            ui.button(
                "Re-read files",
                icon="refresh",
                on_click=lambda: (store.refresh(proj, name), ui.navigate.reload()),
            ).props("flat").tooltip(
                "Accept the current files (after you fixed labels) as the dataset"
            )
            ui.button(icon="delete", on_click=lambda: confirm_delete(name)).props(
                "flat color=negative"
            ).tooltip("Remove from the project (files stay where they are)")
        ui.label(f"Source: {m['source']}").classes("text-sm text-gray-500")
        tasks = [
            f"{TASK_LABELS[t]}" + ("" if not why else f" (not possible: {why})")
            for t, why in m["tasks"].items()
        ]
        ui.label("Tasks: " + "; ".join(tasks)).classes("text-sm")
        with ui.tabs().classes("w-full") as tabs:
            t_health = ui.tab("Health report", icon="health_and_safety")
            t_gallery = ui.tab("Gallery", icon="grid_view")
            t_aug = ui.tab("What the model sees", icon="visibility")
        with ui.tab_panels(tabs, value=t_health).classes("w-full"):
            with ui.tab_panel(t_health):
                HealthView(name).build()
            with ui.tab_panel(t_gallery):
                Gallery(name).build()
            with ui.tab_panel(t_aug):
                AugPreview(name, m["default_task"]).build()


def confirm_delete(name: str) -> None:
    with ui.dialog() as d, ui.card():
        ui.label(f"Remove dataset {name} from the project? The image files are not deleted.")
        with ui.row():
            ui.button(
                "Remove",
                color="negative",
                on_click=lambda: (
                    store.delete_dataset(project(), name),
                    d.close(),
                    ui.navigate.to("/datasets"),
                ),
            )
            ui.button("Cancel", on_click=d.close).props("flat")
    d.open()


class HealthView:
    def __init__(self, name: str) -> None:
        self.name = name
        self.imgsz = 640

    def build(self) -> None:
        with ui.row().classes("items-center"):
            ui.select(
                [320, 480, 640, 800, 960, 1024, 1280, 1536],
                value=self.imgsz,
                label="Training image size",
                on_change=lambda e: self.load(e.value),
            ).classes("w-48")
            help_icon(
                "The checks depend on the size images are shrunk to for training (object sizes in pixels). "
                "Pick the size you plan to use."
            )
        self.box = ui.column().classes("w-full gap-3")
        ui.timer(0.1, lambda: self.load(self.imgsz), once=True)

    async def load(self, imgsz: int) -> None:
        self.imgsz = imgsz
        self.box.clear()
        with self.box:
            ui.spinner(size="lg")
            ui.label(
                "Analysing the dataset (the first time takes a moment: images are fingerprinted for duplicates)..."
            )
        rep = await run.io_bound(store.health_report, project(), self.name, imgsz)
        self.box.clear()
        with self.box:
            self.render(rep)

    def render(self, rep: dict[str, Any]) -> None:
        st = rep["stats"]
        worst = max((c["level"] for c in rep["checks"]), key=["ok", "info", "warn", "bad"].index)
        verdict = {
            "ok": "Looks healthy.",
            "info": "Looks fine; see the notes.",
            "warn": "Usable, but some things deserve attention.",
            "bad": "There are problems that will hurt training. Fix them first if you can.",
        }[worst]
        with ui.card().classes("w-full"):
            with ui.row().classes("items-center"):
                ui.icon(LEVEL_ICON[worst], color=LEVEL_COLOUR[worst]).classes("text-3xl")
                ui.label(verdict).classes("text-lg")
            ui.label(
                f"{st['images_total']} images, {sum(st['instances_total'].values())} labelled objects, {len(st['classes'])} classes. "
                f"Recommended training image size: {rep['recommended_imgsz']} px."
            ).classes("text-sm")
        with ui.grid(columns=2).classes("w-full gap-3"):
            for c in sorted(
                rep["checks"], key=lambda c: -["ok", "info", "warn", "bad"].index(c["level"])
            ):
                with ui.card().classes("w-full").props("flat bordered"):
                    with ui.row().classes("items-center no-wrap"):
                        ui.icon(LEVEL_ICON[c["level"]], color=LEVEL_COLOUR[c["level"]]).classes(
                            "text-2xl"
                        )
                        ui.label(c["title"]).classes("font-bold")
                    ui.label(c["value"]).classes("text-md")
                    with ui.expansion(
                        "What it means / why it matters / what to do",
                        value=c["level"] in ("warn", "bad") and not simple_mode(),
                    ).classes("w-full text-sm"):
                        ui.label(c["meaning"])
                        ui.label("Why it matters: " + c["why"]).classes("text-gray-700")
                        ui.label("What to do: " + c["action"]).classes("font-medium")
                        for d in c["details"][:15]:
                            ui.label("• " + d).classes("mt-mono")
        names = {int(k): v for k, v in st["classes"].items()}
        labels = [names[i] for i in sorted(names)]
        per_split = {
            s: [st["instances"][s].get(str(i), st["instances"][s].get(i, 0)) for i in sorted(names)]
            for s in st["instances"]
        }
        with ui.grid(columns=2).classes("w-full gap-3"):
            with ui.card().classes("w-full"):
                bar_chart(
                    labels,
                    per_split,
                    "Objects per class (stacked by split)",
                    height=f"{max(260, 22 * len(labels))}px",
                    horizontal=len(labels) > 8,
                )
            if st.get("object_size_hist"):
                with ui.card().classes("w-full"):
                    h = st["object_size_hist"]
                    bar_chart(
                        list(h),
                        {"objects": list(h.values())},
                        f"Object size (short side, px) at {st['imgsz']} px",
                    )
                    ui.label(
                        "Objects under 8 px are hard to detect, under 32 px count as 'small'."
                    ).classes("text-xs text-gray-500")
            if st.get("heatmap") and sum(map(sum, st["heatmap"])):
                with ui.card().classes("w-full"):
                    from .common import heatmap as hm

                    n = len(st["heatmap"])
                    hm(
                        st["heatmap"],
                        [""] * n,
                        [""] * n,
                        "Where object centres are (image split in a grid)",
                        height="300px",
                    )
            with ui.card().classes("w-full"):
                opi = st["objects_per_image"]
                bar_chart(
                    [str(k) for k in opi], {"images": list(opi.values())}, "Objects per image"
                )


class Gallery:
    PAGE = 48

    def __init__(self, name: str) -> None:
        self.name = name
        self.filter = "all"
        self.page = 0
        self.layers = ["boxes", "masks", "keypoints"]

    def build(self) -> None:
        ds, _ = store.open_dataset(project(), self.name)
        self.ds = ds
        opts = {
            "all": "All images",
            "split:train": "Training split",
            "split:val": "Validation split",
            "background": "Background (no objects)",
            "errors": "Label problems",
            "tiny": "Tiny objects (< 8 px at 640)",
            "crowded": "Most crowded",
            "duplicates": "Duplicates",
        }
        opts.update({f"class:{i}": f"Class: {n}" for i, n in ds.names.items()})
        with ui.row().classes("items-center"):
            ui.select(
                opts, value=self.filter, label="Show", on_change=lambda e: self.set_filter(e.value)
            ).classes("w-64")
            for layer in ("boxes", "masks", "keypoints"):
                ui.checkbox(
                    layer,
                    value=True,
                    on_change=lambda e, lyr=layer: self.toggle_layer(lyr, e.value),
                )
            self.count = ui.label()
        self.grid = ui.element("div").classes(
            "grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-2 w-full"
        )
        with ui.row():
            ui.button("Previous", icon="chevron_left", on_click=lambda: self.go(-1)).props("flat")
            ui.button("Next", icon="chevron_right", on_click=lambda: self.go(1)).props("flat")
        self.report = (
            store.health_report(project(), self.name, 640)
            if (store.dataset_dir(project(), self.name) / "health_640.json").is_file()
            else {"stats": {"label_errors": {}, "duplicates": []}}
        )
        self.show()

    def set_filter(self, f: str) -> None:
        self.filter, self.page = f, 0
        self.show()

    def toggle_layer(self, layer: str, on: bool) -> None:
        self.layers = [lyr for lyr in self.layers if lyr != layer] + ([layer] if on else [])
        self.show()

    def go(self, d: int) -> None:
        self.page = max(0, self.page + d)
        self.show()

    def show(self) -> None:
        items = health.gallery_filter(self.ds, self.report, self.filter)
        if self.filter == "tiny":
            items = health.smallest_first(items)
        n_pages = max(1, (len(items) + self.PAGE - 1) // self.PAGE)
        self.page = min(self.page, n_pages - 1)
        page_items = items[self.page * self.PAGE : (self.page + 1) * self.PAGE]
        self.count.text = f"{len(items)} images - page {self.page + 1}/{n_pages}"
        self.grid.clear()
        cache = ws().cache_dir
        errs = self.report["stats"].get("label_errors", {})
        with self.grid:
            for it in page_items:
                thumb = render.thumbnail(it, self.ds.names, cache, 320, tuple(self.layers))
                with (
                    ui.card()
                    .classes("p-1 cursor-pointer")
                    .props("flat bordered")
                    .on("click", lambda _, i=it: self.open(i))
                ):
                    ui.image(str(thumb)).classes("w-full")
                    txt = f"{it.split} · {len(it.objects)} obj"
                    if it.key in errs:
                        txt += " · ⚠"
                    ui.label(txt).classes("text-xs text-gray-600")

    def open(self, it: Any) -> None:
        errs = self.report["stats"].get("label_errors", {}).get(it.key, [])
        with ui.dialog().props("maximized") as d, ui.card().classes("w-full h-full"):
            with ui.row().classes("items-center w-full"):
                ui.label(
                    f"{it.image.name}  ({it.split}, {it.width}x{it.height}, {len(it.objects)} objects)"
                ).classes("font-bold")
                ui.space()
                ui.button(icon="close", on_click=d.close).props("flat round")
            img = render.render_item(it, self.ds.names, self.layers + ["labels"], max_side=1600)
            ui.image(img).classes("max-h-[80vh] object-contain")
            counts: dict[str, int] = {}
            for o in it.objects:
                n = self.ds.names.get(o.cls, str(o.cls))
                counts[n] = counts.get(n, 0) + 1
            ui.label(
                ", ".join(f"{k}: {v}" for k, v in counts.items()) or "no objects (background image)"
            )
            for e in errs:
                ui.label("⚠ " + e).classes("text-negative")
            ui.label(str(it.image)).classes("text-xs text-gray-500")
        d.open()


class AugPreview:
    def __init__(self, name: str, task: str) -> None:
        self.name, self.task = name, task
        self.values: dict[str, Any] = dict(P.AUG_PRESETS["standard"])
        self.values["imgsz"] = 640

    def build(self) -> None:
        ui.markdown(
            "These are **real training samples**: resized and augmented exactly as training will do it "
            "(mosaic glues 4 images together). Check that the objects are still recognisable and the labels still fit."
        )
        with ui.row().classes("items-center"):
            ui.select(
                list(P.AUG_PRESETS),
                value="standard",
                label="Augmentation preset",
                on_change=lambda e: self.preset(e.value),
            ).classes("w-48")
            ui.number(
                "Image size",
                value=640,
                step=32,
                on_change=lambda e: self.values.update(imgsz=int(e.value or 640)),
            ).classes("w-32")
            ui.select(
                {t: TASK_LABELS[t] for t in TASK_LABELS},
                value=self.task,
                label="Task",
                on_change=lambda e: setattr(self, "task", e.value),
            ).classes("w-64")
            ui.button("Show samples", icon="refresh", on_click=self.refresh)
        self.help = ui.label(P.AUG_PRESET_HELP["standard"]).classes("text-sm text-gray-600")
        self.grid = ui.element("div").classes("grid grid-cols-3 gap-2 w-full")

    def preset(self, name: str) -> None:
        self.values.update(P.AUG_PRESETS[name])
        self.help.text = P.AUG_PRESET_HELP[name]

    async def refresh(self) -> None:
        from ..augpreview import preview_data, samples

        self.grid.clear()
        with self.grid:
            ui.spinner(size="lg")
        ds, _ = store.open_dataset(project(), self.name)
        try:
            data = await run.io_bound(preview_data, ds, self.task, ws().cache_dir)
            imgs = await run.io_bound(samples, data, self.task, self.values, 9)
        except Exception as exc:  # noqa: BLE001 - shown to the user
            self.grid.clear()
            with self.grid:
                ui.label(f"Could not make samples: {exc}").classes("text-negative")
            return
        self.grid.clear()
        with self.grid:
            for im in imgs:
                ui.image(im).classes("w-full")
