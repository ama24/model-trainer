"""Home, compare, playground, sweeps, settings and glossary pages."""

from __future__ import annotations

import contextlib
import tempfile
from pathlib import Path
from typing import Any

from nicegui import run, ui

from .. import compare as C
from .. import sweeps as S
from ..notify import send
from ..runs.manager import Manager
from ..runs.store import Run
from ..util import IMAGE_EXTS
from .common import empty, frame, help_icon, line_chart, project, state_badge, ws

# ------------------------------------------------------------------------------- home


def home_page() -> None:
    with frame("Home"):
        mgr = Manager(ws())
        proj = project()
        ui.label(f"Project: {proj.name}").classes("text-2xl font-bold")
        for r in mgr.interrupted():
            m = r.metrics()
            with ui.card().classes("w-full bg-orange-50"):
                with ui.row().classes("items-center w-full"):
                    ui.icon("restore", color="deep-orange").classes("text-2xl")
                    ui.label(
                        f"Run '{r.config.name}' was interrupted after epoch {m[-1]['epoch'] if m else 0}. Resume it?"
                    )
                    ui.space()
                    ui.button(
                        "Resume",
                        on_click=lambda r=r: (mgr.resume(r), ui.navigate.to(f"/run/{r.id}")),
                    ).props("color=positive")
                    ui.button("Open", on_click=lambda r=r: ui.navigate.to(f"/run/{r.id}")).props(
                        "flat"
                    )
        active = mgr.active()
        if active:
            ui.label("Training now").classes("text-lg font-bold")
            for r in active:
                s = r.summary()
                with (
                    ui.card()
                    .classes("w-full cursor-pointer")
                    .on("click", lambda _, r=r: ui.navigate.to(f"/run/{r.id}"))
                ):
                    with ui.row().classes("items-center"):
                        state_badge(s["state"])
                        ui.label(
                            f"{s['name']} - epoch {s['epoch'] or 0}/{s['epochs'] or '?'} - {s['model']} on {s['dataset']}"
                        )
        with ui.row().classes("gap-4 w-full"):
            for title, text, icon, link in (
                (
                    "1. Import a dataset",
                    "YOLO, COCO or one folder per class. Get a health report with explanations.",
                    "photo_library",
                    "/datasets",
                ),
                (
                    "2. Set up a run",
                    "Pick a base model, image size, augmentation, epochs. Simple or Advanced.",
                    "add_circle",
                    "/new",
                ),
                (
                    "3. Watch it train",
                    "Live charts, pause/resume, crash-safe checkpoints, phone notifications.",
                    "show_chart",
                    "/runs",
                ),
                (
                    "4. Check & export",
                    "Errors gallery, try it on your own images, export for deployment.",
                    "science",
                    "/playground",
                ),
            ):
                with (
                    ui.card()
                    .classes("w-64 cursor-pointer")
                    .on("click", lambda _, link=link: ui.navigate.to(link))
                ):
                    ui.icon(icon, color="primary").classes("text-3xl")
                    ui.label(title).classes("font-bold")
                    ui.label(text).classes("text-sm text-gray-600")
        runs = [Run(p) for p in proj.runs()[:8]]
        if runs:
            ui.label("Recent runs").classes("text-lg font-bold")
            for r in runs:
                s = r.summary()
                with (
                    ui.row()
                    .classes("items-center cursor-pointer")
                    .on("click", lambda _, r=r: ui.navigate.to(f"/run/{r.id}"))
                ):
                    state_badge(s["state"])
                    score = (
                        f" - {s['main_metric']} {s['main_value']:.3f}"
                        if s["main_value"] is not None
                        else ""
                    )
                    ui.label(f"{s['name']} ({s['model']}, {s['dataset']}){score}")


# ---------------------------------------------------------------------------- compare


def compare_page(runs: str = "") -> None:
    with frame("Compare runs"):
        proj = project()
        all_runs = {p.name: p.name for p in proj.runs()}
        chosen = [r for r in runs.split(",") if r in all_runs] if runs else []
        ui.select(
            all_runs,
            value=chosen,
            multiple=True,
            label="Runs to compare",
            on_change=lambda e: ui.navigate.to("/compare?runs=" + ",".join(e.value)),
        ).classes("w-full").props("use-chips")
        if len(chosen) < 2:
            ui.label("Choose at least two runs.").classes("text-gray-600")
            return
        rs = [Run(ws().find_run(r)) for r in chosen]
        best = C.best(rs)
        if best:
            with ui.card().classes("w-full bg-green-50"):
                ui.label(
                    f"Best: {best['name']} ({best['main metric']} {best['score']:.3f})"
                ).classes("font-bold")
        rows = C.results_table(rs)
        ui.table(
            columns=[
                {"name": k, "label": k, "field": k, "sortable": True, "align": "left"}
                for k in rows[0]
            ],
            rows=rows,
        ).props("dense flat bordered").classes("w-full")
        ui.label("What was different").classes("text-lg font-bold")
        diff = C.settings_diff(rs)
        if diff:
            ui.table(
                columns=[{"name": k, "label": k, "field": k, "align": "left"} for k in diff[0]],
                rows=[{k: str(v) for k, v in d.items()} for d in diff],
            ).props("dense flat bordered").classes("w-full")
        else:
            ui.label("Identical settings (only randomness differs).")
        metrics = C.common_metrics(rs)
        default = [m for m in metrics if "mAP50-95" in m or "accuracy_top1" in m][:1] + [
            m for m in metrics if m.startswith("val/") and "loss" in m
        ][:1]
        sel = (
            ui.select(metrics, value=default, multiple=True, label="Curves")
            .classes("w-full")
            .props("use-chips")
        )
        box = ui.grid(columns=2).classes("w-full")

        def draw() -> None:
            box.clear()
            with box:
                for m in sel.value:
                    with ui.card().classes("w-full"):
                        line_chart(C.curves(rs, m), m)

        sel.on_value_change(lambda _: draw())
        draw()


# ------------------------------------------------------------------------- playground


def playground_page(run_id: str | None = None) -> None:
    with frame("Playground"):
        Playground(run_id).build()


class Playground:
    def __init__(self, run_id: str | None) -> None:
        self.runs = {p.name: p for p in project().runs() if Run(p).best_weights}
        self.run_a = run_id if run_id in self.runs else next(iter(self.runs), None)
        self.run_b: str | None = None
        self.files: list[Path] = []
        self.conf, self.iou = 0.25, 0.7
        self.tmp = Path(tempfile.mkdtemp(prefix="mtrain_play_"))
        self.last_preds: list[Any] = []

    def build(self) -> None:
        if not self.runs:
            empty(
                "Train a model first: the playground uses a finished (or paused) run's best weights.",
                ("New run", "/new"),
            )
            return
        ui.markdown(
            "Drop in your own photos (or a video) to see what the model finds. "
            "**Final check:** use images the model has never seen, ideally from the real setting where it will be used."
        )
        with ui.row().classes("items-center"):
            ui.select(
                list(self.runs),
                value=self.run_a,
                label="Model",
                on_change=lambda e: setattr(self, "run_a", e.value),
            ).classes("w-96")
            ui.select(
                [None, *self.runs],
                value=None,
                label="Compare with (optional)",
                on_change=lambda e: setattr(self, "run_b", e.value),
            ).classes("w-96")
        with ui.row().classes("items-center w-full"):
            ui.label("Confidence")
            ui.slider(
                min=0.01,
                max=0.95,
                step=0.01,
                value=self.conf,
                on_change=lambda e: setattr(self, "conf", e.value),
            ).classes("w-64").props("label-always")
            help_icon(
                "Only detections at least this sure are shown. Lower: finds more, but more false alarms. "
                "The run's results page recommends a value (best F1)."
            )
            ui.label("Overlap (IoU)")
            ui.slider(
                min=0.1,
                max=0.95,
                step=0.05,
                value=self.iou,
                on_change=lambda e: setattr(self, "iou", e.value),
            ).classes("w-48").props("label-always")
            help_icon(
                "Two detections overlapping more than this are merged into one (non-maximum suppression)."
            )
            ui.button("Run again", icon="refresh", on_click=self.predict)
        ui.upload(
            label="Drop images or a video here",
            multiple=True,
            auto_upload=True,
            on_upload=self.uploaded,
            on_multi_upload=lambda _: self.predict(),
        ).props("accept=image/*,video/*").classes("w-full")
        with ui.row().classes("items-center"):
            self.folder = ui.input("... or a folder of images on this computer").classes(
                "w-[480px]"
            )
            ui.button("Load folder", on_click=self.load_folder).props("flat")
        self.out = ui.column().classes("w-full")
        with ui.expansion(
            "Score on a labelled test set (external data)", icon="fact_check"
        ).classes("w-full"):
            ui.label(
                "Pick an imported dataset with the same class names: the model is scored on its validation images."
            ).classes("text-sm")
            ds = ui.select(project().datasets(), label="Dataset").classes("w-64")
            res = ui.label()

            async def score() -> None:
                from ..datasets import store
                from ..evaluate import evaluate_external

                if not ds.value or not self.run_a:
                    return
                res.text = "scoring..."
                m = store.manifest(project(), ds.value)
                src = Path(m["source"])
                yaml_path = (
                    src if src.suffix in (".yaml", ".yml") else next(src.glob("*.yaml"), None)
                )
                if yaml_path is None:
                    res.text = "this dataset has no YOLO yaml (COCO / folders): convert it or import a YOLO version"
                    return
                try:
                    r = await run.io_bound(
                        evaluate_external, Run(self.runs[self.run_a]), yaml_path, ds.value
                    )
                    res.text = r["summary"]
                except Exception as exc:  # noqa: BLE001
                    res.text = f"failed: {exc}"

            ui.button("Score", on_click=score)
        with ui.expansion(
            "Send these images to a new dataset (active learning)", icon="add_photo_alternate"
        ).classes("w-full"):
            ui.label(
                "Saves the images with the model's predictions as labels, as a YOLO dataset you can correct in a labelling "
                "tool and import, then train again. The fastest way to grow a dataset with the hard cases."
            ).classes("text-sm")
            name = ui.input("New dataset name", value="from_playground").classes("w-64")
            ui.button("Save as dataset", on_click=lambda: self.save_dataset(name.value)).props(
                "outline"
            )

    async def uploaded(self, e: Any) -> None:
        dest = self.tmp / Path(e.file.name).name
        await e.file.save(dest)
        self.files.append(dest)

    async def load_folder(self) -> None:
        p = Path(self.folder.value.strip().strip('"'))
        if not p.is_dir():
            ui.notify("Not a folder", type="warning")
            return
        self.files = sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTS)[:200]
        await self.predict()

    async def predict(self) -> None:
        from ..predict import VIDEO_EXTS, predict_images, predict_video

        if not self.files or not self.run_a:
            return
        self.out.clear()
        with self.out:
            ui.spinner(size="lg")
        models = [self.run_a] + ([self.run_b] if self.run_b and self.run_b != self.run_a else [])
        images = [f for f in self.files if f.suffix.lower() in IMAGE_EXTS]
        videos = [f for f in self.files if f.suffix.lower() in VIDEO_EXTS]
        results: dict[str, list[Any]] = {}
        for name in models:
            w = Run(self.runs[name]).best_weights
            results[name] = await run.io_bound(predict_images, w, images, self.conf, self.iou)
        self.last_preds = results[self.run_a]
        self.out.clear()
        with self.out:
            for i, img in enumerate(images):
                with ui.card().classes("w-full"):
                    ui.label(img.name).classes("font-medium")
                    with ui.row().classes("w-full no-wrap"):
                        for name in models:
                            p = results[name][i]
                            with ui.column().classes("flex-1"):
                                if len(models) > 1:
                                    ui.label(name).classes("text-xs text-gray-500")
                                ui.image(p.plotted).classes("w-full")
                                if p.top:
                                    ui.label(", ".join(f"{n} {s:.0%}" for n, s in p.top[:3]))
                                else:
                                    counts: dict[str, int] = {}
                                    for o in p.objects:
                                        counts[o["class"]] = counts.get(o["class"], 0) + 1
                                    ui.label(
                                        (
                                            ", ".join(
                                                f"{k}: {v}" for k, v in sorted(counts.items())
                                            )
                                            or "nothing found"
                                        )
                                        + f" - {p.ms:.0f} ms"
                                    )
            for v in videos:
                ui.label(f"{v.name}: processing video...")
                out = self.tmp / f"{v.stem}_pred.mp4"
                info = await run.io_bound(
                    predict_video,
                    Run(self.runs[self.run_a]).best_weights,
                    v,
                    out,
                    self.conf,
                    self.iou,
                )
                ui.label(f"{info['frames']} frames, {info['detections']} detections. Saved: {out}")
                ui.video(str(out)).classes("w-full")

    def save_dataset(self, name: str) -> None:
        from ..datasets import store
        from ..predict import save_as_dataset
        from ..workspace import safe_name

        if not self.last_preds:
            ui.notify("Run the model on some images first", type="warning")
            return
        out = ws().root / "inbox" / safe_name(name)
        save_as_dataset(self.last_preds, Run(self.runs[self.run_a]).best_weights, out)
        with contextlib.suppress(
            FileExistsError
        ):  # already imported: the files were refreshed in place
            store.import_dataset(project(), out, name=name)
        ui.notify(
            f"Saved to {out} and imported as {name}. Correct the labels there, then 'Re-read files' on the dataset page.",
            type="positive",
            multi_line=True,
        )


# ----------------------------------------------------------------------------- sweeps


def sweeps_page() -> None:
    with frame("Sweeps"):
        ui.label("Hyperparameter sweeps").classes("text-2xl font-bold")
        ui.markdown(
            "A sweep trains several copies of a run with different settings, one after another, and ranks them. "
            "Write the values to try as YAML. Lists are tried in every combination (grid); "
            "`{min, max, log}` ranges are sampled randomly."
        )
        proj = project()
        runs = [p.name for p in proj.runs()]
        if not runs:
            empty("Create a run first: a sweep starts from its settings.", ("New run", "/new"))
            return
        base = ui.select(
            runs, value=runs[0], label="Base run (its settings are the starting point)"
        ).classes("w-96")
        space = ui.codemirror(
            "lr0: [0.001, 0.01]\nimgsz: [640, 960]\n# model.size: [n, s]\n# mosaic: {min: 0.5, max: 1.0}\n",
            language="YAML",
        ).classes("w-full h-40")
        with ui.row().classes("items-center"):
            mode = ui.toggle(["grid", "random"], value="grid")
            n = ui.number("Max runs", value=8, min=1, step=1).classes("w-32")
            budget = ui.number(
                "Time budget (hours, empty = none)", value=None, min=0.1, step=0.5
            ).classes("w-72")

        def create() -> None:
            import yaml

            try:
                sp = yaml.safe_load(space.value) or {}
                assert isinstance(sp, dict) and sp
            except Exception:  # noqa: BLE001
                ui.notify(
                    "The values to try must be YAML like 'lr0: [0.001, 0.01]'", type="negative"
                )
                return
            r = Run(ws().find_run(base.value))
            s = S.create_sweep(
                ws(),
                proj,
                r.config,
                sp,
                mode.value,
                int(n.value),
                float(budget.value) if budget.value else None,
            )
            S.start_sweep(ws(), s["id"])
            ui.notify(f"Sweep {s['id']}: {len(s['runs'])} runs queued", type="positive")
            ui.navigate.reload()

        ui.button("Create and start", icon="play_arrow", on_click=create)
        for s in reversed(S.sweeps(ws())):
            with ui.expansion(f"{s['id']} - {len(s['runs'])} runs", icon="tune").classes("w-full"):
                rows = S.results(ws(), s["id"])
                if rows:
                    ui.table(
                        columns=[
                            {"name": k, "label": k, "field": k, "sortable": True, "align": "left"}
                            for k in rows[0]
                        ],
                        rows=[
                            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
                            for r in rows
                        ],
                    ).props("dense flat").classes("w-full")
                ui.button(
                    "Compare all",
                    on_click=lambda s=s: ui.navigate.to("/compare?runs=" + ",".join(s["runs"])),
                ).props("flat")


# --------------------------------------------------------------------------- settings


def settings_page() -> None:
    with frame("Settings"):
        s = ws().settings()
        n = s["notify"]
        ui.label("Phone notifications").classes("text-xl font-bold")
        ui.markdown(
            "1. Install the free **ntfy** app (Android / iPhone).\n"
            "2. In the app, subscribe to a topic with a private, hard-to-guess name, e.g. `mtrain-k39xq7`.\n"
            "3. Put the same name below and press *Send test*.\n\n"
            "Anyone who knows the topic name can read the messages, so keep it secret."
        )
        enabled = ui.switch("Send notifications", value=n.get("enabled", False))
        topic = ui.input("Topic", value=n.get("topic", "")).classes("w-80")
        server = ui.input("Server", value=n.get("server", "https://ntfy.sh")).classes("w-80")
        events = (
            ui.select(
                ["finished", "failed", "warning", "paused"],
                value=n.get("events"),
                multiple=True,
                label="Notify when",
            )
            .classes("w-96")
            .props("use-chips")
        )

        def save() -> None:
            s["notify"] = {
                "enabled": enabled.value,
                "topic": topic.value.strip(),
                "server": server.value.strip(),
                "events": events.value,
            }
            ws().save_settings(s)
            ui.notify("Saved", type="positive")

        def test() -> None:
            save()
            ok = send(ws().settings(), "test", "model-trainer", "Test notification: it works.")
            ui.notify(
                "Sent - check your phone"
                if ok
                else "Could not send: check the topic and the internet connection",
                type="positive" if ok else "negative",
            )

        with ui.row():
            ui.button("Save", on_click=save)
            ui.button("Send test", on_click=test).props("outline")
        ui.separator()
        ui.label("Other machines (SSH)").classes("text-xl font-bold")
        ui.markdown(
            "Train on another computer, or time exported models on a Raspberry Pi. The machine needs model-trainer "
            "installed and SSH key login (no password). *Experimental.*"
        )
        remotes = list(s.get("remotes") or [])
        box = ui.column().classes("w-full")

        def draw() -> None:
            box.clear()
            with box:
                for i, r in enumerate(remotes):
                    with ui.row().classes("items-center"):
                        ui.label(
                            f"{r['name']}: {r.get('user', '')}@{r['host']}:{r.get('port', 22)} {r.get('workdir', '')} ({r.get('python', 'python3')})"
                        )
                        ui.button(icon="network_check", on_click=lambda r=r: check(r)).props(
                            "flat dense"
                        ).tooltip("Test the connection")
                        ui.button(
                            icon="delete", on_click=lambda i=i: (remotes.pop(i), persist(), draw())
                        ).props("flat dense color=negative")

        async def check(r: dict[str, Any]) -> None:
            from ..remote import Remote

            ok, msg = await run.io_bound(Remote(r).check)
            ui.notify(
                ("OK: " if ok else "Failed: ") + msg,
                type="positive" if ok else "negative",
                multi_line=True,
            )

        def persist() -> None:
            s["remotes"] = remotes
            ws().save_settings(s)

        draw()
        with ui.row().classes("items-end"):
            f = {
                k: ui.input(k, value=v).classes("w-40")
                for k, v in (
                    ("name", ""),
                    ("host", ""),
                    ("user", ""),
                    ("port", "22"),
                    ("workdir", "~/model-trainer-workspace"),
                    ("python", "python3"),
                )
            }

            def add() -> None:
                if not f["name"].value or not f["host"].value:
                    ui.notify("name and host are needed", type="warning")
                    return
                remotes.append({k: v.value for k, v in f.items()})
                persist()
                draw()

            ui.button("Add", on_click=add)
        ui.separator()
        ui.label(f"Workspace: {ws().root}").classes("text-sm text-gray-600")


# --------------------------------------------------------------------------- glossary

GLOSSARY = [
    (
        "Epoch",
        "One pass over all training images. Training usually needs tens to hundreds of epochs.",
    ),
    (
        "Batch size",
        "How many images are processed together before the weights are updated. Limited by GPU memory.",
    ),
    ("Learning rate", "How big each update step is. Too high: unstable; too low: slow."),
    (
        "Loss",
        "A number saying how wrong the model is on the images it just saw. Training lowers it.",
    ),
    (
        "Overfitting",
        "The model memorises the training images instead of learning the general pattern: training loss keeps "
        "falling but validation gets worse. Fix: more data, more augmentation, smaller model, early stopping.",
    ),
    ("Validation set", "Images kept out of training, used only to measure the model honestly."),
    ("Precision", "Of the detections the model reports, the share that are right."),
    ("Recall", "Of the objects that are really there, the share the model finds."),
    (
        "F1",
        "One number combining precision and recall (their harmonic mean). Its peak gives a good confidence threshold.",
    ),
    (
        "IoU",
        "Intersection over union: how much a predicted box overlaps the true box (0 = not at all, 1 = perfectly).",
    ),
    (
        "mAP50",
        "Mean average precision when a detection counts as right if it overlaps the true box by at least 50 %.",
    ),
    (
        "mAP50-95",
        "mAP averaged over overlap thresholds 50 %..95 %: also rewards precise boxes. The main detection score.",
    ),
    (
        "Confidence threshold",
        "Detections less sure than this are dropped. Trade-off between missed objects and false alarms.",
    ),
    (
        "NMS",
        "Non-maximum suppression: removes duplicate detections of the same object (YOLO26 does not need it).",
    ),
    (
        "Augmentation",
        "Random changes to training images (flip, rotate, recolour, mosaic...) so the model generalises.",
    ),
    ("Mosaic", "Augmentation that glues 4 training images into one; very effective for detection."),
    (
        "Pretrained",
        "Weights already trained on a large dataset; your training adapts them (transfer learning).",
    ),
    (
        "Fine-tuning",
        "Continuing training of an existing model on new data, often with a lower learning rate.",
    ),
    (
        "Early stopping",
        "Ending training when the validation score has not improved for a number of epochs (patience).",
    ),
    ("Mixed precision (AMP)", "Computing with 16-bit numbers where safe: faster and less memory."),
    (
        "Class imbalance",
        "Some classes have far more examples than others; the rare ones are learned badly.",
    ),
    (
        "Leakage",
        "Validation images that (nearly) also appear in training: the scores look better than reality.",
    ),
    (
        "Export / ONNX",
        "Converting the model to a format that runs without the training software, e.g. on a Raspberry Pi.",
    ),
    ("FP16 / INT8", "Smaller number formats for faster, smaller models; INT8 can lose accuracy."),
]


def glossary_page() -> None:
    with frame("Glossary"):
        ui.label("Glossary").classes("text-2xl font-bold")
        flt = ui.input("Search").classes("w-64")
        box = ui.column().classes("w-full")

        def draw() -> None:
            box.clear()
            q = (flt.value or "").lower()
            with box:
                for term, text in GLOSSARY:
                    if q and q not in term.lower() and q not in text.lower():
                        continue
                    with ui.row().classes("no-wrap"):
                        ui.label(term).classes("w-48 font-bold")
                        ui.label(text)

        flt.on_value_change(lambda _: draw())
        draw()
