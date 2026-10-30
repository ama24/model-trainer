"""One run: live monitor with controls, 'watch it learn', results, error gallery, export."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nicegui import run, ui
from PIL import Image, ImageDraw

from ..evaluate import evaluate_run, package
from ..export import FORMATS, PRECISION_HELP, export_model
from ..runs.manager import Manager
from ..runs.store import ACTIVE, Run
from ..util import human_time, read_json
from .common import empty, frame, heatmap, help_icon, line_chart, set_series, state_badge, ws

CHART_HELP = {
    "loss": "Losses measure how wrong the model is (lower is better). Training loss should fall steadily. If the validation "
    "loss starts rising while training loss keeps falling, the model is memorising the training images (overfitting).",
    "metrics": "Scores on the validation images (higher is better, 1.0 is perfect). Precision: how many detections are right. "
    "Recall: how many objects are found. mAP50: average precision when a box counts as right at 50 % overlap; "
    "mAP50-95: averaged over stricter overlaps (the main score).",
    "lr": "The learning rate: the step size of each update. It ramps up during warm-up, then decreases.",
}


def runs_page() -> None:
    from .common import project

    with frame("Runs"):
        proj = project()
        mgr = Manager(ws())
        with ui.row().classes("items-center"):
            ui.label("Runs").classes("text-2xl font-bold")
            ui.button("New run", icon="add", on_click=lambda: ui.navigate.to("/new"))
        runs = [Run(p) for p in proj.runs()]
        if not runs:
            empty("No runs yet.", ("Start the wizard", "/new"))
            return
        rows = []
        for r in runs:
            s = r.summary()
            rows.append(
                {
                    "id": s["id"],
                    "name": s["name"],
                    "state": s["state"],
                    "model": s["model"],
                    "dataset": s["dataset"],
                    "task": s["task"],
                    "epoch": f"{s['epoch'] or 0}/{s['epochs'] or '?'}",
                    "score": f"{s['main_value']:.3f}" if s["main_value"] is not None else "",
                    "metric": s["main_metric"] or "",
                    "updated": (s["updated"] or "")[:16].replace("T", " "),
                    "notes": s["notes"],
                }
            )
        cols = [
            {"name": k, "label": k.capitalize(), "field": k, "sortable": True, "align": "left"}
            for k in rows[0]
            if k != "id"
        ]
        table = (
            ui.table(columns=cols, rows=rows, row_key="id", selection="multiple")
            .classes("w-full")
            .props("flat bordered")
        )
        table.on("rowClick", lambda e: ui.navigate.to(f"/run/{e.args[1]['id']}"))
        with ui.row():
            ui.button(
                "Compare selected",
                icon="compare",
                on_click=lambda: ui.navigate.to(
                    "/compare?runs=" + ",".join(r["id"] for r in table.selected)
                ),
            ).props("outline")
        q = mgr.queue()
        if q:
            ui.label("Queue (trains one after another)").classes("text-lg font-bold mt-4")
            for i, e in enumerate(q):
                r = Run(Path(e["run"]))
                with ui.row().classes("items-center"):
                    ui.label(f"{i + 1}. {r.id}")
                    ui.button(
                        icon="arrow_upward",
                        on_click=lambda r=r: (mgr.move_in_queue(r, -1), ui.navigate.reload()),
                    ).props("flat dense")
                    ui.button(
                        icon="arrow_downward",
                        on_click=lambda r=r: (mgr.move_in_queue(r, 1), ui.navigate.reload()),
                    ).props("flat dense")
                    ui.button(
                        icon="close", on_click=lambda r=r: (mgr.stop(r), ui.navigate.reload())
                    ).props("flat dense color=negative").tooltip("Remove from the queue")


def run_page(run_id: str) -> None:
    with frame(f"Run {run_id}"):
        p = ws().find_run(run_id)
        if p is None:
            empty(f"No run {run_id!r}.", ("All runs", "/runs"))
            return
        RunView(Run(p)).build()


class RunView:
    def __init__(self, r: Run) -> None:
        self.run = r
        self.mgr = Manager(ws())
        self.cfg = r.config
        self.last_state: str | None = None
        self.n_metrics = -1

    # ------------------------------------------------------------------------ header
    def build(self) -> None:
        with ui.row().classes("items-center w-full"):
            ui.label(self.cfg.name).classes("text-2xl font-bold")
            self.badge_box = ui.row()
            ui.label(
                f"{self.cfg.model.family}{self.cfg.model.size} · {self.cfg.task} · dataset {self.cfg.dataset}"
            ).classes("text-gray-600")
            ui.space()
            self.buttons = ui.row()
        if self.cfg.notes:
            ui.label(self.cfg.notes).classes("text-sm italic text-gray-600")
        self.banner = ui.column().classes("w-full")
        self.progress_box = ui.column().classes("w-full")
        with ui.tabs().classes("w-full") as tabs:
            t_live = ui.tab("Training", icon="show_chart")
            t_watch = ui.tab("Watch it learn", icon="visibility")
            t_res = ui.tab("Results", icon="fact_check")
            t_exp = ui.tab("Export", icon="output")
            t_log = ui.tab("Events & log", icon="article")
            t_cfg = ui.tab("Settings", icon="settings")
        default = (
            t_res if self.run.results() and self.run.state().get("state") not in ACTIVE else t_live
        )
        with ui.tab_panels(tabs, value=default).classes("w-full"):
            with ui.tab_panel(t_live):
                self.build_live()
            with ui.tab_panel(t_watch):
                self.watch_box = ui.column().classes("w-full")
            with ui.tab_panel(t_res):
                self.results_box = ui.column().classes("w-full")
            with ui.tab_panel(t_exp):
                self.export_box = ui.column().classes("w-full")
            with ui.tab_panel(t_log):
                self.events_box = ui.column().classes("w-full")
                self.log = ui.label().classes(
                    "mt-mono w-full bg-gray-50 p-2 rounded max-h-[60vh] overflow-auto"
                )
            with ui.tab_panel(t_cfg):
                self.build_settings()
        self.tick(force=True)
        ui.timer(2.0, self.tick)

    def tick(self, force: bool = False) -> None:
        st = self.run.state()
        state = st.get("state", "created")
        if state != self.last_state or force:
            self.last_state = state
            self.badge_box.clear()
            with self.badge_box:
                state_badge(state)
            self.render_buttons(state)
            self.render_banner(st)
            self.render_results()
            self.render_export()
        m = self.run.metrics()
        if len(m) != self.n_metrics or force:
            self.n_metrics = len(m)
            self.update_charts(m)
            self.render_watch()
            self.render_events()
        if state in ACTIVE or force:
            self.render_progress(st, m)
            self.log.text = self.run.tail_log(150)

    def render_buttons(self, state: str) -> None:
        self.buttons.clear()
        with self.buttons:
            if state in ("running", "preparing"):
                ui.button("Pause", icon="pause", on_click=self.pause).props("color=orange").tooltip(
                    "Finishes the current epoch, saves everything and frees the GPU. Resume continues exactly there."
                ).mark("pause")
                with ui.button("Stop", icon="stop").props("color=negative outline"):
                    with ui.menu():
                        ui.menu_item(
                            "Stop after this epoch (keeps the best model)",
                            on_click=lambda: self.stop(False),
                        )
                        ui.menu_item("Stop now (mid-epoch)", on_click=lambda: self.stop(True))
            elif state in ("pausing", "stopping", "evaluating"):
                ui.spinner()
                ui.label(
                    {
                        "pausing": "pausing after this epoch...",
                        "stopping": "stopping...",
                        "evaluating": "evaluating the best model...",
                    }[state]
                )
            elif state in ("paused", "interrupted"):
                ui.button("Resume", icon="play_arrow", on_click=self.resume).props(
                    "color=positive"
                ).mark("resume")
                ui.button(
                    "Stop for good", icon="stop", on_click=lambda: self.finish_paused()
                ).props("flat")
            elif state in ("done", "stopped"):
                ui.button("Train longer", icon="more_time", on_click=self.extend_dialog).tooltip(
                    "Continue this run for more epochs"
                )
                ui.button(
                    "Copy & change",
                    icon="content_copy",
                    on_click=lambda: ui.navigate.to(f"/new?from_run={self.run.id}"),
                ).props("outline")
                ui.button(
                    "Try it",
                    icon="science",
                    on_click=lambda: ui.navigate.to(f"/playground?run={self.run.id}"),
                ).props("outline")
            elif state == "queued":
                ui.label("waiting in the queue")
                ui.button(
                    "Remove from queue",
                    on_click=lambda: (self.mgr.stop(self.run), ui.navigate.reload()),
                ).props("flat")
            else:  # created / failed
                ui.button(
                    "Start" if state == "created" else "Try again",
                    icon="play_arrow",
                    on_click=self.start,
                ).props("color=positive").mark("start")
                if state == "failed" and self.run.resume_ckpt.is_file():
                    ui.button("Resume from last epoch", icon="restore", on_click=self.resume).props(
                        "outline"
                    )
                ui.button(
                    "Copy & change",
                    icon="content_copy",
                    on_click=lambda: ui.navigate.to(f"/new?from_run={self.run.id}"),
                ).props("flat")
            if state not in ACTIVE:
                ui.button(icon="delete", on_click=self.delete).props("flat color=negative").tooltip(
                    "Delete this run and its files"
                )

    def render_banner(self, st: dict[str, Any]) -> None:
        self.banner.clear()
        state = st.get("state")
        with self.banner:
            if state == "interrupted":
                m = self.run.metrics()
                with ui.card().classes("w-full bg-orange-50"):
                    ui.label(
                        f"This run was interrupted after epoch {m[-1]['epoch'] if m else 0} (crash, power cut, sleep or a closed "
                        "computer). Nothing is lost: Resume continues from the last finished epoch."
                    ).classes("text-orange-900")
            elif state == "failed":
                with ui.card().classes("w-full bg-red-50"):
                    ui.label("Training failed: " + str(st.get("error", "see the log"))).classes(
                        "text-red-900 whitespace-pre-wrap"
                    )
            elif st.get("note"):
                ui.label(st["note"]).classes("text-sm text-gray-600")

    def render_progress(self, st: dict[str, Any], m: list[dict[str, Any]]) -> None:
        self.progress_box.clear()
        state = st.get("state")
        if state not in ACTIVE and state not in ("paused", "interrupted"):
            return
        p = self.run.progress()
        epochs = int(st.get("epochs") or self.cfg.train_args().get("epochs", 0) or 1)
        done = m[-1]["epoch"] if m else 0
        with self.progress_box:
            frac = done / max(epochs, 1)
            ui.linear_progress(value=frac, show_value=False).props("size=14px rounded")
            times = [x.get("epoch_s", 0) for x in m[-5:] if x.get("epoch_s")]
            per = sum(times) / len(times) if times else None
            eta = per * (epochs - done) if per else None
            text = f"epoch {done}/{epochs} finished"
            if state == "running" and p:
                text += f" · epoch {p.get('epoch')}: batch {p.get('batch')}/{p.get('batches')}"
            if per:
                text += f" · {human_time(per)} per epoch · about {human_time(eta)} left (early stopping may end sooner)"
            if state == "preparing":
                text = (
                    "preparing the dataset (freezing a copy of exactly the images this run uses)..."
                )
            ui.label(text).classes("text-sm")
            last = m[-1] if m else {}
            hw = []
            if last.get("gpu_mem_gb"):
                hw.append(f"GPU memory {last['gpu_mem_gb']:.1f} GB")
            if last.get("gpu_temp_c"):
                hw.append(f"{last['gpu_temp_c']:.0f} °C")
            if last.get("gpu_load_pct") is not None:
                hw.append(f"load {last['gpu_load_pct']:.0f} %")
            if last.get("batch"):
                hw.append(f"batch {last['batch']}")
            if hw:
                ui.label(" · ".join(hw)).classes("text-xs text-gray-500")

    # ------------------------------------------------------------------------- charts
    def build_live(self) -> None:
        with ui.grid(columns=2).classes("w-full gap-3"):
            for key, title in (
                ("loss", "Losses (lower is better)"),
                ("metrics", "Validation scores (higher is better)"),
            ):
                with ui.card().classes("w-full"):
                    with ui.row().classes("items-center"):
                        ui.label(title).classes("font-medium")
                        help_icon(CHART_HELP[key])
                    setattr(self, f"chart_{key}", line_chart({}, height="300px"))
        with ui.grid(columns=2).classes("w-full gap-3"):
            with ui.card().classes("w-full"):
                with ui.row().classes("items-center"):
                    ui.label("Learning rate").classes("font-medium")
                    help_icon(CHART_HELP["lr"])
                self.chart_lr = line_chart({}, height="220px")
            with ui.card().classes("w-full"):
                with ui.row().classes("items-center"):
                    ui.label("Per class (latest epoch)").classes("font-medium")
                    help_icon(
                        "Scores of each class on the validation images after the latest epoch. Classes far below the "
                        "others need more (or better) examples, or are confused with a similar class."
                    )
                self.class_table = (
                    ui.table(columns=[], rows=[]).props("dense flat").classes("w-full")
                )

    def update_charts(self, m: list[dict[str, Any]]) -> None:
        if not m:
            return
        keys = sorted({k for r in m for k in r})
        loss = {
            k: [(r["epoch"], r[k]) for r in m if r.get(k) is not None] for k in keys if "loss" in k
        }
        mets = {
            k.replace("metrics/", ""): [(r["epoch"], r[k]) for r in m if r.get(k) is not None]
            for k in keys
            if k.startswith("metrics/")
        }
        lr = {
            k: [(r["epoch"], r[k]) for r in m if r.get(k) is not None]
            for k in keys
            if k.startswith("lr/")
        }
        set_series(self.chart_loss, loss)
        set_series(self.chart_metrics, mets)
        set_series(self.chart_lr, lr)
        pc = next((r["per_class"] for r in reversed(m) if r.get("per_class")), None)
        if pc:
            cols = [
                {"name": k, "label": k, "field": k, "sortable": True, "align": "left"}
                for k in pc[0]
            ]
            rows = [
                {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in pc
            ]
            self.class_table.columns = cols
            self.class_table.rows = rows
            self.class_table.update()

    # -------------------------------------------------------------------------- watch
    def render_watch(self) -> None:
        self.watch_box.clear()
        epochs = sorted(self.run.watch_dir.glob("epoch_*")) if self.run.watch_dir.is_dir() else []
        with self.watch_box:
            if not epochs:
                ui.label(
                    "After each epoch the model's predictions on the same validation images appear here."
                ).classes("text-gray-600")
                return
            ui.label(
                "The same validation images after each epoch: watch the predictions get better."
            ).classes("text-sm")
            grid = ui.element("div").classes("grid grid-cols-2 md:grid-cols-4 gap-2 w-full")

            def show(i: int) -> None:
                d = epochs[int(i)]
                label.text = f"after epoch {int(d.name.split('_')[1])}"
                grid.clear()
                with grid:
                    for f in sorted(d.glob("*.jpg")):
                        ui.image(str(f)).classes("w-full")

            with ui.row().classes("items-center w-full"):
                ui.slider(
                    min=0,
                    max=len(epochs) - 1,
                    value=len(epochs) - 1,
                    on_change=lambda e: show(e.value),
                ).classes("w-96")
                label = ui.label()
            show(len(epochs) - 1)

    # ------------------------------------------------------------------------ results
    def render_results(self) -> None:
        self.results_box.clear()
        res = self.run.results()
        with self.results_box:
            if not res:
                if self.run.best_weights and self.run.state().get("state") not in ACTIVE:
                    ui.button("Evaluate now", icon="fact_check", on_click=self.evaluate)
                else:
                    ui.label("Results appear when training ends.").classes("text-gray-600")
                return
            with ui.card().classes("w-full bg-blue-50"):
                ui.label(res.get("summary", "")).classes("text-md")
                if res.get("best_epoch"):
                    ui.label(
                        f"Best epoch: {res['best_epoch']} of {res.get('epochs_trained')}."
                    ).classes("text-sm text-gray-600")
            mean = res.get("mean", {})
            with ui.row().classes("gap-4"):
                for k, v in mean.items():
                    if k == "fitness":
                        continue
                    with ui.card().classes("p-2"):
                        ui.label(k.replace("metrics/", "")).classes("text-xs text-gray-600")
                        ui.label(f"{v:.3f}").classes("text-xl font-bold")
            if res.get("per_class"):
                ui.label("Per class").classes("text-lg font-bold")
                pc = res["per_class"]
                ui.table(
                    columns=[
                        {"name": k, "label": k, "field": k, "sortable": True, "align": "left"}
                        for k in pc[0]
                    ],
                    rows=[
                        {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}
                        for r in pc
                    ],
                ).props("dense flat bordered").classes("w-full")
            with ui.grid(columns=2).classes("w-full gap-3"):
                cm = res.get("confusion")
                if cm and cm.get("matrix"):
                    with ui.card().classes("w-full"):
                        with ui.row().classes("items-center"):
                            ui.label("Confusion matrix").classes("font-medium")
                            help_icon(
                                "Rows: what the model predicted; columns: what was really there. The diagonal is right; "
                                "everything else is a mix-up. 'background' = missed objects (column) or false detections (row). "
                                "Click a cell to see those images."
                            )
                        names = list(
                            cm.get("names") or [str(i) for i in range(len(cm["matrix"]) - 1)]
                        )
                        if len(names) < len(cm["matrix"]):
                            names = names + ["background"]
                        heatmap(
                            _normalise_cols(cm["matrix"]),
                            names,
                            names,
                            height="420px",
                            on_click=lambda x, y: self.confusion_images(x, y, names),
                        )
                for name, c in (res.get("curves") or {}).items():
                    with ui.card().classes("w-full"):
                        ui.label(name).classes("font-medium")
                        line_chart(
                            {"all classes": list(zip(c["x"], c["y"]))},
                            height="240px",
                            x_name=c["xlabel"],
                            y_name=c["ylabel"],
                        )
            err = res.get("errors")
            if err:
                self.render_errors(err)

    def render_errors(self, err: dict[str, Any]) -> None:
        ui.label("Where it goes wrong").classes("text-lg font-bold")
        t = err["totals"]
        kinds = {
            "missed": "missed objects",
            "false": "false detections",
            "wrong_class": "wrong class",
            "localisation": "poor box placement",
        }
        with ui.row().classes("gap-4"):
            ui.label(f"right: {t.get('hit', 0)}").classes("text-positive")
            for k, label in kinds.items():
                ui.label(f"{label}: {t.get(k, 0)}")
        rbs = err.get("recall_by_size") or {}
        if rbs:
            ui.label(
                "Found by object size: "
                + ", ".join(
                    f"{b} {v['found']}/{v['objects']}"
                    + (f" ({v['recall']:.0%})" if v["recall"] is not None else "")
                    for b, v in rbs.items()
                )
            ).classes("text-sm")
        detail = read_json(self.run.results_dir / "errors.json", {})
        imgs = [r for r in detail.get("per_image", []) if r["errors"]][:24]
        if not imgs:
            return
        ui.label(
            "Images with the most mistakes (red: missed, orange: false, purple: wrong class, blue: poor placement)"
        ).classes("text-sm text-gray-600")
        names = self._names()
        with ui.element("div").classes("grid grid-cols-2 md:grid-cols-4 gap-2 w-full"):
            for r in imgs:
                with ui.card().classes("p-1").props("flat bordered"):
                    ui.image(draw_errors(r, names)).classes("w-full")
                    ui.label(", ".join(f"{kinds[k]} {r[k]}" for k in kinds if r.get(k))).classes(
                        "text-xs"
                    )

    def confusion_images(self, x: int, y: int, names: list[str]) -> None:
        detail = read_json(self.run.results_dir / "errors.json", {})
        bg = len(names) - 1
        hits = []
        for r in detail.get("per_image", []):
            d = r.get("detail", {})
            if x == bg and y < bg:  # predicted y, nothing there: false detections of class y
                ok = any(o["cls"] == y for o in d.get("false", []))
            elif y == bg and x < bg:  # missed objects of class x
                ok = any(o["cls"] == x for o in d.get("missed", []))
            else:
                ok = any(o["cls"] == y and o.get("true_cls") == x for o in d.get("wrong_class", []))
            if ok:
                hits.append(r)
        with ui.dialog() as dlg, ui.card().classes("w-[90vw] max-w-none"):
            ui.label(f"true {names[x]} / predicted {names[y]}: {len(hits)} image(s)").classes(
                "font-bold"
            )
            if not hits:
                ui.label("No examples in the error list (correct predictions are not stored).")
            with ui.element("div").classes("grid grid-cols-2 md:grid-cols-4 gap-2 w-full"):
                for r in hits[:24]:
                    ui.image(draw_errors(r, self._names())).classes("w-full")
        dlg.open()

    def _names(self) -> dict[int, str]:
        m = read_json(self.run.data_dir / "manifest.json", {})
        return {int(k): v for k, v in (m.get("classes") or {}).items()}

    # ------------------------------------------------------------------------- export
    def render_export(self) -> None:
        self.export_box.clear()
        with self.export_box:
            if not self.run.best_weights:
                ui.label("Export becomes available once the run has weights.").classes(
                    "text-gray-600"
                )
                return
            ui.markdown(
                f"The **model package** is in `{self.run.package_dir}`: `best.pt`, the exports and `model_card.json` "
                "(classes, image size, scores, dataset fingerprint). Other projects read that folder."
            )
            fmt = ui.select(
                {k: f"{v['label']} - {v['where']}" for k, v in FORMATS.items()},
                value="onnx",
                label="Format",
            ).classes("w-[420px]")
            prec = ui.toggle(["fp32", "fp16", "int8"], value="fp32")
            note = ui.label().classes("text-sm text-gray-600")

            def explain() -> None:
                f = FORMATS[fmt.value]
                note.text = (
                    f.get("note", "")
                    + " "
                    + PRECISION_HELP[prec.value]
                    + (
                        ""
                        if prec.value in f["precisions"]
                        else f"  ({f['label']} does not support {prec.value}.)"
                    )
                )

            fmt.on_value_change(lambda _: explain())
            prec.on_value_change(lambda _: explain())
            explain()
            status = ui.label()

            async def go() -> None:
                status.text = "Exporting, re-scoring and timing... (can take a few minutes)"
                try:
                    await run.io_bound(export_model, self.run, fmt.value, prec.value)
                except Exception as exc:  # noqa: BLE001
                    status.text = f"Export failed: {exc}"
                    return
                status.text = "Done."
                self.render_export()

            ui.button("Export", icon="output", on_click=go).mark("export")
            exports = read_json(self.run.package_dir / "exports.json", [])
            if exports:
                rows = [
                    {
                        "format": e["format"],
                        "precision": e["precision"],
                        "size": f"{e['size_mb']} MB",
                        "score": f"{e['main_value']:.3f}"
                        if e.get("main_value") is not None
                        else e.get("rescore_error", "")[:40],
                        "change": f"{e['change_pct']:+.1f} %"
                        if e.get("change_pct") is not None
                        else "",
                        "speed": f"{e['ms_per_image']} ms/img" if e.get("ms_per_image") else "",
                        "path": e["path"],
                    }
                    for e in exports
                ]
                ui.table(
                    columns=[{"name": k, "label": k, "field": k, "align": "left"} for k in rows[0]],
                    rows=rows,
                ).props("dense flat bordered").classes("w-full")
            remotes = [r["name"] for r in ws().settings().get("remotes") or []]
            if remotes and exports:
                with ui.row().classes("items-center"):
                    rsel = ui.select(remotes, value=remotes[0], label="Time it on").classes("w-48")
                    esel = ui.select(
                        {e["path"]: f"{e['format']} {e['precision']}" for e in exports},
                        label="Model",
                        value=exports[-1]["path"],
                    ).classes("w-48")
                    out = ui.label()

                    async def bench() -> None:
                        from ..remote import benchmark_remote

                        out.text = "timing..."
                        try:
                            ms = await run.io_bound(
                                benchmark_remote,
                                ws(),
                                rsel.value,
                                Path(esel.value),
                                int(self.cfg.train_args()["imgsz"]),
                                self.cfg.task,
                            )
                            out.text = f"{ms:.1f} ms per image ({1000 / ms:.1f} images/s)"
                        except Exception as exc:  # noqa: BLE001
                            out.text = f"failed: {exc}"

                    ui.button("Benchmark", on_click=bench).props("outline")

    # ---------------------------------------------------------------- events/settings
    def render_events(self) -> None:
        self.events_box.clear()
        colours = {"info": "text-gray-700", "warn": "text-orange-700", "bad": "text-red-700"}
        with self.events_box:
            for e in reversed(self.run.events()[-50:]):
                ui.label(f"{e['time'][11:19]}  {e['kind']}: {e['text']}").classes(
                    f"text-sm {colours.get(e['level'], '')}"
                )

    def build_settings(self) -> None:
        import yaml

        ui.code(yaml.safe_dump(self.cfg.to_dict(), sort_keys=False), language="yaml").classes(
            "w-full"
        )
        ui.label("Effective Ultralytics arguments:").classes("font-medium")
        ui.code(yaml.safe_dump(self.cfg.train_args(), sort_keys=True), language="yaml").classes(
            "w-full"
        )
        ui.label(f"Folder: {self.run.path}").classes("text-xs text-gray-500")

    # ------------------------------------------------------------------------ actions
    async def start(self) -> None:
        how = await run.io_bound(self.mgr.start, self.run)
        ui.notify(
            "started" if how == "started" else "queued: another run is training", type="positive"
        )
        self.tick(force=True)

    async def resume(self) -> None:
        try:
            how = await run.io_bound(self.mgr.resume, self.run)
        except Exception as exc:  # noqa: BLE001
            ui.notify(str(exc), type="negative")
            return
        ui.notify("resuming" if how == "started" else "queued", type="positive")
        self.tick(force=True)

    def pause(self) -> None:
        self.mgr.pause(self.run)
        ui.notify("Pausing at the end of this epoch", type="info")

    def stop(self, now: bool) -> None:
        self.mgr.stop(self.run, now)
        ui.notify("Stopping" + (" now" if now else " after this epoch"), type="info")

    async def finish_paused(self) -> None:
        self.run.set_state("stopped")
        await self.evaluate()

    async def evaluate(self) -> None:
        ui.notify("Evaluating... (a minute)", type="info")
        try:
            res = await run.io_bound(evaluate_run, self.run)
            await run.io_bound(package, self.run, res)
        except Exception as exc:  # noqa: BLE001
            ui.notify(f"Evaluation failed: {exc}", type="negative")
        self.tick(force=True)

    def extend_dialog(self) -> None:
        m = self.run.metrics()
        done = m[-1]["epoch"] if m else 0
        with ui.dialog() as d, ui.card():
            ui.label(f"Train longer: the run has trained {done} epochs.")
            n = ui.number(
                "New total number of epochs", value=done + max(20, done // 2), min=done + 1, step=10
            )
            ui.label(
                "Training continues from the last epoch with its optimizer state; early stopping patience is raised so it does not stop immediately."
            ).classes("text-sm text-gray-600")

            async def go() -> None:
                d.close()
                try:
                    await run.io_bound(self.mgr.extend, self.run, int(n.value))
                except Exception as exc:  # noqa: BLE001
                    ui.notify(str(exc), type="negative")
                self.tick(force=True)

            ui.button("Continue training", on_click=go)
        d.open()

    def delete(self) -> None:
        with ui.dialog() as d, ui.card():
            ui.label(f"Delete run {self.run.id} and all its files (weights, results)?")
            with ui.row():
                ui.button(
                    "Delete",
                    color="negative",
                    on_click=lambda: (
                        self.mgr.delete(self.run),
                        d.close(),
                        ui.navigate.to("/runs"),
                    ),
                )
                ui.button("Cancel", on_click=d.close).props("flat")
        d.open()


def _normalise_cols(m: list[list[float]]) -> list[list[float]]:
    """Each column (true class) as fractions, so rare classes are as readable as common ones."""
    n = len(m)
    sums = [sum(m[r][c] for r in range(n)) or 1 for c in range(n)]
    return [[m[r][c] / sums[c] for c in range(n)] for r in range(n)]


def draw_errors(r: dict[str, Any], names: dict[int, str], side: int = 640) -> Image.Image:
    colours = {
        "missed": (220, 38, 38),
        "false": (234, 140, 0),
        "wrong_class": (147, 51, 234),
        "localisation": (37, 99, 235),
    }
    try:
        with Image.open(r["image"]) as im:
            img = im.convert("RGB")
    except OSError:
        return Image.new("RGB", (side, side), (80, 80, 80))
    w, h = img.size
    d = ImageDraw.Draw(img)
    lw = max(2, round(max(w, h) / 300))
    for kind, objs in (r.get("detail") or {}).items():
        for o in objs:
            cx, cy, bw, bh = o["box"]
            box = [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]
            d.rectangle(box, outline=colours[kind], width=lw)
            text = names.get(o["cls"], str(o["cls"]))
            if kind == "wrong_class":
                text = f"{text} (is {names.get(o['true_cls'], o['true_cls'])})"
            d.text((box[0] + 2, box[1] + 2), text, fill=colours[kind])
    img.thumbnail((side, side))
    return img
