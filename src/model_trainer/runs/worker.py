"""The training worker: one process per run, started by the manager, never inside the GUI.

    python -m model_trainer.runs.worker <run folder> [start|resume|extend:<epochs>]

What makes runs safe:
- after every epoch a full checkpoint (with the optimizer) is copied atomically to
  checkpoints/resume.pt and verified; Ultralytics strips the optimizer from its own last.pt at the
  end of training, so resume.pt is what pause/resume, crash recovery and "extend" use;
- the run's dataset is frozen in the run folder and checked (sizes, optionally SHA-1) before resuming;
- a heartbeat file lets the GUI tell a running worker from a crashed one;
- the computer is kept awake while training.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import platform
import shutil
import sys
import threading
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any

from .. import notify
from ..datasets import prepare as prep
from ..datasets import store as ds_store
from ..models import weights_name
from ..util import (
    append_jsonl,
    atomic_copy,
    free_disk_gb,
    gpu_status,
    keep_awake,
    now,
    pid_alive,
    read_json,
    trust_system_certificates,
    write_json,
)
from ..workspace import Project, Workspace
from .config import RunConfig
from .store import HEARTBEAT_S, Run

log = logging.getLogger("model_trainer.worker")

MIN_FREE_GB = 2.0


class Worker:
    def __init__(self, run: Run) -> None:
        self.run = run
        self.cfg: RunConfig = run.config
        # <workspace>/projects/<project>/runs/<run>
        self.workspace = Workspace(run.path.parents[3])
        self.project = Project(run.path.parents[1])
        self.settings = self.workspace.settings()
        self._stop_hb = threading.Event()
        self.pause_requested = False
        self.stop_requested = False
        self.in_final_eval = False
        self.last_progress = 0.0
        self.warned: set[str] = set()
        self.best_fitness = -math.inf
        self.best_epoch = 0
        self.batch_used: int | None = None  # what Ultralytics trains with (after AutoBatch)

    # ------------------------------------------------------------------------ helpers
    def event(
        self, kind: str, text: str, level: str = "info", notify_event: str | None = None
    ) -> None:
        append_jsonl(
            self.run.path / "events.jsonl",
            {"time": now(), "kind": kind, "level": level, "text": text},
        )
        log.info("[%s] %s", kind, text)
        if notify_event and self.cfg.options.notify:
            notify.send(self.settings, notify_event, f"{self.cfg.name}: {kind}", text)

    def _heartbeat_loop(self) -> None:
        while not self._stop_hb.wait(HEARTBEAT_S):
            self.run.heartbeat(os.getpid())

    def write_env(self) -> None:
        env: dict[str, Any] = {
            "time": now(),
            "python": sys.version,
            "platform": platform.platform(),
            "argv": sys.argv,
            "gpus": gpu_status(),
        }
        with contextlib.suppress(Exception):
            import torch
            import ultralytics

            env["torch"] = torch.__version__
            env["cuda"] = torch.version.cuda
            env["ultralytics"] = ultralytics.__version__
        from .. import __version__

        env["model_trainer"] = __version__
        write_json(self.run.path / "env.json", env)

    # ------------------------------------------------------------------------ dataset
    def prepare_data(self) -> None:
        manifest = self.run.data_dir / "manifest.json"
        data_yaml = self.run.data_dir / "data.yaml"
        if (
            data_yaml.is_file()
        ):  # the run folder was moved (other disk, remote machine): fix the path
            data = _yaml(data_yaml)
            if Path(str(data.get("path", ""))) != self.run.data_dir:
                from ..util import write_yaml

                data["path"] = str(self.run.data_dir)
                write_yaml(data_yaml, data)
        if manifest.is_file():
            problems = prep.verify(self.run.data_dir)
            if problems:
                raise RuntimeError(
                    "the run's frozen dataset was modified, so resuming would mix two datasets:\n  "
                    + "\n  ".join(problems[:10])
                    + "\nStart a new run (copy this one) instead."
                )
            return
        self.run.set_state("preparing")
        ds, m = ds_store.open_dataset(self.project, self.cfg.dataset)
        if m.get("changed"):
            self.event(
                "dataset", "the dataset files changed since import; using the current files", "warn"
            )
        out = prep.prepare(
            ds,
            self.run.data_dir,
            self.cfg.task,
            prep.ClassPlan.from_dict(self.cfg.classes),
            prep.SplitPlan.from_dict(self.cfg.split),
            subset=float(self.cfg.subset),
            balance=self.cfg.balance,
        )
        out["dataset_fingerprint"] = m["fingerprint"]
        write_json(manifest, out)
        self.event(
            "dataset", f"prepared {out['images']} images, classes {list(out['classes'].values())}"
        )

    def data_arg(self) -> str:
        if self.cfg.task == "classify":
            return str(self.run.data_dir)
        return str(self.run.data_dir / "data.yaml")

    # ------------------------------------------------------------------------- model
    def start_weights(self) -> str:
        m = self.cfg.model
        name = weights_name(m.family, m.size, self.cfg.task)
        if m.start == "scratch":
            return name.replace(".pt", ".yaml")
        if m.start == "run":
            other = self.workspace.find_run(m.source)
            w = Run(other).best_weights if other else None
            if w is None:
                raise RuntimeError(f"run {m.source!r} has no weights to start from")
            return str(w)
        if m.start == "file":
            if not Path(m.source).is_file():
                raise RuntimeError(f"weights file {m.source} not found")
            return m.source
        return name

    # --------------------------------------------------------------------- callbacks
    def attach(self, model: Any) -> None:
        model.add_callback("on_train_start", self.on_train_start)
        model.add_callback("on_train_batch_end", self.on_batch_end)
        model.add_callback("on_model_save", self.on_model_save)
        model.add_callback("on_fit_epoch_end", self.on_epoch_end)

    def on_train_start(self, trainer: Any) -> None:
        original = trainer.final_eval

        def final_eval() -> None:
            if self.pause_requested:  # pausing: no point validating best.pt now
                return
            self.in_final_eval = True
            original()

        trainer.final_eval = final_eval
        self.batch_used = int(trainer.batch_size)
        self.run.set_state(
            "running",
            epochs=int(trainer.epochs),
            start_epoch=int(trainer.start_epoch),
            batch=int(trainer.batch_size),
            started=now(),
            pid=os.getpid(),
        )
        self.event(
            "start",
            f"training epochs {trainer.start_epoch + 1}-{trainer.epochs}, batch {trainer.batch_size}",
        )

    def on_batch_end(self, trainer: Any) -> None:
        i = getattr(trainer, "_mt_batch", 0) + 1
        trainer._mt_batch = i
        t = time.time()
        if t - self.last_progress < 2:
            return
        self.last_progress = t
        nb = len(trainer.train_loader) if trainer.train_loader else 0
        write_json(
            self.run.path / "progress.json",
            {
                "epoch": int(trainer.epoch) + 1,
                "epochs": int(trainer.epochs),
                "batch": i,
                "batches": nb,
                "time": t,
            },
        )
        cmd = self.run.pending_command()
        if cmd == "stop_now":
            self.stop_requested = True
            trainer.stop = True
            self.run.set_state("stopping")

    def on_model_save(self, trainer: Any) -> None:
        last = Path(trainer.last)
        if not last.is_file():
            return
        dst = self.run.resume_ckpt
        tmp = dst.with_name("resume.pt.new")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(last, tmp)
        if not _checkpoint_ok(tmp):
            self.event(
                "checkpoint", "the new checkpoint is damaged; keeping the previous one", "warn"
            )
            tmp.unlink(missing_ok=True)
            return
        os.replace(tmp, dst)
        every = int(self.cfg.options.snapshot_every or 0)
        epoch = int(trainer.epoch) + 1
        if every > 0 and epoch % every == 0:
            snap = self.run.snapshots_dir / f"epoch_{epoch:04d}.pt"
            atomic_copy(last, snap)
            snaps = sorted(self.run.snapshots_dir.glob("epoch_*.pt"))
            for old in snaps[: max(0, len(snaps) - int(self.cfg.options.keep_snapshots))]:
                old.unlink(missing_ok=True)

    def on_epoch_end(self, trainer: Any) -> None:
        if self.in_final_eval:
            return  # Ultralytics reports the final best.pt validation through the same callback
        trainer._mt_batch = 0
        epoch = int(trainer.epoch) + 1
        rec: dict[str, Any] = {
            "epoch": epoch,
            "time": now(),
            "epoch_s": round(float(trainer.epoch_time or 0), 2),
        }
        with contextlib.suppress(Exception):
            rec.update(
                {
                    k: float(v)
                    for k, v in trainer.label_loss_items(trainer.tloss, prefix="train").items()
                }
            )
        rec.update({k: float(v) for k, v in (trainer.metrics or {}).items() if _num(v)})
        rec.update({k: float(v) for k, v in (trainer.lr or {}).items()})
        rec["fitness"] = float(trainer.fitness) if trainer.fitness is not None else None
        with contextlib.suppress(Exception):
            import torch

            if torch.cuda.is_available():
                rec["gpu_mem_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 2)
        gpus = gpu_status()
        if gpus:
            rec["gpu_temp_c"] = gpus[0]["temperature_c"]
            rec["gpu_load_pct"] = gpus[0]["load_pct"]
        rec["batch"] = int(trainer.batch_size)
        with contextlib.suppress(Exception):  # per-class scores of this epoch's validation
            rows = trainer.validator.metrics.summary()
            rec["per_class"] = [
                {
                    k: (float(v) if isinstance(v, (int, float)) or hasattr(v, "item") else v)
                    for k, v in r.items()
                }
                for r in rows
            ]
        append_jsonl(self.run.path / "metrics.jsonl", rec)
        if rec["fitness"] is not None and rec["fitness"] > self.best_fitness:
            self.best_fitness, self.best_epoch = rec["fitness"], epoch
        self.check_health(rec, trainer)
        if self.cfg.options.watch_images:
            with contextlib.suppress(Exception):
                self.watch(trainer, epoch)
        cmd = self.run.pending_command()
        if cmd == "pause":
            self.pause_requested = True
            trainer.stop = True
            self.run.set_state("pausing")
        elif cmd in ("stop", "stop_now"):
            self.stop_requested = True
            trainer.stop = True
            self.run.set_state("stopping")

    def check_health(self, rec: dict[str, Any], trainer: Any) -> None:
        hist = self.run.metrics()
        val_loss = [
            sum(v for k, v in r.items() if k.startswith("val/") and "loss" in k) for r in hist
        ]
        train_loss = [
            sum(v for k, v in r.items() if k.startswith("train/") and "loss" in k) for r in hist
        ]
        if len(hist) >= 8 and "overfit" not in self.warned:
            recent_val = val_loss[-5:]
            recent_train = train_loss[-5:]
            if (
                all(b > a for a, b in zip(recent_val, recent_val[1:]))
                and recent_train[-1] < recent_train[0]
            ):
                self.warned.add("overfit")
                self.event(
                    "overfitting",
                    "validation loss has risen for 5 epochs while training loss keeps falling: the model starts "
                    "memorising. More data or stronger augmentation help; early stopping will keep the best epoch.",
                    "warn",
                    "warning",
                )
        if not all(_num(v) for k, v in rec.items() if "loss" in k) and "nan" not in self.warned:
            self.warned.add("nan")
            self.event(
                "nan",
                "a loss became NaN/Inf. Ultralytics will reload the last good epoch; if it keeps "
                "happening lower the learning rate or turn off AMP.",
                "bad",
                "warning",
            )
        patience = int(getattr(trainer.args, "patience", 0) or 0)
        since = rec["epoch"] - self.best_epoch
        if patience and since >= max(5, patience // 2) and "plateau" not in self.warned:
            self.warned.add("plateau")
            self.event(
                "plateau",
                f"no improvement for {since} epochs (best at epoch {self.best_epoch}). "
                f"Early stopping ends the run after {patience} epochs without improvement.",
                "info",
            )
        temp = rec.get("gpu_temp_c")
        if temp and temp >= 85 and "hot" not in self.warned:
            self.warned.add("hot")
            self.event(
                "gpu_temperature",
                f"GPU at {temp:.0f} C: check the case airflow.",
                "warn",
                "warning",
            )
        free = free_disk_gb(self.run.path)
        if free < MIN_FREE_GB and "disk" not in self.warned:
            self.warned.add("disk")
            self.event(
                "disk",
                f"only {free:.1f} GB free: pausing so checkpoints cannot fail. Free some space, then resume.",
                "bad",
                "warning",
            )
            self.run.request("pause")

    def watch(self, trainer: Any, epoch: int) -> None:
        """Predictions on the same few validation images after each epoch ("watch it learn")."""
        from ultralytics import YOLO

        images = self._watch_images()
        if not images or not Path(trainer.last).is_file():
            return
        out = self.run.watch_dir / f"epoch_{epoch:04d}"
        out.mkdir(parents=True, exist_ok=True)
        model = YOLO(str(trainer.last))
        imgsz = int(trainer.args.imgsz)
        for i, img in enumerate(images):
            r = model.predict(
                str(img), imgsz=imgsz, conf=0.25, verbose=False, device=trainer.device
            )[0]
            from PIL import Image

            im = Image.fromarray(r.plot()[..., ::-1])
            im.thumbnail((480, 480))
            im.save(out / f"{i:02d}.jpg", quality=80)
        del model
        # keep every epoch of the last 10, then every 5th
        olds = sorted(self.run.watch_dir.glob("epoch_*"))[:-10]
        for d in olds:
            if int(d.name.split("_")[1]) % 5:
                shutil.rmtree(d, ignore_errors=True)

    def _watch_images(self) -> list[Path]:
        if not hasattr(self, "_watch"):
            n = int(self.cfg.options.watch_images)
            if self.cfg.task == "classify":
                imgs = sorted(p for p in (self.run.data_dir / "val").rglob("*") if p.is_file())
            else:
                imgs = sorted((self.run.data_dir / "images" / "val").glob("*"))
            step = max(1, len(imgs) // max(n, 1))
            self._watch = imgs[::step][:n]
        return self._watch

    # --------------------------------------------------------------------------- main
    def train(self, mode: str) -> str:
        self.prepare_data()
        if free_disk_gb(self.run.path) < MIN_FREE_GB:
            raise RuntimeError(f"less than {MIN_FREE_GB} GB of free disk space")
        args = self.cfg.train_args()
        if args.get("cache") == "auto":
            args["cache"] = self._auto_cache(int(args["imgsz"])) if mode == "start" else None
        device = self.cfg.options.device or None
        try:
            self._fit(mode, args, device)
        except Exception as exc:  # noqa: BLE001 - only out-of-memory is retried
            lighter = self._lighter_settings(exc, args)
            if lighter is None:
                raise
            change, text = lighter
            self.event("out of memory", text, "warn")
            cfg = self.cfg
            cfg.train.update(change)  # the restart and later resumes keep what fits
            self.run.save_config(cfg)
            # Not retried in this process: the failed attempt keeps its GPU memory and its data
            # loader processes alive, so every in-process retry had less memory than the last.
            return "relaunch"
        if self.pause_requested:
            return "paused"
        return "stopped" if self.stop_requested else "done"

    def _auto_cache(self, imgsz: int) -> str | bool:
        """'ram' when the run's images fit in the free memory now (cache: auto), otherwise off."""
        from ..models import cache_ram_gb, free_ram_gb, ram_cache_fits

        n = sum(read_json(self.run.data_dir / "manifest.json", {}).get("images", {}).values())
        need, free = cache_ram_gb(n, imgsz), free_ram_gb()
        if ram_cache_fits(n, imgsz, free):
            self.event(
                "cache",
                f"images are kept in RAM for faster epochs (about {need:.1f} GB, "
                f"{free:.1f} GB free)",
            )
            return "ram"
        if free is not None:
            self.event(
                "cache",
                f"images are read from disk every epoch: keeping them in RAM needs about "
                f"{need:.1f} GB plus room for training, {free:.1f} GB free",
            )
        return False

    def _lighter_settings(
        self, exc: BaseException, args: dict[str, Any]
    ) -> tuple[dict[str, Any], str] | None:
        """Settings that need less memory after an out-of-memory error; None = give up.

        AutoBatch sizes the batch on ordinary images; images with hundreds of objects need much
        more memory in the loss (target assignment), so the GPU can still run out: half the batch.
        The PC's RAM runs out in the data loader (MemoryError, paging file): the image cache goes
        first, then half the workers (each holds its own images and masks), then half the batch.
        """
        text = f"{type(exc).__name__} {exc}".lower()
        memory = any(
            s in text
            for s in (
                "out of memory",
                "memoryerror",
                "not enough memory",
                "paging file is too small",
            )
        )
        batch = int(self.batch_used or 0)
        if not memory:
            return None
        if "cuda" not in text:  # the PC's RAM, not the GPU
            if args.get("cache") == "ram":
                return {"cache": False}, (
                    "the PC ran out of memory (RAM) with the images kept in RAM: "
                    "restarting without the image cache"
                )
            workers = int(args.get("workers") or 0)
            if workers > 2:
                return {"workers": workers // 2}, (
                    f"the PC ran out of memory (RAM) with {workers} data loader workers: "
                    f"restarting with {workers // 2}"
                )
            if batch > 1:
                return {"batch": batch // 2}, (
                    f"the PC ran out of memory (RAM) with batch {batch}: "
                    f"restarting with batch {batch // 2}"
                )
            return None
        if batch <= 1:
            return None
        return {"batch": batch // 2}, (
            f"the GPU ran out of memory with batch {batch}: restarting with batch {batch // 2}"
        )

    def relaunch_mode(self, mode: str) -> str:
        """How the restarted process continues: keep the finished epochs when there are any."""
        if mode.startswith("extend:"):
            return mode
        return "resume" if self.run.resume_ckpt.is_file() else "start"

    def _fit(self, mode: str, args: dict[str, Any], device: str | None) -> None:
        from ultralytics import YOLO

        if mode == "start":
            if self.cfg.task == "pose" and float(args.get("fliplr", 0)) > 0:
                data = json.loads(json.dumps(_yaml(self.run.data_dir / "data.yaml")))
                if not data.get("flip_idx"):
                    args["fliplr"] = 0.0
                    self.event(
                        "augmentation",
                        "horizontal flip turned off: the dataset does not say which "
                        "keypoints swap when mirrored (flip_idx)",
                        "info",
                    )
            model = YOLO(self.start_weights())
            self.attach(model)
            model.train(
                data=self.data_arg(),
                project=str(self.run.path / "ultra"),
                name="train",
                exist_ok=True,
                device=device,
                save_period=-1,
                plots=True,
                **{k: v for k, v in args.items() if v is not None},
            )
        else:
            ckpt = self.run.resume_ckpt
            if not ckpt.is_file():
                raise RuntimeError("no checkpoint to resume from (the run never finished an epoch)")
            last = self.run.weights_dir / "last.pt"
            overrides: dict[str, Any] = {}
            if mode.startswith("extend:"):
                epochs = int(mode.split(":", 1)[1])
                _set_epochs(ckpt, last, epochs)
                overrides["patience"] = max(int(args.get("patience", 50)), epochs)
                cfg = self.cfg
                cfg.train["epochs"] = epochs
                self.run.save_config(cfg)
                self.event("extend", f"extending to {epochs} epochs")
            else:
                atomic_copy(ckpt, last)
            if int(args.get("batch") or -1) > 0:
                overrides["batch"] = int(args["batch"])  # e.g. lowered after running out of memory
            if "workers" in self.cfg.train:
                overrides["workers"] = int(args["workers"])
            if self.cfg.train.get("cache", "auto") != "auto":  # auto: as decided at the start
                overrides["cache"] = self.cfg.train["cache"]
            model = YOLO(str(last))
            self.attach(model)
            model.train(resume=True, device=device, **overrides)

    def finish(self, outcome: str) -> None:
        from ..evaluate import evaluate_run, package

        if outcome == "paused":
            self.run.set_state("paused")
            self.event(
                "paused", "paused after the epoch; resume continues exactly here", "info", "paused"
            )
            return
        self.run.set_state("evaluating", outcome=outcome)
        try:
            res = evaluate_run(self.run)
            package(self.run, res)
            summary = res.get("summary", "")
        except Exception as exc:  # noqa: BLE001 - the weights are safe even if evaluation fails
            log.exception("evaluation failed")
            summary = f"training {outcome}, but the evaluation failed: {exc}"
            self.event("evaluation", summary, "warn")
        self.run.set_state(outcome, finished=now())
        verb = "finished" if outcome == "done" else "stopped"
        self.event(verb, summary or verb, "info", "finished")


def _checkpoint_ok(path: Path) -> bool:
    """A torch checkpoint is a zip file: test every member's CRC."""
    try:
        with zipfile.ZipFile(path) as z:
            return z.testzip() is None
    except (zipfile.BadZipFile, OSError):
        return False


def _set_epochs(src: Path, dst: Path, epochs: int) -> None:
    import torch

    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    done = int(ckpt.get("epoch", -1)) + 1
    if epochs <= done:
        raise RuntimeError(f"the run already trained {done} epochs; choose more than that")
    ckpt["train_args"]["epochs"] = epochs
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    torch.save(ckpt, tmp)
    os.replace(tmp, dst)


def _num(v: Any) -> bool:
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _yaml(path: Path) -> dict[str, Any]:
    from ..util import read_yaml

    return read_yaml(path) or {}


def _friendly(exc: BaseException) -> str:
    text = str(exc)
    low = text.lower()
    if "out of memory" in low:
        return (
            "the GPU ran out of memory. Lower the batch size or the image size, or pick a smaller model. "
            f"({text.splitlines()[0][:200]})"
        )
    if "download failure" in low or "certificate" in low:
        return (
            "could not download the base model. Check the internet connection, or download the .pt file "
            "yourself and choose 'start from a weights file'. " + text.splitlines()[0][:200]
        )
    if "no such file" in low or "not found" in low:
        return f"a file is missing: {text[:300]}"
    return text[:500]


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    after = next((int(a.split("=", 1)[1]) for a in argv if a.startswith("--after=")), None)
    argv = [a for a in argv if not a.startswith("--after=")]
    run = Run(Path(argv[0]))
    mode = argv[1] if len(argv) > 1 else "start"
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout
    )
    if after:  # restarted after running out of memory: wait until the old process let go of it
        deadline = time.time() + 300
        while pid_alive(after) and time.time() < deadline:
            time.sleep(1)
    trust_system_certificates()
    w = Worker(run)
    run.heartbeat(os.getpid())
    hb = threading.Thread(target=w._heartbeat_loop, daemon=True)
    hb.start()
    run.clear_command()
    run.update_state(pid=os.getpid(), mode=mode)
    w.write_env()
    code = 0
    outcome = ""
    with keep_awake(f"training {run.id}"):
        try:
            outcome = w.train(mode)
            if outcome != "relaunch":
                w.finish(outcome)
        except BaseException as exc:  # noqa: BLE001 - every failure must end up in state.json
            traceback.print_exc()
            reason = _friendly(exc)
            if isinstance(exc, KeyboardInterrupt):
                run.set_state("interrupted", error="interrupted from the keyboard")
            else:
                run.set_state("failed", error=reason, finished=now())
                w.event("failed", reason, "bad", "failed")
            code = 1
        finally:
            run.clear_command()
            w._stop_hb.set()
            (run.path / "progress.json").unlink(missing_ok=True)
    from .manager import Manager

    if outcome == "relaunch":
        Manager(w.workspace).start_process(run, w.relaunch_mode(mode), after=os.getpid())
        return code
    with contextlib.suppress(Exception):
        Manager(w.workspace).start_next_queued()
    return code


if __name__ == "__main__":
    sys.exit(main())
