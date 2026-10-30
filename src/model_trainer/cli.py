"""mtrain: everything the GUI does, from the terminal.

mtrain gui                                   open the app in the browser
mtrain dataset import <folder|yaml|json>     register a dataset (YOLO, COCO, class folders)
mtrain dataset list | check <name>           health report in the terminal
mtrain run new --dataset shapes --task detect --model yolo26n --epochs 100 [--start]
mtrain run list | status | start | pause | resume | stop | extend | eval | export | copy | delete
mtrain predict --run <id> --source <images>  playground in the terminal
mtrain sweep create | start | results        hyperparameter sweeps
mtrain remote check | bench                  SSH machines and Raspberry Pi benchmarks
mtrain notify-test                           send a test notification to your phone
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .util import read_yaml, trust_system_certificates
from .workspace import Workspace


def _out(text: Any = "") -> None:
    if not isinstance(text, str):
        text = json.dumps(text, indent=2, default=str)
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def _ws(args: argparse.Namespace) -> Workspace:
    return Workspace(args.workspace)


def _run(ws: Workspace, run_id: str):
    from .runs.store import Run

    p = ws.find_run(run_id)
    if p is None:
        matches = [r for r in ws.all_runs() if run_id in r.name]
        if len(matches) == 1:
            p = matches[0]
    if p is None:
        raise SystemExit(f"no run {run_id!r} (mtrain run list)")
    return Run(p)


# ------------------------------------------------------------------------------- dataset


def cmd_dataset(args: argparse.Namespace) -> int:
    from .datasets import store

    ws = _ws(args)
    proj = ws.project(args.project)
    if args.action == "import":
        m = store.import_dataset(proj, args.path, args.name, "obb" if args.obb else None, args.copy)
        _out(
            f"imported {m['name']}: {m['images']} images, {len(m['classes'])} classes, labels: {m['label_kind']}"
        )
    elif args.action == "list":
        for name in proj.datasets():
            m = store.manifest(proj, name)
            _out(
                f"{name:30s} {m['format']:8s} {m['label_kind']:10s} {sum(m['images'].values()):7d} images  {len(m['classes'])} classes"
            )
    elif args.action == "check":
        rep = store.health_report(proj, args.name, args.imgsz, force=args.force)
        marks = {"ok": "OK  ", "info": "INFO", "warn": "WARN", "bad": "BAD "}
        for c in rep["checks"]:
            _out(f"[{marks[c['level']]}] {c['title']}: {c['value']}")
            if c["level"] in ("warn", "bad") or args.verbose:
                _out(f"       why: {c['why']}\n       do:  {c['action']}")
        _out(f"recommended image size: {rep['recommended_imgsz']}")
    return 0


# ----------------------------------------------------------------------------------- run


def _parse_model(s: str) -> tuple[str, str]:
    from .models import FAMILIES

    for fam in sorted(FAMILIES, key=len, reverse=True):
        if s.startswith(fam):
            size = s[len(fam) :].lstrip("-")
            return fam, size or "n"
    raise SystemExit(f"unknown model {s!r}: e.g. yolo26n, yolo11s, yolov8m, rtdetr-l")


def cmd_run(args: argparse.Namespace) -> int:
    from .runs.config import ModelChoice, RunConfig
    from .runs.manager import Manager

    ws = _ws(args)
    mgr = Manager(ws)
    a = args.action
    if a == "new":
        proj = ws.project(args.project)
        if args.from_yaml:
            cfg = RunConfig.from_dict(read_yaml(args.from_yaml))
        else:
            from .datasets import store

            if not args.dataset:
                raise SystemExit("--dataset is required")
            m = store.manifest(proj, args.dataset)
            fam, size = _parse_model(args.model)
            train: dict[str, Any] = {}
            for k in ("epochs", "imgsz", "batch", "patience"):
                if getattr(args, k) is not None:
                    train[k] = getattr(args, k)
            for kv in args.set or []:
                k, v = kv.split("=", 1)
                train[k] = _value(v)
            cfg = RunConfig(
                name=args.name or f"{args.dataset}-{args.model}",
                project=proj.name,
                dataset=args.dataset,
                task=args.task or m["default_task"],
                model=ModelChoice(fam, size),
                train=train,
                aug_preset=args.aug,
                balance="oversample" if args.oversample else "none",
                split={"mode": "resplit"} if args.resplit else {"mode": "keep"},
                subset=args.subset,
            )
            if args.drop:
                cfg.classes["drop"] = args.drop
        run = mgr.create(proj, cfg)
        _out(f"created {run.id}")
        if args.start:
            _out(mgr.start(run))
        return 0
    if a == "list":
        for p in ws.all_runs():
            from .runs.store import Run

            s = Run(p).summary()
            score = f"{s['main_value']:.3f}" if s["main_value"] is not None else "-"
            _out(
                f"{s['id']:45s} {s['state']:11s} {s['model']:9s} epoch {s['epoch'] or 0}/{s['epochs'] or '?'}  score {score}"
            )
        q = mgr.queue()
        if q:
            _out("queue: " + ", ".join(Path(e["run"]).name for e in q))
        return 0
    run = _run(ws, args.run)
    if a == "status":
        st = run.state()
        _out({k: v for k, v in st.items() if k != "history"})
        m = run.metrics()
        if m:
            _out(m[-1])
        if args.follow:
            while run.state().get("state") in (
                "preparing",
                "running",
                "pausing",
                "stopping",
                "evaluating",
            ):
                p = run.progress()
                last = run.metrics()[-1] if run.metrics() else {}
                _out(
                    f"epoch {p.get('epoch', '?')}/{p.get('epochs', '?')} batch {p.get('batch', '?')}/{p.get('batches', '?')}  fitness {last.get('fitness')}"
                )
                time.sleep(10)
            _out(run.state().get("state"))
    elif a == "start":
        _out(mgr.start(run))
    elif a == "resume":
        _out(mgr.resume(run))
    elif a == "pause":
        mgr.pause(run)
        _out("pause requested: the run pauses at the end of the current epoch")
    elif a == "stop":
        mgr.stop(run, now=args.now)
        _out("stop requested")
    elif a == "extend":
        _out(mgr.extend(run, args.epochs))
    elif a == "copy":
        _out(mgr.copy(run, args.name).id)
    elif a == "delete":
        mgr.delete(run)
        _out("deleted")
    elif a == "eval":
        from .evaluate import evaluate_external, evaluate_run, package

        if args.data:
            res = evaluate_external(run, Path(args.data), Path(args.data).stem)
        else:
            res = evaluate_run(run)
            package(run, res)
        _out(res["summary"])
    elif a == "export":
        from .export import export_model

        rec = export_model(run, args.format, args.precision)
        _out(rec)
    return 0


def _value(v: str) -> Any:
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    return v


# ------------------------------------------------------------------------------ predict


def cmd_predict(args: argparse.Namespace) -> int:
    from .predict import VIDEO_EXTS, predict_images, predict_video
    from .util import IMAGE_EXTS

    ws = _ws(args)
    weights = Path(args.weights) if args.weights else _run(ws, args.run).best_weights
    if weights is None:
        raise SystemExit("the run has no weights yet")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sources: list[Path] = []
    for s in args.source:
        p = Path(s)
        sources += (
            sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS)
            if p.is_dir()
            else [p]
        )
    images = [p for p in sources if p.suffix.lower() in IMAGE_EXTS]
    for pred in predict_images(weights, images, args.conf, args.iou):
        pred.plotted.save(out / (Path(pred.image).stem + "_pred.jpg"))
        what = ", ".join(f"{o['class']} {o['conf']:.2f}" for o in pred.objects) or ", ".join(
            f"{n} {p:.2f}" for n, p in pred.top[:3]
        )
        _out(f"{Path(pred.image).name}: {what or 'nothing found'} ({pred.ms:.0f} ms)")
    for v in (p for p in sources if p.suffix.lower() in VIDEO_EXTS):
        _out(predict_video(weights, v, out / (v.stem + "_pred.mp4"), args.conf, args.iou))
    return 0


# -------------------------------------------------------------------------------- sweeps


def cmd_sweep(args: argparse.Namespace) -> int:
    from . import sweeps

    ws = _ws(args)
    if args.action == "create":
        base = _run(ws, args.base)
        space = read_yaml(args.space)
        from .workspace import Project

        s = sweeps.create_sweep(
            ws,
            Project(base.path.parents[1]),
            base.config,
            space,
            args.mode,
            args.max_runs,
            args.budget_hours,
            name=args.name,
        )
        _out(f"sweep {s['id']}: {len(s['runs'])} runs created (mtrain sweep start {s['id']})")
    elif args.action == "start":
        _out(sweeps.start_sweep(ws, args.id))
    elif args.action == "results":
        for row in sweeps.results(ws, args.id):
            _out(row)
    return 0


def cmd_remote(args: argparse.Namespace) -> int:
    from . import remote

    ws = _ws(args)
    if args.action == "check":
        ok, msg = remote.get_remote(ws, args.name).check()
        _out(("OK: " if ok else "FAILED: ") + msg)
        return 0 if ok else 1
    run = _run(ws, args.run)
    model = Path(args.model) if args.model else run.package_dir / "best.pt"
    ms = remote.benchmark_remote(
        ws, args.name, model, int(run.config.train_args()["imgsz"]), run.config.task
    )
    _out(f"{args.name}: {ms:.1f} ms per image ({1000 / ms:.1f} images/s)")
    return 0


def cmd_notify(args: argparse.Namespace) -> int:
    from .notify import send

    ok = send(_ws(args).settings(), "test", "model-trainer", "Test notification: it works.")
    _out("sent" if ok else "not sent: set a topic in the Settings page (or settings.json)")
    return 0 if ok else 1


def cmd_gui(args: argparse.Namespace) -> int:
    from .gui.app import run_app

    run_app(Workspace(args.workspace), port=args.port, show=not args.no_browser)
    return 0


# ------------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--workspace", help="workspace folder (default MTRAIN_HOME or ~/model-trainer-workspace)"
    )
    common.add_argument("--project", help="project name (default: the last one used)")
    common.add_argument("-v", "--verbose", action="store_true")
    p = argparse.ArgumentParser(
        prog="mtrain", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("gui", parents=[common], help="open the app in the browser")
    g.add_argument("--port", type=int, default=8090)
    g.add_argument("--no-browser", action="store_true")
    g.set_defaults(func=cmd_gui)

    d = sub.add_parser("dataset", help="import and check datasets")
    ds = d.add_subparsers(dest="action", required=True)
    x = ds.add_parser("import", parents=[common])
    x.add_argument("path")
    x.add_argument("--name")
    x.add_argument(
        "--copy", action="store_true", help="copy (hard-link) the files into the workspace"
    )
    x.add_argument("--obb", action="store_true", help="8-number label lines are rotated boxes")
    ds.add_parser("list", parents=[common])
    x = ds.add_parser("check", parents=[common])
    x.add_argument("name")
    x.add_argument("--imgsz", type=int, default=640)
    x.add_argument("--force", action="store_true")
    d.set_defaults(func=cmd_dataset)

    r = sub.add_parser("run", help="create and control training runs")
    rs = r.add_subparsers(dest="action", required=True)
    x = rs.add_parser("new", parents=[common])
    x.add_argument("--from-yaml", help="a run.yaml to copy the settings from")
    x.add_argument("--dataset")
    x.add_argument("--task", choices=["detect", "segment", "pose", "obb", "classify"])
    x.add_argument("--model", default="yolo26n")
    x.add_argument("--name")
    x.add_argument("--epochs", type=int)
    x.add_argument("--imgsz", type=int)
    x.add_argument("--batch", type=int)
    x.add_argument("--patience", type=int)
    x.add_argument("--aug", default="standard", choices=["none", "light", "standard", "strong"])
    x.add_argument("--drop", nargs="*", help="class names to drop")
    x.add_argument("--oversample", action="store_true", help="repeat images of rare classes")
    x.add_argument(
        "--resplit", action="store_true", help="new stratified, group-aware train/val split"
    )
    x.add_argument("--subset", type=float, default=1.0)
    x.add_argument("--set", nargs="*", help="any Ultralytics argument, e.g. lr0=0.001 mosaic=0.5")
    x.add_argument("--start", action="store_true")
    rs.add_parser("list", parents=[common])
    for name in (
        "status",
        "start",
        "resume",
        "pause",
        "stop",
        "copy",
        "delete",
        "eval",
        "export",
        "extend",
    ):
        x = rs.add_parser(name, parents=[common])
        x.add_argument("run")
        if name == "status":
            x.add_argument("--follow", action="store_true")
        if name == "stop":
            x.add_argument("--now", action="store_true", help="stop mid-epoch")
        if name == "copy":
            x.add_argument("--name")
        if name == "extend":
            x.add_argument("--epochs", type=int, required=True, help="new total number of epochs")
        if name == "eval":
            x.add_argument("--data", help="another dataset yaml (external test set)")
        if name == "export":
            x.add_argument("--format", default="onnx")
            x.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "int8"])
    r.set_defaults(func=cmd_run)

    x = sub.add_parser("predict", parents=[common], help="run a model on images or videos")
    x.add_argument("--run")
    x.add_argument("--weights")
    x.add_argument("--source", nargs="+", required=True)
    x.add_argument("--conf", type=float, default=0.25)
    x.add_argument("--iou", type=float, default=0.7)
    x.add_argument("--out", default="predictions")
    x.set_defaults(func=cmd_predict)

    s = sub.add_parser("sweep", help="hyperparameter sweeps")
    ss = s.add_subparsers(dest="action", required=True)
    x = ss.add_parser("create", parents=[common])
    x.add_argument("--base", required=True, help="run whose settings are the starting point")
    x.add_argument("--space", required=True, help="yaml with the values to try (see sweeps.py)")
    x.add_argument("--mode", default="grid", choices=["grid", "random"])
    x.add_argument("--max-runs", type=int, default=12)
    x.add_argument("--budget-hours", type=float)
    x.add_argument("--name")
    for name in ("start", "results"):
        x = ss.add_parser(name, parents=[common])
        x.add_argument("id")
    s.set_defaults(func=cmd_sweep)

    rm = sub.add_parser("remote", help="SSH machines")
    rms = rm.add_subparsers(dest="action", required=True)
    x = rms.add_parser("check", parents=[common])
    x.add_argument("name")
    x = rms.add_parser(
        "bench", parents=[common], help="time a model on a remote device (Raspberry Pi)"
    )
    x.add_argument("name")
    x.add_argument("--run", required=True)
    x.add_argument("--model", help="exported model file/folder (default the run's best.pt)")
    rm.set_defaults(func=cmd_remote)

    x = sub.add_parser("notify-test", parents=[common], help="send a test notification")
    x.set_defaults(func=cmd_notify)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s"
    )
    trust_system_certificates()
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
