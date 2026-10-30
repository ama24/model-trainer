"""Real (tiny) training runs through the worker process: pause, resume, extend, crash recovery, export.

Slow: downloads the nano base models once and trains for a few epochs (GPU if there is one).
"""

from __future__ import annotations

import os
import signal
import time

import pytest

from model_trainer.datasets import store
from model_trainer.runs.config import ModelChoice, RunConfig
from model_trainer.runs.manager import Manager
from model_trainer.runs.store import Run

pytestmark = pytest.mark.slow
FAST = {"imgsz": 128, "batch": 8, "workers": 0, "patience": 100}


def _wait(run: Run, states: tuple[str, ...], timeout: float = 900) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = run.state()
        if s.get("state") in states:
            return s
        if s.get("state") == "failed" and "failed" not in states:
            pytest.fail(f"run failed: {s.get('error')}\n{run.tail_log(60)}")
        time.sleep(1)
    pytest.fail(f"timeout waiting for {states}; state {run.state()}\n{run.tail_log(60)}")


def _wait_epochs(run: Run, n: int) -> None:
    t0 = time.time()
    while len(run.metrics()) < n:
        if run.state().get("state") in ("failed", "interrupted") or time.time() - t0 > 900:
            pytest.fail(f"no progress: {run.state()}\n{run.tail_log(60)}")
        time.sleep(1)


@pytest.fixture
def ws_with_data(workspace, yolo_seg, folders_ds):
    proj = workspace.project("default")
    store.import_dataset(proj, yolo_seg, name="shapes")
    store.import_dataset(proj, folders_ds, name="cls")
    return workspace


def test_segment_pause_resume_extend_export(ws_with_data):
    ws = ws_with_data
    mgr = Manager(ws)
    cfg = RunConfig(
        name="seg",
        project="default",
        dataset="shapes",
        task="segment",
        model=ModelChoice("yolo26", "n"),
        train={**FAST, "epochs": 5},
        balance="oversample",
        split={"mode": "resplit"},
    )
    cfg.options.snapshot_every = 2
    cfg.options.watch_images = 2
    run = mgr.create(ws.project("default"), cfg)
    assert mgr.start(run) == "started"
    _wait_epochs(run, 2)
    mgr.pause(run)
    _wait(run, ("paused",))
    paused_at = run.metrics()[-1]["epoch"]
    assert run.resume_ckpt.is_file()
    mgr.resume(run)
    _wait(run, ("done",))
    epochs = [m["epoch"] for m in run.metrics()]
    assert epochs == list(range(1, 6)), epochs  # no epoch repeated or skipped across the pause
    assert paused_at < 5
    assert (run.package_dir / "model_card.json").is_file() and run.results()[
        "main_metric"
    ] == "mask mAP50-95"
    assert sorted(p.name for p in run.snapshots_dir.glob("*.pt")) == [
        "epoch_0002.pt",
        "epoch_0004.pt",
    ]
    assert any(run.watch_dir.glob("epoch_*/*.jpg"))
    assert run.metrics()[-1].get("per_class")
    mgr.extend(run, 7)
    _wait(run, ("done",))
    assert [m["epoch"] for m in run.metrics()][-2:] == [6, 7]
    from model_trainer.export import export_model

    rec = export_model(run, "onnx", "fp32", bench_images=3)
    assert (
        rec["path"].endswith(".onnx")
        and rec.get("main_value") is not None
        and rec.get("ms_per_image")
    )


def test_classify_from_folders(ws_with_data):
    mgr = Manager(ws_with_data)
    cfg = RunConfig(
        name="cls",
        project="default",
        dataset="cls",
        task="classify",
        model=ModelChoice("yolo26", "n"),
        train={"imgsz": 64, "batch": 8, "workers": 0, "epochs": 2},
    )
    run = mgr.create(ws_with_data.project("default"), cfg)
    mgr.start(run)
    _wait(run, ("done",))
    assert run.results()["main_metric"] == "top-1 accuracy"


def test_crash_is_interrupted_and_resumable(ws_with_data):
    mgr = Manager(ws_with_data)
    cfg = RunConfig(
        name="crash",
        project="default",
        dataset="shapes",
        task="detect",
        model=ModelChoice("yolo26", "n"),
        train={**FAST, "epochs": 4},
    )
    run = mgr.create(ws_with_data.project("default"), cfg)
    mgr.start(run)
    _wait_epochs(run, 1)
    pid = run.state()["pid"]
    os.kill(
        pid, signal.SIGTERM if os.name != "nt" else signal.SIGTERM
    )  # simulate a crash / power cut
    t0 = time.time()
    while run.state()["state"] != "interrupted":  # the heartbeat goes stale / the pid is gone
        assert time.time() - t0 < 120, run.state()
        time.sleep(1)
    done_before = run.metrics()[-1]["epoch"]
    mgr.resume(run)
    _wait(run, ("done",))
    epochs = [m["epoch"] for m in run.metrics()]
    assert epochs[-1] == 4 and epochs.count(done_before) <= 2
