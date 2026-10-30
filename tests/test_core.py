import subprocess
import zipfile

import pytest

from model_trainer import compare, models, notify, params, sweeps
from model_trainer.cli import build_parser
from model_trainer.evaluate import match_image, plain_summary
from model_trainer.remote import Remote, launch_remote, send_command
from model_trainer.runs import worker
from model_trainer.runs.config import ModelChoice, RunConfig
from model_trainer.runs.manager import Manager
from model_trainer.runs.store import Run


def _cfg(**kw):
    base = {"name": "r", "project": "default", "dataset": "shapes", "task": "detect"}
    base.update(kw)
    return RunConfig(**base)


def test_params_defaults_presets_and_validation():
    d = params.defaults("detect")
    assert d["epochs"] == 100 and "mosaic" in d and "dropout" not in d
    assert "dropout" in params.defaults("classify") and "mosaic" not in params.defaults("classify")
    clean, problems = params.validate(
        {"epochs": "50", "imgsz": 99999, "optimizer": "Nope", "lr0": 0.002, "custom_arg": 1},
        "detect",
    )
    assert clean["epochs"] == 50 and clean["imgsz"] == 2048 and clean["custom_arg"] == 1
    assert "optimizer" not in clean and len(problems) == 2
    assert params.changed({"epochs": 100, "imgsz": 1024}, "detect") == {"imgsz": 1024}
    for p in params.PARAMS:
        assert p.help() and p.label


def test_run_config_merges_preset_and_explicit_values():
    cfg = _cfg(aug_preset="none", train={"mosaic": 0.5, "epochs": 3})
    a = cfg.train_args()
    assert a["mosaic"] == 0.5 and a["fliplr"] == 0 and a["epochs"] == 3
    assert cfg.problems() == []
    assert _cfg(task="pose", model=ModelChoice("rtdetr", "l")).problems()
    assert _cfg(model=ModelChoice(start="run")).problems()
    again = RunConfig.from_dict(cfg.to_dict())
    assert again == cfg


def test_models_registry_and_estimates():
    assert models.weights_name("yolo26", "s", "segment") == "yolo26s-seg.pt"
    assert models.weights_name("rtdetr", "l", "detect") == "rtdetr-l.pt"
    with pytest.raises(ValueError):
        models.weights_name("rtdetr", "l", "segment")
    assert {o["size"] for o in models.options("pose") if o["family"] == "yolo11"} == set("nsmlx")
    small = models.estimate_vram_gb("yolo26", "n", "detect", 640, 16)
    big = models.estimate_vram_gb("yolo26", "x", "detect", 1280, 16)
    assert 1 < small < 4 < big
    assert models.max_batch("yolo26", "m", "segment", 1024, 8) < models.max_batch(
        "yolo26", "n", "detect", 640, 8
    )
    assert models.recommend(100, "detect", 8, 640)[1] == "n"


def test_match_image_sorts_errors():
    gt = [
        (0, (0.2, 0.2, 0.1, 0.1)),
        (1, (0.6, 0.6, 0.1, 0.1)),
        (0, (0.85, 0.85, 0.1, 0.1)),
        (1, (0.4, 0.8, 0.1, 0.1)),
    ]
    pred = [
        (0, (0.2, 0.2, 0.1, 0.1), 0.9),  # hit
        (0, (0.6, 0.6, 0.1, 0.1), 0.8),  # wrong class (true 1)
        (0, (0.88, 0.88, 0.1, 0.1), 0.7),  # poor localisation
        (1, (0.1, 0.9, 0.05, 0.05), 0.6),  # false detection
    ]
    m = match_image(gt, pred)
    assert (
        len(m["hit"]) == 1 and len(m["wrong_class"]) == 1 and m["wrong_class"][0]["true_cls"] == 1
    )
    assert len(m["localisation"]) == 1 and len(m["false"]) == 1 and len(m["missed"]) == 1


def test_plain_summary():
    res = {
        "mean": {"metrics/precision(B)": 0.9, "metrics/recall(B)": 0.8},
        "main_metric": "mAP50-95",
        "main_value": 0.6,
        "recommended_conf": 0.31,
    }
    s = plain_summary(res, "detect")
    assert "about 80" in s and "about 90" in s and "0.31" in s


def test_manager_queue_and_states(workspace, monkeypatch):
    launched = []
    mgr = Manager(workspace)
    monkeypatch.setattr(
        mgr,
        "_launch",
        lambda run, mode: (
            launched.append((run.id, mode)),
            run.set_state("running"),
            run.heartbeat(__import__("os").getpid()),
        ),
    )
    proj = workspace.project("default")
    a = mgr.create(proj, _cfg(name="a"))
    b = mgr.create(proj, _cfg(name="b"))
    assert mgr.start(a) == "started"
    assert mgr.start(b) == "queued" and b.state()["state"] == "queued"
    assert [e["run"] for e in mgr.queue()] == [str(b.path)]
    assert mgr.start_next_queued() is None  # a is still active
    a.set_state("done")
    nxt = mgr.start_next_queued()
    assert nxt is not None and nxt.id == b.id and launched[-1] == (b.id, "start")
    mgr.pause(b)
    assert b.pending_command() == "pause"
    # a crashed worker (stale heartbeat) shows up as interrupted
    (b.path / "heartbeat.json").write_text('{"pid": 999999, "time": 0}')
    assert b.state()["state"] == "interrupted" and mgr.interrupted()[0].id == b.id
    c = mgr.copy(a, "a-again")
    assert c.config.name == "a-again" and c.state()["state"] == "created"
    mgr.delete(c)
    assert not c.path.exists()


def test_store_summary_and_log(workspace):
    run = Run.create(workspace.project("default"), _cfg())
    (run.path / "log.txt").write_text("a\r\nprogress 1\rprogress 2\n\x1b[1mbold\x1b[0m\n")
    assert run.tail_log().splitlines()[-2:] == ["progress 2", "bold"]
    s = run.summary()
    assert s["state"] == "created" and s["model"] == "yolo26n"


def test_checkpoint_check(tmp_path):
    good = tmp_path / "a.pt"
    with zipfile.ZipFile(good, "w") as z:
        z.writestr("data.pkl", b"x" * 100)
    assert worker._checkpoint_ok(good)
    bad = tmp_path / "b.pt"
    bad.write_bytes(good.read_bytes()[:50])
    assert not worker._checkpoint_ok(bad)
    assert "smaller model" in worker._friendly(
        RuntimeError("CUDA out of memory. Tried to allocate")
    )


def test_sweep_expand_and_apply():
    grid = sweeps.expand({"lr0": [0.001, 0.01], "model.size": ["n", "s"]}, "grid", 0)
    assert len(grid) == 4
    rnd = sweeps.expand(
        {"lr0": {"min": 1e-4, "max": 1e-2, "log": True}, "mosaic": [0.5, 1.0]}, "random", 5, seed=1
    )
    assert len(rnd) == 5 and all(1e-4 <= p["lr0"] <= 1e-2 for p in rnd)
    assert rnd == sweeps.expand(
        {"lr0": {"min": 1e-4, "max": 1e-2, "log": True}, "mosaic": [0.5, 1.0]}, "random", 5, seed=1
    )
    c = sweeps.apply(_cfg(), {"lr0": 0.003, "model.size": "s", "aug_preset": "light"})
    assert c.train["lr0"] == 0.003 and c.model.size == "s" and c.aug_preset == "light"


def test_create_sweep(workspace):
    s = sweeps.create_sweep(
        workspace,
        workspace.project("default"),
        _cfg(),
        {"imgsz": [320, 640]},
        max_runs=10,
        budget_hours=1,
    )
    assert len(s["runs"]) == 2
    rows = sweeps.results(workspace, s["id"])
    assert {r["imgsz"] for r in rows} == {320, 640}
    assert not sweeps.over_budget(workspace, s["id"])


def test_compare_settings_diff(workspace):
    proj = workspace.project("default")
    a = Run.create(proj, _cfg(name="a"))
    b = Run.create(proj, _cfg(name="b", train={"imgsz": 960}, model=ModelChoice(size="s")))
    diff = {r["setting"]: r for r in compare.settings_diff([a, b])}
    assert set(diff) == {"imgsz", "model"}
    assert diff["imgsz"][b.id] == 960


def test_remote_commands_are_built_without_running(workspace):
    calls = []

    def fake(cmd):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, "True", "")

    s = workspace.settings()
    s["remotes"] = [
        {
            "name": "box",
            "host": "10.0.0.5",
            "user": "me",
            "port": 2222,
            "workdir": "/w",
            "python": "py",
        }
    ]
    workspace.save_settings(s)
    r = Remote(s["remotes"][0], fake)
    assert r.check() == (True, "connected; CUDA available")
    assert calls[-1][:4] == ["ssh", "-p", "2222", "-o"] and calls[-1][5] == "me@10.0.0.5"
    run = Run.create(workspace.project("default"), _cfg(options={"executor": "ssh:box"}))
    run.data_dir.mkdir()
    (run.data_dir / "manifest.json").write_text('{"files": {}}')
    launch_remote(workspace, run, "start", fake)
    assert any(c[0] == "scp" and "-r" in c for c in calls)
    assert "nohup py -m model_trainer.runs.worker" in calls[-1][-1]
    assert run.state().get("remote") == "box"
    send_command(workspace, run, "pause", fake)
    assert calls[-1][-1].endswith("/command.json")


def test_notify_needs_a_topic():
    assert (
        notify.send(
            {"notify": {"enabled": True, "topic": "", "events": ["finished"]}}, "finished", "t", "m"
        )
        is False
    )
    assert (
        notify.send(
            {"notify": {"enabled": False, "topic": "x", "events": ["finished"]}},
            "finished",
            "t",
            "m",
        )
        is False
    )


def test_cli_parses():
    p = build_parser()
    a = p.parse_args(
        [
            "run",
            "new",
            "--dataset",
            "shapes",
            "--model",
            "yolo11s",
            "--set",
            "lr0=0.001",
            "--drop",
            "circle",
        ]
    )
    assert a.model == "yolo11s" and a.set == ["lr0=0.001"] and a.drop == ["circle"]
    assert p.parse_args(["run", "extend", "x", "--epochs", "200"]).epochs == 200
    from model_trainer.cli import _parse_model, _value

    assert _parse_model("yolo26m") == ("yolo26", "m") and _parse_model("rtdetr-l") == (
        "rtdetr",
        "l",
    )
    assert (
        _value("3") == 3
        and _value("0.5") == 0.5
        and _value("true") is True
        and _value("adam") == "adam"
    )


def test_cli_dataset_commands(workspace, yolo_det, capsys):
    from model_trainer.cli import main

    assert (
        main(
            ["dataset", "import", str(yolo_det), "--name", "d", "--workspace", str(workspace.root)]
        )
        == 0
    )
    assert main(["dataset", "check", "d", "--workspace", str(workspace.root)]) == 0
    out = capsys.readouterr().out
    assert "imported d" in out and "Class balance" in out and "recommended image size" in out
    assert (
        main(["run", "new", "--dataset", "d", "--epochs", "1", "--workspace", str(workspace.root)])
        == 0
    )
    assert main(["run", "list", "--workspace", str(workspace.root)]) == 0
    assert "created" in capsys.readouterr().out
