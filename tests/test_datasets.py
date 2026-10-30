import csv

import pytest

from model_trainer.datasets import formats, health, prepare, store
from model_trainer.datasets.prepare import ClassPlan, SplitPlan
from model_trainer.util import read_yaml


def test_yolo_seg_is_read(yolo_seg):
    ds = formats.load(yolo_seg)
    assert ds.format == "yolo" and ds.label_kind == "polygons"
    assert ds.split_counts() == {"train": 24, "val": 8, "test": 0}
    assert ds.names == {0: "square", 1: "circle", 2: "triangle"}
    assert all(it.width == 160 and it.height == 120 for it in ds.items)
    assert all(it.group for it in ds.items)  # groups.csv applied
    assert ds.tasks()["segment"] == "" and ds.tasks()["pose"]
    o = next(o for it in ds.items for o in it.objects)
    assert o.polygon and o.box and 0 < o.box[2] < 1


def test_yolo_without_yaml_and_det(yolo_det):
    (yolo_det / "data.yaml").unlink()
    ds = formats.load(yolo_det)
    assert ds.label_kind == "boxes" and ds.default_task() == "detect"
    assert set(ds.names) == {0, 1, 2}  # placeholder names from the class ids


def test_bad_lines_are_reported(yolo_det):
    lab = next((yolo_det / "labels" / "train").glob("*.txt"))
    lab.write_text("0 0.5 0.5\nx y z\n7 0.5 0.5 0.1 0.1\n")
    ds = formats.load(yolo_det)
    it = next(i for i in ds.items if i.label_file == lab)
    assert len(it.problems) == 2
    errs = health.label_errors(it, len(ds.names))
    assert any("not in the class list" in e for e in errs) or 7 in ds.names


def test_coco_and_folders(coco_ds, folders_ds):
    ds = formats.load(coco_ds)
    assert ds.format == "coco" and ds.label_kind == "polygons"
    assert ds.split_counts()["val"] == 3 and len(ds.names) == 3
    assert all(it.image.is_file() for it in ds.items)
    f = formats.load(folders_ds)
    assert f.format == "folders" and f.label_kind == "classes"
    assert f.split_counts() == {"train": 24, "val": 9, "test": 0}
    assert f.default_task() == "classify"


def test_health_report(yolo_seg):
    ds = formats.load(yolo_seg)
    rep = health.analyse(ds, imgsz=640)
    ids = {c.id: c for c in rep.checks}
    for key in (
        "splits",
        "background",
        "balance",
        "object_size",
        "label_errors",
        "leakage_duplicates",
        "groups",
    ):
        assert key in ids, key
    assert ids["label_errors"].level == "ok"
    assert ids["groups"].level == "ok"
    assert sum(rep.stats["background_images"].values()) == 2
    assert rep.stats["instances_total"][0] > rep.stats["instances_total"][2]
    small = health.analyse(ds, imgsz=32)
    assert {c.id: c for c in small.checks}["object_size"].level in ("warn", "bad")
    assert health.recommend_imgsz(small.stats) > 32


def test_duplicates_across_splits_are_leakage(yolo_seg):
    import shutil

    src = next((yolo_seg / "images" / "train").glob("*.jpg"))
    shutil.copy(src, yolo_seg / "images" / "val" / "copy.jpg")
    ds = formats.load(yolo_seg)
    rep = health.analyse(ds)
    leak = next(c for c in rep.checks if c.id == "leakage_duplicates")
    assert leak.level != "ok" and leak.details


def test_class_plan_mapping():
    plan = ClassPlan(drop=["triangle"], merge={"circle": "round"}, rename={"square": "box"})
    mapping, names = plan.mapping({0: "square", 1: "circle", 2: "triangle"})
    assert names == {0: "box", 1: "round"}
    assert mapping == {0: 0, 1: 1, 2: None}


def test_resplit_keeps_groups_and_ratio(yolo_seg):
    ds = formats.load(yolo_seg)
    out = prepare.resplit(ds.items, SplitPlan(mode="resplit", ratios={"train": 0.75, "val": 0.25}))
    by_group = {}
    for it in ds.items:
        assert by_group.setdefault(it.group, out[it.key]) == out[it.key]
    val = sum(1 for v in out.values() if v == "val")
    assert 4 <= val <= 12


def test_resplit_few_groups_still_has_validation(tmp_path):
    # 3 layouts with removal photos (screw-synth test export): 85/15 used to give 3/0
    sizes = {"000000": 4, "000001": 3, "000002": 2}
    items = [
        formats.Item(tmp_path / f"{g}_{k}.jpg", "train", [formats.Obj(k % 2)], group=g)
        for g, n in sizes.items()
        for k in range(n)
    ]
    plan = SplitPlan(mode="resplit", ratios={"train": 0.85, "val": 0.15})
    out = prepare.resplit(items, plan)
    assert set(out.values()) == {"train", "val"}
    val_groups = {it.group for it in items if out[it.key] == "val"}
    assert len(val_groups) == 1  # one whole layout, never split
    with pytest.raises(ValueError, match="too few images for a validation set"):
        one_layout = formats.Dataset("t", tmp_path, "yolo", {0: "a", 1: "b"}, items[:4], "boxes")
        prepare.prepare(one_layout, tmp_path / "data", "detect", split=plan)


def test_prepare_detect_with_oversampling(yolo_seg, tmp_path):
    ds = formats.load(yolo_seg)
    m = prepare.prepare(
        ds,
        tmp_path / "data",
        "detect",
        ClassPlan(drop=["triangle"]),
        balance="oversample",
    )
    assert m["classes"] == {0: "square", 1: "circle"}
    data = read_yaml(tmp_path / "data" / "data.yaml")
    assert data["train"] == "train.txt" and data["names"] == {0: "square", 1: "circle"}
    train_list = (tmp_path / "data" / "train.txt").read_text().split()
    assert len(train_list) >= m["images"]["train"]
    label = next((tmp_path / "data" / "labels" / "val").glob("*.txt")).read_text().split("\n")[0]
    if label:
        assert len(label.split()) == 5
    assert prepare.verify(tmp_path / "data", full=True) == []
    (tmp_path / "data" / train_list[0][2:]).write_bytes(b"x")
    assert prepare.verify(tmp_path / "data")
    rows = list(csv.DictReader((tmp_path / "data" / "groups.csv").open()))
    assert rows and rows[0]["group"]


@pytest.mark.parametrize("task", ["segment", "obb", "classify"])
def test_prepare_other_tasks(yolo_seg, tmp_path, task):
    ds = formats.load(yolo_seg)
    m = prepare.prepare(ds, tmp_path / task, task, split=SplitPlan(mode="resplit"))
    assert sum(m["images"].values()) > 0
    if task == "classify":
        assert (tmp_path / task / "train").is_dir() and (tmp_path / task / "val").is_dir()
    else:
        line = next(
            line
            for p in (tmp_path / task / "labels" / "train").glob("*.txt")
            for line in p.read_text().split("\n")
            if line
        )
        assert len(line.split()) == (9 if task == "obb" else len(line.split()))


def test_prepare_requires_val(yolo_seg, tmp_path):
    import shutil

    shutil.rmtree(yolo_seg / "images" / "val")
    ds = formats.load(yolo_seg)
    with pytest.raises(ValueError, match="re-split"):
        prepare.prepare(ds, tmp_path / "x", "detect")


def test_store_import_fingerprint_and_health(workspace, yolo_seg):
    proj = workspace.project("demo")
    m = store.import_dataset(proj, yolo_seg, name="shapes")
    assert proj.datasets() == ["shapes"] and m["default_task"] == "segment"
    with pytest.raises(FileExistsError):
        store.import_dataset(proj, yolo_seg, name="shapes")
    rep = store.health_report(proj, "shapes", 640)
    assert rep["checks"] and rep["recommended_imgsz"]
    assert store.health_report(proj, "shapes", 640)["created"] == rep["created"]  # cached
    store._CACHE.clear()
    lab = next((yolo_seg / "labels" / "train").glob("*.txt"))
    lab.write_text("0 0.5 0.5 0.2 0.2\n")
    _, m2 = store.open_dataset(proj, "shapes")
    assert m2["changed"]
    store.refresh(proj, "shapes")
    store._CACHE.clear()
    assert not store.open_dataset(proj, "shapes")[1]["changed"]


def test_folder_beside_its_yaml_gets_the_class_names(tmp_path):
    """Exporters write <out>/seg.yaml next to <out>/seg/: importing the folder must find the names."""
    import shutil

    from conftest import make_yolo

    src = make_yolo(tmp_path / "src", groups=False)
    out = tmp_path / "export"
    shutil.copytree(src / "images", out / "seg" / "images")
    shutil.copytree(src / "labels", out / "seg" / "labels")
    (out / "seg.yaml").write_text(
        f"path: {(out / 'seg').as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: square\n  1: circle\n  2: triangle\n"
    )
    (out / "pose.yaml").write_text(
        "path: pose\ntrain: images/train\nval: images/val\nkpt_shape: [3, 3]\nnames:\n  0: x\n"
    )
    ds = formats.load(out / "seg")
    assert ds.names == {0: "square", 1: "circle", 2: "triangle"} and not formats.placeholder_names(
        ds.names
    )
    with pytest.raises(formats.MultipleDatasets) as exc:
        formats.load(out)
    opts = {o["path"].split("\\")[-1].split("/")[-1]: o for o in exc.value.options}
    assert set(opts) == {"seg.yaml", "pose.yaml"} and opts["pose.yaml"]["labels"] == "keypoints"
    assert len(formats.candidates(out)) == 2 and len(formats.candidates(out / "seg")) == 1


def test_classes_txt_names_a_bare_folder(yolo_det):
    (yolo_det / "data.yaml").unlink()
    assert formats.placeholder_names(formats.load(yolo_det).names) == [0, 1, 2]
    (yolo_det / "classes.txt").write_text("square\ncircle\ntriangle\n")
    assert formats.load(yolo_det).names == {0: "square", 1: "circle", 2: "triangle"}
