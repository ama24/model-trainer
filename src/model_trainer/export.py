"""Export a trained model for deployment, re-score the exported file and measure its speed.

Exporting can silently lose accuracy (especially FP16 and INT8), so every export is validated again
on the run's validation set and compared with the original PyTorch weights.
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Any

from .evaluate import MAIN_METRIC, _data_arg, validate
from .runs.store import Run
from .util import now, read_json, write_json

log = logging.getLogger("model_trainer.export")

FORMATS: dict[str, dict[str, Any]] = {
    "onnx": {
        "label": "ONNX",
        "where": "anywhere (Windows, Linux, Raspberry Pi with onnxruntime)",
        "precisions": ("fp32", "fp16"),
        "note": "The universal choice. FP16 needs a GPU to export.",
    },
    "engine": {
        "label": "TensorRT",
        "where": "NVIDIA GPUs (fastest there)",
        "precisions": ("fp32", "fp16", "int8"),
        "note": "Needs the TensorRT package; the file only runs on the same GPU model / driver.",
    },
    "openvino": {
        "label": "OpenVINO",
        "where": "Intel CPUs and GPUs",
        "precisions": ("fp32", "fp16", "int8"),
    },
    "ncnn": {
        "label": "NCNN",
        "where": "Raspberry Pi and phones (ARM CPUs)",
        "precisions": ("fp32", "fp16"),
        "note": "Usually the fastest option on a Raspberry Pi.",
    },
    "tflite": {
        "label": "TensorFlow Lite",
        "where": "Android, Coral, microcontrollers",
        "precisions": ("fp32", "fp16", "int8"),
        "note": "Pulls in TensorFlow (large download).",
    },
    "coreml": {"label": "CoreML", "where": "Apple devices", "precisions": ("fp32", "fp16", "int8")},
    "torchscript": {
        "label": "TorchScript",
        "where": "C++ / PyTorch without Python",
        "precisions": ("fp32",),
    },
}
PRECISION_HELP = {
    "fp32": "Full precision: the reference accuracy.",
    "fp16": "Half precision: about half the size, faster on GPUs; usually no visible accuracy loss.",
    "int8": "8-bit integers: smallest and fastest on CPUs/NPUs, but can lose accuracy. Calibrated on training images.",
}


def export_model(
    run: Run,
    fmt: str,
    precision: str = "fp32",
    device: str | None = None,
    rescore: bool = True,
    bench_images: int = 20,
) -> dict[str, Any]:
    from ultralytics import YOLO

    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt}")
    if precision not in FORMATS[fmt]["precisions"]:
        raise ValueError(f"{FORMATS[fmt]['label']} does not support {precision}")
    cfg = run.config
    weights = run.package_dir / "best.pt"
    if not weights.is_file():
        weights = run.best_weights
    if weights is None:
        raise FileNotFoundError("the run has no trained weights yet")
    imgsz = int(cfg.train_args().get("imgsz", 640))
    work = run.package_dir / "exports" / f"{fmt}_{precision}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    tmp_weights = work / "best.pt"
    shutil.copyfile(weights, tmp_weights)  # Ultralytics writes the export next to the weights
    t0 = time.time()
    out = YOLO(str(tmp_weights)).export(
        format=fmt,
        imgsz=imgsz,
        half=precision == "fp16",
        int8=precision == "int8",
        data=_data_arg(run) if precision == "int8" else None,
        device=device or cfg.options.device or None,
    )
    tmp_weights.unlink(missing_ok=True)
    path = Path(out)
    rec: dict[str, Any] = {
        "format": fmt,
        "precision": precision,
        "path": str(path),
        "size_mb": round(_size(path) / 1e6, 2),
        "export_s": round(time.time() - t0, 1),
        "created": now(),
    }
    if rescore:
        try:
            m = validate(
                path, _data_arg(run), imgsz, work / "val", device=device or cfg.options.device
            )
            key = MAIN_METRIC[cfg.task][0]
            rec["main_value"] = float(m.results_dict.get(key, 0))
            ref = run.results().get("main_value")
            if ref:
                rec["reference"] = float(ref)
                rec["change_pct"] = round(100 * (rec["main_value"] - ref) / ref, 2)
        except Exception as exc:  # noqa: BLE001 - some formats cannot be validated on this machine
            rec["rescore_error"] = str(exc)[:300]
    if bench_images:
        try:
            rec["ms_per_image"] = benchmark(path, run, imgsz, bench_images, device)
        except Exception as exc:  # noqa: BLE001
            rec["bench_error"] = str(exc)[:300]
    exports = read_json(run.package_dir / "exports.json", [])
    exports = [e for e in exports if not (e["format"] == fmt and e["precision"] == precision)] + [
        rec
    ]
    write_json(run.package_dir / "exports.json", exports)
    card = read_json(run.package_dir / "model_card.json", {})
    card["exports"] = exports
    write_json(run.package_dir / "model_card.json", card)
    return rec


def benchmark(
    model_path: Path, run: Run, imgsz: int, n: int = 20, device: str | None = None
) -> float:
    """Median milliseconds per image (after a warm-up) on validation images."""
    from ultralytics import YOLO

    if run.config.task == "classify":
        images = sorted(p for p in (run.data_dir / "val").rglob("*") if p.is_file())[:n]
    else:
        images = sorted((run.data_dir / "images" / "val").glob("*"))[:n]
    if not images:
        raise FileNotFoundError("no validation images to time")
    model = YOLO(str(model_path), task=run.config.task)
    for img in images[:3]:
        model.predict(str(img), imgsz=imgsz, verbose=False, device=device or None)
    times = []
    for img in images:
        t = time.perf_counter()
        model.predict(str(img), imgsz=imgsz, verbose=False, device=device or None)
        times.append((time.perf_counter() - t) * 1000)
    times.sort()
    return round(times[len(times) // 2], 2)


def _size(p: Path) -> int:
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return p.stat().st_size if p.exists() else 0
