# model-trainer

A friendly, generic app for training YOLO models: detection, segmentation, pose (keypoints), rotated boxes (OBB)
and classification.

You:
1. load a dataset and check its health;
2. pick a base model from a dropdown;
3. tweak as much or as little as you like;
4. train it, with live charts and pause/resume;
5. evaluate the result, try it on your own images, and export it.

Beginners get recommended values and explanations everywhere (**Simple** mode). Experienced users get every
parameter, raw YAML, sweeps and a CLI (**Advanced** mode).

## Install (Windows, Linux)

Requirements: Python 3.10-3.12, and an NVIDIA GPU with a recent driver (training on the CPU works, but slowly).

```bash
git clone https://github.com/ama24/model-trainer.git
cd model-trainer
python -m venv .venv
.venv\Scripts\activate                 # Linux: source .venv/bin/activate
# PyTorch with CUDA first (pick the right CUDA version on pytorch.org; this is CUDA 12.4):
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[export,dev]"
mtrain gui                              # opens http://localhost:8090
```

To reproduce the exact tested environment, install the pinned versions instead:
`pip install -r requirements.lock.txt`.

Everything (datasets, runs, settings) lives in the **workspace** folder:
- default: `~/model-trainer-workspace`;
- or set `MTRAIN_HOME`, or pass `--workspace`.

## What it does

| Screen | What you can do |
|---|---|
| **Datasets** | Import YOLO, COCO or class-folder datasets (folder, yaml/json or zip). **Health report**: split sizes, class balance and imbalance ratio, rare classes, object size at your training resolution, crowding, object-position heatmap, label errors, duplicate images and leakage between train/val, groups. Each check says what it means, why it matters and what to do. **Gallery** with boxes, masks and keypoints, filtered by class, split or problem. **What the model sees**: real augmented training samples. |
| **New run** (wizard) | 1. Dataset, task (only the ones the labels allow), keep/drop/merge/rename classes, oversampling, new stratified group-aware split, quick-test subset. 2. Base model: YOLO26 / YOLO11 / YOLOv8 / RT-DETR × nano…x-large, with estimated GPU memory, "fits?" and time per epoch, plus a recommendation. Start from pretrained, scratch, an earlier run or a file. 3. Image size (recommended from the object sizes), caching, rect. 4. Augmentation presets plus every slider. 5. Epochs, patience, batch, optimizer, learning rate schedule, loss weights, raw YAML. 6. Review with time, memory and disk estimates and the problems found, then Start, Queue, or Save as template. |
| **Run** | Live losses, scores and learning rate per epoch, per-class table, GPU memory/temperature, ETA. **Pause** (end of epoch, frees the GPU), **Resume**, **Stop**, **Train longer**. **Watch it learn**: the same validation images after every epoch. Warnings for overfitting, plateaus, NaN, a hot GPU and low disk. **Results**: plain-language summary, per-class scores, clickable confusion matrix, PR/F1 curves with the best confidence threshold, error gallery (missed, false, wrong class, poor box), recall by object size. **Export**: ONNX, TensorRT, OpenVINO, NCNN (Raspberry Pi), TFLite, CoreML in FP32/FP16/INT8, each re-scored and timed. |
| **Compare** | Overlay any metric across runs, see the settings that differed, find the best run. |
| **Playground** | Drop images, a folder or a video. One model or two side by side, confidence/IoU sliders. Score on an external labelled set. Save predictions as a new dataset to correct (active learning). |
| **Sweeps** | Grid or random search over any settings, queued, with a time budget, ranked. |
| **Settings** | Phone notifications (ntfy), SSH machines for remote training and Raspberry Pi benchmarks. |

## Safety: runs survive crashes and power cuts

- **Checkpoints:** after every epoch a full checkpoint (with the optimizer state) is saved and verified.
  - Resume continues exactly where training stopped, also after a crash or power loss.
  - Snapshots every N epochs let you go back.
- **Frozen data:** every run trains on its own frozen copy of the dataset (hard links, so no extra space), with
  a SHA-1 of every file. Resume refuses if those files were changed.
- **Detached training:** training runs in its own process. Closing the browser or the app does not stop it.
- **No sleep:** the computer is kept awake while training.
- **Out-of-memory recovery:** when the GPU runs out of memory, training restarts in a fresh process with half
  the batch size and keeps the finished epochs. When the PC's RAM runs out, it first turns off the image cache,
  then halves the data loader workers, then the batch. A fresh process is needed because a failed attempt does
  not give its memory back.
- **Automatic image cache:** with "Cache images: auto" (the default), the training-size images are kept in RAM
  when they fit in the free memory at the start of training (twice their size plus 4 GB), for faster epochs.
  More RAM in the PC is used automatically.

## Command line

```bash
mtrain dataset import path/to/dataset --name parts
mtrain dataset check parts --imgsz 1024
mtrain run new --dataset parts --model yolo26s --epochs 100 --imgsz 1024 --start
mtrain run list
mtrain run status <run> --follow
mtrain run pause <run>      # / resume / stop / extend --epochs 200
mtrain run export <run> --format onnx --precision fp16
mtrain predict --run <run> --source photos/
mtrain sweep create --base <run> --space space.yaml --mode grid --budget-hours 8
mtrain notify-test
```

## Using the models elsewhere

A finished run has a **model package** in `<run>/package/`:
- `best.pt`;
- the exports;
- `model_card.json`: task, classes in order, image size, scores, recommended confidence, the dataset fingerprint and
  class mapping, the training settings and versions.

Check the class list in the card before using a model in another project.

## Development

```bash
pytest -m "not slow"     # fast tests (datasets, GUI with a simulated user, run manager, CLI)
pytest -m slow           # real tiny training runs: pause/resume/extend, crash recovery, export (GPU recommended)
ruff check . && ruff format --check .
```

## Docs

- [docs/USER_GUIDE.md](docs/USER_GUIDE.md): a walkthrough for your first model.
- [docs/DESIGN.md](docs/DESIGN.md): the design.

Licence: AGPL-3.0 (it builds on [Ultralytics](https://github.com/ultralytics/ultralytics), which is AGPL-3.0).
