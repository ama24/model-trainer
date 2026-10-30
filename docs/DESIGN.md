# model-trainer: design

Status: approved 2026-10-29; phases 1-3 implemented 2026-10-30 (see README for what exists; SSH training is experimental).

## Goal

A browser app to train computer-vision models (YOLO family) on any dataset. It should:
- guide inexperienced users with recommended values, plain-language explanations and health checks;
- give experienced users every parameter, reproducible configs, a CLI and a Python API.

It knows nothing about any particular project. Projects such as *supreme-screw-sortering* exchange **files** with
it:
- a dataset goes in (YOLO or COCO format, optionally with a group id per image);
- a **model package** comes out: weights, exported formats, and a model card.

## Principles

- **One app, two modes.**
  - **Simple** shows about 6 settings with recommended values.
  - **Advanced** shows everything.
  - Every parameter has a (?) with what it does, a typical value and the effect of raising or lowering it.
  - Values changed from the default are highlighted.
- **Training runs in its own process.** A browser or GUI crash never kills a run, and the GUI reconnects on restart.
- **Everything is a file.** Settings are readable YAML, and every GUI action has a CLI equivalent.
- **Runs cannot be silently corrupted.** Checkpoints are written safely, data changes are detected, and interrupted
  runs are offered for resume.

## Concepts

| Term | Meaning |
|---|---|
| Project | Folder grouping datasets, runs and models for one goal |
| Dataset | Imported, frozen copy with a fingerprint (hash of every image and label) |
| Run | Dataset + base model + all settings + results; can be paused, resumed, extended, copied, compared |
| Model package | Weights (`.pt`), exports (`.onnx`, …) and `model_card.json` (classes in order, image size, task, metrics, dataset fingerprint, settings, versions) |

## Screens

### 1. Projects (home)

- Recent runs and their state: running, paused, interrupted, done.
- GPU panel: name, free memory, temperature.
- Interrupted runs show a prompt, e.g. "interrupted at epoch 37. Resume?"

### 2. Dataset: import, health check, explore

**Import** from a folder or a zip. The format is detected automatically: YOLO (boxes, polygons, keypoints, rotated
boxes), COCO, classification folders, or Roboflow exports.

**The task follows from the labels:**

| Labels | Tasks offered |
|---|---|
| Polygons / masks | segmentation, detection, rotated boxes |
| Boxes | detection |
| Keypoints | pose |
| Folder per class | classification |
| Boxes / masks | classification of object crops |

**Health report.** Each item is green, amber or red, with *what it means / why it matters / what to do*:
- images per split, image sizes and aspect ratios, share of background images;
- objects and images per class, the imbalance ratio, and classes too rare to learn;
- **object size at the chosen training image size** (with a warning when objects fall below about 8 px);
- a heatmap of object positions, and objects per image;
- label errors: out of bounds, zero size, duplicates, degenerate polygons, unknown class ids;
- leakage: duplicate and near-duplicate images across splits (perceptual hash), and groups split across train and
  validation;
- class co-occurrence and look-alike classes.

**Gallery:**
- boxes, masks and keypoints drawn on each image;
- filters by class, split and issue;
- sorting by object count or smallest object;
- a full-size view with layer toggles;
- **"What the model sees":** the image after resizing and augmentation.

### 3. New run wizard

Every step can be saved as a template.

1. **Dataset & classes**
   - The task is picked from what the labels allow.
   - Classes can be dropped, merged or renamed. The mapping is stored in the model card.
   - Rare classes can be oversampled or weighted.
   - Splits: keep the dataset's own, or re-split by ratio (stratified, keeping groups together).
   - An optional quick-test subset.
2. **Base model**
   - Dropdowns for family (YOLO26, YOLO11, YOLOv8, RT-DETR) and size (n, s, m, l, x).
   - A table with parameters, published accuracy, and **estimated speed and VRAM on this GPU** at the chosen image
     size and batch.
   - Start from: pretrained weights, scratch, an earlier run (continue or fine-tune), or a file.
   - A recommendation based on the dataset.
3. **Data preparation**
   - Image size, recommended from the object-size check.
   - Rectangular batches.
   - Cache in RAM or on disk, with the memory it needs. Default `auto`: the worker picks `ram` at the start
     when 2 x the cache + 4 GB is free (`models.ram_cache_fits`), otherwise off; resumes keep that choice.
   - Number of data loader workers.
4. **Augmentation**
   - Presets: none, light, standard, strong, custom.
   - Every setting has a slider and a live 3×3 preview: flips, rotation, scale, shear, perspective, HSV, mosaic and
     close-mosaic, mixup, copy-paste, blur and noise.
   - Warnings where an augmentation can break meaning (e.g. hue shift when colour defines a class).
5. **Training**
   - Simple: epochs, patience, batch size (auto).
   - Advanced: optimizer, learning rate and final learning rate, schedule, warm-up, momentum, weight decay, frozen
     layers, mixed precision, label smoothing, dropout, loss weights, multi-scale, seed and deterministic mode,
     snapshot interval and how many to keep, validation interval.
   - A raw YAML tab.
6. **Review**
   - A summary with estimated time, disk space and VRAM, plus blocking problems and their fixes.
   - Buttons: Start now, Add to queue, or Save as template.

### 4. Live monitor

- Per-epoch charts:
  - losses (training and validation);
  - precision, recall, mAP50 and mAP50-95;
  - learning rate.
  - Each chart has a "how to read this" note.
- A live per-class score table.
- **"Watch it learn":** fixed validation images with the predictions after each epoch.
- GPU load, memory and temperature; time per epoch; estimated finish time.
- Automatic warnings: overfitting, a plateau, a NaN loss, GPU temperature, low disk space.
- Controls:
  - **Pause** finishes the epoch, checkpoints and frees the GPU.
  - **Resume** continues exactly where it stopped.
  - **Stop** keeps the best model.
  - **Stop & discard** throws the run away.
  - **Extend** adds epochs to a finished run.
- **Phone notifications** when a run finishes, fails or shows a warning.

### 5. Safety

- Checkpoints:
  - `last` (for resuming) and `best` (for using);
  - snapshots every N epochs, keeping the last K.
- Atomic writes: write to a temporary name, then rename. Checkpoints are verified on load.
- Crashes and power loss: the run is marked *interrupted* and offered for resume on the next start.
- Each run records:
  - the exact image list with hashes;
  - the class mapping and the settings;
  - package versions, GPU and driver.
  - **Resume refuses if the dataset changed**, and shows what changed.
- Out of memory: save lighter settings (GPU: half the batch; RAM: half the workers, then the batch), log
  it, and restart the worker as a new process that resumes from the last epoch (`--after=<pid>` waits for
  the old process to exit). Retrying inside the same process leaked GPU memory and data loader processes.
- Windows is kept awake during training.
- Disk space is checked before the start and during the run.

### 6. Results

- Metrics explained in plain language.
- A per-class table.
- A confusion matrix: click a cell to see those images.
- Precision-recall curves.
- A recommended confidence threshold, from the F1 curve.
- An error gallery: missed objects, false detections, wrong class, poor localisation.
- Results by object size.
- Run comparison: curves overlaid, and the settings shown as a diff.

### 7. Playground

- Drop in images, a folder or a video. Use one model, or two side by side.
- Confidence and IoU sliders.
- With an external labelled set, it gives an independent score that shows the domain gap.
- Predictions can be downloaded.
- Later: correct the predictions and send them back into a dataset (active learning).

### 8. Export

- Formats: ONNX, TensorRT, OpenVINO, NCNN, TFLite, CoreML.
- Precision: FP32, FP16 or INT8. **The exported model is re-scored and compared with the original.**
- A speed benchmark on this PC, and later on a Raspberry Pi over SSH.
- The output is a model package.

### 9. Compute

- An *executor* layer.
- Phase 1: the local GPU.
- Later: another PC over SSH, rented cloud GPUs (e.g. RunPod), or Colab, monitored on the same screens.

### 10. Professional extras

- CLI and Python API covering everything.
- YAML configs that can live in git.
- Optional TensorBoard, MLflow or W&B logging.
- Hyperparameter sweeps (grid or random) under a time budget.
- Custom callbacks.
- A glossary page.

## Phases

| Phase | Contents |
|---|---|
| **1: first usable** | YOLO dataset import, health report, gallery, wizard (model dropdowns, task, image size, epochs, augmentation presets, dropping classes), live monitor with pause/resume/stop, checkpoints and crash recovery, keep-awake, phone notifications, results page, playground, ONNX export, local GPU |
| 2 | Augmentation preview, confusion-matrix drill-down, run comparison, queue, merging classes and re-splitting, COCO and classification-folder import, all advanced parameters |
| 3 | All export formats with INT8 checks, Pi benchmark, sweeps, SSH/cloud executor, active learning |

## Decisions

- **Name and visibility:** public repository `model-trainer`.
- **Licence:** AGPL-3.0, required because it builds on Ultralytics (AGPL-3.0).
- **Phone notifications:** [ntfy](https://ntfy.sh). It's free, needs no account, and has an Android/iOS app; you
  subscribe to a private topic name. Telegram can be added later.
- **Technology:**
  - Python ≥ 3.10 and Ultralytics as the first backend;
  - NiceGUI for the browser app;
  - training in a worker subprocess that talks to the GUI through files in the run folder (status, metrics,
    commands).
