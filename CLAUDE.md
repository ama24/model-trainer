# CLAUDE.md — model-trainer

A generic browser app (NiceGUI) and CLI for training YOLO-family models with Ultralytics. The design and the phase
plan are in [docs/DESIGN.md](docs/DESIGN.md). Read it before changing anything.

## Rules
- **Generic:** no project-specific code (screws, cameras, ...). Projects exchange files with the trainer:
  - datasets in YOLO/COCO format go in;
  - model packages come out (weights, exports, `model_card.json`).
- **Workflow:**
  - All code lives on `main`. Each piece of work gets a short-lived branch (`feature/<name>`, `fix/<name>`), then
    a PR into `main`, then the branch is deleted.
  - Never push to `main` directly.
- Run `pytest` and `ruff check .` before every commit.
- **Training runs in a worker subprocess,** never in the GUI process. The two talk through files in the run folder.
- **Writes must be atomic** (temporary file, then rename). Never overwrite a checkpoint in place.
- **Every parameter shown in the GUI** has a help text: what it does, a typical value, and the effect of changing it.
- **Code standards:** type hints, logging instead of print, no hard-coded paths, and seeds for every random choice.
