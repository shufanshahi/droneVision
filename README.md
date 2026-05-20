# droneVision — Aerial Human + Car Detection, Counting & Tracking

An end-to-end computer-vision pipeline for drone / aerial imagery, built on the
[VisDrone2019-DET](http://aiskyeye.com/) dataset and trained with **YOLOv11X**
(Ultralytics). The entire workflow lives in a single, Kaggle-runnable notebook:
[visdrone.ipynb](visdrone.ipynb).

The pipeline covers the full assessment scope: dataset understanding, model
training, detection with human counting, multi-object tracking, and evaluation.

---

## What this project does

| # | Task | Notebook section |
|---|------|------------------|
| 1 | Dataset understanding (EDA) | Section 2 |
| 2 | Preprocessing (label remap, class balancing) | Section 3 |
| 3 | Model training (YOLOv11X) | Section 4 |
| 4 | Detection + human counting (tiled inference) | Section 5 |
| 5 | Multi-object tracking (ByteTrack) | Section 6 |
| 6 | Evaluation + metrics | Section 7 |
| 7 | Discussion of strengths / limitations | Section 8 |

---

## Key design choices

These follow directly from the exploratory analysis in Section 2:

- **Merged class schema (9 classes).** VisDrone's original `pedestrian` and
  `people` classes are visually the same target for this task, so they are
  merged into a single `human` class. The remaining 8 classes shift down by one.
- **High-resolution training (`imgsz=1024`).** A large fraction of aerial
  objects are tiny (under 0.1% of image area), so the model trains at 1024 px to
  preserve feature-map resolution for small targets.
- **Class-balanced oversampling.** The dataset is heavily long-tailed (e.g.
  `awning-tricycle` and `bus` are roughly 25x rarer than `car`). Images
  containing rare classes are repeated so each class contributes more evenly to
  the gradient.
- **Tiled inference with cross-tile NMS.** At test time, a sliding 1024 px
  window (20% overlap) plus a full-frame pass are merged with per-class NMS to
  reliably recover tiny objects.

### 9-class schema

```
0: human      1: bicycle   2: car
3: van        4: truck     5: tricycle
6: awning-tricycle         7: bus       8: motor
```

Original VisDrone IDs 10 (`ignored regions`) and 11 (`others`) are dropped
during the label remap so they never pollute training.

---

## Pipeline overview

### Section 1 — Setup
Central configuration block: `TEST_MODE` flag for a fast smoke-test, the 9-class
schema, the `CLASS_REMAP` table (10 -> 9), dataset paths (Kaggle layout), and
dependency installation (`ultralytics>=8.3.0`, `albumentations`, `supervision`,
`huggingface_hub`).

### Section 2 — Exploratory Data Analysis
Four views that motivate the rest of the pipeline:
1. Per-class object counts (log scale) — exposes the long-tail skew.
2. Bounding-box scale distribution and tiny/small/medium/large buckets.
3. Spatial heatmap of object centres — reveals a strong centre bias.
4. Sample ground-truth visualizations on raw 10-class labels.

### Section 3 — Preprocessing
- Remaps val and test labels to the 9-class schema (so `model.val()` compares
  against the same schema used for training).
- Builds a class-balanced training set via image-level oversampling:
  rare classes (`tricycle`, `awning-tricycle`, `bus`) at 4x, moderate classes
  (`bicycle`, `van`, `truck`) at 2x, everything else 1x.
- Uses symlinks for val/test images to save disk space.
- Writes the Ultralytics dataset YAML (`visdrone_human_car.yaml`).

### Section 4 — Training (YOLOv11X)
Fine-tunes YOLOv11X (~57 M parameters) from the official COCO checkpoint
(`yolo11x.pt`). YOLO11 is anchor-free (`box + cls + dfl` loss; no anchor YAML).

| Knob | Value | Rationale |
|------|-------|-----------|
| `imgsz` | 1024 | Tiny objects need feature-map resolution |
| `batch` | 4 | Larger batches OOM at 1024 px on a single T4 |
| `epochs` | 10 | Validation mAP plateaus by epoch 7-8 |
| `mosaic` | 1.0 | Preserves small objects under scale jitter |
| `copy_paste` | 0.4 | Synthesises rare-class instances |
| `translate` | 0.15 | Pushes objects to borders (counters centre bias) |

Multi-GPU training uses `device='0,1'` when two GPUs are present.

### Section 5 — Inference, Detection & Counting
Implements two-pass tiled inference:
1. Slide a 1024x1024 window across the image with 20% overlap, predict per tile.
2. Run one full-image pass to catch large objects spanning tile borders.
3. Merge all detections with per-class NMS (`torchvision.ops.batched_nms`).

Predictions are saved in normalized YOLO format. Only the task classes
(`human`, `car`) are kept for visualization; humans are counted and drawn with a
clean overlay (green = human, red = car), plus a human-count distribution
histogram across the inferred test slice.

### Section 6 — Multi-Object Tracking (ByteTrack)
Downloads one scene from the
[`Voxel51/visdrone-mot`](https://huggingface.co/datasets/Voxel51/visdrone-mot)
HuggingFace dataset and runs Ultralytics' built-in ByteTrack
(`model.track(..., persist=True)`). ByteTrack is chosen because it keeps
low-confidence detections that simple SORT discards, which helps in dense aerial
crowds. The result is annotated with `supervision` (boxes, labels, traces),
counts unique humans by tracker ID, and is re-encoded to H.264 for inline
playback. A callback strips degenerate (near-zero-size) boxes before they reach
the Kalman filter to avoid NaN covariance errors.

### Section 7 — Evaluation & Metrics
Three views on quality:
1. Canonical Ultralytics validation (`model.val()`) on the remapped test split —
   precision, recall, mAP@0.5, mAP@0.5:0.95, per class.
2. A custom 11-point VOC AP@0.5 curve computed only for `human` + `car` against
   the saved predictions.
3. FPS / latency measurement for single-frame inference at `imgsz=1024`.

### Section 8 — Discussion
Documents strengths (task-aligned schema, small-object handling, imbalance
robustness), limitations (noisy single-frame counting, ~4-7 FPS latency on a T4,
aerial domain shift), challenges encountered, and suggested next steps.

---

## Requirements

- Python 3.12
- PyTorch with CUDA (GPU strongly recommended)
- `ultralytics>=8.3.0` (ships YOLO11), `albumentations`, `supervision`,
  `huggingface_hub`
- `ffmpeg` (for inline H.264 video re-encoding in the tracking section)
- The VisDrone2019-DET dataset, mounted at the path configured in Section 1

The notebook is written for a Kaggle environment. Dataset paths default to:

```
/kaggle/input/datasets/banuprasadb/visdrone-dataset/VisDrone_Dataset
```

with `VisDrone2019-DET-train`, `VisDrone2019-DET-val`, and
`VisDrone2019-DET-test-dev` subfolders, each containing `images/` and `labels/`
(YOLO format). Adjust `DATASET_ROOT` / `WORK_DIR` in Section 1 to run elsewhere.

---

## How to run

1. Open [visdrone.ipynb](visdrone.ipynb) in Kaggle (or a local Jupyter
   environment with a GPU).
2. Ensure the VisDrone dataset is mounted and the paths in Section 1 match.
3. For a quick smoke-test, set `TEST_MODE = True` in Section 1 (runs on
   `SUBSET_SIZE = 50` images per split and 2 epochs).
4. For a full run, leave `TEST_MODE = False`. A complete training run from the
   COCO checkpoint takes roughly 10 hours on a Kaggle T4 x2.
5. Run all cells top to bottom. The Setup block holds all configuration; every
   later section reads from it.

Notes:
- The tracking section (Section 6) requires internet access (turn on
  "Internet -> On" in Kaggle); it degrades gracefully if frames cannot be
  fetched.
- The test-set inference in Section 5 caps at 200 images by default for
  notebook runtime; remove that slice for a complete test-set evaluation.

---

## Repository structure

```
droneVision/
  visdrone.ipynb    The complete pipeline (EDA -> training -> inference -> tracking -> evaluation)
  README.md         This file
```

All intermediate artifacts (balanced training set, remapped labels, dataset
YAML, trained weights, predictions, tracking video) are generated at runtime
under the working directory and are not committed to the repository.
