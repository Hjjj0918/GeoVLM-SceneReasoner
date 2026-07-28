# CalibVLM

Geometry-aware vision-language perception for embodied AI.

This repository is starting with **CalibVLM-lite**, a lightweight research-oriented desktop-scene pipeline. The core idea is to compare ordinary VLM spatial reasoning against a geometry-aware version that also receives structured observations such as object boxes, masks, estimated depth, camera intrinsics, and approximate 3D camera coordinates.

## Stage 1: Dataset Setup and Camera Calibration

Current stage:

- Define the project structure.
- Collect real desktop images.
- Collect chessboard calibration images.
- Estimate camera intrinsics with OpenCV.
- Save intrinsics in a JSON format that later geometry code can consume.

Not included yet:

- YOLO object detection.
- SAM segmentation.
- Depth Anything depth estimation.
- VLM or LLM reasoning.
- Evaluation benchmark.

## Setup

Use Python 3.10 or newer. A conda environment is recommended.

```powershell
python -m pip install -r requirements.txt
```

Stage 1 uses CPU only. Your RTX 4060 8GB is not needed until later model stages.

## Directory Layout

```text
configs/
  camera_intrinsics.example.json
data/
  images/
  calibration/
  annotations.json
outputs/
  detections/
  masks/
  depth/
  object_3d/
  vlm_answers/
  visualizations/
scripts/
  00_check_dataset.py
  01_calibrate_camera.py
report/
  calibvlm_project_note.md
tests/
```

Raw images and generated outputs are ignored by Git by default. The folder structure is preserved with `.gitkeep` files.

## Data Collection

Place desktop scene images in:

```text
data/images/
```

Recommended names:

```text
scene_0001.jpg
scene_0002.jpg
scene_0003.jpg
```

For the MVP, collect 30-50 images. Each image should contain 3-6 common tabletop objects such as a cup, mouse, book, bottle, keyboard, pen, notebook, or laptop.

Place chessboard calibration images in:

```text
data/calibration/
```

Recommended names:

```text
calib_0001.jpg
calib_0002.jpg
calib_0003.jpg
```

Use 10-20 calibration images from varied positions and angles. Keep the chessboard flat and fully visible. The default script assumes a 9 by 6 inner-corner chessboard with 25 mm squares. Change the CLI arguments if your physical board is different.

## Check Dataset

```powershell
python scripts/00_check_dataset.py
```

This prints:

- Number of scene images.
- Number of calibration images.
- Image dimensions.
- Warnings for inconsistent dimensions.
- Warnings for filenames outside the recommended pattern.

Empty folders are allowed at this stage. The script still runs so you can verify the project structure before collecting data.

## Calibrate Camera

Default command:

```powershell
python scripts/01_calibrate_camera.py
```

If your chessboard differs from the default, pass the real inner-corner count and square size:

```powershell
python scripts/01_calibrate_camera.py --rows 9 --cols 6 --square-size-m 0.025
```

Important: `--rows` and `--cols` are inner corners, not the number of black/white squares.

The script writes:

```text
configs/camera_intrinsics.json
```

That file contains:

- `fx`, `fy`, `cx`, `cy`
- `camera_matrix`
- `distortion_coefficients`
- `image_width`, `image_height`
- `reprojection_error`
- calibration board metadata

## Success Criteria

Stage 1 is successful when:

- `python scripts/00_check_dataset.py` runs and reports dataset status.
- Valid chessboard images in `data/calibration/` produce `configs/camera_intrinsics.json`.
- The reprojection error is printed.
- The JSON contains camera intrinsics and distortion coefficients.
- Failure cases are explicit, for example no calibration images or too few detected chessboards.

## Common Calibration Problems

- Wrong `--rows` or `--cols`: use inner corners, not square count.
- Motion blur or glare: retake sharper images.
- Chessboard partly outside the image: keep the whole board visible.
- Too few angles: capture the board near image corners and at different tilts.
- Mixed image resolutions: keep all calibration images at the same resolution.

## GPU Fallback

Stage 1 has no GPU requirement.

For later stages on RTX 4060 8GB:

- Run detection, segmentation, depth, geometry, and VLM reasoning as separate scripts.
- Save intermediate JSON, masks, and depth maps.
- Prefer small model variants first.
- Avoid loading YOLO, SAM, Depth Anything, and a VLM at the same time.
