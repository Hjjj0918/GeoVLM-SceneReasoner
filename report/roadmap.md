# GeoVLM-SceneReasoner Roadmap

## Project Identity

GeoVLM-SceneReasoner: Geometry-Aware Visual Reasoning for Vision-Language Models.

The project evaluates whether vision-language models can reliably answer spatial and physical reasoning questions in real images, and whether explicit object-level geometry improves answer accuracy.

## Core Pipeline

```text
image
-> object detection
-> SAM2 segmentation
-> Depth Anything V2 depth estimation
-> object-level spatial representation
-> VLM / LLM reasoning
-> pure VLM vs geometry-aware comparison
```

## Stage 1: Repository Foundation

Status: completed.

Completed scope:

- Project name, README, and research note.
- Dataset layout under `data/images/`.
- Output layout for detections, masks, depth, geometry, reasoning, evaluations, and visualizations.
- Benchmark question schema in `data/questions.example.json`.
- Object-level geometry example in `configs/geometry_schema.example.json`.
- Pipeline example in `configs/pipeline.example.json`.
- Dataset check script: `scripts/00_check_dataset.py`.
- Question validation script: `scripts/01_validate_questions.py`.
- Unit tests for dataset checks and question validation.

Verification commands:

```powershell
python scripts\00_check_dataset.py
python scripts\01_validate_questions.py --questions data\questions.example.json
python -m pytest -q
```

## Stage 2: Real-Image Benchmark

Goal: build the first usable benchmark split.

Tasks:

- Add 30-50 real images to `data/images/`.
- Use filenames such as `scene_0001.jpg`, `scene_0002.jpg`, and `scene_0003.jpg`.
- Create `data/questions.json` from `data/questions.example.json`.
- Write 3-5 questions per image for the first 5-10 images.
- Cover `closer_farther`, `left_right`, `occlusion`, `support_relation`, and `physical_size`.
- Validate the question file before adding model pipelines.

Exit criteria:

- `python scripts\00_check_dataset.py` reports real images.
- `python scripts\01_validate_questions.py --questions data\questions.json` passes.
- The benchmark contains enough manually checked questions to evaluate a baseline.

## Stage 3: Object Detection

Goal: detect candidate objects and save reusable JSON outputs.

Recommended first implementation:

- Use a small YOLO model variant.
- Run detection stage independently.
- Save per-image detection files under `outputs/detections/`.
- Keep labels, confidence scores, and `bbox_xyxy`.

Exit criteria:

- Each benchmark image has a detection JSON file.
- Detection output can be inspected without loading any other model.

## Stage 4: Segmentation

Goal: generate object masks for detected objects.

Recommended first implementation:

- Use SAM2 tiny or small.
- Use detection boxes as prompts.
- Save masks under `outputs/masks/`.
- Link each mask from the object geometry record.

Exit criteria:

- Each detected object has a mask path or a clear failure reason.
- Masks can be visualized for quick quality checks.

## Stage 5: Depth Estimation

Goal: estimate relative scene depth for spatial reasoning.

Recommended first implementation:

- Use Depth Anything V2 small.
- Save depth maps under `outputs/depth/`.
- Compute object-level median depth from masks.

Exit criteria:

- Each benchmark image has a depth output.
- Each segmented object has a relative depth summary.

## Stage 6: Object-Level Geometry

Goal: combine detection, segmentation, and depth into structured object summaries.

Geometry fields should include:

- object id and label
- bounding box
- bbox center
- mask centroid
- mask area
- relative median depth
- horizontal and vertical position tags
- depth order hints
- occlusion hints

Exit criteria:

- Geometry JSON files are written under `outputs/geometry/`.
- Geometry records match the schema in `configs/geometry_schema.example.json`.

## Stage 7: Reasoning and Evaluation

Goal: compare pure visual reasoning against geometry-aware reasoning.

Evaluation variants:

- Pure VLM: image plus question.
- Geometry-only LLM: object-level geometry plus question.
- GeoVLM: image, geometry context, and question.

Metrics:

- overall accuracy
- per-question-type accuracy
- reasoning consistency
- failure case analysis

Exit criteria:

- Outputs are saved under `outputs/reasoning/`.
- Evaluation tables are saved under `outputs/evaluations/`.
- The report can show whether geometry helps, where it helps, and where it fails.

## Current Next Step

Collect real images and create `data/questions.json`. The model pipeline should start only after the benchmark file validates cleanly.
