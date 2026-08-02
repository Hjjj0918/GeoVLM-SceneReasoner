# GeoVLM-SceneReasoner

**GeoVLM-SceneReasoner: Geometry-Aware Visual Reasoning for Vision-Language Models**

This project asks whether VLMs are reliable on real-image spatial reasoning, and whether explicit object-level geometry can improve their answers.

Core question:

```text
Can detection, segmentation, and depth-derived object geometry improve VLM reasoning about spatial relations, distance, occlusion, support, and physical size?
```

## Motivation

Recent VLM reasoning benchmarks argue that strong multimodal models still struggle with genuinely visual reasoning. EasyARC focuses on true visual reasoning, VisuLogic evaluates vision-centric reasoning categories such as spatial relations and attribute comparison, and VLM2-Bench provides a larger VQA-style reference dataset.

This repo does not try to train a new model. It builds a small benchmark and a staged inference pipeline:

```text
image
-> object detection
-> SAM2 segmentation
-> Depth Anything V2 depth estimation
-> object-level spatial representation
-> VLM / LLM reasoning
-> pure VLM vs geometry-aware comparison
```

## Current Stage

Current stage: real-image benchmark setup, object detection, detection normalization, SAM2 segmentation, review visualizations, Depth Anything V2 depth estimation, object-level geometry extraction, reasoning prompt generation, a geometry-only rule baseline, pipeline failure reporting, and evaluation split generation.

Included:

- Project identity, dataset layout, benchmark question schema, and validation scripts.
- Multi-view image renaming.
- Question scaffold generation for same-scene multi-view captures.
- YOLO detection script.
- Detection normalization for common label corrections.
- SAM2 segmentation from normalized detections.
- Detection and mask visualization scripts for manual review.
- Depth Anything V2 relative depth estimation.
- Object-level geometry extraction from masks and depth maps.
- Prompt generation for Pure VLM, Geometry-only LLM, and GeoVLM comparisons.
- Geometry-only rule baseline with accuracy and coverage summary.
- Failure report generation for missing target objects.
- Evaluation split generation for separating upstream pipeline failures from geometry-available candidates.

Not included yet:

- VLM API or local VLM inference.
- Full comparison tables across Pure VLM, Geometry-only LLM, and GeoVLM.

## Setup

Use Python 3.10 or newer. A conda environment is recommended.

```powershell
python -m pip install -r requirements.txt
```

Stage 1 is CPU-only.

## Directory Layout

```text
configs/
  pipeline.example.json
  geometry_schema.example.json
data/
  images/
  annotations.json
  question_templates.example.json
  questions.example.json
  questions.json
outputs/
  detections/
  detections_normalized/
  masks/
  depth/
  geometry/
  reasoning/
  evaluations/
  visualizations/
scripts/
  00_check_dataset.py
  01_validate_questions.py
  02_rename_images.py
  03_scaffold_questions.py
  04_detect_objects.py
  05_visualize_detections.py
  06_normalize_detections.py
  07_segment_objects.py
  08_visualize_masks.py
  09_estimate_depth.py
  10_extract_geometry.py
  11_build_reasoning_prompts.py
  12_run_geometry_rule_baseline.py
  13_build_failure_report.py
  14_build_evaluation_splits.py
report/
  project_note.md
  roadmap.md
tests/
```

Raw images, generated outputs, model weights, local environments, and `docs/` planning artifacts are ignored by Git.

## Data Collection

Place real images in:

```text
data/images/
```

Recommended names:

```text
scene_0001.jpg
scene_0002.jpg
scene_0003.jpg
```

For an MVP, use 30-50 real images. Desktop scenes are enough. Each image should contain 3-6 common objects such as a cup, mouse, book, bottle, keyboard, phone, pen, laptop, or notebook.

The first version uses image-space relations and relative depth:

- left/right from object centers
- closer/farther from relative depth
- larger/smaller from mask area and depth cues
- possible occlusion from mask/bbox overlap and depth ordering

## Benchmark Questions

Start from:

```text
data/questions.example.json
```

For a single image or a small manual benchmark, copy it to:

```text
data/questions.json
```

For a same-scene multi-view capture, edit:

```text
data/question_templates.example.json
```

Then expand the templates across every image view:

```powershell
python scripts/03_scaffold_questions.py
```

This writes:

```text
data/questions.draft.json
```

Review the object names, answers, and ambiguous views before saving the final benchmark as `data/questions.json`.

Question types:

- `closer_farther`
- `left_right`
- `front_back`
- `occlusion`
- `support_relation`
- `physical_size`

Example:

```json
{
  "question_id": "scene_0001_q001",
  "image": "scene_0001.jpg",
  "question": "Which object is closer to the camera, the cup or the laptop?",
  "type": "closer_farther",
  "target_objects": ["cup", "laptop"],
  "answer": "cup",
  "evaluation": {
    "metric": "exact_match",
    "acceptable_answers": ["cup"]
  }
}
```

## Validation

Check dataset status:

```powershell
python scripts/00_check_dataset.py
```

Validate questions:

```powershell
python scripts/01_validate_questions.py --questions data/questions.example.json
python scripts/01_validate_questions.py --questions data/questions.draft.json
python scripts/01_validate_questions.py --questions data/questions.json
```

## Object Detection

Run a dry run first:

```powershell
python scripts/04_detect_objects.py --dry-run --limit 3
```

Run YOLO detection:

```powershell
python scripts/04_detect_objects.py --model yolo11n.pt --device cuda --overwrite
```

If CUDA is unavailable, use CPU:

```powershell
python scripts/04_detect_objects.py --model yolo11n.pt --device cpu --overwrite
```

Detection JSON files are written to:

```text
outputs/detections/
```

Visualize detection boxes for manual review:

```powershell
python scripts/05_visualize_detections.py --overwrite
```

Visualization images are written to:

```text
outputs/visualizations/detections/
```

Review these images before running segmentation. Low-confidence duplicate boxes or wrong labels should be corrected or filtered before they are used as SAM2 prompts.

Normalize labels after visual review:

```powershell
python scripts/06_normalize_detections.py --overwrite
```

This reads:

```text
outputs/detections/
configs/detection_corrections.example.json
```

and writes:

```text
outputs/detections_normalized/
```

The raw YOLO outputs stay unchanged. Normalized detections preserve the original label in `raw_label`, for example when a closed laptop is detected as `book` and normalized to `laptop`.

## Object Segmentation

Run a dry run first:

```powershell
python scripts/07_segment_objects.py --dry-run --limit 3
```

Run SAM2 segmentation from normalized detections:

```powershell
python scripts/07_segment_objects.py --model sam2_t.pt --device cpu --overwrite
```

If CUDA is available:

```powershell
python scripts/07_segment_objects.py --model sam2_t.pt --device cuda --overwrite
```

This reads:

```text
data/images/
outputs/detections_normalized/
```

and writes:

```text
outputs/masks/<image_stem>/<object_id>.png
outputs/masks/<image_stem>/segments.json
```

Each mask PNG is a binary object mask. Each `segments.json` file preserves the object label, bbox, confidence, mask path, and mask area.

Visualize SAM2 masks for manual review:

```powershell
python scripts/08_visualize_masks.py --overwrite
```

This reads:

```text
data/images/
outputs/masks/
```

and writes:

```text
outputs/visualizations/masks/
```

Each visualization overlays the binary masks on the original image and labels each object with `object_id`, normalized label, and mask area in pixels. Review these images before running depth or geometry extraction, because geometry quality depends on whether the mask actually covers the intended object.

## Depth Estimation

Run a dry run first:

```powershell
python scripts/09_estimate_depth.py --dry-run --limit 3
```

Run Depth Anything V2 on CPU:

```powershell
python scripts/09_estimate_depth.py --device cpu --overwrite
```

If CUDA is available:

```powershell
python scripts/09_estimate_depth.py --device cuda --overwrite
```

By default this uses:

```text
depth-anything/Depth-Anything-V2-Small-hf
```

This reads:

```text
data/images/
```

and writes:

```text
outputs/depth/<image_stem>.npy
outputs/depth/<image_stem>_preview.jpg
outputs/depth/<image_stem>.json
```

The `.npy` file stores the raw relative depth map as `float32`. The `_preview.jpg` file is only for visual review. The `.json` file records the image name, model name, output paths, depth shape, and depth statistics. Depth Anything V2 produces monocular relative depth, so these values should be used for ranking and object-level comparison, not as metric centimeters or meters.

## Geometry Extraction

Run a dry run first:

```powershell
python scripts/10_extract_geometry.py --dry-run --limit 3
```

Extract object-level geometry:

```powershell
python scripts/10_extract_geometry.py --overwrite
```

This reads:

```text
outputs/masks/<image_stem>/segments.json
outputs/masks/<image_stem>/<object_id>.png
outputs/depth/<image_stem>.npy
```

and writes:

```text
outputs/geometry/<image_stem>.json
```

Each geometry JSON contains:

- object centers from bbox and mask centroid
- mask bounds and mask area fraction
- per-object relative depth statistics inside the mask
- coarse position tags such as `left`, `center`, `right`, `top`, `middle`, `bottom`
- depth-order hints such as `near`, `middle`, `far`
- pairwise relations such as `left_of`, `right_of`, `above`, `below`, `closer_than`, `farther_than`, and `similar_depth`

By default the script assumes higher Depth Anything V2 values mean closer objects and records this as:

```text
higher_relative_depth_is_closer
```

If manual review shows the opposite for your environment, rerun with:

```powershell
python scripts/10_extract_geometry.py --lower-depth-is-closer --overwrite
```

## Reasoning Prompts

Build prompt records for the three-way comparison:

```powershell
python scripts/11_build_reasoning_prompts.py --overwrite
```

This reads:

```text
data/questions.json
outputs/geometry/<image_stem>.json
```

and writes:

```text
outputs/reasoning/prompts.jsonl
```

Each JSONL record contains:

- benchmark metadata: `question_id`, `image`, `question`, `type`, `target_objects`, `answer`, and `acceptable_answers`
- `pure_vlm_prompt`: image-only prompt for a VLM
- `geometry_llm_prompt`: text-only prompt using object-level geometry
- `geovlm_prompt`: image plus geometry prompt for a VLM
- `missing_target_objects`: target labels that were not detected in the geometry file

The answer is stored as metadata for evaluation. It is not inserted into the prompt text.

## Geometry Rule Baseline

Run the geometry-only rule baseline:

```powershell
python scripts/12_run_geometry_rule_baseline.py --overwrite
```

This reads:

```text
outputs/reasoning/prompts.jsonl
outputs/geometry/<image_stem>.json
```

and writes:

```text
outputs/reasoning/geometry_rule_baseline.jsonl
outputs/evaluations/geometry_rule_baseline_summary.json
```

The baseline uses simple transparent rules:

- `closer_farther`: pairwise depth relation first, then median relative depth
- `physical_size`: larger mask area fraction
- `support_relation`: target object existence plus a simple position/depth/area heuristic

Each result records `prediction`, `source`, `confidence`, `correct`, and `error_reason`. Questions with missing target objects are answered as `unknown` and counted separately through `error_reason=missing_target_objects`, because those are upstream detection/segmentation failures rather than pure reasoning failures.

The summary reports:

- `end_to_end_accuracy`: correct / total questions
- `answered_accuracy`: correct / answered questions
- `coverage`: answered / total questions
- `unknown_rate`: unknown / total questions
- `accuracy`: compatibility alias for `end_to_end_accuracy`

## Failure Report

Build a report for upstream pipeline failures:

```powershell
python scripts/13_build_failure_report.py --overwrite
```

This reads:

```text
outputs/reasoning/geometry_rule_baseline.jsonl
```

and writes:

```text
outputs/evaluations/pipeline_failure_report.json
outputs/evaluations/pipeline_failure_report.csv
```

Use this report to identify which images and labels caused `missing_target_objects`. These cases should be reviewed separately from reasoning failures.

## Evaluation Splits

Build an evaluation split and a row-level review file:

```powershell
python scripts/14_build_evaluation_splits.py --overwrite
```

This reads:

```text
outputs/reasoning/geometry_rule_baseline.jsonl
```

and writes:

```text
outputs/evaluations/evaluation_splits.json
outputs/evaluations/evaluation_review.csv
```

The script applies an automatic, conservative split:

- `pipeline_failure`: the baseline record has `error_reason=missing_target_objects`. These records indicate that a target object was not available in the upstream detection/segmentation output.
- `geometry_available_candidates`: the target objects were available to the baseline. These records are only candidates, not guaranteed clean examples, because the mask and relative depth still require visual review.
- `manual_review_pending`: the number of `geometry_available_candidates` records that still need mask/depth review.

The CSV contains one row per question, including `question_id`, image, target objects, automatic split, prediction, correctness, error reason, and missing target objects. Review the corresponding files under:

```text
outputs/visualizations/masks/
outputs/depth/*_preview.jpg
```

before treating a candidate as a final evaluation example.

Run tests:

```powershell
python -m pytest -q
```

## Experimental Comparison

The final benchmark should compare:

- **Pure VLM:** image + question.
- **Geometry-only LLM:** object-level geometry JSON/text + question.
- **GeoVLM:** image + object-level geometry text + question.

Metrics:

- accuracy
- closer/farther accuracy
- left/right accuracy
- occlusion accuracy
- physical-size accuracy
- reasoning consistency
- failure case analysis

## GPU Plan

RTX 4060 8GB is enough if the pipeline is staged:

1. detection -> save JSON
2. segmentation -> save masks
3. depth -> save depth maps
4. geometry -> save object-level JSON
5. reasoning -> save answers

Use small model variants first and avoid loading YOLO, SAM2, Depth Anything, and a VLM at the same time.

## References

- EasyARC: Evaluating Vision Language Models on True Visual Reasoning, arXiv:2506.11595.
- VisuLogic: A Benchmark for Evaluating Visual Reasoning in Multi-modal Large Language Models, arXiv:2504.15279.
- VLM2-Bench dataset: `Sterzhang/vlm2-bench` on Hugging Face.
