# Pipeline

This document describes the staged GeoVLM-SceneReasoner pipeline.

## 1. Dataset Check

```powershell
python scripts/00_check_dataset.py
python scripts/01_validate_questions.py --questions data/questions.json
```

The validation scripts check image availability and question schema consistency.

## 2. Object Detection

```powershell
python scripts/04_detect_objects.py --model yolo11n.pt --device cuda --overwrite
```

Use CPU if needed:

```powershell
python scripts/04_detect_objects.py --model yolo11n.pt --device cpu --overwrite
```

Outputs:

```text
outputs/detections/
```

YOLO is a pretrained object detector. It can detect many common object categories, but it can still miss objects or assign the wrong class, especially for unusual viewpoints, blur, occlusion, or closed objects such as a laptop that looks like a book.

## 3. Detection Visualization

```powershell
python scripts/05_visualize_detections.py --overwrite
```

Outputs:

```text
outputs/visualizations/detections/
```

Review these images before segmentation. Bad detections become bad segmentation prompts.

## 4. Detection Normalization

```powershell
python scripts/06_normalize_detections.py --overwrite
```

Inputs:

```text
outputs/detections/
configs/detection_corrections.example.json
```

Outputs:

```text
outputs/detections_normalized/
```

Normalization preserves the original YOLO label in `raw_label` and writes the corrected label to `label`.

## 5. Segmentation

```powershell
python scripts/07_segment_objects.py --model sam2_t.pt --device cuda --overwrite
```

Use CPU if needed:

```powershell
python scripts/07_segment_objects.py --model sam2_t.pt --device cpu --overwrite
```

Inputs:

```text
data/images/
outputs/detections_normalized/
```

Outputs:

```text
outputs/masks/<image_stem>/<object_id>.png
outputs/masks/<image_stem>/segments.json
```

SAM2 is used as an existing segmentation model. In this pipeline it receives detection boxes as prompts, so segmentation quality depends on detection quality.

## 6. Mask Visualization

```powershell
python scripts/08_visualize_masks.py --overwrite
```

Outputs:

```text
outputs/visualizations/masks/
```

Review whether each mask covers the intended object rather than background or adjacent objects.

## 7. Depth Estimation

```powershell
python scripts/09_estimate_depth.py --device cuda --overwrite
```

Use CPU if needed:

```powershell
python scripts/09_estimate_depth.py --device cpu --overwrite
```

Default model:

```text
depth-anything/Depth-Anything-V2-Small-hf
```

Outputs:

```text
outputs/depth/<image_stem>.npy
outputs/depth/<image_stem>_preview.jpg
outputs/depth/<image_stem>.json
```

Depth Anything V2 produces monocular relative depth. The values are useful for ranking and comparison, not for metric centimeter or meter estimates.

## 8. Geometry Extraction

```powershell
python scripts/10_extract_geometry.py --overwrite
```

Inputs:

```text
outputs/masks/<image_stem>/segments.json
outputs/masks/<image_stem>/<object_id>.png
outputs/depth/<image_stem>.npy
```

Outputs:

```text
outputs/geometry/<image_stem>.json
```

Each geometry file contains object centers, mask area, bounding boxes, relative depth statistics, coarse position tags, and pairwise spatial relations.

By default, the script assumes higher relative depth values mean closer objects. If manual review shows the opposite for a depth model or environment, rerun:

```powershell
python scripts/10_extract_geometry.py --lower-depth-is-closer --overwrite
```

## 9. Reasoning Prompts

```powershell
python scripts/11_build_reasoning_prompts.py --overwrite
```

Outputs:

```text
outputs/reasoning/prompts.jsonl
```

Each prompt record contains metadata, a pure VLM prompt, a geometry-only LLM prompt, a GeoVLM prompt, and missing target object information.

## 10. Geometry Rule Baseline

```powershell
python scripts/12_run_geometry_rule_baseline.py --overwrite
```

Outputs:

```text
outputs/reasoning/geometry_rule_baseline.jsonl
outputs/evaluations/geometry_rule_baseline_summary.json
```

The baseline uses transparent rules for supported question types:

- `closer_farther`: pairwise depth relation, then median relative depth
- `physical_size`: mask area fraction
- `support_relation`: simple position, depth, and area heuristic

Missing target objects are answered as `unknown`.

## 11. Failure Report

```powershell
python scripts/13_build_failure_report.py --overwrite
```

Outputs:

```text
outputs/evaluations/pipeline_failure_report.json
outputs/evaluations/pipeline_failure_report.csv
```

Use this report to identify which labels and images caused upstream failures.

## 12. Evaluation Splits

```powershell
python scripts/14_build_evaluation_splits.py --overwrite
```

Outputs:

```text
outputs/evaluations/evaluation_splits.json
outputs/evaluations/evaluation_review.csv
```

The split separates upstream pipeline failures from geometry-available candidates. Candidate records still require manual mask and depth review before final evaluation.

## 13. Clean Subset Generation

After reviewing `outputs/evaluations/evaluation_review.csv`, fill these columns for each geometry-available candidate:

```text
mask_ok
depth_ok
question_valid
final_split
review_notes
```

Then run:

```powershell
python scripts/15_build_clean_subset.py --overwrite
```

Outputs:

```text
outputs/evaluations/clean_subset.json
outputs/evaluations/clean_subset_questions.json
```

Only rows with `automatic_split=geometry_available_candidates` and positive manual review values for `mask_ok`, `depth_ok`, and `question_valid` enter the clean subset. This clean subset should be used for final Pure VLM, Geometry-only LLM, and GeoVLM comparison.

## 14. Clean Geometry Baseline

After generating the clean subset, rerun the geometry-only rule baseline only on reviewed questions:

```powershell
python scripts/12_run_geometry_rule_baseline.py --question-ids outputs/evaluations/clean_subset.json --output outputs/reasoning/geometry_rule_baseline_clean.jsonl --summary outputs/evaluations/geometry_rule_baseline_clean_summary.json --overwrite
```

Outputs:

```text
outputs/reasoning/geometry_rule_baseline_clean.jsonl
outputs/evaluations/geometry_rule_baseline_clean_summary.json
```

This gives a fairer geometry-only baseline by excluding upstream missing-target failures and using the same question set that later VLM and GeoVLM runs should use.
