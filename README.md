# GeoVLM-SceneReasoner

**Geometry-Aware Visual Reasoning for Vision-Language Models**

GeoVLM-SceneReasoner is a lightweight research pipeline for evaluating whether explicit object-level geometry can improve visual spatial reasoning in vision-language models.

The project focuses on real-image questions such as:

- Which object is closer to the camera?
- Is one object left or right of another object?
- Which object is most likely on the table?
- Which object appears physically larger, not just larger in image area?

The repository is designed for small-scale experiments. It does not train a new VLM. Instead, it builds a staged perception and reasoning pipeline that can compare image-only VLM answers against geometry-aware alternatives.

## Motivation

Recent multimodal reasoning benchmarks show that strong VLMs can still fail on genuinely visual reasoning, especially spatial relations, object grounding, occlusion, and physical plausibility. GeoVLM-SceneReasoner studies a practical question:

```text
Can detection, segmentation, and monocular depth provide useful geometry signals for VLM spatial reasoning?
```

The project also separates two failure modes that are often mixed together:

- **Perception failure:** the pipeline misses, mislabels, or poorly segments the target object.
- **Reasoning failure:** the relevant objects and geometry are available, but the reasoning answer is still wrong.

This distinction is important when evaluating whether geometry actually helps VLM reasoning.

## Pipeline

```text
image
-> object detection
-> detection normalization
-> SAM2 segmentation
-> Depth Anything V2 relative depth estimation
-> object-level geometry extraction
-> reasoning prompt generation
-> geometry-only baseline / VLM comparison
-> evaluation split and failure analysis
```

The current implementation provides the full preprocessing and geometry-baseline path. VLM API or local VLM inference is planned but not included yet.

## Features

| Component | Status |
|---|---|
| Dataset structure and validation | Implemented |
| Multi-view image renaming | Implemented |
| Question scaffold generation | Implemented |
| YOLO object detection | Implemented |
| Detection visualization | Implemented |
| Detection label normalization | Implemented |
| SAM2 segmentation from detection boxes | Implemented |
| Mask visualization | Implemented |
| Depth Anything V2 relative depth estimation | Implemented |
| Object-level geometry extraction | Implemented |
| Pure VLM / Geometry-only / GeoVLM prompt generation | Implemented |
| Geometry-only rule baseline | Implemented |
| Pipeline failure report | Implemented |
| Evaluation split generation | Implemented |
| Manual-review clean subset generation | Implemented |
| VLM inference runner | Planned |
| Final comparison tables | Planned |

## Installation

Use Python 3.10 or newer. A conda environment is recommended.

```powershell
python -m pip install -r requirements.txt
```

CUDA is optional but recommended for segmentation and depth estimation. The pipeline is staged so that detection, segmentation, depth, geometry, and reasoning can be run separately.

## Data Preparation

Place images in:

```text
data/images/
```

Recommended naming:

```text
scene_0001_view_00.jpg
scene_0001_view_01.jpg
scene_0002_view_00.jpg
```

For local experiments, 30-50 real desktop or indoor images are enough. Public releases should not include private photos unless they have been reviewed for privacy.

More details are in [docs/data_collection.md](docs/data_collection.md).

## Quick Start

Validate the dataset and questions:

```powershell
python scripts/00_check_dataset.py
python scripts/01_validate_questions.py --questions data/questions.json
```

Run the staged pipeline:

```powershell
python scripts/04_detect_objects.py --model yolo11n.pt --device cuda --overwrite
python scripts/06_normalize_detections.py --overwrite
python scripts/07_segment_objects.py --model sam2_t.pt --device cuda --overwrite
python scripts/08_visualize_masks.py --overwrite
python scripts/09_estimate_depth.py --device cuda --overwrite
python scripts/10_extract_geometry.py --overwrite
python scripts/11_build_reasoning_prompts.py --overwrite
python scripts/12_run_geometry_rule_baseline.py --overwrite
python scripts/13_build_failure_report.py --overwrite
python scripts/14_build_evaluation_splits.py --overwrite
python scripts/15_build_clean_subset.py --overwrite
python scripts/12_run_geometry_rule_baseline.py --question-ids outputs/evaluations/clean_subset.json --output outputs/reasoning/geometry_rule_baseline_clean.jsonl --summary outputs/evaluations/geometry_rule_baseline_clean_summary.json --overwrite
```

If CUDA is unavailable, use `--device cpu` for the detection, segmentation, and depth scripts.

Detailed script documentation is in [docs/pipeline.md](docs/pipeline.md).

## Outputs

Generated outputs are written under:

```text
outputs/
  detections/
  detections_normalized/
  masks/
  depth/
  geometry/
  reasoning/
  evaluations/
  visualizations/
```

These files are ignored by Git because they may contain local images, generated masks, depth maps, and experiment results.

## Evaluation

The intended comparison is:

| Track | Input |
|---|---|
| Pure VLM | image + question |
| Geometry-only LLM | object-level geometry + question |
| GeoVLM | image + object-level geometry + question |

The current implemented baseline is a transparent geometry-only rule system. It reports:

- end-to-end accuracy
- answered accuracy
- coverage
- unknown rate
- missing-target failure counts

The evaluation split script separates records into:

- `pipeline_failure`: target objects are missing from upstream perception output.
- `geometry_available_candidates`: target objects are available, but masks and depth still need manual review.

After manually reviewing candidate rows in `outputs/evaluations/evaluation_review.csv`, build a formal clean subset:

```powershell
python scripts/15_build_clean_subset.py --overwrite
```

The clean subset includes only rows where `automatic_split=geometry_available_candidates` and `mask_ok`, `depth_ok`, and `question_valid` are all marked as `yes`.

Run the geometry-only rule baseline on that reviewed subset:

```powershell
python scripts/12_run_geometry_rule_baseline.py --question-ids outputs/evaluations/clean_subset.json --output outputs/reasoning/geometry_rule_baseline_clean.jsonl --summary outputs/evaluations/geometry_rule_baseline_clean_summary.json --overwrite
```

More details are in [docs/evaluation.md](docs/evaluation.md).

## Limitations

This project uses existing perception models and does not guarantee perfect geometry.

- YOLO may miss or misclassify objects.
- SAM2 masks are prompted by detection boxes, so segmentation can inherit detection errors.
- Depth Anything V2 provides monocular relative depth, not metric 3D distance.
- Image-space size is not the same as real physical size.
- Geometry-available examples still require manual mask and depth review before final evaluation.

These limitations are part of the research question: explicit geometry can help only when the perception signals are good enough, and the pipeline should make perception failures visible instead of hiding them inside a single accuracy number.

## Repository Layout

```text
configs/       Example pipeline and geometry configuration files
data/          Question schemas and local image directory
docs/          Public project documentation
outputs/       Generated artifacts, ignored by Git
scripts/       Pipeline scripts
tests/         Unit tests
```

## Testing

Run:

```powershell
python -m pytest -q
```

## References

- EasyARC: Evaluating Vision Language Models on True Visual Reasoning, arXiv:2506.11595.
- VisuLogic: A Benchmark for Evaluating Visual Reasoning in Multi-modal Large Language Models, arXiv:2504.15279.
- VLM2-Bench dataset: `Sterzhang/vlm2-bench` on Hugging Face.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

External datasets, model checkpoints, and generated annotations may have separate licenses. Verify their terms before redistributing them.
