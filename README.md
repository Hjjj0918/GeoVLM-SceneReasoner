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

The current implementation provides the full preprocessing, geometry-baseline, and model-agnostic VLM inference path.

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
| Model-agnostic VLM inference runner | Implemented |
| Track comparison report | Implemented |

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

After all three VLM tracks finish, compare them without making more API calls:

```powershell
python scripts/17_compare_track_results.py --overwrite
```

This writes:

```text
outputs/evaluations/track_comparison.json
outputs/evaluations/track_disagreements.csv
```

## Run Qwen3-VL-Flash

The VLM inference runner uses an OpenAI-compatible API. The local computer runs the
image pipeline and sends only the VLM reasoning requests to the remote API. Do not
put your API key in source files, JSON files, notebooks, or Git commits.

### 1. Set the API key

In PowerShell, set the key for the current terminal session:

```powershell
$env:DASHSCOPE_API_KEY = "your-dashscope-api-key"
```

Verify that the variable exists without printing the key:

```powershell
if ([string]::IsNullOrWhiteSpace($env:DASHSCOPE_API_KEY)) {
    "DASHSCOPE_API_KEY_NOT_SET"
} else {
    "DASHSCOPE_API_KEY_SET"
}
```

If a new PowerShell window is opened, set the variable again. To save it for
future PowerShell sessions, use Windows user environment variables instead of
hardcoding it in this repository.

### 2. Choose the DashScope endpoint

Use the endpoint that matches the region where the API key was created:

```text
China mainland: https://dashscope.aliyuncs.com/compatible-mode/v1
International:   https://dashscope-intl.aliyuncs.com/compatible-mode/v1
```

The examples below use the China mainland endpoint. Replace
`$QWEN_API_BASE` with the international endpoint if necessary:

```powershell
$QWEN_MODEL = "qwen3-vl-flash"
$QWEN_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
```

### 3. Check the clean evaluation subset

The VLM tracks should use the reviewed clean subset:

```powershell
Get-Content outputs\evaluations\clean_subset.json
```

The current runner reads:

```text
outputs/reasoning/prompts.jsonl
outputs/evaluations/clean_subset.json
data/images/
```

If geometry or prompt code was changed, regenerate these files first:

```powershell
python scripts/10_extract_geometry.py --overwrite
python scripts/11_build_reasoning_prompts.py --overwrite
```

### 4. Run a three-question API test

Run a small test before spending API requests on the complete benchmark:

```powershell
python scripts/16_run_vlm_inference.py `
    --track pure_vlm `
    --provider openai_compatible `
    --model $QWEN_MODEL `
    --api-base $QWEN_API_BASE `
    --api-key-env DASHSCOPE_API_KEY `
    --limit 3 `
    --overwrite
```

The command prints one line after each completed question:

```text
[1/3] scene_0001_view_01_q001 closer_farther -> laptop correct=True latency=8.46s
```

Check the test output:

```powershell
Get-Content outputs\inference\pure_vlm.jsonl
Get-Content outputs\evaluations\pure_vlm_summary.json
```

Confirm that:

- `"model"` is `"qwen3-vl-flash"`;
- `"error"` is `null`;
- `"raw_response"` contains the model response;
- the summary contains the expected number of questions.

The `--limit 3` test is a connectivity and data-flow test, not the final
benchmark result.

### 5. Run all three tracks

Run all tracks with the same model and the same clean question set:

```powershell
python scripts/16_run_vlm_inference.py `
    --track pure_vlm `
    --provider openai_compatible `
    --model $QWEN_MODEL `
    --api-base $QWEN_API_BASE `
    --api-key-env DASHSCOPE_API_KEY `
    --overwrite

python scripts/16_run_vlm_inference.py `
    --track geometry_only `
    --provider openai_compatible `
    --model $QWEN_MODEL `
    --api-base $QWEN_API_BASE `
    --api-key-env DASHSCOPE_API_KEY `
    --overwrite

python scripts/16_run_vlm_inference.py `
    --track geovlm `
    --provider openai_compatible `
    --model $QWEN_MODEL `
    --api-base $QWEN_API_BASE `
    --api-key-env DASHSCOPE_API_KEY `
    --overwrite
```

The three tracks mean:

```text
pure_vlm:      image + question
geometry_only: object-level geometry + question
geovlm:        image + object-level geometry + question
```

Progress is printed after every question by default. Add `--no-progress` only
when running in a silent or scripted environment.

Generated result files:

```text
outputs/inference/pure_vlm.jsonl
outputs/inference/geometry_only.jsonl
outputs/inference/geovlm.jsonl

outputs/evaluations/pure_vlm_summary.json
outputs/evaluations/geometry_only_summary.json
outputs/evaluations/geovlm_summary.json
```

Each JSONL line stores the question ID, image, track, model, raw response,
normalized prediction, correctness, latency, and provider error information.

### 6. Generate the comparison report

After all three tracks finish, compare them without making additional API calls:

```powershell
python scripts/17_compare_track_results.py --overwrite
```

Outputs:

```text
outputs/evaluations/track_comparison.json
outputs/evaluations/track_disagreements.csv
```

Use `track_comparison.json` for aggregate and per-question-type metrics. Use
`track_disagreements.csv` to inspect questions where the tracks disagree.

### 7. Re-run only geometry-dependent tracks after a geometry fix

If `scripts/10_extract_geometry.py` or
`scripts/11_build_reasoning_prompts.py` changes, regenerate geometry and prompts,
then rerun only:

```powershell
python scripts/16_run_vlm_inference.py --track geometry_only --provider openai_compatible --model $QWEN_MODEL --api-base $QWEN_API_BASE --api-key-env DASHSCOPE_API_KEY --overwrite
python scripts/16_run_vlm_inference.py --track geovlm --provider openai_compatible --model $QWEN_MODEL --api-base $QWEN_API_BASE --api-key-env DASHSCOPE_API_KEY --overwrite
python scripts/17_compare_track_results.py --overwrite
```

`pure_vlm` does not use geometry prompts, so it can remain as the fixed baseline
unless the benchmark questions or images change.

## Limitations

This project uses existing perception models and does not guarantee perfect geometry.

- YOLO may miss or misclassify objects.
- SAM2 masks are prompted by detection boxes, so segmentation can inherit detection errors.
- Depth Anything V2 provides monocular relative depth, not metric 3D distance. For `closer_farther`, GeoVLM uses a near-surface `closeness_score` where higher means closer.
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
