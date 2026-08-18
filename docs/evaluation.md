# Evaluation

GeoVLM-SceneReasoner is designed to compare image-only reasoning with geometry-aware reasoning.

## Tracks

| Track | Input | Purpose |
|---|---|---|
| Pure VLM | image + question | Measures the VLM's image-only spatial reasoning. |
| Geometry-only LLM | object-level geometry + question | Measures whether explicit geometry is enough for the question. |
| GeoVLM | image + object-level geometry + question | Measures whether geometry helps a VLM answer more reliably. |

The current repository implements prompt generation, a geometry-only rule baseline, and a model-agnostic VLM inference runner.

## Metrics

The geometry baseline reports:

- `end_to_end_accuracy`: correct answers divided by total questions.
- `answered_accuracy`: correct answers divided by non-unknown answers.
- `coverage`: non-unknown answers divided by total questions.
- `unknown_rate`: unknown answers divided by total questions.
- `error_reasons`: grouped reasons for unknown or failed answers.

End-to-end accuracy includes upstream perception failures. Answered accuracy only evaluates questions where the baseline produced an answer.

## Failure Splits

Run:

```powershell
python scripts/14_build_evaluation_splits.py --overwrite
```

This writes:

```text
outputs/evaluations/evaluation_splits.json
outputs/evaluations/evaluation_review.csv
```

The automatic split is conservative:

- `pipeline_failure`: the baseline record has `error_reason=missing_target_objects`.
- `geometry_available_candidates`: target objects were available to the geometry baseline.
- `manual_review_pending`: candidate records that still require mask and depth review.

`geometry_available_candidates` does not mean the sample is clean. It means the target labels exist in the object-level geometry file. The segmentation mask and relative depth estimate may still be wrong.

The review CSV includes manual review columns:

```text
mask_ok
depth_ok
question_valid
final_split
review_notes
```

## Manual Review

For each candidate row in `evaluation_review.csv`, check:

- the original image in `data/images/`
- the mask visualization in `outputs/visualizations/masks/`
- the depth preview in `outputs/depth/`
- the object-level geometry JSON in `outputs/geometry/`

A final clean subset should include only records where:

- target objects are correctly detected
- masks cover the intended objects
- depth ordering is plausible for the question
- the question answer is valid for that camera view

This review step is necessary before making claims about VLM reasoning accuracy.

## Clean Subset

After manual review, run:

```powershell
python scripts/15_build_clean_subset.py --overwrite
```

This reads:

```text
outputs/evaluations/evaluation_review.csv
data/questions.json
```

and writes:

```text
outputs/evaluations/clean_subset.json
outputs/evaluations/clean_subset_questions.json
```

A row is included only when:

```text
automatic_split == geometry_available_candidates
mask_ok == yes
depth_ok == yes
question_valid == yes
```

Accepted yes-like values include `yes`, `y`, `true`, `1`, `ok`, `pass`, and `passed`. If `final_split` is filled with `reject`, `exclude`, `bad`, or `no`, the row is excluded even if the other fields are positive.

If no rows have been manually marked yet, the clean subset will contain zero questions. That is expected and means the benchmark is still pending human review.

## Clean Geometry Baseline

Run the geometry-only rule baseline on the reviewed subset:

```powershell
python scripts/12_run_geometry_rule_baseline.py --question-ids outputs/evaluations/clean_subset.json --output outputs/reasoning/geometry_rule_baseline_clean.jsonl --summary outputs/evaluations/geometry_rule_baseline_clean_summary.json --overwrite
```

This writes:

```text
outputs/reasoning/geometry_rule_baseline_clean.jsonl
outputs/evaluations/geometry_rule_baseline_clean_summary.json
```

Use this clean-only summary when comparing against Pure VLM and GeoVLM results. The full 111-question summary is still useful for end-to-end pipeline analysis, but it includes missing-target perception failures.

## VLM Inference

The model-agnostic runner is:

```text
scripts/16_run_vlm_inference.py
```

It supports:

```text
pure_vlm
geometry_only
geovlm
```

All tracks are filtered by the question IDs in:

```text
outputs/evaluations/clean_subset.json
```

Run a no-network smoke test:

```powershell
python scripts/16_run_vlm_inference.py --track pure_vlm --provider mock --mock-response laptop --limit 3 --overwrite
```

For actual inference, use an OpenAI-compatible endpoint:

```powershell
$env:OPENAI_API_KEY = "your-key"
python scripts/16_run_vlm_inference.py --track pure_vlm --provider openai_compatible --model your-vision-model --api-base https://your-endpoint/v1 --overwrite
```

The three tracks differ only in their inputs:

```text
Pure VLM: image + pure_vlm_prompt
Geometry-only LLM: geometry_llm_prompt
GeoVLM: image + geovlm_prompt
```

The runner saves both `raw_response` and normalized `prediction`, so model wording can be audited after evaluation. Mock results are for pipeline testing only and must not be reported as VLM accuracy.
