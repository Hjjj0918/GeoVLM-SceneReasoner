# Evaluation

GeoVLM-SceneReasoner is designed to compare image-only reasoning with geometry-aware reasoning.

## Tracks

| Track | Input | Purpose |
|---|---|---|
| Pure VLM | image + question | Measures the VLM's image-only spatial reasoning. |
| Geometry-only LLM | object-level geometry + question | Measures whether explicit geometry is enough for the question. |
| GeoVLM | image + object-level geometry + question | Measures whether geometry helps a VLM answer more reliably. |

The current repository implements prompt generation and a geometry-only rule baseline. VLM inference is planned.

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

