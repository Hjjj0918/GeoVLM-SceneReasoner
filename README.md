# GeoVLM-SceneReasoner

**GeoVLM-SceneReasoner: Geometry-Aware Visual Reasoning for Vision-Language Models**

This is the non-robotics version of CalibVLM. The project asks whether VLMs are reliable on real-image spatial reasoning, and whether explicit object-level geometry can improve their answers.

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

## Stage 1

Current stage: project identity, dataset layout, benchmark question schema, and validation scripts.

Not included yet:

- YOLO detection.
- SAM2 segmentation.
- Depth Anything V2.
- VLM API or local VLM inference.
- Automatic evaluation tables.

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
  questions.example.json
outputs/
  detections/
  masks/
  depth/
  geometry/
  reasoning/
  evaluations/
  visualizations/
scripts/
  00_check_dataset.py
  01_validate_questions.py
report/
  project_note.md
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

You do not need a chessboard or camera calibration for the first version. The project initially uses image-space relations and relative depth:

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
```

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

Your RTX 4060 8GB is enough if the pipeline is staged:

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
