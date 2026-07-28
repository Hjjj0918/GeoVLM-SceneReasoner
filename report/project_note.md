# GeoVLM-SceneReasoner Project Note

## Project

GeoVLM-SceneReasoner: Geometry-Aware Visual Reasoning for Vision-Language Models

## Research Question

Are vision-language models reliable when answering spatial and physical reasoning questions about real images? Can explicit object-level geometry from detection, segmentation, and depth estimation improve answer accuracy and consistency?

## Core Hypothesis

Pure VLM answers are unstable on questions about relative position, relative depth, occlusion, and physical size. A geometry-aware reasoning prompt that includes object boxes, masks, relative depth, and object-level spatial summaries should improve spatial reasoning performance.

## MVP Pipeline

```text
image
-> object detection
-> SAM2 segmentation
-> Depth Anything V2 depth estimation
-> object-level geometry
-> pure VLM / geometry-only LLM / GeoVLM comparison
```

## Benchmark Question Types

- `closer_farther`
- `left_right`
- `front_back`
- `occlusion`
- `support_relation`
- `physical_size`

## Evaluation Variants

- Pure VLM: image and question only.
- Geometry-only LLM: structured object geometry and question only.
- GeoVLM: image, structured object geometry, and question.

## References

- EasyARC: Evaluating Vision Language Models on True Visual Reasoning, arXiv:2506.11595.
- VisuLogic: A Benchmark for Evaluating Visual Reasoning in Multi-modal Large Language Models, arXiv:2504.15279.
- VLM2-Bench dataset, Hugging Face: `Sterzhang/vlm2-bench`.
