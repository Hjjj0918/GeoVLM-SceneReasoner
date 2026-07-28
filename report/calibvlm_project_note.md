# CalibVLM Project Note

## Project

CalibVLM: Geometry-Aware Vision-Language Perception for Embodied AI

## Core Hypothesis

Vision-language models should answer spatial questions more reliably when image input is paired with explicit geometric observations such as bounding boxes, depth estimates, camera intrinsics, and approximate 3D camera coordinates.

## Stage 1 Status

Stage 1 builds the repository skeleton, dataset conventions, and camera calibration workflow. It does not run detection, segmentation, depth estimation, or VLM reasoning yet.

## Planned Benchmark

- Pure VLM: image and question only.
- Geometry-only LLM: structured geometry and question only.
- CalibVLM: image, structured geometry, and question.

## Notes To Fill Later

- Data collection details.
- Camera calibration reprojection error.
- Detection and segmentation failure cases.
- Depth scale calibration approach.
- Spatial QA examples.
- Pure VLM vs CalibVLM comparison table.
