# Data Collection

GeoVLM-SceneReasoner can use self-captured images or public datasets. For a first experiment, self-captured real images are acceptable as long as the benchmark questions are reviewed carefully.

## Local Images

Place images in:

```text
data/images/
```

Recommended naming for multi-view scenes:

```text
scene_0001_view_00.jpg
scene_0001_view_01.jpg
scene_0001_view_02.jpg
```

Use:

```powershell
python scripts/02_rename_images.py
```

to rename local images into a stable sequence.

## Capture Guidelines

For small indoor experiments:

- Keep objects fixed if the goal is to study camera viewpoint changes.
- Capture 30-50 images for an MVP.
- Include 3-6 common objects per scene.
- Avoid severe motion blur, extreme darkness, and heavy reflections.
- Record only non-private scenes if the repository may be published.

Changing yaw, pitch, and roll while keeping objects fixed is valid for a multi-view same-scene benchmark. The answers may still need review because some objects can become partially hidden or ambiguous from certain angles.

## Question Files

The main benchmark file is:

```text
data/questions.json
```

For multi-view captures, start from:

```text
data/question_templates.example.json
```

Then generate a draft:

```powershell
python scripts/03_scaffold_questions.py
```

This writes:

```text
data/questions.draft.json
```

Review the draft before using it as the final benchmark. In particular, check whether each answer is still valid for each view.

## Public Release Notes

Do not publish private images, generated masks, depth maps, API keys, or model weights unless their licenses and privacy status are clear. The repository is configured to ignore:

```text
data/images/
outputs/
*.pt
*.pth
*.onnx
*.safetensors
.env
```

