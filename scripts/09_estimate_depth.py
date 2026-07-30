"""Estimate relative depth maps with Depth Anything V2."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
REPO_ROOT = Path(__file__).resolve().parents[1]


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def collect_image_files(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        return []
    return sorted(
        (path for path in image_dir.iterdir() if is_image_file(path)),
        key=lambda path: path.name.lower(),
    )


def read_image_size(image_path: Path) -> tuple[int, int]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")
    height, width = image.shape[:2]
    return width, height


def output_paths_for_image(output_dir: Path, image_path: Path) -> tuple[Path, Path, Path]:
    depth_path = output_dir / f"{image_path.stem}.npy"
    visualization_path = output_dir / f"{image_path.stem}_preview.jpg"
    metadata_path = output_dir / f"{image_path.stem}.json"
    return depth_path, visualization_path, metadata_path


def normalize_depth_to_uint8(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        return np.zeros(depth.shape, dtype=np.uint8)

    depth_min = float(finite.min())
    depth_max = float(finite.max())
    if depth_max <= depth_min:
        return np.zeros(depth.shape, dtype=np.uint8)

    normalized = (depth - depth_min) / (depth_max - depth_min)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)


def depth_preview(depth: np.ndarray) -> np.ndarray:
    preview = normalize_depth_to_uint8(depth)
    return cv2.applyColorMap(preview, cv2.COLORMAP_INFERNO)


def finite_stat(depth: np.ndarray, stat: str) -> float:
    finite = np.asarray(depth, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    if stat == "min":
        return float(finite.min())
    if stat == "max":
        return float(finite.max())
    if stat == "mean":
        return float(finite.mean())
    raise ValueError(f"Unknown depth statistic: {stat}")


def normalized_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def build_depth_payload(
    image_path: Path,
    image_width: int,
    image_height: int,
    model_name: str,
    depth_path: Path,
    visualization_path: Path,
    depth: np.ndarray,
) -> dict[str, Any]:
    return {
        "image": image_path.name,
        "image_width": image_width,
        "image_height": image_height,
        "model": model_name,
        "depth_format": "relative_depth_float32_npy",
        "depth_path": normalized_path(depth_path),
        "visualization_path": normalized_path(visualization_path),
        "depth_shape": list(depth.shape),
        "depth_min": finite_stat(depth, "min"),
        "depth_max": finite_stat(depth, "max"),
        "depth_mean": finite_stat(depth, "mean"),
        "depth_note": "Depth Anything V2 outputs relative monocular depth, not metric distance.",
    }


def write_depth_outputs(
    image_path: Path,
    output_dir: Path,
    model_name: str,
    depth: np.ndarray,
    overwrite: bool,
) -> Path:
    depth_path, visualization_path, metadata_path = output_paths_for_image(output_dir, image_path)
    existing = [path for path in (depth_path, visualization_path, metadata_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Depth output already exists: {existing[0]}")

    image_width, image_height = read_image_size(image_path)
    depth = np.asarray(depth, dtype=np.float32)
    if depth.shape != (image_height, image_width):
        depth = cv2.resize(depth, (image_width, image_height), interpolation=cv2.INTER_CUBIC).astype(np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(depth_path, depth)
    if not cv2.imwrite(str(visualization_path), depth_preview(depth)):
        raise ValueError(f"OpenCV could not write depth preview: {visualization_path}")

    payload = build_depth_payload(
        image_path=image_path,
        image_width=image_width,
        image_height=image_height,
        model_name=model_name,
        depth_path=depth_path,
        visualization_path=visualization_path,
        depth=depth,
    )
    metadata_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata_path


def prepare_huggingface_environment(cache_dir: Path | None = None) -> None:
    target_dir = cache_dir or REPO_ROOT / "outputs" / "hf_cache"
    target_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(target_dir.resolve()))


def select_device(torch_module: object, requested_device: str | None) -> str:
    if requested_device:
        return requested_device
    cuda = getattr(getattr(torch_module, "cuda", None), "is_available", None)
    if callable(cuda) and cuda():
        return "cuda"
    return "cpu"


def load_depth_estimator(model_name: str, device: str | None = None) -> dict[str, Any]:
    prepare_huggingface_environment()
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    except ImportError as error:
        raise RuntimeError(
            "Depth dependencies are not installed. Install them with: "
            "python -m pip install transformers pillow torch"
        ) from error

    selected_device = select_device(torch, device)
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name)
    model.to(selected_device)
    model.eval()
    return {
        "torch": torch,
        "processor": processor,
        "model": model,
        "device": selected_device,
    }


def estimate_depth_array(estimator: dict[str, Any], image_path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Pillow is not installed. Install it with: python -m pip install pillow") from error

    torch = estimator["torch"]
    processor = estimator["processor"]
    model = estimator["model"]
    device = estimator["device"]

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
        resized_depth = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )

    return resized_depth.squeeze().detach().cpu().numpy().astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate GeoVLM relative depth maps with Depth Anything V2.")
    parser.add_argument("--image-dir", type=Path, default=Path("data/images"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/depth"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None, help="Optional device, for example cuda, cuda:0, or cpu.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N images.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing depth outputs.")
    parser.add_argument("--dry-run", action="store_true", help="List planned outputs without loading the model.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    images = collect_image_files(args.image_dir)
    if args.limit is not None:
        images = images[: args.limit]

    if not images:
        print(f"No image files found in {args.image_dir}")
        return 1

    print(f"Images: {len(images)}")
    print(f"Output dir: {args.output_dir}")
    print(f"Model: {args.model}")

    if args.dry_run:
        for image_path in images:
            depth_path, visualization_path, metadata_path = output_paths_for_image(args.output_dir, image_path)
            print(f"{image_path.name} -> {depth_path}, {visualization_path}, {metadata_path}")
        return 0

    try:
        estimator = load_depth_estimator(args.model, device=args.device)
        for image_path in images:
            depth = estimate_depth_array(estimator, image_path)
            metadata_path = write_depth_outputs(
                image_path=image_path,
                output_dir=args.output_dir,
                model_name=args.model,
                depth=depth,
                overwrite=args.overwrite,
            )
            print(f"Wrote {metadata_path}")
    except (FileExistsError, OSError, RuntimeError, ValueError) as error:
        print(error)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
