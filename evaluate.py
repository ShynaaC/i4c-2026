#!/usr/bin/env python3
"""Standalone KLA inference: restore every image in an input directory."""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model import build_model


SUPPORTED = {".npy", ".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restore all grayscale test images at 2x resolution. Output names match input names."
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing test .npy or image files")
    parser.add_argument("output_dir", type=Path, help="Directory for restored outputs")
    parser.add_argument(
        "--weights", type=Path,
        default=Path(__file__).resolve().parent / "weights" / "restoration_model.pt",
        help="Checkpoint path (default: weights/restoration_model.pt beside this script)",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--no-fp16", action="store_true", help="Disable CUDA FP16 inference")
    return parser.parse_args()


def read_grayscale(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False)
    else:
        array = np.asarray(Image.open(path).convert("F"), dtype=np.float32)
        if float(np.nanmax(array)) > 1.0:
            array /= 255.0 if float(np.nanmax(array)) <= 255.0 else 65535.0
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 3 and 1 in array.shape:
        array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image, got shape {array.shape} from {path}")
    return np.ascontiguousarray(np.clip(np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0))


def write_output(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.clip(array, 0.0, 1.0).astype(np.float32, copy=False)
    if path.suffix.lower() == ".npy":
        np.save(path, array, allow_pickle=False)
    elif path.suffix.lower() in {".tif", ".tiff"}:
        Image.fromarray(array, mode="F").save(path)
    else:
        Image.fromarray(np.rint(array * 255.0).astype(np.uint8), mode="L").save(path)


def main() -> None:
    args = parse_args()
    if not args.input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {args.input_dir}")
    if not args.weights.is_file():
        raise FileNotFoundError(f"Model weights not found: {args.weights}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    files = sorted(p for p in args.input_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)
    if not files:
        raise FileNotFoundError(f"No supported images found under {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    device_name = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
    model = build_model(checkpoint.get("model_config", {}))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval().to(device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        model = model.to(memory_format=torch.channels_last)
    use_fp16 = device.type == "cuda" and not args.no_fp16

    groups: dict[tuple[int, int], list[tuple[Path, np.ndarray]]] = defaultdict(list)
    for path in files:
        image = read_grayscale(path)
        groups[image.shape].append((path, image))

    started = time.perf_counter()
    with torch.inference_mode():
        for items in groups.values():
            for offset in range(0, len(items), args.batch_size):
                batch_items = items[offset : offset + args.batch_size]
                batch = np.stack([image for _, image in batch_items])[:, None]
                tensor = torch.from_numpy(batch).to(device, non_blocking=True)
                if device.type == "cuda":
                    tensor = tensor.contiguous(memory_format=torch.channels_last)
                with torch.amp.autocast("cuda", enabled=use_fp16):
                    restored = model(tensor)
                outputs = restored.float().cpu().numpy()[:, 0]
                for (input_path, _), output in zip(batch_items, outputs):
                    relative = input_path.relative_to(args.input_dir)
                    write_output(args.output_dir / relative, output)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    print(f"Restored {len(files)} images to {args.output_dir} on {device} in {elapsed:.3f}s ({elapsed / len(files):.4f}s/image)")


if __name__ == "__main__":
    main()
