#!/usr/bin/env python3
"""Reproduce training for the KLA grayscale restoration model."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from model import build_model


class PairedNpyDataset(Dataset):
    def __init__(
        self,
        noisy_dir: Path,
        gt_dir: Path,
        names: list[str],
        crop_size: int,
        training: bool,
    ) -> None:
        self.noisy_dir = noisy_dir
        self.gt_dir = gt_dir
        self.names = names
        self.crop_size = crop_size
        self.training = training

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        name = self.names[index]
        low = np.load(self.noisy_dir / name, allow_pickle=False).astype(np.float32)
        gt = np.load(self.gt_dir / name, allow_pickle=False).astype(np.float32)
        low = np.nan_to_num(low, nan=0.0, posinf=1.0, neginf=0.0)
        gt = np.nan_to_num(gt, nan=0.0, posinf=1.0, neginf=0.0)
        if low.ndim != 2 or gt.ndim != 2 or gt.shape != (low.shape[0] * 2, low.shape[1] * 2):
            raise ValueError(f"Invalid pair {name}: low={low.shape}, gt={gt.shape}")

        if self.training:
            hr_crop = min(self.crop_size, gt.shape[0], gt.shape[1])
            hr_crop -= hr_crop % 2
            lr_crop = hr_crop // 2
            top = random.randint(0, low.shape[0] - lr_crop)
            left = random.randint(0, low.shape[1] - lr_crop)
            low = low[top : top + lr_crop, left : left + lr_crop]
            gt = gt[top * 2 : top * 2 + hr_crop, left * 2 : left * 2 + hr_crop]
            if random.random() < 0.5:
                low, gt = low[:, ::-1], gt[:, ::-1]
            if random.random() < 0.5:
                low, gt = low[::-1, :], gt[::-1, :]
            if random.random() < 0.5:
                low, gt = low.T, gt.T

        low = np.ascontiguousarray(np.clip(low, 0.0, 1.0))
        gt = np.ascontiguousarray(np.clip(gt, 0.0, 1.0))
        return torch.from_numpy(low[None]), torch.from_numpy(gt[None])


def restoration_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff = pred - target
    charbonnier = torch.sqrt(diff * diff + 1e-6).mean()
    pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
    pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
    gt_dx = target[..., :, 1:] - target[..., :, :-1]
    gt_dy = target[..., 1:, :] - target[..., :-1, :]
    edge = F.l1_loss(pred_dx, gt_dx) + F.l1_loss(pred_dy, gt_dy)
    return charbonnier + 0.1 * edge


@torch.inference_mode()
def validate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    squared_error = 0.0
    pixels = 0
    for low, gt in loader:
        low, gt = low.to(device), gt.to(device)
        pred = model(low)
        squared_error += F.mse_loss(pred, gt, reduction="sum").item()
        pixels += gt.numel()
    mse = max(squared_error / pixels, 1e-12)
    return -10.0 * math.log10(mse)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Folder containing GT/ and NoisyLR/")
    parser.add_argument("--output", type=Path, default=Path("weights/restoration_model.pt"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--crop-size", type=int, default=96, help="Even HR crop size")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--val-count", type=int, default=160)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    device_name = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)

    noisy_dir, gt_dir = args.data_root / "NoisyLR", args.data_root / "GT"
    noisy_names = {p.name for p in noisy_dir.glob("*.npy")}
    gt_names = {p.name for p in gt_dir.glob("*.npy")}
    names = sorted(noisy_names & gt_names)
    if not names:
        raise FileNotFoundError(f"No paired .npy files found in {noisy_dir} and {gt_dir}")
    missing = (noisy_names ^ gt_names)
    if missing:
        raise ValueError(f"Unpaired files found (first 10): {sorted(missing)[:10]}")

    rng = random.Random(args.seed)
    rng.shuffle(names)
    val_count = min(args.val_count, max(1, len(names) // 10))
    val_names, train_names = names[:val_count], names[val_count:]
    train_data = PairedNpyDataset(noisy_dir, gt_dir, train_names, args.crop_size, True)
    val_data = PairedNpyDataset(noisy_dir, gt_dir, val_names, args.crop_size, False)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=device.type == "cuda", generator=generator,
    )
    val_loader = DataLoader(val_data, batch_size=1, num_workers=args.workers)

    config = {"channels": 32, "blocks": 4, "scale": 2}
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_psnr = float("-inf")
    started = time.time()

    print(json.dumps({"device": str(device), "train_pairs": len(train_names), "val_pairs": len(val_names), "config": config}))
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for low, gt in train_loader:
            low = low.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                pred = model(low)
                loss = restoration_loss(pred, gt)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
        scheduler.step()
        psnr = validate(model, val_loader, device)
        print(f"epoch={epoch:02d}/{args.epochs} loss={running / len(train_loader):.6f} val_psnr={psnr:.3f}dB")
        if psnr > best_psnr:
            best_psnr = psnr
            torch.save(
                {
                    "format_version": 1,
                    "model_config": config,
                    "model_state_dict": model.state_dict(),
                    "scale": 2,
                    "input_clip": [0.0, 1.0],
                    "best_val_psnr_db": best_psnr,
                    "epoch": epoch,
                    "seed": args.seed,
                },
                args.output,
            )
    print(f"saved={args.output} best_val_psnr={best_psnr:.3f}dB elapsed_s={time.time() - started:.1f}")


if __name__ == "__main__":
    main()
