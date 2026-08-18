"""Compact residual network shared by training and standalone inference."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.1 * self.body(x)


class FastRestoreNet(nn.Module):
    """Denoise and super-resolve a grayscale image by exactly 2x.

    A bicubic skip connection provides a conservative OOD fallback. The learned
    branch only predicts the missing residual, which keeps this small model fast.
    """

    def __init__(self, channels: int = 32, blocks: int = 4, scale: int = 2) -> None:
        super().__init__()
        if scale != 2:
            raise ValueError("FastRestoreNet currently supports scale=2 only")
        self.scale = scale
        self.head = nn.Conv2d(1, channels, 3, padding=1)
        self.blocks = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
        self.body = nn.Conv2d(channels, channels, 3, padding=1)
        self.up = nn.Sequential(
            nn.Conv2d(channels, channels * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, 0.0, 1.0)
        base = F.interpolate(
            x, scale_factor=self.scale, mode="bicubic", align_corners=False
        )
        features = self.head(x)
        residual = self.up(self.body(self.blocks(features)) + features)
        return torch.clamp(base + 0.1 * residual, 0.0, 1.0)


def build_model(config: dict | None = None) -> FastRestoreNet:
    config = config or {}
    return FastRestoreNet(
        channels=int(config.get("channels", 32)),
        blocks=int(config.get("blocks", 4)),
        scale=int(config.get("scale", 2)),
    )
