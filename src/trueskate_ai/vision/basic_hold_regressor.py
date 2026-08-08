"""Small full-clip regressor for the basic stationary-hold experiment."""
from __future__ import annotations

import torch
from torch import nn

from trueskate_ai.vision.basic_hold_dataset import HOLD_DURATION_MAX_S, HOLD_DURATION_MIN_S


class BasicHoldRegressor(nn.Module):
    """CNN-per-frame encoder, temporal mean/max pooling, and a 3-value head."""

    def __init__(self, base_channels: int = 16):
        super().__init__()
        if base_channels < 2:
            raise ValueError("base_channels must be >= 2")
        c = base_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(3, c, 5, stride=2, padding=2), nn.GroupNorm(1, c), nn.SiLU(),
            nn.Conv2d(c, c * 2, 3, stride=2, padding=1), nn.GroupNorm(2, c * 2), nn.SiLU(),
            nn.Conv2d(c * 2, c * 4, 3, stride=2, padding=1), nn.GroupNorm(4, c * 4), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(c * 8, c * 4), nn.SiLU(), nn.Linear(c * 4, 3),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError("frames must have shape [batch,time,3,height,width]")
        batch, steps = frames.shape[:2]
        features = self.encoder(frames.flatten(0, 1)).flatten(1).unflatten(0, (batch, steps))
        pooled = torch.cat((features.mean(dim=1), features.amax(dim=1)), dim=1)
        raw = torch.sigmoid(self.head(pooled))
        duration = HOLD_DURATION_MIN_S + raw[:, 2:3] * (HOLD_DURATION_MAX_S - HOLD_DURATION_MIN_S)
        return torch.cat((raw[:, :2], duration), dim=1)

    @torch.no_grad()
    def predict_hold(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return the public native-unit gesture interface ``{x, y, dur}``.

        Values retain the batch dimension, allowing callers to score or replay a
        batch without a tensor-column convention leaking beyond this module.
        """
        prediction = self(frames)
        return {"x": prediction[:, 0], "y": prediction[:, 1], "dur": prediction[:, 2]}
