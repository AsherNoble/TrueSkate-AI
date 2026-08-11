"""Spatial-temporal regressor for the basic stationary-hold experiment.

The rendered touch is a small, local orange mark.  A conventional image encoder
ending in global average pooling is therefore the wrong architecture: it makes
the representation nearly invariant to the mark's position before the x/y head
ever sees it.  This module retains a spatial score map through to a soft-argmax
coordinate readout and uses its temporal evidence to estimate hold duration.
"""
from __future__ import annotations

import torch
from torch import nn

from trueskate_ai.vision.basic_hold_dataset import HOLD_DURATION_MAX_S, HOLD_DURATION_MIN_S


class BasicHoldRegressor(nn.Module):
    """Predict ``[x, y, duration]`` from a full clip without discarding location.

    The spatial head scores every feature-map cell in every frame.  A single
    spatiotemporal soft-argmax reads ``x`` and ``y`` from the rendered trace;
    the temporal head sees the framewise peak/mean trace score, preserving the
    onset-to-liftoff evidence needed for duration.
    """

    def __init__(self, base_channels: int = 16):
        super().__init__()
        if base_channels < 2:
            raise ValueError("base_channels must be >= 2")
        c = base_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(3, c, 5, stride=2, padding=2), nn.GroupNorm(1, c), nn.SiLU(),
            nn.Conv2d(c, c * 2, 3, stride=2, padding=1), nn.GroupNorm(2, c * 2), nn.SiLU(),
            nn.Conv2d(c * 2, c * 4, 3, stride=2, padding=1), nn.GroupNorm(4, c * 4), nn.SiLU(),
        )
        # Deliberately no spatial pooling before this head.  See module docstring.
        self.spatial_score = nn.Conv2d(c * 4, 1, 1)
        self.duration_head = nn.Sequential(
            nn.Conv1d(2, c, 3, padding=1), nn.SiLU(),
            nn.Conv1d(c, c, 3, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool1d(8), nn.Flatten(),
            nn.Linear(c * 8, c * 2), nn.SiLU(), nn.Linear(c * 2, 1),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError("frames must have shape [batch,time,3,height,width]")
        batch, steps = frames.shape[:2]
        encoded = self.encoder(frames.flatten(0, 1))
        height, width = encoded.shape[-2:]
        scores = self.spatial_score(encoded).reshape(batch, steps, height, width)

        # Soft-argmax across both time and space chooses the strongest rendered
        # trace while retaining differentiable coordinate supervision.
        attention = torch.softmax(scores.flatten(1) / 0.15, dim=1).reshape_as(scores)
        y_axis = torch.linspace(0.0, 1.0, height, dtype=frames.dtype, device=frames.device)
        x_axis = torch.linspace(0.0, 1.0, width, dtype=frames.dtype, device=frames.device)
        x = (attention * x_axis.view(1, 1, 1, width)).sum(dim=(1, 2, 3))
        y = (attention * y_axis.view(1, 1, height, 1)).sum(dim=(1, 2, 3))

        # The peak/mean score series preserves the duration evidence rather than
        # collapsing frames before the head can distinguish short from long holds.
        score_series = torch.stack((scores.amax(dim=(2, 3)), scores.mean(dim=(2, 3))), dim=1)
        raw_duration = torch.sigmoid(self.duration_head(score_series))
        duration = HOLD_DURATION_MIN_S + raw_duration * (HOLD_DURATION_MAX_S - HOLD_DURATION_MIN_S)
        return torch.cat((x.unsqueeze(1), y.unsqueeze(1), duration), dim=1)

    @torch.no_grad()
    def predict_hold(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return the public native-unit gesture interface ``{x, y, dur}``."""
        prediction = self(frames)
        return {"x": prediction[:, 0], "y": prediction[:, 1], "dur": prediction[:, 2]}
