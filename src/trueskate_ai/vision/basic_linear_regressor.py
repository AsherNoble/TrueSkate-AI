"""Spatial-temporal MVP-2 regressor for one finite-slope straight drag."""
from __future__ import annotations

import torch
from torch import nn

from trueskate_ai.data.gesture_sampling import BASIC_LINEAR_MAX_S, BASIC_LINEAR_MIN_S


class BasicLinearRegressor(nn.Module):
    """Predict ``[x0,y0,x1,y1,duration]`` while retaining spatial evidence."""

    def __init__(self, base_channels: int = 16):
        super().__init__()
        c = base_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(6, c, 5, stride=2, padding=2), nn.GroupNorm(1, c), nn.SiLU(),
            nn.Conv2d(c, c * 2, 3, stride=2, padding=1), nn.GroupNorm(2, c * 2), nn.SiLU(),
            nn.Conv2d(c * 2, c * 4, 3, padding=1), nn.GroupNorm(4, c * 4), nn.SiLU(),
        )
        self.spatial_score = nn.Conv2d(c * 4, 1, 1)
        self.duration_head = nn.Sequential(
            nn.Conv1d(2, c, 3, padding=1), nn.SiLU(), nn.Conv1d(c, c, 3, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool1d(8), nn.Flatten(), nn.Linear(c * 8, c * 2), nn.SiLU(), nn.Linear(c * 2, 1),
        )

    @staticmethod
    def _read_xy(scores: torch.Tensor, time_prior: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, height, width = scores.shape
        logits = scores.flatten(1) / .15 + time_prior.view(1, steps, 1, 1).expand_as(scores).flatten(1)
        attention = torch.softmax(logits, dim=1).reshape_as(scores)
        xa = torch.linspace(0., 1., width, dtype=scores.dtype, device=scores.device)
        ya = torch.linspace(0., 1., height, dtype=scores.dtype, device=scores.device)
        return ((attention * xa.view(1, 1, 1, width)).sum((1, 2, 3)),
                (attention * ya.view(1, 1, height, 1)).sum((1, 2, 3)))

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError("frames must have shape [batch,time,3,height,width]")
        batch, steps = frames.shape[:2]
        reference = frames[:, :max(1, round(steps * .22))].mean(dim=1, keepdim=True)
        encoded = self.encoder(torch.cat((frames, torch.abs(frames - reference)), dim=2).flatten(0, 1))
        h, w = encoded.shape[-2:]
        scores = self.spatial_score(encoded).reshape(batch, steps, h, w)
        # Endpoint priors choose the first and last half of the rendered trace;
        # the learned spatial scores still select actual onset/offset evidence.
        time = torch.linspace(0., 1., steps, dtype=frames.dtype, device=frames.device)
        x0, y0 = self._read_xy(scores, -4.0 * time)
        x1, y1 = self._read_xy(scores, 4.0 * time)
        series = torch.stack((scores.amax((2, 3)), scores.mean((2, 3))), dim=1)
        duration = BASIC_LINEAR_MIN_S + torch.sigmoid(self.duration_head(series)) * (BASIC_LINEAR_MAX_S - BASIC_LINEAR_MIN_S)
        return torch.cat((x0[:, None], y0[:, None], x1[:, None], y1[:, None], duration), dim=1)

    @torch.no_grad()
    def predict_linear(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        value = self(frames)
        return {"x0": value[:, 0], "y0": value[:, 1], "x1": value[:, 2], "y1": value[:, 3], "dur": value[:, 4]}
