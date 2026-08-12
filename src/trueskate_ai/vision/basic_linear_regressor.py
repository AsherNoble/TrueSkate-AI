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
            # MVP-2 must distinguish both endpoints.  At 128px input width a
            # stride-four map has only 32 x-cells (one cell is ~0.031 in model
            # coordinates), already the whole recovery tolerance.  Keep the
            # first layer dense and downsample only once so soft-argmax has
            # genuine sub-tolerance endpoint evidence.
            nn.Conv2d(6, c, 5, stride=1, padding=2), nn.GroupNorm(1, c), nn.SiLU(),
            nn.Conv2d(c, c * 2, 3, stride=2, padding=1), nn.GroupNorm(2, c * 2), nn.SiLU(),
            nn.Conv2d(c * 2, c * 4, 3, padding=1), nn.GroupNorm(4, c * 4), nn.SiLU(),
        )
        # Start and end are not interchangeable visual concepts.  A shared map
        # plus an externally imposed time prior can collapse to the trace's
        # midpoint; give each endpoint its own learned spatial evidence.
        self.start_score = nn.Conv2d(c * 4, 1, 1)
        self.end_score = nn.Conv2d(c * 4, 1, 1)
        self.duration_head = nn.Sequential(
            nn.Conv1d(2, c, 3, padding=1), nn.SiLU(), nn.Conv1d(c, c, 3, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool1d(8), nn.Flatten(), nn.Linear(c * 8, c * 2), nn.SiLU(), nn.Linear(c * 2, 1),
        )

    @staticmethod
    def _read_xy(scores: torch.Tensor, time_prior: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, height, width = scores.shape
        if time_prior.ndim == 1:
            time_prior = time_prior.unsqueeze(0)
        if time_prior.shape not in {(1, steps), (batch, steps)}:
            raise ValueError("time_prior must have shape [time] or [batch,time]")
        logits = scores.flatten(1) / .15 + time_prior[:, :, None, None].expand_as(scores).flatten(1)
        attention = torch.softmax(logits, dim=1).reshape_as(scores)
        xa = torch.linspace(0., 1., width, dtype=scores.dtype, device=scores.device)
        ya = torch.linspace(0., 1., height, dtype=scores.dtype, device=scores.device)
        return ((attention * xa.view(1, 1, 1, width)).sum((1, 2, 3)),
                (attention * ya.view(1, 1, height, 1)).sum((1, 2, 3)))

    @staticmethod
    def _refine_from_orange(frames: torch.Tensor, reference: torch.Tensor, xy: torch.Tensor,
                            centre_time: torch.Tensor, *, spatial_sigma: float = .085,
                            time_sigma: float = .11) -> tuple[torch.Tensor, torch.Tensor]:
        """Locally refine an endpoint from the rendered orange finger glow.

        The learned heads provide a robust global proposal; restricting this
        deterministic cue to its proposal neighbourhood avoids unrelated orange
        board/game elements that make a global colour detector unreliable.
        """
        batch, steps, _channels, height, width = frames.shape
        red, green, blue = frames.unbind(dim=2)
        motion = torch.abs(frames - reference).mean(dim=2)
        orange = ((red - green + .12).relu() * (green - blue + .12).relu()
                  * (red - .20).relu() * motion)
        time = torch.linspace(0., 1., steps, dtype=frames.dtype, device=frames.device)
        xa = torch.linspace(0., 1., width, dtype=frames.dtype, device=frames.device)
        ya = torch.linspace(0., 1., height, dtype=frames.dtype, device=frames.device)
        time_weight = torch.exp(-.5 * ((time[None, :, None, None] - centre_time[:, None, None, None])
                                       / time_sigma).square())
        x_weight = torch.exp(-.5 * ((xa[None, None, None, :] - xy[:, 0, None, None, None])
                                    / spatial_sigma).square())
        y_weight = torch.exp(-.5 * ((ya[None, None, :, None] - xy[:, 1, None, None, None])
                                    / spatial_sigma).square())
        weight = orange * time_weight * x_weight * y_weight
        normalizer = weight.sum((1, 2, 3)).clamp_min(1e-8)
        refined_x = (weight * xa.view(1, 1, 1, width)).sum((1, 2, 3)) / normalizer
        refined_y = (weight * ya.view(1, 1, height, 1)).sum((1, 2, 3)) / normalizer
        # Keep the neural proposal dominant when the local orange evidence is
        # weak (e.g. motion blur), while correcting the several-pixel tail.
        confidence = (normalizer / normalizer.detach().median().clamp_min(1e-8)).clamp(0., 1.)
        blend = .55 * confidence
        return xy[:, 0] * (1 - blend) + refined_x * blend, xy[:, 1] * (1 - blend) + refined_y * blend

    def forward_with_scores(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the gesture plus raw start/end score maps for inspection."""
        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError("frames must have shape [batch,time,3,height,width]")
        batch, steps = frames.shape[:2]
        reference = frames[:, :max(1, round(steps * .22))].mean(dim=1, keepdim=True)
        difference = torch.abs(frames - reference)
        encoded = self.encoder(torch.cat((frames, difference), dim=2).flatten(0, 1))
        h, w = encoded.shape[-2:]
        start_scores = self.start_score(encoded).reshape(batch, steps, h, w)
        end_scores = self.end_score(encoded).reshape(batch, steps, h, w)
        # The pre-touch reference is the leading ~22% of every aligned clip.
        # Suppress that known no-gesture region, then make only a gentle learned
        # temporal preference.  Strong priors previously let background pixels in
        # the pre-touch prefix dominate early-endpoint attention.
        time = torch.linspace(0., 1., steps, dtype=frames.dtype, device=frames.device)
        active = torch.where(time < .18, torch.full_like(time, -12.0), torch.zeros_like(time))
        evidence = torch.maximum(start_scores, end_scores)
        series = torch.stack((evidence.amax((2, 3)), evidence.mean((2, 3))), dim=1)
        duration = BASIC_LINEAR_MIN_S + torch.sigmoid(self.duration_head(series)) * (BASIC_LINEAR_MAX_S - BASIC_LINEAR_MIN_S)
        # The trace is visible from command onset to liftoff, while later frames
        # contain only board/camera motion.  The old ``+time`` end prior was
        # therefore systematically attracted to post-liftoff pixels.  The
        # direct clip window is -0.5s..~1.77s, so onset is near 0.24 and an
        # estimated liftoff is onset + duration / 2.27.  Broad Gaussian priors
        # retain tolerance for calibration/frame-rate variation while guiding
        # each endpoint head to the causally relevant trace interval.
        onset = time.new_full((batch,), 0.24)
        liftoff = (onset + duration[:, 0] / 2.27).clamp(max=0.88)
        start_prior = active - ((time[None, :] - onset[:, None]) / 0.13).square()
        end_prior = active - ((time[None, :] - liftoff[:, None]) / 0.15).square()
        x0, y0 = self._read_xy(start_scores, start_prior)
        x1, y1 = self._read_xy(end_scores, end_prior)
        x0, y0 = self._refine_from_orange(frames, reference, torch.stack((x0, y0), dim=1), onset)
        x1, y1 = self._refine_from_orange(frames, reference, torch.stack((x1, y1), dim=1), liftoff)
        prediction = torch.cat((x0[:, None], y0[:, None], x1[:, None], y1[:, None], duration), dim=1)
        return prediction, start_scores, end_scores

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.forward_with_scores(frames)[0]

    @torch.no_grad()
    def predict_linear(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        value = self(frames)
        return {"x0": value[:, 0], "y0": value[:, 1], "x1": value[:, 2], "y1": value[:, 3], "dur": value[:, 4]}
