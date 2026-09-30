"""Frozen-checkpoint interventions on how the end point is read (M1-ENDPROBE).

``BasicLinearRegressor`` reads the end point with ONE softmax over every pixel
of every frame, weighted by a Gaussian time prior centred on a liftoff estimated
from the *predicted* duration (``end_onset + duration / CLIP_WINDOW_S``,
sigma 0.15 of the clip).  Frames before liftoff show a trail that has not yet
reached its tip, so weight on them pulls the soft-argmax short along the path.

This module holds the score maps fixed and changes only that read-out, so a
change in recovery isolates the decoder from the features.  Variants that use
the target duration are label-informed diagnostics, never inference-feasible.
"""
from __future__ import annotations

import torch

from trueskate_ai.model1.linear.regressor import CLIP_WINDOW_S, BasicLinearRegressor

END_PRIOR_SIGMA = .15
LIFTOFF_CLAMP = .88
ACTIVE_FROM = .18
SPATIAL_TEMPERATURE = .15

# name -> (liftoff source, time-prior sigma, or None for a single-frame read)
VARIANTS: dict[str, tuple[str, float | None]] = {
    "baseline": ("predicted", END_PRIOR_SIGMA),
    "sigma_0.075": ("predicted", .075),
    "sigma_0.05": ("predicted", .05),
    "oracle_liftoff": ("oracle", END_PRIOR_SIGMA),
    "oracle_liftoff_sigma_0.05": ("oracle", .05),
    "oracle_frame": ("oracle", None),
}


def capture_end_scores(model: BasicLinearRegressor, frames: torch.Tensor
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the model once; return its prediction and the end score maps it read."""
    captured: list[torch.Tensor] = []
    original = BasicLinearRegressor._read_xy

    def recording(scores, time_prior):
        captured.append(scores)
        return original(scores, time_prior)

    model._read_xy = recording  # instance override of the staticmethod
    try:
        prediction = model(frames)
    finally:
        del model._read_xy
    if len(captured) != 2:
        raise RuntimeError(f"expected start and end reads, saw {len(captured)}")
    return prediction, captured[1]


def end_prior(steps: int, liftoff: torch.Tensor, sigma: float) -> torch.Tensor:
    """The regressor's end time prior, with a chosen centre and width."""
    time = torch.linspace(0., 1., steps, dtype=liftoff.dtype, device=liftoff.device)
    active = torch.where(time < ACTIVE_FROM, torch.full_like(time, -12.0), torch.zeros_like(time))
    return active - ((time[None, :] - liftoff[:, None]) / sigma).square()


def liftoff_from_duration(duration_s: torch.Tensor, end_onset: float) -> torch.Tensor:
    return (end_onset + duration_s / CLIP_WINDOW_S).clamp(max=LIFTOFF_CLAMP)


def frame_read(scores: torch.Tensor, liftoff: torch.Tensor) -> torch.Tensor:
    """Spatial soft-argmax of the single frame nearest ``liftoff``."""
    batch, steps, height, width = scores.shape
    index = (liftoff * (steps - 1)).round().long().clamp(0, steps - 1)
    frame = scores[torch.arange(batch, device=scores.device), index]
    attention = torch.softmax(frame.flatten(1) / SPATIAL_TEMPERATURE, dim=1).reshape_as(frame)
    xa = torch.linspace(0., 1., width, dtype=scores.dtype, device=scores.device)
    ya = torch.linspace(0., 1., height, dtype=scores.dtype, device=scores.device)
    return torch.stack(((attention * xa.view(1, 1, width)).sum((1, 2)),
                        (attention * ya.view(1, height, 1)).sum((1, 2))), dim=1)


def attention_time(scores: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
    """Attention-weighted mean normalised time of an end read."""
    batch, steps = scores.shape[:2]
    logits = scores.flatten(1) / SPATIAL_TEMPERATURE + prior[:, :, None, None].expand_as(scores).flatten(1)
    weights = torch.softmax(logits, dim=1).reshape_as(scores).sum((2, 3))
    time = torch.linspace(0., 1., steps, dtype=scores.dtype, device=scores.device)
    return (weights * time[None, :]).sum(1)


def end_variants(model: BasicLinearRegressor, frames: torch.Tensor, target: torch.Tensor
                 ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Return the model prediction, each variant's end ``[B,2]``, and diagnostics."""
    if target.shape[1] != 5:
        raise ValueError("end_variants supports the two-knot [x0,y0,x1,y1,duration] layout only")
    prediction, scores = capture_end_scores(model, frames)
    steps = scores.shape[1]
    liftoffs = {
        "predicted": liftoff_from_duration(prediction[:, 4], model.end_onset),
        "oracle": liftoff_from_duration(target[:, 4].to(prediction.dtype), model.end_onset),
    }
    ends: dict[str, torch.Tensor] = {}
    for name, (source, sigma) in VARIANTS.items():
        if sigma is None:
            ends[name] = frame_read(scores, liftoffs[source])
        else:
            x, y = BasicLinearRegressor._read_xy(scores, end_prior(steps, liftoffs[source], sigma))
            ends[name] = torch.stack((x, y), dim=1)
    baseline_prior = end_prior(steps, liftoffs["predicted"], END_PRIOR_SIGMA)
    diagnostics = {
        "attention_time": attention_time(scores, baseline_prior),
        "predicted_liftoff": liftoffs["predicted"],
        "oracle_liftoff": liftoffs["oracle"],
    }
    return prediction, ends, diagnostics


def along_path_error(end: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Signed end error along the commanded direction (negative = short)."""
    direction = target[:, 2:4] - target[:, 0:2]
    unit = direction / torch.linalg.vector_norm(direction, dim=1, keepdim=True).clamp_min(1e-6)
    return ((end - target[:, 2:4]) * unit).sum(1)
