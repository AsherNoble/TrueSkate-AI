"""Training and evaluation helpers for MVP 2 finite-slope linear drags."""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from trueskate_ai.data.gesture_sampling import BASIC_LINEAR_MAX_S, BASIC_LINEAR_MIN_S
from trueskate_ai.vision.basic_linear_bias import AlongPathBias

RECOVERY_ENDPOINT_TOLERANCE = 0.03
RECOVERY_DURATION_TOLERANCE_S = 0.10


def target_knots(width: int) -> int:
    """Number of trajectory knots encoded in a ``[..., 2K+1]`` target vector."""
    if width < 5 or width % 2 == 0:
        raise ValueError(f"target width {width} is not 2K+1 for any K>=2")
    return (width - 1) // 2


def knot_errors(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-knot Euclidean error, shape ``[batch, K]``."""
    knots = target_knots(prediction.shape[1])
    difference = (prediction[:, :2 * knots] - target[:, :2 * knots]).reshape(-1, knots, 2)
    return torch.linalg.vector_norm(difference, dim=2)


def basic_linear_endpoint_map_loss(scores: torch.Tensor, xy: torch.Tensor,
                                   centre_time: torch.Tensor) -> torch.Tensor:
    """A gentle spatial-temporal score-map target for endpoint attention.

    This uses the *same* 0.15 temperature as ``BasicLinearRegressor._read_xy``.
    It is intentionally an optional low-weight auxiliary: an earlier dense,
    sharp classification objective overwhelmed coordinate learning rather than
    regularising the broad attention distributions behind tail errors.
    """
    if scores.ndim != 4 or xy.shape != (scores.shape[0], 2):
        raise ValueError("scores must be [batch,time,height,width] and xy [batch,2]")
    batch, steps, height, width = scores.shape
    if centre_time.shape != (batch,):
        raise ValueError("centre_time must have shape [batch]")
    time = torch.linspace(0., 1., steps, dtype=scores.dtype, device=scores.device)
    x = torch.linspace(0., 1., width, dtype=scores.dtype, device=scores.device)
    y = torch.linspace(0., 1., height, dtype=scores.dtype, device=scores.device)
    time_error = (time[None, :, None, None] - centre_time[:, None, None, None]) / .055
    x_error = (x[None, None, None, :] - xy[:, 0, None, None, None]) / .035
    y_error = (y[None, None, :, None] - xy[:, 1, None, None, None]) / .035
    target = torch.exp(-.5 * (time_error.square() + x_error.square() + y_error.square())).flatten(1)
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return -(target * F.log_softmax(scores.flatten(1) / .15, dim=1)).sum(dim=1).mean()


def basic_linear_trajectory_map_loss(scores: torch.Tensor, trajectory_xy: torch.Tensor,
                                     trajectory_mask: torch.Tensor) -> torch.Tensor:
    """Score-map CE against the manifest-known position at each active frame.

    Unlike the old endpoint auxiliary this has no guessed onset/liftoff: the
    per-frame target is computed from each sample's aligned ``frame_times`` and
    constant-velocity command.  It supervises only the active path interval.
    """
    if (scores.ndim != 4 or trajectory_xy.shape != (*scores.shape[:2], 2)
            or trajectory_mask.shape != scores.shape[:2]):
        raise ValueError("scores [B,T,H,W], trajectory_xy [B,T,2], and mask [B,T] are required")
    if not torch.any(trajectory_mask):
        raise ValueError("trajectory supervision needs at least one active frame")
    _batch, _steps, height, width = scores.shape
    x = torch.linspace(0., 1., width, dtype=scores.dtype, device=scores.device)
    y = torch.linspace(0., 1., height, dtype=scores.dtype, device=scores.device)
    x_error = (x[None, None, None, :] - trajectory_xy[:, :, 0, None, None]) / .035
    y_error = (y[None, None, :, None] - trajectory_xy[:, :, 1, None, None]) / .035
    target = torch.exp(-.5 * (x_error.square() + y_error.square())).flatten(2)
    target = target / target.sum(dim=2, keepdim=True).clamp_min(1e-12)
    per_frame = -(target * F.log_softmax(scores.flatten(2) / .15, dim=2)).sum(dim=2)
    mask = trajectory_mask.to(dtype=per_frame.dtype)
    return (per_frame * mask).sum() / mask.sum().clamp_min(1.)


def basic_linear_loss(prediction: torch.Tensor, target: torch.Tensor, *,
                      start_scores: torch.Tensor | None = None,
                      end_scores: torch.Tensor | None = None,
                      map_weight: float = 0.0, trajectory_xy: torch.Tensor | None = None,
                      trajectory_mask: torch.Tensor | None = None,
                      trajectory_weight: float = 0.0,
                      trajectory_scores: torch.Tensor | None = None) -> torch.Tensor:
    """Robust endpoint error plus duration error in matched native scales."""
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must both have shape [batch,2K+1]")
    knots = target_knots(prediction.shape[1])
    if knots != 2:
        # MVP-3 gates every knot equally, so the loss weights them equally too;
        # there is no "start is the bottleneck" asymmetry to encode here.
        positions = F.smooth_l1_loss(
            prediction[:, :2 * knots].contiguous(), target[:, :2 * knots].contiguous(), beta=0.03,
        )
        duration_scale = BASIC_LINEAR_MAX_S - BASIC_LINEAR_MIN_S
        duration = F.smooth_l1_loss(
            prediction[:, -1] / duration_scale, target[:, -1] / duration_scale, beta=0.05,
        )
        loss = positions + duration
        if trajectory_weight and trajectory_scores is not None:
            if trajectory_xy is None or trajectory_mask is None:
                raise ValueError("trajectory targets are required when trajectory_weight is positive")
            loss = loss + trajectory_weight * basic_linear_trajectory_map_loss(
                trajectory_scores, trajectory_xy, trajectory_mask,
            )
        return loss
    # Component audit of the best command-held-out checkpoint: duration passes
    # 98.7%, end 88.7%, but start only 78.7%.  Weight the start pair more
    # heavily so optimisation spends capacity on the actual recovery bottleneck.
    # torch 2.12's smooth_l1_loss viewers reject a column slice of a [B,5]
    # tensor ("spans across two contiguous subspaces"), so materialise the
    # endpoint pairs.  Values are unchanged; this only fixes the stride.
    start = F.smooth_l1_loss(prediction[:, :2].contiguous(), target[:, :2].contiguous(), beta=0.03)
    end = F.smooth_l1_loss(prediction[:, 2:4].contiguous(), target[:, 2:4].contiguous(), beta=0.03)
    endpoints = 1.8 * start + end
    duration_scale = BASIC_LINEAR_MAX_S - BASIC_LINEAR_MIN_S
    duration = F.smooth_l1_loss(
        prediction[:, 4] / duration_scale, target[:, 4] / duration_scale, beta=0.05,
    )
    if map_weight < 0 or trajectory_weight < 0:
        raise ValueError("map_weight and trajectory_weight must be non-negative")
    if map_weight == 0 and trajectory_weight == 0:
        return endpoints + duration
    if start_scores is None or end_scores is None or start_scores.shape != end_scores.shape:
        raise ValueError("score maps are required when auxiliary map weights are positive")
    loss = endpoints + duration
    if map_weight:
        onset = target.new_full((len(target),), .24)
        liftoff = (onset + target[:, 4] / 2.27).clamp(max=.88)
        map_loss = basic_linear_endpoint_map_loss(start_scores, target[:, :2], onset)
        map_loss = map_loss + basic_linear_endpoint_map_loss(end_scores, target[:, 2:4], liftoff)
        loss = loss + map_weight * map_loss
    if trajectory_weight:
        if trajectory_xy is None or trajectory_mask is None:
            raise ValueError("trajectory targets are required when trajectory_weight is positive")
        # A dedicated track score map avoids forcing endpoint-specific heads to
        # label every intermediate contact position.  Retain the old two-head
        # auxiliary as a backwards-compatible control when none is supplied.
        if trajectory_scores is None:
            trajectory_loss = basic_linear_trajectory_map_loss(start_scores, trajectory_xy, trajectory_mask)
            trajectory_loss = trajectory_loss + basic_linear_trajectory_map_loss(end_scores, trajectory_xy, trajectory_mask)
        else:
            if trajectory_scores.shape != start_scores.shape:
                raise ValueError("trajectory_scores must match endpoint score-map shape")
            trajectory_loss = basic_linear_trajectory_map_loss(
                trajectory_scores, trajectory_xy, trajectory_mask,
            )
        loss = loss + trajectory_weight * trajectory_loss
    return loss


@torch.no_grad()
def basic_linear_metrics(model: torch.nn.Module, loader, device: torch.device, *,
                         correction: AlongPathBias | None = None) -> dict[str, float]:
    """Report independent endpoint geometry and duration errors.

    ``correction`` is an explicit opt-in: an along-path bias fit on a *different*
    (validation) split.  It is never fit here, so scoring a split with a
    correction cannot tune on that split.
    """
    model.eval()
    start_errors: list[float] = []
    end_errors: list[float] = []
    duration_errors: list[float] = []
    recovered: list[float] = []
    start_recovered: list[float] = []
    end_recovered: list[float] = []
    duration_recovered: list[float] = []
    per_knot: list[list[float]] = []
    for batch in loader:
        prediction = model(batch["frames"].to(device))
        if correction is not None:
            prediction = correction.apply(prediction)
        target = batch["target"].to(device)
        errors = knot_errors(prediction, target)
        per_knot.extend(errors.cpu().tolist())
        # "start" and "end" keep their MVP-2 meaning: the first and last knot.
        start = errors[:, 0]
        end = errors[:, -1]
        duration = torch.abs(prediction[:, -1] - target[:, -1])
        start_errors.extend(start.cpu().tolist())
        end_errors.extend(end.cpu().tolist())
        duration_errors.extend(duration.cpu().tolist())
        start_recovered.extend((start <= RECOVERY_ENDPOINT_TOLERANCE).float().cpu().tolist())
        end_recovered.extend((end <= RECOVERY_ENDPOINT_TOLERANCE).float().cpu().tolist())
        duration_recovered.extend((duration <= RECOVERY_DURATION_TOLERANCE_S).float().cpu().tolist())
        recovered.extend(((errors <= RECOVERY_ENDPOINT_TOLERANCE).all(dim=1)
                          & (duration <= RECOVERY_DURATION_TOLERANCE_S)).float().cpu().tolist())
    if not start_errors:
        raise ValueError("cannot evaluate an empty loader")
    endpoint_errors = start_errors + end_errors
    return {
        "samples": float(len(start_errors)),
        "start_coordinate_median": float(np.median(start_errors)),
        "end_coordinate_median": float(np.median(end_errors)),
        "endpoint_coordinate_median": float(np.median(endpoint_errors)),
        "endpoint_coordinate_p90": float(np.quantile(endpoint_errors, 0.90)),
        "duration_mae": float(np.mean(duration_errors)),
        "duration_p90": float(np.quantile(duration_errors, 0.90)),
        "gesture_recovery_accuracy": float(np.mean(recovered)),
        "start_recovery_accuracy": float(np.mean(start_recovered)),
        "end_recovery_accuracy": float(np.mean(end_recovered)),
        "duration_recovery_accuracy": float(np.mean(duration_recovered)),
        "recovery_endpoint_tolerance": RECOVERY_ENDPOINT_TOLERANCE,
        "recovery_duration_tolerance_s": RECOVERY_DURATION_TOLERANCE_S,
        "knots": float(len(per_knot[0])),
        # Per-knot recovery keeps progress visible while the joint gate, which
        # requires every knot at once, is still failing.
        **{f"knot{index}_recovery_accuracy":
           float(np.mean([row[index] <= RECOVERY_ENDPOINT_TOLERANCE for row in per_knot]))
           for index in range(len(per_knot[0]))},
        **{f"knot{index}_coordinate_median": float(np.median([row[index] for row in per_knot]))
           for index in range(len(per_knot[0]))},
    }


@torch.no_grad()
def basic_linear_recovery_records(model: torch.nn.Module, loader, device: torch.device, *,
                                  correction: AlongPathBias | None = None) -> list[dict[str, float]]:
    """Return per-clip recovery evidence for post-hoc split audits.

    The training loader need only provide tensors; optional ``sample_index`` lets
    callers join records to dataset provenance (device, slope, duration) without
    making model evaluation depend on filesystem metadata.
    """
    model.eval()
    records: list[dict[str, float]] = []
    for batch in loader:
        prediction = model(batch["frames"].to(device))
        if correction is not None:
            prediction = correction.apply(prediction)
        target = batch["target"].to(device)
        errors = knot_errors(prediction, target)
        start, end = errors[:, 0], errors[:, -1]
        duration = torch.abs(prediction[:, -1] - target[:, -1])
        recovered = ((errors <= RECOVERY_ENDPOINT_TOLERANCE).all(dim=1)
                     & (duration <= RECOVERY_DURATION_TOLERANCE_S))
        for index in range(len(start)):
            records.append({
                "start_error": float(start[index]),
                "end_error": float(end[index]),
                "duration_error": float(duration[index]),
                "recovered": float(recovered[index]),
                "knot_errors": [float(v) for v in errors[index].cpu()],
                # Keep the raw pair as well as the error: a tail audit needs to
                # know *where* a missed endpoint landed (short of the trail, past
                # it, or on the opposite end) to tell failure modes apart.
                "predicted": [float(value) for value in prediction[index].cpu()],
                "target": [float(value) for value in target[index].cpu()],
            })
    return records


def passes_basic_linear_acceptance(metrics: dict[str, float]) -> bool:
    """MVP 2 gate: both endpoints must be localised, not merely their midpoint."""
    return (
        # Gate every knot when the metrics carry them; fall back to the MVP-2
        # endpoint pair for legacy two-knot reports that predate per-knot keys.
        max([value for key, value in metrics.items()
             if key.startswith("knot") and key.endswith("_coordinate_median")]
            or [metrics["start_coordinate_median"], metrics["end_coordinate_median"]]) <= 0.03
        and metrics["duration_mae"] <= 0.10
        and metrics["gesture_recovery_accuracy"] >= 0.95
    )
