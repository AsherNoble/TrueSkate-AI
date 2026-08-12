"""Training and evaluation helpers for MVP 2 finite-slope linear drags."""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from trueskate_ai.data.gesture_sampling import BASIC_LINEAR_MAX_S, BASIC_LINEAR_MIN_S

RECOVERY_ENDPOINT_TOLERANCE = 0.03
RECOVERY_DURATION_TOLERANCE_S = 0.10


def basic_linear_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Robust endpoint error plus duration error in matched native scales."""
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 5:
        raise ValueError("prediction and target must both have shape [batch,5]")
    # The acceptance gate is clipped per sample; the baseline achieved strong
    # medians but left a high-error tail just beyond 0.03.  Retain robust local
    # learning while adding a modest squared-error term to pull that tail in.
    endpoint_residual = prediction[:, :4] - target[:, :4]
    endpoints = F.smooth_l1_loss(prediction[:, :4], target[:, :4], beta=0.03)
    endpoint_tail = endpoint_residual.square().mean()
    duration_scale = BASIC_LINEAR_MAX_S - BASIC_LINEAR_MIN_S
    duration = F.smooth_l1_loss(
        prediction[:, 4] / duration_scale, target[:, 4] / duration_scale, beta=0.05,
    )
    return endpoints + 0.35 * endpoint_tail + duration


@torch.no_grad()
def basic_linear_metrics(model: torch.nn.Module, loader, device: torch.device) -> dict[str, float]:
    """Report independent endpoint geometry and duration errors."""
    model.eval()
    start_errors: list[float] = []
    end_errors: list[float] = []
    duration_errors: list[float] = []
    recovered: list[float] = []
    for batch in loader:
        prediction = model(batch["frames"].to(device))
        target = batch["target"].to(device)
        start_errors.extend(torch.linalg.vector_norm(prediction[:, :2] - target[:, :2], dim=1).cpu().tolist())
        end_errors.extend(torch.linalg.vector_norm(prediction[:, 2:4] - target[:, 2:4], dim=1).cpu().tolist())
        duration_errors.extend(torch.abs(prediction[:, 4] - target[:, 4]).cpu().tolist())
        start = torch.linalg.vector_norm(prediction[:, :2] - target[:, :2], dim=1)
        end = torch.linalg.vector_norm(prediction[:, 2:4] - target[:, 2:4], dim=1)
        duration = torch.abs(prediction[:, 4] - target[:, 4])
        recovered.extend(((start <= RECOVERY_ENDPOINT_TOLERANCE)
                          & (end <= RECOVERY_ENDPOINT_TOLERANCE)
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
        "recovery_endpoint_tolerance": RECOVERY_ENDPOINT_TOLERANCE,
        "recovery_duration_tolerance_s": RECOVERY_DURATION_TOLERANCE_S,
    }


@torch.no_grad()
def basic_linear_recovery_records(model: torch.nn.Module, loader, device: torch.device) -> list[dict[str, float]]:
    """Return per-clip recovery evidence for post-hoc split audits.

    The training loader need only provide tensors; optional ``sample_index`` lets
    callers join records to dataset provenance (device, slope, duration) without
    making model evaluation depend on filesystem metadata.
    """
    model.eval()
    records: list[dict[str, float]] = []
    for batch in loader:
        prediction = model(batch["frames"].to(device))
        target = batch["target"].to(device)
        start = torch.linalg.vector_norm(prediction[:, :2] - target[:, :2], dim=1)
        end = torch.linalg.vector_norm(prediction[:, 2:4] - target[:, 2:4], dim=1)
        duration = torch.abs(prediction[:, 4] - target[:, 4])
        recovered = ((start <= RECOVERY_ENDPOINT_TOLERANCE)
                     & (end <= RECOVERY_ENDPOINT_TOLERANCE)
                     & (duration <= RECOVERY_DURATION_TOLERANCE_S))
        for index in range(len(start)):
            records.append({
                "start_error": float(start[index]),
                "end_error": float(end[index]),
                "duration_error": float(duration[index]),
                "recovered": float(recovered[index]),
            })
    return records


def passes_basic_linear_acceptance(metrics: dict[str, float]) -> bool:
    """MVP 2 gate: both endpoints must be localised, not merely their midpoint."""
    return (
        metrics["start_coordinate_median"] <= 0.03
        and metrics["end_coordinate_median"] <= 0.03
        and metrics["duration_mae"] <= 0.10
        and metrics["gesture_recovery_accuracy"] >= 0.95
    )
