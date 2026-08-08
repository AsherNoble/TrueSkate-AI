"""Training and evaluation helpers for the basic Model 1 hold regressor."""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from trueskate_ai.vision.basic_hold_dataset import HOLD_DURATION_MAX_S, HOLD_DURATION_MIN_S


def basic_hold_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Balanced robust error in native output dimensions.

    Duration is scaled to the same approximately unit span as the normalized
    coordinates, preventing it from being accidentally underweighted.
    """
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 3:
        raise ValueError("prediction and target must both have shape [batch,3]")
    position = F.smooth_l1_loss(prediction[:, :2], target[:, :2], beta=0.03)
    duration_scale = HOLD_DURATION_MAX_S - HOLD_DURATION_MIN_S
    duration = F.smooth_l1_loss(
        prediction[:, 2] / duration_scale, target[:, 2] / duration_scale, beta=0.05
    )
    return position + duration


@torch.no_grad()
def basic_hold_metrics(model: torch.nn.Module, loader, device: torch.device) -> dict[str, float]:
    """Compute specified median/90th-percentile position and duration errors."""
    model.eval()
    coordinate_errors: list[float] = []
    duration_errors: list[float] = []
    for batch in loader:
        prediction = model(batch["frames"].to(device))
        target = batch["target"].to(device)
        coordinate_errors.extend(torch.linalg.vector_norm(prediction[:, :2] - target[:, :2], dim=1).cpu().tolist())
        duration_errors.extend(torch.abs(prediction[:, 2] - target[:, 2]).cpu().tolist())
    if not coordinate_errors:
        raise ValueError("cannot evaluate an empty loader")
    return {
        "samples": float(len(coordinate_errors)),
        "coordinate_median": float(np.median(coordinate_errors)),
        "coordinate_p90": float(np.quantile(coordinate_errors, 0.90)),
        "duration_mae": float(np.mean(duration_errors)),
        "duration_p90": float(np.quantile(duration_errors, 0.90)),
    }


def passes_basic_hold_acceptance(metrics: dict[str, float]) -> bool:
    return metrics["coordinate_median"] <= 0.03 and metrics["duration_mae"] <= 0.10
