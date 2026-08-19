"""Validation-fit along-path endpoint bias correction.

The 2026-08-18 failure autopsy found the MVP-2 tail is not scatter: the last
knot's error lives almost entirely *along* the commanded path (perpendicular
error within tolerance for 100% of test clips) and is negative in 85% of them.
That is the signature of soft-argmax over a *cumulative* trail — the rendered
line persists behind the finger, so end attention averages backward from the
tip.  The first knot has no trailing mass behind it and shows no such asymmetry.

A bias is removable in a way that scatter is not.  This module fits one scalar
shift on a *validation* split and applies it unchanged elsewhere, which is a
correction rather than test tuning: nothing here ever reads the split it is
scored on.

Applying the shift needs a direction.  At fit time the commanded path is
available, but at inference it is not, so ``AlongPathBias.apply`` takes the
direction from the model's *own* predicted knots.  The correction is therefore
legal wherever the model runs, not only where labels exist.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

# Below this the path is too short for "along" to have a meaning; such clips are
# excluded from the fit and left untouched by the correction.
MIN_PATH_LENGTH = 1e-6


def _knot_pair(vector: Sequence[float], knot: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (knot position, the position defining its path direction)."""
    values = np.asarray(vector, dtype=np.float64)
    if values.size < 5 or values.size % 2 == 0:
        raise ValueError(f"record vector width {values.size} is not 2K+1 for any K>=2")
    knots = values[:-1].reshape(-1, 2)
    index = range(len(knots))[knot]
    if index == 0:
        raise ValueError("the first knot has no preceding knot to define a direction")
    return knots[index], knots[index - 1]


def signed_along_path_error(record: Mapping[str, Sequence[float]], *, knot: int = -1) -> float | None:
    """Signed error of one knot projected onto its commanded path direction.

    Negative means the prediction fell *short* along the path — the undershoot
    the autopsy measured.  Returns ``None`` for a degenerate (zero-length) path.
    """
    predicted, _ = _knot_pair(record["predicted"], knot)
    target, previous = _knot_pair(record["target"], knot)
    direction = target - previous
    length = float(np.linalg.norm(direction))
    if length < MIN_PATH_LENGTH:
        return None
    return float(np.dot(predicted - target, direction / length))


@dataclass(frozen=True)
class AlongPathBias:
    """One scalar shift, fit on validation records, applied along the path."""

    shift: float
    samples: int
    knot: int = -1
    statistic: str = "mean"

    def apply(self, prediction: torch.Tensor) -> torch.Tensor:
        """Shift the corrected knot forward along the *predicted* path.

        Uses only the prediction, so this is valid at inference time.  Returns a
        new tensor; the input is not modified.  A zero shift is an exact no-op.
        """
        if prediction.ndim != 2 or prediction.shape[1] < 5 or prediction.shape[1] % 2 == 0:
            raise ValueError("prediction must have shape [batch, 2K+1] with K>=2")
        if self.shift == 0.:
            return prediction.clone()
        knots = prediction[:, :-1].reshape(len(prediction), -1, 2)
        index = range(knots.shape[1])[self.knot]
        if index == 0:
            raise ValueError("the first knot has no preceding knot to define a direction")
        direction = knots[:, index] - knots[:, index - 1]
        length = torch.linalg.vector_norm(direction, dim=1, keepdim=True)
        unit = torch.where(length >= MIN_PATH_LENGTH, direction / length.clamp_min(MIN_PATH_LENGTH),
                           torch.zeros_like(direction))
        corrected = prediction.clone()
        offset = 2 * (index if self.knot >= 0 else knots.shape[1] + self.knot)
        corrected[:, offset:offset + 2] = knots[:, index] - self.shift * unit
        return corrected


def fit_along_path_bias(records: Iterable[Mapping[str, Sequence[float]]], *, knot: int = -1,
                        statistic: str = "mean") -> AlongPathBias:
    """Fit the along-path shift from validation recovery records.

    ``statistic`` is ``"mean"`` (matches the autopsy's counterfactual) or
    ``"median"`` (robust to a few large misses).  Records whose path is
    degenerate are skipped.  With no usable record the shift is zero, so the
    correction degrades to a no-op rather than to an arbitrary number.
    """
    if statistic not in {"mean", "median"}:
        raise ValueError("statistic must be 'mean' or 'median'")
    errors = [error for error in
              (signed_along_path_error(record, knot=knot) for record in records)
              if error is not None]
    if not errors:
        return AlongPathBias(shift=0., samples=0, knot=knot, statistic=statistic)
    shift = float(np.mean(errors)) if statistic == "mean" else float(np.median(errors))
    return AlongPathBias(shift=shift, samples=len(errors), knot=knot, statistic=statistic)
