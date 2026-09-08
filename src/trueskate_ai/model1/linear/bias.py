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
legal wherever the model runs, not only where labels exist — and the fit uses
that same predicted chord by default, so the scalar estimated is the quantity
the correction removes.

The commanded-chord decomposition (the 2026-08-18 autopsy's operator) remains
available for comparison, but a bias fit that way must not be applied: see
``AlongPathBias.apply``.
"""
from __future__ import annotations

import hashlib
import math
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


AXES = ("predicted", "commanded")


def signed_along_path_error(record: Mapping[str, Sequence[float]], *, knot: int = -1,
                            axis: str = "predicted") -> float | None:
    """Signed error of one knot projected onto its path direction.

    Negative means the prediction fell *short* along the path — the undershoot
    the autopsy measured.  Returns ``None`` for a degenerate (zero-length) path.

    ``axis`` selects which chord defines "along":

    - ``"predicted"`` (default) matches ``AlongPathBias.apply``, which has only
      the prediction to work with.  Estimator and corrector then share an axis,
      so the fitted scalar is the quantity the correction actually removes.
    - ``"commanded"`` is the 2026-08-18 autopsy's operator, which decomposes
      against the label.  Kept so the two can be compared on real records
      (EQ-002); it is not usable at inference.
    """
    if axis not in AXES:
        raise ValueError(f"axis must be one of {AXES}")
    predicted, predicted_previous = _knot_pair(record["predicted"], knot)
    target, target_previous = _knot_pair(record["target"], knot)
    direction = (predicted - predicted_previous if axis == "predicted"
                 else target - target_previous)
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
    axis: str = "predicted"
    # Which split the shift was fit on.  The "this is not test tuning" claim is
    # only checkable if the artefact says where it came from, so callers record
    # it here and evaluators report it alongside the number.
    fit_on: str = "unspecified"

    def apply(self, prediction: torch.Tensor) -> torch.Tensor:
        """Shift the corrected knot forward along the *predicted* path.

        Uses only the prediction, so this is valid at inference time.  Returns a
        new tensor; the input is not modified.  A zero shift is an exact no-op.
        """
        if self.axis != "predicted":
            # apply() has only the prediction, so it can use no other chord.  A
            # commanded-axis fit applied here silently reproduces the estimator/
            # corrector mismatch this module exists to remove.
            raise ValueError(
                f"a bias fit on the {self.axis!r} axis cannot be applied: apply() corrects along "
                "the predicted chord, so only a predicted-axis fit is consistent")
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
                        statistic: str = "mean", axis: str = "predicted",
                        fit_on: str = "unspecified") -> AlongPathBias:
    """Fit the along-path shift from validation recovery records.

    ``statistic`` is ``"mean"`` (matches the autopsy's counterfactual) or
    ``"median"`` (robust to a few large misses).  Records whose path is
    degenerate are skipped.  With no usable record the shift is zero, so the
    correction degrades to a no-op rather than to an arbitrary number.
    """
    if statistic not in {"mean", "median"}:
        raise ValueError("statistic must be 'mean' or 'median'")
    errors = [error for error in
              (signed_along_path_error(record, knot=knot, axis=axis) for record in records)
              if error is not None]
    if not errors:
        return AlongPathBias(shift=0., samples=0, knot=knot, statistic=statistic, axis=axis,
                             fit_on=fit_on)
    shift = float(np.mean(errors)) if statistic == "mean" else float(np.median(errors))
    return AlongPathBias(shift=shift, samples=len(errors), knot=knot, statistic=statistic, axis=axis,
                         fit_on=fit_on)


def perpendicular_error(record: Mapping[str, Sequence[float]], *, knot: int = -1,
                        axis: str = "commanded") -> float | None:
    """Magnitude of one knot's error across its path.

    Nonnegative by construction — a folded quantity, so its sd is not comparable
    to the signed along-path sd without saying so.

    EQ-008 established that the disagreement between the predicted-chord and
    commanded-chord operators scales as ``perp**2 / |chord|``.  The whole
    predicted-chord design therefore rests on the perpendicular error staying
    near the 0.0032 sd the autopsy measured once, on one split — so an evaluator
    that applies the correction should report this distribution beside it.
    """
    if axis not in AXES:
        raise ValueError(f"axis must be one of {AXES}")
    predicted, predicted_previous = _knot_pair(record["predicted"], knot)
    target, target_previous = _knot_pair(record["target"], knot)
    direction = (predicted - predicted_previous if axis == "predicted"
                 else target - target_previous)
    length = float(np.linalg.norm(direction))
    if length < MIN_PATH_LENGTH:
        return None
    unit = direction / length
    offset = predicted - target
    return float(np.linalg.norm(offset - float(np.dot(offset, unit)) * unit))


def discordant_pairs(before: Sequence[float], after: Sequence[float]) -> tuple[int, int]:
    """Count clips a correction flipped: ``(gained, lost)``.

    A correction is judged on which clips it moves, not on the accuracy delta:
    at n=153 a 3-clip net change is invisible inside its own confidence
    interval, while the discordant counts are exactly what a paired test needs.
    """
    if len(before) != len(after):
        raise ValueError("paired comparison needs equal-length outcome vectors")
    gained = sum(1 for old, new in zip(before, after) if not old and new)
    lost = sum(1 for old, new in zip(before, after) if old and not new)
    return gained, lost


def mcnemar_exact_p(gained: int, lost: int) -> float:
    """Two-sided exact McNemar p for the discordant pairs.

    Under the null the correction flips clips either way with equal
    probability, so the discordant pairs are Binomial(n, 0.5).  Exact rather
    than chi-squared because the counts here are single digits.
    """
    if gained < 0 or lost < 0:
        raise ValueError("discordant counts cannot be negative")
    total = gained + lost
    if total == 0:
        return 1.
    extreme = max(gained, lost)
    tail = sum(math.comb(total, k) for k in range(extreme, total + 1)) / 2 ** total
    return min(1., 2 * tail)


def along_path_fit_key(split: str, indices: Sequence[int]) -> str:
    """Provenance string derived from the index set actually fitted on.

    A caller-written label can claim a provenance the artefact does not have.
    Hashing the indices makes the claim checkable: re-derive the split, re-hash,
    and a mismatch means the shift came from somewhere else.
    """
    digest = hashlib.sha256(",".join(str(int(index)) for index in indices).encode())
    return f"{split}[{len(indices)}]:{digest.hexdigest()[:16]}"
