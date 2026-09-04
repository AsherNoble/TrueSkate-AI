"""Complete-gesture recovery contracts and exact certification bounds."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import beta

from trueskate_ai.data.trajectory_resample import resample_command_at_times

PATH_POINTS = 5
POSITION_TOLERANCE = 0.03
DURATION_TOLERANCE_S = 0.10
SPIN_FRAME_TOLERANCE = 2
CERTIFICATION_MIN_EXAMPLES = 30_000
CERTIFICATION_TARGET = 0.999


@dataclass(frozen=True)
class TouchTrack:
    """One chronological physical touch reconstructed by Model 1."""

    kind: str
    start_s: float
    end_s: float
    points: tuple[tuple[float, float], ...]
    easing_power: float = 1.0

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def one_sided_binomial_lower_bound(
    successes: int,
    total: int,
    *,
    confidence: float = 0.95,
) -> float:
    """Exact one-sided Clopper-Pearson lower confidence bound."""
    if isinstance(successes, bool) or isinstance(total, bool):
        raise ValueError("successes and total must be integer counts")
    if not 0 <= successes <= total or total < 1:
        raise ValueError(f"invalid binomial counts successes={successes}, total={total}")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if successes == 0:
        return 0.0
    return float(beta.ppf(1.0 - confidence, successes, total - successes + 1))


def _validate_track(track: TouchTrack) -> None:
    if track.kind not in {"drag", "spin"}:
        raise ValueError(f"unknown touch-track kind {track.kind!r}")
    if not np.isfinite((track.start_s, track.end_s)).all() or track.end_s < track.start_s:
        raise ValueError(f"invalid touch interval {(track.start_s, track.end_s)}")
    points = np.asarray(track.points, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 1:
        raise ValueError("touch-track points must have shape [N,2]")
    if not np.isfinite(points).all() or np.any((points < 0.0) | (points > 1.0)):
        raise ValueError("touch-track points must be finite normalized coordinates")


def _fixed_time_points(track: TouchTrack, *, count: int = PATH_POINTS) -> np.ndarray:
    _validate_track(track)
    points = np.asarray(track.points, dtype=float)
    if track.kind != "drag" or len(points) < 2:
        raise ValueError("path comparison requires a drag with at least two points")
    return resample_command_at_times(
        points,
        max(track.duration_s, 1e-9),
        knots=count,
        easing_power=track.easing_power,
    )


def linear_drag_recovered(predicted: TouchTrack, target: TouchTrack) -> bool:
    """Legacy linear contract: endpoints and duration only."""
    if predicted.kind != "drag" or target.kind != "drag":
        return False
    predicted_points = np.asarray(predicted.points, dtype=float)
    target_points = np.asarray(target.points, dtype=float)
    if len(predicted_points) < 2 or len(target_points) < 2:
        return False
    endpoint_errors = np.linalg.norm(
        predicted_points[[0, -1]] - target_points[[0, -1]], axis=1
    )
    return bool(
        np.all(endpoint_errors <= POSITION_TOLERANCE)
        and abs(predicted.duration_s - target.duration_s) <= DURATION_TOLERANCE_S
    )


def curved_drag_recovered(predicted: TouchTrack, target: TouchTrack) -> bool:
    """Five fixed-time path points and duration, independent of control-point form."""
    if predicted.kind != "drag" or target.kind != "drag":
        return False
    try:
        errors = np.linalg.norm(_fixed_time_points(predicted) - _fixed_time_points(target), axis=1)
    except ValueError:
        return False
    return bool(
        np.all(errors <= POSITION_TOLERANCE)
        and abs(predicted.duration_s - target.duration_s) <= DURATION_TOLERANCE_S
    )


def spin_interval_recovered(
    predicted: TouchTrack | None,
    target: TouchTrack | None,
    *,
    fps: float = 30.0,
) -> bool:
    """Require the same spin-active state and both edges within two frames."""
    if fps <= 0 or not np.isfinite(fps):
        raise ValueError("fps must be positive and finite")
    if predicted is None or target is None:
        return predicted is target
    if predicted.kind != "spin" or target.kind != "spin":
        return False
    tolerance = SPIN_FRAME_TOLERANCE / fps
    return bool(
        abs(predicted.start_s - target.start_s) <= tolerance + 1e-12
        and abs(predicted.end_s - target.end_s) <= tolerance + 1e-12
    )


def complete_gesture_recovered(
    predicted_tracks: Sequence[TouchTrack],
    target_tracks: Sequence[TouchTrack],
    *,
    subtype: str,
    fps: float = 30.0,
) -> bool:
    """Strictly score strokes, overlap, and spin without forgiving count errors.

    Exact track counts and kinds make every extra, missing, merged, or lost
    overlapping touch a complete-gesture failure.
    """
    if subtype not in {"linear", "curved", "curved_spin"}:
        raise ValueError(f"unknown certification subtype {subtype!r}")
    if len(predicted_tracks) != len(target_tracks):
        return False
    predicted = sorted(predicted_tracks, key=lambda track: (track.start_s, track.kind, track.end_s))
    target = sorted(target_tracks, key=lambda track: (track.start_s, track.kind, track.end_s))
    if [track.kind for track in predicted] != [track.kind for track in target]:
        return False
    for predicted_track, target_track in zip(predicted, target):
        if target_track.kind == "spin":
            if subtype != "curved_spin" or not spin_interval_recovered(
                predicted_track, target_track, fps=fps
            ):
                return False
        elif subtype == "linear":
            if not linear_drag_recovered(predicted_track, target_track):
                return False
        elif not curved_drag_recovered(predicted_track, target_track):
            return False
    target_has_spin = any(track.kind == "spin" for track in target)
    return target_has_spin == (subtype == "curved_spin")


def certification_report(
    outcomes: Mapping[str, Iterable[bool]],
    *,
    confidence: float = 0.95,
    target: float = CERTIFICATION_TARGET,
    min_examples: int = CERTIFICATION_MIN_EXAMPLES,
) -> dict[str, object]:
    """Report and gate each subtype independently; no pooled pass is allowed."""
    reports: dict[str, dict[str, object]] = {}
    required = {"linear", "curved", "curved_spin"}
    if set(outcomes) != required:
        raise ValueError(f"certification requires exactly {sorted(required)}")
    for subtype in sorted(required):
        values = [bool(value) for value in outcomes[subtype]]
        if not values:
            raise ValueError(f"{subtype} certification has no examples")
        successes = sum(values)
        lower = one_sided_binomial_lower_bound(successes, len(values), confidence=confidence)
        reports[subtype] = {
            "examples": len(values),
            "successes": successes,
            "failures": len(values) - successes,
            "point_recovery": successes / len(values),
            "one_sided_lower_bound": lower,
            "enough_examples": len(values) >= min_examples,
            "passes": len(values) >= min_examples and lower > target,
        }
    return {
        "confidence": confidence,
        "target": target,
        "minimum_examples_per_subtype": min_examples,
        "subtypes": reports,
        "passes": all(bool(report["passes"]) for report in reports.values()),
    }
