"""Classical visible-touch detection and segment timing calibration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class TapOnset:
    """One detected rendered-tap onset in video time."""

    onset_s: float
    score: float
    threshold: float


@dataclass(frozen=True)
class TwoAnchorTimingFit:
    """Affine mapping from WDA monotonic time to recording video time."""

    intercept_s: float
    rate: float
    anchor_span_s: float

    def video_time_s(self, wda_monotonic_s: float) -> float:
        return self.intercept_s + self.rate * float(wda_monotonic_s)


@dataclass(frozen=True)
class TapOffsetFit:
    """Robust segment-level fit over detected command-to-pixel offsets."""

    offset_s: float | None
    mad_s: float | None
    candidate_offsets_s: tuple[float, ...]
    inlier_offsets_s: tuple[float, ...]
    accepted: bool
    reason: str | None = None


def detect_tap_onset(
    frames: Sequence[np.ndarray],
    frame_times_s: Sequence[float],
    *,
    point_xy: tuple[float, float],
    command_s: float,
    reference_window_s: float = 0.5,
    immediate_threshold: float = 4.0,
    confirmation_threshold: float = 6.0,
    history_frames: int = 3,
    lookahead_frames: int = 3,
) -> TapOnset | None:
    """Find the first persistent local brightening at a known touch point.

    This is the production form of the evaluated V2 detector. It compares a small
    centre disc with a surrounding ring, then requires an immediate brightness
    increase to persist over three frames. Broad floor motion is largely cancelled
    by the ring. ``None`` is a valid result: callers reject the segment rather than
    manufacture a timing label.
    """
    if len(frames) != len(frame_times_s):
        raise ValueError(f"frames/times length mismatch: {len(frames)} != {len(frame_times_s)}")
    if not frames:
        return None
    if reference_window_s <= 0.0:
        raise ValueError(f"reference_window_s must be > 0, got {reference_window_s}")
    if immediate_threshold <= 0 or confirmation_threshold <= 0:
        raise ValueError("detector thresholds must be positive")
    if history_frames < 1 or lookahead_frames < 1:
        raise ValueError("detector history/lookahead must be positive")

    times = np.asarray(frame_times_s, dtype=np.float64)
    if not np.all(np.isfinite(times)):
        raise ValueError("frame_times_s must be finite")
    if np.any(np.diff(times) < 0.0):
        raise ValueError("frame_times_s must be chronological")

    first = np.asarray(frames[0])
    if first.ndim not in (2, 3):
        raise ValueError(f"frames must be grayscale or BGR/RGB arrays, got {first.shape}")
    if not (0.0 <= point_xy[0] <= 1.0 and 0.0 <= point_xy[1] <= 1.0):
        raise ValueError(f"tap point must be normalised in [0, 1], got {point_xy}")
    height, width = first.shape[:2]
    cx = point_xy[0] * (width - 1)
    cy = point_xy[1] * (height - 1)
    yy, xx = np.ogrid[:height, :width]
    scale = max(width / 828.0, 0.25)
    core_radius = max(2.0, 10.0 * scale)
    ring_inner_radius = max(core_radius + 1.0, 10.0, 20.0 * scale)
    ring_outer_radius = max(ring_inner_radius + 2.0, 20.0, 40.0 * scale)
    squared = (xx - cx) ** 2 + (yy - cy) ** 2
    core = squared <= core_radius**2
    ring = (squared <= ring_outer_radius**2) & (squared > ring_inner_radius**2)
    if not core.any() or not ring.any():
        return None
    contrasts: list[float] = []
    for frame in frames:
        image = np.asarray(frame)
        if image.shape != first.shape:
            raise ValueError(f"all frames must share a shape, got {first.shape} and {image.shape}")
        brightness = image.astype(np.float32)
        if brightness.ndim == 3:
            brightness = brightness[..., :3].mean(axis=2)
        contrasts.append(float(brightness[core].mean() - brightness[ring].mean()))

    contrast = np.asarray(contrasts, dtype=np.float64)
    changes = np.diff(contrast)
    candidate_indices = np.flatnonzero(times[1:] >= command_s - reference_window_s) + 1
    for index in candidate_indices:
        if changes[index - 1] < immediate_threshold:
            continue
        before = np.median(contrast[max(0, index - history_frames):index])
        after = np.median(contrast[index:min(len(contrast), index + lookahead_frames)])
        if after - before >= confirmation_threshold:
            return TapOnset(
                onset_s=float(times[int(index)]),
                score=float(after - before),
                threshold=float(confirmation_threshold),
            )
    return None


def fit_two_anchor_timeline(
    first_wda_monotonic_s: float,
    first_video_s: float,
    last_wda_monotonic_s: float,
    last_video_s: float,
    *,
    min_anchor_span_s: float = 30.0,
    min_rate: float = 0.98,
    max_rate: float = 1.02,
) -> TwoAnchorTimingFit:
    """Fit ``video = intercept + rate * WDA`` from two rendered controls."""
    values = (
        first_wda_monotonic_s, first_video_s,
        last_wda_monotonic_s, last_video_s,
    )
    if not all(np.isfinite(value) for value in values):
        raise ValueError("two-anchor timestamps must be finite")
    span = float(last_wda_monotonic_s - first_wda_monotonic_s)
    if span < min_anchor_span_s:
        raise ValueError(
            f"two-anchor WDA span {span:.3f}s is below required {min_anchor_span_s:.3f}s"
        )
    rate = float((last_video_s - first_video_s) / span)
    if not min_rate <= rate <= max_rate:
        raise ValueError(f"two-anchor clock rate {rate:.6f} is outside [{min_rate}, {max_rate}]")
    intercept = float(first_video_s - rate * first_wda_monotonic_s)
    return TwoAnchorTimingFit(intercept_s=intercept, rate=rate, anchor_span_s=span)


def fit_tap_offsets(
    offsets_s: Sequence[float],
    *,
    min_taps: int = 2,
    max_mad_s: float = 0.10,
    outlier_floor_s: float = 0.05,
) -> TapOffsetFit:
    """Robustly fit one segment offset, abstaining on sparse/noisy evidence.

    A median/MAD pass removes a bad local detector result before deciding whether
    the remaining taps agree tightly enough to use.  Failing closed is deliberate:
    a segment with no reliable calibration retains the aligner's existing offsets
    rather than silently receiving a speculative correction.
    """
    if min_taps < 1:
        raise ValueError(f"min_taps must be >= 1, got {min_taps}")
    if max_mad_s < 0.0 or outlier_floor_s < 0.0:
        raise ValueError("max_mad_s and outlier_floor_s must be >= 0")
    candidates = tuple(float(value) for value in offsets_s if np.isfinite(value))
    if len(candidates) < min_taps:
        return TapOffsetFit(
            offset_s=None,
            mad_s=None,
            candidate_offsets_s=candidates,
            inlier_offsets_s=(),
            accepted=False,
            reason=f"need at least {min_taps} detected taps; found {len(candidates)}",
        )

    values = np.asarray(candidates, dtype=np.float64)
    initial_median = float(np.median(values))
    initial_mad = float(np.median(np.abs(values - initial_median)))
    # Convert MAD to a Gaussian-equivalent scale for clipping, but never make the
    # gate narrower than 50 ms: video timestamps are quantised to 30 fps.
    clip_s = max(outlier_floor_s, 3.0 * 1.4826 * initial_mad)
    inlier_values = values[np.abs(values - initial_median) <= clip_s]
    inliers = tuple(float(value) for value in inlier_values)
    if len(inliers) < min_taps:
        return TapOffsetFit(
            offset_s=None,
            mad_s=None,
            candidate_offsets_s=candidates,
            inlier_offsets_s=inliers,
            accepted=False,
            reason=f"only {len(inliers)} inlier tap(s) after robust clipping",
        )
    offset = float(np.median(inlier_values))
    mad = float(np.median(np.abs(inlier_values - offset)))
    if mad > max_mad_s:
        return TapOffsetFit(
            offset_s=offset,
            mad_s=mad,
            candidate_offsets_s=candidates,
            inlier_offsets_s=inliers,
            accepted=False,
            reason=(f"tap offset MAD {mad:.3f}s exceeds allowed {max_mad_s:.3f}s"),
        )
    return TapOffsetFit(
        offset_s=offset,
        mad_s=mad,
        candidate_offsets_s=candidates,
        inlier_offsets_s=inliers,
        accepted=True,
    )
