"""Frame-difference timing calibration from known stationary taps.

The Model 1 stationary-touch MVP deliberately includes taps at manifest-known
positions.  Those taps are a *timing* clapperboard: compare a pre-touch local
reference against the video around the commanded point, find the first rendered
mark, then fit one command-to-pixel offset for the segment.

Only timing is inferred from pixels here.  The touch position remains the
manifest's commanded position, so this is not positional label leakage.  The
detector is intentionally colour-agnostic: a local frame difference is more
robust to the game's trace colour and video encoding than a hard-coded orange
threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class TapOnset:
    """One detected rendered-tap onset in video time."""

    onset_s: float
    score: int
    threshold: int


@dataclass(frozen=True)
class TapOffsetFit:
    """Robust segment-level fit over detected command-to-pixel offsets."""

    offset_s: float | None
    mad_s: float | None
    candidate_offsets_s: tuple[float, ...]
    inlier_offsets_s: tuple[float, ...]
    accepted: bool
    reason: str | None = None


def _roi(point_xy: tuple[float, float], shape: tuple[int, ...], radius_norm: float) -> tuple[slice, slice]:
    """Return a clipped screen-normalised patch around a commanded touch point."""
    if len(shape) < 2:
        raise ValueError(f"frame needs at least 2 dimensions, got shape={shape}")
    if not (0.0 <= point_xy[0] <= 1.0 and 0.0 <= point_xy[1] <= 1.0):
        raise ValueError(f"tap point must be normalised in [0, 1], got {point_xy}")
    if not (0.0 < radius_norm <= 0.5):
        raise ValueError(f"radius_norm must be in (0, 0.5], got {radius_norm}")
    height, width = shape[:2]
    cx = int(round(point_xy[0] * (width - 1)))
    cy = int(round(point_xy[1] * (height - 1)))
    rx = max(4, int(round(width * radius_norm)))
    ry = max(4, int(round(height * radius_norm)))
    return (
        slice(max(0, cy - ry), min(height, cy + ry + 1)),
        slice(max(0, cx - rx), min(width, cx + rx + 1)),
    )


def _changed_pixels(frame: np.ndarray, reference: np.ndarray, *, pixel_delta: int) -> int:
    """Count locally changed pixels without uint8-wraparound artefacts."""
    diff = np.abs(frame.astype(np.int16) - reference.astype(np.int16))
    if diff.ndim == 3:
        diff = diff.max(axis=2)
    return int(np.count_nonzero(diff >= pixel_delta))


def detect_tap_onset(
    frames: Sequence[np.ndarray],
    frame_times_s: Sequence[float],
    *,
    point_xy: tuple[float, float],
    command_s: float,
    reference_window_s: float = 0.5,
    radius_norm: float = 0.06,
    pixel_delta: int = 20,
    min_changed_pixels: int = 12,
    persistence_frames: int = 2,
) -> TapOnset | None:
    """Find a tap mark's first visible frame in a local video window.

    ``frames`` must cover a short period before ``command_s`` and a later search
    period.  A median of the pre-command patch is the static-scene reference.
    Detection requires two consecutive changed frames; taps render for roughly six
    30fps frames on device, while the persistence rule rejects one-frame codec noise.
    ``None`` is a valid result: callers should exclude an undetectable tap rather
    than force a timing label from it.
    """
    if len(frames) != len(frame_times_s):
        raise ValueError(f"frames/times length mismatch: {len(frames)} != {len(frame_times_s)}")
    if not frames:
        return None
    if reference_window_s <= 0.0:
        raise ValueError(f"reference_window_s must be > 0, got {reference_window_s}")
    if pixel_delta < 1 or min_changed_pixels < 1 or persistence_frames < 1:
        raise ValueError("pixel_delta, min_changed_pixels, and persistence_frames must be >= 1")

    times = np.asarray(frame_times_s, dtype=np.float64)
    if not np.all(np.isfinite(times)):
        raise ValueError("frame_times_s must be finite")
    if np.any(np.diff(times) < 0.0):
        raise ValueError("frame_times_s must be chronological")

    first = np.asarray(frames[0])
    if first.ndim not in (2, 3):
        raise ValueError(f"frames must be grayscale or BGR/RGB arrays, got {first.shape}")
    ys, xs = _roi(point_xy, first.shape, radius_norm)
    patches: list[np.ndarray] = []
    for frame in frames:
        image = np.asarray(frame)
        if image.shape != first.shape:
            raise ValueError(f"all frames must share a shape, got {first.shape} and {image.shape}")
        patches.append(image[ys, xs])

    reference_indices = np.flatnonzero(
        (times >= command_s - reference_window_s) & (times < command_s)
    )
    # Three frames give the median enough protection from a fading previous mark.
    if len(reference_indices) < 3:
        return None
    reference = np.median(
        np.stack([patches[int(index)] for index in reference_indices], axis=0), axis=0
    )
    reference_scores = np.asarray(
        [_changed_pixels(patches[int(index)], reference, pixel_delta=pixel_delta)
         for index in reference_indices],
        dtype=np.float64,
    )
    noise_median = float(np.median(reference_scores))
    noise_mad = float(np.median(np.abs(reference_scores - noise_median)))
    # The fixed floor preserves sensitivity on a perfectly static recording; the
    # robust noise term adapts to h264 shimmer without letting a one-frame spike
    # define the threshold.
    threshold = max(
        min_changed_pixels,
        int(np.ceil(noise_median + max(5.0, 3.0 * 1.4826 * noise_mad))),
    )
    scores = np.asarray(
        [_changed_pixels(patch, reference, pixel_delta=pixel_delta) for patch in patches],
        dtype=np.int64,
    )
    candidate_indices = np.flatnonzero(times >= command_s)
    for index in candidate_indices:
        stop = min(len(scores), int(index) + persistence_frames)
        if stop - int(index) < persistence_frames:
            break
        if bool(np.all(scores[int(index):stop] >= threshold)):
            return TapOnset(
                onset_s=float(times[int(index)]),
                score=int(scores[int(index)]),
                threshold=threshold,
            )
    return None


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
