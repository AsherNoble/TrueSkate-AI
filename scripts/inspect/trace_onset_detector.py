"""Small classical detectors for the first visible orange trace frame.

Version 1 is the detector evaluated on 2026-09-13. It intentionally remains
simple so that later variants can be compared with the recorded baseline.
"""

from __future__ import annotations

import numpy as np


def circular_mask(
    height: int,
    width: int,
    center_x: float,
    center_y: float,
    radius: float,
) -> np.ndarray:
    """Return a boolean circle mask in image coordinates."""
    yy, xx = np.ogrid[:height, :width]
    return (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius**2


def warm_colour_image(frame: np.ndarray) -> np.ndarray:
    """Measure how much red exceeds both green and blue at each pixel."""
    values = np.asarray(frame, dtype=np.float32)
    return np.minimum(
        values[..., 0] - values[..., 1],
        values[..., 0] - values[..., 2],
    )


def detect_v1(
    frames: np.ndarray,
    center_x: float,
    center_y: float,
    *,
    radius: float = 12.0,
    threshold: float = 2.0,
) -> int | None:
    """Return the first local frame whose warm-colour increase crosses 2.

    The returned index is relative to ``frames``. Frame zero cannot be selected
    because the method requires its predecessor. This is the exact selection
    rule used for the recorded wide-window baseline.
    """
    values = np.asarray(frames)
    if values.ndim != 4 or values.shape[-1] < 3:
        raise ValueError("frames must have shape (time, height, width, channels)")
    if len(values) < 2:
        return None

    mask = circular_mask(
        values.shape[1], values.shape[2], center_x, center_y, radius
    )
    scores = warm_colour_image(values)[..., mask].mean(axis=-1)
    crossings = np.flatnonzero(np.diff(scores) >= threshold)
    return int(crossings[0] + 1) if len(crossings) else None


def local_brightness_contrast(
    frames: np.ndarray,
    center_x: float,
    center_y: float,
    *,
    core_radius: float = 10.0,
    ring_inner_radius: float = 20.0,
    ring_outer_radius: float = 40.0,
) -> np.ndarray:
    """Return centre brightness minus nearby background brightness per frame."""
    values = np.asarray(frames, dtype=np.float32)
    if values.ndim != 4 or values.shape[-1] < 3:
        raise ValueError("frames must have shape (time, height, width, channels)")
    height, width = values.shape[1:3]
    core = circular_mask(height, width, center_x, center_y, core_radius)
    outer = circular_mask(height, width, center_x, center_y, ring_outer_radius)
    inner = circular_mask(height, width, center_x, center_y, ring_inner_radius)
    ring = outer & ~inner
    if not core.any() or not ring.any():
        raise ValueError("centre and background ring must overlap the image")
    brightness = values[..., :3].mean(axis=-1)
    return brightness[:, core].mean(axis=-1) - brightness[:, ring].mean(axis=-1)


def detect_v2(
    frames: np.ndarray,
    center_x: float,
    center_y: float,
    *,
    immediate_threshold: float = 4.0,
    confirmation_threshold: float = 6.0,
    history_frames: int = 3,
    lookahead_frames: int = 3,
) -> int | None:
    """Find the first persistent, local brightening at the commanded point.

    Nearby brightness is subtracted to reduce responses to broad scene motion.
    A candidate must brighten immediately and remain brighter over a short
    look-ahead. The candidate frame, rather than a later confirmation frame, is
    returned. This is an exploratory second version, not a certified detector.
    """
    if history_frames < 1 or lookahead_frames < 1:
        raise ValueError("history_frames and lookahead_frames must be positive")
    contrast = local_brightness_contrast(frames, center_x, center_y)
    if len(contrast) < 2:
        return None
    changes = np.diff(contrast)
    for frame_index, immediate_change in enumerate(changes, start=1):
        if immediate_change < immediate_threshold:
            continue
        before = np.median(contrast[max(0, frame_index - history_frames) : frame_index])
        after = np.median(
            contrast[frame_index : min(len(contrast), frame_index + lookahead_frames)]
        )
        if after - before >= confirmation_threshold:
            return frame_index
    return None
