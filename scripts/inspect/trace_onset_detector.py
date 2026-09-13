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
