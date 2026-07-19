"""Canonical spatial peak decoding for Model 1 touch heatmaps.

Model 1 renders touch centers on the inclusive pixel grid: a normalised
coordinate of 1.0 lands at pixel ``width - 1`` or ``height - 1``.  Decoding
uses the same convention so predictions round-trip without an inward bias.

The decoder first collapses connected local-maximum plateaus to their centroid,
then applies deterministic score-ordered greedy NMS.  This preserves distinct
simultaneous touches while preventing quantised or noisy bumps from becoming
multiple detections.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import math

import numpy as np


@dataclass(frozen=True)
class TouchPeak:
    """One distinct heatmap maximum in normalised screen coordinates."""

    x: float
    y: float
    score: float


def _square_maximum_filter(values: np.ndarray, radius: int) -> np.ndarray:
    """Pure-NumPy separable maximum filter with nearest-edge padding."""

    if radius == 0:
        return values.copy()
    size = 2 * radius + 1
    horizontal = np.lib.stride_tricks.sliding_window_view(
        np.pad(values, ((0, 0), (radius, radius)), mode="edge"),
        size,
        axis=1,
    ).max(axis=-1)
    return np.lib.stride_tricks.sliding_window_view(
        np.pad(horizontal, ((radius, radius), (0, 0)), mode="edge"),
        size,
        axis=0,
    ).max(axis=-1)


def _component_centers(mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return four-connected component coordinates without a SciPy dependency."""

    remaining = mask.copy()
    components: list[tuple[np.ndarray, np.ndarray]] = []
    height, width = remaining.shape
    for start_y, start_x in np.argwhere(remaining):
        if not remaining[start_y, start_x]:
            continue
        remaining[start_y, start_x] = False
        queue = deque([(int(start_y), int(start_x))])
        ys: list[int] = []
        xs: list[int] = []
        while queue:
            y, x = queue.popleft()
            ys.append(y)
            xs.append(x)
            for neighbor_y, neighbor_x in (
                (y - 1, x),
                (y + 1, x),
                (y, x - 1),
                (y, x + 1),
            ):
                if (
                    0 <= neighbor_y < height
                    and 0 <= neighbor_x < width
                    and remaining[neighbor_y, neighbor_x]
                ):
                    remaining[neighbor_y, neighbor_x] = False
                    queue.append((neighbor_y, neighbor_x))
        components.append(
            (np.asarray(ys, dtype=np.intp), np.asarray(xs, dtype=np.intp))
        )
    return components


def extract_touch_peaks(
    heatmap: np.ndarray,
    *,
    threshold: float = 0.30,
    max_peaks: int = 2,
    nms_radius_px: int = 6,
) -> list[TouchPeak]:
    """Extract distinct local maxima from one ``[H, W]`` heatmap.

    Max-combined Gaussian targets can represent several simultaneous touches.
    A maximum filter identifies candidate maxima, connected flat maxima are
    represented by their centroid, and greedy Euclidean NMS removes duplicate
    candidates around the same touch.
    """
    hm = np.asarray(heatmap, dtype=np.float64)
    if hm.ndim != 2:
        raise ValueError(f"heatmap must be 2-D, got shape {hm.shape}")
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be finite and in [0,1], got {threshold}")
    if max_peaks < 1:
        raise ValueError(f"max_peaks must be >= 1, got {max_peaks}")
    if nms_radius_px < 0:
        raise ValueError(f"nms_radius_px must be >= 0, got {nms_radius_px}")
    if hm.shape[0] < 1 or hm.shape[1] < 1:
        return []

    hm = np.nan_to_num(hm, nan=-np.inf, posinf=1.0, neginf=-np.inf)
    if not np.any(hm >= threshold):
        return []

    local_max = hm == _square_maximum_filter(hm, nms_radius_px)
    candidates: list[tuple[float, float, float]] = []
    for yy, xx in _component_centers(local_max & (hm >= threshold)):
        score = float(hm[yy, xx].max())
        # Quantised/saturated bumps can have a broad flat maximum.  Their
        # centroid is stable; choosing an arbitrary argmax pixel is not.
        candidates.append((score, float(xx.mean()), float(yy.mean())))

    # Stable spatial tie-breaking makes identical-score decoding reproducible.
    candidates.sort(key=lambda item: (-item[0], item[2], item[1]))
    kept: list[tuple[float, float, float]] = []
    radius_squared = float(nms_radius_px * nms_radius_px)
    for candidate in candidates:
        _, px, py = candidate
        if any(
            (px - kept_x) ** 2 + (py - kept_y) ** 2 <= radius_squared
            for _, kept_x, kept_y in kept
        ):
            continue
        kept.append(candidate)
        if len(kept) == max_peaks:
            break

    height, width = hm.shape
    x_scale = max(width - 1, 1)
    y_scale = max(height - 1, 1)
    return [
        TouchPeak(x=px / x_scale, y=py / y_scale, score=score)
        for score, px, py in kept
    ]


__all__ = ["TouchPeak", "extract_touch_peaks"]
