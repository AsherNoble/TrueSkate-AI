"""Core CV pipeline: frame → touch state.

Extracts per-frame touch positions from True Skate's visual traces using
HSV color filtering, temporal differencing, and hotspot localization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TraceExtractorConfig:
    """Configuration for trace extraction parameters."""

    # HSV thresholds for orange/warm trace color
    hsv_lower: tuple[int, int, int] = (0, 50, 120)
    hsv_upper: tuple[int, int, int] = (35, 255, 255)

    # HUD mask boundaries (pixels, for 1170x2532 resolution)
    hud_top: int = 420
    hud_bottom_from: int = 1500  # mask from this row to bottom
    hud_left: int = 70

    # Morphological kernel size
    morph_kernel_size: int = 5

    # Active touch detection thresholds
    new_pixel_ratio_active: float = 0.1   # above this → active touch
    new_pixel_ratio_fade: float = 0.05    # below this → fading trace
    solidity_active: float = 0.55         # minimum solidity for active blob

    # Minimum blob area (pixels) to consider
    min_blob_area: int = 100

    # Warm hue upper bound for hotspot computation
    warm_hue_upper: int = 35
    warm_hue_lower_wrap: int = 170  # hues above this also count as warm


@dataclass
class TouchState:
    """Touch state for a single frame."""

    frame_number: int
    touch1_active: bool = False
    touch1_x: float = 0.0
    touch1_y: float = 0.0
    touch2_active: bool = False
    touch2_x: float = 0.0
    touch2_y: float = 0.0

    def as_row(self) -> list:
        """Return as a flat list for CSV output."""
        return [
            self.frame_number,
            int(self.touch1_active), self.touch1_x, self.touch1_y,
            int(self.touch2_active), self.touch2_x, self.touch2_y,
        ]

    @staticmethod
    def csv_header() -> str:
        return "frame_number,touch1_active,touch1_x,touch1_y,touch2_active,touch2_x,touch2_y"


@dataclass
class _BlobInfo:
    """Internal representation of a detected trace blob."""

    area: int
    centroid: tuple[float, float]  # (x, y) in pixel coords
    solidity: float
    mask: np.ndarray  # single-blob binary mask
    hotspot_peak: Optional[tuple[int, int]] = None  # (x, y) pixel coords
    new_pixel_ratio: float = 0.0
    is_active: bool = False


class TraceExtractor:
    """Extracts touch state from individual video frames.

    Maintains internal state (previous frame data) for temporal features.
    Frames must be fed sequentially via `process_frame()`.
    """

    def __init__(self, config: Optional[TraceExtractorConfig] = None) -> None:
        self.config = config or TraceExtractorConfig()
        self._prev_mask: Optional[np.ndarray] = None
        self._prev_touches: list[tuple[float, float]] = []  # previous frame's touch positions (x, y)
        self._hud_mask: Optional[np.ndarray] = None
        self._frame_h: int = 0
        self._frame_w: int = 0

    def reset(self) -> None:
        """Reset internal state for processing a new video."""
        self._prev_mask = None
        self._prev_touches = []
        self._hud_mask = None

    def _ensure_hud_mask(self, h: int, w: int) -> np.ndarray:
        """Create or return cached HUD mask for the given frame dimensions."""
        if self._hud_mask is not None and self._frame_h == h and self._frame_w == w:
            return self._hud_mask

        cfg = self.config
        # Scale HUD boundaries proportionally if resolution differs from reference (1170x2532)
        scale_y = h / 2532
        scale_x = w / 1170

        top = int(cfg.hud_top * scale_y)
        bottom_from = int(cfg.hud_bottom_from * scale_y)
        left = int(cfg.hud_left * scale_x)

        mask = np.ones((h, w), dtype=np.uint8) * 255
        mask[:top, :] = 0
        mask[bottom_from:, :] = 0
        mask[:, :left] = 0

        self._hud_mask = mask
        self._frame_h = h
        self._frame_w = w
        return mask

    def _extract_color_mask(self, hsv_frame: np.ndarray, hud_mask: np.ndarray) -> np.ndarray:
        """Extract warm-colored trace pixels via HSV filtering + morphological cleanup."""
        cfg = self.config
        lower = np.array(cfg.hsv_lower)
        upper = np.array(cfg.hsv_upper)

        color_mask = cv2.inRange(hsv_frame, lower, upper)
        color_mask = cv2.bitwise_and(color_mask, hud_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (cfg.morph_kernel_size, cfg.morph_kernel_size))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)

        return color_mask

    def _compute_hotspot(self, hsv_frame: np.ndarray, hud_mask: np.ndarray) -> np.ndarray:
        """Compute hotspot intensity map (brightness × saturation in warm hues)."""
        cfg = self.config
        v_ch = hsv_frame[:, :, 2].astype(np.float32)
        s_ch = hsv_frame[:, :, 1].astype(np.float32)
        h_ch = hsv_frame[:, :, 0].astype(np.float32)

        warm_mask = ((h_ch < cfg.warm_hue_upper) | (h_ch > cfg.warm_hue_lower_wrap)).astype(np.float32)
        hotspot = (v_ch / 255.0) * (s_ch / 255.0) * warm_mask
        hotspot[hud_mask == 0] = 0
        return hotspot

    def _find_blobs(self, color_mask: np.ndarray, hsv_frame: np.ndarray,
                    hotspot: np.ndarray) -> list[_BlobInfo]:
        """Find and characterize trace blobs using connected component analysis."""
        cfg = self.config
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(color_mask)

        blobs: list[_BlobInfo] = []
        for j in range(1, num_labels):
            area = stats[j, cv2.CC_STAT_AREA]
            if area < cfg.min_blob_area:
                continue

            cx, cy = centroids[j]
            blob_mask = (labels == j).astype(np.uint8)

            # Compute solidity
            contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            solidity = 0.0
            if contours:
                cnt = contours[0]
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = area / max(hull_area, 1)

            # Find hotspot peak within this blob
            blob_hotspot = hotspot.copy()
            blob_hotspot[blob_mask == 0] = 0
            peak_val = blob_hotspot.max()
            peak_pos = None
            if peak_val > 0:
                peak_idx = np.unravel_index(blob_hotspot.argmax(), blob_hotspot.shape)
                peak_pos = (int(peak_idx[1]), int(peak_idx[0]))  # (x, y)

            blobs.append(_BlobInfo(
                area=area,
                centroid=(cx, cy),
                solidity=solidity,
                mask=blob_mask,
                hotspot_peak=peak_pos,
            ))

        # Sort by area descending
        blobs.sort(key=lambda b: -b.area)
        return blobs

    def _classify_blobs(self, blobs: list[_BlobInfo], color_mask: np.ndarray) -> list[_BlobInfo]:
        """Determine which blobs represent active touches vs fading traces."""
        cfg = self.config

        for blob in blobs:
            if self._prev_mask is not None:
                # Compute new pixel ratio for this blob's region
                blob_current = cv2.bitwise_and(blob.mask * 255, color_mask)
                blob_prev = cv2.bitwise_and(blob.mask * 255, self._prev_mask)
                new_pixels = cv2.bitwise_and(blob_current, cv2.bitwise_not(blob_prev))
                new_count = int(np.sum(new_pixels > 0))
                blob_area = int(np.sum(blob_current > 0))
                blob.new_pixel_ratio = new_count / max(blob_area, 1)
            else:
                # First frame: assume active if blob is large enough
                blob.new_pixel_ratio = 1.0

            # Active if: new pixels are appearing AND shape is reasonably solid
            blob.is_active = (
                blob.new_pixel_ratio > cfg.new_pixel_ratio_fade
                and blob.solidity > cfg.solidity_active
            )

            # Stronger confidence: clearly active
            if blob.new_pixel_ratio > cfg.new_pixel_ratio_active:
                blob.is_active = True

        return blobs

    def _assign_touches(self, active_blobs: list[_BlobInfo],
                        h: int, w: int) -> tuple[list[bool], list[tuple[float, float]]]:
        """Assign active blobs to touch1/touch2 with temporal consistency."""
        if not active_blobs:
            return [], []

        # Get touch positions from hotspot peaks (or centroids as fallback)
        positions: list[tuple[float, float]] = []
        for blob in active_blobs[:2]:  # max 2 touches
            if blob.hotspot_peak is not None:
                px, py = blob.hotspot_peak
            else:
                px, py = blob.centroid
            # Normalize to [0, 1]
            nx = px / w
            ny = py / h
            positions.append((nx, ny))

        if len(positions) == 1:
            return [True], [positions[0]]

        # Two touches: assign using nearest-neighbor to previous frame
        if len(self._prev_touches) == 2:
            # Try both assignments, pick the one with lower total distance
            d_same = (self._dist(positions[0], self._prev_touches[0]) +
                      self._dist(positions[1], self._prev_touches[1]))
            d_swap = (self._dist(positions[0], self._prev_touches[1]) +
                      self._dist(positions[1], self._prev_touches[0]))
            if d_swap < d_same:
                positions = [positions[1], positions[0]]
        elif len(self._prev_touches) == 1:
            # One previous touch: assign closest to touch1
            d0 = self._dist(positions[0], self._prev_touches[0])
            d1 = self._dist(positions[1], self._prev_touches[0])
            if d1 < d0:
                positions = [positions[1], positions[0]]

        return [True] * len(positions), positions

    @staticmethod
    def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def process_frame(self, bgr_frame: np.ndarray, frame_number: int) -> TouchState:
        """Process a single BGR frame and return the detected touch state.

        Args:
            bgr_frame: The frame in BGR color space (as read by cv2.VideoCapture).
            frame_number: The 0-based frame index.

        Returns:
            TouchState with detected touch positions.
        """
        h, w = bgr_frame.shape[:2]
        hud_mask = self._ensure_hud_mask(h, w)

        hsv_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
        color_mask = self._extract_color_mask(hsv_frame, hud_mask)
        hotspot = self._compute_hotspot(hsv_frame, hud_mask)

        # Find and classify blobs
        blobs = self._find_blobs(color_mask, hsv_frame, hotspot)
        blobs = self._classify_blobs(blobs, color_mask)

        active_blobs = [b for b in blobs if b.is_active]

        # Assign to touch slots
        actives, positions = self._assign_touches(active_blobs, h, w)

        # Build touch state
        state = TouchState(frame_number=frame_number)
        if len(actives) >= 1 and actives[0]:
            state.touch1_active = True
            state.touch1_x = round(positions[0][0], 6)
            state.touch1_y = round(positions[0][1], 6)
        if len(actives) >= 2 and actives[1]:
            state.touch2_active = True
            state.touch2_x = round(positions[1][0], 6)
            state.touch2_y = round(positions[1][1], 6)

        # Update internal state
        self._prev_mask = color_mask
        self._prev_touches = positions if positions else []

        logger.debug(
            "Frame %d: t1=%s (%.3f, %.3f) t2=%s (%.3f, %.3f) blobs=%d active=%d",
            frame_number,
            state.touch1_active, state.touch1_x, state.touch1_y,
            state.touch2_active, state.touch2_x, state.touch2_y,
            len(blobs), len(active_blobs),
        )

        return state
