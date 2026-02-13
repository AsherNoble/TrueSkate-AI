"""Debug visualization for validating labeler output.

Produces annotated videos or frame strips showing detected traces,
touch positions, and active/inactive indicators.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .trace_extractor import TouchState, TraceExtractor, TraceExtractorConfig

logger = logging.getLogger(__name__)


class LabelVisualizer:
    """Creates annotated debug visualizations from video + labels."""

    # Colors (BGR)
    COLOR_TOUCH1 = (0, 255, 0)       # green
    COLOR_TOUCH2 = (255, 100, 0)     # blue
    COLOR_INACTIVE = (128, 128, 128) # gray
    COLOR_MASK = (0, 120, 255)       # orange overlay
    COLOR_TEXT_BG = (0, 0, 0)
    COLOR_TEXT = (255, 255, 255)

    def __init__(self, config: Optional[TraceExtractorConfig] = None) -> None:
        self.config = config or TraceExtractorConfig()

    @staticmethod
    def load_labels_csv(csv_path: str | Path) -> list[TouchState]:
        """Load touch states from a CSV file produced by VideoLabeler."""
        states: list[TouchState] = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                states.append(TouchState(
                    frame_number=int(row["frame_number"]),
                    touch1_active=bool(int(row["touch1_active"])),
                    touch1_x=float(row["touch1_x"]),
                    touch1_y=float(row["touch1_y"]),
                    touch2_active=bool(int(row["touch2_active"])),
                    touch2_x=float(row["touch2_x"]),
                    touch2_y=float(row["touch2_y"]),
                    spin_control_active=bool(int(row.get("spin_control_active", 0))),
                ))
        return states

    def create_debug_video(
        self,
        video_path: str | Path,
        states: list[TouchState],
        output_path: str | Path,
        show_mask: bool = True,
    ) -> None:
        """Create an annotated debug video with trace overlays and touch markers.

        Args:
            video_path: Path to the original video.
            states: Touch states (from labeler or loaded CSV).
            output_path: Path for the output annotated video.
            show_mask: Whether to overlay the detected trace mask.
        """
        video_path = Path(video_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        extractor = TraceExtractor(self.config) if show_mask else None
        if extractor:
            extractor.reset()

        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num >= len(states):
                break

            state = states[frame_num]
            annotated = frame.copy()

            # Overlay trace mask if requested
            if show_mask and extractor and frame.mean() >= 1.0:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hud_mask = extractor._ensure_hud_mask(h, w)
                color_mask = extractor._extract_color_mask(hsv, hud_mask)

                mask_overlay = np.zeros_like(annotated)
                mask_overlay[color_mask > 0] = self.COLOR_MASK
                annotated = cv2.addWeighted(annotated, 0.7, mask_overlay, 0.3, 0)

            # Draw touch indicators
            self._draw_touch(annotated, state, w, h)

            # Draw info text
            self._draw_info(annotated, state, w, h)

            writer.write(annotated)
            frame_num += 1

        cap.release()
        writer.release()
        logger.info("Debug video saved to %s (%d frames)", output_path, frame_num)

    def create_frame_strip(
        self,
        video_path: str | Path,
        states: list[TouchState],
        output_path: str | Path,
        frame_indices: Optional[list[int]] = None,
        max_frames: int = 20,
        strip_height: int = 300,
    ) -> None:
        """Create a horizontal strip of annotated frames for quick inspection.

        Args:
            video_path: Path to the original video.
            states: Touch states.
            output_path: Path for output image (PNG/JPG).
            frame_indices: Specific frame indices to include. If None, samples evenly.
            max_frames: Maximum frames in the strip (when frame_indices is None).
            strip_height: Height of each frame in the strip.
        """
        video_path = Path(video_path)
        output_path = Path(output_path)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if frame_indices is None:
            step = max(1, total // max_frames)
            frame_indices = list(range(0, total, step))[:max_frames]

        scale = strip_height / h
        thumb_w = int(w * scale)

        strip = np.zeros((strip_height, thumb_w * len(frame_indices), 3), dtype=np.uint8)

        for i, fi in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret or fi >= len(states):
                continue

            state = states[fi]
            annotated = frame.copy()
            self._draw_touch(annotated, state, w, h)
            self._draw_info(annotated, state, w, h)

            thumb = cv2.resize(annotated, (thumb_w, strip_height))
            strip[:, i * thumb_w:(i + 1) * thumb_w] = thumb

        cap.release()
        cv2.imwrite(str(output_path), strip)
        logger.info("Frame strip saved to %s (%d frames)", output_path, len(frame_indices))

    def _draw_touch(self, frame: np.ndarray, state: TouchState, w: int, h: int) -> None:
        """Draw touch position circles on the frame."""
        radius = max(12, min(w, h) // 50)
        thickness = max(2, radius // 4)

        if state.touch1_active:
            px = int(state.touch1_x * w)
            py = int(state.touch1_y * h)
            cv2.circle(frame, (px, py), radius, self.COLOR_TOUCH1, thickness)
            cv2.circle(frame, (px, py), 3, self.COLOR_TOUCH1, -1)
        if state.touch2_active:
            px = int(state.touch2_x * w)
            py = int(state.touch2_y * h)
            cv2.circle(frame, (px, py), radius, self.COLOR_TOUCH2, thickness)
            cv2.circle(frame, (px, py), 3, self.COLOR_TOUCH2, -1)

    def _draw_info(self, frame: np.ndarray, state: TouchState, w: int, h: int) -> None:
        """Draw frame number and touch info as text overlay."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.4, w / 2000)
        thick = max(1, int(scale * 2))
        y_offset = int(30 * (h / 800))

        lines = [f"F{state.frame_number}"]
        if state.spin_control_active:
            lines.append("SPIN")
        if state.touch1_active:
            lines.append(f"T1: ({state.touch1_x:.3f}, {state.touch1_y:.3f})")
        if state.touch2_active:
            lines.append(f"T2: ({state.touch2_x:.3f}, {state.touch2_y:.3f})")
        if not state.touch1_active and not state.touch2_active:
            lines.append("No touch")

        for i, line in enumerate(lines):
            y = y_offset + i * int(y_offset * 1.2)
            # Background rectangle for readability
            (tw, th), _ = cv2.getTextSize(line, font, scale, thick)
            color = (0, 255, 255) if line == "SPIN" else self.COLOR_TEXT  # yellow for SPIN
            cv2.rectangle(frame, (5, y - th - 4), (10 + tw, y + 4), self.COLOR_TEXT_BG, -1)
            cv2.putText(frame, line, (8, y), font, scale, color, thick)
