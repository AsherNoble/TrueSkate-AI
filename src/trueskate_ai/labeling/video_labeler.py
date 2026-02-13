"""Processes full video clips → labeled CSV/tensors.

Reads an MP4 file frame-by-frame, runs trace extraction, and outputs
per-frame touch state labels.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .trace_extractor import TouchState, TraceExtractor, TraceExtractorConfig

logger = logging.getLogger(__name__)


class VideoLabeler:
    """Processes a video file and produces per-frame touch labels."""

    def __init__(self, config: Optional[TraceExtractorConfig] = None) -> None:
        self.config = config or TraceExtractorConfig()
        self.extractor = TraceExtractor(self.config)

    def label_video(self, video_path: str | Path) -> list[TouchState]:
        """Process all frames in a video and return touch states.

        Args:
            video_path: Path to an MP4 video file.

        Returns:
            List of TouchState, one per frame.

        Raises:
            FileNotFoundError: If the video file does not exist.
            RuntimeError: If the video cannot be opened.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info("Processing %s: %d frames @ %.1f fps", video_path.name, total_frames, fps)

        self.extractor.reset()
        states: list[TouchState] = []
        frame_num = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip completely black frames
            if frame.mean() < 1.0:
                states.append(TouchState(frame_number=frame_num))
                logger.debug("Frame %d: black/empty, skipping", frame_num)
                frame_num += 1
                continue

            state = self.extractor.process_frame(frame, frame_num)
            states.append(state)

            if frame_num % 100 == 0 and frame_num > 0:
                logger.info("  Processed %d / %d frames", frame_num, total_frames)

            frame_num += 1

        cap.release()
        logger.info("Done: %d frames processed", len(states))
        return states

    @staticmethod
    def save_csv(states: list[TouchState], output_path: str | Path) -> None:
        """Save touch states to a CSV file.

        Args:
            states: List of TouchState from label_video().
            output_path: Path for the output CSV file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(TouchState.csv_header().split(","))
            for state in states:
                writer.writerow(state.as_row())

        logger.info("Labels saved to %s (%d rows)", output_path, len(states))

    @staticmethod
    def save_tensor(states: list[TouchState], output_path: str | Path) -> None:
        """Save touch states as a PyTorch tensor file (.pt).

        Tensor shape: (N, 7) where columns are:
        [touch1_active, touch1_x, touch1_y, touch2_active, touch2_x, touch2_y, spin_control_active]

        Args:
            states: List of TouchState from label_video().
            output_path: Path for the output .pt file.
        """
        import torch

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = np.array([
            [int(s.touch1_active), s.touch1_x, s.touch1_y,
             int(s.touch2_active), s.touch2_x, s.touch2_y,
             int(s.spin_control_active)]
            for s in states
        ], dtype=np.float32)

        tensor = torch.from_numpy(data)
        torch.save(tensor, str(output_path))
        logger.info("Tensor saved to %s (shape %s)", output_path, tensor.shape)


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry point for video labeling."""
    parser = argparse.ArgumentParser(
        description="Label True Skate video frames with touch positions.",
    )
    parser.add_argument("video", type=str, help="Path to input MP4 video")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output CSV path (default: <video_stem>_labels.csv)")
    parser.add_argument("--tensor", "-t", type=str, default=None,
                        help="Also save as PyTorch tensor (.pt)")
    parser.add_argument("--visualize", "-v", action="store_true",
                        help="Generate annotated debug video after labeling")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level (default: INFO)")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    video_path = Path(args.video)
    output_csv = Path(args.output) if args.output else video_path.with_name(f"{video_path.stem}_labels.csv")

    labeler = VideoLabeler()
    states = labeler.label_video(video_path)
    labeler.save_csv(states, output_csv)

    if args.tensor:
        labeler.save_tensor(states, args.tensor)

    if args.visualize:
        from .visualize import LabelVisualizer
        viz_output = video_path.with_name(f"{video_path.stem}_debug.mp4")
        visualizer = LabelVisualizer()
        visualizer.create_debug_video(video_path, states, viz_output)

    print(f"Labels written to {output_csv}")


if __name__ == "__main__":
    main()
