"""Extract per-frame touch labels from a directory of frames.

Usage:
    python3 extract_trace.py path/to/frames/
    python3 extract_trace.py path/to/frames/ --output labels.csv
    python3 extract_trace.py path/to/frames/ --log-level DEBUG
"""

import csv
import logging
import os
import sys
from pathlib import Path

# Allow running without installing the package or activating the venv
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2

from trueskate_ai.labeling.trace_extractor import TouchState, TraceExtractor

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def find_frames(frames_dir: Path) -> list[Path]:
    frames = sorted(
        p for p in frames_dir.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not frames:
        print(f"No image frames found in {frames_dir}")
        sys.exit(1)
    return frames


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract per-frame touch labels from a directory of frames."
    )
    parser.add_argument("frames_dir", help="Directory containing frame images")
    parser.add_argument("--output", "-o", default=None,
                        help="Output CSV path (default: <frames_dir>_labels.csv)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    frames_dir = Path(args.frames_dir)
    if not frames_dir.is_dir():
        print(f"Not a directory: {frames_dir}")
        sys.exit(1)

    output_csv = Path(args.output) if args.output else frames_dir.parent / f"{frames_dir.name}_labels.csv"

    frames = find_frames(frames_dir)
    print(f"Found {len(frames)} frames in {frames_dir}")

    extractor = TraceExtractor()
    states: list[TouchState] = []

    for frame_num, frame_path in enumerate(frames):
        bgr = cv2.imread(str(frame_path))
        if bgr is None:
            logging.warning("Could not read %s, skipping", frame_path.name)
            states.append(TouchState(frame_number=frame_num))
            continue

        state = extractor.process_frame(bgr, frame_num)
        states.append(state)

        if frame_num % 100 == 0 and frame_num > 0:
            print(f"  Processed {frame_num} / {len(frames)} frames")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(TouchState.csv_header().split(","))
        for state in states:
            writer.writerow(state.as_row())

    print(f"Labels written to {output_csv} ({len(states)} rows)")


if __name__ == "__main__":
    main()
