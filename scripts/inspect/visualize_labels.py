"""Visualise touch labels over extracted frames, producing a debug video.

Usage:
    python3 visualize_labels.py path/to/frames_labels.csv
    python3 visualize_labels.py path/to/frames_labels.csv --frames-dir path/to/frames/
    python3 visualize_labels.py path/to/frames_labels.csv --fps 30 --no-mask
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np

from trueskate_ai.labeling.trace_extractor import TraceExtractor, TraceExtractorConfig
from trueskate_ai.labeling.visualize import LabelVisualizer

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


def infer_frames_dir(csv_path: Path) -> Path:
    """Guess the frames directory from the CSV filename.

    extract_trace.py saves as <frames_dir>_labels.csv next to the frames dir,
    so strip the _labels.csv suffix to recover the original directory name.
    """
    name = csv_path.stem  # e.g. double_kickflip_003_60fps_labels
    if name.endswith("_labels"):
        name = name[: -len("_labels")]
    candidate = csv_path.parent / name
    if candidate.is_dir():
        return candidate
    print(f"Could not infer frames directory from {csv_path}.")
    print(f"Tried: {candidate}")
    print("Pass --frames-dir explicitly.")
    sys.exit(1)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualise CSV touch labels as an annotated debug video."
    )
    parser.add_argument("csv", help="Path to the labels CSV")
    parser.add_argument("--frames-dir", default=None,
                        help="Directory of frame images (inferred from CSV name if omitted)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output video path (default: <csv_stem>_debug.mp4 alongside CSV)")
    parser.add_argument("--fps", type=float, default=60.0,
                        help="Output video frame rate (default: 60)")
    parser.add_argument("--no-mask", action="store_true",
                        help="Skip trace mask overlay (faster)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        sys.exit(1)

    frames_dir = Path(args.frames_dir) if args.frames_dir else infer_frames_dir(csv_path)
    stem = csv_path.stem[: -len("_labels")] if csv_path.stem.endswith("_labels") else csv_path.stem
    output_path = Path(args.output) if args.output else csv_path.parent / f"{stem}_debug.mp4"

    states = LabelVisualizer.load_labels_csv(csv_path)
    print(f"Loaded {len(states)} labels from {csv_path}")

    frames = find_frames(frames_dir)
    print(f"Found {len(frames)} frames in {frames_dir}")

    first = cv2.imread(str(frames[0]))
    if first is None:
        print(f"Could not read first frame: {frames[0]}")
        sys.exit(1)
    h, w = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, args.fps, (w, h))

    viz = LabelVisualizer()
    extractor = None if args.no_mask else TraceExtractor(TraceExtractorConfig())
    if extractor:
        extractor.reset()

    COLOR_MASK = LabelVisualizer.COLOR_MASK

    for frame_num, frame_path in enumerate(frames):
        if frame_num >= len(states):
            break

        bgr = cv2.imread(str(frame_path))
        if bgr is None:
            logging.warning("Could not read %s, writing blank frame", frame_path.name)
            writer.write(np.zeros((h, w, 3), dtype=np.uint8))
            continue

        annotated = bgr.copy()
        state = states[frame_num]

        # Trace mask overlay
        if extractor and bgr.mean() >= 1.0:
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            hud_mask = extractor._ensure_hud_mask(h, w)
            color_mask = extractor._extract_color_mask(hsv, hud_mask)
            overlay = np.zeros_like(annotated)
            overlay[color_mask > 0] = COLOR_MASK
            annotated = cv2.addWeighted(annotated, 0.7, overlay, 0.3, 0)

        viz._draw_touch(annotated, state, w, h)
        viz._draw_info(annotated, state, w, h)
        writer.write(annotated)

        if frame_num % 100 == 0 and frame_num > 0:
            print(f"  Rendered {frame_num} / {len(frames)} frames")

    writer.release()
    print(f"Debug video saved to {output_path}")


if __name__ == "__main__":
    main()
