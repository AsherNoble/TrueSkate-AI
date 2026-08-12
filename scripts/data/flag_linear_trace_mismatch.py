"""Non-destructively exclude MVP-2 clips whose orange trace contradicts its label.

The broad gameplay/menu guard cannot catch a subtler failure mode: an aligned
clip may contain an unrelated orange animation while its recorded command label
belongs to another touch event.  This script checks for the rendered orange
finger blob near the *recorded* start and end coordinates at the corresponding
timeline windows.  It writes a ``.trace_mismatch`` marker; no source video or
metadata is changed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from trueskate_ai.vision.basic_hold_dataset import _decode_frames
from trueskate_ai.vision.basic_linear_dataset import discover_basic_linear_samples


def _orange_delta_mask(frame: np.ndarray, reference: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    difference = np.abs(frame.astype(np.int16) - reference.astype(np.int16)).sum(axis=2)
    return ((hsv[:, :, 0] >= 5) & (hsv[:, :, 0] <= 28) & (hsv[:, :, 1] >= 80)
            & (hsv[:, :, 2] >= 90) & (difference >= 70))


def _has_orange_near(mask: np.ndarray, xy: tuple[float, float], radius: float) -> bool:
    height, width = mask.shape
    x, y = int(round(xy[0] * (width - 1))), int(round(xy[1] * (height - 1)))
    rx, ry = max(2, int(round(radius * width))), max(2, int(round(radius * height)))
    left, right = max(0, x - rx), min(width, x + rx + 1)
    top, bottom = max(0, y - ry), min(height, y + ry + 1)
    return bool(mask[top:bottom, left:right].any())


def _consistent(sample: Path, *, radius: float, start_window_s: float, end_window_s: float) -> tuple[bool, str]:
    meta = json.loads((sample / "meta.json").read_text())
    frames = _decode_frames(sample)
    if len(frames) < 8:
        return False, "too_few_frames"
    times = np.asarray(meta.get("frame_times", []), dtype=np.float32)
    if len(times) != len(frames):
        return False, "missing_frame_times"
    reference = np.mean(frames[:7], axis=0)
    masks = [_orange_delta_mask(frame, reference) for frame in frames]
    start, end = (tuple(float(v) for v in point) for point in meta["waypoints"])
    duration = float(meta["duration"])
    start_ok = any(_has_orange_near(mask, start, radius)
                   for mask, time in zip(masks, times) if abs(float(time)) <= start_window_s)
    end_ok = any(_has_orange_near(mask, end, radius)
                 for mask, time in zip(masks, times) if abs(float(time) - duration) <= end_window_s)
    if not start_ok:
        return False, "start_not_visible_at_label"
    if not end_ok:
        return False, "end_not_visible_at_label"
    return True, "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--radius", type=float, default=.065,
                        help="Normalised endpoint neighbourhood (default: %(default)s)")
    parser.add_argument("--start-window-s", type=float, default=.20)
    parser.add_argument("--end-window-s", type=float, default=.24)
    args = parser.parse_args()
    paths, _stats = discover_basic_linear_samples(args.data)
    if args.limit:
        paths = paths[:args.limit]
    counts: dict[str, int] = {}
    for sample in paths:
        ok, reason = _consistent(sample, radius=args.radius, start_window_s=args.start_window_s,
                                 end_window_s=args.end_window_s)
        counts[reason] = counts.get(reason, 0) + 1
        marker = sample / ".trace_mismatch"
        if not args.dry_run:
            if ok:
                marker.unlink(missing_ok=True)
            else:
                marker.write_text(f"MVP-2 label/trace consistency gate: {reason}\n")
    print(json.dumps({"samples": len(paths), "counts": counts,
                      "consistent_fraction": counts.get("ok", 0) / max(1, len(paths))}, indent=2))


if __name__ == "__main__":
    main()
