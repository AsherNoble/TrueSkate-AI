"""Measure how directly the rendered orange line reveals MVP-2 endpoints.

This is a read-only diagnostic: it never alters clips, labels, or models.  It
uses the known pre-touch prefix as background and evaluates a few robust colour
mask reductions against the recorded endpoints.
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
    # Calibrated real clips place the touch trace around hue 7..23.  Saturation
    # plus a background-difference floor rejects static orange game art.
    return ((hsv[:, :, 0] >= 5) & (hsv[:, :, 0] <= 28) & (hsv[:, :, 1] >= 80)
            & (hsv[:, :, 2] >= 90) & (difference >= 70))


def _largest_component(mask: np.ndarray) -> tuple[int, np.ndarray] | None:
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    candidates = [(stats[index, cv2.CC_STAT_AREA], centroids[index]) for index in range(1, count)
                  if stats[index, cv2.CC_STAT_AREA] >= 3]
    if not candidates:
        return None
    area, xy = max(candidates, key=lambda item: item[0])
    return int(area), np.asarray(xy, dtype=np.float32)


def _trace_endpoints(masks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray] | None:
    """Track the large, continuously moving orange component until it fades."""
    candidates = [_largest_component(mask) for mask in masks]
    # The finger blob covers hundreds of source pixels; tiny stationary orange
    # changes from the board/UI are eliminated by this threshold.  Find the
    # contiguous high-area run with the greatest physical displacement.
    runs: list[list[tuple[int, np.ndarray]]] = []
    current: list[tuple[int, np.ndarray]] = []
    for candidate in candidates:
        if candidate is not None and candidate[0] >= 250:
            current.append(candidate)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    if not runs:
        return None
    run = max(runs, key=lambda values: float(np.linalg.norm(values[-1][1] - values[0][1])))
    if len(run) < 2:
        return None
    start = run[0][1]
    # The final fade frame can slightly pull the blob centroid back.  The farthest
    # tracked location is the most stable visual estimate of liftoff.
    end = max((xy for _area, xy in run), key=lambda xy: float(np.linalg.norm(xy - start)))
    return start, end


def _error(prediction: np.ndarray | None, target: np.ndarray, width: int, height: int) -> float | None:
    if prediction is None:
        return None
    return float(np.linalg.norm(prediction / np.array([width - 1, height - 1]) - target))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 means all strict clips")
    parser.add_argument("--examples", type=int, default=0,
                        help="Include up to this many per-clip errors in the JSON report")
    args = parser.parse_args()
    paths, _stats = discover_basic_linear_samples(args.data)
    if args.limit:
        paths = paths[:args.limit]
    start_errors: list[float] = []
    end_errors: list[float] = []
    missed = 0
    examples: list[dict] = []
    for sample in paths:
        meta = json.loads((sample / "meta.json").read_text())
        frames = _decode_frames(sample)
        reference = np.mean(frames[:7], axis=0)
        times = np.asarray(meta["frame_times"], dtype=np.float32)
        endpoints = _trace_endpoints([_orange_delta_mask(frame, reference) for frame in frames])
        if endpoints is None:
            missed += 1
            continue
        width, height = frames[0].shape[1], frames[0].shape[0]
        start_xy, end_xy = endpoints
        start = _error(start_xy, np.asarray(meta["waypoints"][0]), width, height)
        end = _error(end_xy, np.asarray(meta["waypoints"][1]), width, height)
        if start is None or end is None:
            missed += 1
        else:
            start_errors.append(start)
            end_errors.append(end)
            if len(examples) < args.examples:
                examples.append({"sample": str(sample), "start_error": start, "end_error": end,
                                 "duration": float(meta["duration"])})
    print(json.dumps({
        "samples": len(paths), "matched": len(start_errors), "missed": missed,
        "start_median": float(np.median(start_errors)) if start_errors else None,
        "end_median": float(np.median(end_errors)) if end_errors else None,
        "start_p90": float(np.quantile(start_errors, .9)) if start_errors else None,
        "end_p90": float(np.quantile(end_errors, .9)) if end_errors else None,
        "both_within_0.03": float(np.mean((np.asarray(start_errors) <= .03)
                                           & (np.asarray(end_errors) <= .03))) if start_errors else None,
        "examples": examples,
    }, indent=2))


if __name__ == "__main__":
    main()
