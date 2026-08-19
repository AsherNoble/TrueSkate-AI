"""Render held-out MVP-2 failures as annotated images for visual classification.

Consumes an ``autopsy_failures`` report plus locally fetched sample dirs and
writes one panel per failing clip: the clip's peak trail evidence with the
commanded and predicted endpoints marked.  Numbers say how far a prediction
missed; only looking says whether the trail was occluded, absent, or simply
misread.
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

from trueskate_ai.vision.basic_hold_dataset import _decode_frames  # noqa: E402

_SCALE = 3


def _trail_evidence(frames: list[np.ndarray]) -> np.ndarray:
    """Background-differenced orange response, max-projected over the clip."""
    stack = np.stack(frames).astype(np.float32) / 255.0
    reference = stack[:max(1, round(len(stack) * .22))].mean(axis=0, keepdims=True)
    motion = np.abs(stack - reference).mean(axis=3)
    blue, green, red = stack[..., 0], stack[..., 1], stack[..., 2]
    orange = (np.clip(red - green + .12, 0, None) * np.clip(green - blue + .12, 0, None)
              * np.clip(red - .20, 0, None) * motion)
    return orange.max(axis=0)


def _panel(sample: Path, record: dict) -> np.ndarray:
    frames = _decode_frames(sample)
    evidence = _trail_evidence(frames)
    height, width = evidence.shape
    normalised = evidence / max(float(evidence.max()), 1e-6)
    canvas = cv2.applyColorMap((normalised * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    canvas = cv2.resize(canvas, (width * _SCALE, height * _SCALE), interpolation=cv2.INTER_NEAREST)

    def mark(x: float, y: float, colour: tuple[int, int, int], glyph: str) -> None:
        point = (int(x * width * _SCALE), int(y * height * _SCALE))
        cv2.drawMarker(canvas, point, colour, cv2.MARKER_CROSS, 14, 2)
        cv2.putText(canvas, glyph, (point[0] + 6, point[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, .35, colour, 1, cv2.LINE_AA)

    # A k-knot record is 2K+1 wide, so unpacking five names breaks on k=3
    # autopsies.  Draw the whole polyline and label its first and last knot.
    commanded = record["commanded"]
    predicted = record["predicted"]
    knots = (len(commanded) - 1) // 2
    commanded_knots = [(commanded[2 * i], commanded[2 * i + 1]) for i in range(knots)]
    predicted_knots = [(predicted[2 * i], predicted[2 * i + 1]) for i in range(knots)]
    for (ax, ay), (bx, by) in zip(commanded_knots, commanded_knots[1:]):
        cv2.line(canvas, (int(ax * width * _SCALE), int(ay * height * _SCALE)),
                 (int(bx * width * _SCALE), int(by * height * _SCALE)), (0, 255, 0), 1)
    for index, ((cx, cy), (px, py)) in enumerate(zip(commanded_knots, predicted_knots)):
        if index == 0:
            commanded_glyph, predicted_glyph = "S", "s"
        elif index == knots - 1:
            commanded_glyph, predicted_glyph = "E", "e"
        else:
            commanded_glyph, predicted_glyph = str(index), str(index)
        mark(cx, cy, (0, 255, 0), commanded_glyph)
        mark(px, py, (255, 255, 255), predicted_glyph)
    caption = (f"end_err={record['end_error']:.3f} gap_end={record['trail_gap_end']:.3f} "
               f"{record['device']}")
    cv2.putText(canvas, caption, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, .38, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True, help="autopsy_failures JSON")
    parser.add_argument("--corpus", type=Path, required=True, help="local root holding the sample dirs")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    written = 0
    for record in report["failing_records"]:
        sample = args.corpus / record["sample"]
        if not sample.exists():
            print(f"missing {sample}")
            continue
        target = args.out / (record["sample"].replace("/", "__") + ".png")
        assert cv2.imwrite(str(target), _panel(sample, record))
        written += 1
    print(f"wrote {written} panels to {args.out}")


if __name__ == "__main__":
    main()
