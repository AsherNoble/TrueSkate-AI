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

    x0, y0, x1, y1, _duration = record["commanded"]
    px0, py0, px1, py1, _predicted_duration = record["predicted"]
    cv2.line(canvas, (int(x0 * width * _SCALE), int(y0 * height * _SCALE)),
             (int(x1 * width * _SCALE), int(y1 * height * _SCALE)), (0, 255, 0), 1)
    mark(x0, y0, (0, 255, 0), "S")
    mark(x1, y1, (0, 255, 0), "E")
    mark(px0, py0, (255, 255, 255), "s")
    mark(px1, py1, (255, 255, 255), "e")
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
