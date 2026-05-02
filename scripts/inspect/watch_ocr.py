"""Live OCR monitor — run for 40 s and print every detected trick to terminal.

Usage:
    python scripts/watch_ocr.py
    python scripts/watch_ocr.py --duration 60 --fps 8
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from trueskate_ai.rl.device_worker import DEVICES, DeviceWorker
from trueskate_ai.sim.trick_info_reader import detect_trick_with_diagnostics, ensure_ocr_backend_ready


def main() -> None:
    parser = argparse.ArgumentParser(description="Live OCR monitor for True Skate.")
    parser.add_argument("--duration", type=float, default=40.0, help="How long to run in seconds.")
    parser.add_argument("--fps", type=float, default=6.0, help="Screenshot capture rate.")
    parser.add_argument("--device-name", type=str, default=None)
    args = parser.parse_args()

    ensure_ocr_backend_ready()

    device_cfg = next((d for d in DEVICES if d["name"] == args.device_name), DEVICES[0]) if args.device_name else DEVICES[0]
    worker = DeviceWorker(device_cfg)
    worker.connect()

    interval = 1.0 / args.fps
    deadline = time.monotonic() + args.duration
    last_trick = None
    frame_count = 0

    print(f"\nWatching OCR for {args.duration:.0f}s at {args.fps:.0f} fps — perform tricks now!\n")

    try:
        while time.monotonic() < deadline:
            t0 = time.monotonic()

            png = worker.driver.get_screenshot_as_png()
            arr = np.frombuffer(png, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            frame_count += 1

            result, diag = detect_trick_with_diagnostics(frame)

            remaining = deadline - time.monotonic()
            ts = f"[{args.duration - remaining:5.1f}s]"

            if result is not None:
                if result != last_trick:
                    status_icon = {"landed": "✓", "failed": "✗", "unknown": "?"}.get(result.status, "?")
                    print(f"{ts} {status_icon} {result.trick}  ({result.status})")
                    last_trick = result
            elif diag["anchor_found"]:
                print(f"{ts} anchor={diag['anchor_status']} but no OCR match  candidates={diag['ocr_candidates']}")

            elapsed = time.monotonic() - t0
            wait = interval - elapsed
            if wait > 0:
                time.sleep(wait)

    except KeyboardInterrupt:
        pass
    finally:
        worker.disconnect()

    print(f"\nDone. {frame_count} frames captured.")


if __name__ == "__main__":
    main()
