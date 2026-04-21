"""Utility to verify and tune spin button logical coordinates on-device."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from trueskate_ai.rl.device_worker import DEVICES, DeviceWorker


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tap candidate spin-button coordinates to verify timing/placement."
    )
    parser.add_argument("--x", type=float, default=25.0, help="Logical X coordinate.")
    parser.add_argument("--y", type=float, default=362.0, help="Logical Y coordinate.")
    parser.add_argument(
        "--device-index", type=int, default=0, help="Index into DeviceWorker DEVICES list."
    )
    parser.add_argument("--repeat", type=int, default=2, help="Number of taps to send.")
    parser.add_argument(
        "--interval", type=float, default=0.35, help="Seconds between taps."
    )
    args = parser.parse_args()

    if args.device_index < 0 or args.device_index >= len(DEVICES):
        raise SystemExit(f"Invalid --device-index {args.device_index}; expected 0..{len(DEVICES)-1}")

    worker = DeviceWorker(DEVICES[args.device_index])
    worker.connect()
    try:
        worker.ensure_foreground()
        for i in range(args.repeat):
            worker.driver.execute_script("mobile: tap", {"x": args.x, "y": args.y})
            print(f"Tap {i + 1}/{args.repeat} at ({args.x:.1f}, {args.y:.1f})")
            if i + 1 < args.repeat:
                time.sleep(args.interval)
    finally:
        worker.disconnect()


if __name__ == "__main__":
    main()

