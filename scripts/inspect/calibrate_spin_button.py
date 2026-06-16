"""Utility to verify and tune spin button normalised coordinates on-device.

Takes normalised [0, 1] coords (the spin_button_xy convention) and converts
them to this device's logical pixels via scale_to_device before tapping.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]  # scripts/inspect/ -> repo root
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from trueskate_ai.rl.device_worker import DEVICES, DeviceWorker
from trueskate_ai.sim.gestures import scale_to_device


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tap candidate spin-button coordinates to verify timing/placement."
    )
    # Normalised [0, 1] spin-button coords (matches DEVICES / spin_button_xy
    # convention). Converted to logical pixels per-device at tap time.
    parser.add_argument("--x", type=float, default=0.0604,
                        help="Normalised [0,1] X (spin_button_xy convention).")
    parser.add_argument("--y", type=float, default=0.4040,
                        help="Normalised [0,1] Y (spin_button_xy convention).")
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
        # Convert normalised coords to this device's logical pixels, the same
        # transform gesture execution uses, so calibration matches gameplay.
        px, py = scale_to_device(args.x, args.y, worker.device_w, worker.device_h)
        for i in range(args.repeat):
            worker.driver.execute_script("mobile: tap", {"x": px, "y": py})
            print(f"Tap {i + 1}/{args.repeat} at norm ({args.x:.4f}, {args.y:.4f}) "
                  f"-> logical ({px:.1f}, {py:.1f})")
            if i + 1 < args.repeat:
                time.sleep(args.interval)
    finally:
        worker.disconnect()


if __name__ == "__main__":
    main()

