"""Replay a gesture recipe from a trick library JSON on-device.

Connects to the phone via Appium, loads a trick library entry, and fires
the selected recipe (best or median) via the custom WDA endpoint
/wda/perform_trick_gestures — one HTTP call per gesture, Python-timed delays.

Usage:
    python scripts/inspect/execute_trick.py --library <json_path> [--mode best|median]

Trick library schema and gesture conventions: GESTURES.md at the repo root.
"""
import argparse
import json
import socket
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from trueskate_ai.rl.device_worker import DEVICES, DeviceWorker
from trueskate_ai.sim.gesture_recipe import execute_gesture_recipe


def _load_recipe(library_path: Path, mode: str) -> tuple[str, dict]:
    data = json.loads(library_path.read_text())
    trick = data.get("trick")
    if not trick:
        sys.exit(f"ERROR: no 'trick' field in {library_path}")
    key = f"{mode}_gestures"
    if key not in data:
        sys.exit(f"ERROR: key '{key}' not found in {library_path}")
    return trick, data[key]


def _is_wda_running(port: int, timeout: int = 2) -> bool:
    """Check if WDA is responding on the given port."""
    try:
        resp = requests.get(f"http://localhost:{port}/status", timeout=timeout)
        return resp.status_code == 200
    except (requests.exceptions.RequestException, socket.error):
        return False


def _get_active_devices() -> list[dict]:
    """Return list of devices with WDA currently running."""
    active = []
    for device in DEVICES:
        if _is_wda_running(device["wda_port"]):
            active.append(device)
    return active


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a trick library gesture recipe on-device.")
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--mode", choices=["best", "median"], default="median")
    args = parser.parse_args()

    if not args.library.exists():
        sys.exit(f"ERROR: library file not found: {args.library}")

    trick, recipe = _load_recipe(args.library, args.mode)
    print(f"Executing: trick='{trick}' mode={args.mode}")

    active_devices = _get_active_devices()
    if not active_devices:
        sys.exit("ERROR: No devices with WDA running. Start services with: python scripts/launch_services.py")

    print(f"Found {len(active_devices)} active device(s): {[d['name'] for d in active_devices]}")
    print()

    for device in active_devices:
        print(f"[{device['name']}] Executing...")
        worker = DeviceWorker(device, calibrate_touch_on_connect=False)
        worker.connect()
        try:
            execute_gesture_recipe(
                worker.driver,
                recipe,
                device_w=worker.device_w,
                device_h=worker.device_h,
                wda_url=f"http://127.0.0.1:{worker._cfg['wda_port']}",
                timing_device_key=worker.device_id,
            )
            print(f"[{device['name']}] Gestures fired.")
        finally:
            worker.disconnect()
        print()


if __name__ == "__main__":
    main()
