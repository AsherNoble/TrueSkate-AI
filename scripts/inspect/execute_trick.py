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

from selenium.webdriver.common.action_chains import ActionChains

from trueskate_ai.rl.gestures import (
    PUSH_DURATION,
    PUSH_EASING,
    PUSH_END,
    PUSH_PRE_DELAY,
    PUSH_START,
    scale_to_device,
)
from trueskate_ai.rl.device_worker import DEVICES, DeviceWorker
from trueskate_ai.sim.touch_actions import (
    _easing_to_segment_durations,
    build_curved_drag,
    make_touch_pointer,
    reset_position,
)


def _wda_waypoints(points, total_duration, easing):
    """Convert gesture points + easing into WDA waypoint dicts (duration_ms per segment)."""
    n_segments = len(points) - 1
    total_ms = int(total_duration * 1000)
    if easing is None:
        durations = [max(1, total_ms // n_segments)] * n_segments
    else:
        durations = _easing_to_segment_durations(n_segments, total_ms, easing)
    x0, y0 = points[0]
    waypoints = [{"x": round(x0), "y": round(y0), "duration_ms": 0}]
    for (x, y), dur in zip(points[1:], durations):
        waypoints.append({"x": round(x), "y": round(y), "duration_ms": dur})
    return waypoints


def _fire_gesture(points, duration, easing, wda_url):
    """Fire a single gesture via the WDA endpoint. Blocks until gesture completes."""
    payload = {"gestures": [{"waypoints": _wda_waypoints(points, duration, easing)}]}
    resp = requests.post(f"{wda_url.rstrip('/')}/wda/perform_trick_gestures",
                         json=payload, timeout=15)
    resp.raise_for_status()


def _execute_gestures(gestures_data: list, delays: list, wda_url: str):
    """Fire multiple gestures sequentially with Python-controlled inter-gesture delays.

    WDA's synthesizeEventWithRecord blocks until the gesture completes, so the
    requests.post call returns exactly when one gesture finishes. We then sleep the
    remaining delay before firing the next. Each gesture is an independent touch
    sequence — no shared path, no phantom swipe between them.

    Args:
        gestures_data: List of gesture dicts, each with 'points', 'duration', 'easing_power'
        delays: List of N-1 delays (delay after gesture i before gesture i+1)
        wda_url: WDA base URL
    """
    for i, gesture in enumerate(gestures_data):
        points = gesture["points"]
        duration = gesture["duration"]
        easing_power = gesture["easing_power"]
        easing = (lambda t, p=easing_power: t ** p) if easing_power != 1.0 else None

        t0 = time.perf_counter()
        _fire_gesture(points, duration, easing, wda_url)

        if i < len(delays):
            elapsed = time.perf_counter() - t0
            remaining = delays[i] - elapsed
            if remaining > 0:
                time.sleep(remaining)


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


def execute_recipe(
    driver,
    recipe: dict,
    *,
    device_w: float,
    device_h: float,
    wda_url: str,
    timing_device_key: str | None = None,
) -> None:
    """Fire the push + N-gesture sequence from a decoded recipe.
    
    Args:
        recipe: Dict with 'gestures' (list of gesture dicts) and 'delays' (list of N-1 delays)
    """
    gestures = recipe["gestures"]
    delays = recipe["delays"]

    # Normalize gesture points to device coordinates
    normalized_gestures = []
    for gesture in gestures:
        normalized_gesture = dict(gesture)
        normalized_gesture["points"] = [
            scale_to_device(x, y, device_w, device_h) for x, y in gesture["points"]
        ]
        normalized_gestures.append(normalized_gesture)

    # --- Step 1: push ---
    push_start = scale_to_device(PUSH_START[0], PUSH_START[1], device_w, device_h)
    push_end = scale_to_device(PUSH_END[0], PUSH_END[1], device_w, device_h)
    push_easing = lambda t: t ** PUSH_EASING  # noqa: E731
    finger2 = make_touch_pointer("finger2")
    build_curved_drag(finger2, [push_start, push_end], total_duration=PUSH_DURATION, easing=push_easing)
    ActionChains(driver, devices=[finger2]).perform()

    remaining_pre_delay = PUSH_PRE_DELAY - PUSH_DURATION
    if remaining_pre_delay > 0:
        time.sleep(remaining_pre_delay)

    # --- Step 2: execute all gestures via custom WDA endpoint ---
    _execute_gestures(normalized_gestures, delays, wda_url)

    time.sleep(0.7)
    reset_position(driver, device_w, device_h)


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
            execute_recipe(
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
