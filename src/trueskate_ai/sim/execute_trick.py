"""Replay a gesture recipe from a trick library JSON on-device.

Connects to the phone via Appium, loads a trick library entry, and fires
the selected gesture set (best or median) via a custom WDA endpoint that
executes both gestures as one XCSynthesizedEventRecord — device-native
timing, zero phantom touches between gestures.

Usage:
    python -m trueskate_ai.sim.execute_trick --library <json_path> [--mode best|median]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from selenium.webdriver.common.action_chains import ActionChains

from trueskate_ai.rl.action_param import (
    _PUSH_DURATION,
    _PUSH_EASING,
    _PUSH_END,
    _PUSH_PRE_DELAY,
    _PUSH_START,
    norm_to_device,
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


def _execute_two_gestures(g0_points, g1_points, g0_duration, g1_duration,
                          delay, easing0, easing1, wda_url):
    """Fire two gestures sequentially with Python-controlled inter-gesture delay.

    WDA's synthesizeEventWithRecord blocks until the gesture completes, so the
    requests.post call returns exactly when g0 finishes. We then sleep the
    remaining delay before firing g1. Each gesture is an independent touch
    sequence — no shared path, no phantom swipe between them.
    """
    t0 = time.perf_counter()
    _fire_gesture(g0_points, g0_duration, easing0, wda_url)
    elapsed = time.perf_counter() - t0

    remaining = delay - elapsed
    if remaining > 0:
        time.sleep(remaining)

    _fire_gesture(g1_points, g1_duration, easing1, wda_url)


def _load_recipe(library_path: Path, mode: str) -> tuple[str, dict]:
    data = json.loads(library_path.read_text())
    trick = data.get("trick")
    if not trick:
        sys.exit(f"ERROR: no 'trick' field in {library_path}")
    key = f"{mode}_gestures"
    if key not in data:
        sys.exit(f"ERROR: key '{key}' not found in {library_path}")
    return trick, data[key]


def execute_recipe(
    driver,
    recipe: dict,
    *,
    device_w: float,
    device_h: float,
    wda_url: str,
    timing_device_key: str | None = None,
) -> None:
    """Fire the push + two-gesture sequence from a decoded recipe."""
    g0, g1 = recipe["gestures"]
    delay = recipe["delays"][0]
    g0_points = [norm_to_device(x, y, device_w, device_h) for x, y in g0["points"]]
    g1_points = [norm_to_device(x, y, device_w, device_h) for x, y in g1["points"]]
    push_start = norm_to_device(_PUSH_START[0], _PUSH_START[1], device_w, device_h)
    push_end = norm_to_device(_PUSH_END[0], _PUSH_END[1], device_w, device_h)

    p0 = g0["easing_power"]
    easing0 = (lambda t, p=p0: t ** p) if p0 != 1.0 else None
    p1 = g1["easing_power"]
    easing1 = (lambda t, p=p1: t ** p) if p1 != 1.0 else None
    push_easing = lambda t: t ** _PUSH_EASING

    # --- Step 1: push ---
    finger2 = make_touch_pointer("finger2")
    build_curved_drag(finger2, [push_start, push_end], total_duration=_PUSH_DURATION, easing=push_easing)
    ActionChains(driver, devices=[finger2]).perform()

    remaining_pre_delay = _PUSH_PRE_DELAY - _PUSH_DURATION
    if remaining_pre_delay > 0:
        time.sleep(remaining_pre_delay)

    # --- Step 2: scoop + flick via custom WDA endpoint ---
    _execute_two_gestures(
        g0_points=g0_points,
        g1_points=g1_points,
        g0_duration=g0["duration"],
        g1_duration=g1["duration"],
        delay=delay,
        easing0=easing0,
        easing1=easing1,
        wda_url=wda_url,
    )

    time.sleep(0.7)
    reset_position(driver)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a trick library gesture recipe on-device.")
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--mode", choices=["best", "median"], default="median")
    args = parser.parse_args()

    if not args.library.exists():
        sys.exit(f"ERROR: library file not found: {args.library}")

    trick, recipe = _load_recipe(args.library, args.mode)
    print(f"Executing: trick='{trick}' mode={args.mode}")

    worker = DeviceWorker(DEVICES[0], calibrate_touch_on_connect=False)
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
        print("Gestures fired.")
    finally:
        worker.disconnect()


if __name__ == "__main__":
    main()
