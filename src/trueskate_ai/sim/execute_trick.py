"""Replay a gesture recipe from a trick library JSON on-device.

Connects to the phone via Appium, loads a trick library entry, and fires
the selected gesture set (best or median) using the same push + two-finger
perform sequence as the CMA-ES loop — but driven directly from decoded
curved_drag arguments rather than a raw parameter vector.

Usage:
    python -m trueskate_ai.sim.execute_trick --library <json_path> [--mode best|median]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.actions.pointer_input import PointerInput

from trueskate_ai.rl.action_param import (
    APPIUM_LATENCY_OFFSET,
    _PUSH_DURATION,
    _PUSH_EASING,
    _PUSH_END,
    _PUSH_PRE_DELAY,
    _PUSH_START,
)
from trueskate_ai.rl.device_worker import DEVICES, DeviceWorker
from trueskate_ai.sim.touch_actions import build_curved_drag


def _load_recipe(library_path: Path, mode: str) -> tuple[str, dict]:
    """Load trick name and gesture recipe from a library JSON.

    Returns:
        (trick_name, recipe_dict)
    """
    data = json.loads(library_path.read_text())

    trick = data.get("trick")
    if not trick:
        sys.exit(f"ERROR: no 'trick' field in {library_path}")

    key = f"{mode}_gestures"
    if key not in data:
        sys.exit(f"ERROR: key '{key}' not found in {library_path}")

    return trick, data[key]


def execute_recipe(driver, recipe: dict) -> None:
    """Fire the push + two-finger gesture sequence from a decoded recipe.

    Args:
        driver: Appium WebDriver instance.
        recipe: Dict with "gestures" (list of 2) and "delays" (list of 1),
                as produced by build_trick_library.py.
    """
    g0, g1 = recipe["gestures"]
    delay = recipe["delays"][0]

    p0 = g0["easing_power"]
    easing0 = (lambda t, p=p0: t ** p) if p0 != 1.0 else None
    p1 = g1["easing_power"]
    easing1 = (lambda t, p=p1: t ** p) if p1 != 1.0 else None
    push_easing = lambda t: t ** _PUSH_EASING

    # --- Step 1: push (single-finger, separate perform) ---
    finger2 = PointerInput("touch", "finger2")
    build_curved_drag(
        finger2, [_PUSH_START, _PUSH_END],
        total_duration=_PUSH_DURATION, easing=push_easing,
    )
    ActionChains(driver, devices=[finger2]).perform()

    remaining_pre_delay = _PUSH_PRE_DELAY - _PUSH_DURATION
    if remaining_pre_delay > 0:
        time.sleep(remaining_pre_delay)

    # --- Step 2: scoop + flick (two-finger perform) ---
    finger0 = PointerInput("touch", "finger0")
    finger1 = PointerInput("touch", "finger1")

    # Points are stored as lists in JSON; convert to tuples for build_curved_drag
    g0_points = [tuple(p) for p in g0["points"]]
    g1_points = [tuple(p) for p in g1["points"]]

    build_curved_drag(finger0, g0_points, total_duration=g0["duration"], easing=easing0)

    finger1.create_pointer_move(x=g1_points[0][0], y=g1_points[0][1], duration=0)
    adjusted_delay = delay - APPIUM_LATENCY_OFFSET
    pause_secs = max(0.0, g0["duration"] + adjusted_delay)
    if pause_secs > 0:
        finger1.create_pause(pause_secs)

    build_curved_drag(finger1, g1_points, total_duration=g1["duration"], easing=easing1)

    ActionChains(driver, devices=[finger0, finger1]).perform()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a trick library gesture recipe on-device."
    )
    parser.add_argument("--library", type=Path, required=True,
                        help="Path to the trick library JSON")
    parser.add_argument("--mode", choices=["best", "median"], default="median",
                        help="Which gesture set to replay (default: median)")
    args = parser.parse_args()

    if not args.library.exists():
        sys.exit(f"ERROR: library file not found: {args.library}")

    trick, recipe = _load_recipe(args.library, args.mode)

    print(f"Executing: trick='{trick}' mode={args.mode}")
    worker = DeviceWorker(DEVICES[0])
    worker.connect()

    try:
        execute_recipe(worker.driver, recipe)
        print("Gestures fired.")
    finally:
        worker.disconnect()


if __name__ == "__main__":
    main()
