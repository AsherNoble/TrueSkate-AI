"""Replay a gesture recipe from a trick library JSON on-device.

Connects to the phone via Appium, loads a trick library entry, and fires
the selected gesture set (best or median) using the same push + two-finger
perform sequence as run_cmaes — but driven directly from decoded curved_drag
arguments rather than a raw parameter vector.

Usage:
    python experiments/rl_poc/execute_trick.py --library <json_path> --trick <name> [--mode best|median]
"""
import argparse
import json
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
for _p in [str(_HERE), str(_REPO_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.actions.pointer_input import PointerInput

from action_param import (
    APPIUM_LATENCY_OFFSET,
    _PUSH_DURATION,
    _PUSH_EASING,
    _PUSH_END,
    _PUSH_PRE_DELAY,
    _PUSH_START,
)
from run_cmaes import connect_driver
from trueskate_ai.sim.touch_actions import build_curved_drag


def _load_recipe(library_path: Path, trick: str, mode: str) -> dict:
    """Load and return the gesture recipe for *trick* from a library JSON."""
    data = json.loads(library_path.read_text())

    if data.get("trick", "").lower() != trick.lower():
        sys.exit(
            f"ERROR: library trick is '{data.get('trick')}', "
            f"not '{trick}'"
        )

    key = f"{mode}_gestures"
    if key not in data:
        sys.exit(f"ERROR: key '{key}' not found in {library_path}")

    return data[key]


def _execute_recipe(driver, recipe: dict) -> None:
    """Fire the push + two-finger gesture sequence from a decoded recipe."""
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
    parser.add_argument("--trick", type=str, required=True,
                        help="Trick name to look up (case-insensitive)")
    parser.add_argument("--mode", choices=["best", "median"], default="median",
                        help="Which gesture set to replay (default: median)")
    args = parser.parse_args()

    if not args.library.exists():
        sys.exit(f"ERROR: library file not found: {args.library}")

    recipe = _load_recipe(args.library, args.trick, args.mode)

    print(f"Executing: trick='{args.trick}' mode={args.mode}")
    driver, _ = connect_driver()

    try:
        _execute_recipe(driver, recipe)
        print("Gestures fired.")
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
