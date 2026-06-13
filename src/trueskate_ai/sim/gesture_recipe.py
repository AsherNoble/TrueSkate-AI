"""Gesture recipe execution for True Skate.

execute_gesture_recipe() fires the complete push + N-gesture sequence
from a decoded trick library recipe. Gesture recipe schema and coordinate
conventions are documented in GESTURES.md at the repo root.
"""
import threading
import time

import requests
from selenium.webdriver.common.action_chains import ActionChains

from trueskate_ai.sim.gestures import (
    PUSH_DURATION,
    PUSH_EASING,
    PUSH_END,
    PUSH_PRE_DELAY,
    PUSH_START,
    scale_to_device,
)
from trueskate_ai.sim.touch_actions import (
    build_curved_drag,
    easing_to_segment_durations,
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
        durations = easing_to_segment_durations(n_segments, total_ms, easing)
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


def execute_gesture_recipe(
    driver,
    recipe: dict,
    *,
    device_w: float,
    device_h: float,
    wda_url: str,
    spin_button_xy: tuple[float, float] | None = None,
    timing_device_key: str | None = None,
) -> None:
    """Fire the push + N-gesture sequence from a decoded recipe.

    Args:
        recipe: Dict with 'gestures' (list of gesture dicts), 'delays' (list of
            N-1 delays), and optionally 'spin' ({'enabled', 't_start', 't_end'}).
        spin_button_xy: normalised [0, 1] coords of the rotate button; required
            for an enabled spin block to fire.
    """
    gestures = recipe["gestures"]
    delays = recipe["delays"]

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

    # --- Step 2: optional spin button, tapped on/off from a background thread
    # synchronized with the gesture window (estimated total = durations + gaps) ---
    spin = recipe.get("spin")
    spin_thread = None
    if spin and spin.get("enabled") and spin_button_xy is not None:
        sx, sy = scale_to_device(spin_button_xy[0], spin_button_xy[1], device_w, device_h)
        total = max(
            0.01,
            sum(g["duration"] for g in gestures) + sum(max(0.0, d) for d in delays),
        )
        start_offset = float(spin["t_start"]) * total
        end_offset = float(spin["t_end"]) * total

        def _spin_runner(t0, _s=start_offset, _e=end_offset):
            d0 = max(0.0, (t0 + _s) - time.monotonic())
            if d0 > 0:
                time.sleep(d0)
            driver.execute_script("mobile: tap", {"x": sx, "y": sy})
            d1 = max(0.0, (t0 + _e) - time.monotonic())
            if d1 > 0:
                time.sleep(d1)
            driver.execute_script("mobile: tap", {"x": sx, "y": sy})

        spin_thread = threading.Thread(
            target=_spin_runner, args=(time.monotonic(),), daemon=True
        )
        spin_thread.start()

    # --- Step 3: execute all gestures via custom WDA endpoint ---
    _execute_gestures(normalized_gestures, delays, wda_url)

    if spin_thread is not None:
        spin_thread.join(timeout=5.0)

    time.sleep(0.7)
    reset_position(driver, device_w, device_h)
