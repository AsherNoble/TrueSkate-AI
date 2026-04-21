"""Decode and execute trick-conditioned policy actions."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.actions.pointer_input import PointerInput

from trueskate_ai.rl.action_param import norm_to_device
from trueskate_ai.sim.touch_actions import build_curved_drag

_CANONICAL_W = 375.0
_CANONICAL_H = 812.0
_ACTION_DIM = 42
_SLOT_COUNT = 4

_TOP_BOUND_SCALE = 1.3
_BOTTOM_BAND_SCALE = 1.15
_X_MIN = 0.0
_X_MAX = _CANONICAL_W
_Y_MIN = _CANONICAL_H * ((448.0 * _TOP_BOUND_SCALE) / 896.0)
_Y_BASE_MAX = _CANONICAL_H * (750.0 / 896.0)
_Y_MAX = _Y_MIN + ((_Y_BASE_MAX - _Y_MIN) * _BOTTOM_BAND_SCALE)
_DURATION_MIN = 0.03
_DURATION_MAX = 0.8
_EASING_MIN = 0.3
_EASING_MAX = 3.0
_DELAY_MIN = -0.3
_DELAY_MAX = 0.8

_PUSH_PRE_DELAY = 0.5
_PUSH_DURATION = 0.02
_PUSH_EASING = 2.0
_PUSH_START = (
    _CANONICAL_W * (350.0 / 414.0),
    _CANONICAL_H * ((224.0 * _TOP_BOUND_SCALE) / 896.0),
)
_PUSH_END = (_CANONICAL_W * (350.0 / 414.0), _CANONICAL_H * (672.0 / 896.0))


def _map_from_unit(value: float, lo: float, hi: float) -> float:
    return lo + ((value + 1.0) * 0.5) * (hi - lo)


@dataclass(frozen=True)
class GestureSlot:
    points: list[tuple[float, float]]
    duration: float
    easing_power: float
    enabled: bool


@dataclass(frozen=True)
class SpinControl:
    enabled: bool
    t_start: float
    t_end: float


@dataclass(frozen=True)
class ActionPlan:
    slots: list[GestureSlot]
    delays: list[float]
    spin: SpinControl


def decode_action_vector(action: np.ndarray) -> ActionPlan:
    """Decode a normalized [-1, 1] action vector into a structured plan."""
    flat = np.asarray(action, dtype=np.float64).reshape(-1)
    if flat.shape[0] != _ACTION_DIM:
        raise ValueError(f"Expected {_ACTION_DIM} action values, got {flat.shape[0]}")

    slots: list[GestureSlot] = []
    idx = 0
    for _ in range(_SLOT_COUNT):
        x0 = _map_from_unit(float(flat[idx + 0]), _X_MIN, _X_MAX)
        y0 = _map_from_unit(float(flat[idx + 1]), _Y_MIN, _Y_MAX)
        x1 = _map_from_unit(float(flat[idx + 2]), _X_MIN, _X_MAX)
        y1 = _map_from_unit(float(flat[idx + 3]), _Y_MIN, _Y_MAX)
        x2 = _map_from_unit(float(flat[idx + 4]), _X_MIN, _X_MAX)
        y2 = _map_from_unit(float(flat[idx + 5]), _Y_MIN, _Y_MAX)
        duration = _map_from_unit(float(flat[idx + 6]), _DURATION_MIN, _DURATION_MAX)
        easing = _map_from_unit(float(flat[idx + 7]), _EASING_MIN, _EASING_MAX)
        enabled = float(flat[idx + 8]) >= 0.0
        slots.append(
            GestureSlot(
                points=[(x0, y0), (x1, y1), (x2, y2)],
                duration=duration,
                easing_power=easing,
                enabled=enabled,
            )
        )
        idx += 9

    delays = [
        _map_from_unit(float(flat[idx + 0]), _DELAY_MIN, _DELAY_MAX),
        _map_from_unit(float(flat[idx + 1]), _DELAY_MIN, _DELAY_MAX),
        _map_from_unit(float(flat[idx + 2]), _DELAY_MIN, _DELAY_MAX),
    ]
    idx += 3

    spin_enabled = float(flat[idx + 0]) >= 0.0
    t0 = np.clip(_map_from_unit(float(flat[idx + 1]), 0.0, 1.0), 0.0, 1.0)
    t1 = np.clip(_map_from_unit(float(flat[idx + 2]), 0.0, 1.0), 0.0, 1.0)
    spin = SpinControl(enabled=spin_enabled, t_start=min(t0, t1), t_end=max(t0, t1))

    return ActionPlan(slots=slots, delays=delays, spin=spin)


def _slot_starts(plan: ActionPlan) -> list[float]:
    starts = [0.0]
    for i in range(1, _SLOT_COUNT):
        prev = starts[i - 1] + plan.slots[i - 1].duration + plan.delays[i - 1]
        starts.append(max(0.0, prev))
    return starts


def _finger_pause(finger: PointerInput, pause_secs: float) -> None:
    if pause_secs > 0:
        finger.create_pause(pause_secs)


def _tap_at_time(driver, start_time: float, target_offset: float, tap_xy: tuple[float, float]) -> None:
    delay = max(0.0, (start_time + target_offset) - time.monotonic())
    if delay > 0:
        time.sleep(delay)
    driver.execute_script("mobile: tap", {"x": tap_xy[0], "y": tap_xy[1]})


def _execute_static_push(driver, *, device_w: float, device_h: float) -> None:
    """Run the CMA-ES-style static push before trick gestures."""
    push_start = norm_to_device(_PUSH_START[0], _PUSH_START[1], device_w, device_h)
    push_end = norm_to_device(_PUSH_END[0], _PUSH_END[1], device_w, device_h)
    push_easing = lambda t: t**_PUSH_EASING

    finger_push = PointerInput("touch", "finger_push")
    build_curved_drag(
        finger_push,
        [push_start, push_end],
        total_duration=_PUSH_DURATION,
        easing=push_easing,
    )
    ActionChains(driver, devices=[finger_push]).perform()

    remaining_pre_delay = _PUSH_PRE_DELAY - _PUSH_DURATION
    if remaining_pre_delay > 0:
        time.sleep(remaining_pre_delay)


def execute_action_plan(
    driver,
    plan: ActionPlan,
    *,
    device_w: float,
    device_h: float,
    spin_button_xy: tuple[float, float] = (25.0, 362.0),
) -> None:
    """Execute a decoded action plan on-device."""
    _execute_static_push(driver, device_w=device_w, device_h=device_h)

    starts = _slot_starts(plan)
    fingers = [PointerInput("touch", "finger0"), PointerInput("touch", "finger1")]
    finger_available = [0.0, 0.0]
    finger_has_actions = [False, False]
    has_gesture = False

    for slot_idx, slot in enumerate(plan.slots):
        if not slot.enabled:
            continue
        points = [norm_to_device(x, y, device_w, device_h) for x, y in slot.points]
        requested_start = starts[slot_idx]
        finger_idx = 0 if finger_available[0] <= finger_available[1] else 1
        actual_start = max(requested_start, finger_available[finger_idx])

        # WDA requires any pause to be preceded by pointerMove.
        # We position at slot start before optional waiting.
        fingers[finger_idx].create_pointer_move(
            x=points[0][0], y=points[0][1], duration=0
        )
        _finger_pause(fingers[finger_idx], actual_start - finger_available[finger_idx])

        p = slot.easing_power
        easing = (lambda t, power=p: t**power) if p != 1.0 else None
        build_curved_drag(
            fingers[finger_idx],
            points,
            total_duration=slot.duration,
            easing=easing,
        )
        finger_available[finger_idx] = actual_start + slot.duration
        finger_has_actions[finger_idx] = True
        has_gesture = True

    total_duration = max(finger_available + [0.01])

    if not has_gesture and plan.spin.enabled:
        spin_point = norm_to_device(spin_button_xy[0], spin_button_xy[1], device_w, device_h)
        time.sleep(plan.spin.t_start * total_duration)
        driver.execute_script("mobile: tap", {"x": spin_point[0], "y": spin_point[1]})
        time.sleep(max(0.0, (plan.spin.t_end - plan.spin.t_start) * total_duration))
        driver.execute_script("mobile: tap", {"x": spin_point[0], "y": spin_point[1]})
        return

    action_thread: threading.Thread | None = None
    if plan.spin.enabled and has_gesture:
        spin_point = norm_to_device(spin_button_xy[0], spin_button_xy[1], device_w, device_h)
        start_offset = plan.spin.t_start * total_duration
        end_offset = plan.spin.t_end * total_duration

        def _spin_runner(start_time: float) -> None:
            _tap_at_time(driver, start_time, start_offset, spin_point)
            _tap_at_time(driver, start_time, end_offset, spin_point)

        start_time = time.monotonic()
        action_thread = threading.Thread(target=_spin_runner, args=(start_time,), daemon=True)
        action_thread.start()

    if has_gesture:
        active_fingers = [fingers[i] for i in range(len(fingers)) if finger_has_actions[i]]
        ActionChains(driver, devices=active_fingers).perform()
    if action_thread is not None:
        action_thread.join(timeout=max(1.0, total_duration + 0.5))


def execute_action_vector(
    driver,
    action: np.ndarray,
    *,
    device_w: float,
    device_h: float,
    spin_button_xy: tuple[float, float] = (25.0, 362.0),
) -> ActionPlan:
    """Decode and execute a trick-conditioned action vector."""
    plan = decode_action_vector(action)
    execute_action_plan(
        driver,
        plan,
        device_w=device_w,
        device_h=device_h,
        spin_button_xy=spin_button_xy,
    )
    return plan
