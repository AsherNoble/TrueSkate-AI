"""PPO gesture parameterization: decode a policy output vector and execute it on device.

Decodes a normalised [-1, 1] action vector into a GestureRecipe (4 gated gesture slots,
inter-slot delays, spin control) and executes it via Appium/WDA. Gesture coordinates are
normalised [0, 1] throughout; scale_to_device() converts to logical points at execution.

Gesture and coordinate conventions: GESTURES.md at the repo root.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trueskate_ai.sim.gestures import (
    DEFAULT_SPIN_BUTTON_XY,
    Y_BOUND_MAX,
    Y_BOUND_MIN,
    execute_static_push,
    scale_to_device,
)
from trueskate_ai.sim.touch_actions import (
    PointerInput,
    build_curved_drag,
    make_touch_pointer,
    perform_pointer_actions,
)

_ACTION_DIM = 42
_SLOT_COUNT = 4

_X_MIN = 0.0
_X_MAX = 1.0
_Y_MIN = Y_BOUND_MIN
_Y_MAX = Y_BOUND_MAX
_DURATION_MIN = 0.03
_DURATION_MAX = 3
_EASING_MIN = 0.3
_EASING_MAX = 3.0
_DELAY_MIN = -0.3
_DELAY_MAX = 2


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
class GestureRecipe:
    slots: list[GestureSlot]
    delays: list[float]
    spin: SpinControl


def decode_gesture_params(action: np.ndarray) -> GestureRecipe:
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

    return GestureRecipe(slots=slots, delays=delays, spin=spin)


def _slot_starts(recipe: GestureRecipe) -> list[float]:
    starts = [0.0]
    for i in range(1, _SLOT_COUNT):
        prev = starts[i - 1] + recipe.slots[i - 1].duration + recipe.delays[i - 1]
        starts.append(max(0.0, prev))
    return starts


def _finger_pause(finger: PointerInput, pause_secs: float) -> None:
    if pause_secs > 0:
        finger.create_pause(pause_secs)


def execute_gesture_recipe(
    driver,
    recipe: GestureRecipe,
    *,
    device_w: float,
    device_h: float,
    spin_button_xy: tuple[float, float] = DEFAULT_SPIN_BUTTON_XY,
    on_post_push=None,
) -> None:
    """Execute a decoded gesture recipe on-device.

    spin_button_xy: normalised [0, 1] coords for the spin/rotation button.
    """
    execute_static_push(driver, device_w=device_w, device_h=device_h, on_post_push=on_post_push)

    starts = _slot_starts(recipe)
    fingers = [make_touch_pointer("finger0"), make_touch_pointer("finger1")]
    finger_available = [0.0, 0.0]
    finger_has_actions = [False, False]

    for slot_idx, slot in enumerate(recipe.slots):
        if not slot.enabled:
            continue
        points = [scale_to_device(x, y, device_w, device_h) for x, y in slot.points]
        requested_start = starts[slot_idx]
        finger_idx = 0 if finger_available[0] <= finger_available[1] else 1
        actual_start = max(requested_start, finger_available[finger_idx])

        # Position before waiting to keep WDA pointer sequencing stable, then
        # RE-ISSUE the start move via include_start_move=True: WDA drops a
        # standalone zero-duration move when a pause follows it, so pointer_down
        # would otherwise fire at the pointer origin — the spurious top-left
        # swipe that opens the menu (same fix as execute_n_slot_gestures).
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
            include_start_move=True,
        )
        finger_available[finger_idx] = actual_start + slot.duration
        finger_has_actions[finger_idx] = True

    total_duration = max(finger_available + [0.01])

    # Spin is a HOLD control: a dedicated finger is held DOWN on the rotate button
    # from t_start*total to t_end*total, scheduled INSIDE the same single W3C
    # perform() as the drags. (A prior design fired `mobile: tap` from a background
    # thread concurrent with the perform — that cancelled the in-flight gesture on
    # the shared WDA session and nullified the trick; see rl_poc journal 2026-06-14.
    # This mirrors touch_actions.execute_n_slot_gestures' held-finger spin path.)
    active_fingers = [fingers[i] for i in range(len(fingers)) if finger_has_actions[i]]
    if recipe.spin.enabled:
        bx, by = scale_to_device(spin_button_xy[0], spin_button_xy[1], device_w, device_h)
        start_offset = max(0.0, recipe.spin.t_start * total_duration)
        hold = max(0.0, (recipe.spin.t_end - recipe.spin.t_start) * total_duration)
        sf = make_touch_pointer("spin")
        # Position, wait, then RE-ISSUE the move before pointer_down (same WDA
        # dropped-move workaround as the slots above). Kept as a verbatim mirror
        # of touch_actions' spin block — deliberately not _finger_pause — so the
        # two implementations stay diffable against each other.
        sf.create_pointer_move(x=bx, y=by, duration=0)
        if start_offset > 0:
            sf.create_pause(start_offset)
        sf.create_pointer_move(x=bx, y=by, duration=0)
        sf.create_pointer_down()
        if hold > 0:
            sf.create_pause(hold)
        sf.create_pointer_up(0)
        active_fingers.append(sf)

    if active_fingers:
        perform_pointer_actions(driver, active_fingers)


def execute_gesture_params_vector(
    driver,
    action: np.ndarray,
    *,
    device_w: float,
    device_h: float,
    spin_button_xy: tuple[float, float] = DEFAULT_SPIN_BUTTON_XY,
    on_post_push=None,
) -> GestureRecipe:
    """Decode a normalised [-1, 1] gesture parameter vector and execute it on-device."""
    recipe = decode_gesture_params(action)
    execute_gesture_recipe(
        driver,
        recipe,
        device_w=device_w,
        device_h=device_h,
        spin_button_xy=spin_button_xy,
        on_post_push=on_post_push,
    )
    return recipe
