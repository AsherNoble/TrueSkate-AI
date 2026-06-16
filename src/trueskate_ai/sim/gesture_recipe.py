"""Gesture recipe execution for True Skate.

execute_gesture_recipe() fires the complete push + N-gesture sequence from a
decoded trick library recipe. It delegates to the SAME canonical execution the
CMA-ES eval path uses (sim.gestures.execute_static_push +
sim.touch_actions.execute_n_slot_gestures), so a replayed recipe reproduces
exactly what was optimized — including spin, which is a HOLD finger scheduled
inside one W3C payload (not a concurrent tap that cancels the in-flight gesture).
Gesture recipe schema and coordinate conventions are documented in GESTURES.md.
"""
import time

from trueskate_ai.sim.gestures import execute_static_push, scale_to_device
from trueskate_ai.sim.touch_actions import execute_n_slot_gestures, reset_position


def execute_gesture_recipe(
    driver,
    recipe: dict,
    *,
    device_w: float,
    device_h: float,
    spin_button_xy: tuple[float, float] | None = None,
    timing_device_key: str | None = None,
) -> None:
    """Fire the push + N-gesture sequence from a decoded recipe.

    Mirrors rl.cmaes.action_param.execute_gesture_params (the eval path): same
    push, same per-slot scheduling, same HOLD-finger spin — the only differences
    are that this starts from an already-decoded recipe and, being a standalone
    replay, resets the board afterwards.

    Args:
        recipe: Dict with 'gestures' (each {'points', 'duration', 'easing_power'}),
            'delays' (list of N-1 inter-gesture delays), and optionally 'spin'
            ({'enabled', 't_start', 't_end'}).
        spin_button_xy: normalised [0, 1] coords of the rotate button; required
            for an enabled spin block to fire (None → the spin block is a no-op).
        timing_device_key: device key for touch-timing calibration lookup.
    """
    gestures_points = []
    gestures_durations = []
    easings = []
    for g in recipe["gestures"]:
        gestures_points.append(
            [scale_to_device(x, y, device_w, device_h) for x, y in g["points"]]
        )
        gestures_durations.append(g["duration"])
        p = g["easing_power"]
        easings.append((lambda t, p=p: t ** p) if p != 1.0 else None)

    # Push — execute_static_push honours the PUSH_COUNT/PUSH_END_Y env overrides,
    # so replay pushes identically to eval (the old inline copy ignored them).
    execute_static_push(driver, device_w=device_w, device_h=device_h)

    # Spin is a HOLD control: scale the button to device points and let
    # execute_n_slot_gestures schedule the held finger inside the same W3C
    # payload as the drags. Fires only when enabled and a button coord is given.
    spin = recipe.get("spin")
    spin_button_pt = None
    if spin is not None and spin.get("enabled") and spin_button_xy is not None:
        spin_button_pt = scale_to_device(
            spin_button_xy[0], spin_button_xy[1], device_w, device_h
        )

    execute_n_slot_gestures(
        driver,
        gestures_points=gestures_points,
        gestures_durations=gestures_durations,
        delays=recipe["delays"],
        easings=easings,
        device_key=timing_device_key,
        spin=spin,
        spin_button_pt=spin_button_pt,
    )

    # Standalone replay: settle, then reset the board for the next attempt.
    time.sleep(0.7)
    reset_position(driver, device_w, device_h)
