"""Shared gesture constants and execution utilities for all RL pipelines."""
import time

from selenium.webdriver.common.action_chains import ActionChains

CANONICAL_W: float = 375.0
CANONICAL_H: float = 812.0

# Push constants — static board push executed before each trick attempt.
# Derived from original 414×896 design coordinates, scaled to canonical space.
_LEGACY_W = 414.0
_LEGACY_H = 896.0
PUSH_PRE_DELAY: float = 0.5
PUSH_DURATION: float = 0.02
PUSH_EASING: float = 2.0
PUSH_START: tuple[float, float] = (
    350.0 * CANONICAL_W / _LEGACY_W,
    224.0 * CANONICAL_H / _LEGACY_H,
)
PUSH_END: tuple[float, float] = (
    350.0 * CANONICAL_W / _LEGACY_W,
    672.0 * CANONICAL_H / _LEGACY_H,
)


def norm_to_device(x: float, y: float, device_w: float, device_h: float) -> tuple[float, float]:
    """Map a canonical-space point (375×812) into a device's logical points."""
    scale = device_w / CANONICAL_W
    action_h = CANONICAL_H * scale
    y_offset = (device_h - action_h) / 2.0
    return x * scale, y_offset + (y * scale)


def execute_static_push(
    driver,
    *,
    device_w: float,
    device_h: float,
    on_post_push=None,
) -> None:
    """Execute the static board push before trick gestures.

    Must be a separate Appium perform() call — bundling 3+ fingers in one
    perform() triggers iOS's system three-finger gesture (undo/redo),
    swallowing all touches before True Skate sees them.
    """
    from trueskate_ai.sim.touch_actions import build_curved_drag, make_touch_pointer  # noqa: PLC0415

    push_start = norm_to_device(PUSH_START[0], PUSH_START[1], device_w, device_h)
    push_end = norm_to_device(PUSH_END[0], PUSH_END[1], device_w, device_h)
    push_easing = lambda t: t ** PUSH_EASING  # noqa: E731

    finger = make_touch_pointer("finger_push")
    build_curved_drag(finger, [push_start, push_end], total_duration=PUSH_DURATION, easing=push_easing)
    ActionChains(driver, devices=[finger]).perform()
    if on_post_push is not None:
        on_post_push()

    remaining = PUSH_PRE_DELAY - PUSH_DURATION
    if remaining > 0:
        time.sleep(remaining)
