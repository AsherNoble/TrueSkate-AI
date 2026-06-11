"""Shared gesture execution utilities and constants.

Provides scale_to_device() — the sole coordinate transform in the pipeline —
and the static push gesture fired before every trick attempt. Coordinate and
gesture conventions are documented in GESTURES.md at the repo root.
"""
import time

from selenium.webdriver.common.action_chains import ActionChains

PUSH_PRE_DELAY: float = 0.5
PUSH_DURATION: float = 0.02
PUSH_EASING: float = 2.0
PUSH_START: tuple[float, float] = (
    0.7658,
    0.3044,
)
PUSH_END: tuple[float, float] = (
    0.7658,
    0.6797,
)

Y_BOUND_MIN: float = 0.12   # top of the valid RL gesture area
Y_BOUND_MAX: float = 0.88   # bottom of the valid RL gesture area (avoids home indicator)
X_BOUND_MIN: float = 0.12   # left edge hosts in-game buttons (spin ~x=0.06, clip/replay
X_BOUND_MAX: float = 1.0    # below it) — gestures grazing them open the replay-clip UI


def scale_to_device(x: float, y: float, device_w: float, device_h: float) -> tuple[float, float]:
    """Map a normalised coordinate point (x, y in [0, 1]) to a device's logical points."""
    return x * device_w, y * device_h


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

    push_start = scale_to_device(PUSH_START[0], PUSH_START[1], device_w, device_h)
    push_end = scale_to_device(PUSH_END[0], PUSH_END[1], device_w, device_h)
    push_easing = lambda t: t ** PUSH_EASING  # noqa: E731

    finger = make_touch_pointer("finger_push")
    build_curved_drag(finger, [push_start, push_end], total_duration=PUSH_DURATION, easing=push_easing)
    ActionChains(driver, devices=[finger]).perform()
    if on_post_push is not None:
        on_post_push()

    remaining = PUSH_PRE_DELAY - PUSH_DURATION
    if remaining > 0:
        time.sleep(remaining)
