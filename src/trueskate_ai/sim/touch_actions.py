"""Touch actions for True Skate via Appium XCUITest driver.

All coordinates are in **logical points** (414x896 on iPhone 11).
If your model outputs pixel coordinates (828x1792), divide by the
device scale factor (2 for iPhone 11) before passing them here.

Usage in run_model.py:
    from trueskate_ai.sim.touch_actions import swipe, tap, pixels_to_points
    x, y = pixels_to_points(px_x, px_y, scale=2)
    tap(driver, x, y)
    swipe(driver, 100, 600, 100, 300, duration=0.5)
"""
import time

from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.actions.pointer_input import PointerInput

# iPhone 11: 828x1792 pixels, 414x896 points, @2x
DEFAULT_SCALE_FACTOR = 2


def pixels_to_points(px_x, px_y, *, scale=DEFAULT_SCALE_FACTOR):
    """Convert pixel coordinates to logical points for Appium."""
    return px_x / scale, px_y / scale


def tap(driver, x, y, *, pre_delay=0.0, post_delay=0.0):
    """Single tap at (x, y) in logical points."""
    if pre_delay:
        time.sleep(pre_delay)
    driver.execute_script('mobile: tap', {'x': x, 'y': y})
    if post_delay:
        time.sleep(post_delay)


def swipe(driver, start_x, start_y, end_x, end_y, *, duration=0.5):
    """Swipe from (start_x, start_y) to (end_x, end_y) in logical points.

    Uses mobile: dragFromToForDuration because mobile: swipe only
    accepts a direction string, not coordinates.
    """
    driver.execute_script('mobile: dragFromToForDuration', {
        'fromX': start_x,
        'fromY': start_y,
        'toX': end_x,
        'toY': end_y,
        'duration': duration,
    })


def long_press(driver, x, y, *, duration=1.0):
    """Press and hold at (x, y) in logical points. Duration in seconds."""
    driver.execute_script('mobile: touchAndHold', {
        'x': x,
        'y': y,
        'duration': duration,
    })


def flick(driver, start_x, start_y, end_x, end_y):
    """Fast flick gesture (kick / push) using a very short drag."""
    driver.execute_script('mobile: dragFromToForDuration', {
        'fromX': start_x,
        'fromY': start_y,
        'toX': end_x,
        'toY': end_y,
        'duration': 0.1,
    })


def double_tap(driver, x, y):
    """Double tap at (x, y) in logical points."""
    driver.execute_script('mobile: doubleTap', {'x': x, 'y': y})


def drag(driver, start_x, start_y, end_x, end_y, *, duration=1.0):
    """Slow drag — useful for board positioning and controlled movements."""
    driver.execute_script('mobile: dragFromToForDuration', {
        'fromX': start_x,
        'fromY': start_y,
        'toX': end_x,
        'toY': end_y,
        'duration': duration,
    })


def reset_position(driver):
    """Tap the reset button to return the board to its starting position."""
    driver.tap([(187, 50)])


def two_finger_tap(driver, x, y):
    """Two-finger tap at (x, y) in logical points."""
    driver.execute_script('mobile: twoFingerTap', {'x': x, 'y': y})


def _constant_easing(t):
    """Linear easing — constant velocity."""
    return t


def ease_in(t, power=2):
    """Accelerating easing: slow start, fast end. power=2 is quadratic."""
    return t ** power


def ease_out(t, power=2):
    """Decelerating easing: fast start, slow end."""
    return 1.0 - (1.0 - t) ** power


def ease_in_out(t, power=2):
    """Accelerate then decelerate (S-curve)."""
    if t < 0.5:
        return 0.5 * (2 * t) ** power
    return 1.0 - 0.5 * (2 * (1.0 - t)) ** power


def _easing_to_segment_durations(n_segments, total_duration_ms, easing):
    """Convert an easing function to per-segment durations in ms.

    The easing maps normalized progress [0,1] -> normalized time [0,1].
    We evaluate it at each segment boundary to get cumulative time
    fractions, then diff to get per-segment durations.
    """
    boundaries = [easing(i / n_segments) for i in range(n_segments + 1)]
    raw = [boundaries[i + 1] - boundaries[i] for i in range(n_segments)]
    # Normalize so they sum to exactly total_duration_ms
    raw_sum = sum(raw)
    durations = [max(1, int(d / raw_sum * total_duration_ms)) for d in raw]
    return durations


def curved_drag(driver, points, *, total_duration=0.5, easing=None):
    """Drag along a curved path defined by a sequence of (x, y) points.

    Uses W3C Actions API to chain pointer moves through intermediate
    waypoints.

    Args:
        points: list of (x, y) tuples in logical points — at least 2.
        total_duration: total gesture time in seconds.
        easing: optional function mapping [0,1] -> [0,1] that controls
            velocity profile. None = constant velocity. Use ease_in,
            ease_out, ease_in_out, or any custom callable.
            You can also pass a lambda for polynomial easing, e.g.:
                easing=lambda t: t**3        # cubic acceleration
                easing=lambda t: t**0.5      # sqrt deceleration
    """
    if len(points) < 2:
        raise ValueError("curved_drag needs at least 2 points")

    n_segments = len(points) - 1
    total_ms = int(total_duration * 1000)

    if easing is None:
        durations = [max(1, total_ms // n_segments)] * n_segments
    else:
        durations = _easing_to_segment_durations(n_segments, total_ms, easing)

    finger = PointerInput("touch", "finger")
    actions = ActionChains(driver, devices=[finger])

    x0, y0 = points[0]
    finger.create_pointer_move(x=x0, y=y0, duration=0)
    finger.create_pointer_down()

    for (x, y), dur in zip(points[1:], durations):
        finger.create_pointer_move(x=x, y=y, duration=dur)

    finger.create_pointer_up(0)
    actions.perform()


def build_curved_drag(finger, points, *, total_duration=0.5, easing=None):
    """Append curved-drag pointer actions to an existing PointerInput.

    Same path/easing logic as curved_drag(), but does not create its own
    PointerInput or call perform() — use this to compose multiple fingers
    into a single ActionChains.perform() call.

    Args:
        finger: PointerInput device to append actions to.
        points: list of (x, y) tuples in logical points — at least 2.
        total_duration: total gesture time in seconds.
        easing: optional function mapping [0,1] -> [0,1]. See curved_drag().
    """
    if len(points) < 2:
        raise ValueError("build_curved_drag needs at least 2 points")

    n_segments = len(points) - 1
    total_ms = int(total_duration * 1000)

    if easing is None:
        durations = [max(1, total_ms // n_segments)] * n_segments
    else:
        durations = _easing_to_segment_durations(n_segments, total_ms, easing)

    x0, y0 = points[0]
    finger.create_pointer_move(x=x0, y=y0, duration=0)
    finger.create_pointer_down()

    for (x, y), dur in zip(points[1:], durations):
        finger.create_pointer_move(x=x, y=y, duration=dur)

    finger.create_pointer_up(0)


