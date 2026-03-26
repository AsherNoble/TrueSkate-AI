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


def two_finger_tap(driver, x, y):
    """Two-finger tap at (x, y) in logical points."""
    driver.execute_script('mobile: twoFingerTap', {'x': x, 'y': y})


def curved_drag(driver, points, *, total_duration=0.5):
    """Drag along a curved path defined by a sequence of (x, y) points.

    Uses W3C Actions API to chain pointer moves through intermediate
    waypoints. Each segment gets an equal share of total_duration.

    Args:
        points: list of (x, y) tuples in logical points — at least 2.
        total_duration: total gesture time in seconds.
    """
    if len(points) < 2:
        raise ValueError("curved_drag needs at least 2 points")

    finger = PointerInput("touch", "finger")
    actions = ActionChains(driver, devices=[finger])

    # Move to start and press down
    x0, y0 = points[0]
    actions.w3c_actions.pointer_action.move_to_location(x0, y0)
    actions.w3c_actions.pointer_action.pointer_down()

    # Move through each subsequent point
    segment_ms = int(total_duration * 1000 / (len(points) - 1))
    for x, y in points[1:]:
        actions.w3c_actions.pointer_action.move_to_location(x, y, duration=segment_ms)

    actions.w3c_actions.pointer_action.pointer_up()
    actions.perform()


