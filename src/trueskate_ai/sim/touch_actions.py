"""Low-level Appium gesture primitives for True Skate.

build_curved_drag() and make_touch_pointer() are the core building blocks;
callers compose them into multi-finger perform() payloads. All coordinates
accepted here are device logical points — callers are responsible for scaling
from normalised [0, 1] via scale_to_device() (see sim/gestures.py) before
passing points to these functions.

Gesture and coordinate conventions: GESTURES.md at the repo root.
"""
import logging
import time
from dataclasses import dataclass
from itertools import count
from statistics import median

from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.actions.pointer_input import PointerInput

_POINTER_SEQ = count()
_DEFAULT_COMBINED_THRESHOLD = 0.6
_MIN_COMBINED_THRESHOLD = 0.45
_MAX_COMBINED_THRESHOLD = 0.75
_MAX_SEQUENTIAL_COMPENSATION = 0.25


@dataclass(frozen=True)
class TouchTimingCalibration:
    """Per-device/session timing calibration for two-slot gesture scheduling."""
    sequential_overhead_s: float
    combined_nonneg_threshold_s: float
    source: str = "default"


_DEFAULT_CALIBRATION = TouchTimingCalibration(
    sequential_overhead_s=0.0,
    combined_nonneg_threshold_s=_DEFAULT_COMBINED_THRESHOLD,
    source="default",
)
_TIMING_CALIBRATIONS: dict[str, TouchTimingCalibration] = {}


def make_touch_pointer(prefix="finger"):
    """Create a unique touch pointer id to avoid stale-id state collisions."""
    return PointerInput("touch", f"{prefix}_{next(_POINTER_SEQ)}")


def _clamp(value, low, high):
    return max(low, min(high, value))


def get_touch_timing_calibration(device_key=None) -> TouchTimingCalibration:
    """Return session calibration for a device, or defaults when unavailable."""
    if not device_key:
        return _DEFAULT_CALIBRATION
    return _TIMING_CALIBRATIONS.get(device_key, _DEFAULT_CALIBRATION)


def set_touch_timing_calibration(device_key, calibration: TouchTimingCalibration) -> None:
    """Persist per-device/session timing calibration."""
    if not device_key:
        return
    _TIMING_CALIBRATIONS[device_key] = calibration


def tap(driver, x, y, *, pre_delay=0.0, post_delay=0.0):
    """Single tap at (x, y) in logical points."""
    if pre_delay:
        time.sleep(pre_delay)
    driver.execute_script('mobile: tap', {'x': x, 'y': y})
    if post_delay:
        time.sleep(post_delay)


def long_press(driver, x, y, *, duration=1.0):
    """Press and hold at (x, y) in logical points. Duration in seconds."""
    finger = make_touch_pointer("press")
    actions = ActionChains(driver, devices=[finger])
    finger.create_pointer_move(x=x, y=y, duration=0)
    finger.create_pointer_down()
    finger.create_pause(duration)
    finger.create_pointer_up(0)
    actions.perform()


def double_tap(driver, x, y):
    """Double tap at (x, y) in logical points."""
    driver.execute_script('mobile: doubleTap', {'x': x, 'y': y})


def reset_position(driver, device_w: float, device_h: float):
    """Tap the reset button to return the board to its starting position."""
    driver.tap([(0.5 * device_w, 0.0558 * device_h)])


def skip_loading_screen(driver, device_w: float, device_h: float, *, duration: float = 1.0):
    """Dismiss the True Skate loading screen by holding at normalised position (0.8454, 0.8393).

    When the app relaunches after being backgrounded, a loading screen appears.
    Uses a press-and-hold because a single tap is sometimes insufficient.
    """
    long_press(driver, 0.8454 * device_w, 0.8393 * device_h, duration=duration)


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


def easing_to_segment_durations(n_segments, total_duration_ms, easing):
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
        durations = easing_to_segment_durations(n_segments, total_ms, easing)

    finger = make_touch_pointer("finger")
    actions = ActionChains(driver, devices=[finger])

    x0, y0 = points[0]
    finger.create_pointer_move(x=x0, y=y0, duration=0)
    finger.create_pointer_down()

    for (x, y), dur in zip(points[1:], durations):
        finger.create_pointer_move(x=x, y=y, duration=dur)

    finger.create_pointer_up(0)
    actions.perform()


def build_curved_drag(
    finger,
    points,
    *,
    total_duration=0.5,
    easing=None,
    include_start_move=True,
):
    """Append curved-drag pointer actions to an existing PointerInput.

    Same path/easing logic as curved_drag(), but does not create its own
    PointerInput or call perform() — use this to compose multiple fingers
    into a single ActionChains.perform() call.

    Args:
        finger: PointerInput device to append actions to.
        points: list of (x, y) tuples in logical points — at least 2.
        total_duration: total gesture time in seconds.
        easing: optional function mapping [0,1] -> [0,1]. See curved_drag().
        include_start_move: when True, prepend a zero-duration move to
            points[0]. Set False if caller already positioned this finger
            (e.g., to satisfy WDA before a pause).
    """
    if len(points) < 2:
        raise ValueError("build_curved_drag needs at least 2 points")

    n_segments = len(points) - 1
    total_ms = int(total_duration * 1000)

    if easing is None:
        durations = [max(1, total_ms // n_segments)] * n_segments
    else:
        durations = easing_to_segment_durations(n_segments, total_ms, easing)

    x0, y0 = points[0]
    if include_start_move:
        finger.create_pointer_move(x=x0, y=y0, duration=0)
    finger.create_pointer_down()

    for (x, y), dur in zip(points[1:], durations):
        finger.create_pointer_move(x=x, y=y, duration=dur)

    finger.create_pointer_up(0)


def perform_pointer_actions(driver, fingers):
    """Send one combined W3C pointer payload for all provided fingers."""
    # Clear prior pointer state where supported; driver support is inconsistent
    # across Appium/Selenium versions.
    release_actions = getattr(driver, "release_actions", None)
    if callable(release_actions):
        release_actions()
    encoded = [finger.encode() for finger in fingers]
    max_len = max((len(source["actions"]) for source in encoded), default=0)
    for source in encoded:
        missing = max_len - len(source["actions"])
        if missing > 0:
            source["actions"].extend({"type": "pause", "duration": 0} for _ in range(missing))
    payload = {"actions": encoded}
    driver.execute("actions", payload)
    if callable(release_actions):
        release_actions()


def _build_quick_probe_drag(x, y):
    finger = make_touch_pointer("probe")
    build_curved_drag(
        finger,
        [(x, y), (x, y + 8.0)],
        total_duration=0.03,
        easing=None,
    )
    return finger


def calibrate_touch_timing(driver, *, device_key, device_w, device_h, samples=3):
    """Measure sequential perform overhead and derive a per-device cutoff."""
    base_x = device_w * 0.80
    base_y = device_h * 0.30
    measured = []

    for _ in range(max(1, samples)):
        first = _build_quick_probe_drag(base_x, base_y)
        t0 = time.monotonic()
        ActionChains(driver, devices=[first]).perform()
        t1 = time.monotonic()

        second = _build_quick_probe_drag(base_x + 6.0, base_y)
        ActionChains(driver, devices=[second]).perform()
        t2 = time.monotonic()

        first_elapsed = t1 - t0
        second_elapsed = t2 - t1
        overhead = max(0.0, second_elapsed - first_elapsed)
        measured.append(overhead)

    seq_overhead = _clamp(float(median(measured)), 0.0, _MAX_SEQUENTIAL_COMPENSATION)
    # Lower threshold slightly when measured overhead is high so we prefer
    # sequential mode sooner on unstable/high-latency stacks.
    tuned_threshold = _clamp(
        _DEFAULT_COMBINED_THRESHOLD - (seq_overhead * 0.5),
        _MIN_COMBINED_THRESHOLD,
        _MAX_COMBINED_THRESHOLD,
    )
    calibration = TouchTimingCalibration(
        sequential_overhead_s=seq_overhead,
        combined_nonneg_threshold_s=tuned_threshold,
        source="probe",
    )
    set_touch_timing_calibration(device_key, calibration)
    return calibration


def execute_two_slot_gestures(
    driver,
    *,
    g0_points,
    g1_points,
    g0_duration,
    g1_duration,
    delay,
    easing0=None,
    easing1=None,
    device_key=None,
    combined_nonneg_threshold_override=None,
    sequential_compensation_override=None,
    force_single_payload=False,
):
    """Execute two gesture slots with shared, calibrated delay scheduling."""
    slot2_start = g0_duration + delay
    if slot2_start < 0:
        raise ValueError(
            f"Invalid delay={delay:.3f}s for slot1 duration={g0_duration:.3f}s: "
            "slot2 would start before slot1 starts."
        )

    calibration = get_touch_timing_calibration(device_key)
    threshold = (
        float(combined_nonneg_threshold_override)
        if combined_nonneg_threshold_override is not None
        else calibration.combined_nonneg_threshold_s
    )
    use_combined = force_single_payload or delay < 0 or delay <= threshold

    if use_combined:
        finger0 = make_touch_pointer("finger0")
        finger1 = make_touch_pointer("finger1")
        build_curved_drag(finger0, g0_points, total_duration=g0_duration, easing=easing0)

        if force_single_payload and slot2_start > 0:
            # Keep delayed finger parked away from slot2 start until actual start
            # tick to avoid WDA phantom transitions from a pre-positioned source.
            finger1.create_pointer_move(x=g0_points[0][0], y=g0_points[0][1], duration=0)
            finger1.create_pause(slot2_start)
            build_curved_drag(
                finger1,
                g1_points,
                total_duration=g1_duration,
                easing=easing1,
                include_start_move=True,
            )
        else:
            finger1.create_pointer_move(x=g1_points[0][0], y=g1_points[0][1], duration=0)
            if slot2_start > 0:
                finger1.create_pause(slot2_start)
            build_curved_drag(
                finger1,
                g1_points,
                total_duration=g1_duration,
                easing=easing1,
                include_start_move=False,
            )
        perform_pointer_actions(driver, [finger0, finger1])
        branch = "combined-forced" if force_single_payload else "combined"
        effective_delay = delay
    else:
        finger0 = make_touch_pointer("finger0")
        build_curved_drag(finger0, g0_points, total_duration=g0_duration, easing=easing0)
        ActionChains(driver, devices=[finger0]).perform()

        base_compensation = (
            float(sequential_compensation_override)
            if sequential_compensation_override is not None
            else calibration.sequential_overhead_s
        )
        compensation = min(delay, max(0.0, base_compensation))
        sleep_for = max(0.0, delay - compensation)
        if sleep_for > 0:
            time.sleep(sleep_for)

        finger1 = make_touch_pointer("finger1")
        build_curved_drag(finger1, g1_points, total_duration=g1_duration, easing=easing1)
        ActionChains(driver, devices=[finger1]).perform()
        branch = "sequential"
        effective_delay = sleep_for

    logging.info(
        "touch scheduler branch=%s delay=%.3fs slot2_start=%.3fs threshold=%.3fs calibration=%s seq_overhead=%.3fs effective_delay=%.3fs",
        branch,
        delay,
        slot2_start,
        threshold,
        calibration.source,
        base_compensation if branch == "sequential" else calibration.sequential_overhead_s,
        effective_delay,
    )
