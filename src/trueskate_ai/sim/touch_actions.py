"""Low-level Appium gesture primitives for True Skate.

build_curved_drag() and make_touch_pointer() are the core building blocks;
callers compose them into multi-finger perform() payloads. All coordinates
accepted here are device logical points — callers are responsible for scaling
from normalised [0, 1] via scale_to_device() (see sim/gestures.py) before
passing points to these functions.

Gesture and coordinate conventions: GESTURES.md at the repo root.
"""
import logging
import os
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
# Minimum stagger (s) between consecutive finger-DOWNS in a combined multi-finger
# payload. True Skate reads a simultaneous 2-finger touch as a camera gesture that
# opens the park editor (confirmed: nslot/recipe open it ~8.6%/3.2% of the time,
# flick never; a 0.12s stagger drops that to 0). Env-gated so only the SLS collector
# enables it (set TRUESKATE_MIN_FINGER_STAGGER_S); CMA-ES executes unchanged.
_MIN_FINGER_STAGGER_S = float(os.environ.get("TRUESKATE_MIN_FINGER_STAGGER_S", "0") or "0")


@dataclass(frozen=True)
class TouchTimingCalibration:
    """Per-device/session timing calibration for N-slot gesture scheduling."""
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


def execute_n_slot_gestures(
    driver,
    *,
    gestures_points,
    gestures_durations,
    delays,
    easings=None,
    device_key=None,
    combined_nonneg_threshold_override=None,
    sequential_compensation_override=None,
    force_single_payload=False,
    spin=None,
    spin_button_pt=None,
):
    """Execute N gesture slots with shared, calibrated delay scheduling.

    Args:
        gestures_points: list of N gesture point lists (each in device logical points).
        gestures_durations: list of N gesture durations in seconds.
        delays: list of N-1 inter-gesture delays in seconds. Negative = overlap.
        easings: optional list of N easing callables (None entries → constant velocity).
        spin: optional decoded spin control dict {"enabled", "t_start", "t_end"}
            (fractions of the schedule's total duration). True Skate's spin is a
            HOLD control: when enabled, an extra finger is held DOWN on the spin
            button from t_start·total to t_end·total, scheduled inside the same
            W3C payload as the drags.
        spin_button_pt: (x, y) logical points of the spin button — required for
            spin to fire (already scaled by the caller).

    A delay negative enough to place a later slot before an earlier one is not
    rejected: slots are reordered by absolute start time so the schedule is
    always valid. The delay parameter is honoured exactly — only the execution
    order is relabelled — so the schedule stays continuous as a delay sweeps
    through the crossover point.

    When spin fires, the combined single-payload path is forced so the held
    spin finger shares one perform() with the drags (no concurrent WDA command
    to cancel the in-flight gesture).
    """
    n = len(gestures_points)
    if n < 1:
        raise ValueError("execute_n_slot_gestures requires at least 1 gesture")
    if len(gestures_durations) != n:
        raise ValueError(
            f"gestures_durations length {len(gestures_durations)} does not match "
            f"gestures_points length {n}"
        )
    expected_delays = max(0, n - 1)
    if len(delays) != expected_delays:
        raise ValueError(
            f"delays length {len(delays)} does not match N-1 ({expected_delays}) for N={n}"
        )
    if easings is None:
        easings = [None] * n
    elif len(easings) != n:
        raise ValueError(
            f"easings length {len(easings)} does not match gestures_points length {n}"
        )

    # Absolute start time of each slot from the delay chain. A sufficiently
    # negative delay places a later slot before an earlier one, which yields a
    # negative start time here.
    raw_starts = [0.0]
    for i in range(1, n):
        raw_starts.append(raw_starts[i - 1] + gestures_durations[i - 1] + delays[i - 1])

    # Reorder slots by absolute start time so the schedule is always valid.
    # A stable sort preserves the original order for simultaneous starts.
    order = sorted(range(n), key=lambda i: raw_starts[i])
    if order != list(range(n)):
        gestures_points = [gestures_points[i] for i in order]
        gestures_durations = [gestures_durations[i] for i in order]
        easings = [easings[i] for i in order]
        raw_starts = [raw_starts[i] for i in order]

    # Normalise so the earliest slot starts at t=0, then recompute the
    # inter-slot delays for the (possibly reordered) schedule. Sorting
    # guarantees every start is non-decreasing, so each delay >= -duration.
    starts = [s - raw_starts[0] for s in raw_starts]
    if _MIN_FINGER_STAGGER_S > 0.0:
        for i in range(1, n):
            starts[i] = max(starts[i], starts[i - 1] + _MIN_FINGER_STAGGER_S)
    delays = [
        starts[i + 1] - starts[i] - gestures_durations[i] for i in range(n - 1)
    ]

    calibration = get_touch_timing_calibration(device_key)
    threshold = (
        float(combined_nonneg_threshold_override)
        if combined_nonneg_threshold_override is not None
        else calibration.combined_nonneg_threshold_s
    )

    # Combined when: forced; any overlap requires a single payload; or every gap
    # is short enough that a single bundled payload preserves the schedule better
    # than separate WDA calls.
    # Spin forces the combined single-payload path: one perform() in flight
    # racing the spin-button tap thread (the topology the PPO path validated).
    spin_active = bool(spin and spin.get("enabled") and spin_button_pt is not None)

    has_delays = len(delays) > 0
    any_negative = has_delays and any(d < 0 for d in delays)
    all_under_threshold = has_delays and all(d <= threshold for d in delays)
    use_combined = (
        force_single_payload
        or spin_active
        or any_negative
        or (has_delays and all_under_threshold)
    )

    base_compensation = (
        float(sequential_compensation_override)
        if sequential_compensation_override is not None
        else calibration.sequential_overhead_s
    )

    if use_combined:
        fingers = []
        for i in range(n):
            finger = make_touch_pointer(f"finger{i}")
            start_x, start_y = gestures_points[i][0]
            if i == 0:
                build_curved_drag(
                    finger,
                    gestures_points[i],
                    total_duration=gestures_durations[i],
                    easing=easings[i],
                )
            elif force_single_payload and starts[i] > 0:
                # Park parked finger away from its actual start until the scheduled tick
                # to avoid WDA phantom transitions from a pre-positioned source.
                finger.create_pointer_move(x=start_x, y=start_y, duration=0)
                finger.create_pause(starts[i])
                build_curved_drag(
                    finger,
                    gestures_points[i],
                    total_duration=gestures_durations[i],
                    easing=easings[i],
                    include_start_move=True,
                )
            else:
                finger.create_pointer_move(x=start_x, y=start_y, duration=0)
                if starts[i] > 0:
                    finger.create_pause(starts[i])
                # include_start_move re-issues the start move immediately
                # before pointer_down: WDA drops a standalone zero-duration
                # move when a pause follows it, so pointer_down would
                # otherwise fire at the pointer origin (top-left) — observed
                # as a spurious top-left downward swipe that opens the menu.
                build_curved_drag(
                    finger,
                    gestures_points[i],
                    total_duration=gestures_durations[i],
                    easing=easings[i],
                    include_start_move=True,
                )
            fingers.append(finger)

        # Spin button is a HOLD control: an extra finger goes DOWN at
        # t_start·total and UP at t_end·total, scheduled INSIDE the same W3C
        # payload as the drags. (A prior design tapped it from a background
        # thread via `mobile: tap` concurrent with perform() — that cancelled
        # the in-flight gesture on the shared WDA session and nullified the
        # trick. A finger in the same payload has no such conflict.)
        if spin_active:
            spin_total = max(
                (starts[i] + gestures_durations[i] for i in range(n)), default=0.01
            )
            start_offset = max(0.0, float(spin["t_start"]) * spin_total)
            hold = max(0.0, (float(spin["t_end"]) - float(spin["t_start"])) * spin_total)
            bx, by = spin_button_pt
            sf = make_touch_pointer("spin")
            # Position, wait, then RE-ISSUE the move before pointer_down: WDA
            # drops a standalone zero-duration move when a pause follows it, so
            # the down would otherwise fire at the pointer origin (0,0).
            sf.create_pointer_move(x=bx, y=by, duration=0)
            if start_offset > 0:
                sf.create_pause(start_offset)
            sf.create_pointer_move(x=bx, y=by, duration=0)
            sf.create_pointer_down()
            if hold > 0:
                sf.create_pause(hold)
            sf.create_pointer_up(0)
            fingers.append(sf)

        perform_pointer_actions(driver, fingers)

        branch = (
            "combined-spin" if spin_active
            else "combined-forced" if force_single_payload
            else "combined"
        )
        effective_delays = list(delays)
    else:
        # Sequential — N separate WDA calls with compensated delays in between.
        effective_delays = []
        for i in range(n):
            finger = make_touch_pointer(f"finger{i}")
            build_curved_drag(
                finger,
                gestures_points[i],
                total_duration=gestures_durations[i],
                easing=easings[i],
            )
            ActionChains(driver, devices=[finger]).perform()
            if i < n - 1:
                compensation = min(delays[i], max(0.0, base_compensation))
                sleep_for = max(0.0, delays[i] - compensation)
                if sleep_for > 0:
                    time.sleep(sleep_for)
                effective_delays.append(sleep_for)
        branch = "sequential"

    logging.info(
        "touch scheduler branch=%s n=%d order=%s delays=%s starts=%s threshold=%.3fs "
        "calibration=%s seq_overhead=%.3fs effective_delays=%s",
        branch,
        n,
        order,
        [round(d, 3) for d in delays],
        [round(s, 3) for s in starts],
        threshold,
        calibration.source,
        base_compensation,
        [round(d, 3) for d in effective_delays],
    )
