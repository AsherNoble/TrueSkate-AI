"""Action parameterization for the CMA-ES 360 flip experiment.

Bridges a flat 17-float numpy parameter vector to actual touch gestures
executed via curved_drag(). CMA-ES optimizes this vector; this module
handles bounds, unpacking, and execution.

Parameter layout (17 total):
    Slot 1 (scoop):  x0,y0, x1,y1, x2,y2, duration, easing_power  → indices 0–7
    Slot 2 (flick):  x0,y0, x1,y1, x2,y2, duration, easing_power  → indices 8–15
    Delay 1→2: index 16

Easing power controls the velocity profile passed to curved_drag():
    power < 1.0  — decelerating (fast start, slow end)
    power = 1.0  — constant velocity (linear)
    power > 1.0  — accelerating (slow start, fast end)

Screen: 414×896 logical points (iPhone 11 @2x).
"""
import sys
import time
from pathlib import Path

import numpy as np
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.actions.pointer_input import PointerInput

# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------

# fmt: off
_BOUNDS_RAW = [
    # Slot 1
    [0, 414],    # x0
    [448, 750],  # y0  — capped at 750 to avoid the iOS home indicator zone
    [0, 414],    # x1
    [448, 750],  # y1
    [0, 414],    # x2
    [448, 750],  # y2
    [0.03, 0.8], # duration
    [0.3, 3.0],  # easing_power
    # Slot 2
    [0, 414],    # x0
    [448, 750],  # y0
    [0, 414],    # x1
    [448, 750],  # y1
    [0, 414],    # x2
    [448, 750],  # y2
    [0.03, 0.8], # duration
    [0.3, 3.0],  # easing_power
    # Delay
    [-0.3, 0.8], # delay 1→2
]
# fmt: on

PARAM_BOUNDS: np.ndarray = np.array(_BOUNDS_RAW, dtype=np.float64)
"""(17, 2) array of (min, max) per parameter."""

# ---------------------------------------------------------------------------
# Initial mean — informed prior for a 360 flip
# ---------------------------------------------------------------------------

# Slot 1: scoop — left-to-right arc across the TAIL (y~680, not 575)
_SCOOP = [100, 680, 220, 690, 340, 675, 0.06, 1.2]

# Slot 2: flick — upward-left diagonal from mid-board (the kickflip component)
_FLICK = [270, 590, 220, 530, 160, 470, 0.05, 0.7]

# Delay: slight overlap — flick starts just before scoop finishes
_DELAY = [-0.02]

INITIAL_MEAN: np.ndarray = np.array(_SCOOP + _FLICK + _DELAY, dtype=np.float64)
"""17-element informed prior for a plausible 360 flip."""

# ---------------------------------------------------------------------------
# Initial sigma
# ---------------------------------------------------------------------------

# Parameter type → sigma mapping
_COORD_SIGMA = 40.0
_DUR_SIGMA = 0.15
_EASING_SIGMA = 0.5
_DELAY_SIGMA = 0.15

# Indices by type:
#   duration:     6, 14
#   easing_power: 7, 15
#   delay:        16
_SIGMA_MAP = {
    6: _DUR_SIGMA, 7: _EASING_SIGMA,
    14: _DUR_SIGMA, 15: _EASING_SIGMA,
    16: _DELAY_SIGMA,
}

INITIAL_SIGMA: np.ndarray = np.array(
    [_SIGMA_MAP.get(i, _COORD_SIGMA) for i in range(17)],
    dtype=np.float64,
)
"""Per-parameter initial step sizes for CMA-ES."""

# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def clamp_params(params: np.ndarray) -> np.ndarray:
    """Clamp each parameter to its bounds.

    Replaces any NaN or inf values with the midpoint of that parameter's
    bounds before clipping. CMA-ES can occasionally sample non-finite
    values, and np.clip does not catch them.

    Args:
        params: 17-element float array from CMA-ES.

    Returns:
        New array with each value clipped to [min, max] per PARAM_BOUNDS.
    """
    midpoints = (PARAM_BOUNDS[:, 0] + PARAM_BOUNDS[:, 1]) / 2
    params = np.where(np.isfinite(params), params, midpoints)
    return np.clip(params, PARAM_BOUNDS[:, 0], PARAM_BOUNDS[:, 1])


def unpack_action(params: np.ndarray) -> dict:
    """Unpack a clamped 17-float parameter vector into a structured dict.

    Args:
        params: 17-element float array (should already be clamped).

    Returns:
        Dict with keys:
            "gestures": list of 2 dicts, each with "points" (list of 3
                (x, y) tuples), "duration" (float, seconds), and
                "easing_power" (float).
            "delays": list of 1 float — inter-gesture delay in seconds.
    """
    gestures = []
    for slot in range(2):
        base = slot * 8
        points = [
            (float(params[base + 0]), float(params[base + 1])),
            (float(params[base + 2]), float(params[base + 3])),
            (float(params[base + 4]), float(params[base + 5])),
        ]
        duration = float(params[base + 6])
        easing_power = float(params[base + 7])
        gestures.append({"points": points, "duration": duration, "easing_power": easing_power})

    delays = [float(params[16])]
    return {"gestures": gestures, "delays": delays}


APPIUM_LATENCY_OFFSET = 0.8
"""Approximate Appium/WDA round-trip latency in seconds, subtracted when
computing pause durations so delay values reflect true real-world timing."""

# Static pre-execution push parameters (not optimized by CMA-ES)
_PUSH_PRE_DELAY = 0.5
"""Delay before each trick execution during which the push occurs (seconds)."""
_PUSH_START = (350, 224)
"""Push start position: right side, upper half (x=350, y=224)."""
_PUSH_END = (350, 672)
"""Push end position: right side, lower half (x=350, y=672)."""
_PUSH_DURATION = 0.02
"""Push duration (seconds)."""
_PUSH_EASING = 2.0
"""Push easing power — accelerating (slow start, fast end) for realistic push dynamics."""


def execute_action(driver, params: np.ndarray) -> None:
    """Clamp, unpack, and execute a 17-float action on the device.

    Fires three gesture slots in a single W3C Actions perform() call:
      - finger0: scoop (slot 1)
      - finger1: flick (slot 2), offset by latency-adjusted delay
      - finger2: static downward push (right side), occurring during pre-delay

    Args:
        driver: Appium WebDriver instance.
        params: 17-element float array from CMA-ES.
    """
    _repo_root = Path(__file__).resolve().parents[2]
    if str(_repo_root / "src") not in sys.path:
        sys.path.insert(0, str(_repo_root / "src"))

    from trueskate_ai.sim.touch_actions import build_curved_drag  # noqa: PLC0415

    action = unpack_action(clamp_params(params))
    g0, g1 = action["gestures"]
    delay = action["delays"][0]

    # Debug: print scoop, flick, and delay
    print(
        f"\n[SCOOP]\n"
        f"  points:  {[f'({p[0]:.0f}, {p[1]:.0f})' for p in g0['points']]}\n"
        f"  duration: {g0['duration']:.3f}s\n"
        f"  easing:   {g0['easing_power']:.2f}\n"
        f"[FLICK]\n"
        f"  points:  {[f'({p[0]:.0f}, {p[1]:.0f})' for p in g1['points']]}\n"
        f"  duration: {g1['duration']:.3f}s\n"
        f"  easing:   {g1['easing_power']:.2f}\n"
        f"[DELAY]\n"
        f"  {delay:.3f}s (+ {_PUSH_PRE_DELAY:.3f}s static pre-delay)\n"
    )

    p0 = g0["easing_power"]
    easing0 = (lambda t, p=p0: t ** p) if p0 != 1.0 else None
    p1 = g1["easing_power"]
    easing1 = (lambda t, p=p1: t ** p) if p1 != 1.0 else None
    push_easing = lambda t: t ** _PUSH_EASING  # accelerating (ease_in)

    # --- Step 1: static push (single-finger, separate perform) ---
    # Must be a separate perform() call — bundling 3 fingers in one perform()
    # triggers iOS's system three-finger gesture (undo/redo), swallowing all touches
    # before True Skate sees them.
    finger2 = PointerInput("touch", "finger2")
    build_curved_drag(
        finger2, [_PUSH_START, _PUSH_END],
        total_duration=_PUSH_DURATION, easing=push_easing
    )
    ActionChains(driver, devices=[finger2]).perform()

    # Wait out the remaining pre-delay after the push finishes
    remaining_pre_delay = _PUSH_PRE_DELAY - _PUSH_DURATION
    if remaining_pre_delay > 0:
        time.sleep(remaining_pre_delay)

    # --- Step 2: scoop + flick (two-finger perform) ---
    finger0 = PointerInput("touch", "finger0")
    finger1 = PointerInput("touch", "finger1")

    # Slot 1 on finger0
    build_curved_drag(finger0, g0["points"], total_duration=g0["duration"], easing=easing0)

    # Offset finger1 start: slot1 duration + delay, minus Appium latency
    # WDA requires a pointerMove before any pause, so position finger1 first.
    finger1.create_pointer_move(x=g1["points"][0][0], y=g1["points"][0][1], duration=0)
    adjusted_delay = delay - APPIUM_LATENCY_OFFSET
    pause_secs = max(0.0, g0["duration"] + adjusted_delay)
    if pause_secs > 0:
        finger1.create_pause(pause_secs)

    # Slot 2 on finger1
    build_curved_drag(finger1, g1["points"], total_duration=g1["duration"], easing=easing1)

    # Scoop + flick execute simultaneously
    ActionChains(driver, devices=[finger0, finger1]).perform()


# ---------------------------------------------------------------------------
# Sanity-check entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Initial mean (informed 360-flip prior) ===")
    action = unpack_action(INITIAL_MEAN)
    for i, g in enumerate(action["gestures"]):
        print(f"  Gesture {i + 1}: points={g['points']}, duration={g['duration']:.3f}s, easing_power={g['easing_power']:.2f}")
    print(f"  Delays: {action['delays']}")

    rng = np.random.default_rng(42)
    print("\n=== 3 random samples (uniform within bounds) ===")
    for sample_idx in range(3):
        raw = rng.uniform(PARAM_BOUNDS[:, 0], PARAM_BOUNDS[:, 1])
        action = unpack_action(clamp_params(raw))
        print(f"\n  Sample {sample_idx + 1}:")
        for i, g in enumerate(action["gestures"]):
            print(f"    Gesture {i + 1}: points={g['points']}, duration={g['duration']:.3f}s, easing_power={g['easing_power']:.2f}")
        print(f"    Delays: {action['delays']}")
