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
    [0.0, 0.8],  # delay 1→2
]
# fmt: on

PARAM_BOUNDS: np.ndarray = np.array(_BOUNDS_RAW, dtype=np.float64)
"""(17, 2) array of (min, max) per parameter."""

# ---------------------------------------------------------------------------
# Initial mean — informed prior for a 360 flip
# ---------------------------------------------------------------------------

# Slot 1: scoop — horizontal left-to-right swipe across the tail (~x=140, y=590)
_SCOOP = [120, 590, 220, 585, 320, 580, 0.25, 1.0]
# Slot 2: flick — north-easterly swipe from right-of-center board
_FLICK = [270, 680, 320, 620, 370, 560, 0.08, 1.0]
# Delay: almost immediate — scoop and flick happen in quick succession
_DELAY = [0.03]

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


def execute_action(driver, params: np.ndarray) -> None:
    """Clamp, unpack, and execute a 17-float action on the device.

    Executes two gesture slots sequentially via curved_drag(), with a
    time.sleep() inter-gesture delay between them.

    Args:
        driver: Appium WebDriver instance.
        params: 17-element float array from CMA-ES.
    """
    _repo_root = Path(__file__).resolve().parents[2]
    if str(_repo_root / "src") not in sys.path:
        sys.path.insert(0, str(_repo_root / "src"))

    from trueskate_ai.sim.touch_actions import curved_drag  # noqa: PLC0415

    action = unpack_action(clamp_params(params))

    for i, gesture in enumerate(action["gestures"]):
        power = gesture["easing_power"]
        easing_fn = (lambda t, p=power: t ** p) if power != 1.0 else None
        curved_drag(driver, gesture["points"], total_duration=gesture["duration"], easing=easing_fn)
        if i < len(action["delays"]):
            time.sleep(action["delays"][i])


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
