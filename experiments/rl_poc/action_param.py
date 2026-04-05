"""Action parameterization for the CMA-ES 360 flip RL experiment.

Bridges a flat 23-float numpy parameter vector to actual touch gestures
executed via curved_drag(). CMA-ES optimizes this vector; this module
handles bounds, unpacking, and execution.

Parameter layout (23 total):
    Slot 1 (scoop):  x0,y0, x1,y1, x2,y2, duration  → indices 0–6
    Slot 2 (flick):  x0,y0, x1,y1, x2,y2, duration  → indices 7–13
    Slot 3 (catch):  x0,y0, x1,y1, x2,y2, duration  → indices 14–20
    Delay 1→2: index 21
    Delay 2→3: index 22

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
    [448, 896],  # y0
    [0, 414],    # x1
    [448, 896],  # y1
    [0, 414],    # x2
    [448, 896],  # y2
    [0.03, 0.6], # duration
    # Slot 2
    [0, 414],    # x0
    [448, 896],  # y0
    [0, 414],    # x1
    [448, 896],  # y1
    [0, 414],    # x2
    [448, 896],  # y2
    [0.03, 0.6], # duration
    # Slot 3
    [0, 414],    # x0
    [448, 896],  # y0
    [0, 414],    # x1
    [448, 896],  # y1
    [0, 414],    # x2
    [448, 896],  # y2
    [0.03, 0.6], # duration
    # Delays
    [0.01, 0.4], # delay 1→2
    [0.01, 0.4], # delay 2→3
]
# fmt: on

PARAM_BOUNDS: np.ndarray = np.array(_BOUNDS_RAW, dtype=np.float64)
"""(23, 2) array of (min, max) per parameter."""

# ---------------------------------------------------------------------------
# Initial mean — informed prior for a 360 flip
# ---------------------------------------------------------------------------

# Slot 1: scoop — curved swipe from tail, arcing rightward
_SCOOP = [200, 780, 280, 680, 340, 600, 0.25]
# Slot 2: flick — quick swipe from center board upward/leftward
_FLICK = [250, 650, 220, 580, 200, 520, 0.08]
# Slot 3: catch — tap near center (collapsed waypoints)
_CATCH = [210, 600, 210, 600, 210, 600, 0.05]
# Delays: tight scoop→flick, longer flick→catch for board rotation
_DELAYS = [0.03, 0.35]

INITIAL_MEAN: np.ndarray = np.array(
    _SCOOP + _FLICK + _CATCH + _DELAYS, dtype=np.float64
)
"""23-element informed prior for a plausible 360 flip."""

# ---------------------------------------------------------------------------
# Initial sigma
# ---------------------------------------------------------------------------

_COORD_SIGMA = 40.0   # ~±80 pts exploration range
_DUR_SIGMA = 0.1      # reasonable spread for durations/delays

# Indices of duration/delay parameters: 6, 13, 20, 21, 22
_DUR_INDICES = {6, 13, 20, 21, 22}

INITIAL_SIGMA: np.ndarray = np.array(
    [_DUR_SIGMA if i in _DUR_INDICES else _COORD_SIGMA for i in range(23)],
    dtype=np.float64,
)
"""Per-parameter initial step sizes for CMA-ES."""

# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def clamp_params(params: np.ndarray) -> np.ndarray:
    """Clamp each parameter to its bounds.

    CMA-ES samples can fall outside the feasible region; always clamp
    before unpacking or executing.

    Args:
        params: 23-element float array from CMA-ES.

    Returns:
        New array with each value clipped to [min, max] per PARAM_BOUNDS.
    """
    return np.clip(params, PARAM_BOUNDS[:, 0], PARAM_BOUNDS[:, 1])


def unpack_action(params: np.ndarray) -> dict:
    """Unpack a clamped 23-float parameter vector into a structured dict.

    Args:
        params: 23-element float array (should already be clamped).

    Returns:
        Dict with keys:
            "gestures": list of 3 dicts, each with "points" (list of 3
                (x, y) tuples) and "duration" (float, seconds).
            "delays": list of 2 floats — inter-gesture delays in seconds.
    """
    gestures = []
    for slot in range(3):
        base = slot * 7
        points = [
            (float(params[base + 0]), float(params[base + 1])),
            (float(params[base + 2]), float(params[base + 3])),
            (float(params[base + 4]), float(params[base + 5])),
        ]
        duration = float(params[base + 6])
        gestures.append({"points": points, "duration": duration})

    delays = [float(params[21]), float(params[22])]
    return {"gestures": gestures, "delays": delays}


def execute_action(driver, params: np.ndarray) -> None:
    """Clamp, unpack, and execute a 23-float action on the device.

    Executes three gesture slots sequentially via curved_drag(), with
    time.sleep() inter-gesture delays between them.

    Args:
        driver: Appium WebDriver instance.
        params: 23-element float array from CMA-ES.
    """
    # Resolve import relative to repo root regardless of working directory
    _repo_root = Path(__file__).resolve().parents[2]
    if str(_repo_root / "src") not in sys.path:
        sys.path.insert(0, str(_repo_root / "src"))

    from trueskate_ai.sim.touch_actions import curved_drag  # noqa: PLC0415

    action = unpack_action(clamp_params(params))
    gestures = action["gestures"]
    delays = action["delays"]

    for i, gesture in enumerate(gestures):
        curved_drag(driver, gesture["points"], total_duration=gesture["duration"])
        if i < len(delays):
            time.sleep(delays[i])


# ---------------------------------------------------------------------------
# Sanity-check entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Initial mean (informed 360-flip prior) ===")
    action = unpack_action(INITIAL_MEAN)
    for i, g in enumerate(action["gestures"]):
        print(f"  Gesture {i + 1}: points={g['points']}, duration={g['duration']:.3f}s")
    print(f"  Delays: {action['delays']}")

    rng = np.random.default_rng(42)
    print("\n=== 3 random samples (uniform within bounds) ===")
    for sample_idx in range(3):
        raw = rng.uniform(PARAM_BOUNDS[:, 0], PARAM_BOUNDS[:, 1])
        action = unpack_action(clamp_params(raw))
        print(f"\n  Sample {sample_idx + 1}:")
        for i, g in enumerate(action["gestures"]):
            print(f"    Gesture {i + 1}: points={g['points']}, duration={g['duration']:.3f}s")
        print(f"    Delays: {action['delays']}")
