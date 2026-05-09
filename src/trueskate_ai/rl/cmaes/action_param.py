"""CMA-ES gesture parameterization: bounds, decode, and execution.

Bridges a flat 17-float numpy parameter vector to gesture execution on device.
CMA-ES optimizes this vector; this module handles bounds, unpacking, and execution.

Parameter layout (17 total):
    Slot 1:  x0,y0, x1,y1, x2,y2, duration, easing_power  → indices 0–7
    Slot 2:  x0,y0, x1,y1, x2,y2, duration, easing_power  → indices 8–15
    Delay 1→2: index 16

Coordinate, easing, and recipe conventions are documented in GESTURES.md.
"""
import numpy as np

from trueskate_ai.rl.gestures import (
    PUSH_DURATION,
    PUSH_EASING,
    PUSH_END,
    PUSH_PRE_DELAY,
    PUSH_START,
    execute_static_push,
    scale_to_device,
)

# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------

# # Legacy dimensions — used only for converting the original gesture coords
# # to canonical space when defining bounds and initial mean.
# _LEGACY_W = 414.0
# _LEGACY_H = 896.0


# def _to_canonical_x(x: float) -> float:
#     return x * (CANONICAL_W / _LEGACY_W)
#
#
# def _to_canonical_y(y: float) -> float:
#     return y * (CANONICAL_H / _LEGACY_H)


# fmt: off
_BOUNDS_RAW = [
    # Slot 1
    [0.0, 1],        # x0
    [0.5, 0.8371],  # y0
    [0.0, 1],        # x1
    [0.5, 0.8371],  # y1
    [0.0, 1],        # x2
    [0.5, 0.8371],  # y2
    [0.03, 0.8], # duration
    [0.3, 3.0],  # easing_power
    # Slot 2
    [0.0, 1],        # x0
    [0.5, 0.8371],  # y0
    [0.0, 1],        # x1
    [0.5, 0.8371],  # y1
    [0.0, 1],        # x2
    [0.5, 0.8371],  # y2
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

# Slot 1: pop flick — southward swipe from the tail area
_SCOOP = [
    0.4485, 0.6920,
    0.4595, 0.7001,
    0.4595, 0.8348,
    0.06, 1.2,
]

# Slot 2: flick — rightward swipe from the upper-mid board area (canonical coords)
_FLICK = [
    0.4485, 0.5836,
    0.6017, 0.5714,
    0.7548, 0.5636,
    0.05, 0.9,
]

# Delay: slight overlap — flick starts just before scoop finishes
_DELAY = [0.3]

INITIAL_MEAN: np.ndarray = np.array(_SCOOP + _FLICK + _DELAY, dtype=np.float64)
"""17-element informed prior for a plausible 360 flip."""

# ---------------------------------------------------------------------------
# Initial sigma
# ---------------------------------------------------------------------------

_COORD_SIGMA = 0.10
_DUR_SIGMA = 0.15
_EASING_SIGMA = 0.5
_DELAY_SIGMA = 0.15

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
    """
    midpoints = (PARAM_BOUNDS[:, 0] + PARAM_BOUNDS[:, 1]) / 2
    params = np.where(np.isfinite(params), params, midpoints)
    return np.clip(params, PARAM_BOUNDS[:, 0], PARAM_BOUNDS[:, 1])


def unpack_gesture_params(params: np.ndarray) -> dict:
    """Unpack a clamped 17-float parameter vector into a gesture recipe dict.

    Returns:
        Dict with keys:
            "gestures": list of 2 dicts, each with "points" (list of 3
                (x, y) tuples in normalised [0, 1]), "duration" (float, seconds),
                and "easing_power" (float).
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


def execute_gesture_params(
    driver,
    params: np.ndarray,
    device_w: float,
    device_h: float,
    on_post_push=None,
    timing_device_key: str | None = None,
) -> None:
    """Clamp, unpack, and execute a 17-float gesture parameter vector on the device."""
    from trueskate_ai.sim.touch_actions import execute_two_slot_gestures  # noqa: PLC0415

    recipe = unpack_gesture_params(clamp_params(params))
    g0, g1 = recipe["gestures"]
    delay = recipe["delays"][0]

    g0_points = [scale_to_device(x, y, device_w, device_h) for x, y in g0["points"]]
    g1_points = [scale_to_device(x, y, device_w, device_h) for x, y in g1["points"]]

    p0 = g0["easing_power"]
    easing0 = (lambda t, p=p0: t ** p) if p0 != 1.0 else None
    p1 = g1["easing_power"]
    easing1 = (lambda t, p=p1: t ** p) if p1 != 1.0 else None

    execute_static_push(driver, device_w=device_w, device_h=device_h, on_post_push=on_post_push)

    execute_two_slot_gestures(
        driver,
        g0_points=g0_points,
        g1_points=g1_points,
        g0_duration=g0["duration"],
        g1_duration=g1["duration"],
        delay=delay,
        easing0=easing0,
        easing1=easing1,
        device_key=timing_device_key,
    )


# ---------------------------------------------------------------------------
# Sanity-check entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Initial mean (informed 360-flip prior) ===")
    recipe = unpack_gesture_params(INITIAL_MEAN)
    for i, g in enumerate(recipe["gestures"]):
        print(f"  Gesture {i + 1}: points={g['points']}, duration={g['duration']:.3f}s, easing_power={g['easing_power']:.2f}")
    print(f"  Delays: {recipe['delays']}")

    rng = np.random.default_rng(42)
    print("\n=== 3 random samples (uniform within bounds) ===")
    for sample_idx in range(3):
        raw = rng.uniform(PARAM_BOUNDS[:, 0], PARAM_BOUNDS[:, 1])
        recipe = unpack_gesture_params(clamp_params(raw))
        print(f"\n  Sample {sample_idx + 1}:")
        for i, g in enumerate(recipe["gestures"]):
            print(f"    Gesture {i + 1}: points={g['points']}, duration={g['duration']:.3f}s, easing_power={g['easing_power']:.2f}")
        print(f"    Delays: {recipe['delays']}")
