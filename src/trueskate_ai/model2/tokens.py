"""Gesture tokenization for the Model 2 sequence policy.

A **gesture** is the unit Model 2 predicts: one curved gesture of 3 points (the same
slot CMA-ES optimizes) plus the delay since the previous gesture.
Human play and the policy both emit a *stream* of gestures; a trick (e.g. a 360
flip = scoop + flick) is just a short sub-sequence of gestures.

    gesture = [x0, y0, x1, y1, x2, y2, duration, easing_power, delay_before]   # GESTURE_DIM = 9

Coordinates/duration/easing/delay bounds are the SINGLE source of truth from
`rl/cmaes/action_param.py` (a gesture = one gesture slot + the inter-gesture
delay row), so the token space matches what `execute_gesture_params` runs.

The network regresses in NORMALISED [0, 1] space (each dim mapped through its
bound); `encode`/`decode` convert to/from the native units used by the gesture
recipe + executor. Spin is intentionally omitted in v1 (the journal's expert
corpus is non-spin park play); add a trailing gate dim later if needed.
"""
from __future__ import annotations

import numpy as np

from trueskate_ai.sim.gesture_params import PARAMS_PER_SLOT, build_param_bounds

# One gesture = the 8 per-slot params + 1 inter-gesture delay.
GESTURE_DIM = PARAMS_PER_SLOT + 1  # 9

# (GESTURE_DIM, 2) [min, max] — first 8 rows = one slot's bounds, last row = the
# inter-gesture delay bound. Both pulled from action_param so there is exactly
# one place that defines these ranges.
_SLOT_BOUNDS = build_param_bounds(1, use_spin=False)              # (8, 2)
_DELAY_BOUND = build_param_bounds(2, use_spin=False)[PARAMS_PER_SLOT * 2 : PARAMS_PER_SLOT * 2 + 1]  # (1, 2)
GESTURE_BOUNDS = np.vstack([_SLOT_BOUNDS, _DELAY_BOUND]).astype(np.float64)  # (9, 2)

_LO = GESTURE_BOUNDS[:, 0]
_HI = GESTURE_BOUNDS[:, 1]
_SPAN = _HI - _LO


def encode(gestures: np.ndarray) -> np.ndarray:
    """Native gesture params -> normalised [0, 1]. Accepts (..., GESTURE_DIM)."""
    arr = np.asarray(gestures, dtype=np.float64)
    if arr.shape[-1] != GESTURE_DIM:
        raise ValueError(f"expected last dim {GESTURE_DIM}, got {arr.shape[-1]}")
    return np.clip((arr - _LO) / _SPAN, 0.0, 1.0)


def decode(norm: np.ndarray) -> np.ndarray:
    """Normalised [0, 1] -> native gesture params. Accepts (..., GESTURE_DIM)."""
    arr = np.asarray(norm, dtype=np.float64)
    if arr.shape[-1] != GESTURE_DIM:
        raise ValueError(f"expected last dim {GESTURE_DIM}, got {arr.shape[-1]}")
    return _LO + np.clip(arr, 0.0, 1.0) * _SPAN


def gesture_to_recipe(gesture: np.ndarray) -> dict:
    """One native gesture -> a single-gesture recipe dict (drops delay_before).

    The result plugs straight into `unpack_gesture_params`'s output shape /
    `execute_gesture_params` for replay: {"gestures": [one slot], "delays": []}.
    """
    s = np.asarray(gesture, dtype=np.float64)
    pts = [(float(s[0]), float(s[1])), (float(s[2]), float(s[3])), (float(s[4]), float(s[5]))]
    return {"gestures": [{"points": pts, "duration": float(s[6]), "easing_power": float(s[7])}],
            "delays": []}


def gestures_to_param_vector(gestures: np.ndarray) -> tuple[list[float], int]:
    """A run of N native gestures -> the (9N-1) CMA-ES param vector + N.

    Packs N slots then the N-1 inter-gesture delays (each gesture's delay_before,
    skipping the first), matching `action_param.unpack_gesture_params` layout so
    a predicted gesture-chunk executes via `execute_gesture_params`.
    """
    arr = np.asarray(gestures, dtype=np.float64).reshape(-1, GESTURE_DIM)
    n = arr.shape[0]
    vec: list[float] = []
    for s in arr:
        vec.extend(float(v) for v in s[:PARAMS_PER_SLOT])
    vec.extend(float(v) for v in arr[1:, PARAMS_PER_SLOT])  # delays between gestures
    return vec, n
