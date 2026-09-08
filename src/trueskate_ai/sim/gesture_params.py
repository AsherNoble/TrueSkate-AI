"""Shared gesture parameterization: bounds, decode, and execution.

Bridges a flat numpy parameter vector to gesture execution on device.
Collectors, recipe replay and Model 2 share this representation.
The number of gestures (N) is inferred from the vector layout.

Parameter layout for N gestures (vector length 8N + (N-1) = 9N - 1):
    Slot i (i in 0..N-1): x0,y0, x1,y1, x2,y2, duration, easing_power
        → indices i*8 .. i*8+7
    Delays:               N-1 inter-gesture delays
        → indices 8N .. 8N + (N-2)
    Spin (optional, only when use_spin): gate, t_start, t_end
        → 3 trailing params; appended only for spin-family curricula
        (use_spin=True). Vector length becomes 8N + (N-1) + 3 = 9N + 2.

The no-spin (9N-1 ≡ 8 mod 9) and spin (9N+2 ≡ 2 mod 9) length classes are
disjoint, so ``infer_layout`` recovers (N, use_spin) from a vector length alone.

Coordinate, easing, and recipe conventions are documented in GESTURES.md.
"""
import numpy as np

from trueskate_ai.sim.gestures import (
    PUSH_DURATION,
    PUSH_EASING,
    PUSH_END,
    PUSH_PRE_DELAY,
    PUSH_START,
    X_BOUND_MAX,
    X_BOUND_MIN,
    Y_BOUND_MAX,
    Y_BOUND_MIN,
    execute_static_push,
    scale_to_device,
)

# Per-slot layout: 3 waypoints * 2 coords + duration + easing
PARAMS_PER_SLOT = 8

# Optional trailing spin block: [gate, t_start, t_end]. Appended after the
# delays only when a curriculum sets use_spin=True (spin-family tricks that
# need True Skate's rotate button). gate is thresholded at >= 0 (mirrors the
# PPO SpinControl gate); t_start/t_end are normalised fractions of the gesture
# schedule's total duration. See GESTURES.md.
SPIN_PARAMS = 3


def param_vector_length(num_gestures: int, use_spin: bool = False) -> int:
    """Total parameters for N gestures: 8N + (N-1) delays (+3 spin if use_spin)."""
    if num_gestures < 1:
        raise ValueError(f"num_gestures must be >= 1, got {num_gestures}")
    base = PARAMS_PER_SLOT * num_gestures + max(0, num_gestures - 1)
    return base + (SPIN_PARAMS if use_spin else 0)


def infer_layout(n_params: int) -> tuple[int, bool]:
    """Recover (num_gestures, use_spin) from a param-vector length.

    No-spin vectors have length 8N + (N-1) = 9N - 1 (≡ 8 mod 9); spin vectors
    add a trailing 3-param block → 9N + 2 (≡ 2 mod 9). The two residue classes
    are disjoint, so the layout is unambiguous.
    """
    if n_params >= PARAMS_PER_SLOT and (n_params + 1) % 9 == 0:
        return (n_params + 1) // 9, False
    if n_params >= PARAMS_PER_SLOT + SPIN_PARAMS and (n_params - SPIN_PARAMS + 1) % 9 == 0:
        return (n_params - SPIN_PARAMS + 1) // 9, True
    raise ValueError(
        f"Cannot infer layout from param length {n_params}; expected "
        f"8N+(N-1) (no-spin) or 8N+(N-1)+3 (spin) for some N >= 1."
    )


def infer_num_gestures(n_params: int) -> int:
    """Recover N from a param-vector length (spin-aware). Inverse of param_vector_length."""
    return infer_layout(n_params)[0]


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------

# fmt: off
_SLOT_BOUNDS: list[list[float]] = [
    [X_BOUND_MIN, X_BOUND_MAX],    # x0
    [Y_BOUND_MIN, Y_BOUND_MAX],    # y0
    [X_BOUND_MIN, X_BOUND_MAX],    # x1
    [Y_BOUND_MIN, Y_BOUND_MAX],    # y1
    [X_BOUND_MIN, X_BOUND_MAX],    # x2
    [Y_BOUND_MIN, Y_BOUND_MAX],    # y2
    [0.03, 0.8],                   # duration
    [0.3, 3.0],                    # easing_power
]
_DELAY_BOUNDS: list[float] = [-0.3, 0.8]
_SPIN_BOUNDS: list[list[float]] = [
    [-1.0, 1.0],   # spin gate: enabled when >= 0 (mirrors PPO SpinControl gate)
    [0.0, 1.0],    # t_start — fraction of total gesture duration
    [0.0, 1.0],    # t_end
]
# fmt: on


def build_param_bounds(num_gestures: int, use_spin: bool = False) -> np.ndarray:
    """Construct an (N_params, 2) bounds array for N gestures (+spin if use_spin)."""
    rows: list[list[float]] = []
    for _ in range(num_gestures):
        rows.extend(_SLOT_BOUNDS)
    for _ in range(max(0, num_gestures - 1)):
        rows.append(_DELAY_BOUNDS)
    if use_spin:
        rows.extend(_SPIN_BOUNDS)
    return np.array(rows, dtype=np.float64)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def clamp_params(params: np.ndarray, bounds: np.ndarray | None = None) -> np.ndarray:
    """Clamp each parameter to its bounds.

    Replaces any NaN or inf values with the midpoint of that parameter's
    bounds before clipping. CMA-ES can occasionally sample non-finite
    values, and np.clip does not catch them.

    Args:
        params: parameter vector to clamp.
        bounds: optional (N_params, 2) array of (min, max). When omitted,
            num_gestures is inferred from ``len(params)``.
    """
    if bounds is None:
        n, use_spin = infer_layout(len(params))
        bounds = build_param_bounds(n, use_spin)
    midpoints = (bounds[:, 0] + bounds[:, 1]) / 2
    params = np.where(np.isfinite(params), params, midpoints)
    return np.clip(params, bounds[:, 0], bounds[:, 1])


def unpack_gesture_params(
    params: np.ndarray, num_gestures: int | None = None, use_spin: bool = False
) -> dict:
    """Unpack a clamped parameter vector into a gesture recipe dict.

    Args:
        params: parameter vector of length 8N + (N-1) (+3 if use_spin).
        num_gestures: N. If None, (N, use_spin) are inferred from ``len(params)``.
        use_spin: whether a trailing 3-param spin block is present. Ignored when
            ``num_gestures`` is None (inferred from the vector length instead).

    Returns:
        Dict with keys:
            "gestures": list of N dicts, each with "points" (list of 3
                (x, y) tuples in normalised [0, 1]), "duration" (float, seconds),
                and "easing_power" (float).
            "delays": list of N-1 floats — inter-gesture delays in seconds.
            "spin" (only when use_spin): {"enabled": bool, "t_start": float,
                "t_end": float} — decoded spin-button control (gate thresholded
                at >= 0; t_start/t_end sorted, clipped to [0, 1]).
    """
    if num_gestures is None:
        num_gestures, use_spin = infer_layout(len(params))
    expected = param_vector_length(num_gestures, use_spin)
    if len(params) != expected:
        raise ValueError(
            f"Expected {expected} params for {num_gestures} gestures "
            f"(use_spin={use_spin}), got {len(params)}."
        )

    gestures = []
    for slot in range(num_gestures):
        base = slot * PARAMS_PER_SLOT
        points = [
            (float(params[base + 0]), float(params[base + 1])),
            (float(params[base + 2]), float(params[base + 3])),
            (float(params[base + 4]), float(params[base + 5])),
        ]
        duration = float(params[base + 6])
        easing_power = float(params[base + 7])
        gestures.append({"points": points, "duration": duration, "easing_power": easing_power})

    delay_start = num_gestures * PARAMS_PER_SLOT
    delays = [float(params[delay_start + i]) for i in range(num_gestures - 1)]
    recipe = {"gestures": gestures, "delays": delays}

    if use_spin:
        spin_start = delay_start + max(0, num_gestures - 1)
        gate = float(params[spin_start])
        t0 = min(1.0, max(0.0, float(params[spin_start + 1])))
        t1 = min(1.0, max(0.0, float(params[spin_start + 2])))
        recipe["spin"] = {
            "enabled": gate >= 0.0,
            "t_start": min(t0, t1),
            "t_end": max(t0, t1),
        }
    return recipe


def execute_gesture_params(
    driver,
    params: np.ndarray,
    device_w: float,
    device_h: float,
    *,
    num_gestures: int | None = None,
    use_spin: bool = False,
    spin_button_xy: tuple[float, float] | None = None,
    on_post_push=None,
    timing_device_key: str | None = None,
    static_push: bool = True,
) -> None:
    """Clamp, unpack, and execute a gesture parameter vector on the device.

    Args:
        params: vector of length 8N + (N-1) (+3 if it carries a spin block).
        num_gestures: N. If None, (N, use_spin) are inferred from ``len(params)``
            — so a spin-length vector is auto-detected; callers need not pass
            ``use_spin`` explicitly.
        use_spin: whether the vector carries a trailing spin block (ignored when
            ``num_gestures`` is None).
        spin_button_xy: normalised [0, 1] coords of True Skate's rotate button.
            Required for spin to fire; when None, an enabled spin block is a no-op.
        static_push: fire the board-propelling static push before the gestures.
            True for CMA-ES single-trick eval (reset+push+trick). A streaming
            policy (Model 2 receding-horizon) sets False so it does not inject a
            push before every predicted stroke.
    """
    from trueskate_ai.sim.touch_actions import execute_n_slot_gestures  # noqa: PLC0415

    if num_gestures is None:
        num_gestures, use_spin = infer_layout(len(params))

    bounds = build_param_bounds(num_gestures, use_spin)
    recipe = unpack_gesture_params(clamp_params(np.array(params), bounds), num_gestures, use_spin)

    gestures_points = []
    gestures_durations = []
    easings = []
    for g in recipe["gestures"]:
        gestures_points.append([scale_to_device(x, y, device_w, device_h) for x, y in g["points"]])
        gestures_durations.append(g["duration"])
        p = g["easing_power"]
        easings.append((lambda t, p=p: t ** p) if p != 1.0 else None)

    if static_push:
        execute_static_push(driver, device_w=device_w, device_h=device_h, on_post_push=on_post_push)

    spin = recipe.get("spin")
    spin_button_pt = None
    if spin is not None and spin.get("enabled") and spin_button_xy is not None:
        spin_button_pt = scale_to_device(
            spin_button_xy[0], spin_button_xy[1], device_w, device_h
        )

    execute_n_slot_gestures(
        driver,
        gestures_points=gestures_points,
        gestures_durations=gestures_durations,
        delays=recipe["delays"],
        easings=easings,
        device_key=timing_device_key,
        spin=spin,
        spin_button_pt=spin_button_pt,
    )


# ---------------------------------------------------------------------------
# Sanity-check entrypoint
# ---------------------------------------------------------------------------
