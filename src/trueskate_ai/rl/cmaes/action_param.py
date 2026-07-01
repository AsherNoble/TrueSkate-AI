"""CMA-ES gesture parameterization: bounds, decode, and execution.

Bridges a flat numpy parameter vector to gesture execution on device.
CMA-ES optimizes this vector; this module handles bounds, unpacking, and
execution. The number of gestures (N) is curriculum-defined.

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
# Initial sigma
# ---------------------------------------------------------------------------

_COORD_SIGMA = 0.10
_DUR_SIGMA = 0.15
_EASING_SIGMA = 0.5
_DELAY_SIGMA = 0.15
_SPIN_GATE_SIGMA = 0.4   # large enough for CMA-ES to flip the gate on/off
_SPIN_T_SIGMA = 0.2


def build_initial_sigma(num_gestures: int, use_spin: bool = False) -> np.ndarray:
    """Per-parameter initial step sizes for CMA-ES, sized for N gestures (+spin)."""
    sigma = np.full(param_vector_length(num_gestures, use_spin), _COORD_SIGMA, dtype=np.float64)
    for slot in range(num_gestures):
        base = slot * PARAMS_PER_SLOT
        sigma[base + 6] = _DUR_SIGMA
        sigma[base + 7] = _EASING_SIGMA
    delay_start = num_gestures * PARAMS_PER_SLOT
    for d in range(max(0, num_gestures - 1)):
        sigma[delay_start + d] = _DELAY_SIGMA
    if use_spin:
        spin_start = delay_start + max(0, num_gestures - 1)
        sigma[spin_start] = _SPIN_GATE_SIGMA
        sigma[spin_start + 1] = _SPIN_T_SIGMA
        sigma[spin_start + 2] = _SPIN_T_SIGMA
    return sigma


def build_coordinate_mask(num_gestures: int, use_spin: bool = False) -> np.ndarray:
    """Boolean mask: True where the param is a waypoint coord (x or y).

    Used by warm-start to shrink sigma on coordinates while keeping
    duration/easing/delay/spin sigma at defaults (spin params are NOT
    coordinates → left False so warm-start keeps spin free to move).
    """
    mask = np.zeros(param_vector_length(num_gestures, use_spin), dtype=bool)
    for slot in range(num_gestures):
        base = slot * PARAMS_PER_SLOT
        mask[base : base + 6] = True
    return mask


# ---------------------------------------------------------------------------
# Initial mean — default 2-gesture prior (informed 360-flip)
# ---------------------------------------------------------------------------

# Southward gesture from the tail area
_DEFAULT_SCOOP = [
    0.4485, 0.6920,
    0.4595, 0.7001,
    0.4595, 0.8348,
    0.06, 1.2,
]

# Rightward gesture from the upper-mid board area
_DEFAULT_FLICK = [
    0.4485, 0.5836,
    0.6017, 0.5714,
    0.7548, 0.5636,
    0.05, 0.9,
]

# Slight overlap — flick starts just before scoop finishes
_DEFAULT_DELAY = 0.3

# Neutral spin prior appended to a non-spin warm-start / default mean when a
# curriculum enables spin. gate at the 0.0 threshold means ~half the initial
# population tries the rotate button (unbiased exploration from a non-spin
# warm-start); the interior hold window is then tuned by CMA-ES.
_DEFAULT_SPIN_BLOCK = [0.0, 0.2, 0.8]


def default_spin_block() -> list[float]:
    """Neutral [gate, t_start, t_end] appended to a non-spin mean under use_spin."""
    return list(_DEFAULT_SPIN_BLOCK)


def default_initial_mean(num_gestures: int) -> np.ndarray:
    """Default informed initial mean. Only defined for num_gestures == 2.

    The two-gesture prior is the 360-flip scoop+flick. For N != 2 the
    curriculum must supply explicit priors (initial_means/initial_delays)
    or a warm_start.
    """
    if num_gestures != 2:
        raise NotImplementedError(
            f"default_initial_mean is only defined for num_gestures=2 "
            f"(got {num_gestures}); curriculum must supply explicit priors."
        )
    return np.array(_DEFAULT_SCOOP + _DEFAULT_FLICK + [_DEFAULT_DELAY], dtype=np.float64)


def build_initial_mean_from_priors(
    initial_means,
    initial_delays,
) -> np.ndarray:
    """Flatten per-slot means + inter-gesture delays into a single param vector.

    Args:
        initial_means: sequence of N sequences of PARAMS_PER_SLOT floats.
        initial_delays: sequence of N-1 floats. ``None`` is treated as empty
            (only valid when N == 1).
    """
    if initial_delays is None:
        initial_delays = ()
    flat: list[float] = []
    for slot_idx, slot in enumerate(initial_means):
        if len(slot) != PARAMS_PER_SLOT:
            raise ValueError(
                f"initial_means[{slot_idx}] must have {PARAMS_PER_SLOT} values, got {len(slot)}"
            )
        flat.extend(float(v) for v in slot)
    flat.extend(float(v) for v in initial_delays)
    return np.array(flat, dtype=np.float64)


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

if __name__ == "__main__":
    print("=== Default initial mean (informed 360-flip prior, N=2) ===")
    mean_2 = default_initial_mean(2)
    print(f"  param vector length: {len(mean_2)} (expected {param_vector_length(2)})")
    recipe = unpack_gesture_params(mean_2, num_gestures=2)
    for i, g in enumerate(recipe["gestures"]):
        print(f"  Gesture {i + 1}: points={g['points']}, duration={g['duration']:.3f}s, easing_power={g['easing_power']:.2f}")
    print(f"  Delays: {recipe['delays']}")
    bounds_2 = build_param_bounds(2)
    print(f"  bounds shape: {bounds_2.shape}")
    sigma_2 = build_initial_sigma(2)
    print(f"  sigma: {sigma_2.tolist()}")
    mask_2 = build_coordinate_mask(2)
    print(f"  coord mask True count: {mask_2.sum()} (expected 12 = 6 per slot * 2 slots)")

    print("\n=== Synthetic N=3 prior ===")
    synthetic_means = [
        [0.45, 0.69, 0.45, 0.70, 0.46, 0.83, 0.06, 1.2],
        [0.45, 0.58, 0.60, 0.57, 0.75, 0.56, 0.05, 0.9],
        [0.40, 0.55, 0.50, 0.55, 0.60, 0.55, 0.05, 1.0],
    ]
    synthetic_delays = [0.15, 0.15]
    mean_3 = build_initial_mean_from_priors(synthetic_means, synthetic_delays)
    print(f"  param vector length: {len(mean_3)} (expected {param_vector_length(3)})")
    recipe = unpack_gesture_params(mean_3, num_gestures=3)
    for i, g in enumerate(recipe["gestures"]):
        print(f"  Gesture {i + 1}: points={g['points']}, duration={g['duration']:.3f}s, easing_power={g['easing_power']:.2f}")
    print(f"  Delays: {recipe['delays']}")
    bounds_3 = build_param_bounds(3)
    print(f"  bounds shape: {bounds_3.shape}")
    mask_3 = build_coordinate_mask(3)
    print(f"  coord mask True count: {mask_3.sum()} (expected 18 = 6 per slot * 3 slots)")
    assert infer_num_gestures(len(mean_3)) == 3, "infer_num_gestures round-trip failed for N=3"
    assert infer_num_gestures(len(mean_2)) == 2, "infer_num_gestures round-trip failed for N=2"
    print("  infer_num_gestures round-trip: OK")

    print("\n=== Spin block (N=2, use_spin=True) ===")
    assert param_vector_length(2, use_spin=True) == 20, "spin N=2 length should be 20"
    assert param_vector_length(2, use_spin=False) == 17, "no-spin N=2 length should be 17"
    assert infer_layout(20) == (2, True), "infer_layout(20) should be (2, True)"
    assert infer_layout(17) == (2, False), "infer_layout(17) should be (2, False)"
    assert infer_layout(26) == (3, False), "infer_layout(26) should be (3, False)"
    assert infer_layout(29) == (3, True), "infer_layout(29) should be (3, True)"
    spin_mean = np.array(_DEFAULT_SCOOP + _DEFAULT_FLICK + [_DEFAULT_DELAY] + [0.5, 0.2, 0.7])
    assert len(spin_mean) == 20
    spin_bounds = build_param_bounds(2, use_spin=True)
    spin_sigma = build_initial_sigma(2, use_spin=True)
    spin_mask = build_coordinate_mask(2, use_spin=True)
    assert spin_bounds.shape == (20, 2) and len(spin_sigma) == 20 and len(spin_mask) == 20
    assert spin_mask.sum() == 12, "spin params must not be coordinates"
    spin_recipe = unpack_gesture_params(clamp_params(spin_mean, spin_bounds), 2, use_spin=True)
    sp = spin_recipe["spin"]
    print(f"  decoded spin: {sp}")
    assert sp["enabled"] is True, "gate 0.5 >= 0 → enabled"
    assert abs(sp["t_start"] - 0.2) < 1e-6 and abs(sp["t_end"] - 0.7) < 1e-6, "t ordering"
    # gate below 0 → disabled; reversed t's get sorted
    off = unpack_gesture_params(
        clamp_params(np.array(_DEFAULT_SCOOP + _DEFAULT_FLICK + [_DEFAULT_DELAY] + [-0.5, 0.8, 0.3]), spin_bounds),
        2, use_spin=True,
    )["spin"]
    assert off["enabled"] is False and off["t_start"] < off["t_end"], "gate<0 disabled + sorted"
    assert "spin" not in unpack_gesture_params(mean_2, num_gestures=2), "no-spin recipe has no spin key"
    print("  spin round-trip: OK")

    rng = np.random.default_rng(42)
    print("\n=== 2 random samples (N=2, uniform within bounds) ===")
    for sample_idx in range(2):
        raw = rng.uniform(bounds_2[:, 0], bounds_2[:, 1])
        recipe = unpack_gesture_params(clamp_params(raw, bounds_2), num_gestures=2)
        print(f"\n  Sample {sample_idx + 1}:")
        for i, g in enumerate(recipe["gestures"]):
            print(f"    Gesture {i + 1}: points={g['points']}, duration={g['duration']:.3f}s, easing_power={g['easing_power']:.2f}")
        print(f"    Delays: {recipe['delays']}")
