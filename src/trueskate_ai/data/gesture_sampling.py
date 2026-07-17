"""Random gesture samplers for trace/frame collection.

Single source of truth for the gestures the trace collectors fire. Four modes,
mixed by `sample_mixture`, give the sequence model a broad (frame -> known-gesture)
corpus across visual domains:

- "flick"  — a board-centered outward flick (the original self-label sampler);
             trace-rich, board roughly stationary. Executed via curved_drag.
- "nslot"  — a full random N-slot gesture vector within the CMA-ES bounds;
             broadest timing/overlap coverage. Executed via execute_gesture_params
             (pushes first, like a real trick fire).
- "recipe" — a converged trick-library recipe, jittered within bounds; dense
             coverage near the real-trick manifold. Packed to a vector and
             executed via execute_gesture_params.
- "spin"   — a random N-slot gesture with the spin (rotate-button) HOLD guaranteed
             active, for spin-family coverage. Same held-finger execution path as
             nslot (execute_gesture_params -> execute_n_slot_gestures), gated by
             sample_mixture's `spin_frac`.

The executed gesture is always the label — outcome (land / wall-bump / whiff) is
irrelevant for this corpus, which is what lets it run in obstacle-heavy SLS parks.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trueskate_ai.rl.cmaes.action_param import (
    PARAMS_PER_SLOT,
    SPIN_PARAMS,
    build_param_bounds,
    clamp_params,
)
from trueskate_ai.sim.gestures import (
    X_BOUND_MAX,
    X_BOUND_MIN,
    Y_BOUND_MAX,
    Y_BOUND_MIN,
)

# Flick START-point sampling region. The orange finger-trace follows the FINGER
# path anywhere on screen — NOT only on/near the board (verified live: a static
# push far from the board still renders a full trace). So starts are sampled
# broadly, near-uniformly over the safe screen bounds, giving Model 1 (a per-frame
# touch LOCALIZER) positional coverage with no blind spots. A small inset keeps
# onsets off the very edge; the global X/Y bounds remain the single source.
_START_INSET = 0.02
_FLICK_START_X = (X_BOUND_MIN, X_BOUND_MAX - _START_INSET)                 # ~0.12 .. 0.98
_FLICK_START_Y = (Y_BOUND_MIN + _START_INSET, Y_BOUND_MAX - _START_INSET)  # ~0.14 .. 0.86

# Every flick must leave a full, clean trace: guarantee this minimum start->end
# displacement (post-bounds-clip). Edge starts are aimed inward to satisfy it.
_FLICK_MIN_REACH = 0.15

# A True Skate update added a "Bolt Challenges" indicator (the "(N/M)" circle-
# exclamation) in the TOP-LEFT corner. A gesture finger that lands on it opens the
# Bolt Challenges modal, which the collector then blindly fires into — corrupting
# ~12% of samples. Frame-onset forensics traced every modal-open to a waypoint near
# (0.14, 0.20-0.24); this corner rect covers the indicator's hit target plus margin,
# and clamp_in_bounds() keeps ALL waypoints out of it (pushed right, off the strip).
_BOLT_EXCL_X = (X_BOUND_MIN, 0.20)
_BOLT_EXCL_Y = (Y_BOUND_MIN, 0.30)

# Per-param jitter sigmas for "recipe" mode, in the vector's native units,
# indexed within a slot as [x0,y0,x1,y1,x2,y2,duration,easing].
_COORD_JITTER = 0.03
_DUR_JITTER = 0.05
_EASING_JITTER = 0.2
_DELAY_JITTER = 0.05
_SPIN_T_JITTER = 0.05

# Minimum hold window (fraction of total gesture duration) for a guaranteed-spin
# sample, so the rotate button is HELD long enough to be visible in the frames —
# a uniform t_start/t_end can otherwise collapse to a near-zero-length press.
_SPIN_MIN_HOLD = 0.25


@dataclass
class GestureSample:
    """A sampled gesture in one of four executable forms.

    kind == "flick": use waypoints/duration/easing_power (curved_drag, no push).
    kind in {"nslot","recipe","spin"}: use params/num_gestures/use_spin
    (execute_gesture_params).
    """
    kind: str
    waypoints: list[tuple[float, float]] | None = None
    duration: float | None = None
    easing_power: float | None = None
    params: list[float] | None = None
    num_gestures: int | None = None
    use_spin: bool = False
    source: str | None = None  # recipe filename for kind == "recipe"

    def meta(self) -> dict:
        """JSON-serialisable description for the sample's meta.json."""
        d: dict = {"gesture_distribution": self.kind}
        if self.kind == "flick":
            d.update(
                waypoints=self.waypoints,
                duration=self.duration,
                easing_power=self.easing_power,
            )
        else:
            d.update(
                params=self.params,
                num_gestures=self.num_gestures,
                use_spin=self.use_spin,
            )
            if self.source:
                d["recipe_source"] = self.source
        return d


def _flick_end(rng: np.random.Generator, sx: float, sy: float, mag: float) -> tuple[float, float]:
    """End point ~mag from (sx, sy). Uniform direction while it keeps a full trace
    in-bounds; otherwise aimed toward screen center so edge starts flick inward."""
    for _ in range(8):
        ang = float(rng.uniform(0, 2 * np.pi))
        ex = float(np.clip(sx + mag * np.cos(ang), X_BOUND_MIN, X_BOUND_MAX))
        ey = float(np.clip(sy + mag * np.sin(ang), Y_BOUND_MIN, Y_BOUND_MAX))
        if np.hypot(ex - sx, ey - sy) >= _FLICK_MIN_REACH:
            return ex, ey
    # every uniform angle clipped short (start hugs an edge): aim at the interior
    cx = (X_BOUND_MIN + X_BOUND_MAX) / 2
    cy = (Y_BOUND_MIN + Y_BOUND_MAX) / 2
    ang = float(np.arctan2(cy - sy, cx - sx))
    ex = float(np.clip(sx + mag * np.cos(ang), X_BOUND_MIN, X_BOUND_MAX))
    ey = float(np.clip(sy + mag * np.sin(ang), Y_BOUND_MIN, Y_BOUND_MAX))
    return ex, ey


def sample_flick(rng: np.random.Generator) -> dict:
    """A flick from a broadly-sampled start point, reaching outward in a random dir.

    Start is sampled near-uniformly over the safe screen bounds (_FLICK_START_X/_Y)
    so Model 1 sees touch onsets everywhere, not just at board center. Every flick
    keeps a full trace (>= _FLICK_MIN_REACH); starts near an edge are aimed inward.
    Returns the legacy dict {waypoints, duration, easing_power} so the original
    self-labeled-trace collector can import this verbatim.
    """
    sx = float(rng.uniform(*_FLICK_START_X))
    sy = float(rng.uniform(*_FLICK_START_Y))
    mag = float(rng.uniform(0.18, 0.45))  # flick reach (normalised)
    ex, ey = _flick_end(rng, sx, sy, mag)
    if int(rng.integers(0, 2)) == 0:
        waypoints = [(sx, sy), (ex, ey)]  # straight flick
    else:
        mx = float(np.clip((sx + ex) / 2 + rng.uniform(-0.08, 0.08), X_BOUND_MIN, X_BOUND_MAX))
        my = float(np.clip((sy + ey) / 2 + rng.uniform(-0.08, 0.08), Y_BOUND_MIN, Y_BOUND_MAX))
        waypoints = [(sx, sy), (mx, my), (ex, ey)]  # curved flick
    duration = float(rng.uniform(0.12, 0.4))  # flicks are fast
    easing_power = float(rng.uniform(0.6, 2.0))
    return {"waypoints": waypoints, "duration": duration, "easing_power": easing_power}


def sample_nslot(rng: np.random.Generator, num_gestures: int, use_spin: bool) -> GestureSample:
    """A full random gesture vector, uniform within the CMA-ES bounds."""
    bounds = build_param_bounds(num_gestures, use_spin)
    raw = rng.uniform(bounds[:, 0], bounds[:, 1])
    params = clamp_params(np.asarray(raw, dtype=np.float64), bounds)
    return GestureSample(
        kind="nslot",
        params=[float(v) for v in params],
        num_gestures=num_gestures,
        use_spin=use_spin,
    )


def sample_spin(rng: np.random.Generator, num_gestures: int) -> GestureSample:
    """A random N-slot gesture with the spin HOLD guaranteed ACTIVE.

    Like sample_nslot(use_spin=True), but the spin gate is forced enabled and the
    hold window spans at least _SPIN_MIN_HOLD of the gesture. A uniform spin block
    would leave ~half the samples gate-off (no spin at all) and allow near-zero
    hold windows — both dilute genuine spin coverage. The base gesture stays fully
    random (this corpus is outcome-agnostic), so the label is a random gesture with
    a visibly-held rotate button: exactly the (frames -> gesture) pair the video
    model needs to learn spin-family tricks. Tagged kind="spin" so the corpus is
    filterable, but executes on the identical held-finger path as an nslot sample.
    """
    s = sample_nslot(rng, num_gestures, use_spin=True)
    assert s.params is not None
    t_start = float(rng.uniform(0.0, 1.0 - _SPIN_MIN_HOLD))
    t_end = float(rng.uniform(t_start + _SPIN_MIN_HOLD, 1.0))
    # gate: the enable threshold is >= 0; the 0.05 floor keeps a margin off it.
    gate = float(rng.uniform(0.05, 1.0))
    s.params[-SPIN_PARAMS:] = [gate, t_start, t_end]
    s.kind = "spin"
    return s


def recipe_to_vector(recipe: dict) -> tuple[list[float], int, bool]:
    """Pack a decoded trick recipe (gestures/delays/spin) into a param vector.

    Inverse of action_param.unpack_gesture_params for the canonical 3-waypoint
    slot layout. Raises ValueError if a gesture isn't 3 waypoints.
    """
    gestures = recipe["gestures"]
    vec: list[float] = []
    for g in gestures:
        pts = g["points"]
        if len(pts) != 3:
            raise ValueError(f"recipe gesture has {len(pts)} waypoints, expected 3")
        for px, py in pts:
            vec.extend([float(px), float(py)])
        vec.extend([float(g["duration"]), float(g["easing_power"])])
    vec.extend(float(d) for d in recipe.get("delays", []))
    spin = recipe.get("spin")
    use_spin = bool(spin)
    if use_spin:
        vec.extend([
            1.0 if spin.get("enabled") else -1.0,
            float(spin["t_start"]),
            float(spin["t_end"]),
        ])
    n = len(gestures)
    expected = PARAMS_PER_SLOT * n + max(0, n - 1) + (SPIN_PARAMS if use_spin else 0)
    if len(vec) != expected:
        raise ValueError(f"packed length {len(vec)} != expected {expected} for N={n}")
    return vec, n, use_spin


def _load_recipe_vectors(recipe_paths: list[Path], mode: str = "median") -> list[tuple[list[float], int, bool, str]]:
    """Load and pack recipes from trick-library JSONs; skip any that don't fit."""
    out: list[tuple[list[float], int, bool, str]] = []
    for p in recipe_paths:
        try:
            data = json.loads(Path(p).read_text())
            recipe = data.get(f"{mode}_gestures") or data.get("best_gestures") or data.get("median_gestures")
            if not recipe:
                continue
            vec, n, use_spin = recipe_to_vector(recipe)
            out.append((vec, n, use_spin, Path(p).name))
        except (ValueError, KeyError, json.JSONDecodeError, OSError):
            continue  # incompatible/legacy library — skip, don't crash the run
    return out


def sample_recipe(
    rng: np.random.Generator, recipe_vectors: list[tuple[list[float], int, bool, str]]
) -> GestureSample:
    """Jitter a random converged recipe within its bounds."""
    vec, n, use_spin, name = recipe_vectors[int(rng.integers(0, len(recipe_vectors)))]
    bounds = build_param_bounds(n, use_spin)
    arr = np.asarray(vec, dtype=np.float64)
    # Per-param jitter: coords/dur/easing within each slot, then delays, then spin t's.
    sigma = np.empty_like(arr)
    for slot in range(n):
        base = slot * PARAMS_PER_SLOT
        sigma[base:base + 6] = _COORD_JITTER          # 3 waypoints * 2 coords
        sigma[base + 6] = _DUR_JITTER
        sigma[base + 7] = _EASING_JITTER
    delay_start = n * PARAMS_PER_SLOT
    sigma[delay_start:delay_start + max(0, n - 1)] = _DELAY_JITTER
    if use_spin:
        sigma[-SPIN_PARAMS] = 0.0                       # keep the gate sign stable
        sigma[-SPIN_PARAMS + 1:] = _SPIN_T_JITTER
    jittered = arr + rng.normal(0.0, 1.0, size=arr.shape) * sigma
    params = clamp_params(jittered, bounds)
    return GestureSample(
        kind="recipe",
        params=[float(v) for v in params],
        num_gestures=n,
        use_spin=use_spin,
        source=name,
    )


def _push_out_bolt_zone(x: float, y: float) -> tuple[float, float]:
    """Push a waypoint out of the top-left Bolt-Challenges indicator rect by moving
    it RIGHT to the rect's edge — clearing the whole left-edge button strip in that
    band (see _BOLT_EXCL_X/Y). Points outside the rect pass through unchanged."""
    x0, x1 = _BOLT_EXCL_X
    y0, y1 = _BOLT_EXCL_Y
    if x0 <= x < x1 and y0 <= y < y1:  # half-open: x==x1 is already outside
        return x1, y
    return x, y


def _restore_min_reach(sx: float, sy: float, ex: float, ey: float) -> tuple[float, float]:
    """If the bolt-zone push shrank a flick's start->end displacement below
    _FLICK_MIN_REACH, push the end point further out along the same direction
    (clipped to bounds) to restore the guarantee _flick_end originally made."""
    dx, dy = ex - sx, ey - sy
    dist = float(np.hypot(dx, dy))
    if dist >= _FLICK_MIN_REACH:
        return ex, ey
    if dist < 1e-9:
        dx, dy, dist = 1.0, 0.0, 1.0  # degenerate: start==end, pick an arbitrary direction
    scale = _FLICK_MIN_REACH / dist
    ex2 = float(np.clip(sx + dx * scale, X_BOUND_MIN, X_BOUND_MAX))
    ey2 = float(np.clip(sy + dy * scale, Y_BOUND_MIN, Y_BOUND_MAX))
    return ex2, ey2


def clamp_in_bounds(s: GestureSample) -> GestureSample:
    """Defensive chokepoint: guarantee a sampled gesture lies within the RL
    coordinate bounds (X_BOUND_MIN..MAX, Y_BOUND_MIN..MAX) AND outside the top-left
    Bolt-Challenges indicator rect (_BOLT_EXCL_*), so the collector can never
    execute — or mislabel — an out-of-bounds action or one that opens the Bolt modal,
    whatever the sampling path or any future change. Clamps the SAMPLE itself
    (mutates in place), so the saved label always matches what executes. Normalised
    coords are absolute (0/1 = screen edges); clamping constrains, it never rescales.
    """
    if s.kind == "flick" and s.waypoints is not None:
        pushed = [
            _push_out_bolt_zone(
                float(np.clip(x, X_BOUND_MIN, X_BOUND_MAX)),
                float(np.clip(y, Y_BOUND_MIN, Y_BOUND_MAX)))
            for x, y in s.waypoints
        ]
        # The bolt-zone push can move the start point close enough to the end
        # point to violate _FLICK_MIN_REACH (see _restore_min_reach); only the
        # first/last waypoints define that displacement, so re-check just those.
        sx, sy = pushed[0]
        pushed[-1] = _restore_min_reach(sx, sy, *pushed[-1])
        s.waypoints = pushed
    elif s.params is not None and s.num_gestures is not None:
        bounds = build_param_bounds(s.num_gestures, s.use_spin)
        arr = clamp_params(np.asarray(s.params, dtype=np.float64), bounds)
        for slot in range(s.num_gestures):
            b = slot * PARAMS_PER_SLOT
            for c in (0, 2, 4):  # the slot's 3 waypoint (x, y) pairs
                arr[b + c], arr[b + c + 1] = _push_out_bolt_zone(float(arr[b + c]), float(arr[b + c + 1]))
        s.params = [float(v) for v in arr]
    return s


def sample_mixture(
    rng: np.random.Generator,
    *,
    fracs: tuple[float, float, float] = (0.6, 0.25, 0.15),
    spin_frac: float = 0.0,
    num_gestures: int = 2,
    use_spin: bool = False,
    recipe_vectors: list[tuple[list[float], int, bool, str]] | None = None,
) -> GestureSample:
    """Draw one gesture from the flick / nslot / recipe / spin mixture, guaranteed
    within the coordinate bounds (via clamp_in_bounds).

    fracs = (flick, nslot, recipe) is the non-spin base mix; if no recipes are
    available the recipe share is redistributed to nslot (so the mix never silently
    stalls). spin_frac is a TRUE share of all fires in [0, 1]: the base mix keeps
    its internal ratios but is scaled to the remaining (1 - spin_frac), so
    spin_frac=0.2 yields ~20% guaranteed-spin gestures (a held rotate button)
    whatever fracs sums to. It is the knob to grow the spin-family corpus,
    independent of `use_spin` (which only makes the plain nslot branch
    spin-capable, ~half of those gate-off).
    """
    f_flick, f_nslot, f_recipe = fracs
    if not recipe_vectors:
        f_nslot += f_recipe
        f_recipe = 0.0
    # spin_frac as a raw weight would dilute: e.g. defaults (sum 1.0) + 0.2 give
    # 0.2/1.2 ≈ 17%, not 20%. Scale the base mix to (1 - spin_frac) instead so
    # the advertised share is exact.
    f_spin = min(1.0, max(0.0, spin_frac))
    base_total = f_flick + f_nslot + f_recipe
    if base_total <= 0 and f_spin <= 0:
        raise ValueError("sample_mixture: all mixture weights are zero")
    if base_total > 0:
        scale = (1.0 - f_spin) / base_total
        f_flick, f_nslot, f_recipe = f_flick * scale, f_nslot * scale, f_recipe * scale
    total = f_flick + f_nslot + f_recipe + f_spin
    r = float(rng.uniform(0, total))
    if r < f_flick:
        g = sample_flick(rng)
        s = GestureSample(
            kind="flick",
            waypoints=g["waypoints"],
            duration=g["duration"],
            easing_power=g["easing_power"],
        )
    elif r < f_flick + f_nslot:
        s = sample_nslot(rng, num_gestures, use_spin)
    elif r < f_flick + f_nslot + f_recipe:
        s = sample_recipe(rng, recipe_vectors)
    else:
        s = sample_spin(rng, num_gestures)
    return clamp_in_bounds(s)


def load_recipe_vectors(recipe_dir: Path, mode: str = "median") -> list[tuple[list[float], int, bool, str]]:
    """Pack all trick-library recipes under recipe_dir (top-level *.json only)."""
    paths = sorted(Path(recipe_dir).glob("*.json"))
    return _load_recipe_vectors(paths, mode=mode)
