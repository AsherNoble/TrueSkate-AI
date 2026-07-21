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
- "spin_flick" — a single-finger flick with the spin HOLD guaranteed active and
             OUTLASTING the drag (curved_drag_with_spin_hold — no push, both
             touches label-accounted). The only spin form Model 1 can train on
             (meta stays flick-shaped); splits the spin_frac slice with "spin"
             at _SPIN_FLICK_SHARE.

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

# Stationary-touch ("static") arm — the Model 1 MVP. Hold durations span short to
# long so the model sees a wide range of touch lifetimes; the floor is above the
# ~0.2s a bare tap already renders, so a "hold" is always distinguishable from a tap.
_HOLD_MIN_S = 0.10
_HOLD_MAX_S = 1.50
# Share of the static slice that are instantaneous taps rather than holds. Taps are
# the Δ clapperboard; holds carry the timing supervision. Mirrors _SPIN_FLICK_SHARE.
_TAP_SHARE = 0.20

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

# spin_flick hold window (ABSOLUTE seconds, relative to the flick touch-down at
# payload t=0). Drag-first: the spin press joins >= 0.12s after the flick's
# touch-down — the same stagger execute_n_slot_gestures enforces between drag
# downs, because near-simultaneous multi-finger downs read as the park-editor
# camera gesture. The hold OUTLASTS the flick into the tail window, like real
# spin play (button held through the rotation), so the pressed button stays
# visible for >= ~12 frames at 30fps.
_SPIN_FLICK_HOLD_START = (0.12, 0.35)
_SPIN_FLICK_HOLD_END_AFTER_DRAG = (0.3, 0.9)   # added to the flick duration
_SPIN_FLICK_MIN_HOLD_S = 0.4

# Within the spin_frac slice: share drawn as spin_flick (Model-1 trainable,
# single finger + held button) vs "spin" (nslot + held button, Model-2 fuel).
_SPIN_FLICK_SHARE = 0.5


@dataclass
class GestureSample:
    """A sampled gesture in one of seven executable forms.

    kind == "flick": use waypoints/duration/easing_power (curved_drag, no push).
    kind == "spin_flick": flick fields + spin_hold_*_s
    (curved_drag_with_spin_hold, no push).
    kind in {"nslot","recipe","spin"}: use params/num_gestures/use_spin
    (execute_gesture_params).
    kind in {"hold","tap"}: use point/hold_duration_s (long_press / tap). These are
    STATIONARY touches — zero path length. They exist for the Model 1 MVP: a held
    point has an unambiguous (x, y) and a known onset AND liftoff, so it supervises
    localisation and touch timing without the direction/speed ambiguity a completed
    drag trace carries. Measured on-device (Stage 0): a stationary touch renders the
    normal orange mark at the commanded point, visible for hold + ~0.16s; a tap
    renders for ~0.2s (~6 frames at 30fps).
    """
    kind: str
    waypoints: list[tuple[float, float]] | None = None
    duration: float | None = None
    easing_power: float | None = None
    params: list[float] | None = None
    num_gestures: int | None = None
    use_spin: bool = False
    source: str | None = None  # recipe filename for kind == "recipe"
    # Stationary-touch kinds ("hold" / "tap"). hold_duration_s is 0.0 for a tap.
    point: tuple[float, float] | None = None
    hold_duration_s: float | None = None
    # spin_flick hold window, ABSOLUTE seconds from the payload start (= flick
    # touch-down). Params-spin kinds carry their window inside params; meta()
    # decodes it so every consumer reads the same named fields.
    spin_hold_start_s: float | None = None
    spin_hold_end_s: float | None = None
    # Stamped by the collector (worker.spin_button_xy) just before execution so
    # the logged coord always matches the button the held finger actually hit.
    spin_button_xy: tuple[float, float] | None = None

    def meta(self) -> dict:
        """JSON-serialisable description for the sample's meta.json."""
        d: dict = {"gesture_distribution": self.kind}
        if self.kind in ("flick", "spin_flick"):
            d.update(
                waypoints=self.waypoints,
                duration=self.duration,
                easing_power=self.easing_power,
            )
            if self.kind == "spin_flick":
                d.update(
                    spin_active=True,
                    spin_hold_start_s=self.spin_hold_start_s,
                    spin_hold_end_s=self.spin_hold_end_s,
                    # The hold outlasts the drag by construction, so the W3C
                    # payload (and the gesture call window the aligner anchors
                    # on) ends when the button lifts, not when the drag does.
                    payload_total_s=max(self.duration or 0.0, self.spin_hold_end_s or 0.0),
                )
        elif self.kind in ("hold", "tap"):
            # A stationary touch is fully described by where and for how long. The
            # label pipeline turns this into a single constant-position touch
            # interval (_TouchInterval(constant_xy=...)), the same representation
            # already used for the spin-button hold.
            d.update(
                point=[float(self.point[0]), float(self.point[1])] if self.point else None,
                hold_duration_s=float(self.hold_duration_s or 0.0),
                payload_total_s=float(self.hold_duration_s or 0.0),
            )
        else:
            d.update(
                params=self.params,
                num_gestures=self.num_gestures,
                use_spin=self.use_spin,
            )
            if self.source:
                d["recipe_source"] = self.source
            if self.use_spin and self.params:
                d.update(_params_spin_fields(self.params, self.num_gestures or 1))
        if self.spin_button_xy is not None and d.get("spin_active"):
            d["spin_button_xy"] = [float(self.spin_button_xy[0]), float(self.spin_button_xy[1])]
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


def sample_spin_flick(rng: np.random.Generator) -> GestureSample:
    """A single-finger flick with the spin (rotate) button HELD past the drag.

    The spin form Model 1 can train on: meta stays flick-shaped (waypoints/
    duration/easing_power pass the dataset's flick filter) and the held button
    is carried as an explicit hold window, so the extra finger is a labelled
    touch, never unmodelled noise. Drag-first, spin press joining >=
    _SPIN_FLICK_HOLD_START[0] later (editor mitigation — see constants above);
    the hold runs into the tail window like real spin play. Straight flicks are
    materialised as 3 waypoints (exact midpoint — same path) so meta ==
    execution == label math.
    """
    f = sample_flick(rng)
    wps = [tuple(p) for p in f["waypoints"]]
    if len(wps) == 2:
        (sx, sy), (ex, ey) = wps
        wps = [(sx, sy), ((sx + ex) / 2.0, (sy + ey) / 2.0), (ex, ey)]
    dur = float(f["duration"])
    hold_start = float(rng.uniform(*_SPIN_FLICK_HOLD_START))
    hold_end = dur + float(rng.uniform(*_SPIN_FLICK_HOLD_END_AFTER_DRAG))
    hold_end = max(hold_end, hold_start + _SPIN_FLICK_MIN_HOLD_S)
    return GestureSample(
        kind="spin_flick",
        waypoints=wps,
        duration=dur,
        easing_power=float(f["easing_power"]),
        spin_hold_start_s=hold_start,
        spin_hold_end_s=hold_end,
    )


def schedule_total_s(durations: list[float], delays: list[float]) -> float:
    """Nominal N-slot schedule length in seconds: earliest-start-normalised max
    slot end. Mirrors execute_n_slot_gestures' spin_total (pre-stagger — the
    collector's 0.12s finger stagger can stretch the real schedule slightly)."""
    starts = [0.0]
    for i in range(1, len(durations)):
        starts.append(starts[i - 1] + durations[i - 1] + delays[i - 1])
    base = min(starts)
    return max(s - base + d for s, d in zip(starts, durations))


def _params_spin_fields(params: list[float], num_gestures: int) -> dict:
    """Decoded spin provenance for a params-vector (spin-layout) sample.

    Emits spin_active plus the nominal hold window in ABSOLUTE seconds from
    schedule start — the same reference frame spin_flick uses — so stats and
    Model 2 never re-derive from the raw trailing [gate, t_start, t_end] block.
    """
    gate, t0, t1 = (float(v) for v in params[-SPIN_PARAMS:])
    if gate < 0:  # gate-off: spin block present but the hold never fires
        return {"spin_active": False}
    durations = [float(params[i * PARAMS_PER_SLOT + 6]) for i in range(num_gestures)]
    d0 = num_gestures * PARAMS_PER_SLOT
    delays = [float(v) for v in params[d0:d0 + max(0, num_gestures - 1)]]
    total = schedule_total_s(durations, delays)
    ts, te = sorted((t0, t1))  # unpack_gesture_params orders the window the same way
    return {
        "spin_active": True,
        "spin_hold_start_s": ts * total,
        "spin_hold_end_s": te * total,
        "payload_total_s": total,
    }


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


def sample_hold(rng: np.random.Generator, *, min_s: float = _HOLD_MIN_S,
                max_s: float = _HOLD_MAX_S) -> GestureSample:
    """A stationary press at a uniformly-sampled interior point, held min_s..max_s.

    The duration is what makes a hold worth more than a tap: the rendered mark
    lasts as long as the finger is down, so onset AND liftoff are both observable
    and the model has to learn touch timing, not just position.
    """
    x = float(rng.uniform(*_FLICK_START_X))
    y = float(rng.uniform(*_FLICK_START_Y))
    return GestureSample(kind="hold", point=(x, y),
                         hold_duration_s=float(rng.uniform(min_s, max_s)))


def sample_tap(rng: np.random.Generator) -> GestureSample:
    """An instantaneous tap at a uniformly-sampled interior point.

    Kept as its own arm because a tap is the crispest possible timing marker: it
    renders a ~0.2s mark with a sharp onset, which is what makes it usable as the
    command->pixel (Δ) clapperboard.
    """
    x = float(rng.uniform(*_FLICK_START_X))
    y = float(rng.uniform(*_FLICK_START_Y))
    return GestureSample(kind="tap", point=(x, y), hold_duration_s=0.0)


def clamp_in_bounds(s: GestureSample) -> GestureSample:
    """Defensive chokepoint: guarantee a sampled gesture lies within the RL
    coordinate bounds (X_BOUND_MIN..MAX, Y_BOUND_MIN..MAX) AND outside the top-left
    Bolt-Challenges indicator rect (_BOLT_EXCL_*), so the collector can never
    execute — or mislabel — an out-of-bounds action or one that opens the Bolt modal,
    whatever the sampling path or any future change. Clamps the SAMPLE itself
    (mutates in place), so the saved label always matches what executes. Normalised
    coords are absolute (0/1 = screen edges); clamping constrains, it never rescales.
    """
    if s.kind in ("flick", "spin_flick") and s.waypoints is not None:
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
        ex, ey = _restore_min_reach(sx, sy, *pushed[-1])
        # The reach restore extends along start->end, which can land the end
        # BACK inside the bolt rect (end pushed to the rect's edge with the
        # start further right → the direction points left). Zone exclusion is
        # the hard invariant (it opens the Bolt modal), so it gets the last
        # word — accepting a rare shorter-than-_FLICK_MIN_REACH trace.
        pushed[-1] = _push_out_bolt_zone(ex, ey)
        s.waypoints = pushed
    elif s.kind in ("hold", "tap") and s.point is not None:
        s.point = _push_out_bolt_zone(
            float(np.clip(s.point[0], X_BOUND_MIN, X_BOUND_MAX)),
            float(np.clip(s.point[1], Y_BOUND_MIN, Y_BOUND_MAX)))
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
    static_frac: float = 0.0,
    num_gestures: int = 2,
    use_spin: bool = False,
    recipe_vectors: list[tuple[list[float], int, bool, str]] | None = None,
) -> GestureSample:
    """Draw one gesture from the flick / nslot / recipe / spin / static mixture,
    guaranteed within the coordinate bounds (via clamp_in_bounds).

    fracs = (flick, nslot, recipe) is the non-spin base mix; if no recipes are
    available the recipe share is redistributed to nslot (so the mix never silently
    stalls). spin_frac is a TRUE share of all fires in [0, 1]: the base mix keeps
    its internal ratios but is scaled to the remaining (1 - spin_frac), so
    spin_frac=0.2 yields ~20% guaranteed-spin gestures (a held rotate button)
    whatever fracs sums to. It is the knob to grow the spin-family corpus,
    independent of `use_spin` (which only makes the plain nslot branch
    spin-capable, ~half of those gate-off). The spin slice itself splits
    _SPIN_FLICK_SHARE spin_flick (Model-1 trainable) / rest nslot-spin.

    static_frac is the same kind of true share for STATIONARY touches (the Model 1
    MVP arm): it splits _TAP_SHARE taps / rest holds. static_frac=1.0 gives a
    pure hold/tap run, which is what the MVP collection uses.
    """
    f_flick, f_nslot, f_recipe = fracs
    if not recipe_vectors:
        f_nslot += f_recipe
        f_recipe = 0.0
    # spin_frac/static_frac as raw weights would dilute: e.g. defaults (sum 1.0)
    # + 0.2 give 0.2/1.2 ≈ 17%, not 20%. Scale the base mix to the remainder
    # instead so the advertised shares are exact.
    f_spin = min(1.0, max(0.0, spin_frac))
    f_static = min(1.0, max(0.0, static_frac))
    if f_spin + f_static > 1.0:
        raise ValueError(
            f"sample_mixture: spin_frac + static_frac must be <= 1.0, "
            f"got {f_spin} + {f_static}")
    base_total = f_flick + f_nslot + f_recipe
    if base_total <= 0 and f_spin <= 0 and f_static <= 0:
        raise ValueError("sample_mixture: all mixture weights are zero")
    if base_total > 0:
        scale = (1.0 - f_spin - f_static) / base_total
        f_flick, f_nslot, f_recipe = f_flick * scale, f_nslot * scale, f_recipe * scale
    total = f_flick + f_nslot + f_recipe + f_spin + f_static
    r = float(rng.uniform(0, total))
    if r >= total - f_static:
        # static slice: taps are the crisp Δ marker, holds carry touch timing
        s = (sample_tap(rng) if float(rng.uniform(0.0, 1.0)) < _TAP_SHARE
             else sample_hold(rng))
        return clamp_in_bounds(s)
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
    elif float(rng.uniform(0.0, 1.0)) < _SPIN_FLICK_SHARE:
        s = sample_spin_flick(rng)
    else:
        s = sample_spin(rng, num_gestures)
    return clamp_in_bounds(s)


def load_recipe_vectors(recipe_dir: Path, mode: str = "median") -> list[tuple[list[float], int, bool, str]]:
    """Pack all trick-library recipes under recipe_dir (top-level *.json only)."""
    paths = sorted(Path(recipe_dir).glob("*.json"))
    return _load_recipe_vectors(paths, mode=mode)
