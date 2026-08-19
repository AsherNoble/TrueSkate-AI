# Plan — Model 1 MVP 3: curved drags, no spin

Written 2026-08-18. Successor to MVP 2 (two-point constant-velocity linear drag,
current leader 94.12% strict joint recovery). Scope: a single-finger **curved**
drag with **no spin control active**.

## The controlling finding: the "curve" is a polyline, not a spline

`curved_drag` / `build_curved_drag` (`src/trueskate_ai/sim/touch_actions.py:154`)
do **not** fit a Bezier or spline. They chain W3C `create_pointer_move` calls
straight through the waypoints, so a 3-waypoint gesture is exactly **two straight
segments** meeting at the middle waypoint. Within each segment the pointer moves
linearly in time.

`easing_to_segment_durations` (`touch_actions.py:139`) then splits the total time
by evaluating the easing at each segment boundary. For the project's
`easing_power` convention (`easing(t) = t**p`, GESTURES.md) and 2 segments:

    boundaries = [0, 0.5**p, 1]   ->   segment 1 gets s_k = 0.5**p of the duration

| easing_power | 0.60 | 0.80 | 1.00 | 1.25 | 1.50 | 2.00 | 3.00 |
|---|---|---|---|---|---|---|---|
| knot time `s_k` | .660 | .574 | .500 | .420 | .354 | .250 | .125 |

Inverting, `p = log(s_k) / log(0.5)`. **`easing_power` *is* the bend's time**, and
the bend is directly visible in the clip. So the whole gesture is exactly
identifiable from pixels — no hidden parameter — provided the bend itself is
visible (see the contract below).

Two consequences:

1. **The MVP-2 line-fit decoder generalises directly.** A polyline is *linear in
   its control points*: with piecewise-linear hat basis functions `B0(s)`,
   `Bm(s)`, `B1(s)` and a knot at `s_k`, per-frame contact positions give a
   closed-form weighted least-squares solve for `(p0, pm, p1)` — the same
   machinery as `_fit_constant_velocity`, a 3x3 normal-equation solve instead of
   2x2, with IRLS reweighting unchanged. Build MVP 3 on `--line-fit` from day
   one; do **not** re-derive the two-soft-argmax baseline.
2. **The MVP-2 tail lesson transfers.** The 2026-08-18 autopsy showed the end
   endpoint suffers a systematic along-path undershoot (mean -0.0095, negative in
   85% of clips) because the rendered trail is cumulative and a soft-argmax
   averages backward from the tip. MVP 3's end endpoint has the identical
   exposure; its mid waypoint does not (it has trail on both sides). Expect the
   end to remain the weakest point and budget for the same bias correction.

## Contract (the part that must be right before any collection)

A trainable MVP-3 event is one single-finger, **three-waypoint**, non-spin drag,
labelled `[x0, y0, xm, ym, x1, y1, duration, easing_power]`.

**The identifiability constraint is the critical addition.** If `pm` is collinear
with the `p0 -> p1` chord *and* `s_k = 0.5`, the gesture is pixel-identical to a
straight constant-velocity drag and `pm` is simply unrecoverable — an irreducible
label error of exactly the kind the MVP-2 autopsy proved we do not currently have
and must not introduce. Therefore:

- **Minimum bend:** perpendicular distance from `pm` to the `p0-p1` chord must be
  **>= 0.04** normalised — comfortably above the 0.03 recovery tolerance.
- Note the existing `sample_flick` (`gesture_sampling.py:232`) jitters the mid
  waypoint by only +/-0.08 around the chord midpoint, so a large share of its
  curves are near-collinear. **MVP 3 needs its own sampler**; do not reuse
  `sample_flick`.
- Carry over from MVP 2: per-segment reach floor (each segment >= 0.12 normalised)
  so neither leg degenerates; duration in `0.30-1.20s`; no taps, holds,
  multi-touch, or spin. `spin_active` must be false.
- **Stage 3a** fixes `easing_power = 1.0` (knot at `s_k = 0.5`) and learns shape
  only: 7 targets. **Stage 3b** samples `easing_power` in `[0.6, 2.0]` and adds
  the 8th. Splitting these keeps the first result interpretable — a Stage-3a
  failure is a shape problem, not a timing one.

## Acceptance gate, and an honest expectation

Same tolerances as MVP 2: every waypoint within **0.03**, duration within
**0.10s**. But the gate now spans **three** points, so it is strictly harsher:

| per-point recovery | joint (3 pts) | x duration 0.987 |
|---|---|---|
| MVP-2's actual (start 1.000, end 0.954), mid like end | 91.01% | 89.83% |
| all three at 0.98 | 94.12% | 92.90% |
| all three at 0.99 | 97.03% | 95.77% |

**Do not expect MVP 3 to open near MVP 2's 94%.** Simply adding a third gated
point to today's per-point accuracy lands near 90%. Reaching a 95% gate requires
roughly 99% per waypoint. State this before the first run so the first number is
not misread as a regression.

Stage 3b adds an `easing_power` tolerance, which needs its own declared value.
Recommend deriving it from the knot time rather than picking one blind: a knot
tolerance of +/-0.03 normalised clip time maps to a wider `p` tolerance near
`p = 0.6` than near `p = 2.0`, so a flat tolerance on `p` is not a flat tolerance
on anything physical. **Gate on recovered knot time, and report `p` derived from it.**

## Work items

1. **Sampler + contract** — `sample_basic_curve_mixture` in
   `src/trueskate_ai/data/gesture_sampling.py` alongside the existing
   `sample_basic_linear_mixture`, with `BASIC_CURVE_*` constants (min bend, min
   segment reach, duration range). Mutually exclusive with the linear and hold
   modes, its own persisted per-device seed, its own corpus root.
2. **Strict loader** — `basic_curve_dataset.py` mirroring `basic_linear_dataset.py`:
   admit only calibrated, menu-clean, non-spin, exactly-3-waypoint clips meeting
   the bend and reach floors; command key covers all three waypoints plus duration
   and easing.
3. **Decoder** — extend `BasicLinearRegressor`'s fit to a knotted polyline
   (`_fit_polyline`, 3x3 solve, same IRLS). Reuse `_frame_positions` verbatim.
   Whether this is a flag on the existing class or a sibling depends on how much
   the head shapes diverge; prefer extending, since the basis change is the only
   real difference.
4. **Collector mode** — `--basic-curves` in `scripts/data/collect_sls_xctest.py`,
   with the same `--tap-calibrate --no-reset` timing-gate requirement that makes
   MVP-2 labels trustworthy.
5. **Protocol** — exact-command holdout, device-balanced, validation-only
   selection, single test evaluation. Unchanged from MVP 2 and not up for
   renegotiation after seeing a number.

## Sequencing note

MVP 3 needs new collection, and collection is currently paused. It is therefore
gated behind the MVP-2 endpoint work finishing, which needs no rig at all. The
cheap, rig-free part of MVP 3 that can start now is items 1-3: sampler, loader,
and the polyline solver, all of which are unit-testable against synthetic tracks
exactly as the MVP-2 line fit was.
