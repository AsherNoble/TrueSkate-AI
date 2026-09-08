# Plan — Model 1 MVP 3: fixed-time trajectory sampling

Written 2026-08-19. Supersedes the earlier MVP-3 draft, which was built around
"three waypoints plus an easing power". Asher's constraint is that the gesture
object must eventually approximate arbitrary shapes — a Z, not just a bend — so
the representation is chosen now for that endpoint even though MVP 3 itself stays
restricted to basic curves.

## The representation decision

**Stop predicting semantic waypoints and easing. Predict the finger's position at
K fixed, evenly-spaced times, plus the duration.**

Target: `[(x_0,y_0), (x_1,y_1), ..., (x_{K-1},y_{K-1}), duration]`, where point `k`
is where the finger was at `t = k/(K-1) * duration`.

Three facts make this the right choice, all verified against the executor:

1. **`curved_drag` is already a polyline.** It chains W3C `create_pointer_move`
   calls through the waypoints (`sim/touch_actions.py:154`); there is no spline.
   So every gesture the executor can physically produce *is* a polyline, and a Z
   is simply a 4-waypoint one. Arbitrary shape is already expressible — the
   question was only ever how to parameterise it.
2. **The representation is execution-complete.** With `easing=None`,
   `easing_to_segment_durations` splits time equally across segments. So replaying
   the K predicted points as `curved_drag(points, total_duration=duration,
   easing=None)` reproduces exactly the sampled trajectory. No decoding step, no
   waypoint-fitting, no easing inversion.
3. **It absorbs easing instead of predicting it.** A command with *any*
   `easing_power` can be resampled to positions at evenly-spaced times (segment
   durations are deterministic, interpolation within a segment is linear). Verified:
   `p=2.0, K=5` gives knot times `[0, .062, .249, .561, .998]`, which resample
   cleanly. **`easing_power` therefore disappears as a predicted parameter** — the
   time-warp is encoded in *where the points are*. One fewer head, one fewer
   tolerance to invent, and the collector may keep firing any easing it likes.

**This also removes the degeneracy that forced the old plan's minimum-bend rule.**
The earlier draft required the middle waypoint to sit >= 0.04 off the chord, because
"where is the bend" is unanswerable for a straight gesture. Under fixed-time
sampling the question is instead "where was the finger at half time", which always
has an answer, including for a perfectly straight constant-velocity drag (it is the
midpoint). **Drop the minimum-bend constraint** — it was an artefact of the old
parameterisation and would have needlessly restricted collection.

## The physical ceiling on shape fidelity

Shape detail is capped by contact frames, i.e. `duration x 30fps`. Allowing >= 2
frames per segment:

| duration | contact frames | max usable K |
|---|---|---|
| 0.20s | 6 | 4 |
| 0.30s | 9 | 5 |
| 0.50s | 15 | 8 |
| 0.80s | 24 | 13 |
| 1.20s | 36 | 19 |

**A Z performed in 0.3s is not recoverable at 30fps, by any model.** Complex shapes
require either longer gestures or a higher-rate capture. This belongs in the MVP-n
roadmap as a stated limit, not a surprise discovered later.

## Roadmap: only K and the collection contract change

| | K | shapes | notes |
|---|---|---|---|
| MVP 3 | **3** | basic curves (quadratic-like) | matches the existing recipe schema's 3-waypoint slot exactly |
| MVP 4 | 5 | S-curves, shallow multi-bends | needs duration >= 0.30s |
| MVP 5+ | 9 | Z and sharp direction changes | needs duration >= 0.60s |

**The architecture never changes across these — only `K`.** That is the whole point
of the choice. Each step is a re-collection and a retrain, not a redesign.

## Acceptance gate, honestly stated

Gate: **every one of the K points within 0.03**, and duration within 0.10s — i.e.
max deviation along the path. That is the execution-relevant criterion (the path
must be right *everywhere*, not on average), and it degrades sharply with K:

| per-knot | K=3 | K=5 | K=9 | K=17 |
|---|---|---|---|---|
| 98.0% | 94.1 | 90.4 | 83.4 | 70.9 |
| 99.0% | 97.0 | 95.1 | 91.4 | 84.3 |
| 99.5% | 98.5 | 97.5 | 95.6 | 91.8 |

**MVP 3's 95% target needs ~98.5% per knot.** MVP 2's current per-endpoint figures
are 100% start / 95.4% end, so the end-point work already in flight is a
prerequisite, not a parallel nicety. Report per-knot recovery alongside the joint
number so progress stays visible when the joint gate is still failing.

## Decoder

The MVP-2 robust line fit generalises directly and should be reused, not replaced:

- A fixed-time-knot polyline is **linear in its knot positions**. With piecewise-linear
  hat basis functions over the K knots, per-frame contact positions give a
  closed-form weighted least-squares solve — the existing `_fit_constant_velocity`
  with a K x K system instead of 2 x 2, and IRLS reweighting unchanged.
- Knot times are *fixed*, so there are no free breakpoints to estimate. The only
  timing unknowns remain onset and duration, exactly as in MVP 2.
- `_frame_positions` is reused verbatim.
- The MVP-2 autopsy finding still applies to the **last** knot: it is the terminus of
  a cumulative trail and will show the same along-path undershoot. Interior knots
  have trail on both sides and should not. Carry the validation-fit bias correction.

## Work items

1. `resample_command_at_times()` — given commanded waypoints + duration +
   easing_power, return positions at K evenly-spaced times. Pure function, exactly
   mirrors `easing_to_segment_durations` + within-segment linear interpolation.
   **This is the keystone: it converts every existing and future command into the
   new target, whatever its easing or waypoint count.**
2. `sample_basic_curve_mixture()` in `data/gesture_sampling.py` — MVP-3 contract:
   3 waypoints, non-spin, duration 0.30-1.20s, per-segment reach floor so neither
   leg degenerates. **No minimum-bend rule.**
3. `basic_curve_dataset.py` — strict loader mirroring `basic_linear_dataset.py`,
   emitting the K-point resampled target. Admit only calibrated, menu-clean,
   editor-clean, non-spin clips.
4. `_fit_polyline()` in `basic_linear_regressor.py` — K x K weighted least squares
   with IRLS, generalising `_fit_constant_velocity`.
5. Collector `--basic-curves` mode, with the `--tap-calibrate --no-reset` timing
   gate that makes labels trustworthy.
6. Protocol unchanged: exact-command holdout, device-balanced, validation-only
   selection, single test evaluation.

Items 1-4 need no rig and are unit-testable against synthetic tracks, exactly as the
MVP-2 line fit was. Item 5 needs collection, which is currently paused.
