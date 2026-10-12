# Execution transfer function: is the trace wrong in a predictable way?

Draft design, 2026-10-12. Nothing here has run on a device. L0 is laptop-only;
L1 needs explicit operator authorization (bounded rig time, RIG-BUSY lock,
healthy WDA `ae50404a` untouched).

## The idea in one line

Model 1 does not need the executor to be *faithful*; it needs it to be
*deterministic, observable and correctly labelled*. If the on-screen touch is a
fixed function of the command, Model 1 learns that function's inverse from the
data, and commands it emits reproduce the observed touch when replayed through
the same executor.

## Model of what happens

Labels today: command waypoints, linearly interpolated in command time, anchored
at `onset_s = fit(submitted_to_ios)`; the fit comes from die-five marker taps in
the same recording (`capture.py:156`, `dataset.targets`). Decompose the touch
actually drawn on screen as

    a(t) = W_c(t - onset_s - delta) + noise

- `W_c`: deterministic, command-dependent warp/shape (XCTest interpolation,
  segment boundary handling, holds, game-side trail rendering).
- `delta`: per-request start offset (XCTest start-up latency varies per request;
  markers and samples may also differ systematically).
- `noise`: everything not repeatable.

| Component | If unpredictable | If predictable or measurable |
|---|---|---|
| `W_c` | label noise Model 1 cannot remove | absorbed: Model 1 learns `W^-1` (same-executor replay is exact) |
| `delta` | onset/timing label noise (about ±1–2 frames seen) | correct labels per clip (WDA timing or observed onset) |
| `noise` | irreducible; must be < 1 frame | n/a |

Two limits to the "Model 1 absorbs it" argument:

1. **Observability.** If `W` maps different commands to the *same* video, the
   label is ambiguous and no model can recover it. Example: if a slow segment
   followed by a hold is executed as an instant jump plus a longer hold, it looks
   identical to a genuinely fast segment plus a long hold.
2. **Executor-specific labels.** On human clips Model 1 would output
   `W^-1(human touch)`: the command that reproduces the human touch *through
   this WDA executor*. That suits replay via the same executor; a different
   executor (Pico HID) needs `W` written down explicitly. So characterise `W`
   either way.

## What round 6 already suggests (offline, exploratory)

- Per-clip offsets have both signs and are mostly within about ±30 ms where the
  trace is measurable (e.g. clip 67 leads by a steady ~25 ms; clip 159 is ~0
  in its first segment, then +20 to +35 ms).
- Clip 67 has a ~170 ms lead just before its final 172 ms hold (visually
  confirmed). Apparent leads in 85 and 127 were tracker artifacts (the paths run
  along bright straight scenery at the horizon). So the "hold absorbed into the
  preceding move" hypothesis is **not established**.
- Ratings: gestures whose hold comes first were 5/5 Meet. Gestures with a mid
  or late hold failed 4/8, against 12/92 for the rest (Fisher p ≈ 0.02; post
  hoc, n = 8).
- The deployed WDA turns every `pointerMove`, including a zero-length hold, into
  `moveToPoint:atOffset:(offset+duration)`. Any hold or segment distortion
  therefore comes from XCTest's interpolation or delivery, which is
  deterministic code and testable without a device.
- The video tracker loses the head under the board, on self-overlapping paths,
  on pale ground and along collinear scenery. Video alone is a weak instrument
  for fine timing.

## Experiment

### L0: XCTest synthesis in the iOS Simulator (laptop, free, no device)

Use the WDA repository's simulator test harness: build each payload through
`FBW3CActionsSynthesizer` and
(a) dump the synthesized event record (pointer events with offsets and points),
and (b) deliver it to a touch-logging view that records `UITouch` location and
timestamp.

Probes:

- Single moves of 100, 200, 400, 800 and 1200 ms, at fixed length.
- Two-segment kinks at 30°, 90° and 150°.
- Holds of 50, 100 and 200 ms, placed first, in the middle and last.
- 1, 2, 5 and 10 segments at a fixed total duration.
- All 100 exact round-6 payloads.

This gives the interpolation law (linear or eased), event rate, hold handling
and kink handling exactly, with no measurement noise.

### L1: device repeatability (XR1, bounded, needs authorization)

- **Commands:** the L0 probe set (~40 commands) plus 20 round-6 payloads (10
  Meet, 10 Fail, including every mid/late-hold gesture). Each command is
  repeated 5 times, randomised and interleaved, giving ~300 requests in about
  30 one-minute recordings with die-five calibration.
- **Placement:** one park with plain ground, paths kept clear of the resting
  board and the horizon.
- **Measurements:** per-request WDA timing (`preparation`, `submitted_to_ios`,
  `ios_completion_callback`), plus video lag with an improved tracker: mask
  scenery that is collinear with the path using pre-onset frames, and
  cross-check against the human onset marks.

### L2: analysis (freeze before looking at L1 data)

Per request:

- `delta` = median lag in the first segment.
- Warp `w(tau)` = lag − `delta`.

| Check | Criterion |
|---|---|
| D1 deterministic | SD of `w` across the 5 repeats ≤ 8 ms (half a frame) at ≥ 90% of measurable frames |
| D2 predictable | a `W` fitted on probes (or taken from L0) predicts held-out replays within 1 frame at ≥ 95% of measurable frames |
| D3 offset correctable | `delta` predicted within 1 frame for ≥ 95% of requests, from WDA timing or the observed onset |
| D4 observable | no command class collapses (segment shorter than 1 frame, lost hold, merged kinks) |

Outcomes:

- **All pass:** reframe qualification as below; no executor change.
- **D4 fails for a class (e.g. holds):** drop that class from sampling, or
  change how it is sent (e.g. W3C `pause` instead of a zero-length move), and
  re-check in L0.
- **D3 fails:** re-anchor each clip's onset from the video.
- **D1 fails:** the executor is stochastic, so fidelity (not prediction) is
  required; investigate before collecting.

Cheap pre-L1 checks (read-only file copies, no device actions): correlate
per-clip `delta` from round 6 with each request's WDA `submitted→callback`
minus duration (in the rig `wda-timing.json`, not in the local mirror), and
with segment count and duration.

## Reframed qualification (proposal for the operator)

1. **Automated per clip:** the tracker trace agrees with the *predicted*
   on-screen trace `Ŵ(command)` within 0.03 (normalized) / 1 frame wherever it
   is measurable. The review ring shows `Ŵ(command)`, not the raw command.
2. **Human per clip:** only what automation cannot check: menu/HUD,
   extra contacts, truncation, gross misbehaviour, plus spot checks of the
   predicted-trace ring.
3. **Labels:** remain the commands actually sent (Model 1 absorbs `W`), with the
   onset re-anchored per clip if D3 requires it.
4. **Provenance:** `W` is versioned with the executor (WDA revision and XCTest/iOS
   version), because Model 1 labels are executor-specific.
