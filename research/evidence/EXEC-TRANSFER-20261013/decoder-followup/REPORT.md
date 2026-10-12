# Decoder floor follow-up — EXEC-TRANSFER-20261013

The frozen decoder's 95/100 perfect-target result is reproducible, but is **not
a representability ceiling imposed by the existing 100 ms gesture contract**.
The smallest successful research ablation keeps every selected knot time and
refits only its internal coordinates. It scores 100/100 after integer viewport
quantization and compilation through the frozen gesture/payload implementation.
No metric, collection gate, production decoder, or contract was changed.

More capable free-knot and boundary-interval fits also score 100/100. Their
advantages become visible on independent synthetic cases, but they do not cure
imposed alignment error. These are offline representability diagnostics, not
trained-model results, qualification, human review, or executed-path fidelity.

## Frozen inputs and verification

- Implementation: `59f1782d7a5d43228d3232cb3bf9cce16bc52aa0`, read-only checkout
  `tmp/worktrees/curved-model1-v1`.
- Already-reviewed round-6 manifest: SHA256
  `b3daad332ce5b1f575f83b73acc4fc34af429c50003dd5ea95a28a78286bcfca`.
  All 100 metadata hashes and all three implementation file hashes are recorded
  in `provenance.json`; they match the earlier oracle diagnostic.
- Oracle input coordinates here are first cast to `float32`, exactly as
  `dataset.targets()` supplies model targets. Perfect onset/release logits select
  the nearest native frames. Refitting functions receive only these coordinates,
  native times, selected boundary indices and the baseline prediction; they do
  not read command knots, calibrated event times, or the original command.
- An independent minimax dynamic program exactly reproduces baseline points,
  integer times and residuals for **all 964 audit/synthetic cases**, asserted
  during the run. The baseline audit again yields 95 passes and 71 fits at .01.
- All final scores call the unchanged frozen `evaluate_path`: normalized
  Euclidean error ≤.03 for ≥95% of inclusive contact frames, with both selected
  boundaries within two frames. No relaxation or alternative metric selects a
  passing result. Missing frames remain penalized.
- The three integer/quantized research variants pass the existing duration,
  interval, segment-cap, normalized-bound and quantized-segment control-map
  checks on **all 100+864 cases**. Each also successfully constructs a frozen
  `Gesture`, compiles its original fitted coordinates to the W3C payload, and
  asserts that the command's integer positions exactly match the re-scored
  path. Payload hashes are stored. This verifies offline compilation, not device
  execution. Quantization is truncation to 414×896 logical coordinates.
- A separate compiled-clock check applies each clip's known calibration rate
  to the emitted integer command times, recomputes the actual nearest release
  frame, and uses the unchanged path evaluator. All three research variants
  retain100/100 audit passes and respectively838/851/858 synthetic passes.
  This checks the command-clock/video-clock distinction under the existing
  calibration hypothesis; it does not measure future replay timing.
- Fourteen deterministic repeat fits, including all five original failures,
  reproduce the integer-fit prediction exactly. All starts are deterministic;
  no original knots seed the optimizer. Local optimization is not guaranteed
  to find a global minimax solution.

## What removes the audit floor

| Variant | Audit passes /100 | Audit .01 fits /100 | Synthetic passes /864 | Synthetic .01 fits /864 |
|---|---:|---:|---:|---:|
| Frozen baseline | 95 | 71 | 810 | 534 |
| Fixed knot times; refit internal coordinates; integer viewport | **100** | 73 | 838 | 555 |
| Fit fractional knots; round to legal integer ms; integer viewport | **100** | 87 | 851 | 602 |
| Fit boundaries inside their frame intervals plus fractional knots; legal integer ms/viewport | **100** | 96 | **858** | 749 |

The baseline uses each selected frame's coordinate as a mandatory polyline knot.
It therefore optimizes the placement of a subset of observed samples, rather
than fitting a general piecewise-linear path. A true turn between frames has no
sample at its apex. Regressing internal knot positions with fixed times already
removes all five audit failures, at negligible additional cost. It is the best
first research candidate because it isolates one restriction.

The coordinate refit fixes endpoints, builds the piecewise-linear hat-basis
matrix at the existing knots, and solves two least-squares systems for internal
coordinates. Coordinates are clipped to normalized bounds, then compiled and
re-scored. It uses the existing selected segment count and timeline.

The free-knot fit additionally allocates durations on a simplex with a hard
100 ms minimum and the existing ten-segment cap. It tries the baseline segment
count through the feasible cap, using baseline-derived and uniform deterministic
starts, fixes endpoints, and fits internal positions. Integer rounding projects
the schedule back onto ≥100 ms intervals before quantization and re-scoring.
All outputs retain 300–1200 ms durations. This is a bounded local least-squares
research solver, not an exact free-breakpoint minimax algorithm.

The interval variant treats a nearest-frame onset/release class as its Voronoi
time interval. It jointly fits continuous event times within those intervals,
internal knots and positions, retaining hard duration/interval bounds. It
fits **all** observations at actual native times, including overwritten boundary
targets. Integer rounding keeps release inside its original classified interval.
An inferred fractional boundary is not independent timing evidence; equivalent
fits may remain possible.

The pass criterion is less demanding than the internal max-error .01 target.
For example, original failure 77 is rescued by the integer free-knot fit with
30/31 good contact frames (96.77%), despite a residual above .03 at one frame.
Thus 100/100 passes must not be described as perfect path reconstruction.

## What does not explain away the floor

| Isolated ablation | Audit passes /100 | Synthetic passes /864 |
|---|---:|---:|
| Remove integer-ms rounding, retain sampled knot positions/times | 95 | 810 |
| Require calibrated `100×clock_rate` ms on the rounded frame grid | 93 | 744 |
| Supply exact event boundaries, retain frame-grid knots/coordinates (privileged oracle) | 94 | 703 |
| Lower minimum interval to99 ms (incompatible hypothesis) | 95 | 810 |
| Lower minimum interval to95 ms (incompatible hypothesis) | 95 | 810 |
| Lower minimum interval to83 ms (incompatible hypothesis) | 95 | 812 |
| Lower minimum interval to66 ms (incompatible hypothesis) | 96 | 818 |
| Lower minimum interval to33 ms (incompatible hypothesis) | 100 | 855 |

Sub-millisecond rounding is not the main audit limitation. Nor does inserting
the calibration rate into a frame-grid spacing test repair the mismatch: a
nominal100 ms grid edge becomes forbidden whenever `clock_rate>1`, forcing a
longer sampled edge or a smaller feasible segment count. Original audit rates
span approximately0.999677–1.000622, but even this small range can cross a hard
spacing boundary. Free knots avoid demanding that true turns land on frames.

Lowering segment spacing is explicitly outside curved-v1. It can improve
approximation by using extra short edges, but is unnecessary to eliminate this
audit's pass floor. No recommendation to change the100 ms contract follows.

## Synthetic challenge and endpoint-label artifact

The 864 new deterministic cases form the complete factorial of four shapes
(line, kink, alternating turns, movement–hold–movement), four intervals
(100/103/117/151 ms), three amplitudes(.08/.25/.48), six onset phases
(0/2/5/8/11/14 ms), and three clock rates(.999/1/1.001), on a native60 Hz
timeline. All actual commands satisfy curved-v1, and all cases are scored.
This deliberately concentrated challenge is not a prevalence estimate or a
validation/test cohort.

Baseline failures concentrate in large-amplitude, minimum-duration turns near
unfavourable frame phases. All288 amplitude.08 cases pass the baseline;237/288
amplitude.48 cases do. The final interval fit's six remaining failures are
the .48-amplitude,100 ms alternating-turn/hold cases at8 ms phase, at each rate.
Each has18/19 good frames:94.7368%, which misses the unchanged95% requirement.

An exact original command, evaluated at its true calibrated event times, scores
100/100 audit but only856/864 synthetic. This is a **label/evaluation convention
artifact**, not loss in executing an original command: both `targets()` and
`evaluate_path()` overwrite the nearest onset/release frame positions with exact
endpoints, even when that frame is inside the real contact interval. A fast
command can move more than.03 between the actual event and the nearest frame.
One bad endpoint frame in a19-frame contact is already>5%.

In the audit, the largest endpoint overwrite is approximately.03270 normalized
distance and only one sample exceeds.03 at either boundary. Twelve exact
commands exceed the internal.01 maximum because of endpoint overwrites, although
all pass the .03/95% score. Endpoint phases are up to8.05 ms from the nearest
frame. `endpoint-diagnostics.json` preserves every case.

The privileged exact-boundary/rate-aware fitter attains an internal event-based
.01 fit on all864 synthetic cases yet scores856/864 under the frozen frame
metric, independently exposing this distinction. Its internal objective moves
boundary observations to the supplied event times; its fit count is therefore
not directly comparable with the main table. No label or metric change was
made or used for acceptance. Any future proposal to harmonize event/frame
semantics needs separate review and synthetic compatibility tests.

## Imposed timing shifts remain costly

The earlier uniform-shift experiment is repeated using float32 coordinates and
the same unchanged evaluation against the original commands. These are
sensitivity probes, not measured noise rates. Counts are out of100.

| Imposed shift | Baseline | Coordinate refit/quantized | Free-knot integer/quantized | Interval integer/quantized |
|---|---:|---:|---:|---:|
| −16.67 ms | 70 | 72 | 70 | 70 |
| −8.33 ms | 80 | 81 | 83 | 84 |
| 0 ms | 95 | 100 | 100 | 100 |
| +8.33 ms | 82 | 81 | 81 | 83 |
| +16.67 ms | 68 | 67 | 67 | 71 |

Decoding away the perfect-target floor does not establish robustness to actual
temporal alignment error. Some shifted cases regress. Better decoding and
independent timing accuracy remain separate requirements.

## Cost and next experiment

Measured on this laptop across the100 audit cases; these are CPU decoder costs
with perfect targets, not end-to-end inference or training-validation timings.

| Operation | Median | 95th percentile | Maximum | Total across100 |
|---|---:|---:|---:|---:|
| Baseline decode | 3.63 ms | 12.98 ms | 14.63 ms | .535 s |
| Additional coordinate refit | .056 ms | .081 ms | .157 ms | .0059 s |
| Additional integer free-knot fit | 2.65 ms | 41.59 ms | 167.82 ms | .951 s |
| Additional interval fit | 5.04 ms | 91.33 ms | 185.66 ms | 1.376 s |

Simply scaling these observed totals to1250 clips gives approximately.074 s
additional coordinate-refit cost,11.9 s free-knot cost, or17.2 s interval-fit
cost per evaluation, beyond baseline. This is arithmetic extrapolation from
this small audit, not a validation benchmark. Model prediction noise and
different path/segment distributions can change optimizer convergence and cost.

Prioritize a synthetic noise study of the minimal fixed-time coordinate refit
before considering production changes: independent vs temporally correlated
coordinate noise, biased endpoints, exact vs shifted boundary classes, with
frozen seeds and unchanged acceptance. Compare its safety/rounding checks and
worst latency against the more capable solvers. No real training, new research
tranche, holdout selection, cloud work, rig access, or gesture execution was
performed or is implied by this recommendation.

## Reproduction and artifacts

Run from any directory with the repository `.venv` interpreter. Absolute paths
are preserved in `provenance.json`; `--synthetic-stride 1` means every synthetic
case receives every fitted variant.

```sh
'/Users/ashernoble/Projects/Robotics & hardware/TrueSkate-AI/.venv/bin/python' \
  '/Users/ashernoble/Projects/Robotics & hardware/TrueSkate-AI/tmp/worktrees/executor-research-20261013/research/evidence/EXEC-TRANSFER-20261013/decoder-followup/decoder_followup.py' \
  --source '/Users/ashernoble/Projects/Robotics & hardware/TrueSkate-AI/tmp/worktrees/curved-model1-v1' \
  --mirror '/private/tmp/claude-503/-Users-ashernoble-Projects-Robotics---hardware-TrueSkate-AI/e0f1c48e-973a-4ff3-a26a-f5f2350d9a43/scratchpad/audit6-mirror' \
  --out '/Users/ashernoble/Projects/Robotics & hardware/TrueSkate-AI/tmp/worktrees/executor-research-20261013/research/evidence/EXEC-TRANSFER-20261013/decoder-followup' \
  --synthetic-stride 1
```

Run `timing_sensitivity.py` with the same `--source`, `--mirror` and `--out`
arguments for the shift table, and `compiled_timing_check.py` for the compiled
clock check. The scripts need NumPy/SciPy and the existing
repository environment; no packages were installed.

`audit-summary.json`, `synthetic-summary.json` and
`timing-sensitivity-summary.json` are readable aggregates. Complete predictions,
per-case metrics, compatibility flags and payload hashes are in deterministic
gzip JSON files. `timings-and-repeats.json` preserves per-case CPU timings and
repeat checks. `provenance.json` includes Python/NumPy/SciPy versions, input
hashes, source SHA, script hash and exact invocation. No commits were made.
