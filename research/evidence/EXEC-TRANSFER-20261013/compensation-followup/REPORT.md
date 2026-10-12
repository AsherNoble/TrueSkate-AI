# Executor compensation feasibility — offline follow-up

The coordinating agent completed this analysis from preserved measurements
after the delegated follow-up stalled. No new simulator or physical requests
were issued. The retrospective split and five duration candidates were fixed
in [PLAN-FROZEN.md](PLAN-FROZEN.md) before these fits. Broad reported effects
and some examples were already known; these are command holdouts, not a new
blind experiment or untouched research cohort.

## What can be inverted

The local Recap cubic is monotonic in both coordinates, so its single-leg
progress curve has a unique mathematical inverse. Let `x=B_x(u)`,
`q=B_y(u)`. The command phase corresponding to a desired progress is
`x=B_x(B_y^-1(q))`. [reference.py](reference.py) implements both directions;
10,001 grid checks have maximum forward/inverse discrepancy `5.56e-16`.
Strict monotonicity also follows from the positive cubic derivative terms,
since each axis has `0 < control1 < control2 < 1`.

This API is a reference clock transform, **not an executable compensation
payload**. For one fixed-duration W3C move, delivered progress is approximately
`f(t/D)` under the continuous single-leg abstraction. Requiring its nominal
endpoint at target time `T` fixes `D=T`; a different scalar duration cannot make nonlinear `f(t/T)`
equal `t/T` at every intermediate time. Prewarping progress requires additional
control freedom, such as more waypoints. Each added segment can restart easing
and change scheduling, so simply sampling the inverse does not invert the full
executor. Bounds, corners, zero-distance holds and pointer-up must be evaluated
together.

## Duration candidates on command holdouts

Seven development commands are fixed in the plan: four long single moves,
two subdivisions and one two-segment 17 ms boundary control. The linear candidate
fits duration error against segment count only on those seven commands.
Other candidates have no fitted parameters. Scores weight commands equally
using five-repeat median contact duration, not individual callbacks as replicates.

| Candidate | Round-6 MAE, 100 commands | Round-6 max | Other main probes MAE, 43 commands | Boundary MAE, 50 commands |
|---|---:|---:|---:|---:|
| Requested total |25.91 ms|101.33 ms|44.98 ms|81.86 ms|
| Nearest total tick |25.00 ms|100.00 ms|44.57 ms|81.99 ms|
| Sum rounded segment ticks |25.50 ms|116.67 ms|48.45 ms|80.34 ms|
| Sum segment ticks rounded up |**2.00 ms**|33.33 ms|42.25 ms|52.67 ms|
| Linear segment-count correction |9.12 ms|39.03 ms|59.87 ms|96.14 ms|

The ceiling candidate is informative on the current round-6 domain, whose
intervals are at least 100 ms. Its 95th-percentile absolute error is 16.667 ms.
It does not establish exact scheduling even there: the worst error is two
frames. The boundary maximum is 333.33 ms and the other-probe maximum 666.67 ms.
These deliberately challenging sets include short segments, pauses and
different contact durations, so their MAEs are not prevalence estimates.

This predicts received **contact duration only**. It neither predicts every
waypoint arrival nor evaluates an adjusted replay. The candidate receives
time/point sequences; it does not model pause-vs-move event semantics or all
backend state. A low duration residual cannot authorize shortening commands,
altering holds or changing production labels.

## Reachability and ambiguity

The saved controls show that ten 15 ms and ten 16 ms schedules can yield identical
received sequences at one timestamp, whereas ten 17/20/25/33 ms schedules each
last 333.333 ms. Single short moves and mixed long/short sequences behave
differently. This rules out using one universal per-segment floor or a simple
total-duration inverse across these cases. The finite tested grid cannot prove
that every untested intermediate effect is unreachable by some other command.

The observed collision prevents unique recovery of the original schedule from
that received stream. Effect imitation could still use a right inverse on a
reachable effect set. A coordinate/timing average of two commands is not
automatically a right inverse. Holds and release are separate requirements:
zero-distance moves preserve stationary phases with quantization; replacing
them by W3C pauses changes motion and is unsupported.

The expert's 57–115 ms figures are visible moving spans, not independently
measured touch contacts. Unknown endpoint holds prevent a total-contact
exclusion claim. They nevertheless motivate checking whether rapid movement,
turns and neighboring actions can be reproduced, rather than comparing only
total contact duration. The existing 100 ms segment constraint and the measured
backend behavior both matter. No physical fast-stroke reachability claim is
established by this analysis.

## Smallest useful next checks

1. Freeze a small simulator test set with unseen interval phases, mixed segment
   lengths, sharp corners and initial/middle/final zero-distance holds. Compare
   any proposed adjusted payload with exact desired per-frame XY, waypoint
   arrival and release time. Record raw timestamps and callback arrival
   separately; vary logger buffering. Include safety and integer compilation.
   Do not accept total duration alone or retune on the scored cases.
2. If that succeeds, propose a separately authorized physical-helper test using
   repeated safe payloads and the existing healthy WDA. Verify OS/runtime and
   delivered events without assuming the simulator law. True Skate transfer
   additionally needs rendering-phase evidence; one clock-alignment fit to a
   few glow edges is insufficient. Keep recordings at one minute and smoke
   output isolated; no collector/watchdog or open-ended collection.
3. For the decoder, first test the cheap fixed-time coordinate refit under
   synthetic independent/correlated noise, endpoint bias and shifted boundary
   classes. Its perfect-target gain does not justify deployment by itself.

These are proposals, not performed or approved physical tests. No production
compensation, metric change, new corpus tranche or paid compute was used.

## Reproduction

Run the existing repository `.venv/bin/python` with the absolute path to
`reference.py`. It reads only the preserved simulator evidence and generates
`results.json`, including all 200 command predictions, seven development IDs,
input hashes, candidate errors and inverse checks. No external scratch corpus
is required. The original frozen plan's human-duration wording is interpreted
with the movement-span caveat above; the plan itself is preserved unchanged.
