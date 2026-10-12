# DECODER-NOISE-20261013

The minimal fixed-knot coordinate refit remains useful under independent XY
noise, but is not a broadly robust replacement decoder. On 43,200 paired
synthetic stress cases, guarded passes rise 15,992→17,127: 1,467 gains against
332 regressions. The paired difference is +2.63 percentage points, with a
fixture-cluster bootstrap 95% interval of +2.10 to +3.22 points. The composition
is deliberately challenging and arbitrary; this aggregate is not a real-model
accuracy or prevalence estimate.

The candidate also creates 440 protected-control failures in cases where the
baseline is safe, while rescuing 393 unsafe baseline paths. Total compiler
rejections increase 10,884→10,931. Every rejected command remains a failure;
there is no baseline fallback or repair. These results support a guarded
sequential research pilot, and do not support production promotion.

## Frozen comparison

The [plan](PLAN-FROZEN.md) was recorded before fixture generation and scoring.
Seeds 2026101301/02/03 govern the cohort, corruption and bootstrap. The fixture
content hash is
`ca4f641697c22c8087c909c2f45f343fa5e7187970d6d45b4a472f9699b9eb8f`.

There are 24 generated commands per family: line, kink, alternating turns,
movement–hold–movement, sampled arch, near the expanded left spin control,
near the expanded bottom controls, and near the expanded top controls.
All 192 original commands pass the frozen timing, normalized-bound and
quantized protected-control contract. No fixture-generation rejection was
needed; no fixture was selected using decoder performance. These are new
generated inputs, independent of the prior 864-case factorial and 100-clip
qualification diagnostic. No corpus or trained checkpoint was read.

Each command receives a random onset phase and calibration rate in
[.999,1.001] on a 120-frame 60 Hz timeline. Existing `targets()` creates the
float32 observations, including its nearest-frame endpoint overwrites.
The baseline is the unchanged production `decode()`. The candidate calls the
unchanged prior research `refit_grid_coordinates()`: endpoints, selected knot
times, segment count and boundary classes stay fixed; only internal coordinates
are fit by least squares. Neither decoder receives true commands or event
times; these are used for fixture target generation and scoring.

Both predictions use integer 414×896 viewport truncation. All outputs are scored
on their intended quantized paths, including paths rejected by the compiler;
those metric-only scores do not imply executability. The primary **admissible
pass** requires the unchanged joint `.03`/`95%`/`±2-frame` score and successful
original timing, bounds, per-segment control-map and compiler checks. Positive
results never bypass a safety check. Every candidate failure is retained.

## Corruptions and boundary errors

The coordinate grid is .005,.01,.02,.03,.04,.06 normalized units plus clean.
Independent Gaussian and stationary AR(1), rho=.9, levels specify pre-clipping
radial RMS (per-axis SD=level/√2). Linear drift runs from −v to +v over contact,
with endpoint vector magnitude equal to the level. Endpoint-local bias uses
independent start/end directions and exponential two-frame decay. Inputs are
clipped to [0,1] for both variants to preserve a bounded coordinate head.
One fixed random field per fixture/type is scaled across levels.

These are hypothetical stress levels, not estimated prediction errors. A
Gaussian RMS level of .03 is not a guarantee that every point lies inside the
.03 tolerance. Coherent drift and endpoint magnitudes have different error
distributions from framewise Gaussian noise.

Every coordinate condition is crossed with onset and release class shifts
independently in {−1,0,+1}. The same Gaussian-shaped logits are used for both
variants. A shifted class changes the selected contact interval, while native
times and XY stay fixed; this is not a uniform temporal shift of the whole
trajectory. The decoder's original duration mask chooses the nearest legal
pair where the intended peaks conflict with 300–1200 ms restrictions. This
changes 400 onset selections and 75 release selections out of 43,200 cases;
all intended and actual classes are retained in the ledger.

## Exact boundary classes

Clean XY and exact classes score 184→189/192: five gains, no regressions and
no safety failures. Remaining failures are `alternating-10`, `alternating-18`
and `alternating-23`; their candidate good-contact fractions are .9434,.9111
and .9474. This independent challenge therefore does not reproduce the prior
audit's 100/100 result. Keeping the baseline's knot times is still a real
restriction on large, rapid alternating turns.

The table below uses **admissible passes out of 192** with exact classes. Gains
and regressions are paired; all compiler rejections count as failures.

| Corruption | Level | Baseline | Refit | Gains | Regressions |
|---|---:|---:|---:|---:|---:|
| Clean |0|184|189|5|0|
| Independent XY |.01|135|147|13|1|
| Independent XY |.03|36|60|25|1|
| Independent XY |.06|1|4|4|1|
| AR(1) |.01|129|137|9|1|
| AR(1) |.03|17|19|5|3|
| AR(1) |.06|0|0|0|0|
| Linear drift |.01|139|142|4|1|
| Linear drift |.03|61|63|5|3|
| Linear drift |.06|0|0|0|0|
| Endpoint bias |.01|146|149|4|1|
| Endpoint bias |.03|107|110|5|2|
| Endpoint bias |.06|0|0|0|0|

The fixed-coordinate least-squares refit can average independent errors at
internal knots. It cannot average away persistent drift, and fixes the same
possibly biased endpoints as the baseline. Its limited correlated-noise
benefit and observed regressions are therefore consequential limitations,
not evidence for a universal noise tolerance.

## Boundary errors with clean coordinates

These are admissible passes out of 192; no control-map failure occurs in these
clean-coordinate conditions.

| Onset shift | Release shift | Baseline | Refit |
|---:|---:|---:|---:|
|−1|−1|154|158|
|−1|0|171|175|
|−1|+1|157|162|
|0|−1|174|179|
|0|0|184|189|
|0|+1|173|179|
|+1|−1|90|93|
|+1|0|175|182|
|+1|+1|155|160|

Being within the ±2-frame boundary allowance is insufficient by itself.
Missing contact frames remain penalized by the unchanged 95% joint score;
shortening the interval at both ends is particularly damaging. Better knot
coordinates do not repair missing contact or establish temporal alignment.

## Safety and pilot decision

All observed contract failures are protected-control rejections from the
original compiler. There are no relaxed timing, interval, segment-count or
normalized-bound exceptions. The three near-control families deliberately
place true paths only a few logical points outside expanded hitboxes; their
rejection rates cannot estimate deployment incident rates.

The 440 newly unsafe candidate paths show that a coordinate refit does not
preserve the baseline's protected-control safety geometrically. Clipping to
normalized bounds cannot substitute for checking every quantized segment.
The unchanged compiler successfully rejects these commands. No command was
executed.

The refit is a reasonable **research pilot candidate with mandatory original
guards** because clean and independently noisy inputs improve at low cost.
It is not a universal replacement: 332 paired admissibility regressions,
new unsafe paths, coherent-noise weakness and boundary sensitivity preclude
promotion on this evidence. A sequential pilot must preserve those failures
in its denominator and measure actual model errors and end-to-end usefulness.
It must not treat these synthetic levels as empirically calibrated noise.

## Cost, verification and reproduction

CPU latency results and final verification are recorded in `summary.json`
and `verification.json`. Timings exclude evaluation; the candidate pipeline
includes the common frozen decode, coordinate refit, quantization and guard/
compiler work. The laptop was used for parallel research, so tail measurements
are observations under shared load, not a deployment SLA.

| Operation | Median ms |p95 ms |p99 ms |Maximum ms |
|---|---:|---:|---:|---:|
| Frozen decode |3.470|9.725|12.566|126.487|
| Additional coordinate refit |.0497|.0856|.1333|3.752|
| Baseline including quantization/guards |3.567|9.842|12.724|126.955|
| Candidate including quantization/guards |3.619|9.898|12.804|127.133|

The full run took 202.03 seconds. The original decoder unit test
passes, and archive integrity passes. The verifier checks all 43,200 unique
paired records, regenerates the fixture content byte-for-byte, and replays
120 prespecified rows with identical scores and payload/guard outcomes.

Run from any directory:

```sh
'/Users/ashernoble/Projects/Robotics & hardware/TrueSkate-AI/.venv/bin/python' \
  '/Users/ashernoble/Projects/Robotics & hardware/TrueSkate-AI/tmp/worktrees/decoder-noise-20261013/research/evidence/DECODER-NOISE-20261013/study.py' \
  --source '/Users/ashernoble/Projects/Robotics & hardware/TrueSkate-AI/tmp/worktrees/decoder-noise-20261013' \
  --out '/Users/ashernoble/Projects/Robotics & hardware/TrueSkate-AI/tmp/worktrees/decoder-noise-20261013/research/evidence/DECODER-NOISE-20261013'
```

Run `verify.py` with the same arguments. It regenerates the entire fixture
cohort, verifies frozen plan/source hashes and every row's acceptance logic,
and exactly replays 120 prespecified paired cases spanning all families,
corruption types and representative boundary patterns. `fixtures.json.gz`
and `paired-results.json.gz` preserve all generated inputs and per-case scores,
guards, errors, selected classes, segment counts, payload hashes and timings.
`summary.json` includes every level/pattern/family and fixture-cluster
bootstrap intervals. `provenance.json` records source hashes, invocation,
platform and versions. No existing evidence files or shared STATUS/JOURNAL
were edited.
