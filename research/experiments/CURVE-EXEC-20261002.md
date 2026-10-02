# CURVE-EXEC-20261002 — Cubic in time and execution fidelity

Status: **inconclusive pilot; main/confirmation stopped**. Implementation and
offline verification complete. Bounded pilot authorized by “Execute the plan”.
Both XRs are in **Inbound**, observed by the operator. Eight XR1 diagnostics and
three controls ran; XR2, identical-command repeats and dense probes did not run.
The measurement floor therefore remains unquantified. No model training,
collection jobs, service restarts, overlap or spin are authorized by this run.

Base: `ac74a7f`, isolated `research/curved-gestures` worktree. Behavioural cloning
remains authoritative; linear/scaling Model 1 and all checkpoint schemas remain
unchanged. Curve recovery/generation belongs to the next tranche.

## Frozen protocol

The executed protocol, seeded commands and executable source hashes are in
[manifest.json](../evidence/CURVE-EXEC-20261002/manifest.json), frozen before the
pilot (SHA256 `f7ac2a4622c8d6a557c483b6bd0b34e699e2a0191b6493dfab5a5d73ebbbdc1b`).
The corrected final tools have a separate
[implementation manifest](../evidence/CURVE-EXEC-20261002/implementation-manifest.json);
this version was verified offline and was not replayed on devices. Original
execution and rejected admission records are preserved. Preserve all attempts; freeze the extractor, uncertainty, budget and
eligible spacing rules in a main-run addendum before looking at main outcomes.

Representation: `cubic_in_time_v1`, four 2D Bernstein coefficients plus duration
(**nine numbers**), evaluated directly at normalized time. No independent easing.
Endpoint-constrained least squares uses eight evenly timed positions. A single
cubic is not a universal trajectory description. Reject unsafe coefficient hulls
and quantized segments; never clip or move coefficients. Sample within
x=[0.27,0.92], y=[0.22,0.82]; expanded XR controls and global bounds still apply.

Round total duration once, then cumulative boundaries. Evaluate at their actual
normalized times, quantize logical device points, and pass positive integer-ms
segment durations to one down/multiple moves/one up W3C request. Separate curves
are separate requests. Labels share these exact boundaries and quantized points;
legacy labels/execution deliberately retain their old truncation behavior.

Spacing is the controlled variable: maximum segment durations **10, 20, 40 ms**;
minimum 1 ms. Alternatively accept explicit N and a minimum-duration floor;
reject incompatible constraints. Record N and every actual duration. For N=32,
nominal spacing is 9.4 ms at 0.30 s and 37.5 ms at 1.20 s. The user-supplied
[gptd-swift PR3](https://github.com/MobileBoostHQ/gptd-swift/pull/3) report about
XCTest interpolation and sub-16ms collapse is unverified on our rig; the PR could
not be independently retrieved during planning. It is a hypothesis to test.

Offline: twelve seeded curves, four families (variable-speed straight, arc,
S-curve, reversal) × 0.30/0.60/1.20 s, with rotations, both directions and varied
distances. Evaluate N=2/4/8/16/32 and spacing rules. Bound interpolation via C''
and add coordinate quantization; eligibility requires **≤0.005** for every curve.
N=2/4 remain offline only. Report synthetic circle/corner/multiple-bend fitting
residuals separately; they do not affect executor scores.

## Bounded stages

1. Pilot: **36** diagnostics. Each XR: eight N=16 commands (four families,
   short/long), the same eight repeated, and two identical short N=32 S-curves.
   Verify continuity, recording, calibration, visibility and dense-path collapse.
2. Main: **144** diagnostics. Same twelve curves × three spacing rules × two
   repeats × two XRs. Randomized within device, paired identities retained.
   Rules failing offline are diagnostics excluded from selection.
3. Confirmation: **24** diagnostics. Twelve fresh seeded curves per XR under
   the selected spacing rule; derive N by duration, not a fixed N.

Maximum **204** diagnostics, excluding controls. At most eight per one-minute
recording. Controls at 1/30/57 s, diagnostics at 6/12/18/24/36/42/48/54 s,
stop at 59 s. Reduced final batches still span the middle control. Reset/settle
between recordings only. Run on training-server with observed park provenance,
isolated tmp output, foreground/gameplay guards, running root RemoteXPC tunnel,
one recorder-start attempt, complete timing reports and decoded-frame checks.
Abort on contamination, recorder failure, incomplete timing or overrun. Retain
failed recordings and logs; never replace failures or restart healthy WDA.

## Measurement and preregistered floor

Use original pixels and source PTS at native ~30fps. Fit video timebase using
first/last controls; the middle control is held out. Require the existing
calibration checks and a held-out error no larger than two native frames. Prior
evidence: eight gestures within one frame, middle miss 38 ms; this does not
certify finer timing precision.

Primary spatial measurement is a command-blind orange-mask skeleton: no command
geometry, rule or N is an extractor input. Subtract pre-command orange background;
retain gaps, branches and competing components as indeterminate. Never snap or
choose branches by requested path. Leading endpoints are contact candidates only
when identifiable from new orange pixels. Fading never establishes liftoff.
Route indeterminate cases to blinded human review. Export original native frames
with source PTS; requested curves and counts are hidden.

Seeded random **15%** human audit, stratified by marginal device/family/duration/
rule, with blinded **10%** reannotation; reviews are additional. Pilot repeated
commands quantify repeatability separately from extractor/human disagreement.
Conservative floor = max(audited spatial uncertainty, half maximum identical-
command paired discrepancy). Audited uncertainty = maximum extractor-human
centreline discrepancy + maximum blinded within-human discrepancy.

- Floor ≤0.005: primary budget **0.01**.
- Floor >0.005: primary result **inconclusive**, revised budget **0.03**.
- Floor >0.025 or unbounded/insufficient visibility: stop before main,
  **inconclusive**.

Do not choose a new budget after seeing main. An audit exceeding frozen pilot
uncertainty invalidates conclusions. Both budgets remain reported. Normalized
Euclidean distance is anisotropic: 0.01 is 4.14 pt horizontally or 8.96 pt
vertically on 414×896. The remaining 0.005 after compilation is 2.07 horizontal
or 4.48 vertical points. This matches the Model 1 recovery metric.

## Scores and selection

Separate representation, compilation and observed execution. Report observed
shape versus quantized command and versus continuous cubic; use the latter for
the positional gate. Report maximum symmetric trail distance, endpoint error,
time-aligned contact errors with frame uncertainty, visible motion duration,
progress at u=0.25/0.50/0.75, visibility, indeterminate cases, interruptions,
failures, and WDA submission-to-completion/request overhead separately.

Duration error ≤0.10 s and collapse checks apply to every duration. Progress
error ≤two native frames applies **only to 0.60/1.20 s**. Two frames (~67ms)
poorly discriminate 0.30 s quarter intervals (75ms). Reversal crossings remain
indeterminate when temporal identity is ambiguous. WDA callbacks do not timestamp
individual touches. Report by device, family, total duration, N and actual spacing;
repeats are paired observations, not independent curve samples.

Choose the **coarsest eligible spacing rule** meeting every criterion: offline
bound ≤0.005; main evaluability ≥90% overall and ≥80% per device/family; every
evaluable shape/endpoint error plus uncertainty within chosen budget; duration
and applicable progress timing within limits; no collapse, touch interruption or
contamination; fresh confirmation passes the same rule. No eligible rule or an
inadequate measurement floor requires diagnosis, not new training.

## Command manifest and artifacts

Run Python from repository root using the existing `.venv`. Worktree tools use
`PYTHONPATH=src` because the shared environment's editable install points at the
primary checkout. Do not install another environment or upgrade rig dependencies.

```sh
PYTHONPATH=src /absolute/repo/.venv/bin/python scripts/inspect/prepare_curve_exec.py --out /absolute/new/freeze
PYTHONPATH=src /absolute/rig/.venv/bin/python scripts/collection/probe_cubic_curves.py --manifest /absolute/freeze/manifest.json --stage pilot --device iPhone_XR --segment-index 0 --park Inbound --wda-revision b5ace21788b5f5dc4cf0e0759f8bb8a79ab83ae6 --out /absolute/isolated/pilot/iPhone_XR/segment_00
PYTHONPATH=src /absolute/repo/.venv/bin/python scripts/inspect/measure_curve_exec.py --recording /absolute/isolated/pilot/iPhone_XR/segment_00
PYTHONPATH=src /absolute/repo/.venv/bin/python scripts/inspect/build_curve_audit.py --measurements /absolute/measurement.json --out /absolute/new/blind-audit
PYTHONPATH=src /absolute/repo/.venv/bin/python scripts/inspect/build_curve_audit.py --export /absolute/blind-curve-annotations.json --private-map /absolute/blind-audit-private.json --out /absolute/new/imported-annotations.json
PYTHONPATH=src /absolute/repo/.venv/bin/python scripts/inspect/report_curve_exec.py --manifest /absolute/freeze/manifest.json --stage pilot --measurements /absolute/measurement.json --annotations /absolute/imported-annotations.json --out /absolute/new/pilot-gate.json
```

Later recordings require preceding calibration admission; main requires a frozen
passing pilot gate, confirmation a passing main gate. JSON gates remain
inconclusive without complete cohorts and audits. Native videos and annotation
exports stay in isolated tmp storage, not Git or training loaders. Compact frozen
protocol and offline evidence are tracked. Results and verification follow below.

## Results — 2026-10-02

[Offline evidence](../evidence/CURVE-EXEC-20261002/offline.json):

| Comparison | Maximum normalized bound | Curves ≤0.005 |
|---|---:|---:|
| N=2 | 0.156969 | 0/12 |
| N=4 | 0.040078 | 0/12 |
| N=8 | 0.011005 | 6/12 |
| N=16 | 0.003765 | 12/12 |
| N=32 | 0.001890 | 12/12 |
| maximum 10 ms | 0.001707 | 12/12 |
| maximum 20 ms | 0.003267 | 12/12 |
| maximum 40 ms | 0.008522 | 10/12 |

Eligible offline rules: **10/20 ms**. The 40 ms rule remains a known-deviation
diagnostic excluded from selection. Maximum time-aligned representation
residuals: circle **0.0662**, corner **0.0562**, multiple bends **0.1225**.
These are synthetic representation limits, not model recovery scores.

[Retained pilot summary](../evidence/CURVE-EXEC-20261002/pilot-summary.json) and
[frozen stop decision](../evidence/CURVE-EXEC-20261002/pilot-gate.json):

- XR1 completed eight diagnostics (each family at 0.30/1.20 s, N=16) plus
  start/middle/end controls. Foreground/gameplay guards and WDA timing records
  passed; the recorder returned idle. No failed gesture was replaced.
- Original admission rejected the source-frame count: OpenCV decoded **1773**
  frames, while FFprobe and stream metadata both reported **1772**. The staged
  executable source and original failed admission remain preserved.
- Separate offline reanalysis uses FFmpeg `fps_mode=passthrough` with exactly
  1772 source frames and their original PTS. No truncation or resampling. A
  synthetic test proves selected native pixels and times match a full decode,
  and both surplus and missing frames reject.
- Reanalysis timing calibration passed: anchor span **56.131 s**, fitted rate
  **0.9996550**, middle residual **−5.823 ms**. Native median frame interval is
  **33.333 ms**; maximum observed gap is **50 ms**. These describe this recording,
  not a new cross-device timing certification.
- Primary orange extraction yielded **0/8 evaluable** gestures and **zero
  identifiable contact centres** in all eight. Moving orange/yellow scenery and
  board graphics survive static colour-background subtraction; gaps and branches
  remain ambiguous. Pixel inspection confirms these competing colour sources.
  No observations were snapped onto command geometry or promoted to passing labels.
- The measurement method is inadequate. Both 0.01 and 0.03 execution results are
  **inconclusive**; no spacing rule selected. The positional floor, repeated-command
  variability and blinded human uncertainty remain unquantified. Dense collapse
  probes, main and confirmation were withheld. WDA callback durations are retained
  as overhead evidence and are not interpreted as individual touch timestamps.
- A blinded native-frame review bundle covers the eight indeterminate cases,
  the seeded audit sample and repeated annotations. No human audit results have
  been fabricated. Review is for diagnosis; it does not retroactively complete
  the missing pilot cohort or authorize main.

Raw artifacts: rig
`/Users/training-server/trueskate-ai-runtime/tmp/CURVE-EXEC-20261002-f7ac2a46/`;
laptop worktree `tmp/curve-pilot-recordings/`. Source bundle/video SHA256 values
are in the summary. The original rig bundle and laptop copy preserve the executed
source separately from the corrected final tools. Original videos, detailed masks,
source PTS and annotation exports remain outside Git and training corpora.

Cleanup: both recorders idle. The recorded UUID
`297307E6-51D7-4E5F-9036-F65568B20127` was absent from the documented read-only
attachment listing. Seven other XR1 and three XR2 UUIDs were listed; their
provenance was not established and none was deleted. Active rig release and
healthy WDA services were unchanged; collector jobs remain off.

Verification: **466 passed, 2 skipped** in the repository suite using the existing
`.venv`; the localhost test required sandbox permission. Archive integrity passed
against `origin/main`. Tests cover fitting/degeneracies, unsafe hulls, deterministic
sampling, spacing floors, payload/label boundary agreement, native-source decode,
blinded extraction ambiguities, failure cleanup, audit resolution and floor gates.

Next prerequisite is a command-blind extractor that distinguishes the visible
finger trail from moving orange scenery and supports audited spatial uncertainty.
Diagnose it on retained pixels before proposing another bounded pilot; do not
change representation or add training on this evidence.
