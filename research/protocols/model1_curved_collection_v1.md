# Curved Model 1 — bounded collection, audit and training

Agreed 2026-10-11. Implementation is additive and awaiting live qualification.
This supersedes the provisional sampling, duration and 15-waypoint choices in
[the October 3 design record](model1_curved_mvp2.md). Existing linear results
remain the supported baseline. No curved recovery result is claimed yet.

## Frozen scope

One continuous contact, represented by 2–11 normalized timed waypoints:
1–10 straight movements or pauses, each at least 100 ms, with total duration
sampled uniformly from integer 300–1200 ms. Duration limits the feasible segment
count. One W3C request contains one down/up pair. There is no extra subdivision.
Repeated adjacent positions are pauses. Segment lengths and durations vary.

Half the samples are random chains. Half come from a straight-to-single-arc
family, up to a half-circle, using the fewest segments whose geometric sagitta
is at most two logical points. Feasible geometry is constructed inside convex
control-safe regions rather than repeatedly rejecting candidates. Quantization
adds a separate error: labels use the exact integer-coordinate payload; ideal
arc parameters remain diagnostic metadata. Random chains are not uniform over
the mathematical space of all possible curves.

All generated paths stay inside the existing gesture bounds and avoid every
expanded control hitbox. Normalized Euclidean position tolerance is 0.03;
it is anisotropic on a 414×896 display (12.42 horizontal or 26.88 vertical
logical points). Report logical-point error too.

## Parks, sessions and fixed limits

[config/parks.json](../../config/parks.json) is the operator's complete installed
library, recorded verbatim. Never correct its spelling, punctuation or spaces.
Exclude SLS 2016 Munich as the reserved challenge. Sample five parks from the
sorted remaining names with seed 20261011. The draw is:

1. Skatepark: Inbound
2. SLS 2016 Newark
3. SLS 2015: Super Crown
4. SLS 2015: LosAngeles
5. Macba

At most five waypoint identities per park are available. The collector requires
the operator's current park, waypoint identity (1–5), device and settings note;
these are fixed within a session. Reset uses the game's saved waypoint. Motion,
airborne state and a moving camera do not disqualify a sample.

Each qualification round contains 100 fresh automatically admitted clips:
20 per park, ten from each sampler in each park. Human review passes a round at
**at least 95/100 meet-spec**, with no gross or systematic defect. Failures and
unclear judgements do not pass. A failed round requires a diagnosed fix and a
new pipeline fingerprint before another fresh 100. There are **five rounds
total**, initial included. A technical abort explicitly declared a failed round
also consumes one round. Five failures stop the program for redesign.

Once qualified, freeze that pipeline and collect a separate **14,000 admitted
production clips**, 2,800 per park. Pilot clips are excluded. Select a seeded
random **12,500**, 2,500 per park. The pool has ten sessions per park, each with
280 admitted clips and an equal sampler mix. Assign eight entire sessions to
training, one to validation and one to test before collection, using a seeded
shuffle. Random selection within park/partition produces 2,000/250/250 per park
(10,000/1,250/1,250 total). Keep surplus clips and original recordings.

The persistent SQLite ledger counts **20,000 candidate gesture dispatches
maximum**, including qualification, rejected clips and unknown dispatch
outcomes. It commits spending before the RPC. Planned but never dispatched
commands do not spend the budget; controls and resets are separately identified
in the execution records. There is no automatic RPC replay. A shortfall is
reported rather than silently reducing quotas or starting a replacement run.
Use the same run directory across all rounds and production sessions.

No extra mandatory final audit is introduced. A model failure does not launch
another collection loop or impose a fixed two-training-run limit.

## Native video and admission

Request 60 fps XCTest recordings of at most one minute. Verify actual source
PTS: effective 58–62 fps, median frame interval 15–18 ms, strictly increasing.
Decoded frame count must equal the probed native frame count. A clip is exactly
120 consecutive native frames, with no padding, temporal sampling or fabricated
frames; its largest gap may not exceed 50 ms. Encoded clip PTS must agree with
the source PTS to within 1 ms.

Choose uniformly among valid source-frame crop starts, preserving at least five
real frames before contact and five after release. Exclude every other sample,
reset and calibration contact. This spreads onset across the feasible clip
timeline. Use all 120 frames at 64×144 initially. Variable clip lengths and faster
expert gestures remain later extensions.

Each recording uses the established wide die-five start/end markers (71 logical
point offsets), a held-out middle marker and per-recording affine clock fit.
Require at least 55 s anchor separation, accepted marker consensus and middle
error within two native frames. Store commands, actual WDA submission times,
source timestamps, calibration, decoded counts, code fingerprint and park
provenance. Never shift an individual gesture until its overlay looks right.

Keep foreground, menu/editor, root and per-device RemoteXPC tunnel, empty
attachment and disk-space guards. Check every candidate clip frame for menu
contamination. Stop the invocation on its first recorder/calibration failure.
Never restart healthy WDA, weaken admission or silently clean preserved evidence.
Every live invocation is one explicit park/session and 1–28 recorded minutes;
it stops earlier when its quota or global budget is reached. There are no
background collectors, watchdogs or notifications.

## Human qualification

The temporary viewer decodes every native frame, displays actual PTS, and shows
one toggleable hollow target ring of radius ten logical points. A translucent
logical grid is optional. There are no added zoom controls and no redundant
orange-contact confirmation. Review the active accessibility cursor when it
represents contact. Mark first active-contact and first clear-after-release
frames. Boundaries must agree with the nearest target frame within one frame.

“Meet” also confirms the entire active path follows within 0.03 normalized
distance, with full intended contact and no extra contact, menu or truncation.
If visibility or timing prevents that judgement, choose unclear. Reasons are
required for fail/unclear and systemic flags. Exports bind all 100 unique samples
to the frozen manifest hash; repeat imports must be identical. Importing the
review ends the viewer server. A rejected export still requires the operator's
viewer to be stopped once its path has been returned.

## Training and evaluation

Use the separate `model1.curved` model and checkpoint schema. The network takes
RGB plus consecutive-frame difference, a c16 CNN and temporal mixer, then
predicts framewise spatial heatmaps and learned contact boundaries. It receives
actual frame times and frame intervals. There is no privileged early onset.

Dense position targets interpolate the executed quantized command on the native
timeline, including nearest-frame endpoint supervision. Loss is onset and
release cross entropy, masked Smooth-L1 position error scaled by 0.03, plus
0.1 times spatial heatmap cross entropy (Gaussian sigma 0.035 normalized).

The decoder jointly chooses a 300–1200 ms contact interval. A deterministic
minimax dynamic program selects freely placed timed breakpoints, 1–10 segments
of at least 100 ms. Choose the smallest segment count within 0.01 of the dense
prediction; otherwise report the best feasible fit and its residual. This is
representation error, not the accuracy threshold against ground truth. Predicted
paths are never automatically executed; control safety is checked separately.

Initial recipe: 40 epochs, batch 8, c16, AdamW, learning rate 0.001, weight decay
0.01, constant schedule, seed 0. Immutable completed-epoch checkpoints contain
optimizer, RNG, loader state, recipe, implementation and dataset fingerprints.
Resume rejects changed inputs or recipes. Validation selects the checkpoint by
joint pass rate, breaking ties with mean path error. Evaluate the selected
checkpoint on test once; an interrupted exposure is recorded rather than hidden.

A clip passes when both contact boundaries are within two native frames and
at least 95% of target contact frames have normalized error ≤0.03 on the original
timeline. Missing predicted contact is an error. No alignment or time stretching
is allowed. Report per-clip errors, timing errors, missing frames, fit residuals,
logical-point errors, and strata by park, sampler, duration, segment count and
onset quarter. Targets are ≥90% overall and ≥80% in each park. A miss is a baseline
result to diagnose, not permission to recollect indefinitely.

Before training, freeze ten human clips, two per selected park, each one contact
inside this MVP's scope and 120 native frames. Their file hashes are bound to
training. They are qualitative evidence: the viewer puts the actual video beside
the same synchronized frame with the inferred contact. No human gesture labels
are invented. These five-park session splits are not an unseen-park benchmark.

## Modal cost and launch boundary

Use the new `trueskate-curved-model1-v1` corpus volume and `curved-v1/<run>` under
the existing model volume. Sequential tar bundles preserve clip bytes and
metadata; safe staging verifies hashes and only permits expected regular files.
Train/validation uint8 cache is approximately 34.8 GiB at 120×144×64×3.

Pin L4, four CPU cores and 64 GiB memory. A separately approved benchmark is
capped at 900 execution seconds and measures loading, training and evaluation
throughput before quoting the full run. At the [2026-10-11 published rates](https://modal.com/pricing),
requested compute is about $1.4993/hour, or $0.3748 for 15 execution minutes.
Build/startup, transfer, storage and teardown are additional; the wrapper reserves
one dollar of estimate headroom. This is a runtime-bound estimate, not a provider
billing cap. Storage is quoted conservatively at $0.09/GiB-month before account
allowances. Reverify rates and obtain explicit spend approval before upload or
compute. Merely importing or requesting `quote` starts no remote work.

The implementation and offline evidence are recorded in
[M1-CURVED-20261011](../experiments/M1-CURVED-20261011.md).
