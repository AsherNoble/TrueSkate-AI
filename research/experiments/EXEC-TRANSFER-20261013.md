# EXEC-TRANSFER-20261013 — executor timing, observability and curved labels

Status: laptop-only research continuation. The physical XR transfer remains
unqualified. No collection, rig operation, paid compute, production training,
or change to the frozen qualification gates was performed.

## Recovered task and evidence

Claude session `e0f1c48e-973a-4ff3-a26a-f5f2350d9a43`, subsequently resumed as
`380dc89c-3d2b-43ff-bcc5-4794e1404d2e`, stopped during the analysis of the
100 round-6 clips. Three research agents were unfinished: XCTest transfer,
trail measurement, and label tolerance. Human coverage and executor alternatives
had returned reports. All three unfinished assignments were resumed; focused
follow-ups investigate decoder representation, simulator-to-video transfer and
the feasibility of executor compensation.

The operator's question was whether a predictable difference between commanded
and visible motion could be learned by Model 1, rather than requiring the raw
trail to match a linearly timed command. The operator also requested signed lag
measurement after clarifying that mismatches went in both directions.

Implementation under investigation:
`59f1782d7a5d43228d3232cb3bf9cce16bc52aa0`; WDA
`ae50404aac12d9f8c41f6c3fa8776e97975eaef5`. Development is isolated on
`codex/executor-research-20261013`. The original checkouts are preserved. The
original temporary scratch directory disappeared during final verification;
compact evidence was already copied, but movies and large caches are now
unavailable locally. [Evidence index](../evidence/EXEC-TRANSFER-20261013/README.md)
lists reports, scripts, hashes, reproduction commands and external artifacts.

Round 6 remains **83 Meet, 16 Fail, 1 Unclear**, below the 95-Meet gate.
The review was imported by the prior session: one of five reviewed rounds and
144 of 20,000 candidate dispatches used. Its exact export hash is
`4085a0096e2d1d594ce6c60f45fbba73ebe6ba565b03bc8bb08caac95c0fb5b9`.
This continuation reproduced the score and verified every local clip and
metadata hash against the frozen manifest (100 samples, 178,022,307 bytes).
That media check passed before the scratch directory disappeared. A subsequent
preserved-only verification rechecks all 100 commands, recording separation and
forward-comparison arithmetic; it explicitly reports missing original profiles
and movies rather than treating those source checks as passed again.

## What the trail can establish

The original tracker reported 59/100 clips with at least three accepted frames:
20 apparent leads, 31 within ±8 ms, eight apparent lags, and 41 unmeasurable.
The median was −2.53 ms. These are **exploratory tracker outputs**, not validated
direction labels or an executor-noise estimate. The accepted sample is selective,
the ±8 ms threshold is arbitrary, and the nominal clock is calibrated command
time rather than an independent touch timestamp.

The resumed [trail study](../evidence/EXEC-TRANSFER-20261013/trail-model/REPORT.md)
sampled all 100 clips and measured 728 clear fixed spots in 23 clips. The
clip-balanced trail width at half intensity was about 45 native pixels at peak,
37 after 50 ms, 29 after 100 ms, and 23 after 150 ms. Intensity halved after
92 ms and crossed the old detector's threshold about 194 ms after the local
peak. Visibility depends on background and trail age.

The assumed 24-pixel cap is smaller than the 27–31-pixel threshold edge measured
in two clean stationary holds. An illustrative four-pixel correction changes
the old accepted-frame median to +1.19 ms and its counts to 15 apparent leads,
31 within ±8 ms, 13 apparent lags. This local speed approximation is a
sensitivity check, not a corrected measurement. It demonstrates that small
global lead/lag claims are not robust to cap geometry.

Clip 67's supposed 170 ms lead is invalid evidence of hold collapse. Its final
moving segment nearly exactly retraces its predecessor (direction cosine
−0.999968, endpoint cross-line distance 2.56 native pixels). Recent old glow
at the endpoint is misclassified as a future head. The overlap filter uses that
future candidate time, incorrectly ages the old glow, and the finite-path edge
score invents visible darkness beyond the endpoint. Deck interference compounds
the ambiguity. Increasing the fade constant alone does not fix this circular
test.

Stationary holds are not timing-observable from position alone. The old
500 ms persistence rule also erases genuine long-held glow. Deck intersections
and generic texture-mask flags are different quantities; a glow can remain
visible over deck texture. Touch-to-display latency and subtle corner cutting
cannot be separated from executor and renderer effects using these videos alone.

## Label semantics and tolerance

Write delivered input as `a = E(c)` and video as `v = R(a, game state)`.
An inverse model trained with command labels can learn a command reproducing
an observed effect through the same executor. On human video that command is
generally `E`-specific; it need not equal the human's original touch path.

The resumed [label study](../evidence/EXEC-TRANSFER-20261013/label-tolerance/REPORT.md)
ran 15 CPU fits (three independent seeds, five conditions, 600 held-out toy
videos per seed). A deterministic invertible warp gave mean normalized command
error 0.00452 and fresh replay error 0.00438, despite a raw command/trace gap
near 0.055. Independent ±25 ms onset noise raised fresh replay error to 0.03044.
These deliberately favorable timing-only toys demonstrate mechanisms, not XR
performance. Fresh stochastic replay uses a new noise draw rather than replaying
through a noise-free nominal executor.

Determinism is insufficient by itself: the target must be observable, the model
must have sufficient information/capacity, and the executor and renderer must
remain stable across devices, builds, parks and state. Unique original-command
recovery requires injectivity; effect imitation can instead use a validated
right inverse on the desired effect set. Averaging ambiguous command labels
does not generally provide such a right inverse. The synthetic collapse example
is hypothetical, separate from the invalidated clip-67 claim.

Primary literature supports noncausal context and effect-based latent actions,
but supplies no universal ±25 ms tolerance for this task. VPT's 20 Hz control
rate is not a 50 ms label-noise allowance, and compensated deployment latency
is different from timestamp corruption. See the study's primary-source notes.
At normalized local speed `v`, timing error `δ` gives roughly `v|δ|` positional
error. A 0.03 bound therefore yields a speed-dependent timing budget, not a
universal one-frame rule.

## Simulator delivery

The resumed [XCTest study](../evidence/EXEC-TRANSFER-20261013/xctest-transfer/REPORT.md)
issued 745 randomized requests (49 probes and every
round-6 command, five repeats each), 255 boundary controls, and 90 callback
logging controls. These are simulator requests, not independent physical-device
trials. The local iPhone XR simulator runs iOS 18.6; matching the physical XRs'
OS, framework and rendering behavior has not been established.

The local Recap framework selects a cubic timing curve with control points
`(0.2620539963, 0.1794590056)` and `(0.7582859993, 0.8902440071)`.
Solving its x coordinate for time and evaluating y predicts the measured
100/400/800/1200 ms single-segment paths to within 0.247 logical points after
nearest-frame alignment. This is source-backed local-runtime evidence for
nonlinear interpolation, not a universal physical-XR transfer function.

Repeated commands have identical common-frame coordinates; the largest
round-6 duration standard deviation across five repeats is 0.3882 ms. Samples
follow approximately 60 Hz cadence; exported schedules are frame-quantized.
Coordinates fall on a half-point grid. Round-6 spatial polyline deviation is
at most 0.344 points, while end-to-end duration error for those commands
ranges from about −8.5 to +101.3 ms. Sixty-two of 100 commands end over one
frame late. Segment count explains some duration error, but a fixed offset or
single time scale does not describe the measurements.

The boundary controls distinguish command compilation from delivery: all 255
native point/type/timestamp records exactly match their W3C inputs. Nevertheless,
ten 15 or 16 ms moves collapse into one received timestamp, with identical
received sequences for the two schedules. Ten 17/20/25/33 ms moves each take
333.3 ms. A ≥17 ms guard prevents the observed collapse but does not preserve
timing; these cases do not establish a universal 33.3 ms segment law.

Zero-distance moves preserve stationary intervals with additional frame
quantization. W3C pauses do not emit stationary events and can stretch the next
movement instead, so substituting pauses is unsafe. The exact clip-67 payload
does not show a 170 ms lead in the simulator: movement finishes at 966.7 ms
versus a 934 ms command, with an approximately 183 ms final hold.

The logging control compares synchronous-before, buffered-memory and
synchronous-after runs. All six tested paths have identical frame-indexed
coordinates, with optional stationary/coalesced-sample differences. An earlier
1.376 s callback stall did not recur; neither its cause nor a callback-delivery
jitter law is identified. UITouch timestamps are distinct from callback arrival
times and from the game's display clock.

## Physical-video transfer

The [frozen forward comparison](../evidence/EXEC-TRANSFER-20261013/sim-video-followup/REPORT.md)
is **inconclusive**. It screened all 100 clips without using either model's
predicted position to select the visible front. Only 27 of 4,872 screened active
frames qualify under the primary conservative visibility screen. The held-out
set contributes 19 frames from three recordings; frames are not independent
replicates. Clip 67 contributes no measurements.

With one global alignment fitted on development recordings only, interpolated
simulator schedules reduce recording-balanced held-out residual from 18.79 to
13.00 native pixels. But one held-out recording worsens, the descriptive
three-recording bootstrap includes no improvement, and holding samples until
the next delivered touch reverses the result (34.19 versus 18.79 pixels).
One-frame phase changes exceed the apparent advantage. The physical videos
therefore neither certify the simulator law nor reject its easing mechanism.
Further fitting of these sparse edges cannot recover the missing touch/render
clock.

## Research decoder feasibility

The [decoder follow-up](../evidence/EXEC-TRANSFER-20261013/decoder-followup/REPORT.md)
reproduces 95/100 recovery from perfect targets with the frozen decoder. Refitting
internal coordinates at its existing knot times raises this to 100/100 after
integer-coordinate compilation, without changing the 100 ms segment contract
or scoring metric. The restriction is that the baseline anchors knots to
observed frame coordinates; this is not a hard representability ceiling of the
gesture contract.

On 864 independent synthetic cases, recovery is 810 for the baseline, 838 for
fixed-time coordinate refitting, 851 for legal free knots, and 858 for a joint
boundary-interval fit. All three research variants compile under the frozen
gesture and safety checks. Compilation and perfect-target reconstruction do
not establish trained-model robustness or executed-path accuracy. The minimal
refit's additional median CPU cost is 0.056 ms in this audit; model-noise behavior
remains untested.

Timing alignment remains consequential: an imposed +8.33 ms shift reduces
baseline/refit recovery to 82/81, and +16.67 ms to 68/67. These are hypothetical
sensitivity probes, not estimates of real timing noise. The study also finds
that overwriting nearest boundary-frame positions with exact endpoints can
make an exact original command fail the frozen score on fast synthetic cases
(8/864). This is a label/evaluation convention artifact, not execution loss.
No metric change was made or used for acceptance.

## Compensation feasibility

The [offline compensation check](../evidence/EXEC-TRANSFER-20261013/compensation-followup/REPORT.md)
implements the mathematical inverse of the local single-leg easing curve, but
does not claim an executable inverse for an entire W3C stream. A scalar duration
change cannot linearize a fixed-duration nonlinear leg; added waypoints also
change easing and scheduling.

Five duration candidates were evaluated under a retrospective, declared
seven-command development split. On the other 100 round-6 commands, summing
each segment rounded up to a 60 Hz interval predicts contact duration with
2.00 ms mean absolute error, versus 25.91 ms for the requested total. Its worst
round-6 error remains 33.33 ms, and its boundary/probe errors reach hundreds of
milliseconds. These are command-weighted five-repeat medians, not independent
device trials. This supports a restricted scheduling hypothesis for current
long segments, not a universal law or a compensation fix. Intermediate XY,
corners, holds and release must be validated together before using adjusted
commands. No adjusted replay was performed.

## Human motion support

The preserved eight-stroke expert extraction contains seven visible moving
spans of 57–115 ms and a 550 ms catch trajectory. These are trail-derived
estimates, not independently logged touch-down/up times. Stationary endpoint
holds could make full contacts longer. The current curved-v1 sampler, contract
and decoder admit only 300–1200 ms contacts with segments at least 100 ms. The raw
extracted fast trajectories fall outside that timing representation, exposing
a motion-support concern; the extraction alone cannot prove that seven complete
contacts violate the duration bound. One player/trick/park cannot estimate
population frequencies.

The [independent support diagnostic](../evidence/EXEC-TRANSFER-20261013/oracle/human-support.json)
finds peak-speed 0.03 timing budgets around 3.4–4.5 ms for those fast strokes
under an exact local path model. These are geometric sensitivities, not measured
timing tolerances. A 0.03 normalized circle is an ellipse in logical coordinates,
with semiaxes 12.42 and 26.88 points; the viewer's ten-point ring is a position
marker, not that tolerance contour. Under the extraction's inferred event
partition, finite, non-negative 120-frame windows cannot isolate any of the
seven fast spans because neighboring motion is too close. Only the catch has
a feasible range, subject to available source context. This is conditional on
the extracted contact partition. Merely adding short durations does not address
multi-contact windows, unknown endpoint holds or the executor's executable
trajectory set.

## Decisions supported now

Keep round 6 failed and the existing corpus/training gate intact. Characterize
executor delivery independently of glow, declare whether labels mean executor
commands or raw human touches, and evaluate fresh same-executor replay on
supported human effects. A model of expected rendered motion can inform a
future review overlay only after held-out device validation; it cannot
retroactively convert the failed review into qualification.

Prioritize the minimal coordinate-refit noise diagnostic and a held-out
simulator compensation check before changing production. A future physical
transfer test needs independently observed touch timing or a validated
render/phase model, straight legs plus corners and zero-distance holds, repeated
exact payloads, and predeclared failure criteria. Keep recording segments at
one minute and collection output isolated with honest park provenance. Such a
physical test remains a separate, explicitly bounded proposal requiring
operator authorization; it was not run here.
