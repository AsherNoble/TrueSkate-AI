# Simulator schedule versus round-6 video: frozen plan

Frozen 2026-10-13 before extracting new head measurements or fitting video alignment.

Question: do independently measured simulator forward schedules explain moving
trail positions in the 100 physical round-6 clips better than the nominal linear
command schedule? This is a selected-region transfer test, not collection
qualification, a timing certification, or proof of physical event delivery.

Inputs: read-only scratch `audit6-mirror/audit-6/manifest.json`, corresponding
raw clips, prior `research/2-trail-model/data/<id>.npz` raw profiles, and
`research/1-xctest-transfer/resume-20261013/r6-transfer.json`. Prior lag outputs
are exploratory context only. Simulator schedules use actual touch-B as zero;
movie time is the existing two-control calibrated command clock. Their unknown
constant offset is fitted on development recordings only.

Split: recordings ordered by minimum sample ID. Development: first IDs
51,71,91,111,131. Held out: 61,81,101,121,151. Entire recordings remain together.
This supplies one recording of each source-prefix pair to each side; it does
not imply new-device independence. No clip-specific shifts, warp fits, or model
selection on held-out error are allowed.

Measurement: use the prior native-pixel 4-pixel path grid and perpendicular
profiles. The warm core is the maximum at offsets -8..8 px; subtract its
median at pre-onset times <=-50 ms, then Gaussian-smooth along path at sigma
4 px. Find downward threshold crossings at excess 25, without using either
prediction to choose a crossing. Require median signal >=40 from edge-48 to
edge-16 px and <=10 from edge+16 to edge+48 px, at least 80% of samples in
the bright interval above threshold and all samples in the dark interval below
threshold. Require a unique qualifying crossing per frame. Nominal clip-active
time must be >=0 and <=command duration+100 ms.

Require edge +-60 px to be in bounds, >70 px along-path from vertices and
endpoints, command speed 0.5..4 native px/ms, >160 px from all path locations
whose arc coordinate differs by >160 px, no prior texture-proxy flag, median
local noise <=8, and pre-onset core <12. These conservative spatial exclusions
also reject recently retraced paths; holds, endpoint dwell, beyond-path dark
evidence, and clip 67 terminal retrace cannot contribute. The estimated centre
is edge minus 29 px along the current straight path. Perpendicular centre is
unmeasured; comparing XY means observed along-path positions, not independent
2-D localization.

Visual validation precedes fitting: generate source-frame crops at the earliest,
middle and latest accepted frame of every selected clip, and full-frame views
for dubious cases. Reject an entire selected run only for predeclared defects:
board/deck intersection, visible scenery stripe mistaken for glow, multiple
heads/retracing, threshold crossing inside a kink/endpoint, or no visible glow.
Log every judgment and show coverage before/after. Reviewers see source frames
and detected edges only, not prediction errors or model identities.

Prediction: nominal linear XY from command points versus measured simulator XY
(native scale = 2 x logical points), interpolated forward in time. Fit each
model one global shift on development only, minimizing equal-recording mean
of per-clip median Euclidean residual over a grid -150..150 ms, step 1 ms.
Evaluate once on held-out recordings. Aggregate per clip first, then recording;
never count correlated frames as independent replicates. Also show unshifted
errors, signed along-path errors, prediction separation and per-recording
values. Bootstrap only recordings, with the small number of clusters explicit.

Uncertainty: rerun measurement thresholds 20/25/30 and cap radii 27/29/31 px;
fit shifts on development for each setting. Bracket simulator interpolation
with sample-hold. Show +/- one native movie frame time sensitivity and
explain that this does not certify marker/calibration/rendering uncertainty.
Nominal cap27..31 is based on stationary caps; moving-cap transfer is unproven.
The trail width/fade is age-dependent, so fading local peaks and old trail
segments are not treated as contact positions. No lag-curve inversion is used.

Decision: support only if simulator wins by recording and improvement retains
direction across cap/threshold/sampling settings, with enough held-out coverage
to describe the observed waveform. Reject only the tested simulator-video law
if linear consistently wins under those checks. Otherwise report inconclusive.
Any excluded regions or inadequate coverage remain unidentified. A positive
result still cannot isolate app rendering from event delivery or validate
physical holds/corners. All software/source conclusions remain simulator-only.
