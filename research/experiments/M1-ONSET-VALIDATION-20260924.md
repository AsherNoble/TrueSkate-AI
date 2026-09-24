# M1-ONSET-VALIDATION-20260924 — Held-out trace-onset check

Status: all 24 human labels and the preregistered comparison complete. No new
collection, corpus edits, training or exclusion decision.

## Question

Does the unusual two-anchor calibration rate predict a late first visible
trace in compact clips? If so, does the error shrink toward the end of a
high-rate Los Angeles recording, as expected from a falsely early *start*
calibration detection? Do low-rate recordings elsewhere show the opposite
pattern expected from a falsely early *end* detection?

## Frozen selection

The [preceding balanced audit](M1-AUDIT-20260924.md) supplied the hypothesis.
This follow-up uses 24 clips from 12 different one-minute recordings, none of
which appeared in the 140-clip audit. Each recording contributes its earliest
and latest admitted gesture. A random seed (6998894493552140267) selected:

- Four XR1/Los Angeles high-rate recordings (rate > 1.004, start-control
  detector score < 20).
- Four XR1/Los Angeles ordinary-rate recordings (abs(rate - 1) < 0.0008,
  start-control detector score >= 25).
- Four low-rate recordings (rate < 0.995) in other parks: two Skateboard GB,
  one Kansas City and one Super Crown.

The 24 clips were shuffled individually. The viewer displays neither park,
rate, pair, expected onset nor swipe overlay. The exact preselected paths,
order, rates and categories are frozen in
[selection.json](../evidence/M1-ONSET-VALIDATION-20260924/selection.json).
All 24 videos were checked to decode to 32 frames.

## Human task and analysis

For each clip, mark the first displayed frame where a *new* swipe trace is
visible, or mark it uncertain. The viewer saves locally and exports
model1-onset-validation-20260924.json. Frame numbers shown to the operator
are one-based; the export includes both zero-based and one-based numbers.

After labels arrive, compare each human frame with the first stored
frame_times >= 0 frame, using the exact source timestamps rather than an
invented constant frame grid. Report all 24 differences, uncertain marks,
category summaries and each recording's early-to-late difference. The predicted
pattern is larger positive onset delay early in high-rate Los Angeles
recordings, ordinary timing in the matched Los Angeles group, and potentially
larger delay late in low-rate other-park recordings. These predictions are
falsifiable; a small set does not justify a universal exclusion threshold.

The compact clips do not contain the calibration touches themselves. Even if
the rate predicts swipe error, identifying the visual false-positive mechanism
will require a new bounded recording that retains its original video.

## Results

The operator labelled all 24 clips; none was uncertain. The
[label export](../evidence/M1-ONSET-VALIDATION-20260924/labels.json) and
[exact timing fields](../evidence/M1-ONSET-VALIDATION-20260924/timing.json)
are preserved. Every clip's first stored non-negative frame time is displayed
frame 8.

| Segment category | Early clip first trace | Late clip first trace |
|---|---|---|
| Four high-rate Los Angeles recordings | 13, 18, 17, 12 | 8, 9, 9, 9 |
| Four ordinary-rate Los Angeles recordings | 8, 8, 8, 8 | 8, 8, 8, 8 |
| Four low-rate other-park recordings | 9, 9, 9, 9 | 19, 16, 19, 16 |

All four high-rate Los Angeles pairs improve toward the end control; all four
low-rate other-park pairs worsen toward the end control. The ordinary-rate
Los Angeles recordings stay aligned at both ends. This supports the predicted
anchor-error direction in all twelve pairs, on recordings unseen in the first
audit.

For an early start-control false detection, approximate actual-onset delay as
(rate - 1) × (end-control WDA time - gesture WDA time). For an early end-control
false detection, use (1 - rate) × (gesture WDA time - start-control WDA time).
Map that delay to the first exact stored source-frame time at or after it. These
two simple anchor-error formulas match 22/24 human onset frames exactly and
the remaining two within one frame. Ordinary-rate recordings use zero delay.
The early/late directions were specified before annotation; this exact-frame
projection was calculated afterward. It assumes the true clock rate is
approximately 1.0 and does not independently identify the detector's false
visual event.

The result strongly validates the calibration-rate anomaly as a timing-risk
signal at the selected extremes, rather than a park-wide defect or a constant
one-frame compact-video phase error. The high-rate Los Angeles start-control
scores are 12.4–13.3, while the low-rate other-park end-control scores are
6.4–16.3, consistent with weaker detections at the implicated anchors.
Because the selection deliberately sampled extreme rates, it does not set a
safe cutoff for the whole corpus. The earlier screen of 1,063 clips at
abs(rate - 1) > 0.002 remains a risk estimate, not an exclusion decision.

## Artifact

Rig viewer:
/Users/training-server/trueskate-ai-runtime/tmp/model1-24-onset-validation-20260924/viewer/
(served on Tailscale port 8771). The viewer generator is
[build_linear_onset_validation_viewer.py](../../scripts/inspect/build_linear_onset_validation_viewer.py).
