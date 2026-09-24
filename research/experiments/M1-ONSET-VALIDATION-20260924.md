# M1-ONSET-VALIDATION-20260924 — Held-out trace-onset check

Status: selection and blinded viewer prepared; human labels pending. No new
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

## Artifact

Rig viewer:
/Users/training-server/trueskate-ai-runtime/tmp/model1-24-onset-validation-20260924/viewer/
(served on Tailscale port 8771). The viewer generator is
[build_linear_onset_validation_viewer.py](../../scripts/inspect/build_linear_onset_validation_viewer.py).
