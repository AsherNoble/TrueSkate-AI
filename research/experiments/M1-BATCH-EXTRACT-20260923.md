# M1-BATCH-EXTRACT-20260923 — Decode each recording once

Status: opt-in implementation validated offline; not yet deployed for collection.

## Cause and proposed change

The September exact-PTS collector spends about five minutes per one-minute
recording, admitting roughly ten clips. Recent segment directories show the
recording and save phase takes about one minute; alignment takes another three
to four minutes. The existing direct extractor launches one FFmpeg command per
clip. Each command opens the full recording and selects 32 exact source frames.
The aligner also probes the full recording's frame timestamps separately for
each of the two calibration windows and a third time for clip extraction.

The opt-in `--batch-direct-video` path computes the same source frame numbers and
source timestamps, then uses one FFmpeg decode with separate output branches.
It preserves the two-anchor calibration, command labels, 32-frame requirement,
source PTS metadata and contamination filters. The current per-clip extractor
remains the default and is the fallback if a batch fails or emits an invalid
frame count. `BASIC_LINEAR_BATCH_DIRECT_VIDEO=1` selects the new path in the
bounded linear wrapper.

The candidate probes the source timestamps once and passes that unchanged list
to both calibration windows and the clip extractor. It does not change the
detector, window boundaries, selected frame numbers or timestamp calculations.

## Offline evidence

- The full test suite passed: 332 passed, 2 skipped. Targeted tests compare
  end-to-end metadata and decoded clip pixels, and force a batch failure to
  exercise the per-clip fallback.
- A retained 65.445-second XR2 phone recording with 1,962 decoded source frames
  was tested on the rig's 8 GB Intel MacBook Air while the two production
  collectors remained active. For ten clips, the deployed release took 41.993
  seconds and the candidate took 11.198 seconds. All ten outputs had the same
  source timestamps and exactly 32 pixel-identical decoded frames. Report and
  isolated outputs are in the rig's
  `tmp/model1-batch-validation-20260923/ten/` directory.
- A three-clip rig comparison was slower in batch (13.318 versus 9.501 seconds)
  under concurrent load. The one-minute collector normally produces about ten
  clips, so the ten-clip result is the relevant throughput check. This does not
  yet establish the speedup of a complete recorded/calibrated segment.
- On the same phone recording, the release took 43.841 seconds for its first
  full-video PTS probe, then 64.785 and 58.844 seconds for the start/end
  calibration windows (each of which repeated the probe). The candidate took
  35.791 seconds for one probe and 17.498 and 22.321 seconds for the two windows
  using the shared timestamps. Times vary with the concurrent production load;
  the structural improvement is removing two full-video probes. A regression
  test confirms both windows receive identical pixel arrays and timestamps.

## Promotion gate

Stage a clean committed release and run one isolated one-minute segment on
each XR in its operator-confirmed park. Require accepted two-anchor calibration,
strict clip admission, 32 increasing source times per clip, and no regressions in
foreground or recorder handling. Switch the bounded production collectors only
between segments, preserving their output directories, next seeds, strict
targets and previous release for rollback.
