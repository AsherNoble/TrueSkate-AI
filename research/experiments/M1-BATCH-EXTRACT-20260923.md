# M1-BATCH-EXTRACT-20260923 — Decode each recording once

Status: opt-in implementation validated offline and on both XRs; deployed to
the bounded replacement collection on 2026-09-23.

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

- The full local test suite passed: 333 passed, 2 skipped. Targeted tests compare
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

## On-device promotion

A clean candidate release at `1b8497e66cbde00928eac310b446ceecadf6096a`
ran one isolated one-minute segment per XR. XR1 in The Workshop admitted 10/10
strict clips; XR2 in SLS 2013 Kansas City admitted 9/9. Both had accepted
two-anchor calibration, correct device and park provenance, 32 increasing
source-relative times and exactly 32 decoded frames per clip. Neither emitted
`.menu` or `.trace_mismatch` samples or used the per-clip fallback. The strict
audit passed for both. Logs and outputs are retained under the rig's
`tmp/model1-batch-validation-20260923/` directory.

The candidate sessions reached `.aligned` in 169.4 seconds on XR1 and 169.0
seconds on XR2; the last ten old-release production segments had medians of
304.7 and 303.9 seconds, respectively, from segment start to `.aligned`.
These are observed comparisons, not a long-run throughput guarantee.

The stable rig link was switched from release `13554c9` to `1b8497e` between
production segments. The previously authorized bounded XR1 and XR2 collectors
were resumed with `BASIC_LINEAR_BATCH_DIRECT_VIDEO=1`, preserving their output
directories, persisted next seeds and strict targets of 4,050 and 3,800.
Release `13554c9` remains available for rollback. WDA was not restarted.

The first resumed production segments admitted 11 XR1 and 10 XR2 clips. The
existing strict corpus audit passed each session with correct park/device
provenance and no exact-command duplicates; all clips had a decodable
`frames.mp4`. Their aligners accepted two-anchor calibration, and neither
logged a batch fallback. The next live collection processes included
`--batch-direct-video` and advanced the persisted seeds to `235779031` (XR1)
and `58035956` (XR2). Those first production segments reached `.aligned` about
143–148 seconds after their named session start; the next segments began
roughly 167–170 seconds after the prior wrapper launch. The Appium disconnect
warning still appears after successful sessions on both the old and new
release; it did not stop either bounded wrapper.
