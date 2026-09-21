# M1-20260921 — Compact-video frame phase

- **Question:** why does the trace appear one output frame before the gesture
  timing stored in the replacement linear corpus?
- **Human audit:** the operator reviewed 35 clips across 16 XR2 sessions and
  marked the first visible trace as displayed frame 7 in every clip. The
  synthetic `frame_times` place gesture time zero between displayed frames 7
  (`-0.0613 s`) and 8 (`+0.0118 s`), so the animated command begins on frame 8.
  Export: `/Users/ashernoble/Downloads/linear-corpus-audit-reviews.json`.
- **Calibration-control check:** one original tranche recording survived an
  interrupted alignment with both centre controls intact:
  `/Users/training-server/trueskate-ai-runtime/data/basic_linear_v2_20260918/iPhone_XR2/iPhone_XR2_20260918_064521/segment_00000.mov`.
  For both controls, the production detector selected the first meaningful
  centre change. The preceding-frame contrast deltas were `-0.067` and
  `+0.004`; selected-frame deltas were `+11.223` and `+11.478`. The detector did
  not introduce the observed one-frame delay.
- **Timeline check:** for the first preserved swipe, the two-anchor WDA fit
  predicted video time `7.433331 s`; the first visible source frame was
  `7.441667 s`, an `8.336 ms` residual. The WDA-to-video fit is within one native
  30 fps frame and does not explain a 73 ms compact-clip slot error.
- **Cause:** the direct extractor used FFmpeg's default nearest rounding in its
  `fps` filter. It assigned the first post-onset source frame to output index 6,
  whose metadata says `-0.0613 s`. The metadata was a synthetic output grid and
  did not account for that nearest-frame choice.
- **Candidate fix:** `fps=...:start_time=0:round=up` assigns source pixels to the
  first output slot at or after their source time, while explicitly filling the
  zero-time slot. Across the eight preserved swipes, the old extractor's local
  detector chose index 6 for 8/8. The candidate chose index 7 for 7/8; one scene
  was detector-inconclusive because of local floor brightness. All candidate
  clips decoded as exactly 32 frames on the rig. The first preserved swipe moved
  from `-0.0613 s` to `+0.0118 s`, matching its raw source onset.
- **Decision:** make causal upward rounding canonical for compact extraction.
  Add a real-FFmpeg regression fixture that forbids future source pixels in a
  pre-onset output slot. Do not repair this cheap 1,109-clip tranche; recollect
  after the corrected extractor passes a bounded on-device smoke test.
- **Smoke-test guard finding:** two isolated XR2 validation segments correctly
  failed admission because their end control was not detected. Their preserved
  recordings show that iOS Control Center's Game Mode panel was already covering
  True Skate from frame zero through the final frame; neither run is evidence
  that a sampled swipe opened it. Appium's `query_app_state` still returned True
  Skate as foreground, while WDA `/wda/activeAppInfo` correctly returned
  `com.apple.springboard`. The device guard now consults that frontmost-bundle
  endpoint at connection and before/after every gesture. A SpringBoard overlay
  causes True Skate to be reactivated and the instrumented segment to be
  discarded. Final clean on-device validation remains pending.
