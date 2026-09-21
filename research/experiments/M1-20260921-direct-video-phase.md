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
- **Rejected interim fix:** `fps=...:start_time=0:round=up` fixed the first
  preserved swipe and moved 7/8 local-detector results from index 6 to index 7;
  one scene was detector-inconclusive. A later clean XR2 segment still produced
  mixed detector positions (six index 6, four index 7, one missed, with one
  unrelated late scene change). The local detector is imperfect on moving
  scenery, but the larger design problem remains: resampled pixels are paired
  with invented times rather than the selected source frames' PTS. Rounding is
  therefore not a trustworthy timing contract.
- **Final fix:** probe the source recording's frame PTS once, choose 32 exact
  source frame numbers from each gesture window, encode those frames directly,
  and store those same source PTS relative to the fitted gesture onset. A
  preserved raw swipe encodes in 0.85 s, produces exactly 32 frames, and places
  its first visible trace at source-relative `+0.008336 s` (index 7). A
  real-FFmpeg regression fixture verifies that an unambiguous onset's encoded
  frame and stored source PTS agree.
- **Decision:** exact source-frame selection is canonical for compact linear
  clips. Do not repair this cheap 1,109-clip tranche; recollect after the exact
  extractor passes a bounded on-device smoke test. A later inventory confirmed
  that the tranche contains 1,109 compact clips but only one surviving original
  `.mov`, and that recording has no emitted compact clips because its alignment
  was interrupted. The complete tranche therefore cannot be faithfully
  regenerated from original pixels; it remains an audit artifact only.
- **Smoke-test guard finding:** two isolated XR2 validation segments correctly
  failed admission because their end control was not detected. Their preserved
  recordings show that iOS Control Center's Game Mode panel was already covering
  True Skate from frame zero through the final frame; neither run is evidence
  that a sampled swipe opened it. Appium's `query_app_state` still returned True
  Skate as foreground, while WDA `/wda/activeAppInfo` correctly returned
  `com.apple.springboard`. The device guard now consults that frontmost-bundle
  endpoint at connection and before/after every gesture. A SpringBoard overlay
  causes True Skate to be reactivated and the instrumented segment to be
  discarded. After the operator closed Control Center, a third bounded segment
  passed the guard and two-anchor calibration (rate `0.99994434`), admitted 11/11
  payload clips, decoded all clips at 32 frames, and left True Skate frontmost.
  That run used the rejected rounding candidate. A final bounded XR2 segment
  using the exact source-frame extractor then passed the same checks: two-anchor
  calibration was accepted (`rate=0.99994441`), 11/11 payload clips were
  admitted, every compact clip decoded to 32 frames with 32 stored source
  relative timestamps, and WDA still reported `com.trueaxis.skate` frontmost.
  The extractor is therefore validated on-device for this collection path.
