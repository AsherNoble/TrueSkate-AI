# ALIGN-20260913 — Instrumented-WDA onset alignment

- Date: 2026-09-13.
- Question: can WebDriverAgent's internal `submitted_to_ios` timestamp align a
  separately-executed gesture to its video frames accurately enough to self-label
  traces? Bears on the STATUS open question on labelling expert recordings.
- Instrument: WDA fork `feature/gesture-timing-instrumentation` (build revision
  `b5ace217`) records eight boundary marks per `POST /actions` (monotonic and
  epoch): request_entered, preparation_started, preparation_finished,
  submitted_to_ios, ios_completion_callback, stability_wait_started,
  stability_wait_finished, request_finished. Capture is opt-in via
  `POST /wda/actionTiming`; zero overhead when disabled (`timing` stays nil).
  Not merged; hand-launched on XR2 outside the service supervisor.
- Method: probe `probe_gesture_timing.py` sends calibration taps + linear swipes
  as SEPARATE `/actions` requests with ~1s gaps, records 30fps video, retrieves
  the timing records. A human marks the first visible frame of each touch in a
  frame-by-frame annotator (`tools/build_viewer.py`) that shows no predictions or
  expected times. Analysis anchors on the FIRST calibration mark:
  `pred[i] = mark[0] + (submitted_to_ios[i] - submitted_to_ios[0])`;
  `residual[i] = human_mark[i] - pred[i]`.
- Data: iPhone_XR2, park "Skateboard GB 2024", one session block, four
  recordings — run05 (3 cal + 6 gestures, duration-repeat); run06/07/08
  (1 cal + 5/7/8 randomised linear gestures with constant speed, seeds
  606/707/808). Timing records, manifests
  and human labels retained under `tmp/timing-audit-run0*` and Downloads exports.
- Result: 28/28 non-anchor gestures within one video frame. Pooled residual
  mean −1.5 ms, sd 13.4 ms, max 28.8 ms (0.87 frame). Per-run max residual:
  run05 0.87, run06 0.51, run07 0.61, run08 0.63 frame. The residual sd (13.4 ms)
  is at the two-frame quantisation floor √2·(33.3/√12)=13.6 ms, so true submit→
  onset jitter is **≤ ~13 ms** (a ceiling set by 30 fps labelling, not a point
  estimate).
- Strong-null (red-team): substituting other timestamps into Method A gives
  per-run max residuals — `t_call_start` (host send time, no instrumentation)
  1.5–7.2 frames; `request_finished` ~26–28; `ios_completion_callback` ~28.
  `submitted_to_ios` (0.5–0.87) beats the free host clock by up to ~7 frames, so
  it captures real per-gesture host→submit variation (60–140 ms sd) and is not
  equivalent to "gestures sent ~1 s apart". Non-circular: blind frame marks vs
  WDA's internal clock, no shared fitted parameter.
- Red-team verdict (experiment-red-team): **CONFIRMED**, with two claims trimmed
  below. Residuals reproduced exactly from raw files; record↔label pairing clean.
- Trim 1: the ~27 ms per-run spread in *absolute* submit→onset latency
  (159.7/144.8/132.5/139.0 ms) is confounded with phone↔host clock skew
  (`submitted_to_ios` is device-clock, `started_at_epoch_s` host-clock; run05 is
  ~33 min before 06–08; means are not monotonic in wall-clock order). It is NOT a
  physical touch-latency drift. Per-recording calibration is still the robust
  choice — absolute latency needs a clock sync never verified — and Method A
  (within-recording deltas over ~20 s) is immune.
- Trim 2 (power): n=28 gestures, 4 recordings, one device, one park, one
  afternoon. Rule of three: 0/28 exceedances certifies only a failure rate
  < ~11%, not "every gesture" (max is already 0.87 frame). Single-anchor runs
  inject the anchor's ±½-frame quantisation as a per-run bias (−11.5/+5.5/−6.4/
  +8.3 ms); the pooled −1.5 ms mean is partly those biases averaging out, not
  per-run sub-ms accuracy.
- Provenance limits: one device, one park, one session block; human marks are
  frame-quantised (±1 frame). Absolute latency depends on unverified clock sync.
- Conclusion: `submitted_to_ios` plus one calibration touch per recording aligns
  every gesture onset within one frame across randomised linear gestures with
  constant speed, and beats the
  uninstrumented host clock decisively. This validates the alignment step for
  instrumented-WDA self-labelling. Not yet certified: cross-park/session/device
  generalisation, and that a single anchor transfers across a full recording —
  next test is a multi-anchor recording (2–3 calibrations spread across the
  timeline; confirm inter-anchor residuals < 1 frame).
