# DECODER-NOISE-20261013: frozen synthetic stress plan

This plan precedes generation, scoring or inspection of experiment results.
The study is authorized as an offline synthetic experiment. It changes no
production decoder, model, data, scoring rule, shared research index or rig.

- Source base: `dc159db075f60336ba2543fc7e607c00f5188c51`.
- Cohort seed: `2026101301`; corruption seed: `2026101302`; bootstrap seed:
  `2026101303`. NumPy PCG64; one fixed corruption field per fixture/type, scaled
  across levels. Both variants receive identical float32 arrays and logits.
- Generate 192 legal commands: 24 per family: line, kink, alternating turns,
  movement–hold–movement, sampled arch, near expanded left spin boundary,
  near expanded bottom controls, near expanded top controls. Durations and
  integer segment times vary independently, with >=100 ms intervals,
  300–1200 ms total and <=10 intervals. Fixture generation rejects illegal
  original commands before noise; rejection attempts and fixture hashes are
  recorded. It never selects cases using decoder performance.
- Each fixture uses a 120-frame 60 Hz timeline, random subframe onset phase,
  and a generated calibration rate in [.999, 1.001]. `targets()` supplies
  float32 coordinates and nearest-frame boundary classes with its existing
  endpoint overwrite convention. This is a new generated challenge cohort,
  not a training, validation, qualification or human-trace holdout.
- Coordinate conditions: clean, independent Gaussian XY noise, stationary
  AR(1) noise with rho=.9, linear drift from -v to +v over contact, and
  endpoint-local bias with exponential two-frame decay. Grid:
  .005, .01, .02, .03, .04, .06 normalized units plus clean. Gaussian levels
  specify radial RMS before clipping (per-axis SD=level/sqrt(2)); drift and
  endpoint levels specify endpoint vector magnitude. These are stress levels,
  not empirical prediction-error estimates. Clip inputs to [0,1] for both
  variants, matching a bounded coordinate head; record clipping frequency.
- Fully cross every coordinate condition with onset and release class shifts
  independently in {-1,0,+1}: 25 coordinate conditions x9 boundary patterns
  x192 fixtures =43,200 paired cases. A boundary-class error changes logits,
  not native timestamps or XY data. This differs from shifting the whole
  continuous trajectory in time.
  Boundary logits are -0.5*((frame-shifted_class)/.75)^2, so the existing joint
  duration mask can choose the nearest legal pair if the shifted peaks violate
  its limits. Record intended and selected classes; no one-hot tie fallback.
- Compare frozen `decode()` against the minimal existing
  `refit_grid_coordinates()` research function. Keep its chosen knot times,
  segment count and fixed endpoints. Refit receives no true command or event
  timing. Quantize both variants through the same existing integer viewport
  compiler and score the compiled coordinates with unchanged `evaluate_path`.
- Report unchanged metric pass and separately admissible pass (metric plus
  every original timing/bounds/control-map/compiler check). Report invalid
  candidates as failures; never drop, retry, replace or fall back to baseline.
- Preserve each case's gains, regressions, safety failure reasons and metric
  values. Summarize paired effects by corruption level/boundary pattern and
  family. Bootstrap paired differences by original fixture (1,000 draws), not
  by treating repeated corruptions as independent samples.
- Measure baseline decode and incremental coordinate-refit latency plus
  compilation/guard overhead on this laptop; report median/p95/p99/max.
  Evaluation time is excluded. No deployment latency or physical fidelity is
  inferred. Inspect clean and noisy results together; do not tune the method
  or grid after seeing scores.
- A favorable result can nominate a sequential pilot; it cannot certify a
  trained model or deployment. Regressions and safety failures take priority
  in the candidacy decision. No simulator, phone, cloud or corpus is used.
