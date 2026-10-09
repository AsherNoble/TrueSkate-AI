# Model 1 replay fine-tuning — discussion note, 2026-10-09

Proposed follow-on to known-gesture training; implementation and optimizer are
not settled. This records a possible build, not authorization to collect or train.

- **Objective:** recover gestures that reproduce the full observed gameplay.
  Matching a trick name is insufficient: pop/flick speed and timing change the
  visible motion even when the named trick is the same. Do not assume different
  gestures are interchangeable for video matching.
- **Replay loop:** generate a gesture sequence, execute it through Appium from a
  waypoint, and collect its reference video. Model 1 predicts gestures from that
  video; reset to the same waypoint, execute the prediction through the same
  setup, and automatically grade replay/reference differences. This can improve
  Model 1 directly; Model 2 is not required. Begin within the current single-drag
  MVP scope before extending to sequences.
- **Training mechanism:** the real game/Appium is not differentiable, so a scalar
  grade alone cannot supply ordinary backpropagation. One candidate is to execute
  small variations around a prediction, retain reliably better candidates, and
  train video-to-gesture recovery on them alongside original known-gesture
  examples. Compare gameplay throughout the clip, excluding cursor/trail overlays;
  avoid alignment that conceals timing errors.
- **Repeatability first:** replay the original sequence repeatedly from the same
  waypoint, check starting motion/state, and measure both the size and onset of
  trajectory divergence. Same commands do not prove identical effective touches
  or initial state. Existing [Bluetooth repeats](../experiments/HID-POINTER-20261004.md#7-repeatability-identical-schedules-different-tricks)
  produced different trick/landing outcomes, but do not isolate intrinsic game
  randomness from input delivery, reset differences or sensitivity to tiny errors.
- **Noise and feedback:** human playability supports the possibility of reliable
  control, not elimination of unpredictable noise. Determine whether variation
  needs correction between gestures or during a gesture, and whether a correction
  can arrive before the outcome is committed. Measure the whole observation,
  inference and delivery delay; Pico's report interval is not reaction latency.
  Repeated replay grades must distinguish improvements from lucky executions;
  exact video reproduction may have a noise floor even when reliable play is possible.
- **Practical direction:** consider a useful, imperfect curved Model 1 followed by
  replay refinement rather than making perfect hardware timing a prerequisite.
  Reliable repeatability and grading still need validation. Consistent execution
  bias may be compensable; intrinsic unpredictable variation cannot be learned
  away. Preserve gesture-recovery audits alongside the gameplay objective.
