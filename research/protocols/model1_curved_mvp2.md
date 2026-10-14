# Model 1 MVP 2.0 — arbitrary single-drag recovery

Operator requirements recorded 2026-10-03. This is a design record, not a
collection/training launch or a frozen workload. Explore mode applies
(AGENTS.md); audit sizes marked certify-mode apply only on request.

## Success criterion

The MVP succeeds when the intended trick lands (the AGENTS.md north star). The
operator's practical test is an expert demonstration of a single trick, the
kickflip first, followed by Model 1 gesture inference and execution of the
inferred gestures in the Workshop. Score whether the intended trick is
reproduced; Model 2 is not required. Retain positional accuracy and human
overlay review alongside this end-to-end outcome so failures can be diagnosed.

Use demonstrations whose strokes are single-finger and sequential, such as the
kickflip's pop then flick, and recover each stroke separately. Overlapping
contacts exceed the current scope. Flatground reduces obstacle/location
variation, but compare reasonably matched initial speed, heading and
grounded/airborne state. This replay-test control does not impose a
stationary-start restriction on the training corpus. Choose the held-out
demonstrations and repetitions before evaluating a trained candidate.

## Scope and representation

One continuous finger drag, no spin-button contact. Recover arbitrary expert
finger paths, including multiple bends, loops, reversals, pauses and changing
speed. The nine-number single cubic remains a diagnostic family, not the
expressive ceiling for this model.

Use a time-parameterized trajectory p(t) = (x(t), y(t)) with an extensible number
of pieces. Timed waypoints with piecewise interpolation are the initial design
candidate; multiple spline pieces are another implementation option. Spatial
geometry alone, with a single total duration and constant-speed traversal, is
insufficient. Preserve normalized coordinates and touch-down/up times. Final
interpolation and duration bounds are not yet frozen. The operator chooses a
provisional cap of **15 timed waypoints including both endpoints**, allowing
2–15 points (up to 14 intervals). No manual survey of expert recordings is
required before trying this MVP. Revisit the cap if observed recovery needs it;
do not silently smooth away corners or reversals to fit a fixed small vector.
The finite representation approximates paths within a measured tolerance; it
cannot promise unlimited temporal/spatial detail from finite video.

## Execution-fidelity viewing

Next curved diagnostic viewer: one moving **unfilled ring of radius 10 logical
points**, no drawn trajectory line or accumulated target trail. Scale its radius
with the logical-to-video display transform so the interior stays unobscured. Its center evaluates the time-parameterized
target at each native video timestamp, including pauses and changing speed.
Allow the circle to be hidden so it does not obscure the observed contact.

Use five-contact die-five calibration markers as explicitly directed by the
operator, with per-recording clock alignment and independent timing checks.
Calibration is derived from those contacts, not adjusted per gesture until the
overlay looks right. Keep controls outside gesture clips. Verify the deployed
marker configuration when implementing the next bounded pilot: older repository
records contain several comparison settings and center-only diagnostic runners.
Retain the abstract target and the exact quantized, timed WDA command separately,
so compiler approximation and delivery errors remain distinguishable. The target
used as a training label must agree with the chosen execution/label contract.

Human qualitative review determines whether the target circle follows the
visible trace; automatic menu/editor classification is not requested for this
pilot. The latest linear single-movement duration result does not establish
curved multi-movement delivery fidelity or expert-gesture minimum duration.
The operator reports that traces are always visible in expert demonstrations;
this is the intended observation domain.

## Collection coverage and game state

Accept gestures beginning in any gameplay state: stationary, rolling, airborne,
turning, landing, with moving camera/background. Do not require a stationary
board at gesture t=0 or use motion as a sample exclusion. Calibration-control
preparation, if needed, is distinct from training-example admission.

Coverage means avoiding accidental concentration of generated examples in easy
shapes or screen regions. A sampler should cover locations, orientations,
lengths, straight and curved portions, corners, loops/reversals, timing/speed
profiles, and gameplay states. These are sampling dimensions rather than a
closed vocabulary or restrictions on recovered expert gestures. Real trick
strokes are fast: the filmed kickflip's pop takes 75 ms and its flick 100 ms
([record](../experiments/KICKFLIP-REPEATABILITY-20261009.md)), so the speed
range must reach them (SYSTEM_MAP, blocker 1). The operator prioritizes the
widest useful variety, including simple/common motions and complex paths across
moving gameplay states. Exact quotas, initial sample count and park/session mix
remain to be chosen; begin with broad coverage and refine using observed
omissions rather than requiring an expert-video survey.

## Accuracy and audits

For each native frame timestamp t_f during contact, evaluate both trajectories
and measure ||p_pred(t_f) - p_target(t_f)||_2. The operator requests the closest
practical equivalent to the linear model's accepted error, with perfection
deferred if the MVP works. The current linear scorer uses normalized Euclidean
position tolerance **0.03** and absolute duration tolerance **0.10 s**
(`model1/linear/training.py`). Reuse these as provisional MVP tolerances until
the outcome-tolerance test (AGENTS.md, "Precision from outcomes") gives a
measured bound, applying position comparisons along the whole path, including
endpoints, rather than only to start/end. Report per-frame distributions and
per-gesture pass rates, plus touch-down/up timing errors; final
aggregation/contact-boundary treatment needs an explicit implementation contract
before scoring.

The inherited normalized metric is anisotropic: a pure 0.03 x displacement is
12.42 logical points on XR, a pure y displacement is 26.88. Also report isotropic
logical-point errors for interpretation; do not call 0.03 a single pixel radius.
Parameter-vector error is not the primary metric: different parameterizations
can describe the same trajectory.

Keep two checks distinct: operator audit establishes that generated labels agree
with visible execution; model evaluation compares predictions to those verified
targets. In explore mode an operator spot-check of about 20 clips is enough
unless a decision depends on an exact rate; certify-mode audits use about 100
held-out clips. Keep recording/session splits independent, avoid reusing final
audit clips to tune the compiler or scorer, and preserve raw video, commands,
calibration and human labels. A passing finite audit supplies evidence, not
universal certification.

## Meaning of WDA subdivision

Curves may be translated into timed straight movements while keeping one finger
down. The implementation must reproduce the desired geometry and speed at
that level. The model's 15-waypoint budget and the execution movement count
are separate concepts; interpolation may require additional movements. The
observed 10–15 ms whole-gesture response boundary does not establish a minimum
duration for movements inside a continuous contact. Human ring-overlay review
will check whether the translation follows the intended trajectory.

## Remaining decisions before scaled collection

- Timed-waypoint interpolation under the provisional 15-point cap.
- Curved command subdivision that passes calibrated human overlay review.
- Exact frame/contact scoring contract using the provisional linear tolerances;
  maximum gesture duration.
- Initial collection mixture and held-out split/audit selection.
