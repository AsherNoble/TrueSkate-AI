# Model 1 MVP 2.0 — arbitrary single-drag recovery

Operator requirements recorded 2026-10-03. This is a design record, not a
collection/training launch or a frozen workload.

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
interpolation, point budget and duration bounds are not yet frozen. Select these
against a positional approximation tolerance on representative expert paths;
do not silently smooth away corners or reversals to fit a fixed small vector.
The finite representation approximates paths within a measured tolerance; it
cannot promise unlimited temporal/spatial detail from finite video.

## Execution-fidelity viewing

Next curved diagnostic viewer: one moving target circle, no drawn trajectory
line or accumulated target trail. Its center evaluates the time-parameterized
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
closed vocabulary or restrictions on recovered expert gestures. Exact quotas,
initial sample count and park/session mix remain to be chosen; diversity should
be checked against actual expert demonstrations.

## Accuracy and audits

For each native frame timestamp t_f during contact, evaluate both trajectories
and measure ||p_pred(t_f) - p_target(t_f)||_2. Use isotropic screen units (e.g.
logical points after applying screen width/height), not unadjusted Euclidean
normalized x/y distances on a non-square display. Report typical and tail path
errors, per-gesture pass rates and touch-down/up timing errors. Parameter-vector
error is not the primary metric: different parameterizations can describe the
same trajectory. Numerical tolerances remain open.

Keep two checks distinct: operator audit establishes that generated labels
agree with visible execution; model evaluation compares predictions to those
verified targets. Continue approximately 100-clip held-out human audits. Keep
recording/session splits independent, avoid reusing final audit clips to tune
the compiler or scorer, and preserve raw video, commands, calibration and human
labels. A passing finite audit supplies evidence, not universal certification.

## Remaining decisions before scaled collection

- Piecewise representation/interpolation and a demonstrated approximation budget.
- Curved command subdivision that passes calibrated human overlay review.
- Numerical position/timing tolerances, duration and point limits.
- Initial collection mixture and held-out split/audit selection.
