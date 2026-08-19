# Experiment queue — Model 1 (behavioural cloning)

journal: experiments/rl_poc_experiment_journal.md
plans: experiments/model1_mvp2_999_plan.md, experiments/model1_mvp3_plan.md
run with: the `experiment-queue` skill — one item per invocation, verdict journaled, stop.

Ordered by the owner. Cheap/offline first, paid work gated, holdout work gated twice.

## EQ-001 — Implement the validation-fit end-bias correction (offline)
- status: done: INCONCLUSIVE — implemented and verified, but red team CONFOUNDED the claim it
  supports (design constants read off the test split; effect is 3 clips on n=153, McNemar p=0.375).
  See the 2026-08-19 EQ-001 journal entry. Do not re-propose "the cheap 2 points" without EQ-008/009.
- tier: FREE
- hypothesis: the end-endpoint along-path undershoot is a reproducible scalar bias, so a
  shift fit on the validation split and applied unchanged to test is a legitimate
  correction rather than test-set tuning.
- method: add the correction to the decode path behind an explicit flag, fit from
  validation recovery records only; unit-test on synthetic records with a known injected
  along-path bias (recovered shift within tolerance, no effect on the perpendicular
  component, no-op when the bias is zero). No cloud run, no checkpoint access.
- expected: tests green; a known injected bias of b is recovered to within 10%.
- kill: the correction cannot be expressed without touching test-split statistics — in
  which case it is test tuning and the whole line is abandoned.
- why: gates EQ-002, which is the cheapest ~2 points available anywhere in this project.

## EQ-002 — Does the predicted-chord operator reproduce the commanded-chord counterfactual?
- status: todo
- tier: PAID
- blocked-by: EQ-008, EQ-009
- hypothesis: the shipped correction (direction from the *predicted* chord) behaves the same as the
  autopsy's operator (direction from the *commanded* chord) on real clips, within 1 clip.
- method: one Modal evaluation of `basic_linear_linear_mixed_fresh_holdout_20260813` over its exact
  existing fresh split: fit on validation, apply to test, record the **discordant pair counts**
  (clips that flip pass->fail and fail->pass), not the accuracy delta. Report McNemar exact p and
  the Clopper-Pearson interval alongside the point estimate. Evaluation only, on the split this
  checkpoint has already been scored on — consumes no new holdout.
- expected: discordant pairs of roughly b=3-4 / c=1, i.e. the operators agree to within a clip.
  EQ-008's red team simulated this operator difference over 6,120 clips and found b=0, c=0 with
  per-clip displacement median 1.0e-4 / P99 5.5e-4 — so the prior is agreement; this run is the
  real-record confirmation, and it is worth doing only because a simulation cannot see a real
  perpendicular-error distribution wider than the autopsy's 0.0032 sd.
- kill: the two operators disagree on more clips than the correction gains — then the predicted-chord
  direction is not a valid stand-in and the correction needs the commanded chord, which is not
  available at inference.
- why: NOT worth ~2 points. The red team established the effect is 3 clips on n=153, two-sided
  p=0.375, with a confidence interval whose lower bound sits below the 0.95 gate. This run buys
  *operator agreement*, which is a prerequisite for using the correction at inference at all — and
  it can only be demonstrated as significant on the >=3,000-clip holdout of EQ-007.

## EQ-003 — Duration: the binding constraint above 98.7%
- status: todo
- tier: PAID
- hypothesis: duration failures are a distinct, characterisable population (not scatter),
  as the endpoint failures turned out to be.
- method: dump per-clip duration error against commanded duration, dx, slope band, device
  and frame count over the existing split; classify the failures the way the endpoint
  autopsy did before proposing any model change.
- expected: a taxonomy with counts; a dominant bucket.
- kill: duration error is unstructured across every covariate — then it is a precision
  limit and needs capacity or more frames, not a decoder fix.
- why: at 98.69% (the endpoint ceiling on the current checkpoint) duration is 100% of the
  residual, and it has had zero attention.

## EQ-004 — Resolve or bury the line-fit decoder
- status: todo
- tier: PAID
- hypothesis: the line fit's 83.17% (vs baseline 90.10%) is a configuration artefact —
  `trajectory_weight=0.02` untuned, equal knot weighting instead of the baseline's 1.8x
  start weighting, and a cold onset head trained inside the same 40 epochs.
- method: `trajectory_weight` sweep at K=2 on the frozen 2,022-command split with baseline
  knot weighting restored; same epochs, same seed, one variable at a time.
- expected: >= 90.10% at some weight if the artefact explanation holds.
- kill: no setting reaches the baseline — then the line fit is closed in the journal as a
  falsified architectural bet so it cannot be re-proposed.
- why: it was the primary bet of the MVP-2 plan and is currently a regression; leaving it
  ambiguous invites re-litigating it every session.

## EQ-005 — Derive the Model-1 fidelity target instead of asserting 99%
- status: todo
- tier: PAID
- hypothesis: Model 2 has a tolerance to Model-1 error, and it is not 99% per knot.
- method: inject controlled noise of magnitude eps into ground-truth gesture parameters,
  train Model 2 on the perturbed targets, and find the eps at which its performance
  degrades. Sensitivity study, synthetic perturbation, no new collection.
- expected: a curve giving the eps that Model 2 tolerates, converted into a per-knot
  recovery requirement.
- kill: Model 2 is degenerate at every eps including zero — then the requirement question
  is premature and this closes until Model 2 has a working baseline.
- why: "99%" is currently an assertion. The 2026-07-19 stroke-recovery reframe measured
  *current* fidelity and returned a negative; this measures the *requirement*, which has
  never been measured. It could raise the bar or lower it — both are useful.

## EQ-006 — Owner actions that gate everything downstream
- status: blocked
- tier: FREE (to document) / owner decision (to execute)
- blocked-by: owner
- items: (a) paid Apple Developer account — the free 7-day team means any restart, reboot
  or crash ends collection until an interactive sign-in at the rig; (b) the autooffload
  `MIN_SPIN_FRAC=0.8` gate against the collectors' `--spin-frac 0.5`, which strands 295 GiB
  locally and will never offload as configured; (c) confirm the target volume has room
  before flipping (b) — `trueskate-corpus` was effectively full on both axes.
- why: no amount of model work substitutes for these, and (a) is the single biggest
  schedule risk to certification.

## EQ-007 — Certification protocol for a >= 99% claim
- status: blocked
- tier: HOLDOUT
- blocked-by: EQ-006, collection
- hypothesis: n/a — this is a measurement protocol, predeclared before any data is spent.
- method: >= 3,000 untouched unique commands, device-balanced; headline is the
  Clopper-Pearson 95% lower bound, not the point estimate; staged in tranches of 1,000;
  test evaluated exactly once; no post-hoc tolerance changes.
- expected: n/a.
- kill: n/a.
- why: by the rule of three, zero failures in the current 303-command slice certifies only
  99.01%. Until this exists, no 99% claim is measurable, and any loop optimising toward
  one is optimising noise.

## EQ-008 — Make the fit and the apply share one axis, and test the case that matters
- status: done: CONFIRMED (restated) — axes agree to ~4e-5, 0.14% of tolerance. Red team CONFOUNDED
  the stated reason: delta = E[perp^2]/|chord|, driven by PERPENDICULAR error and first-order
  insensitive to first-knot error (x0 removed: 3.6e-5; perp x5: 8.9e-4, at the gate). Transfer is
  data-dependent on perp sd 0.0032 holding. See the 2026-08-19 EQ-008 journal entry.
- tier: FREE
- hypothesis: the correction is currently measured on the commanded chord and applied on the
  predicted chord; making them consistent changes the fitted shift by less than the noise floor.
- method: project the fit onto the same predicted chord `.apply()` uses (`signed_along_path_error`
  currently discards the predicted `previous` knot); add tests whose first knot is displaced from
  the commanded start, so the predicted and commanded chords genuinely differ — every current test
  has them agreeing to 2.1e-8 rad, and would pass even if `.apply()` read the target.
- expected: fitted shift moves by < 0.001; the new tests fail before the change and pass after.
- kill: the consistent fit changes the shift materially — then the two operators are not
  interchangeable and the autopsy's numbers do not transfer at all.
- why: without this, EQ-002 measures a different estimator than the one that will ship.

## EQ-009 — The missing fit-on-validation -> apply-on-test entry point
- status: done: CONFIRMED after correction — `evaluate_bias_correction()` exists and takes its split
  identity from the checkpoint. Red team CONFOUNDED the first version: the disjointness assertion was
  vacuous and a re-derived split on a grown corpus would have put trained-on commands in "test".
  See the 2026-08-19 EQ-009 journal entry.
- tier: FREE
- hypothesis: n/a — implementation.
- method: add a Modal entry point that fits `AlongPathBias` on the validation split and evaluates
  the test split with it, emitting discordant-pair counts. Record provenance (which split the shift
  was fit on, sample count) on the `AlongPathBias` so the "not test tuning" guarantee is carried by
  the artefact rather than by a docstring.
- expected: runnable, CPU smoke passes locally.
- kill: n/a.
- why: nothing currently calls the correction; EQ-002 was not runnable as written.

## EQ-010 — Verify perpendicular error sd on the EQ-002 split before trusting the axis transfer
- status: todo
- tier: PAID
- blocked-by: EQ-009
- hypothesis: the perpendicular error sd on the split EQ-002 runs is near the autopsy's 0.0032, so
  the predicted/commanded axis agreement measured in EQ-008 transfers.
- method: fold into EQ-002's single evaluation — report the perpendicular error distribution
  (sd, P90, P99) alongside the discordant-pair counts. No separate run.
- expected: sd within ~2x of 0.0032, i.e. |delta| stays under ~1.5e-4.
- kill: sd above ~0.016 (5x), where the axis disagreement reaches the 0.001 gate and the shipped
  operator is no longer interchangeable with the one the autopsy measured.
- why: EQ-008 established delta scales with the SQUARE of perpendicular error, so the whole
  predicted-chord design rests on a number measured once, on one checkpoint, on one split.

## EQ-011 — Sweep the Modal evaluators that ignore the checkpoint's knot count
- status: done: CONFIRMED at k=2, FALSIFIED at k=3 — resolving knots made k=3 datasets constructible
  but three evaluators decode a hardcoded 5-wide layout, so the sweep turned two crashes into silent
  mislabelled artefacts until `_require_two_knots` was added. Red team CONFOUNDED; the call-site
  guard was also inverted and is now a structural check. See the 2026-08-19 EQ-011 journal entry.
- tier: FREE
- hypothesis: six evaluators build `BasicLinearClipDataset` without passing `knots` from the payload,
  so any k>2 checkpoint is scored against a 2-knot target and throws a shape error after loading the
  whole corpus (or, worse, silently compares the wrong components if a future change makes the
  shapes compatible).
- method: pass `knots` from the payload at every call site, as EQ-009 now does; extend the existing
  `_payload_resolution` call-site guard to cover the knot count too. (The helper was renamed
  `_payload_dataset_kwargs` in the course of this item.)
- expected: no behaviour change for k=2 checkpoints; k=3 checkpoints become evaluable.
- kill: n/a — this is a defect sweep.
- why: MVP-3 introduced k=3 checkpoints, so this is now live rather than hypothetical, and it wastes
  a full corpus load before failing.

## EQ-012 — Make the two endpoint-decoding evaluators knot-general
- status: todo
- tier: FREE
- hypothesis: `audit_endpoint_residuals` and `autopsy_failures` can read first/last knot and duration
  from a 2K+1 vector as `basic_linear_bias` and `knot_errors` already do, rather than refusing k>2.
- method: replace the hardcoded `[:, :2]` / `[:, 2:4]` / `[:, 4]` slices with knot-indexed reads;
  drop their `_require_two_knots` guard once the bodies are general; add a k=3 synthetic test.
- expected: identical output at k=2 (regression-checked against a stored k=2 artefact), and correct
  first/last-knot decomposition at k=3.
- kill: the along/perpendicular decomposition is not meaningful for an interior knot — then keep the
  refusal and say so in the docstring.
- why: EQ-011 stopped them lying; it did not make them usable, and MVP-3 is producing k=3 checkpoints
  now.
