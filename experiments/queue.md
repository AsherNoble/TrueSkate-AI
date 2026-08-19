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
- status: done: CONFIRMED — operators agree (disagreement 4.9e-5) AND flip the identical 3 clips;
  94.12 -> 96.08, end median -20.9%, p90 -24.2%. But gained 3 / lost 0, p=0.25: NOT significant, and
  96.08% does NOT pass the 0.95 gate (CP lower bound 91.66%). See the 2026-08-19 EQ-002 journal entry.
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
- status: done: CONFOUNDED then resolved — kill criterion RETRACTED. Duration failures are an
  ONSET/HEADROOM effect: headroom<2 frames fails 2/6 vs 1/300 above (Fisher p=9.6e-4), two clips have
  commanded liftoff past the clip end. Typical error is sub-frame (0.19 frames median). Next step is
  the aligner/window, NOT capacity. Cost $0 (existing artefacts). See the 2026-08-19 EQ-003 entry.
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
- status: done: CONFIRMED — all three decided 2026-08-19. (a) Apple paid account DECLINED and settled;
  free-tier re-signing is routine maintenance, not a schedule risk — plan §6 and memory amended, do
  not re-propose. (b) `MIN_SPIN_FRAC` 0.8 -> 0.3 and collector `--spin-frac` 0.5 -> 0.2, given the
  expert corpus is ~5% spin-active. (c) offload target -> `trueskate-corpus-v2` (v1 is full; the
  migration was already underway). See the 2026-08-19 EQ-006 journal entry.
- follow-up: EQ-014 applies (b) and (c) at the rig — NOT yet applied.

## EQ-014 — Apply the offload fix at the rig (owner-executed)
- status: todo
- tier: FREE (config only) but the FIRST RUN MOVES AND DELETES 295 GiB
- blocked-by: owner — the remote plist edit was blocked here, and the deletion warrants a human
- hypothesis: with the gate below the collectors' actual spin fraction and the target pointed at a
  volume with room, the stranded 295 GiB offloads and local disk recovers.
- method: on the rig, with collectors stopped (verified 2026-08-19: 0 loaded, 0 running):
    P=~/Library/LaunchAgents/com.trueskate.autooffload.plist
    cp -n "$P" "$P.bak.20260819"
    /usr/libexec/PlistBuddy -c "Set :EnvironmentVariables:MIN_SPIN_FRAC 0.3" "$P"
    /usr/libexec/PlistBuddy -c "Set :EnvironmentVariables:MODAL_VOLUME trueskate-corpus-v2" "$P"
    launchctl bootout gui/$(id -u)/com.trueskate.autooffload 2>/dev/null
    launchctl bootstrap gui/$(id -u) "$P"     # env is read at LOAD; editing alone changes nothing
  Then watch `logs/autooffload.log` for the first eligible session instead of the hourly SKIP.
- expected: `iPhone_XR_20260814_042825` becomes eligible; ~295 GiB uploads to corpus-v2; local free
  disk rises from 81 GiB.
- kill: `trueskate-corpus-v2` lacks capacity — then the wall has only moved and a third volume or a
  retention policy is needed BEFORE anything is deleted locally.
- why: this is the dominant lever on local disk and it has been stranded since the session was
  collected. **Verify corpus-v2 capacity first — the pipeline deletes locally after upload, and this
  is post-anchor-fix corpus, the half whose timing is recoverable.**

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
- status: done: CONFIRMED — test perpendicular sd 0.003165 (p99 0.0148) against a 0.016 kill; the axis
  transfer is safe on this split. Folded into EQ-002's single run as designed.
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
- status: done: CONFIRMED (narrow) — endpoint decomposition generalised to k-knot vectors; k=2 output
  bit-identical. NOT the same as "the audits are k=3-correct": red team found the score-peak keys were
  mislabelled for every line-fit (hence every k>2) checkpoint, `recovered` silently got stricter, and
  the failure renderer broke on k=3. All fixed. See the 2026-08-19 EQ-012 journal entry.
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

## EQ-013 — Evidence columns for interior-knot failures
- status: done: CONFIRMED — per-knot trail evidence in both records and summary; trail arithmetic
  extracted to the unit-tested `nearest_trail_gaps`, now flat in K. Red team CONFIRMED (narrowly):
  the summary was still endpoint-only until fixed, and the cost benchmark was withdrawn as not
  measuring this code. See the 2026-08-19 EQ-013 journal entry.
- tier: FREE
- hypothesis: `autopsy_failures` can report a `trail_gap` per knot, not just for first and last, so a
  k=3 clip that failed on its interior knot gets the same evidence-vs-misread verdict the endpoints do.
- method: run `nearest()` for every knot; emit `trail_gap_knot{i}` / `trail_frame_knot{i}` alongside
  the existing start/end keys (which stay, for k=2 artefact compatibility).
- expected: identical keys and values at k=2 plus the new per-knot ones; at k=3 every failing knot has
  an evidence column.
- kill: `nearest()` cost scales badly enough per knot to dominate the autopsy — then sample frames.
- why: EQ-012 made `recovered` gate every knot, so a clip can now fail on a knot the report offers no
  evidence about. The diagnostic is weaker than the gate it explains.

## EQ-015 — Give late-onset clips enough headroom to contain their own liftoff
- status: done: INVALID — my `audit_clip_headroom` was CIRCULAR (`frame_times` is synthesised from
  aligner constants, so `tail` is an affine function of commanded duration and measures nothing). The
  EQ-003 truncation mechanism SURVIVES on RENDERED headroom: 6 clips <2 frames (2.0%), 1 negative,
  monotone dose-response, and corr collapses to +0.039 once low-headroom clips are excluded. No
  unapplied Delta in this corpus. See the 2026-08-20 EQ-015 entry.
- tier: FREE to analyse, PAID to retrain
- hypothesis: duration failures are clips whose commanded liftoff falls at or past frame 31, caused by
  late `trail_frame_start` combined with an aligner Δ currently applied as 0. Restoring headroom
  removes 2 of the 3 known failures without touching the model.
- method: (1) FREE — measure the headroom distribution across the whole corpus, not just these 306
  clips, and count how many clips have headroom < 2; (2) decide between applying the measured Δ ≈
  +1.11s in the aligner versus lengthening `CLIP_WINDOW_S` / the 32-frame sampling; (3) PAID — retrain
  only if (1) shows a material fraction affected.
- expected: a single-digit percent of clips affected; those clips' duration undershoot disappears.
- kill: headroom < 2 is vanishingly rare corpus-wide — then the 3 failures are a curiosity, not a
  lever, and duration is genuinely at its precision floor.
- why: EQ-003 showed capacity cannot fix these clips; the evidence is outside the window.

## EQ-016 — The `strong` trail mask is saturated
- status: todo
- tier: FREE
- hypothesis: `trail_frames_present` is 32/32 on every one of 306 clips because the "strong" threshold
  (0.25 x per-clip max) admits the persistent rendered trail before touchdown and after liftoff.
- method: sweep the threshold on cached clips; report the fraction of frames flagged versus threshold,
  and whether any threshold makes trail presence track the commanded contact interval.
- expected: either a threshold that recovers a real onset/liftoff signal, or a demonstration that the
  trail genuinely never vanishes.
- kill: no threshold separates contact from non-contact — then trail presence carries no timing
  information and the field should be renamed to say what it measures.
- why: if trail presence tracked contact, duration would be readable directly; EQ-003 suggests it does
  not, and that assumption is load-bearing for the whole timing story.

## EQ-017 — Does train share recording sessions with validation and test?
- status: done: CONFIRMED — 100% of validation and test clips sit in a training session; zero sessions
  unique to test. Expected under the design, not a defect. But session identity is a WEAK nuisance
  here (failures do not bunch by session); the real coverage gap is park/day/device. See the
  2026-08-20 EQ-017 entry.
- tier: FREE
- hypothesis: the exact-command holdout is command-disjoint but not session-disjoint (111 sessions are
  shared between val and test), so train may share sessions — and therefore park, lighting and board
  pose — with the evaluated splits.
- method: recompute the split locally from the corpus listing and report session overlap between
  train / validation / test; quantify how many test commands have a train clip from the same session.
- expected: substantial session sharing, since commands are assigned randomly within sessions.
- kill: sessions are already disjoint — then there is nothing to fix.
- why: it does not invalidate the command-disjoint protocol, but it bounds how much generalisation the
  held-out numbers actually demonstrate, and EQ-007's certification protocol should decide this
  deliberately rather than inherit it.

## EQ-018 — Do the clip videos hold as many frames as their labels claim?
- status: running
- tier: FREE (cheap CPU, header reads only)
- hypothesis: `frame_times` is synthesised by the aligner and nothing verifies the extracted mp4
  against it, while `_decode_even_frames` stretches whatever frames exist across the sequence length —
  so a short video yields a clip whose pixels are time-compressed relative to labels claiming the
  nominal schedule.
- method: for every clip, compare `len(meta["frame_times"])` against the mp4's container frame count.
- expected: either a clean corpus (all 32) or a shortfall population concentrated in the low-headroom
  clips.
- kill: all videos hold the claimed frame count — then the residual is tap-calibration jitter and the
  fix is an onset-deviation filter, not a harness repair.
- why: this single number decides which fix EQ-003's duration tail needs, and a shortfall would be a
  label-timing bug affecting the whole corpus, not just duration.

## EQ-019 — Fix the legacy session key, then audit the 2,022-command split
- status: todo
- tier: FREE
- hypothesis: `_segment_key`'s `legacy:<dir>` fallback collapses every legacy sample into one bucket,
  so session counts for the 2,022-command corpus are meaningless and the cross-corpus claim cannot be
  made.
- method: derive the session from the sample path (`relative_to(root).parts[1]`) instead of the meta
  key; re-run the overlap audit on both corpora.
- expected: a real train session count, and a defensible statement about whether 90.10%/93.07% share
  sessions too.
- kill: legacy paths carry no usable session structure either — then say so and stop claiming anything
  about that corpus.
- why: EQ-017's headline is sound but its train session count is wrong, and the cross-corpus sentence
  was withdrawn for lack of evidence.

## EQ-020 — Quantify the real generalisation gap: park, day, device
- status: todo
- tier: FREE to measure
- hypothesis: the evaluated split is one park (`the_workshop`), one four-hour window (2026-08-13), and
  146 XR / 7 XR2 clips — so park/day/device coverage, not session identity, bounds what the held-out
  numbers demonstrate.
- method: tabulate clips and recovery by park, session date and device across the corpus; report the
  XR2 recovery interval given its tiny support.
- expected: a coverage table showing the held-out set exercises one condition, with XR2 severely
  under-supported (already 71.4% on 7 clips).
- kill: the corpus already spans multiple parks/days in the evaluated split — then coverage is fine.
- why: EQ-007's certification protocol should choose its axes deliberately; EQ-017 showed session
  identity is the wrong axis to worry about.
