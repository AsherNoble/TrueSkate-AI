# Experiment queue — Model 1 (behavioural cloning)

> **Read `CONSOLIDATION — what is actually true after EQ-001..EQ-031` (2026-08-20) in the journal
> before this queue.** Many earlier entries were later retracted; that entry is the single source of
> truth and supersedes them.
>
> **Re-prioritisation for the owner (2026-08-20).** EQ-016..EQ-032 went deep into duration diagnostics
> and produced a clean attribution, but **no item since EQ-002 has moved the headline accuracy**, and
> the two highest-leverage items in this queue have never run: **EQ-005** (derive Model 1's target from
> Model 2's tolerance, instead of asserting 99%) and **EQ-004** (resolve or bury the line-fit decoder).
> Both are PAID. EQ-007 certification stays blocked on collection, which is blocked on EQ-024's axis
> decision. Suggest running EQ-005 next rather than continuing the duration line.

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
- cost: MEASURED 2026-08-21 (three short calibration runs, `--no-evaluate-test`, ~$0.20-0.30 total).
  Per-epoch steady state is **29 s on one container and 78 s on another** — `gpu="any"` draws from
  {T4, L4, A10} and the draw is worth 2.7x. Epoch 1 carries a stable ~163 s decode/cache premium,
  paid once per RUN (the frame cache is an in-process dict, so each of the 6 runs re-pays it).
  Per run: 22 min (fast draw) to 55 min (slow draw). Six runs plus `memory=16384` billing
  ($0.128/hr, omitted from the first estimate) and the ~60-90 s test-evaluation tail the
  calibration did not measure: **~$2.5 best case, ~$5.1 realistic worst (L4), well inside $10.**
- **PREREQUISITE, and it is about the science not the budget: pin the GPU.** Six runs on `gpu="any"`
  are six independent hardware draws, and the calibration showed identical-seed epoch-1 validation
  differing across draws (start_med 0.0296 / 0.0262 / 0.0296) — i.e. cuDNN algorithm choice moves
  the number the sweep is trying to compare. Set `MODAL_TRAIN_GPU=L4` (or T4) for every run of the
  sweep; the plumbing was added 2026-08-21 and defaults to `any` for one-off runs. `_device()` now
  prints `device=cuda name=...` so the draw is always attributable.
- **stale premise, corrected:** this item says the line-fit runs used "equal knot weighting instead of
  the baseline's 1.8x start weighting". At K=2 `basic_linear_training.py:173` ALREADY applies
  `1.8*start + end`, so there is nothing to restore and no code change is needed. One of the three
  confounds this item was built on does not exist.
- why: it was the primary bet of the MVP-2 plan and is currently a regression; leaving it
  ambiguous invites re-litigating it every session.

## EQ-005 — Derive the Model-1 fidelity target instead of asserting 99%
- status: blocked — and the 2026-08-20 "tooling gap" finding is RETRACTED. Expert play has no manifests
  (verified: no meta.json under data/extracted_frames), and SLS gestures are random so a sequence model
  learns nothing from them. Structure and labels sit on opposite sides, so the Model-1 gate is REAL.
  See the 2026-08-20 EQ-034 retraction entry.
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
- status: done: CONFIRMED — the config change was necessary but NOT sufficient. A latent ARG_MAX
  bug in `offload_corpus_to_modal.sh:216` made every park above ~13k sample dirs enumerate as
  EMPTY (stderr discarded), so the session retried forever at 0 batches. Fixed with `find`;
  20344 dirs -> 227 batches, uploading. Red team CONFIRMED, no deletion hazard. Capacity
  resolved: corpus-v2 is a Modal **v2-format** volume (trueskate-corpus holds >550k entries,
  above v1's 500k hard cap), so the only cap that applies is 262,144 files PER DIRECTORY and
  the layout peaks at 20,344. See the 2026-08-20 EQ-014 journal entry.
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
- status: blocked on collection only — the axis decision is SETTLED (EQ-024, owner, 2026-08-21)
- tier: HOLDOUT
- blocked-by: collection
- hypothesis: n/a — this is a measurement protocol, predeclared before any data is spent.

### Certified axes (owner decision, 2026-08-21 — ALL FOUR)

The holdout certifies a Model-1 recovery rate that is simultaneously:
  (a) **command-disjoint** — no exact gesture command in the holdout appears in train or validation,
      enforced by `split_with_fresh_command_holdout` and failing closed on any overlap;
  (b) **device-balanced** — both XR1 and XR2 present, each contributing at least 40% of clips, so the
      headline is not an XR1 number with XR2 decoration (the current corpus is 95% one device);
  (c) **park-disjoint** — the holdout contains clips from at least one park that contributed ZERO
      training clips. As of 2026-08-20 XR1 is on SLS 2015 Super Crown and XR2 on SLS 2013 Kansas City,
      so this falls out of collecting from both phones; the park held out must be named BEFORE
      collection ends, not chosen afterwards from whichever park scored best;
  (d) **day-disjoint** — no collection session (`iPhone_XR*_YYYYMMDD_HHMMSS`) contributes clips to both
      the holdout and the training set, and the holdout spans >= 2 calendar days.

Explicitly NOT certified: any device other than iPhone XR/XR2; any park outside the SLS-arena family
plus The Workshop; True Skate versions other than the one collected under; expert human gestures (the
corpus is agent-generated random SLS-mix gestures — see the CONSOLIDATION entry).

- method: >= 3,000 untouched unique commands satisfying (a)-(d) above; headline is the
  Clopper-Pearson 95% lower bound, not the point estimate; staged in tranches of 1,000;
  test evaluated exactly once; no post-hoc tolerance changes. If a tranche fails to satisfy (b)-(d),
  it is discarded and recollected — it is NOT evaluated and then explained.
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
- status: done: kill criterion FIRES — pixel evidence (balacc 0.846) LOSES to a constant [7,19] window
  using no pixels (0.900). `trail_frames_present` is a tautology (per-frame amax), so EQ-015's "trail
  persists" claim is retracted; my replacement claim is retracted too. See the 2026-08-20 EQ-016 entry.
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
- status: done: CONFIRMED — NO. All 3,040 clips hold 31 frames against 32 claimed (decode-verified,
  header agrees). Root cause `align_xctest_traces.py:375` leaves 1/30s tail margin and the frame count
  is never asserted. Pixels sit on a 31/30-stretched timebase while labels do not. Fixing it requires a
  RETRAIN, not a re-eval. See the 2026-08-20 EQ-018 entry.
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
- status: done: CONFIRMED — session key now pattern-matched (a fixed path index gave the PARK on the 2k
  layout; caught before journaling). The 2,022-command corpus is 100% session-shared (233 train / 171
  test sessions, 0 unique to test), restoring EQ-017's withdrawn cross-corpus claim. See the 2026-08-20
  EQ-019 entry.
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
- status: done: CONFIRMED — one park corpus-wide, fresh is 95% XR (969/49), evaluated split is one
  4h window on one day. Overnight gap to legacy confirmed real (15.6h) from capture timestamps.
  Retracted "XR2 scores 71.4%" (Fisher p=0.056 across 5 post-hoc slices = not evidence). See the
  2026-08-20 EQ-020 entry.
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

## EQ-021 — Fix the extractor: real tail margin plus a frame-count assertion
- status: done: PARTIAL — the frame-count assertion is shipped and tested (a short extract is now
  rejected, not stretched). The tail margin is UNVALIDATED: this machine's ffmpeg writes containers
  advertising 32 frames of which only 30 decode, so it cannot reproduce the rig's behaviour. See the
  2026-08-20 EQ-021 entry.
- tier: FREE to change, PAID to revalidate (requires a retrain)
- hypothesis: extracting with `-t dur + 2/fps` (keeping `-frames:v 32`) and asserting the produced
  frame count equals `max_frames` yields clips whose pixels and labels share one timebase.
- method: change `_extract_sample_video`, add the assertion so a short extract fails the sample loudly
  instead of being stretched silently; re-extract a pilot segment and confirm 32 frames and a 2.2667s
  span; only then consider a corpus re-extract.
- expected: 32 frames, span matching `frame_times`, and the assertion catching any regression.
- kill: the source `.mov` files are gone for most of the corpus, so re-extraction is impossible — then
  the fix applies only to future collection and the existing corpus keeps its documented 31/30 skew.
- why: it is a silent label-timing defect affecting every clip, and it is cheap to prevent going
  forward even if the existing corpus cannot be repaired.
- CAUTION: do NOT shorten `frame_times` to 31 as a shortcut — that preserves the truncated content. And
  do not re-evaluate existing checkpoints on corrected clips without retraining; they would
  underestimate duration by ~3.2%.

## EQ-022 — Is EQ-003's duration tail calibration residual or aligner phase jitter?
- status: todo
- tier: FREE where a `.mov` survives
- hypothesis: `-ss` input seek lands on the source frame at or after the requested start, so frame 0
  carries a per-clip offset in [0, 1/30s) that the `fps` filter re-stamps away — up to +/-0.45
  label-frames of phase jitter, independent of tap calibration.
- method: for a handful of segments whose `.mov` still exists, re-run extraction with
  `-copyts` / `-show_entries frame=pts_time` before the `fps` filter and measure frame 0's true offset
  clip-to-clip.
- expected: a spread of order 1/30s, uncorrelated with the tap-calibration offset.
- kill: frame 0's offset is constant across clips — then the jitter is calibration residual after all
  and EQ-003's attribution stands.
- why: EQ-003 attributed the whole low-headroom tail to tap-calibration residual. If a second,
  independent source of per-clip timing error exists, that attribution is wrong and the fix differs.

## EQ-023 — Validate the tail margin on the rig
- status: todo
- tier: FREE (rig time only, no cloud spend)
- blocked-by: a surviving segment `.mov` on the rig
- hypothesis: with `-t duration + 2/source_fps`, the rig's ffmpeg produces exactly 32 decodable frames
  where it previously produced 31.
- method: on the rig, run the patched `_extract_sample_video` against a surviving `.mov` and count
  decodable frames; compare against the same call without the margin. Confirm header and decode agree,
  as they do corpus-wide.
- expected: 31 -> 32 decodable frames, header matching decode.
- kill: the margin does not move the count on the rig either — then the cause is not tail margin, and
  the assertion alone (reject and re-extract with a wider window) is the whole fix.
- why: EQ-021 could not validate this locally because this machine's ffmpeg exhibits a different
  pathology (container advertises frames that do not decode). The fix must not be believed until it is
  measured where it will run.

## EQ-024 — Predeclare which axes EQ-007 certifies
- status: done: RESOLVED (2026-08-21) — owner chose ALL FOUR axes (command-disjoint,
  device-balanced, park-disjoint, day-disjoint). The protocol is written into EQ-007 above,
  including the explicit list of what is NOT certified. Nothing more to run.
- tier: FREE (a decision + a written protocol)
- blocked-by: owner
- hypothesis: n/a — this is a design decision that must be made before collection, not after.
- method: choose explicitly, and write into the EQ-007 protocol, which of these the >=3,000-command
  holdout is meant to certify: (a) unseen COMMANDS only, same park/device/day — cheapest, and what the
  current corpus already supports; (b) unseen commands + device parity (needs ~10x more fresh XR2);
  (c) unseen commands + a second park (needs collection in a park never trained on); (d) session- or
  day-disjoint. Each costs collection time, and none can be added after the fact.
- expected: a one-paragraph protocol naming the certified axes and explicitly listing what is NOT
  certified.
- kill: n/a.
- plain-language version (asked for 2026-08-20): you will spend collection time building a test
  set, look at it exactly ONCE, and announce a number. This decides what that number is allowed
  to claim. Today Model 1 trains and tests on one park, one day, one phone — so a good score
  means only "it handles swipes it hasn't seen", not "a new park", "the other phone", or
  "a different day". Those axes cannot be added after collection.
- RECOMMENDATION (mine, pending owner sign-off): take all four. As of 2026-08-20 XR1 is on SLS
  2015 Super Crown and XR2 on SLS 2013 Kansas City, so the second-park and device-parity axes
  now cost almost nothing — they fall out of collecting from both phones as they currently sit.
  Day-disjointness costs only a calendar gap. Declining them is the expensive choice.
- why: EQ-020 showed the current holdout certifies (a) while its name ("fresh holdout") suggests more.
  Inheriting that silently into a 3,000-clip certification would make the headline number sound
  broader than it is.

## EQ-025 — Measure the trailing edge, the only part that can carry duration
- status: done: CONFOUNDED then resolved — split by source falsified the red team's legacy-contamination
  hypothesis: the bad mode is in FRESH (MAE 3.18 vs legacy 2.29). On fresh, evidence LOSES to the
  pixel-free constant on MAE (3.18 vs 2.82) while winning on gate (48% vs 27%). The model is ~12x
  better on the comparable population. See the 2026-08-20 EQ-025 entry.
- tier: PAID (cheap CPU)
- hypothesis: the contact interval's leading edge is constant by construction (frame 7 for every clip),
  so any duration signal lives entirely in the trailing edge; a detector scored only on that edge, and
  only over frames >= 7, is the honest test of whether pixels beat the nominal schedule.
- method: for each clip, estimate the liftoff frame from trail evidence and compare against the
  commanded liftoff index; report the distribution of |estimated - commanded| in frames, beside a
  baseline that predicts the MEAN liftoff index (no pixels) and one that predicts from commanded
  duration (an oracle upper bound).
- expected: a per-clip edge error in frames, directly comparable to the 0.073s quantum and the 0.10s
  gate.
- kill: the evidence-based edge is no better than predicting the mean liftoff index — then pixels carry
  no duration information at this resolution and duration is capped by the trail's rendering, not by
  the model.
- why: EQ-016 showed aggregate frame-classification is dominated by a structural constant. The edge is
  the only place duration can come from, and it has never been measured directly.

## EQ-026 — Growth-based liftoff edge, the only untested member of the family
- status: done: kill criterion does NOT fire — RETRACTED my "family exhausted" claim. The midpoint of
  growth and fade is near-unbiased and beats the pixel-free constant (MAE 2.821 vs 3.041, paired sign
  p=0.035); last_increase also wins (p=0.0032) and doubles the gate fraction. All remain ~11x worse
  than the model. Red team's floor hypothesis falsified (0.34% pinned). See the 2026-08-20 entry.
- tier: PAID (cheap CPU)
- hypothesis: `peak` is a spatial MAX, so it tracks the newest bright trail segment and decays —
  "last frame above threshold" is a fade timer, not a contact-end detector. An estimator based on trail
  GROWTH (the last frame at which new trail pixels appear / the knee of spatial extent) is better
  specified and may beat both the fade estimator and the constant baseline.
- method: fresh-source clips only; estimate liftoff as the last frame where the count of above-threshold
  pixels increases materially; PRE-COMMIT the threshold before looking at the result; report a PAIRED
  test against the constant baseline, and the gate fraction alongside MAE.
- expected: MAE below the constant's 2.816 frames on fresh, and a lighter tail than the fade
  estimator's p90 of 9.0.
- kill: growth-based edge is no better than fade-based on fresh — then presence/geometry edge decoding
  is exhausted as a route and duration is limited by something the trail does not encode at 13.7fps.
- why: EQ-025 ruled out the fade estimator but the family was not exhausted, and the practical claim
  ("presence-edge decoding is not a route to duration") is the one that would cost money if wrong.

## EQ-027 — A difference-based duration reader
- status: done: INVALID for its hypothesis — the leading edge is constant by construction and is erased
  by the reference subtraction, so no cancellation was available and a "difference" is an absolute index
  up to a fixed affine map. Differencing HURT (r 0.451 single-edge vs 0.415 differenced). The ~0.19s
  plateau vs the model's 0.0189s survives and the 10x is now a LOWER bound. See the 2026-08-20 entry.
- tier: PAID (cheap CPU)
- hypothesis: every estimator so far reads an ABSOLUTE frame index against the label grid, so each pays
  the EQ-018 timebase skew and the `-ss` phase jitter in full. A duration is a DIFFERENCE of two edges,
  which cancels any clip-constant offset — the one structural advantage the trained model has that none
  of these estimators were given.
- method: estimate duration directly as (fade edge - growth edge) x quantum, and as (last_increase -
  first_increase); compare against commanded DURATION rather than against an absolute edge index; paired
  sign test against a constant-duration predictor (the corpus mean duration).
- expected: a materially lower error than the absolute-index estimators, since the shared offset drops
  out.
- kill: the difference reader is no better than the absolute ones — then the per-clip noise is not a
  shared offset and the timebase caveat, while real, is not what limits these estimators.
- why: EQ-026 left this as the single structural asymmetry between the estimators and the model, and it
  is the last cheap thing that could change the picture before duration work moves to the model itself.

## EQ-028 — Track the trail HEAD, not a per-frame scalar
- status: done: kill does NOT fire — head tracking is the best reader (r 0.582 vs 0.451, MAE 0.163s vs
  0.189s). But RETRACTED the headline: the model's `duration_head` takes only a 2xT scalar series
  (spatial max + mean), so it sees NO position and the gap is not about geometry. Much of the ratio is
  the 0.021s integer-frame quantisation floor. See the 2026-08-20 EQ-028 entry.
- tier: PAID (cheap CPU)
- hypothesis: every reader so far collapses each frame to a scalar (pixel count or max brightness) and
  uses none of the trail's POSITION. Projecting above-threshold pixels onto the commanded start->end
  axis and taking the max projection per frame gives the trail HEAD, whose advance stops at contact
  end — a direct kinematic read, immune to fade.
- method: per frame, project above-threshold pixel coordinates onto the unit chord; take the max
  projection; estimate liftoff as the frame after which the projection stops advancing (last frame with
  a material increase). Fresh-only, threshold pre-committed at 0.35, paired sign test against both the
  constant and against EQ-026's midpoint, reported as effect size (r, R^2) not p-value.
- expected: r with duration materially above 0.451 (the best single scalar reader) if position carries
  what the scalars miss.
- kill: head-projection is no better than the scalar readers — then the gap to the model is not about
  trail geometry at all, and duration work moves to the model/architecture with the trail line closed
  for real.
- why: this is the one structurally different family left, and the model demonstrably reads geometry
  spatially (endpoints to 0.006 normalised units) while every reader tried so far throws position away.

## EQ-029 — Is the gap the evidence map or the decoder?
- status: done: CONFOUNDED on the clean split (three variables bundled) but the DECISIVE result stands:
  decoder ~2.6x [2.1,3.2], front end ~3.3x, and per-clip errors are UNCORRELATED with disjoint failure
  sets (0 both / 27 handcrafted-only / 2 model-only), so the residual is a genuine front-end advantage
  and IS addressable — not the EQ-025 data defect. 153 distinct commands. See the 2026-08-20 entry.
- tier: PAID (one short training run)
- hypothesis: the model's duration advantage comes from either (a) a learned evidence map beating a
  hand-crafted colour x motion filter, or (b) a learned temporal decoder over the whole series beating
  a single hand-picked event. EQ-028 separates neither.
- method: feed the hand-crafted `trail_evidence` scalar series (per-frame spatial max and mean, the same
  2xT shape `duration_head` consumes) into a freshly-initialised copy of `duration_head`, trained on the
  same split with the same schedule. Compare against 0.163s (hand-crafted front end + hand-picked event)
  and 0.0189s (learned front end + learned decoder).
- expected: a number between the two that attributes the gap.
- kill: n/a — this is an attribution experiment; every outcome is informative.
- why: EQ-028 proved the model's duration path is the SAME reader family as the hand-crafted ones, so
  the remaining question is which half of the pipeline carries the advantage. It is the last cheap thing
  that changes what duration work should target.

## EQ-030 — Fix the head estimator, then go sub-frame
- status: todo
- tier: PAID (cheap CPU)
- hypothesis: (a) the per-frame max should be a cumulative max — the code contradicts its own comment and
  one flicker latches the estimate to the clip end; (b) clipping the head to [0,1] removes post-liftoff
  board-motion contamination (p90 reach 1.22); (c) interpolating the head-advance profile instead of
  emitting an integer index is the ONLY way any reader beats the 0.021s quantisation floor.
- method: re-run EQ-028 with `head.cummax(dim=0)` and a [0,1] clip as a bug-fix arm; then add a sub-frame
  arm that fits the advance profile and reads liftoff by interpolation.
- expected: the bug fix lowers p90 materially; the sub-frame arm is the only one that can approach the
  model's 0.024s residual sd.
- kill: sub-frame interpolation does not beat the integer read — then the reader's noise is not
  quantisation-limited and the floor argument does not apply to it.
- why: EQ-028's estimator has a known bug and a known contamination path, and the quantisation floor means
  no integer estimator can ever be compared fairly to the model.

## EQ-031 — Decompose the three variables EQ-029 bundled
- status: done: RETRACTED "temporal shape dominates" — it is CAPACITY. linear_64 (full series) equals
  linear_6, and conv_32 equals mlp_64, so temporal information and conv structure each buy ~nothing;
  nonlinearity over 6 scalars buys 1.65x and lands within 16% of the decoder. See the 2026-08-20 entry.
- tier: PAID (cheap CPU)
- hypothesis: EQ-029's "decoder 2.6x" bundles decoder architecture, fitting budget and per-clip
  normalisation; the front-end 3.3x bundles {learned filter, temporal mixer, duration-supervised
  training}. Each is separable with the data already extracted.
- method: (a) ridge over ~5 hand-picked scalars of the same normalised series (last_rising, head edge,
  extent max, peak max, contact-fraction) on the same 2,734 train / 153 test split — isolates
  MULTIVARIATE reading from SHAPE reading; (b) re-run `extract()` with no normalisation and with a
  corpus-global scale — isolates the normalisation knob; (c) note that separating temporal_mixer from
  the learned filter needs a retrain with `temporal_mixer=False` and is the one genuinely expensive arm.
- expected: (a) lands between 0.163 and 0.0629 and says how much is shape; (b) moves the 0.0629 by less
  than the 2.6x margin, or does not.
- kill: n/a — attribution, every outcome informative.
- why: EQ-029 showed the residual is addressable, so knowing WHICH half to attack is now the binding
  question for duration work.

## EQ-032 — Does capacity on the LEARNED evidence series close the remaining 3.3x?
- status: done: PARTIAL — a fresh head on the FROZEN learned series reaches 0.01703s (vs 0.06289s from
  the hand-crafted series), so the map's temporal envelope already contains the duration signal. But
  RETRACTED "exceeds the model" (checkpoint selection is lexicographic, duration only a tie-breaker)
  and "joint training adds nothing" (map_weight=0, so the map was BUILT BY the duration loss). See the
  2026-08-20 EQ-032 entry.
- tier: PAID (cheap CPU)
- hypothesis: EQ-031 showed capacity over the evidence series is the dominant decoder factor. The
  remaining 3.3x sits in the front end (learned map + temporal mixer + duration-supervised training).
  Extracting the MODEL's own `max(start_scores, end_scores)` series and feeding it to the same mlp_6 /
  mlp_64 arms says how much of that 3.3x is the map itself versus end-to-end training.
- method: run the checkpoint to dump `evidence.amax((2,3))` and `evidence.mean((2,3))` per clip, then
  train the same arms on it; compare against 0.0629 (hand-crafted series) and 0.0189 (the model).
- expected: a number that splits the front-end factor into "better series" versus "trained jointly".
- kill: the learned series with a fresh head matches 0.0189 — then the front end is entirely the map and
  joint training adds nothing.
- why: it is the last cheap decomposition; after it, duration work is an architecture decision rather
  than an attribution question.

## EQ-033 — Is the map duration-legible WITHOUT duration supervision?
- status: todo
- tier: PAID (one full retrain)
- hypothesis: EQ-032 froze a map that was itself built by the duration loss (`endpoint_map_weight=0`,
  `trajectory_map_weight=0`, so duration is one of only two gradient sources shaping the score maps).
  Detaching the duration path from the maps says whether the map is duration-legible on its own.
- method: retrain with `series` built from `evidence.detach()` (`basic_linear_regressor.py:260`) so no
  duration gradient reaches the encoder; freeze; fit the same fresh conv head; compare against 0.01703
  (duration-supervised map) and 0.06289 (hand-crafted series).
- expected: a number that decides whether "the front end is the map" or "the front end is duration
  supervision shaping the map".
- kill: n/a — attribution; both outcomes are informative.
- why: EQ-032's headline rests on a map that duration supervision produced, so the attribution is
  circular until this runs. It is the last question in the duration line and the only one needing a
  retrain.

## EQ-034 — Ground-truth mode for build_bc_clips: a Model 2 corpus without Model 1
- status: done: RETRACTED as a route to EQ-005. Still worth building as a PLUMBING/CAPACITY check only
  (does Model 2's architecture fit gesture sequences and does the training path run on real frames?),
  never as a tolerance study — SLS gestures are random, so an epsilon sweep on them measures nothing.
- tier: FREE to build, PAID to train
- hypothesis: `clip.json` can be written directly from the command manifests, giving a Model 2 training
  corpus with PERFECT gesture labels and no dependence on Model 1. The pieces already exist:
  `_schedule_from_meta` (`temporal_trace_dataset.py:562`) yields GT drag waypoints — the same GT the
  2026-07-19 stroke-recovery study used — and `assemble_strokes` (`bc/assemble.py:88`) is the shared
  assembler.
- method: add a `--ground-truth` mode to `build_bc_clips.py` that bypasses the Model 1 forward pass and
  emits `clip.json` from the manifest; round-trip it through `SequenceDataset` exactly as `--smoke`
  does; then train Model 2 on it.
- expected: a Model 2 that trains on real frames with a reported metric — the first such baseline.
- kill: Model 2 fails to learn even with perfect gestures — then the sequence architecture is the
  problem and Model 1's fidelity is irrelevant to it, which is itself decisive.
- why: this is the gate on EQ-005, and EQ-005 is the item that turns the project's "99% on Model 1"
  target from an assertion into a measured requirement. It also decouples the two models: Model 2 can
  be developed in parallel with Model 1 rather than behind it.

## EQ-035 — The epsilon sweep: derive Model 1's actual requirement
- status: blocked — needs a STRUCTURED gesture corpus with labels, which does not exist. See EQ-036.
- tier: PAID
- blocked-by: EQ-036
- hypothesis: Model 2 tolerates gesture-parameter error up to some epsilon; that epsilon, converted into
  a per-knot recovery rate, IS Model 1's requirement.
- method: with the GT corpus from EQ-034, perturb gesture parameters by controlled epsilon (endpoint
  noise and duration noise, swept independently since EQ-003 showed they behave differently), retrain
  Model 2 at each level, and find where performance departs from the epsilon=0 baseline.
- expected: a curve giving the tolerated epsilon, converted to a per-knot requirement via the MVP-3
  joint-gate arithmetic.
- kill: performance is flat in epsilon across the whole plausible range — then Model 2 is insensitive to
  gesture fidelity and the 99% target is simply wrong.
- why: "99%" has been an assertion since 2026-07-19. This measures it. If the tolerated epsilon is loose,
  the current 94.12% may already suffice and the milestone is closer than assumed.

## EQ-036 — Two gates to Model 2, and the project tracks only one
- status: done: RESOLVED (2026-08-20) — the gate was mostly already closed and nobody knew.
  (i) The SLS arenas were INSTALLED on 2026-06-14 (`vision_sequence_leap_journal.md:75`), so
  option (a) was executed two months ago. (ii) The 295 GiB stranded session
  `iPhone_XR_20260814_042825` is entirely in `sls_2015_super_crown` — 20,344 domain-matched
  samples that existed all along, stuck behind the EQ-014 offload bug. EQ-014 and EQ-036 were
  the SAME blocker. (iii) Owner has since put XR1 on SLS 2015 Super Crown and XR2 on SLS 2013
  Kansas City, giving option (a) + option (c) simultaneously. Residual: no collector is
  running (see EQ-044), so the park change records nothing yet.
- tier: FREE to decide, collection to execute
- finding: Model 1 must be accurate enough AND must transfer to the parks the expert corpus uses. EQ-020
  measured the MVP corpus as ONE park (`the_workshop`, 3,040/3,040 clips); the journal (2026-06) records
  the expert clips as SLS-arena parks, expert transfer as a DOMAIN-GAP problem rather than a
  data-quantity one, and those parks as NOT INSTALLED (store/download only).
- options: (a) install/download an SLS-arena park and collect there, making Model 1's training domain
  match the expert corpus; (b) re-record the expert corpus in an installed park (The Workshop, Glass
  House, Underpass) so the domains match from the other side — cheapest if Asher is willing to replay;
  (c) multi-park collection for domain robustness, which was started in 2026-06 and never finished.
- why: a 99% single-park Model 1 does not imply a usable labeller for expert play, so the current target
  can be met in full and still not unblock Model 2.
- **CORRECTION (owner, 2026-08-21) — option (b) was never on the table and should not have been listed.**
  Model 2 v1 trains on expert gameplay from ONE park, **SLS 2015 Super Crown**, on purpose: the expert
  corpus is only ~3 hours, and holding the park fixed spends that data on sequence structure instead of
  park-invariance. So there is nothing to re-record. What Model 1 must do is transfer to SLS 2015 Super
  Crown specifically — which the 20,344-sample `iPhone_XR_20260814_042825` session (now on
  `trueskate-corpus-v2`) is exactly the data for. Multi-park collection serves Model 1 robustness and
  the EQ-007 park-disjointness axis (EQ-044), NOT Model 2.

## EQ-037 — Model 2 plumbing check
- status: done: CONFIRMED — `train_sequence_model.py --smoke` runs (3.62M params, loss 1.41 -> 0.032).
  Architecture and training loop are sound on synthetic data; real-frame training remains gated on
  labels. NOTE: the run overwrote `notebooks/models/sequence_model.pth` (gitignored, unrecoverable);
  almost certainly a prior smoke artefact. See the 2026-08-20 EQ-037 entry.

## EQ-038 — A diagnostic must not be able to clobber a model artefact
- status: done: CONFIRMED — smoke now resolves to tmp/, explicit --out still wins, test added. Sweep
  found train_scene_classifier has no smoke mode and train_trace_extractor was already safe (I patched
  it anyway, introduced a bug, reverted). See the 2026-08-20 EQ-038 entry.
- tier: FREE
- hypothesis: `train_sequence_model.py --smoke` defaults `--out` to
  `notebooks/models/sequence_model.pth`, so a throwaway diagnostic overwrites a production path. The
  same pattern may exist in other scripts.
- method: default the smoke path to `tmp/`; sweep the other entry points for diagnostics that write to
  `notebooks/models/` or another durable location by default; add a test asserting the smoke default is
  under `tmp/`.
- expected: one-line fix here, plus whatever the sweep turns up.
- kill: n/a — defect fix.
- why: it already cost one checkpoint this session. Model artefacts are gitignored, so a clobber is
  unrecoverable and silent.

## EQ-039 — `--epochs` is ignored by the Model 2 smoke path
- status: done: CONFIRMED — two independent caps removed, flag verified for 1/2/4 epochs, real-run
  default preserved (I briefly changed it and reverted). Fourth name-vs-behaviour mismatch in this
  queue. See the 2026-08-20 EQ-039 entry.
- tier: FREE
- hypothesis: `train_sequence_model.py:112` stops on `smoke and ep >= 2`, so `--epochs 1` still runs
  three epochs and the flag misreports what happened.
- method: honour `--epochs` under `--smoke` (or reject the combination explicitly), and assert it.
- expected: `--epochs N` runs N epochs in both modes.
- kill: n/a — cosmetic defect.
- why: harmless today, but a flag that silently does something other than what it says is the same
  class of problem as `trail_frames_present` and the synthesised `frame_times` — three of which have
  already produced retracted conclusions in this queue.

## EQ-040 — Audit the codebase for name-vs-behaviour mismatches
- status: done: CONFIRMED — two more found. `gesture_start_monotonic` holds EPOCH seconds, and its mere
  PRESENCE is a mode switch selecting the start-relative label branch. Six total, four of which cost
  retractions. See the 2026-08-20 EQ-040 entry.
- tier: FREE
- hypothesis: four fields/flags in this pipeline have been found to not do what their names say —
  `trail_frames_present` (constant by construction), `trail_frame_start` (an argmin, not an onset),
  `frame_times` (synthesised, not measured), `--epochs` (ignored under --smoke). Three produced
  retracted conclusions. There are likely more, and they are cheap to find but expensive to discover
  by being misled.
- method: enumerate the fields emitted by the diagnostic/audit paths and the flags on the entry points;
  for each, state what the name implies and what the code does; flag every divergence. Prioritise
  anything a conclusion has been or could be drawn from (timing fields especially).
- expected: a short table of divergences, each either renamed or documented at the definition site.
- kill: no further divergences found — then the four were the whole of it, which is itself worth
  knowing.
- why: this queue's dominant failure mode has not been bad experiments, it has been trusting a name.
  The cost of one such mistake (EQ-015's circular audit, EQ-016's two retractions) far exceeds the cost
  of the sweep.

## EQ-041 — State the label convention instead of inferring it from key presence
- status: todo (owner sign-off — touches every corpus on disk)
- tier: FREE to implement, but it is a DATA-FORMAT change
- hypothesis: `_is_end_relative()` decides whether touches are placed start- or end-relative purely from
  WHICH KEYS EXIST in a meta dict (`temporal_trace_dataset.py:586-593`). A corpus written by a different
  aligner version silently gets a different label convention, with no version field and no error.
- method: (a) write an explicit `label_time_base: "start" | "end"` into new metas and read it first,
  keeping `_is_end_relative` as a legacy fallback; (b) rename `gesture_start_monotonic` ->
  `gesture_start_epoch_s` (it holds `t_call_start_epoch_s`, not a monotonic clock), still reading the
  old key for existing corpora; (c) assert the convention at dataset load so a mismatch fails loudly.
- expected: no behaviour change on existing corpora; new ones self-describe.
- kill: n/a — defect fix.
- why: this is the same shape as EQ-018 (a schedule asserted rather than checked) and is the mismatch on
  the register most likely to produce a future wrong number. It is also the riskiest to change blind,
  which is why it wants sign-off rather than a loop tick.

## EQ-042 — Spot-check sample integrity before the offloader deletes 295 GiB
- status: todo
- tier: FREE (read-only, on the rig)
- blocked-by: none, but must run BEFORE the 227th batch completes
- hypothesis: every uploaded sample dir is byte-complete, so the `REMOTE == LOCAL` guard — which
  counts `meta.json` ONLY and checks nothing about the 32 frames beside it — is a sufficient
  precondition for the irreversible local delete.
- method: sample ~50 `sample_*` dirs spread across the batch index range; for each, compare the local
  file count against `modal volume ls trueskate-corpus-v2 /<sess>/<park>/<sample>`. Two dirs were
  already checked by the red team (33 local vs 33 remote, both clean).
- expected: 50/50 exact file-count matches.
- kill: any mismatch — then the guard is insufficient and the delete must be blocked (bootout the job)
  until the offloader verifies frame counts, not just `meta.json` presence.
- why: the red team raised this as the one integrity gap it could not close: 2 of 20,344 certifies
  essentially nothing, and the delete is irreversible post-anchor-fix corpus.

## EQ-043 — Kill the 9-day Modal retry zombie (owner-executed)
- status: todo (owner — my `kill` was blocked by the sandbox classifier)
- tier: FREE (stops spend)
- finding: `modal run scripts/cloud/train_basic_hold_modal.py --run-label hold_1008_baseline_20260811`
  (rig PID 57026, started 2026-08-11 06:31, app `ap-kLhkNVbdx6NuxjBghvIGz2`, still `ephemeral`) has
  retried a DETERMINISTIC import-time crash — `IndexError` on `Path(__file__).resolve().parents[2]`
  — **1,298 times** over 9 days. It cannot succeed.
- method: `kill 57026` on the rig. Then fix `_ROOT = Path(__file__).resolve().parents[2]` in the
  remote module (the container's `__file__` is shallower than the repo's) before any relaunch.
- why: it is charging a $10 budget for an outcome that is impossible, and it obscures real runs in
  `modal app list`.

## EQ-044 — Collection is stopped on both phones
- status: todo (owner)
- tier: FREE to restart; collection time to execute
- finding: no `com.trueskate.collect.*` job is loaded. `logs/collect_xr1.log` ends 2026-08-17 with
  `WDA is not responding at 127.0.0.1:8100`; `logs/collect_xr2.log` ends 2026-08-06. The watchdogs
  log `fleet state checked` every 2 min and do NOT flag that collection is down — because the
  collectors are intentionally unloaded, the watchdog has nothing to complain about.
- method: let the 227-batch offload finish first (it frees ~295 GiB and owns the uplink), then
  `launchctl kickstart` both collectors. XR1 will need `scripts/launch_services.py` for WDA.
- **constraint added by EQ-024 (2026-08-21):** this collection now feeds the EQ-007 holdout, which
  must be device-balanced, park-disjoint and day-disjoint. So: keep BOTH phones collecting (not
  one), keep them on DIFFERENT parks, name the held-out park before collection ends, and let the
  run span at least two calendar days. Collecting from one phone in one park for one day would
  silently reduce the certification back to axis (a).
- **scope, set by the owner (2026-08-21): ~FOUR parks is sufficient.** Not an open-ended multi-park
  programme. `--per-park-hours` rotation across four parks on two phones satisfies park-disjointness
  with one park held out, and the day-disjointness axis falls out of the rotation taking >1 day.
- **this is a MODEL 1 requirement and has nothing to do with Model 2.** Model 2 v1 trains on expert
  gameplay from a SINGLE park (SLS 2015 Super Crown) by deliberate choice — see the note on EQ-036.
  Do not let "multi-park" leak across the two models.
- open question: should the watchdog distinguish "collectors deliberately unloaded" from "collectors
  should be running and are not"? As built it is silent in both cases — the exact blindness that
  `unattended-jobs-self-heal-not-blind` warns about.
- why: XR1/XR2 are now on SLS 2015 Super Crown / SLS 2013 Kansas City (EQ-036), which records nothing
  while no collector runs.

## EQ-045 — The rig's copy of the ARG_MAX fix is uncommitted and revertible
- status: todo (owner)
- tier: FREE
- finding: the rig is on `feature/dashboard-sls-preview` at `463316d` with
  `scripts/ops/offload_corpus_to_modal.sh` modified in the working tree (backup at
  `.bak.20260820`). Any `git checkout`/`git pull --rebase` there silently restores the ARG_MAX glob
  and reproduces the exact silent-zero-batches failure. The same fix IS committed on
  `feature/behavourial-cloning` locally.
- method: owner's call — commit it on the rig's branch, or merge/cherry-pick the local commit.
- why: this is a bug whose whole signature is that it looks like success.
