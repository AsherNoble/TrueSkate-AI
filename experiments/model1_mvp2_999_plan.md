# Plan — MVP 2 (linear-drag Model 1) to 99.9%

Written 2026-08-18. Branch `feature/behavourial-cloning`. Target: 99.9% strict joint
recovery (start <=0.03, end <=0.03, duration <=0.10s) on an untouched held-out slice.

Current valid leader: **94.12%** (fresh-holdout pooled temporal mixer, 144/153) and
**93.07%** (fixed-split 4-model ensemble, 282/303). Gate to date was 95%; 99.9% is a
different problem, not a harder version of the same one.

## 0. The controlling result — this is a tail problem, not a precision problem

Fit a Rayleigh to the observed endpoint-error medians and ask what recovery that bulk
distribution alone would produce at the 0.03 tolerance:

| run | endpoint | median err | implied sigma | Rayleigh P(err>0.03) | predicted recovery | observed |
|---|---|---|---|---|---|---|
| fresh 94.12% | end | 0.00921 | 0.00782 | 0.064% | 99.94% | 95.42% |
| fresh 94.12% | start | 0.00635 | 0.00539 | ~0% | ~100% | 100.0% |
| ens 93.07% | end | 0.00998 | 0.00848 | 0.19% | 99.81% | 95.71% |
| ens 93.07% | start | 0.00627 | 0.00533 | ~0% | ~100% | 98.35% |

**The bulk of the error distribution is already at 99.9%.** The observed failure rate is
20–70x fatter than its own core. Test P90 is 0.01875 — well inside a 0.03 tolerance —
yet ~5% of clips miss. That shape means a *discrete outlier population*, not incremental
imprecision.

Direct consequences, and they invert the current roadmap:

- Shrinking median error further buys ~nothing. More capacity, more epochs, more
  ensemble members, finer time priors — all of that moves the core, which is not the
  binding constraint. The 90.1 -> 93.07 ensemble gain was already the cheap part.
- The whole 99.9% question reduces to: **what are the ~5% of clips that fail, and is
  their cause in the model, the labels, or the pixels?** That is an autopsy, and it is
  answerable offline, today, with checkpoints and corpora that already exist.
- 99.9% is only reachable if the tail cause is removable. If it turns out ~1% of clips
  have labels the pixels cannot support, no architecture reaches 99.9% at this tolerance
  and the target must be redefined (see §5).

Supporting evidence that the tail is structured, not random: recovery splits by geometry
(low slope 93.1% / mid 86.9% / high 86.4% on the 2k benchmark) and the failing component
is almost always **end**, never start. A random precision limit would not respect slope
bands or prefer one endpoint.

## 1. The second controlling result — 99.9% is currently unmeasurable

Rule of three: zero failures in N gives a 95% lower bound of 1 - 3/N.

| N test clips | 95% LB with 0 failures |
|---|---|
| 303 (current) | 99.01% |
| 1,000 | 99.70% |
| 3,000 | 99.900% |
| 10,000 | 99.970% |

Current fresh test slices are 153–303 unique commands. **A perfect score on those cannot
distinguish 99% from 99.99%.** Certifying 99.9% needs >=3,000 untouched held-out unique
commands with zero failures, or ~10,000 to tolerate a few. That test set alone is larger
than the entire corpus collected to date (2,022 legacy + ~1,000 fresh + 412 partial
balanced holdout).

So the goal decomposes into two independent, roughly equal problems: **kill the tail**
(§2–3) and **build a holdout big enough to prove it** (§4).

## 2. Phase 1 — Tail autopsy (offline, no rig, do this first)

Nothing else should be built until the failure taxonomy exists. Cost: hours.

1. Extend `basic_linear_recovery_records` (`vision/basic_linear_training.py`) to join each
   record to provenance: sample path, device, dx, slope, duration, both endpoints'
   commanded coords, predicted coords. It already accepts `sample_index` for exactly this.
2. Re-run the two leader checkpoints over their held-out slices, dump every clip's record,
   and pull the ~9 (fresh) and ~21 (fixed-split) failures.
3. Render each failure: frames with commanded endpoints, predicted endpoints, and the
   model's start/end score maps overlaid (`forward_with_scores` already returns them).
4. Classify each failure into a bucket, with counts:
   - **occlusion** — endpoint sits under the hub bar / board / a HUD element
   - **no-render** — trail absent or truncated (dead zone, UI element, Bolt Challenges rect)
   - **timing** — the endpoint's frame is not among the 32 sampled frames
   - **attention collapse** — score map peaks mid-trace or on the wrong endpoint
   - **label-pixel disagreement** — trail is clean and complete but ends >0.03 from the
     commanded point (this is the fatal bucket; see Phase 2)
   - **quantisation** — peak is right, soft-argmax read is off by ~1 map cell

**Gate:** a written taxonomy with counts. The bucket that dominates dictates Phase 3;
do not start Phase 3 before this.

Prior on the answer (to be confirmed or killed, not assumed): the end-only, slope-dependent
signature points at attention collapse plus label-pixel disagreement, not occlusion.

## 3. Phase 2 — Measure the label-noise floor (offline, decisive)

The labels come from the *command manifest*, not from pixels. If Appium's ActionChains
drag under- or overshoots, or liftoff renders short, the commanded `(x1,y1)` is simply
not where the trail ends — and that error is irreducible for any model.

- Reuse the frame-difference-at-commanded-point primitive and
  `scripts/inspect/diagnose_linear_orange_trace.py`'s component tracker. For a few thousand
  strict clips, extract the rendered trail's actual extremes and report the full
  distribution of |rendered - commanded| for start and end separately.
- Report P50 / P90 / P99 / P99.9 and the fraction exceeding 0.03.
- Note the confound honestly: the colour extractor is itself imperfect (it matched only
  13/150 clips in the earlier audit at strict thresholds). Loosen it for *measurement*
  and hand-verify a sample; the earlier audit's low match rate was a detector limitation,
  so a low match rate here is not evidence of clean labels.

**Gate — this is the go/no-go for the whole 99.9% target:**
- fraction >0.03 is **< 0.05%** -> labels support 99.9%; proceed to Phase 3 unchanged.
- fraction is **0.05–1%** -> 99.9% needs pixel-derived label correction: refit each clip's
  label to the observed trail (keeping the command as prior), or tighten the collection
  contract to exclude the offending geometries.
- fraction is **> 1%** -> 99.9% at 0.03 with command labels is impossible. Go to §5.

## 4. Phase 3 — Model changes that attack a tail (not a median)

Evaluate every one of these on the *frozen* 2,022-command benchmark split first (cheap,
directly comparable to 90.1% / 93.07%). Promote only winners to fresh holdout.

**4a. Robust constant-velocity line fit — the primary bet.**
The command is analytically `x(t) = x0 + (x1-x0)*t/T`. The model currently reads two
independent soft-argmax endpoints, so each endpoint rests on essentially one moment of
evidence — maximally exposed to a single bad frame. Instead: regress a per-frame contact
position across all ~30 frames, then solve a **closed-form weighted least-squares fit** of
the constant-velocity line and *read the endpoints off the fitted line*.
- Averaging ~30 measurements cuts endpoint variance ~5x, but the real gain is that a
  single occluded/missed frame can no longer move the answer.
- Wrap it in IRLS / Huber weights on per-frame residuals so outlier frames self-demote.
  This is the mechanism that converts a fat tail into a Gaussian one.
- **This is not the failed trajectory-map control.** That run (`trajectory_weight=0.005`,
  70.6% and regressing) decoded endpoints with time-softmax windows over the path map and
  fused through a cold `sigmoid(-4)` gate — it never fit a line and never had a robust
  loss. Reuse its per-frame supervision (`trajectory_xy` / `trajectory_mask` are already
  emitted by `basic_linear_dataset`), discard its decoder, and drop the fusion gate in
  favour of training the line-fit head as the primary path.

**4b. Resolution / two-stage refinement — the quantisation bucket.**
Input is 128x288 from a 512x1104 raster; one stride-2 leaves a 64x144 score map, so one
x-cell is 0.0156 — over half the entire 0.03 tolerance. A thin rendered line is likely
sub-pixel at 128 wide and aliased by `INTER_AREA`.
- Ablation A: input width 128 -> 256 (4x compute), map cell 0.0078.
- Ablation B: coarse-to-fine cascade — take the coarse endpoint, crop a window from the
  *source raster at native resolution*, refine there. Standard sub-pixel localisation
  cascade, and it costs far less than a global resolution bump.
- `vision/basic_linear_refinement.py` already does a post-hoc local colour nudge; treat it
  as the hand-coded ancestor of 4b and measure whether the learned crop head beats it.

**4c. Abstain / disagreement signal — build it regardless.**
Ensemble member disagreement and score-map peak sharpness are cheap per-clip confidence
proxies. Even if 99.9% raw is reached, downstream needs to know which predictions to
trust. Validate that flagged clips are enriched for failures; do not use it to filter the
test set.

**Explicitly deprioritised** (they move the median, which is not the constraint): more
ensemble members, longer training, larger base_channels, further time-prior grid sweeps.

## 5. Phase 4 — A protocol that can certify 99.9% (predeclare before collecting)

- **Test size:** >=3,000 untouched unique commands, device-balanced across XR1/XR2.
  10,000 if the Phase 2/3 results suggest the true rate will land near 99.9% rather than
  comfortably above.
- **Reporting:** headline number is the **Clopper-Pearson 95% lower bound**, not the point
  estimate. Pass = LB >= 99.9%. This kills the "94.12% is basically 95%" ambiguity that has
  recurred through this MVP.
- **One-shot discipline, unchanged:** validation-only epoch/ensemble selection, test
  evaluated exactly once, no post-hoc tuning, exact-command disjointness enforced fail-closed.
- **Staged spend:** evaluate on a 1,000-clip tranche first. If it fails, do not burn the
  remaining 2,000 — diagnose, fix, and collect a replacement tranche. A held-out set is
  consumed by every look at it.
- **Fallback target, decided in advance, not after a miss:** if Phase 2 proves the label
  floor blocks 99.9% at 0.03, the honest restatements are (a) 99.9% at a tolerance the
  labels support, or (b) 99.9% on the accepted subset with a calibrated abstain rate
  (report both the accuracy and the coverage). Both are legitimate; silently relaxing the
  tolerance after seeing the test result is not.

## 6. Phase 5 — Collection and ops (the schedule risk)

Scale needed: ~3,000–10,000 held-out clips plus a training corpus that should grow with
it (target ~20,000–30,000 total). Measured burst rate was ~379 clips/h across both
phones; assume ~300/h sustained fleet -> ~10 fleet-hours per 3,000 clips, so days, not
weeks, **given healthy phones**.

Blockers and prerequisites, in priority order:

1. **Signing — OWNER DECISION 2026-08-19: staying on the free 7-day team. Closed.**
   Both WDA ports are UP only because a *running* WDA survives expiry; any restart of
   `com.trueskate.services`, any phone reboot, or any crash ends collection until an
   interactive Apple ID sign-in happens at the rig. This plan previously called a paid
   account ($99/yr) the highest-leverage non-model action. **Asher has declined it, and the
   premise it rested on no longer holds:** the original risk assessment assumed he was
   remote from the rig for long stretches. He is now within reach of the rig for at least an
   hour every day and re-signs on the 7-day cycle without difficulty, so an expiry costs
   hours rather than days. Treat re-signing as routine scheduled maintenance, not as a
   schedule risk to the collection plan. What remains true and worth keeping: do not restart
   a healthy `com.trueskate.services` to test something (memory
   `wda-signing-free-team-7day-expiry`), because that converts a working rig into one
   needing a hands-on 2FA sign-in.
2. **Verify the balanced-holdout corpus before trusting it.** `basic_linear_xctest_balanced_holdout`
   has 412 `meta.json` but only 304 distinct waypoint/duration commands and a 367/45
   XR1/XR2 split. Some of the gap is calibration taps, but run the strict loader and
   confirm one-command-per-clip and the device balance before it is used for anything.
3. **XR2 throughput.** XR2 contributed 45 of 412. Its intermittent-calibration issue and
   its WDA provisioning history make a device-balanced 3,000-clip holdout the long pole.
   Either fix XR2 properly or predeclare an unbalanced protocol honestly.
4. **Storage/inode headroom.** The main `trueskate-corpus` volume is effectively full
   (997.9/1024 GB, 2.37M files against a 500k inode limit). Confirm the MVP volumes have
   room for a 10x corpus *before* collecting; video-encoded clips (one `frames.mp4` per
   sample) are what make this tractable — keep that, never revert to PNG-per-frame.

## 7. Ordering

1. Phase 1 autopsy (offline, hours) — **start here, unblocked**
2. Phase 2 label-noise floor (offline, hours-to-a-day) — **decides whether 99.9% is real**
3. Phase 3 4a robust line fit on the frozen 2k benchmark; 4b resolution ablations in parallel
4. Phase 5 item 1 (paid Apple account) in parallel from day one — it gates everything after
5. Phase 4 protocol predeclared and committed, then collection, then one-shot evaluation

Phases 1 and 2 need no rig, no collection, and no Modal capacity, and either could
invalidate the rest of the plan. They are cheap; run them before spending anything else.
