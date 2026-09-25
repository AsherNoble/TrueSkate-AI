# M1-DIE5-COMPARE-20260925 — Five-touch vs single-touch calibration

Status: protocol preregistered before any comparison recording. Phase 0 started.
No collection, corpus edits or training are authorized by this record.

## Question

In the conditions that produce false calibration detections, does the
five-touch die marker reduce wrong anchor frames versus the single centre
touch, without rejecting too many markers?

Main risk: M1-LA-ORIGINALS reproduced the fault in 0/7 recordings. The design
must preserve the production start-control timing after reset.

## Design

Paired: every marker is a die-five touch (centre `(207,448)`, corners ±35 pt).
Each marker is scored by:

- **A** — the unchanged production centre-only `detect_tap_onset`.
- **B** — the same detector at all five points; accept only if ≥4 points
  agree within ±1 frame, onset = median of agreeing frames; otherwise reject.

Detector B, its search windows and the comparator are frozen in code before
Phase 2. Corner check: corners (49.5 pt) may intrude on A's ring (~40 pt), so
a subset of recordings alternates real single touches with die-five markers to
compare A's scores and errors on both.

## Phases

0. Offline: per device/park/anchor bad-fit prevalence (`abs(rate - 1) > 0.002`)
   from existing segment metadata to size Phase 2; implement B and comparator
   with synthetic tests. The die-five and LA-originals recordings are
   development data only.
1. Feasibility, both XRs: a 50 ms die-five marker is visible at all five
   points, does not move the board or leave gameplay. Failure stops the study.
2. Recordings in an isolated directory with originals retained, not training
   data. One-minute production-style segments: start marker at production
   timing after reset, end marker, ~4 mid markers between linear swipes.
   Los Angeles ~20 recordings per XR; control park (Workshop or Kansas City)
   ~8 per XR.
3. Blind human labels of first visible centre-touch frame for every A/B
   disagreement or no-result, plus a random 25% (min 40) of agreements,
   shuffled together, no detector output shown.
4. False-alarm test: both detectors on windows with no WDA-recorded touch
   (≥1.5 s from any command). Any detection is a false alarm.

## Measures

Primary: Los Angeles per-marker gross-error rate (>1 frame from human label),
A vs B, McNemar. Secondary: exact/±1 rates, no-result rates, false alarms per
null window, and production-calibration replay (start+end fit): share of
segments outside `abs(rate - 1) > 0.002` and mid-marker residuals.

## Adoption criteria

1. Los Angeles: B has significantly fewer gross errors than A (p < 0.05) and
   none on accepted markers.
2. B rejects ≤5% of markers.
3. B is no worse than A in the control park.
4. The corner check shows A is unaffected by die-five geometry.

A pass does not certify untested parks or repair the existing corpus.

## Phase 0 results (2026-09-25)

Prevalence, read-only over all 1,409 aligned segments / 13,592 clips of
`model1_linear_replacement_20260922` ([fit-prevalence.json](../evidence/M1-DIE5-COMPARE-20260925/fit-prevalence.json),
[script](../../scripts/inspect/calibration_fit_prevalence.py)):

| Device / park | Segments | High rate (start implicated) | Low rate (end implicated) |
|---|---:|---:|---:|
| XR1 Los Angeles | 207 | 90 (859 clips) | 8 |
| XR1 Super Crown | 115 | 0 | 1 |
| XR1 Workshop | 425 | 0 | 0 |
| XR2 Kansas City | 393 | 0 | 4 |
| XR2 Skateboard GB | 182 | 0 | 4 |
| XR2 Super Crown | 87 | 0 | 3 |

Every high-rate segment is XR1 Los Angeles and implicates the start control
(median start score 13.3 vs 30.0 for ordinary fits). Failures come in streaks
across the 17-hour run and thin out after ~15:00; M1-LA-ORIGINALS saw 0/4 XR1
two days later. The fault is state-dependent, so reproduction is not assured.
Segments that failed calibration emitted no clips, so these are lower bounds.

Detector B is `trueskate_ai.collection.die_five_calibration` (frozen: ≥4/5
votes within ±1 frame, lower median, unchanged per-point detector defaults).
Synthetic tests cover a centre-only early decoy that fools A but not B, and
rejection below four points. Development replay on the three M1-DIE5 pilot
recordings: B accepted 12/12 with 5/5 votes, identical frames to the pilot.

Corner check risk is lower than assumed: at native 828 px width the ring's
outer radius is 40 px = 20 logical points, well inside the 49.5 pt corners.
The alternating single/die-five subset is retained anyway.

## Sizing amendment (before any Phase 2 recording)

- XR1 Los Angeles: 40 recordings. Gate: if the first 20 show no A start-anchor
  anomaly, pause and discuss rather than continue.
- XR2 Los Angeles: 10 recordings (no corpus history; M1-LA-ORIGINALS 0/3).
- Controls: XR1 Workshop 8, XR2 Kansas City 8.
- Six discordant markers with B=0 are needed for two-sided exact McNemar
  p < 0.05 (p = 0.031); five give only p = 0.0625.

## Phase 1 results (2026-09-25) — passed

Operator confirmed both XRs in SLS 2015 Los Angeles. Rig release `1b8497e`,
both WDAs healthy, tunnel running, no collector active. The generalised
`probe_five_simultaneous_taps.py` (`--device`, `--hold-s 0.05`) ran two
sub-minute recordings of four markers per XR, reset before recording. All four
recordings ended in gameplay.

[Analysis](../evidence/M1-DIE5-COMPARE-20260925/phase1-feasibility.json)
([script](../../scripts/inspect/analyze_die_five_feasibility.py)): detector B
accepted 16/16 markers with **5/5 votes on the same frame** every time. Mean
grey change outside the marker area from onset−1 to onset+15 frames was
0.002–0.069 levels, no larger than the pre-marker change (0.006–0.385), so the
50 ms marker does not visibly move the board or camera. Visual crops confirm
a static board; a difference image shows the five marks clearly, but they are
faint soft glows on the board in raw frames. Phase 3 labellers should expect
subtle marks.

**Correction (M1-DIE5-R100, 2026-09-25):** searching each marker from its own
call time shows XR2 markers 1 and 2 resolved to the same frame in both
recordings. XR2 marker 1 fell before the board settled and was never seen
separately, so "16/16" is really 14 distinct markers accepted. See
[M1-DIE5-R100](M1-DIE5-R100-20260925.md#phase-1-results-2026-09-25--passed-with-a-caveat). Originals: rig
`/Users/training-server/trueskate-ai-runtime/tmp/die5-compare-phase1-20260925/`
and local `tmp/die5-compare-phase1-20260925/` (not backed up by Git).

## Phase 2 capture method (fixed before recording)

`collect_sls_xctest.py --die-five-experiment` runs the unchanged production
basic-linear loop (flags from `scripts/ops/mvp_collect_linear.sh`: 50 ms
controls, reset before segment with 1.5 s settle, one-minute segments, WDA
timing capture, foreground/menu guards). Only three things differ: start and
end controls are die-five markers in one WDA request; four mid markers follow
payload samples 2/4/6/8 after the usual tail, alternating die-five and single
centre touch (the corner check runs in every recording); the aligner is never
spawned, so originals are retained and no clips are emitted. A mid marker that
opens non-gameplay UI discards the segment.

## Notes during Phase 2 (2026-09-25)

Protocol clarification before labelling: single-touch mid markers enter the
blind set at 25% (min 20) for the corner check. Viewer items are frame-exact
native crops (±90 pt around centre) with seeded ±0.3 s window jitter and a
difference-view toggle; the key is stored outside the served directory
([builder](../../scripts/inspect/build_die_five_label_viewer.py)).

Mechanism observation (not a Phase 2 result): the retained M1-LA-ORIGINALS
XR2 segment 3 (start score 10.8, rate 1.0018) shows the board still rotating
and sliding under the centre point after the pre-segment reset. Its bright
graphic stripes sweep the detector core; the single-touch detector fired at
1.647 s, ~30 ms after WDA submission, too early for a rendered touch. The rate
error implies ~98 ms (~3 frames) early. This suggests a separate candidate
fix — wait for a static scene before the start control — to test after this study.

## Amendment after design red-team (2026-09-25, before any Phase 2 result was read)

An experiment-red-team review (read-only; verified no file changes) returned
FIXABLE. Adopted, superseding earlier text where they conflict:

1. **Primary analysis:** Los Angeles *start* markers only, one per recording
   (the recording is the unit). McNemar on "gross error" (detected >1 frame
   from the human label). No-result is a separate outcome, never counted as a
   gross error or as correct. Report exact counts, the rule-of-three bound
   (0 errors in n ⇒ rate < 3/n), and device/time clustering of discordant pairs.
2. **Gate and maximum N:** an A-anomaly is an A start+end fit with
   `abs(rate - 1) > 0.002`. Maximum N is fixed at 40 XR1 LA + 10 XR2 LA
   recordings; no continuation beyond it to reach significance. If the first
   20 XR1 recordings have no A-anomaly, pause and discuss.
3. **Labelling:** 100% of Los Angeles start and end markers are labelled;
   other die-five mid markers use the disagreement + 25% (min 40) rule, with
   agreement-stratum rates inverse-probability weighted (len/k). Single mid
   markers 25% (min 20). The human target is the first frame any touch mark is
   visible. Record who labelled and confirm they had no access to analysis or
   key files.
4. **Rejection criterion** applies to start markers per device (≤5% each),
   not pooled.
5. **False alarms:** reported per second of scanned no-touch time. A
   post-reset null window (video start to 0.1 s before the start marker's WDA
   submission) is added, since it covers the moving-board fault state.
6. **Corner check** (corrected geometry: production decodes at 256 px, so the
   ring is ~16–32 pt, leaving ~17 pt to the 49.5 pt corners; glow radius
   unmeasured). Pass if, on labelled mid markers, A's within-±1 rate on
   die-five is no more than 10 points below single, and A's median score ratio
   die-five/single is 0.8–1.25. It covers post-swipe conditions only.
7. **Mid-marker residuals** of both fits are scored against human labels, not
   detector output.
8. Viewer frame indices use the aligner's own frame-time probe with a length
   assertion.

Not adopted without operator approval: a forced-motion enrichment arm
(shortened post-reset settle) — it adds device time and a new condition.
Scope: a pass shows fewer wrong start anchors on fresh recordings; it does not
repair the 859 affected clips or compare against a settle-before-start fix.

## Phase 2 results before human labels (2026-09-25)

Capture: rig code `e6d1436` (staged copy, release `1b8497e` venv, linked
`.env`). An initial XR1 attempt failed before touching the device because the
staged copy lacked `.env` (Appium selected a simulator); it was relaunched.
XR1 Los Angeles: 20/20 segments saved; XR2 Los Angeles: 10/10 (run alongside,
operator-approved). No guard trips, discards or lost segments. Originals
retained on the rig and copied locally with `scp -r` (sizes verified).

The analysis JSON is withheld from Git until labelling ends, so the labeller
stays blind; its SHA-256 is in
[phase2-analysis-sha256.txt](../evidence/M1-DIE5-COMPARE-20260925/phase2-analysis-sha256.txt).

- **Gate:** none of the 20 XR1 recordings had an A-anomaly (max A rate
  1.00077). Per the amendment, the remaining 20 XR1 recordings are paused for
  operator discussion.
- **Fault reproduced once, on XR2:** one XR2 recording had an A start+end fit
  rate of 1.00606. It is the recording with the highest pre-start centre
  motion. There, B's start marker reached only 2/5 votes and B returned no
  result, so the segment would be rejected rather than mislabelled.
  Human labels will decide whether A's start was a gross error.
- **Agreement:** A and B differ on 3/30 start markers and 2/30 end markers;
  where both detect a mid die-five marker, they never differ.
- **B no-result:** starts 1/30 (0/20 XR1, 1/10 XR2), ends 1/30, die-five mids
  12/60. The XR2 start rate of 1/10 exceeds the ≤5% criterion, but n is small.
- **False alarms (no touch sent):**

  | Condition | Scanned | A alarms | B alarms |
  |---|---:|---:|---:|
  | Steady windows | 157.5 s | 7 | 0 |
  | Post-reset windows | 47.7 s | 1 | 0 |

- **Motion link (exploratory):** 9/30 recordings had a moving board before the
  start. On 0.25 s screenshot-spaced change, still boards measure ≤0.58 and
  moving boards ≥2.9, rising to 14–24 by the start marker. All five B start
  markers with fewer than 5 votes came from moving-board recordings. In the
  7 M1-LA-ORIGINALS recordings, the only moving one was the only weak/high-rate one.

**Mechanism of B rejections (post-hoc):** in most rejections the two *upper*
corners fire 10–25 frames before the centre and lower corners, which agree
with each other. Mid markers follow a swipe, and scenery above centre
triggers first. The per-point detector keeps only its first onset, so an early
false trigger hides the marker at that point.

Exploratory **B′** (`earliest_candidate_consensus`) keeps every per-point
candidate and takes the earliest frame where ≥4 points agree within ±1 frame.
It is post-hoc and is not substituted for preregistered B. Any adoption needs
a confirmatory test on new recordings.

**Candidate root-cause fix:** `--start-settle-threshold` (`7b57d4f`, opt-in,
off by default) waits after the pre-segment reset until two successive 0.25 s
screenshot differences are below the threshold. The footage above suggests a
threshold of 1.5. If the board keeps rolling, the wait will time out; that
case needs a re-reset or a skip, which is untested.

## Exploratory label-free rate check (2026-09-25)

Fitted start+end rates, using the start/end frames from each detector over
the 30 Phase 2 recordings:

| Detector | Fitted | Median rate error | Max | Beyond 0.002 |
|---|---:|---:|---:|---:|
| A | 30/30 | 311 ppm | 6,056 ppm | 1 |
| B | 28/30 | 202 ppm | 575 ppm | 0 |
| B′ | 30/30 | 197 ppm | 575 ppm | 0 |

- **XR2 anomaly recording.** A's start gives +6,056 ppm. B′'s start is 10
  frames later with 5/5 votes and gives −189 ppm, consistent with A having
  fired early on the moving board.
- **Other disagreements.** In the three other A/B start/end disagreements (all
  moving-board recordings or ends), B's rate is closer to zero: A +421…+765 ppm
  against B −185…+151 ppm.
- **B′ false alarms.** B′ fired in 6/90 steady no-touch windows (B: 0), always
  with exactly 4/5 votes. The contested start had 5/5. A hybrid rule is a
  hypothesis for a new confirmatory test, not a finding: use B, and fall back
  to B′ only at 5/5 votes.

Rate plausibility is a proxy; human labels remain the primary evidence. See
[M1-SETTLE-20260925](M1-SETTLE-20260925.md) for the settle-wait smoke test. It
produced still start controls (a manipulation check); its effect on anchor
errors is untested (red-team: CONFOUNDED as originally worded).

## Phase 3 ready (2026-09-25)

The blind viewer was built with seed 2509250400 from the sealed Phase 2
analysis. It has 132 items: 30 start, 30 end, 12 mid die-five with
disagreement or no result, 40 mid die-five with agreement (25% floor 40) and
20 single mids (floor 20).

- It is served locally at `http://127.0.0.1:8780/` from
  `tmp/die5-compare-phase2-20260925/label-viewer/`.
- The key is `tmp/die5-compare-phase2-20260925/label-key.json`, outside the
  served directory (404 when requested).
- The page payload contains only item IDs (`000`…) and frame paths.

After export, score with
`scripts/inspect/score_die_five_compare.py --analysis … --key … --labels … --out …`.

## Exploratory label-free mid-marker check (2026-09-25)

Each mid marker's detection was compared with the time predicted from its WDA
submission by B's start+end fit. That fit is independent of the mid marker's
own detection; the 28 segments with a B fit were used. Aggregates only, so the
labeller stays blind:

| Detector | Within ±1 frame | >2 frames off |
|---|---:|---:|
| A on die-five mids | 51/56 | 4 (all 18–25 frames early) |
| A on single mids | 47/49 | 2 (12–23 frames early) |
| B on accepted die-five mids | 45/45 | 0 |
| B′ on die-five mids | 53/56 | 2 (≤19 frames) |

- A's early false triggers after swipes occur at similar rates on single and
  die-five markers. That is consistent with scenery rather than marker
  geometry (see corner check).
- B's cost is rejection, not wrong frames; B′ trades rejections for errors.
- This is a proxy that assumes the start+end fit is correct. Human labels
  remain primary.

The scoring pipeline was dry-run end to end on the real analysis/key files
with stand-in labels (values meaningless); it completes in ~20 min because it
probes every frame timestamp.

## Exploratory: multi-anchor consistency with the single-touch detector

Using only A's detections on all six markers per Phase 2 recording (start,
four mids, end), with the clock rate fixed at 1 (corpus median about +50 ppm;
the ±600 ppm spread is anchor quantisation), each anchor's offset
(video − WDA) was compared with the recording's median offset. Anchors more than
1.5 frames from the median were flagged:

| Anchor | Flagged |
|---|---:|
| Start | 1/30 (the XR2 anomaly start, −9.8 frames) |
| End | 0/30 |
| Mid | 6/113 (A's early false triggers after swipes) |

This suggests a cheaper production candidate than five-touch markers. Add one
or two extra single centre controls per segment, preferably after a settle
wait, and replace the two-point fit with a robust multi-anchor fit: median
offset, or Theil–Sen/RANSAC if the rate may drift (ALIGN-20260913 once saw
−1,900 ppm). Recordings whose anchors disagree would be rejected. This is post hoc and
exploratory; it needs its own test, and it costs one or two payload slots per
segment.

### Implemented as an opt-in path (not validated on device)

- `fit_multi_anchor_timeline`: exhaustive pairwise consensus (RANSAC over all
  anchor pairs within the 0.98–1.02 rate bounds), 1.5-frame inlier tolerance,
  ≥3 inliers spanning ≥30 s, then least squares on the inliers. Theil–Sen was
  tried first and rejected: with four anchors, one outlier contaminates half
  the pairwise slopes.
- The aligner gets a new `timing_alignment == "wda_submitted_multi_anchor"`
  path (method `wda-submitted-multi-centre-controls-v1`). The two-anchor path
  is unchanged.
- The collector gets single centre mid controls without
  `--die-five-experiment`: `--mid-markers N --mid-marker-kinds single`, with
  `--mid-control-gap-s` (default 1 s) so each control lands after the previous
  clip window. Manifests then declare multi-anchor timing.
- The timing screen passes multi-anchor clips.

**Offline replay on all 44 recorded Los Angeles segments** (Phase 2 plus both
settle runs), using A's detections on every marker:

- all 44 fits were accepted;
- outliers were the XR2 anomaly start (−10.1 frames) and 9 mid markers (A's
  early post-swipe triggers);
- the largest fitted rate deviation was 756 ppm, against 6,056 ppm for A's
  two-anchor fit.

Replay uses die-five mid markers seen through A's centre point, not production
single controls with the new gap. A bounded device smoke test of the production
path is the next step. Operator approval is recommended because it emits
isolated clips.

## Held-out check of the post-hoc B′ and hybrid (2026-09-25)

The 17 settle-run recordings (M1-SETTLE, M1-MULTIANCHOR tree excluded) were
captured after B′ and the "B, else B′ at 5/5" hybrid were proposed. They also
carry die-five markers, so they serve as held-out data. The reference is the
start+end fit (B's where available). Label-free proxy:

| Detector | Correct (≤1 frame) | No result | Gross (>2 frames) | Null-window alarms |
|---|---:|---:|---:|---:|
| A | 32/34 | 0 | 2 | 4/53 |
| B | 28/34 | 6 | 0 | 0/53 |
| B′ | 31/34 | 0 | 3 | 2/53 |
| Hybrid | 28/34 | 6 | 0 | 0/53 |

- **B′ does not hold up.** On held-out data it makes more gross errors than A
  and fires in no-touch windows. It is dropped as a candidate.
- **The hybrid adds nothing.** No rejected marker reached 5/5 under B′.
- **B replicates its Phase 2 pattern:** no wrong frames and no false alarms,
  with ~18% no-result on post-swipe mid markers (Phase 2: 20%).

## Code review note (2026-09-25)

A `/code-review` of `beccf3f..HEAD` found that B's consensus accepts positions
within ±1 frame of one shared candidate, so agreeing positions may be up to 2
frames apart. The protocol text said "within ±1 frame". The frozen code was not
changed after data collection. The docstring now states the actual rule, and
results are interpreted under it. The review also fixed two collector bugs:
production mid controls ignored `--no-menu-guard`, and `--mid-marker-every 0`
was not rejected outside die-five mode.
