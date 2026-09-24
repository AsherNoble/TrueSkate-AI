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
subtle marks. Originals: rig
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
[M1-SETTLE-20260925](M1-SETTLE-20260925.md) for the settle-wait smoke test that
removed the moving-board start condition in 11/11 recordings.
