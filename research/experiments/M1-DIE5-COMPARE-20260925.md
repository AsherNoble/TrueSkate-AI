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
