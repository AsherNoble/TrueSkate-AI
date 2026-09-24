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
