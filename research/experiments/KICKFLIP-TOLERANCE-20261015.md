# KICKFLIP-TOLERANCE-20261015 — how far can the kickflip's touches be off and still land?

- **Question:** how much position error does a known-good kickflip tolerate before it stops landing?
- **Result:** landed 10/10 at 0, 8/10 at 2% of the screen, 1/9 at 5% and 0/10 at 10%.
- **Decision:** labels, Model 1 and the robotic hand need positions within ~2% of the screen. Trick Model 1
  MVP v0 places only 37% of test strokes that well, so the pops' end comes first.
- **Next:** item 10 in [SYSTEM_MAP.md](../SYSTEM_MAP.md) (a 16–33 ms end hold on pops), then the 1,000-clip batch.
- **Link:** evidence `KICKFLIP-TOLERANCE-20261015/kickflip-tolerance-f50bb7d1/` (laptop); review export and
  tally in [research/evidence/KICKFLIP-TOLERANCE-20261015/](../evidence/KICKFLIP-TOLERANCE-20261015/).

Explore mode; plan item 1, approved 2026-10-15.

## What ran

- **Gesture:** `candidate_19` from [KICKFLIP-REPEATABILITY-20261009](KICKFLIP-REPEATABILITY-20261009.md)
  (landed 19/19): push, pop, flick and catch through WDA's direct stroke path, one request per gesture.
- **Error:** every point of the pop and the flick moved by one fixed distance in a random direction per
  attempt; push and catch unchanged. Distance is Euclidean in normalized screen units (x/414, y/896),
  the metric used for Model 1 recovery. Levels 0, 0.02, 0.05 and 0.1; 10 attempts each in a seeded,
  shuffled order (seed 20261015).
- **Where:** XR1 in The Workshop at the saved flatground waypoint, facing the closed door; existing
  stance, camera and physics. The operator confirmed readiness at 21:38 Sydney on 2026-10-15; the run
  started at about 22:12 (staged commit `f50bb7d1`).
- **Admission:** the repeatability run's gates (marker calibration, decoded frame counts, foreground).
  Attempt 18 (5%) was a technical failure: the recorder stop failed with "Unable to locate XCTest
  screen recording". Its stuck attachment was deleted on the operator's OK, and it was never rerun.
- **Review:** the operator judged all 39 attempts on the review page, blind to the level: native
  frames from 0.5 s before the pop to 3 s after the flick, L (landed), F (not) or U (unsure), with notes.

## Results

| Distance (share of the screen) | Landed | Not | Landing rate |
|---|---:|---:|---|
| 0 | 10 | 0 | 10/10 |
| 0.02 | 8 | 2 | 8/10 |
| 0.05 | 1 | 8 | 1/9 |
| 0.1 | 0 | 10 | 0/10 |

No attempt was marked unsure.

What the misses looked like, from the operator's notes:
- **2%:** both misses were caught a quarter turn late and landed primo. One landing was close to primo
  before it tipped back.
- **5%:** about half were near misses (caught a quarter to half a turn late, or half a turn too far).
  The rest failed outright: no pop, half a backflip, or a "soggy" flip.
- **10%:** the pop or flick failed: three pop shove-its, several that never left the ground, and one
  close miss.

## What it means

- The kickflip's position bar is about 2% of the screen. Even 2% costs some landings (8/10), so
  closer is better; somewhere between 2% and 5% it collapses.
- Model 1's labels are good enough when their error is below that bar. MVP v0 has 37% of test strokes
  within 2% everywhere along the path, and 90% within 5%
  ([MODEL1-TRICK-MVP-20261016](MODEL1-TRICK-MVP-20261016.md)), so it misses the bar. Its main fault is
  stroke ends that fall short, most of all on pops. That matches the pops whose visible trail stops
  10–20 pt short in the trick-speed trials.
- The robotic hand's position spec is the same ~2%.
- The error here moved the whole pop and flick together; a model's error varies along the stroke.
  The MVP is scored against the same bar, using the worst point of each path.

## Evidence

- Laptop: `/Users/ashernoble/TrueSkate-evidence/KICKFLIP-TOLERANCE-20261015/kickflip-tolerance-f50bb7d1/`:
  plan, attempts (recordings, execution and admission), the review export, `summary.json` and `summary.md`.
- In Git: `research/evidence/KICKFLIP-TOLERANCE-20261015/review-report.json` (the operator's export) and
  `summary.json` (levels, verdicts and counts per attempt).
- Code: `scripts/collection/probe_kickflip_tolerance.py`, `src/trueskate_ai/research/kickflip_tolerance.py`
  and `kickflip_review.py`. The pre-squash history is kept at
  `archive/squash-20261016/research/kickflip-tolerance`.
