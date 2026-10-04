# HID-POINTER-20261004 — Can a Bluetooth pointer replay an expert's gesture sequence?

Status: calibrated and validated; demo replay graded 3/3 Minor; corrected-stroke
replays recorded, grades pending.
XR2 only. No collection, no training admission.

## Question

XCTest could not replay the expert Workshop demo. Five pushes, two flicks and a
catch need several separate touches with short gaps between them, and XCTest
failed in every mode across four rounds
([CURVE-AUDIT-20261003](CURVE-AUDIT-20261003.md), on `research/curved-execution-audit`):
- separate records waited ≥ ~0.25 s between gestures;
- one record joined the gestures together ("conjoined");
- holding an anchor finger linked the strokes to it.

The XRs (iOS 18.7.6 / 18.7.10) cannot be jailbroken. Can a microcontroller acting as
a mouse through AssistiveTouch play the whole sequence from its own clock instead?

## Setup

- **Hardware.** A Freenove ESP32-WROVER-E on the rig's USB. Firmware
  `hardware/hid_pointer/hid_pointer.ino`: a BLE HID mouse with a relative 3-button
  report.
- **Firmware protocol.** The board stores an event list (`E t_us dx dy buttons`) and
  plays it on `GO`, busy-waiting on `micros()`.
- **Host link.** A persistent serial bridge (`hid_bridge.py`) keeps the port open,
  because opening the port resets the board.
- **Phone.** XR2 with AssistiveTouch on, "Perform Touch Gestures", default Tracking
  Speed. True Skate in The Workshop.

## Results

### 1. Getting accepted

The board bonded, but AssistiveTouch never adopted it until encryption was fixed.
On esp32 core 3.3.12, `setAuthenticationMode(uint8_t)` never enables security
([NOTES](../../hardware/hid_pointer/NOTES.md)).

Absolute-position reports are ignored by iOS. Relative reports work: the cursor
appears, and presses become touches.

iOS runs the link at a 15 ms connection interval and refuses 11.25 ms. Every report
is therefore quantised to a 15 ms grid. A schedule placed on that grid shifts by one
constant delay and keeps its internal gaps. The board itself fires every event within
5 µs of schedule.

### 2. Pointer gain

- **Method.** 49 board-timed drags across 6 recordings. Each drag: home, park, press,
  n equal reports, lift. The displacement is read from the hovering cursor before the
  press and after the lift (`gain_measure.py`, Hough circle on the cursor disc).
  Evidence: [gain-drags](../evidence/HID-POINTER-20261004/gain-drags),
  [fit](../evidence/HID-POINTER-20261004/gain-fit.txt).
- **Result.** Per-report displacement depends only on the length of the count vector,
  and is applied along it:

  | \|c\| | points per report |
  | --- | --- |
  | < 3 | 0.2017\|c\| |
  | 3–25.8 | 1.0009\|c\| − 2.107 |
  | 25.8–71.7 | 2.4485\|c\| − 39.405 |
  | ≥ 71.7 | 0.965\|c\| + 66.997 (max 190 at 127) |

  The fit is 0.14 pt rms per report (max 0.44); the worst whole drag is off by 0.95 pt.
- **No dependence on report timing.** 6 counts: 0.630 / 0.622 / 0.640 pt per count at
  15 / 30 / 60 ms. 16 counts: 0.816 / 0.833 / 0.796.
- **No dependence on history.** 16 counts gave 14.0 / 13.75 / 14.0 / 13.9 pt per report
  at n = 1 / 2 / 4 / 16; 45 counts gave 71.0 / 70.8 / 70.75.
- **Same on both axes.** Vertical: 3.87 / 13.88 / 38.9 pt per report. Horizontal:
  3.91 / 13.9 / 39.0.
- **Vector, not per axis.** (−3, 4) moved 2.90 pt per report at −0.6, 0.8. The vector
  model predicts 2.89; per-axis predicts (−0.9, 1.9).
- **The planner** (`trueskate_ai.control.hid_pointer`):
  - inverts the curve per report on integer counts;
  - aims each report from the modelled position, so error does not accumulate;
  - homes and parks at a measured point, (109.1, 368.2) pt, ±0.5.

### 3. Validation

Details in [taps and combined lift](../evidence/HID-POINTER-20261004/validation-taps-and-combined-lift.txt)
and [lift modes](../evidence/HID-POINTER-20261004/validation-lift-modes.txt).

- **Positioning.** Six hover-then-press targets across the screen landed 0.35–2.15 pt
  from target (mean 1.27). The goal was ±5.
- **Flick end points.** Five planned 240 pt flicks (three reports each) came to rest
  1.7–2.8 pt from target.
- **The lift must be its own report.** With the lift on the last move report, iOS
  dropped that move and the touch ended 80 pt short. A still lift 1 ms after the last
  move ("immediate") keeps the full path and usually rides the same connection event,
  so the finger doesn't dwell before leaving. "immediate" is the default.

### 4. Demo replay

- **Run.** [Plan](../evidence/HID-POINTER-20261004/demo-replay/plan.json) and
  [runs](../evidence/HID-POINTER-20261004/demo-replay/runs.json): 3 repeats of the
  extracted gestures (`extracted-gestures.json` from CURVE-AUDIT), all 8 strokes in one
  board schedule.
- **Board timing.** Every event within 3 µs of schedule.
- **Clip timing.** After removing one constant offset, the 15 ms grid puts each press
  within ±6 ms of the clip.
- **Modelled path ends** are within 1.1 pt.
- **What it looked like** (spot check of one repeat):
  - the pushes and both flicks arrive as separate strokes at the clip's spacing;
  - the board flips and lands. That repeat's trick scored as "360 Pop Shove-It", where
    the clip shows "360 Flip".
- **Grades: 3/3 Minor**, side by side with the clip
  ([grades](../evidence/HID-POINTER-20261004/demo-replay/grades.json)). Every XCTest
  round was Major. Operator comments:
  - "All the gestures appear to have been executed at the correct times and more or
    less the correct positions."
  - The miss was a minor difference in the first flick (the scoop).
  - One repeat did a 360 Pop Shove-It: "it spun the correct amount but just didn't
    include a flip component".

### 5. Stroke ends were short (an extraction bias)

The demo strokes were extracted as the centroid of each frame's new trail
(CURVE-AUDIT). That centroid sits mid-segment. Fast strokes therefore start about
half a frame of motion late and stop about half a frame short. The frames show it:
the scoop's trail begins on the board's tail edge and runs ~40 pt past the last
extracted point ([frames](../evidence/HID-POINTER-20261004/demo-replay-v3/flick-frames-new-trail-magenta.png)).

`scripts/inspect/extract_demo_strokes.py` reads the finger instead:
- on fast frames (≥ 1.5 glow radii of motion), the leading edge of the new trail,
  less the glow radius (12–13.5 pt, measured as half the trail's width);
- the press is the first piece's trailing edge, timed by extrapolating that frame's
  speed;
- frames are added while the leading edge still advances;
- slow frames keep the centroid. The catch is unchanged.

Changes against the replayed extraction, in path length (pt):

| stroke | old | corrected | press moved | lift moved |
| --- | --- | --- | --- | --- |
| flick A (scoop) | 162 | 213 | 15 pt, 6.5 ms earlier | 35 pt |
| flick B | 202 | 247 | 0 | 43 pt |
| pushes | 422–507 | 452–527 | 0–13 pt | 7–26 pt |
| catch | 185 | 185 | 0 | 0 |

The corrected points span each visible trail from end to end
([overlay](../evidence/HID-POINTER-20261004/demo-replay-v3/stroke-ends-v2-green-v3-magenta.png)).

### 6. Replays with corrected stroke ends

- **Run.** 3 repeats ([plan](../evidence/HID-POINTER-20261004/demo-replay-v3/plan.json),
  [runs](../evidence/HID-POINTER-20261004/demo-replay-v3/runs.json)). Board timing was
  within 1 µs; presses within ±7 ms of the clip.
- **Outcomes varied across identical runs.** The three repeats scored an FS Pop
  Shove-It, an Inward Heelflip, and a 360 Pop Shove-It that landed upside down
  ("FAILED"). The board's input was identical each time.
- **Leading hypothesis (untested).** Pointer updates arrive on a 15 ms grid, but the
  game samples touches once per 16.7 ms frame. One frame in nine receives two updates,
  which changes the per-frame movement the game sees in a ~50 ms flick. Which frame
  that is depends on each run's random Bluetooth phase.
- **Grades: 1 Minor, 2 Major**
  ([grades](../evidence/HID-POINTER-20261004/demo-replay-v3/grades.json)).
  - **Minor** (the upside-down run): "SO SO close. It 360 pop shoved and did half of a
    kickflip". This is the first replay with a flip component.
  - **Major** (the other two): "The scoop was a frame or two too early and wasn't
    quite right."
  - **Caveat: likely viewer sync.** The viewer synced each replay on its first push,
    detected in a ~30 fps recording. The two Major replays got offsets 66 and 33 ms
    larger than the Minor one, which displays them that much early.
  - **Board timing inside a replay.** The scoop is placed within ±7 ms of the clip
    relative to push 1.
  - **Conclusion.** With n = 3 per condition, the corrected strokes can't be called
    better or worse than the old ones.

## Conclusions so far

The pointer route solves what XCTest could not: separate touches with short, exact
gaps between them, from one schedule. Positioning error is ~1–2 pt. The timing
floor is the 15 ms Bluetooth grid, and no gesture is ever conjoined. The operator
graded the first replays 3/3 Minor.

Extracting strokes from trail centroids shortens fast strokes. Read the leading
edge instead (`extract_demo_strokes.py`). This also affects any Model 1 labels made
the old way.

Identical schedules can end in different tricks, so a single replay is not a
measurement of fidelity.

## Next

- Operator grading of the corrected-stroke replays.
- Measure run-to-run variance before more fidelity work: about 10 identical repeats
  and their trick outcomes. If the Bluetooth/frame beat explains it, test whether
  True Skate uses touch timestamps or per-frame deltas, for example by scheduling the
  same flick at controlled phases.
- Spin (planned step 6): a capacitive pad on the spin button, driven by a board GPIO on
  the same schedule, so spin shares the board's clock with the gestures.
- Re-measure the gain if iOS updates or the AssistiveTouch Tracking Speed changes.
