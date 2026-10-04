# HID-POINTER-20261004 — Can a Bluetooth pointer replay an expert's gesture sequence?

Status: calibrated and validated; demo replay recorded, operator grades pending.
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
- **Grades.** Operator grades, side by side with the clip: pending.

## Conclusions so far

The pointer route solves what XCTest could not: separate touches with short,
exact gaps between them, from one schedule. Positioning error is ~1–2 pt. The
timing floor is the 15 ms Bluetooth grid; no gesture is ever conjoined.

## Next

- Operator grading of the three replays.
- Spin (planned step 6): a capacitive pad on the spin button, driven by a board GPIO on
  the same schedule, so spin shares the board's clock with the gestures.
- Re-measure the gain if iOS updates or the AssistiveTouch Tracking Speed changes.
