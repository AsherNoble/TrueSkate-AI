# HID pointer pilot notes

XR2 (iOS 18.7.10), Freenove ESP32-WROVER-E, esp32 core 3.3.12, AssistiveTouch on with
"Perform Touch Gestures", default Tracking Speed. Record:
[HID-POINTER-20261004](../../research/experiments/HID-POINTER-20261004.md).

## Status (2026-10-04)

Working. The board is a bonded Bluetooth mouse that AssistiveTouch turns into one touch.
It plays whole gesture schedules from its own clock, and the demo replays as separate
strokes. Plan schedules with `trueskate_ai.control.hid_pointer` and play them with
`scripts/collection/run_pointer_schedule.py`.

## Operating the board

- Flash at 115200 baud (`UploadSpeed=115200`); 921600 fails verification. `arduino-cli`
  needs `TMPDIR` set to a writable directory on the rig.
- Opening or closing the CH340 serial port resets the ESP32 and drops Bluetooth.
  `hid_bridge.py` opens the port once (DTR/RTS low) and relays commands over TCP
  (127.0.0.1:8765). Talk to it with `hid_client.PointerClient`, never by opening the port.
- A reflash without `EraseFlash=all` keeps the bond. iOS does **not** reconnect by itself
  after the board restarts: tap "TrueSkate Pointer" in Settings → Bluetooth.
- After changing the GATT layout or name, erase flash (`EraseFlash=all`) and forget the
  device on the phone: iOS caches the services and the name per bonded address.
- The AssistiveTouch cursor shows in XCTest recordings: a ~17 pt grey disc,
  translucent while hovering and darker while pressed.

## What iOS accepts

- **Relative reports only.** An absolute (digitizer-style X/Y) report map pairs, but
  iOS shows no cursor and makes no touches. iPhone has no pointer-acceleration switch.
- **Encryption is required.** Pairing must bond: see the security bug below.
- **Connection interval 15 ms.** iOS connects at 30 ms and then moves to 15 ms with
  peripheral latency 4. `PARAMS 12 12 0 400` (15 ms, latency 0) is accepted. Anything
  allowing 11.25 ms is refused (status 15). Reports leave on the next connection
  event, so planners use a 15 ms grid.
- **At most ~2 reports per connection event.** Faster reports are queued, then dropped
  with no error: 60 reports sent 1 ms apart delivered 33–34 over ~250 ms. Send at most
  one per 15 ms. Even then ~6% of reports reach the screen a frame early or late
  (`hover_rate_probe.py` / `hover_rate_measure.py`, record section 8).

## Pointer gain (measured 2026-10-04, 49 drags)

One report of count vector c moves the cursor D(|c|) points along c:

| \|c\| | D (pt per report) |
| --- | --- |
| below 3 | 0.2017 \|c\| |
| 3 to 25.8 | 1.0009 \|c\| − 2.107 |
| 25.8 to 71.7 | 2.4485 \|c\| − 39.405 |
| 71.7 to 127 | 0.965 \|c\| + 66.997 (190 pt at 127) |

- The fit is 0.14 pt rms per report (max 0.44).
- **Timing doesn't matter.** Reports 15, 30 and 60 ms apart gave the same gain.
- **No memory.** 16 counts moved 14.0 / 13.75 / 14.0 / 13.9 pt per report at
  n = 1 / 2 / 4 / 16.
- **x and y behave the same.**
- **Vector, not per axis.** (−3, 4) moved 2.90 pt per report along (−0.6, 0.8), exactly
  D(5). A per-axis model is off by 2×.
- **Homing.** 25 reports of (−127, −127) pin the cursor in the corner. Then 40 reports of
  (3, 11) park it at (109.1, 368.2) pt, repeatable to ±0.5 pt.

The measurements are re-runnable:
- `gain_probe.py` (rig) plays the drags and records them;
- `gain_measure.py` reads the hovering cursor before and after each drag;
- `gain_fit.py` refits the curve.

Re-measure after an iOS update or any AssistiveTouch Tracking Speed change.

## Touches

- **The lift must be its own report.** A report that moves and lifts loses its move:
  the touch ends one step short.
- **Lift timing.** A still lift 1 ms after the last move ("immediate") keeps the whole
  path, and usually rides the same connection event, so the finger does not dwell
  before it leaves.
- **Press.** A still report, with moves on the following slots.
- **Trail.** True Skate draws its orange trail only while a touch moves; a stationary
  press shows nothing until it moves.
- **Pushes.** Vertical drags on the floor push the board. Horizontal ones leave it alone.
- **Validation.** Hover-then-press landed 0.35–2.15 pt from six targets across the screen.

## History: the security bug (2026-10-04)

At first XR2 connected and subscribed, but AssistiveTouch listed the device greyed out
with an endless spinner (`2026-10-04-assistivetouch-spinner.png`).

`hid_log.py` logged every GATT access. The pairing sequence had **no authentication
step**: connect → MTU 517 → manufacturer → PnP ID ×2 → manufacturer → HID info →
report map → report reference → CCCD write 1. The link stayed unencrypted.

Cause: on esp32 core 3.3.12 (Bluedroid), `BLESecurity::setAuthenticationMode(uint8_t)`
only stores the flags. It never sets `m_securityEnabled`, so
`BLEDevice::gattServerEventHandler` skips `startSecurity()` on connect.

Fix: use `setAuthenticationMode(bonding, mitm, sc)`.
