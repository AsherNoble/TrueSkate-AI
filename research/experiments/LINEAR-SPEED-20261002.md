# LINEAR-SPEED-20261002 — Duration-only linear drag sweep

The operator authorized sixteen diagnostic drags on XR1 in Inbound: normalized
(0.27, 0.78) → (0.91, 0.30), at 600, 400, 300, 200, 100, 50, 20 and 10 ms,
slow→fast in the first minute and fast→slow in the second. The entire diagonal
clears the expanded XR control map. Every frozen drag payload has exactly one
pointer-down, one positive-duration linear pointer-move and one pointer-up;
the initial zero-duration placement happens before contact. Duration is the
only drag-payload difference. Logical coordinates use the existing Selenium
integer conversion of normalized coordinates multiplied by 414×896.

The [manifest](../evidence/LINEAR-SPEED-20261002/manifest.json) was frozen before
execution. Start/middle/end controls remain at 1/30/57 seconds; diagnostic slots
are 6/12/18/24/36/42/48/54 seconds. Each labelled reset begins three seconds
before its diagnostic, followed by the existing centre-settle detector
(threshold 2, two consecutive quiet screenshots) and foreground/gameplay guard.
Preparation and execution must finish before the next scheduled request. The
scheduler's chosen sleep tolerance is 100 ms; it never compresses missed slots.
All reset requests are frozen and retained in the raw WDA report, so the
nineteen expected records include eight resets and eleven diagnostic/control
requests. Existing full-source decoding, gameplay screening, start/end affine
calibration and held-out middle control checks gate progression to recording 2.
These diagnostic resets do not change the production no-in-recording-reset rule.

## Attempt and result

The root RemoteXPC tunnel was running, XR1 WDA was healthy at instrumented
revision `b5ace21788b5f5dc4cf0e0759f8bb8a79ab83ae6`, the recorder was idle, and
no collector was active. The pre-recording reset passed settle after 2.173 s.
One recorder start succeeded. The first scheduled sleep exceeded the chosen
100 ms tolerance; execution aborted **before any control or diagnostic**.
The attempted recorder was stopped once and its original saved. WDA capture
contains zero records and zero dropped records. Recording 2 was not started.
No replacement, collector job, notification or WDA restart followed.

The original has 34 decoded native frames with PTS from 0.071667 to 1.221667 s.
The viewer remux uses `-copyts`, retains all 34 source PTS exactly, and exposes
native stepping, slow motion and all sixteen duration-labelled buttons as
disabled/unexecuted. The first frame shows the operator-confirmed gameplay
scene; there is no executed drag to classify. Visible feedback and gameplay
response are **not assessed**, rather than classifying an unexecuted request as
indeterminate. The repeatable transition range is **unmeasured**. At 30 fps,
20/10 ms commands would occupy less than one frame; tap-like rendering alone
could not establish path collapse even in a completed run.

The failure log did not capture the exact wake lateness. The scheduler now
records each wake time and lateness for future diagnosis; no further device
attempt was made after that logging change. The executed source is preserved
in the isolated rig directory and local bundle; hashes are in the
[summary](../evidence/LINEAR-SPEED-20261002/summary.json).

## Artifacts and verification

- Rig artifacts: `/Users/training-server/trueskate-ai-runtime/tmp/LINEAR-SPEED-20261002/`.
- Local originals: `tmp/linear-speed-worktree/tmp/linear-speed-run/recordings/`.
- Executed source bundle: `tmp/linear-speed-bundle.tgz` (ignored, retained).
- Existing viewer addition: http://127.0.0.1:8767/linear-sweep/.
- Synthetic tests check frozen JSON reconstruction, payload shape, reversed
  schedules, reset lead times, preparation/execution/sleep overruns, recorder
  start failure, incomplete timing and partial-evidence preservation.
- Source FFmpeg decode count and lossless-remux frame PTS match. The loopback
  viewer's HTTP byte-range response is 206. Browser UI inspection was unavailable
  in this session; its JavaScript was syntax-checked.

Collection stays off. No training dataset, checkpoint or model interface changed.
