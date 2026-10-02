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

## Authorized redo and native-frame viewer

The operator found the original UI unusable and explicitly overrode the
no-replacement rule for this occasion, requesting the unexecuted sixteen again.
The original failure and manifest remain unchanged. A gesture-free rig check
measured a one-second host sleep returning 146 ms late. Deadline waiting now
wakes 250 ms early and finishes in short sleeps, retaining the 100 ms lateness
limit; a rig check returned within 1 ms, and synthetic coalescing coverage passes.

Two partial forward attempts completed five drags each before preparation for
50 ms overran. A partial reverse attempt completed 10/20/50 ms before preparation
for 100 ms overran. These **thirteen additional partial-attempt drags** are
preserved and not substituted into the complete comparison. The eventual pair
contains all sixteen planned drags in two complete one-minute recordings, with
nineteen successful WDA requests in each (eight resets, eight drags, three controls).
There was no collector launch, WDA restart, recorder-start retry or training.

The three-second preparation lead, settle threshold 2, 250 ms sampling interval,
two consecutive quiet comparisons, foreground checks and one-minute bounds are
unchanged. Settle images come from newly delivered full-resolution WDA MJPEG
frames rather than repeatedly requesting slow screenshots. The subsequent
gameplay guard reuses the just-captured settle frame and still queries Appium
and WDA foreground state. This removes redundant image requests without
changing the detector thresholds. Appium sessions close before offline decode;
no healthy WDA service is restarted.

Both original full-video admissions failed with the same gameplay-contamination
heuristic. They remain failed. Separate diagnostics retain the first flagged
source frames: repeat 1 frame 569 (19.12 s) and repeat 2 frame 414 (13.936667 s).
Visual inspection shows ordinary park gameplay in both, with bottom-strip
scenery colours rather than replay/menu controls. This identifies these first
flags as false positives; it is not a wholesale certification of all frames or
a change to collection admission. The remaining eight were explicitly executed
as an independent diagnostic repeat under the operator's sixteen-drag request,
retaining its own failed admission rather than overriding the first result.

Separate start/end fits and the held-out middle checks pass on both recordings:
14.300 ms and 9.253 ms residuals, with >55 s anchor spans. Each original and viewer
remux decodes to **1,767 native frames**, with every source PTS preserved. The
native-frame player uses decoded JPEG frames, including exact source indices and
PTS; this avoids the reproduced black video playback in Safari. Two image panes
keep the preceding frame visible while the next loads. Duration buttons pair the
two repeats, and annotation fields sit in a collapsed panel. Safari verification
confirmed actual gameplay rendering, playback, clip selection and one-frame
stepping. The old viewer URL redirects to the completed
[sixteen-clip viewer](http://127.0.0.1:8767/linear-redone/).

Assistant visual assessment from native-frame sheets, with gameplay response
recorded separately (operator assessment remains primary):

| Command durations | Visible feedback in both repeats | Gameplay response |
| --- | --- | --- |
| 600, 400, 300, 200, 100 ms | Moving contact / trail | Board turns or rotates; scene/camera moves |
| 50 ms | Visible trail; motion sampling differs between repeats | Strong board rotation / airborne response |
| 20 ms | Indeterminate; no confident moving trail identification | Strong rotation / airborne response |
| 10 ms | Tap-like stationary spot near touch start | Small scene/camera displacement; no comparable large rotation observed |

The repeatable **visible-trail transition is between 20 and 50 ms**. This is not
a measured input-collapse threshold: 20/10 ms are below one native frame, and
the 20 ms gameplay response persists despite poor touch rendering. The latter
half of repeat 2 also resets into a different area of Inbound, so scene and
board response are not perfectly held constant across the complete pair.
No orange-mask extraction or model training was used for the visual assessment.

Complete manifests, original execution/timing reports, diagnostic residuals,
per-drag assessments and source bundle hashes are in the
[redo summary](../evidence/LINEAR-SPEED-20261002/redo/summary.json).
Complete raw sources remain at
`/Users/training-server/trueskate-ai-runtime/tmp/LINEAR-SPEED-20261002-redo3/recordings/recording_1/`
and
`/Users/training-server/trueskate-ai-runtime/tmp/LINEAR-SPEED-20261002-repeat2b/recordings/recording_2/`.
Local originals and diagnostic flag images are under
`tmp/linear-speed-worktree/tmp/linear-speed-redo3/recordings/`.
