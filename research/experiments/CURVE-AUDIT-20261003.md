# CURVE-AUDIT-20261003 — Direct timed-waypoint pilot

Status: **stopped on the first execution failure; 0/100 calibrated audit clips**.
This is an execution/recording feasibility result, not a curve-match verdict.

## Frozen protocol

Seed `2026100305`; manifest SHA-256
`db5c843c9a4f741e50948d7939fd601415bea0a8261da65d5b621b4358359670`.
50 paths, each assigned once to XR1/Inbound and XR2/Skateboard GB 2024.
Ten paths each: arcs, S curves, multiple bends, loops and reversals. Each family
has two paths at each of 150/300/600/900/1200 ms. Waypoint counts 5/8/11/15 occur
13/13/12/12 times; five timing profiles each occur ten times. Whole quantized
paths clear expanded controls. Abstract coordinates and exact outgoing payloads
were frozen before any recording.

Ten alternating one-minute segments were planned, ten samples each at
5/10/15/20/25/35/40/45/50/54 seconds. Die-five markers use `(207,448)` and corners
±71 points, held 50 ms at 1/30/57 seconds. Stop at 59 seconds. There are no resets
or stationary-board checks; menu/editor assessment is human. Collection jobs
remain off. No training or notifications were started.

## Attempt and stop

The first invocation failed before connecting or recording because Intel/ARM
math libraries differed by a few ulps in regenerated abstract positions.
The validator now permits only 1e-12 abstract-position differences; manifest
integrity, integer payloads, quantized positions, timestamps and order remain
exact checks. The frozen manifest was not changed.

The actual batch sent the start marker and path `p27` on XR1. This is a
15-waypoint, 900 ms multiple-bend path. Its call took **5.044694 seconds**, beyond
the next 10-second slot. The runner stopped immediately without compressing the
schedule, issuing replacements, continuing on XR2 or restarting services.

WDA reported two successful requests with complete instrumentation. For the
sample, preparation took **3.750958 seconds**; submission to the iOS completion
callback took **1.199558 seconds**. This localizes most request latency to WDA
preparation, but does not establish its cause or the path's visual fidelity.

The partial original is 10,293,261 bytes. All **301 source frames** decoded,
matching 301 native PTS (0.108333–10.158333 seconds). Middle/end markers were
never sent, so start/end alignment and the independent middle check are
unavailable. The recording is rejected, not promoted into a blinded clip.
The recorder returned idle. Shortfall: 99 unattempted sample requests and
**100 missing calibrated audit clips**.

## Preserved artifacts

- Rig: `/Users/training-server/trueskate-ai-runtime/tmp/curved-audit-20261003/`.
  `code/frozen-manifest.json` and the isolated executed source; `recordings/`
  contains the raw original, exact planned payloads, partial execution log,
  WDA timings and batch failure. The stable release was not changed.
- Laptop: repository `tmp/curved-audit-20261003-artifacts/`, containing the same
  manifest/raw evidence plus offline rejection and native-decode diagnostics.
- [Small diagnostic and recording hash](../evidence/CURVE-AUDIT-20261003/partial-diagnostic.json).
  Ignored raw recordings are not backed up by Git tags.

## Implementation and verification

Added the timed-waypoint interface, deterministic manifest generator, serial
bounded runner, five-point calibration/native-decode verifier, blinded native-frame
viewer builder and human-assessment importer. Viewer storage is separate from the
135-clip audit. A cyan unfilled ten-point ring is updated in the same image-load
callback as each native frame; targets are derived only from quantized command
positions and calibrated time. Every clip uses the same three-second window.

Synthetic checks cover balance/reproducibility, integer interval sums, single
contact, caps, full-path safety, pause/reversal interpolation, consensus failure,
independent-middle rejection, complete/partial lifecycle, exact native decoding,
public-label exclusion, rating exclusivity, saved comments/defaults and unchanged
old storage. The importer reports only explicitly saved human ratings and records
that device and park are confounded.

The 100-clip viewer and private *recorded-source* mapping could not be built because
no segment passed. No human ratings or execution-fidelity findings are claimed.
A revised schedule or WDA investigation followed by another device run requires
new authorization; this batch's stop rule remains in force.

## Post-stop analysis (read-only, no device actions)

WDA preparation time was compared across existing rig timing records
(`wda-timing.json`, `preparation_started` → `preparation_finished`) and the
segment counts in each run's `planned.json`.

| Run | Request | Moves/segments | WDA preparation (s) | Submission → iOS callback (s) |
|---|---|---:|---:|---:|
| LINEAR-SPEED-20261002-redo1 | linear, 100–600 ms | 2 | 0.107–0.153 | 0.319–0.822 |
| CURVE-EXEC-20261002 pilot | cubic, 300 ms | 16 | 0.853–0.988 | 0.623–0.633 |
| CURVE-EXEC-20261002 pilot | cubic, 1200 ms | 16 | 0.854–0.890 | 1.469–1.477 |
| CURVE-SPEED-20261002 | cubic, 300/200/120 ms | 16 | 4.304 / 4.300 / 0.904 | 0.627 / 0.498 / 0.231 |
| CURVE-JAGGED-20261002 | 600 ms / 300 ms | 48 / 56 | 2.770 / 14.358 | 0.785 / 0.262 |
| This audit, `p27` | 900 ms | 14 | 3.751 | 1.200 |

Findings:

- Preparation is the variable cost. Two-segment linear requests prepare in
  ~0.1 s; multi-segment requests take ~0.9 s at best and 2.8–14.4 s at worst. The
  same 16-segment form took 0.9 s and 4.3 s in different runs, so segment count
  alone does not predict it.
- Submission → iOS callback behaves consistently: about the commanded duration
  plus ~0.27 s (centre controls, 50 ms hold: ~0.28 s). The 1.200 s for `p27`'s
  900 ms gesture fits that pattern and is not by itself a fidelity defect. (Two
  short dense requests returned in less than their commanded duration; that
  deserves a check in any rerun.)
- WDA source (appium-webdriveragent 11.1.4, `FBBaseActionsSynthesizer.m:46-53`,
  `FBW3CActionsSynthesizer.m:278-311`): each viewport `pointerMove` is resolved
  through `[application coordinateWithNormalizedOffset:]` and then `.screenPoint`.
  **Hypothesis, untested:** each resolution queries the app's accessibility state,
  so preparation scales with move count and with how expensive True Skate's UI
  tree is to query at that moment.

Implications for a rerun (needs authorization; this batch's stop rule stands):
the fixed 5 s sample spacing cannot absorb multi-second preparation. Options
are (a) a schedule with slack sized to observed preparation, or (b) removing the
per-move cost in the project's own WDA build, then verifying with a no-recording
timing probe across 2/5/8/11/15 waypoints before any audit run.

## WDA fix: one app snapshot per request (2026-10-03)

The XR1 WDA log confirmed the hypothesis above: every viewport-origin
`pointerMove` resolved an `XCUICoordinate`, and each resolution requested an
accessibility snapshot of True Skate. The five-finger marker took 10 snapshots
(~30 ms each); `p27` took 31 (~125 ms each, the first stalling 3.3 s).

Fork commit `ae50404a` (AsherNoble/WebDriverAgent `master`) resolves the app
origin once per request in portrait and computes viewport/pointer points
arithmetically. Element origins and other orientations are unchanged. A new
simulator integration test checks synthesized points against per-item
`XCUICoordinate` resolution; 123 unit tests pass. Four existing tap/long-press
integration tests are flaky on Xcode 26.6 simulators with and without the patch
(4/20 failures each over five iterations).

Both XRs now run builds of `ae50404a`; the signing profiles were renewed
(expire 2026-10-10) and trusted on each phone. No-recording probe
(`scripts/inspect/probe_wda_preparation.py`, seed 20261003; evidence in
`../evidence/CURVE-AUDIT-20261003/wda-probe/`), WDA preparation in seconds:

| Waypoints | XR1 median / max (n=5) | XR2 median / max (n=2) |
|---:|---:|---:|
| 2 | 0.073 / 0.248 | 0.129 / 0.131 |
| 5 | 0.079 / 0.083 | 0.110 / 0.111 |
| 8 | 0.076 / 0.275 | 0.099 / 0.118 |
| 11 | 0.074 / 0.254 | 0.226 / 0.347 |
| 15 | 0.070 / 0.245 | 0.240 / 0.357 |
| 17 | 0.077 / 0.107 | 0.110 / 0.120 |
| 33 | 0.067 / 0.078 | 0.197 / 0.301 |
| 57 | 0.066 / 0.081 | 0.194 / 0.291 |

All 56 requests succeeded. Preparation no longer grows with move count. XR1
meets every planned limit (median ≤0.15 s, max ≤0.5 s). XR2 meets the maximum
but its median exceeds 0.15 s at 11/15/33/57 waypoints; with n=2 this is not
separated from park/game-state noise. WDA logged two snapshots per request
regardless of size (the second is the per-request orientation query). iOS
submission→callback stayed at duration + 0.22–0.32 s except 33 waypoints on both
phones (+0.44/+0.47 s); at 300 ms those segments are ~9 ms, the sub-16 ms regime
reported to make XCTest compress paths. The five-touch marker still succeeds
(preparation 0.06–0.13 s, iOS 0.29 s).

The audit can be re-run with `--wda-revision ae50404aac12d9f8c41f6c3fa8776e97975eaef5`
once the user authorizes it; this record does not re-open the stopped batch.

## Run 2 stop and protocol amendment v2 (frozen before run 3)

Run 2 (`ae50404a`, same frozen v1 manifest) executed XR1 segment 1 completely:
13/13 requests on schedule, 0.4–1.6 s each. Calibration then rejected it. Start
and middle markers had 5/5 agreeing onsets, but the end marker had 2 votes:
per-point onsets C/TL/TR/BL/BR = 13/13/26/25/17 window frames. With no resets the
board had drifted onto the quarter-pipe and sat under the C/TL points. It began
sliding about 0.4 s before the marker, so board motion, not touches, fired C, TL
and BR; TR/BL at 25–26 match the start/middle onsets (26–27).
[Annotated frames](../evidence/CURVE-AUDIT-20261003/run2-end-marker-frames.png).
The stop rule held: no replacements, and XR2 did not run.

User-approved amendment (schedule `v2`, identity
`curved-execution-20261003-v2-reset-before-markers`):

- Reset the board before recording and wait 3 s, then start the segment
  (start marker at 1 s).
- Reset at 27 s before the 30 s middle marker, and at 55 s before the end
  marker, which moves 57 → 58 s. The last sample slot moves 54 → 53 s so the
  longest (1.2 s) sample finishes before the reset. Start–end span is 57 s.
- Reset tap: (207, 49) pt, 50 ms hold, as in the linear speed sweep.
- Paths, payloads, device/park assignment, per-device order, the 2-frame
  middle check and the stop rule are unchanged. Samples still run in whatever
  state the board is in; resets touch only the calibration lead-ins.

Run 3 uses a fresh output directory with the v2 manifest frozen on the rig.

## Run 3 and amendment v3 (frozen before run 4)

Run 3 (v2 manifest `d7f4e164…`, `ae50404a`) verified segments 1–9: all 50 XR1
paths and 40 XR2 paths, with every start/middle/end marker accepted. XR2 segment
10 executed all ten samples, but its last sample (`p09`, 1.2 s at 53 s) returned
at 54.86 s; the post-request foreground guard then passed the 55 s reset slot
and the runner stopped on "overrun after guard". The reset and end marker never
ran, so segment 10 is uncalibrated and preserved, not used. The v2 margin test
had assumed 0.5 s of WDA overhead; run 3 shows up to 0.65 s plus the guard.

User-authorized replacement (run 4): re-run only XR2 segment 10 under schedule
`v3` (identity `curved-execution-20261003-v3-reset-before-markers`). Sample slots
25→24, 50→49 and 53→52 s give every sample at least 1.8 s after a 1.2 s gesture
before the next command. Markers, resets, paths, order and the stop rule are
unchanged. The viewer takes segments 1–9 from run 3 and segment 10 from run 4,
and only if paths, device, park and sample order match. The private map records
each clip's source manifest.

## Run 4 and human assessment results

Run 4 (v3 manifest `7680e3be…`) re-ran only XR2 segment 10. All three markers
had 5/5 votes, and the held-out middle error was −17.7 ms. The 100-clip blinded
viewer combined run 3 segments 1–9 (v2) with run 4 segment 10 (v3); conditions
matched, and the private map records each clip's source manifest.

The operator rated all 100 clips with no missing ratings: **66 Good, 21 Minor,
2 Major, 11 Unclear**
([results](../evidence/CURVE-AUDIT-20261003/curved-audit-results-20261003.json),
[export](../evidence/CURVE-AUDIT-20261003/blind-gesture-assessments-20261003.json)).
The viewer server was shut down when the report was returned.

| Mean time between points | n | Good | Minor | Major | Unclear | Good / judged |
|---|---:|---:|---:|---:|---:|---:|
| < 16 ms | 8 | 1 | 5 | 2 | 0 | 12% |
| 16–33 ms | 18 | 6 | 6 | 0 | 6 | 50% |
| 33–67 ms | 24 | 16 | 5 | 0 | 3 | 76% |
| 67–150 ms | 28 | 23 | 3 | 0 | 2 | 88% |
| ≥ 150 ms | 22 | 20 | 2 | 0 | 0 | 91% |

- **Duration:** 150 ms paths 7/20 Good (both Majors), 300 ms 8/20, then 600 ms
  17/20, 900 ms 15/20, 1200 ms 19/20. Within a duration, more waypoints did
  worse at 150 ms (5/8/11/15 points: 4/6, 2/6, 1/4, 0/4 Good). This points to
  spacing between points, not point count itself; both Majors had ≈6–7 ms
  segments (150 ms, 15 points).
- **Timing profile:** pause-containing paths were weakest (11/20 Good, 9
  Minor). Shape families differed less, except multiple bends: 8 Unclear, all
  four clips of each of its 300 ms and 900 ms path pairs.
- **Device/park (confounded):** XR1/Inbound 32 Good, 2 Major; XR2/Skateboard GB
  34 Good, 0 Major. 36/50 paths received the same rating on both phones.
- **Overlay timing caveat:** three comments note the trace appearing one frame
  before the ring (frame 21 vs 22); one notes the ring visibly lagging the
  trace tip. A ~1-frame calibration offset is within the 2-frame alignment
  limit, but it may have pushed some ratings toward Minor.

Interpretation (exploratory, single rater, n=100): direct timed-waypoint
execution is faithful when points are at least ~33 ms apart (about one 30 fps
frame). Below ~16 ms it degrades, consistent with the reported XCTest
behaviour of compressing densely spaced waypoints. A curve executor should cap
the point count at about duration/33 ms + 1, rather than sending fixed dense
polylines; a 150 ms flick then gets ~5 points.

## Expert demo replay (XR2, The Workshop)

The operator recorded a 360 flip on flat ground: 5 pushes (85–115 ms each), two
trick flicks (~50 ms each, starts 168 ms apart) and a 550 ms catch. Timed paths
were extracted deterministically: new orange trail pixels per 60 fps frame,
with motion ending at the last frame adding ≥1,000 pixels. The operator checked
the extraction against a moving-dot overlay
([gestures](../evidence/CURVE-AUDIT-20261003/demo-replay/extracted-gestures.json)).

- **Bundled (one W3C request, one pointer per gesture):** 9/9 replays rated
  Major. Every one was drawn as a single conjoined chain at native (~16 ms),
  33 ms and 50 ms spacing
  ([grades](../evidence/CURVE-AUDIT-20261003/demo-replay/bundled-grades.json),
  [key](../evidence/CURVE-AUDIT-20261003/demo-replay/bundled-key.json)). This
  repeats the May 2026 finding, now a hard rule in GESTURES.md.
- **One `perform_trick_gestures` request per gesture:** each call returns about
  270–330 ms after its gesture ends, so starts drift late. Push 5 started
  +447 ms late, and flick B started 315–343 ms after flick A instead of 168 ms
  ([timings](../evidence/CURVE-AUDIT-20261003/demo-replay/separate-runs.json)).
  Operator grades: 9/9 Major. The conjoining is gone, but the added delay
  between gestures is obvious and ruins the outcome at every spacing
  ([grades](../evidence/CURVE-AUDIT-20261003/demo-replay/separate-grades.json),
  [key](../evidence/CURVE-AUDIT-20261003/demo-replay/separate-key.json)).
  Expert sequences need on-device scheduling of one touch record per gesture.
- **On-device schedule, one record per gesture** (WDA `feature/gesture-schedule`
  `1a1589c7`): XCTest rejected the second record, submitted 333 ms after the
  first, with "only one gesture can be performed at a time". The first record
  (a 100 ms push) completed 0.341 s after submission. Separate records therefore
  cannot start within about 0.25 s of the previous gesture's end, whatever
  schedules them.
- **One record, one path per gesture with distinct finger indices** (`243c2cb0`):
  XCTest accepted it, but each 3.23 s sequence reported completion after
  1.54–1.87 s, and the operator rated 9/9 Major: conjoined again and much faster
  than the demo ([grades](../evidence/CURVE-AUDIT-20261003/demo-replay/paths-grades.json),
  [key](../evidence/CURVE-AUDIT-20261003/demo-replay/paths-key.json),
  [runs](../evidence/CURVE-AUDIT-20261003/demo-replay/paths-runs.json)). Distinct
  finger indices do not prevent joining. The leading but unverified explanation
  is that multi-path records drop idle time between paths, playing gestures back
  to back. The bundled W3C replays also completed in about 2 s.
- **WDA idle-wait settings** (`waitForIdleTimeout`, `animationCoolOffTimeout`
  set to 0): no effect. A 100 ms push call still took 0.42–0.45 s.
- **Anchor finger:** a still finger held through the whole indexed-path record
  at (0.96, 0.60) or (0.04, 0.70), 3 replays each. Records now completed in
  4.13–4.22 s, so the gaps were no longer collapsed. The operator still rated
  6/6 Major. True Skate draws the anchor as a touch. Once a gesture finger lifts
  while the anchor is down, the trail gains a line toward the anchor, so the
  trick flicks became a zig-zag. The game scored POP SHOVE-IT NOSE MANUAL, not
  the 360 flip
  ([grades](../evidence/CURVE-AUDIT-20261003/demo-replay/anchor-grades.json),
  [frames](../evidence/CURVE-AUDIT-20261003/demo-replay/anchor-replay1-sheet.png)).
- **Status:** via XCTest synthesis, sequential gestures are either at least
  ~0.25 s apart (separate records), gap-collapsed (one record), or linked to any
  concurrently held finger (one record with an anchor). None reproduces the
  demo's 118–170 ms sequence. Lower-level options (a jailbreak plus system-wide
  touch injection, or a single-pointer HID through AssistiveTouch) are outside
  XCTest.
- **Device iOS versions** (read with `ideviceinfo`, 2026-10-04): XR1 iOS 18.7.6
  (22H320), XR2 iOS 18.7.10 (22H374). The `iphoneos18.2` in WDA xctestrun names
  is the build SDK, not the device OS. Public A12 jailbreaks (Dopamine 3.x) stop
  at iOS 18.7.1, so neither phone can be jailbroken, and neither can be
  downgraded. The next route under test is a hardware pointer: a microcontroller
  acting as an AssistiveTouch HID mouse, with the gesture schedule timed on the
  board, plus a capacitive pad for the spin button on the same clock.


## Integration clarification (2026-10-05)

The opening status describes the first pilot. Subsequent separately authorized
runs 3+4 completed all 100 assessments as recorded above; the negative demo replay
findings remain unresolved. Device/build/jailbreak statements above are historical
observations from their stated dates. The spacing suggestion is exploratory, not
curved/spin certification. Current source uses explicit schedule matching and v2
content-bound reviews; it does not relabel or upgrade these historical v1 artifacts.
