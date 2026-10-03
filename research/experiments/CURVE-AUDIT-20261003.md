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
