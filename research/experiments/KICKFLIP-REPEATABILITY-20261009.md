# KICKFLIP-REPEATABILITY-20261009 — XR1 push/pop/flick repeatability

Status: implemented and checked offline; the operator confirmed XR1 is charged
and ready in Workshop. Initial live setup stopped at WDA build identity: the
runner reported `unversioned`, rather than required `ae50404a`. One reserved setup
attempt made only its initial reset/settle; **no recorder start or gameplay
contacts occurred**. This failed start counts toward the 24-attempt cap. WDA
was left healthy. The operator explicitly approved switching to the preserved,
signed, previously validated runner; deployment/continuation is underway.
Model 1 is not involved.
The operator authorized bounded setup, then 20 repeats only after
accepting the candidate recording in QuickTime. Collection stays OFF.

## Frozen procedure

- XR1 (`iPhone_XR`), WDA 8100, 414×896 logical coordinates. The normal Appium
  server on 4723 lacks Appium 3 session discovery. Use a temporary diagnostic
  Appium on 4725 with `xcuitest:xctest_screen_record,*:session_discovery` enabled;
  keep the healthy WDA and existing supervised services running. Require idle
  WDA before connection and verify both Appium and WDA ownership on every guard.
  Record the selected Appium port in immutable context. Use an
  unobstructed Workshop flatground waypoint, unchanged stance/camera/physics.
  Record the readiness statement and waypoint/settings notes; retain initial
  scene images for every attempt. A settled image is not proof of identical state.
- Historical push: `(0.7658, 0.3044)` → `(0.7658, 0.6797)`, 20 ms, once, then
  480 ms after the blocking request returns. The two-point push's easing setting
  does not encode acceleration. No PUSH_COUNT/PUSH_END_Y environment overrides.
- Two seeds: `tail_20260611/kickflip_2g_20260611_095155.json` best recipe, and the
  first two strokes of `kickflip_handguess_20260613.json` (without its catch).
  Libraries are historical candidates, not guarantees of current kickflips;
  stance and incomplete generating provenance remain relevant.
- The mined pop/flick durations start at 180/210 ms, gap 620 ms; the hand guess
  starts at 60/50 ms, gap deliberately widened from 30 to 480 ms for sequential
  WDA delivery. Preserve the existing compiler's easing and ms truncation; store
  both requested parameters and exact encoded durations/payloads.
- Search the two seeds with pop/flick duration factors 0.8/1/1.2, centre first,
  then ±80 ms gap variants. This prepares 22 candidates; all whole paths and
  quantized coordinates must avoid existing protected controls. Cap **all setup
  attempts, including confirmations and failed starts, at 24**.
- One request per gameplay contact, in push → pop → flick order. No catch,
  spin, overlapping contacts, implicit scheduling compensation or bundling.
  Record actual submission gaps; requested gaps are not physical timing proof.

Each trial has its own ≤60 s XCTest recording at requested 60 fps. After a
pre-recording reset/settle, centre controls run at 1.5/30/57 s; labelled resets
run at 3/22/49 s and settle before the following guard/event. Push starts at 14 s;
pop follows the historical post-response wait; flick follows the explicit gap.
The gameplay window ends before the reset at 22 s. Stop targets 59 s including
recorder-start latency, with one retrieval attempt on interruption/failure.

Validate WDA build `ae50404aac12d9f8c41f6c3fa8776e97975eaef5`, exact timing record
counts, native dimensions/PTS and full FFmpeg decode counts. Keep gameplay and
foreground guards, reserving guard/screenshot time before each submission slot.
Full screenshot/menu/editor guards surround the three-contact sequence and all
controls/resets. Pop/flick retain per-contact Appium/WDA ownership and foreground
guards; costly screenshots do not consume their short gesture gaps. Every native
video frame still receives the same menu/editor checks during admission.
Start/end controls define the video clock with ≥55 s anchor
span; the held-out middle must agree within two native frames. Unobservable
controls or technical faults stop progress; do not relax admission or silently
replace recordings. Preserve original movies and errors separately from outcomes.

## Operator review and continuation

Use the first three assessed attempts of a candidate, including failures. At
least two must be verified landed KICKFLIPs before presenting it as ready. Render
all three gameplay windows into a QuickTime-compatible preview; retain source
hashes and clipping provenance. Open the actual copied recording on the laptop
with `open -a 'QuickTime Player' <absolute-movie-path>`.

Wait for explicit operator acceptance. Approval binds the exact manifest (including execution-code hashes),
candidate, settings, three source movies and preview. If changes are requested,
continue bounded setup or create a newly documented manifest after steering;
an approved recipe cannot silently change. If the cap is reached, stop and show
current evidence rather than launching another search round automatically.

After acceptance, one command runs exactly 20 frozen attempts. A failed or
interrupted batch cannot automatically restart or replace trials. Analysis counts
all outcomes, including failed tricks and unassessed videos, with technical failures
reported separately. Align inspection movies once at the first push; retain later
timing differences. Export source-frame snapshots before each contact and at
0.1/0.25/0.5/1/2/5 s after the encoded flick end, including actual frame PTS and
sampling offsets. Never treat rendered comparisons as native-frame measurements.

Report submission-gap distributions, callback latencies, calibration residuals,
frame gaps, trick/landing counts and observed gameplay divergence. Requested
durations and callback latencies are not independently measured touch boundaries.
This measures **whole-system repeatability**, not intrinsic game randomness or
future controller reaction latency. No video-loss/pose grader or training is added.

## Commands

Entrypoint: `scripts/collection/probe_kickflip_repeatability.py`. Use the existing
`.venv`, an isolated staged source tree and absolute paths. All artifacts remain
under `tmp/`; `training_admission=false` throughout.

`--root /absolute/experiment prepare` freezes the candidates offline;
`status` shows candidate IDs and progress. On the rig, `run-setup --candidate
candidate_01 --env-file /absolute/rig/.env --operator-ready-note '…'
--settings-note '…'` makes **one** attempt. Inspect native source video and
use `assess --trial setup/trial_01 --trick KICKFLIP --status landed
--evidence-note 'source-video observation …'`; labels are explicit observations,
not inferred from the library name. Confirm a successful candidate twice within
the setup cap; all candidate attempts must be assessed.

For the temporary Appium server add `--appium-port 4725` to every live command.
Session discovery must succeed; never bypass it after a permission error.

After resolving a failure before gameplay, prepare a successor with
`prepare --previous-setup /absolute/predecessor/experiment`. This copies and hashes
all predecessor evidence (including any short retrieved recording) and carries
its spent setup budget. Executed/partial gameplay or an approved/main batch
cannot migrate through this path. Read-only
WDA identity and idle checks now run before reserving an attempt/resetting.

Copy the experiment directory to the laptop, then `preview --candidate
candidate_01` renders and opens QuickTime (use `--no-open` when rendering on the
rig). Only after the operator accepts, `approve --candidate candidate_01
--operator-statement 'exact acceptance …'` writes the approval artifact. Transfer
that reviewed directory back to its isolated rig output if review happened locally.
`run-repeatability` requires the same readiness/settings and env-file arguments;
it will refuse an absent/changed approval or any previously started batch.

Finally `assess --trial repeats/trial_01 …` records each observed outcome and
`report --out /absolute/new-report-directory` generates JSON, Markdown, source
snapshots and comparisons against the first admitted repeat. Update this record
with the live source SHA, actual settings, paths, counts and conclusions.

## Offline verification

Synthetic tests cover separate contacts, timing waits, failed starts/interruption,
single retrieval, attempt caps, first-three review selection, immutable approvals,
native decode/calibration rejection, and exactly 20 repeats with no replacements.
Synthetic FFmpeg movies exercise preview/side-by-side rendering, source snapshots
and exact 60-frame native decoding. These are test artifacts, not research runs.

## Live preflight — 2026-10-09

Execution source: `b2c6ca3d4c58ec6ab6eca5ea7e581c73d09be215` on
`research/kickflip-repeatability`. The Appium 3 endpoint is `/appium/sessions`,
protected by `*:session_discovery`; the supervised 4723 server lacks this flag.
A temporary loopback-only server on 4725 passed idle discovery. Existing WDA,
Appium 4723 and the root recording tunnel were left running.

Rig evidence root:
`/Users/training-server/trueskate-ai-runtime/tmp/kickflip-repeatability-b2c6ca3d`.
Local copy:
`tmp/kickflip-repeatability-live-preflight-b2c6ca3d`.
`experiment/setup/trial_01/execution.json` retains the identity error, zero
recorded events and no video. A separate read-only session confirmed timing
schema 1, `build_revision=unversioned`, capture disabled and zero records.
All diagnostic sessions were disconnected; the recorder is idle. No trick
assessment, QuickTime candidate preview or 20-repeat batch exists.
The temporary Appium 4725 was stopped while restart permission was pending,
then relaunched for the authorized continuation.

Preserved replacement: `wda-timing-deployment/viewport-cache-20261003/derived-data/Build/Products`
under the rig runtime `tmp/`. The runner passes deep strict codesign verification;
its profile expires **2026-10-10 04:51:31 UTC**. Its WebDriverAgentLib binary
embeds the exact expected revision and has SHA-256
`cc418bc0e19c07d0b70d8ff2112e4331e6d322ed3085757fbb4748b0db651d6d`.
This is the already-tested CURVE-AUDIT build, not a freshly compiled candidate.
Full inspection is in `wda-recovery-preflight.json`. The merged rig WDA source
is clean at `ae50404a`; that alone does not identify the currently loaded binary.
`wda-recovery-profilecheck.json` confirms the replacement profile includes XR1.
Correction: the first profile check selected an unused `WebDriverAgent-timing`
DerivedData directory (`gbz…`), whose profile expired 2026-09-20. It did not identify
the current supervisor's products. `xcodebuild -showBuildSettings` resolves the
active project to `WebDriverAgent-dfaybgzjcctyfxgzkzvwwwsjarle/Build/Products`;
its original profile expires **2026-10-10 05:12:09 UTC**. The earlier claim that
the active runner's profile had expired was incorrect.

Do not accept `unversioned`, overwrite the failed attempt, reset the setup budget,
or weaken admission. Document the deployment and resolved failure before
continuing. Preserve the failed start separately from gameplay
outcomes. Twenty repeats still require actual QuickTime review acceptance.

## Approved deployment and continuation — 2026-10-09

The operator replied **“Yes”** to switching XR1 to the verified build and continuing
setup. Copied the validated signed products into the current project's default
DerivedData selection, preserving the complete original as sibling
`Products.before-kickflip-20261009`. Requested restart of only XR1's WDA runner;
the existing supervisor relaunched XR1 services using `test-without-building`.
No WDA source or service definition was edited and no rebuild was needed. WDA
reported ready, build time Oct 2 22:04:17, no active session; True Skate remained
foreground. Receipt: `wda-deployment-receipt.json` in the original rig evidence root.

Continuation source `c0ea83acf94c60312120991c1cd7ffa4fe171f21` is staged at
`/Users/training-server/trueskate-ai-runtime/tmp/kickflip-repeatability-c0ea83ac`.
Its experiment manifest retains all predecessor evidence under
`history/predecessor`, including the failed start, and starts the next attempt at
`setup/trial_02`. Gesture candidate hashes, required WDA revision and admission
gates are unchanged. Thirty-nine focused offline tests passed, including immutable
failed-start history, budget carry-forward and rejection of recorded-run migration.

The timing identity passed on trial 02, but the first control exceeded its
100 ms scheduling-lateness gate by 207.546 ms. No actions were submitted; its
1,125,338-byte original and zero-record timing report were retrieved and retained.
Read-only guard profiling measured screenshot retrieval at 0.36–0.41 s, duplicate
image decoding/checks at 0.53–0.59 s, and ownership/foreground checks at about
0.18 s. This is implementation overhead, not evidence about game repeatability.

Repaired the schedule to reserve 1.5 s for full guards, moved the first calibration
slot to 1.5 s (anchor span remains ≥55 s), and moved reset slots to 25/52 s for
settling/guard slack. Stop checks run before the final wait plus a fresh foreground
check at stop. The push/pop/flick recipe and 100 ms lateness gate remain unchanged.
Images are decoded once for live checks; native BGR frames are converted directly
to RGB for identical admission predicates, avoiding a PNG encode/decode round trip.
The next successor retains both pre-gameplay technical failures and starts at
attempt 03, leaving 22 of the original 24 setup attempts available.

Trial 03 submitted only its start control and reset, then failed the unchanged
settling gate. Initial reset settling took 4.6 s; the in-recording allowance
expired at 3.417 s after differences 10.351, 3.295 and 0.676 (two consecutive
differences below 2 are required). Its 8,589,481-byte movie and two timing records
were retained. No push/pop/flick ran. Expanded calibration-only slack: first reset
3 s → push 14 s, middle reset 22 s → control 30 s, final reset 49 s → control 57 s.
Gesture durations and inter-contact timing, settling threshold, held-out residual,
native-frame and maximum-lateness gates remain unchanged. All three pre-gameplay
failures count toward the original cap; next attempt is 04, with 21 remaining.
