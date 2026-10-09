# KICKFLIP-REPEATABILITY-20261009 — XR1 push/pop/flick repeatability

Status: setup paused at 6/24 attempts. Trial 05 landed a 360 flip; trial 06
overran its flick slot. The procedure is revised (mined seed only; flick gap
waited after the pop request returns) and recording is blocked until XR1's
RemoteXPC tunnel is restored. See **Trials 05–06 and procedure revision** below;
the earlier sections record the original procedure and preflight history.
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
controls. Resets/pop/flick retain per-contact Appium/WDA ownership and foreground
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

Trial 04 completed only the start control, then the redundant pre-reset screenshot
check overran that reset's deadline by 212.031 ms. It was stopped and retrieved
before gameplay. Resets now use the same fast foreground/ownership guard as
pop/flick; every reset is followed by settling and a full scene check before the
next control or gameplay. This removes duplicate image work from the short
control→reset gap. A synthetic 1.4 s full-guard/4.6 s settling case verifies all
submission slots, gesture gaps and the one-minute stop without relaxing a gate.
Four retained pre-gameplay failures count; next attempt is 05 (20 slots remaining).

## Trials 05–06 and procedure revision — 2026-10-09

Source `c7acea0a`, rig root `…/tmp/kickflip-repeatability-c7acea0a`.

**Trial 05** (`candidate_01`, mined centre recipe) completed all nine commands and
passed admission: 3,541 native frames, maximum frame gap 33.33 ms, held-out
middle residual 1.741 ms. WDA submission gaps were push→pop 0.931 s and
pop→flick 0.803 s. The source video shows a landed **360 FLIP** (score 146), not a
kickflip. That observation is recorded in `assessment.json`.

**Trial 06** (`candidate_02`, hand guess: 60/50 ms, 480 ms gap) stopped with
`gesture schedule overrun: 0.362315s`. Every WDA call took 0.69–0.70 s, compared
with 0.44–0.66 s in trial 05. The flick was due 0.539 s after the pop call
started, but the pop call alone returned at 0.705 s. Push and pop executed; the
flick was withheld. The 20,089,340-byte original was retrieved. Under trial 06's
latency, the mined recipe's flick deadline (pop start + 0.799 s) would also have
been missed.

**Cause.** The runner timed the flick from pop *start*, which assumed less WDA
latency than was measured. The CMA-ES path that found the mined recipe ran gaps
above 0.6 s as separate requests and slept the gap *after* the pop request
returned (`execute_n_slot_gestures`, sequential branch, zero compensation). Its
flick therefore landed about 0.4 s later than in trial 05. The hand guess's
original 30 ms gap ran as one bundled payload. Widening it to 480 ms produced a
different recipe.

**Decision.** The operator chose "whatever we are likely to be using from now
on". Collection uses one request per contact, because bundled run04b was
rejected for joined gestures. Model 2 times strokes relative to the previous
stroke's completion. The revised procedure therefore keeps separate requests and
waits each gap after the previous request returns, as the push→pop wait already
did. The hand-guess seed is removed, leaving 11 mined candidates. Lateness,
guards, calibration, admission and the 24-attempt cap are unchanged; WDA latency
now appears in measured submission gaps rather than causing overruns.

**Recording-cleanup fault.** Every recording in this experiment logged `No tunnel
found for device` at stop, so its XCTest attachment was not deleted. The root
`com.trueskate.remotexpc-tunnel` daemon was running, but its registry held
**zero tunnels**: XR2's tunnel was removed on 2026-10-07 after `SSL read failed`,
and XR1 returned 404. The existing `launchctl … state = running` preflight could
not detect this. XR1 holds 12 UUID attachments. Five are retrieved originals
from trials 02–06. Seven come from the late-September `xr-control-4380162` run,
where retrieval failed with `Unable to locate XCTest screen recording`, so the
phone may hold their only copies. Preserve them to host storage before any
deletion. Live runs now also require a registry entry for XR1 and an Appium
cleanup dry-run reporting zero UUID attachments, checked before the first
connection and before each repeat.

**Continuation.** Restarting the tunnel daemon requires the operator (sudo).
Then preserve the seven unretrieved attachments, clean up with the documented
wrapper, and prepare a successor with
`prepare --previous-setup …/c7acea0a/experiment --superseded-reason '…'`. The
successor carries trial 05 as a superseded-procedure outcome and trial 06 as a
technical failure during gameplay. Both count toward the cap, so the next
attempt is 07 with 18 remaining. Neither can count as evidence for the new
candidates.

**Tunnel restored and attachments preserved — 2026-10-09.** The operator ran
`sudo launchctl kickstart -k system/com.trueskate.remotexpc-tunnel` on the rig.
The registry then returned XR1 (HTTP 200, USB, 69 services). XR2 was not
registered and is not used here. All 12 XR1 attachments were copied with
`devicectl device copy from` to
`…/tmp/recovered-xctest-attachments-20261009/`; `inventory.json` there records
each file's size, duration and SHA-256. The five experiment attachments match
the retained trial 02–06 originals byte for byte. The seven `xr-control-4380162`
recordings (51.7–62.4 s; 64.8–74.4 MB) exist only in this preserved copy. With
WDA and Appium idle and `/wda/video` null, the cleanup wrapper's dry-run listed
exactly these 12 UUIDs. Deletion is pending operator permission.
The successor is staged at `…/tmp/kickflip-repeatability-cd63e203` (source
`cd63e203`). It was prepared offline: 6 prior attempts carried, 11 candidates.

## Trial 07 and operator-described kickflip — 2026-10-09

Source `cd63e203`, rig root `…/tmp/kickflip-repeatability-cd63e203`. **Trial 07**
(mined centre recipe, flick gap after pop return) ran all nine commands within
25 ms of their slots. WDA submission gaps were push→pop 0.930 s and pop→flick
1.300 s, and the pop call took 656 ms. The source video shows a **360 FLIP**
(score 280) followed by `PRIMO SLIDE — FAILED`; the board landed graphic side
up. Close frames show the mined "pop" stroke entirely off the board. Its trail
runs right of and below the tail, from (0.90, 0.63) down then left. The board
neither pitches nor gains speed (6 mph). The second stroke starts on the tail at
(0.54, 0.65) and alone pops and flips the board; speed jumps to 14 mph. In this
camera and stance the mined recipe is a single-stroke 360 flip, so trial 05's
landed 360 flip is consistent with the same mechanism.

The operator described a kickflip:
1. pop: straight down from the tail tip, about 150 logical points, as a
   ~0.75 s hold followed immediately by a fast southward move;
2. flick: fast, (0.5, 0.5) → (0.8, 0.5);
3. catch: stationary hold at (0.5, 0.5) after the board has flipped.

These strokes need roughly 0.1 s spacing, but separate WDA requests cannot start
closer than about 1 s apart. The repository's trick executor
(`execute_n_slot_gestures`, used by CMA-ES and Model 2 inference) bundles closely
timed strokes into one W3C request with one finger per stroke. This candidate
set does the same. The push remains a separate request, followed by the
historical 0.48 s wait after it returns. Each finger re-issues its start move
after a leading pause, as the executor does, because WDA drops a zero-duration
move followed by a pause.

Centre recipe, t = 0 at pop touch-down: pop from (0.50, 0.67), hold 0.75 s, then
a 0.10 s move 150 points down. That move is split into four accelerating
segments (easing power 0.5; encoded 50/20/15/13 ms, lift at 0.848 s). The flick
starts 0.05 s after the pop lifts and lasts 0.06 s. The catch holds 0.30 s,
starting 0.25 s after the flick. Eleven variants each change one uncertain
number: pop length 100/190 points (190 is the longest clear of the bottom bar),
pop move 0.06/0.16 s, flick gap 0/0.12 s, flick 0.04/0.10 s, catch delay
0.15/0.40 s, and pop hold 0.40 s. The two-of-three landed-kickflip rule, QuickTime
review, admission gates and the 24-attempt cap are unchanged. The successor
carries trials 01–07 (7 attempts, 17 remaining).

**Trial 08** (`6a733550`, operator centre recipe) ran all eight commands within
19 ms of their slots; the trick request took 2,006 ms. The board never left the
ground. From the start of the pop hold, the video shows a second touch glow near
(0.5, 0.5) on the front half of the board as well as the tail touch. That glow
moves right with the flick and remains as the catch.

**WDA phantom touch.** `FBW3CActionsSynthesizer` at `ae50404a` builds touches
per finger as follows:
- a `pointerMove` with no current touch starts one at the move's end offset;
- a `pointerDown` is ignored only at index 1 after a move;
- any other `pointerDown` starts a new touch.

The delayed fingers used `move(0) → pause → move(0) → down`, so each created a
touch at t = 0 that was never lifted, plus the intended touch. The flick finger
therefore pinned the board while the tail popped. Delayed fingers now hover to
their start point for the whole wait, `move(duration = down time) → down`, which
yields exactly one touch. A test replays WDA's rules on every finger.
**Wider impact (not changed here):** `_build_spin_hold_finger` and the
`execute_n_slot_gestures` combined branch use the same leading
move/pause/move/down pattern. This likely adds an unlabelled spin-button or
start-point touch from payload start in spin and combined trick payloads.
Verify on device before relying on those labels.

**Trial 09** (`27b4734c`, single-touch fingers) ran all eight commands within
15 ms; the trick request took 1,705 ms. Only the tail touch is visible during
the hold, and the flick and catch arrive on schedule. The 0.75 s tail hold braked
the board from 4 to 0 mph. The pop then produced only a small hop; the flick
rolled the board onto its edge and it dropped back onto its wheels at an angle.
No trick banner appeared.

**Pop revision (operator).** The hold is not a separate gesture. The pop is one
stroke that starts very slowly and accelerates roughly exponentially
("ppppOP", like a board's snap). The pop now moves along 16 equal-distance
steps from the tail tip. Their timing follows position
(e^(k·t/T) − 1)/(e^k − 1), with centre T = 0.8 s, k = 6 and 150 points: the
first step takes 435 ms and the last steps 8–9 ms, ending at ≈1,100 pt/s.
Millisecond truncation lifts the finger at 0.791 s. The flick and catch timing
is unchanged relative to the lift. Variants: k = 4/8, T = 0.5/1.1 s, length
100/190 points, flick gap 0/0.12 s, flick 0.04 s, catch delay 0.15/0.40 s.

**Trial 10** (`3fc10e10`, exponential pop, centre) ran all eight commands within
13 ms. The slow creep held the board at 4 mph, unlike the separate hold; speed
fell to about 1 mph only at the snap. The outcome matched trial 09: a small hop,
the flick rolled the board onto its edge, and no banner. The operator chose a
quicker 0.5 s pop with a 0.5 s pop-to-flick gap, added as `candidate_13`;
existing candidate IDs are unchanged.

**Trial 11** (`36aac272`, `candidate_13`: 0.5 s pop, 0.5 s pop-to-flick gap) ran
the planned payload; payload SHA and hover durations match. iOS nevertheless
reported completion 1.080 s after submission for a payload ending at 1.602 s.
In the video, the flick trail starts 0.067 s after the pop trail appears, so the
0.5 s gap was not played. Simple gestures complete about 0.23 s after their
encoded duration. On that basis, trials 09, 10 and 11 played ≈1.23, 1.23 and
0.85 s against 1.51, 1.45 and 1.60 s planned. The shortfall roughly matches
their finger-free intervals (0.30, 0.30 and 0.75 s).

**Finding: within one WDA actions request, intervals with no finger down are
collapsed; holds with a finger down keep real time.** This likely explains the
"joined gestures" of bundled run04b. It also implies that the
`execute_n_slot_gestures` combined branch has not honoured positive inter-stroke
gaps. Separate requests and the Pico, which keeps its own clock, are unaffected.

**Operator reference.** The operator filmed a landed KICKFLIP (score 99) at
`tmp/Example Kickflip.MP4`: 750×1624 at 60 fps, the same aspect ratio as XR, with
the board at x 0.40–0.60 and y 0.44–0.70 as on XR1. Glow tracking gives these
normalized positions, with t = 0 at pop touch-down:
- pop (0.505, 0.56) → (0.555, 0.815) over 0.075 s (~3,000 pt/s, near-constant
  speed, starting mid-board; no hold or creep visible);
- 0.13 s with no finger down;
- flick (0.537, 0.51) → (0.715, 0.565) over 0.10 s;
- 0.43 s with no finger down;
- catch: hold at (0.50, 0.505) for 0.52 s.

The operator asked for straight-line gestures only. The candidate set reproduces
these strokes as single constant-speed moves. A stationary keep-alive finger at
off-board (0.85, 0.80) spans the whole trick, so no finger-free interval exists
for WDA to collapse; trial 07 showed off-board touches have no visible effect.
Variants: pop 0.05/0.10 s, flick gap 0.08/0.18 s, flick 0.06/0.14 s, catch gap
0.30/0.55 s. The historical push is unchanged. The operator's own push was a
~0.12 s diagonal stroke (0.81, 0.27) → (0.74, 0.52), ending 0.37 s before the pop.

**Trial 12** (`2dfb06f8`, filmed strokes with an off-board keep-alive finger)
played in real time: iOS completion came 1.492 s after submission for a 1.255 s
payload. The pop worked (4 → 13 mph), and the game scored **TRICK COMPLETED:
OLLIE**, score 38, also completing the park's Ollie challenge. Trails were drawn
from the keep-alive point to the other strokes, and the flick rolled the board
only partway.

## Correction — attempts 08–12 repeated documented results (2026-10-09)

The operator identified trial 12 as an execution bug. Attempts 08–12 re-tested
executors already rejected in
[CURVE-AUDIT-20261003](CURVE-AUDIT-20261003.md) and GESTURES.md's **one touch
contact per request** hard rule (May 2026, archived
`wda_trick_gestures_plan.md`):
- multi-path records run as parallel finger tracks and are drawn as one conjoined
  chain;
- distinct-path records drop idle time, which I misreported as a "new finding";
- hover moves emit real touches;
- an anchor (keep-alive) finger gains trail lines whenever another finger lifts;
- on-device scheduled separate records are rejected ("only one gesture can be
  performed at a time");
- separate requests cannot start within ~0.25 s of the previous gesture's end.

The earlier "wider impact" note on `_build_spin_hold_finger` is withdrawn as
unverified: GESTURES.md treats the spin hold as a legitimate simultaneous
contact, and whether its leading pause starts the hold early is untested.
Trial 07's speculation is unaffected.

**Revised procedure (operator: no attempt cap; keep iterating until the kickflip
is reproduced).** `SETUP_LIMIT` is raised to 200. Pop, flick and catch are each
sent as **one straight contact in its own XCTest record** via WDA's direct
`/wda/perform_trick_gestures` endpoint, which avoids W3C preparation (~0.10 s)
and the stability wait (~0.07 s) measured on trial 12. The flick is sent as soon
as the pop returns, so the expected gap is ~0.25–0.3 s against the filmed
0.13 s; this is the XCTest floor. The catch waits for the remaining filmed gap.
There are no guards between strokes; full guards surround the sequence, and
admission still checks every native frame. The direct endpoint is not
timing-instrumented, so its strokes are marked `instrumented=false`, and their
video times are estimated from the rig clock plus the median rig-to-WDA offset of
the instrumented requests. Calibration still uses the instrumented centre
controls. `run-setup --defer-admission` plus `admit` lets the slow native-frame
admission run on the laptop against the SHA-bound original.

**Trials 13–14** (`f089b174`, separate direct records, laptop admission). Both
showed clean, separate pop and flick trails, with no conjoining. Trial 13
(0.10 s flick): the flick call started 0.331 s after the pop call; the board
flipped about halfway and landed upside down, with no banner. Trial 14
(`candidate_04`, 0.06 s flick): the pop request took 426 ms rather than 331 ms
(WDA jitter), so the flick started 0.427 s after the pop. The board completed a
flip and landed: **HARD FLIP, score 130**. The next candidates keep the 0.06 s or
faster flick and remove its downward slant (a likely shove-it input).

**Trials 15–18.**
- Trial 15 (filmed pop, horizontal 0.06 s flick): KICKFLIP, then FAILED;
  under-rotated, landed upside down.
- Trial 16 (short quick pop to y 0.73 in 0.05 s, horizontal 0.06 s flick):
  HARD FLIP, score 91, landed.
- Trials 17 and 18 (`candidate_10`: filmed pop, horizontal 0.04 s flick, flick
  ~0.43 s after the pop): **KICKFLIP**, scores 85 and 98, landed.

The operator rates 17 and 18 as kickflips of the "rocket flip" kind: the board
flips while pitched nose-up rather than levelled. In the operator's video, the
flick finger first slides ~50 ms up the deck toward the nose, which is the
levelling motion; a later flick also meets a more nose-up board. Added straight
0.04 s flicks angled up the deck (end y 0.46 and 0.42).

**Correction and three-point flicks.** Frame tracking of the operator's flick
shows a near-stationary press on the deck (≈3 pt drift over ~50 ms), not a slide
up the deck, followed by a ~50 ms snap right and slightly down. The operator now
allows two- or three-point flicks. `direct_stroke` takes up to three points with
one duration per segment. `candidate_16`–`18` press for 50 or 80 ms, then snap
horizontally (40 ms) or along the filmed slant (50 ms). Two-point payloads are
unchanged.

**Trial 19** (`candidate_10`, third attempt): KICKFLIP, score 50, landed, then
"Line Ended". `candidate_10` is therefore 3/3 landed KICKFLIPs (85, 98, 50) and
qualifies for review; the operator calls these rocket flips. The operator adds that
their flick follows a curved, downward-ish arc before going straight outward.
`candidate_19`–`21` use a slower down-right first segment (40–60 ms), then a fast
outward segment (30–40 ms).

**Push replaced (operator).** The historical push lasts 20 ms. LINEAR-LENGTH-20261003
found visible traces in only 7/15 gestures at 20 ms and no board response at
10 ms. The operator reports it appears as a flicker and did not move the board
in trial 19. Candidates now use the operator's filmed push as a straight direct
stroke: (0.813, 0.27) → (0.74, 0.522), 228 pt over 0.13 s. The pop follows
0.37 s after the push ends, as filmed. With ~0.25 s endpoint return, the runner
waits a further `pop_wait_s` = 0.12 s with no guard; the full guard runs
immediately before the push. All gameplay strokes are now uninstrumented direct
records. Calibration, resets and controls remain instrumented W3C requests. The
unused W3C push builder was removed. Trials 01–20 used the 20 ms push.
