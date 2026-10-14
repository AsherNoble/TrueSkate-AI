# LEARN-REPLAY-PILOT-20261013 — prospective small learning/replay diagnostic

Status: **Gate 1 go (operator, 2026-10-14); XR1 WDA rebuilt under a new bundle ID
after a tunnel/WDA outage; context bound; capture session `pilot-00` started
2026-10-14 01:34 UTC.** The original prepared record follows unchanged.
The operator authorized the parallel experiments and dependent stages on
2026-10-13. This is an isolated diagnostic, with no production qualification,
bulk collection, paid training or human-demonstration claim.

## Dependency order

1. Independently log XR1 UIKit touch delivery and measure repeatability on the
   frozen physical cohort, then record the representative True Skate replays.
   The missing separately signed touch logger currently blocks this step.
2. Review that measurement before collecting the pilot. Proceed only if the
   contact stream is complete, the mapping is repeatable enough for meaningful
   paired comparisons, and the required gameplay/recording guards pass. Report
   the full position and boundary spread; do not infer a universal frame-level
   label tolerance from simulator determinism. Any numeric go/no-go thresholds
   must be frozen before the first physical request, alongside the physical
   experiment; a failed measurement cannot be repaired by changing a threshold.
   Those pragmatic criteria are now frozen in the physical plan, digest
   `0c02c970e82d754fc90903dddc92431b678131b5aad9edb71f422a084a1970ed`.
   Its current gate is closed and unmeasured; threshold fields describe required
   evidence, not completed physical observations.
3. Bind a verified current park, waypoint, settings, device/iOS/WDA versions and
   isolated capture fingerprint to the frozen command plan before dispatch.
   Unknown context remains unknown; stale heartbeats or old screenshots are
   insufficient. Build an explicitly research-only capture/manifest adapter;
   do not manufacture a production-qualified ledger or relabel pilot samples as
   production. Existing recording admission and calibration checks apply.
4. Capture at most 600 candidate commands, nominally 600 new clips, in twelve
   separately recorded sessions. The frozen whole-session partition is 400
   training, 100 validation and 100 test. Each session has 25 arc and 25 random
   chain commands. One-minute segments, start/middle/end calibration, all-frame
   menu/editor checks, native 120-frame windows, foreground/tunnel/attachment/
   disk checks and failure caps remain unchanged. Rejections consume the budget;
   no automatic replacement, retry or silent cohort shrink. An incomplete
   nominal corpus stops this pilot for diagnosis. Use exactly five one-minute
   recordings per session, ten candidate slots per recording. The current
   collector's frozen slot constants provide three calibration markers, two
   in-recording resets and one pre-roll reset per minute: at most 360 support
   requests across 60 recordings, 960 total capture touch requests. A session is
   a distinct recorded block; no source recording crosses a split. Twelve
   contiguous blocks do not establish twelve independent game populations.
5. Freeze all media/metadata/payload hashes and session separation before
   training. Labels are exact quantized compiled commands on the calibrated
   command clock, with native PTS, not reconstructed trail heads or UIKit
   coordinates substituted for commands. Preserve raw recordings and traces.
6. Train the existing curved architecture locally with its frozen recipe:
   channels 16, batch8, 40 epochs, AdamW lr 0.001/weight decay 0.01, seed 0.
   Checkpoint every epoch and select using validation only. No real training has
   run; local CPU availability is verified, but full training duration is not
   estimated. Keep checkpoint compatibility and record the actual implementation
   hash. The production `train` CLI intentionally rejects this smaller corpus;
   a separate research adapter must use the unchanged core training engine.
7. Expose the test split once. Use the existing decoder as primary. A coordinate
   refit remains an offline paired diagnostic until its robustness results
   justify another explicit experiment; do not select the decoder on this test.
   Compile every prediction through the full curved bounds, duration, segment
   and protected-control contract. Invalid predictions are reported failures,
   never clipped, repaired or dispatched silently.
8. Replay the 20 prospectively selected test commands: five original and five
   predicted requests each, frozen randomized order, at most 200 scientific
   requests. Persist spending before every RPC, including timeouts. No retry of
   an unknown outcome. Use matched reset/context, calibration and fresh capture;
   reverify context each session and report residual board-state variation.
   Reset/settle does not prove identical game state. Use twenty one-minute
   recordings with the same ten-slot/three-marker/three-reset schedule: at most
   120 support requests and 320 total replay touch requests. Invalid predictions
   leave their slots empty and counted; never replace them or move another job
   into that slot. Both capture and replay stop on the first recording failure.

## Outcomes and analysis

Report existing command-recovery metrics, invalid predictions, admission
failures and all 200 request outcomes. Compare predicted replay against the
five-repeat original-command distribution, with clip-level paired effects and
intervals over the 20 reference commands. Report full XY/time/contact metrics
where independently observable. Keep UIKit delivery, glow/trail rendering and
game state/outcome similarity separate. Holds and release cannot be assigned
from glow position alone; ambiguous gameplay frames remain unmeasurable.

This can diagnose whether the network learns a useful command inverse through
the same executor. It cannot certify human-video transfer, unseen parks,
unique command recovery or production readiness. Round 6 remains 83 Meet/16 Fail/
1 Unclear, and its qualification/dispatch ledger is untouched.
Four hundred training clips may be underpowered. Report epoch training loss
and validation learning curves; a negative result cannot refute learnability
of a deterministic executor warp.

## Prepared artifacts and current blocker

[`prepare.py`](../evidence/LEARN-REPLAY-PILOT-20261013/prepare.py) deterministically
generates 600 legal commands and the 200-request replay order without accessing
a device. The selected 20 test IDs are frozen before any media or model output.
The generated compressed plan, source and hash are preserved with this record.
It has no live dispatch command and no production manifest compatibility.
Plan digest: `f06fce86b507533e64b84520fb6c6719f0d9df99bd00c36913390a526bb1be6c`.
Its five synthetic cohort/mutation checks pass; this is preparation validation,
not evidence of learning or physical execution.

Physical preflight on 2026-10-13 found XR1 healthy on iOS 18.7.6 but no installed
touch logger. The rig has zero valid signing identities and only WDA-scoped
profiles; laptop signing identities report revocation and no usable profiles.
Reusing the runner identity would disturb healthy WDA. Therefore zero physical
scientific requests, zero pilot clips, zero training epochs and zero pilot replay
requests have been executed. See
[PHYSICAL-TRANSFER-20261013](PHYSICAL-TRANSFER-20261013.md) for the measured
preflight and unsigned helper. A separate valid helper identity/profile is the
required external prerequisite; WDA is not restarted or rebuilt.

## Gate 1 and capture tooling, 2026-10-14

The physical measurement is recorded in
[PHYSICAL-TRANSFER-20261013](PHYSICAL-TRANSFER-20261013.md). The frozen helper gate
failed only on hold observability, the operator-approved post-hoc hold check passed
30/30, and Gate 4 passed at its one-frame limit. The operator chose **go** on XR1,
with these conditions ([decision](../evidence/LEARN-REPLAY-PILOT-20261013/gate1-decision.json)):

- Preparation starts 2.25 s ahead of each slot instead of 1.75 s, set before any
  pilot touch. Game attempt 4 needed up to 1.58 s. A later start cannot shorten any
  margin, and admission is unchanged.
- The screenshot check before each recording's first marker is skipped, as in
  game attempts 3–4. The foreground guard still precedes every touch.
- XR2 stays out: its delivery is unmeasured and its WDA is not running.
- The training venue is chosen after capture, from a measured local epoch and a
  Modal quote.

[`bind_context.py`](../evidence/LEARN-REPLAY-PILOT-20261013/bind_context.py) writes a
new bound file, read-only on the rig. It pins the plan digest, the operator's verbatim
context, device/iOS/True Skate/WDA versions, the schedule, the curved pipeline
fingerprint and the runner hashes. [`capture_pilot.py`](../evidence/LEARN-REPLAY-PILOT-20261013/capture_pilot.py)
captures one session (five recordings) per invocation through the game cohort's
path. Sessions run in plan order and spending is written before every RPC. It
refuses a rerun, an out-of-order session or a budget overrun. A recording failure
or any rejected candidate stops the pilot for diagnosis (Gate 2). At the measured
rate of about 13.5 min of extraction per recording, capture needs about 15 h of rig
time. Its 13 synthetic checks pass with the 5 preparation checks.

**Rig incident.** At 2026-10-13 13:32:52 UTC, about three minutes after the game
cohort's last clip was settled, the root RemoteXPC tunnel logged `SSL read failed in
device-to-tun loop` and removed XR1's tunnel. The tunnel service kept running with
zero tunnels. The services supervisor relaunched WDA once from the existing build
(`test-without-building`, timing revision unchanged). It then reported WDA wedged
and looped through more than 120 restart attempts, while XR1 remained visible on
USB and to CoreDevice. The cause of the tunnel failure is unknown. No pilot touch
has been sent.

The operator restarted the tunnel service; XR1's tunnel returned, but WDA did not.
The runner's 7-day profile (`com.asher.WebDriverAgentRunner.xctrunner`, team
`BK75JV5YK4`) expired on 2026-10-10, and the healthy runner had survived only
because it was never relaunched. The supervisor's first restart ended it. Every
relaunch since fails with `0xe8008011 (This provisioning profile has expired.)`. The
supervisor's headless `build-for-testing` fallback exited 65: Xcode could not
register that app identifier for the team and found no profile for it. The operator
chose to rebuild WDA at `ae50404a` with the timing-revision define, the 2026-10-07
procedure, from the rig's desktop Terminal using
[`rebuild_wda.sh`](../evidence/LEARN-REPLAY-PILOT-20261013/rebuild_wda.sh). The
script pauses and restores the supervisor, which then installs and launches the build.

The first rebuild failed the same way: Apple would not register
`com.asher.WebDriverAgentRunner` for the operator's only team, the free Personal Team
`BK75JV5YK4` that had issued the expired profiles. With the operator's approval, the
expired runner was uninstalled from XR1, keeping it within the free three-app limit.
WDA was then rebuilt at `ae50404a` with the timing define and the command-line
override `PRODUCT_BUNDLE_IDENTIFIER=com.ashernoble.WebDriverAgentRunner`. No source
changed. The first attempt at that build ran full-width and starved WindowServer into
a watchdog logout of the desktop session (the rig did not reboot; tunnel and supervisor
recovered). The script now builds with two low-priority jobs. The second attempt
succeeded: profile `com.ashernoble.WebDriverAgentRunner.xctrunner`, two devices,
expiring **2026-10-21 00:57 UTC**. The supervisor installed and launched it, and WDA
reported `build_revision` `ae50404a`. Appium attaches by URL, so the pilot path does
not depend on the bundle ID. Only the XR2 Wi-Fi diagnostic script names the old one.

The operator confirmed XR1 on Macba after the tunnel restart. The settings note
`none` dates from 2026-10-13 and was not restated. The operator set a still wallpaper
and disconnected Screen Sharing. The read-only binding
([`bound-context.json`](../evidence/LEARN-REPLAY-PILOT-20261013/bound-context.json),
digest `ce6caa38…`) records XR1 on iOS 18.7.6 (22H320), True Skate 1.6.11, WDA
`ae50404a`, the 2.25 s schedule and runner hashes, which match commit `fd2f6ee6`.
