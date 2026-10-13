# LEARN-REPLAY-PILOT-20261013 — prospective small learning/replay diagnostic

Status: prepared; execution blocked by the physical repeatability prerequisite.
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
