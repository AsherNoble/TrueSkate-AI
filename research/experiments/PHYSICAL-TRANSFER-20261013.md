# PHYSICAL-TRANSFER-20261013 — physical XR input repeatability

Status: **blocked before execution; zero touch requests and recordings**.
The operator authorized the parallel physical, decoder and simulator diagnostics
on 2026-10-13. The physical prerequisite cannot currently be measured without a
separately signed UIKit logger. Healthy WDA remains running and collection is off.

Read-only preflight verified XR1 is physical iPhone XR on iOS 18.7.6/build 22H320,
True Skate 1.6.11 frontmost, the root RemoteXPC tunnel running, recorder idle and
collector/watchdog jobs disabled. The rig release is clean 189dfded0e1de31794d7dd9b88773dd3abfc8ba8.
The current park/waypoint/settings remain unknown. The available Clapperboard
source is a screen toggle, not an independently timed touch logger.

The rig has zero valid signing identities. Its existing provisioning profiles
permit only WDA-specific app IDs. Both laptop signing identities carry revocation
annotations and no local provisioning profile is available. Reusing a WDA app ID
or restarting/re-signing its healthy runner would violate the experiment's
constraints; neither happened. The laptop's existing Tailscale connection was
restored without resetting its preferences.

The fixed future workload has 80 helper requests (eight safe trajectories × 10)
and 20 game requests (four representative trajectories × 5), plus six calibration
and six reset requests: **112 total, no automatic replacements**. It imports the
current guard-aware recording schedule. Its digest is
`0c02c970e82d754fc90903dddc92431b678131b5aad9edb71f422a084a1970ed`.
Pragmatic position/release/observability pilot gates are frozen before any request;
they do not modify curved qualification or establish universal label tolerances.

An isolated buffered UIKit logger compiles cleanly to unsigned arm64. It retains
native/coalesced positions and touch timestamps, callback clocks and scheduled
display callbacks separately. It has not been signed, installed or launched.
The analyzer's 13 synthetic checks pass, including timing-clock separation,
coalescing/duplication, cancellation/overflow and incomplete-cohort gate closure.
No physical transformation, deterministic-warp transfer or repeatability result
is claimed. The dependent learning/replay pilot remains blocked.

See the [full report](../evidence/PHYSICAL-TRANSFER-20261013/REPORT.md),
[frozen plan](../evidence/PHYSICAL-TRANSFER-20261013/frozen-plan.json),
[filtered preflight](../evidence/PHYSICAL-TRANSFER-20261013/preflight.json),
[unsigned build](../evidence/PHYSICAL-TRANSFER-20261013/helper/build.json) and
[synthetic validation](../evidence/PHYSICAL-TRANSFER-20261013/synthetic-verification.json).
Only valid separate helper signing and verified current context can unblock the
next bounded step. No device touch, recording, rig service restart, production
source change, paid compute or human notification occurred.
