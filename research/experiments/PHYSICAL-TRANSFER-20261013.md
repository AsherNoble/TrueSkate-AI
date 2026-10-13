# PHYSICAL-TRANSFER-20261013 — physical XR input repeatability

Status: **helper cohort executed 2026-10-13; frozen helper gate failed on hold
observability only; operator-approved post-hoc hold amendment passes; game cohort
stopped in three attempts (schedule overruns; 1 of 20 scientific requests spent)**.
The original pre-execution record follows unchanged, then the helper result.
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

## Helper cohort, 2026-10-13

The operator signed the helper under the free Personal Team (`BK75JV5YK4`, fallback
bundle `com.ashernoble.touchlogger.physical20261013`; the first profile covered only
XR2 until XR1 was selected in Xcode). It was installed on XR1 without touching WDA.
A first launch attempt was refused (developer not yet trusted; **0 touches spent**).
After the operator trusted it, all **80/80 frozen requests completed**, in order,
spend-before-RPC, with no retry. Directory copy of the logs timed out; exact-path
copy retrieved the single launch log (SHA `43d6c039…`), which holds exactly 80
batches, all bound to their own request windows. WDA stayed healthy throughout.

**Frozen helper gate: failed.** All 80 contacts are complete, uncancelled and
unoverflowed; every waypoint was observed within 0 pt; position and release
criteria pass on every path. The only failures are 6/30 hold runs whose hold
occupancy spanned 0 ms (4/10 `initial_hold_600`) or 8.3 ms (2/10 `middle_hold_800`),
below the frozen 50 ms. Those runs lack UIKit's stationary end-of-hold sample; their
post-hold sample positions equal the path's modal run, with timestamps within
0.85 ms. This is a sampling/observability difference, not different delivered
motion, but the frozen rule counts unobservable as failure and is not re-thresholded.

| Path | Requested ms | Delivered ms (SD) | p95 / worst normalised repeat deviation | Max spatial SD pt |
|---|---:|---:|---:|---:|
| single_400 | 400 | 400.0 (0.00) | 0.0002 / 0.0010 | 0.20 |
| single_800 | 800 | 800.0 (0.00) | 0.0001 / 0.0005 | 0.08 |
| split_800 | 800 | 808.3 (0.00) | 0.0001 / 0.0004 | 0.09 |
| corner_600 | 600 | 600.0 (0.00) | 0.0002 / 0.0008 | 0.26 |
| unequal_600 | 600 | 600.0 (0.00) | 0.0002 / 0.0017 | 0.21 |
| initial_hold_600 | 600 | 608.3 (0.00) | 0.0035 / 0.0054 | 1.51 |
| middle_hold_800 | 800 | 816.7 (0.00) | 0.0004 / 0.0028 | 0.69 |
| final_hold_600 | 600 | 600.0 (0.13) | 0.0002 / 0.0010 | 0.19 |

Gate limits: p95 ≤0.01, worst ≤0.03, release p95 deviation ≤8.333 ms (observed
≤0.25 ms). Durations sit on the 8.33 ms digitizer grid; three hold/split paths gain
a fixed +8.3 or +16.7 ms. Each path departs from the linearly time-parameterised
command by a repeatable 4.8–9.9 pt (median per-run maximum), consistent with a
deterministic delivery warp. UIKit touch-began minus WDA `submitted_to_ios` (phone
clocks) is 12.75 ms median, SD 1.94 ms, p05–p95 10.6–15.5 ms, maximum 24.3 ms.
Ten repeats per path do not bound rare failures; no True Skate rendering, game
timing or learnability is established by the helper.

Evidence: [helper cohort](../evidence/PHYSICAL-TRANSFER-20261013/helper-cohort-20261013T121131Z/)
(redacted receipts, spend log, compressed touch log, [summary](../evidence/PHYSICAL-TRANSFER-20261013/helper-cohort-20261013T121131Z/summary.json),
[provenance](../evidence/PHYSICAL-TRANSFER-20261013/helper-cohort-20261013T121131Z/provenance.json)).
Runner `run_helper.py` at `f311c399` dispatched; `6d3b3ebc --finish` retrieved and
analysed. The game cohort and learning/replay pilot remain gated.

### Post-hoc hold amendment (operator-approved 2026-10-13)

After seeing the failure, the operator approved replacing only the hold check, so
the **frozen gate remains the decision of record (failed)**.
[`amend_holds.py`](../evidence/PHYSICAL-TRANSFER-20261013/amend_holds.py) passes a
hold if its point is observed, no sample in the commanded hold window is over 0.5 pt
from it, and post-hold motion (samples after the window that have left the hold
point) equals the path's modal run within 1 ms. Two implementation choices were made
after seeing the data. A first draft wrongly counted the stationary hold-point
sample as motion. Motion compares distinct consecutive positions because one run
(`helper-013`) carries an extra stationary endpoint sample. **Amended gate: passed,
30/30 holds.** Under the literal sample-by-sample wording, `helper-013` alone fails.
See [`helper-gate-amended.json`](../evidence/PHYSICAL-TRANSFER-20261013/helper-cohort-20261013T121131Z/helper-gate-amended.json).

## Game cohort attempts, 2026-10-13

Context, verbatim from the operator: park `Macba`; settings note `none`. True Skate
waypoints are not numbered: each park has a few built-in waypoints and reset returns
the board to the nearest one. The run records the waypoint as that description.
The "waypoint 1–5" request came from the curved CLI and collector, which require an
integer 1–5 (`scripts/collection/curved_model1.py`, `collection.py`); that is an
agent invention, not an operator convention.

Both attempts stopped with `preparation overrun` at the first control slot (1.5 s),
before any calibration or scientific touch: one pre-roll reset each, **2 support
touches total, 0 of 20 scientific**. The operator approved the single retry. The
recorder took 1.40 s and 0.85 s from host start to its own start timestamp, against
a rig median of 0.067 s (p90 0.258 s, previous maximum 0.672 s, n=307). At Macba the
pre-touch screenshot check already takes 1.07 s median (1.25 s maximum, n=3 curved
recordings, one of which hit the same error), so the inherited production schedule
leaves about 0.4 s for recorder start. Recorder idle, zero attachments, True Skate
frontmost and the lock released were verified afterwards. No further attempt was made.
Evidence: [attempts](../evidence/PHYSICAL-TRANSFER-20261013/game-cohort-attempts-20261013/summary.json).

A third attempt (operator-approved, skipping only the redundant screenshot check
before the first marker) delivered the start marker and `game-001`, then stopped
with `schedule overrun after foreground guard` at the 9 s slot: sample-slot
preparation took 1.71 s and 1.78 s against the 1.75 s lead (Macba round-6 median
1.07 s). Spent across attempts: 3 pre-roll resets, 1 marker, **1 scientific request
(`game-001`, no clip)**. Raw WDA screenshots took 0.40–0.43 s; the host-side menu
and editor checks took 0.41–0.60 s and 0.47–0.50 s. An active Screen Sharing session
(ScreensharingAgent ~45% CPU, WindowServer 26–29%, video wallpaper 21–29%) is the
leading, untested explanation. The rig was verified clean afterwards.
