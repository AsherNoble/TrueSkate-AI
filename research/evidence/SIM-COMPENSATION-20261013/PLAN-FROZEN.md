# Prospective simulator compensation — frozen before measurement

2026-10-13. Base `dc159db0`; isolated branch `codex/sim-compensation-20261013`.
User authorization: bounded simulator compensation, no production or physical changes.

`FROZEN.json` fixes all desired paths, nominal and adjusted exact payloads, four
development / sixteen held-out commands, algorithm hash, scoring constants and
randomization seed 2026101303 before any action request. These geometries and
interval combinations are newly constructed; previous 1,090 simulator requests
are development evidence only. No held-out outcome can change the candidate.

Candidate `legal_ceil_clock_ls_v1` enumerates ≤10 legal intervals, each ≥100 ms,
total 300–1200 ms. It can subdivide longer moving legs, chooses integer request
durations from a previous ceil-to-60-Hz scheduling approximation, and least-squares
fits added coordinates along each original straight leg. Mandatory original
corners, stationary endpoints and first/last points remain fixed. It selects by
predicted normalized spatial RMS plus release and semantic-boundary penalties.
The model is approximate: old ceiling-duration results do not establish full
scheduling, and exact 100 ms intervals have known exceptions. All emitted integer
coordinates pass unchanged `Gesture` and `TimedWaypoints.payload()` safety and
schema checks. No production compiler change or W3C pause substitution is used.
An unchanged payload is labeled a constrained fallback, not a compensation gain.

The 16 held-out cases include straight legs at 317/463/739 ms, sharp corners,
unequal 109–357 ms intervals, initial/middle/final holds, all hold placements in
one contact, and stress cases with three or ten minimum-length intervals. The
stress cases expose contract limits, not an inferred physical-device limit.

Requests: 20 commands ×2 variants ×4 repeats =160 measurement requests maximum,
randomized together; no automatic retry. Up to10 separately recorded setup
smokes allowed (planned two identical400ms controls). A ledger entry is flushed
before each attempted request. Any exception, missing begin/end, cancellation,
HTTP error, or source/safety mismatch stops the run without replacement.

Scores use raw touch timestamps relative to delivered began, with coalesced
points retained; callback arrival is reported separately. Observed positions are
linearly interpolated only for the fixed 60Hz comparison grid; this is an analysis
choice, not a claim about a game's renderer. The desired path is its frozen
piecewise-linear integer path. The grid covers the longer of desired/delivered
contact duration, holding endpoint position outside motion; release/contact
mismatch is scored separately so endpoint padding cannot hide lift error.
Waypoint arrival is earliest sample within1point of the desired waypoint near
that segment. Hold onset/departure use contiguous residence within1point; omitted
stationary callbacks imply up to a frame of boundary uncertainty, reflected in
the frozen17.667ms timing threshold. Raw callbacks/phases remain available.

Per-case full-effect gate on four-repeat median: normalized spatial RMS≤.005,
maximum≤.01, all waypoint/hold/release absolute errors≤17.667ms. Overall success
requires all16 held-out cases passing and at least25% reduction in command-weighted
mean RMS relative to nominal. Improvements limited to some trajectories are
reported as partial, not a successful general inverse. These extra diagnostics
do not relax the existing collection gates or establish game equivalence.

Restore the initial simulator state (dedicated L0-XR-iOS18.6 was Shutdown on
entry), stop owned WDA/logger/build processes, retain all setup attempts and raw
logs in ignored worktree tmp, and commit compact evidence plus this experiment
record only. No rig, SSH, phone, paid service, model training, push or merge.
