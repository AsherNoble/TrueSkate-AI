# PHYSICAL-TRANSFER-20261013

Status: **physical execution blocked; zero touch requests and zero recordings**.
The operator authorized the parallel diagnostic on 2026-10-13. Read-only rig
preflight, an isolated unsigned UIKit helper, the frozen workload and synthetic
analysis validation are complete. Physical input repeatability is unmeasured;
the dependent learning/replay pilot gate remains closed.

## Measured prerequisite state

[`preflight.json`](preflight.json) was captured at
2026-10-12T23:47:50.827694Z (2026-10-13 Sydney). The device UDID was read from the
rig `.env` and matched CoreDevice hardware details; only its hash is preserved.

| Item | Observed state |
|---|---|
| XR1 | Physical iPhone XR, iOS 18.7.6, build 22H320 |
| True Skate | 1.6.11, `com.trueaxis.skate` frontmost |
| WDA | Healthy; build time Oct 2 2026 22:04:17; no session reported |
| Recorder | `/wda/video` is null; no start attempted |
| Root RemoteXPC tunnel | Running |
| Collector/watchdog jobs | Both devices' collection and watchdog jobs disabled |
| Rig release | Clean `189dfded0e1de31794d7dd9b88773dd3abfc8ba8` |
| Current park, waypoint and settings | **Unknown; not inferred from historical notes** |
| Rig signing identities | `security find-identity` reports zero valid identities |
| Installed touch logger | None identified in the complete72-app inventory |

The rig's two provisioning profiles are specifically scoped to WDA. The runner
profile expired 2026-10-10 05:12:09 UTC. The other profile expires
2026-10-14 10:13:56 UTC but permits only `com.asher.WebDriverAgentRunner`.
Neither authorizes this helper. Profile metadata was extracted without treating
it as a validated signing authorization. Healthy WDA was not rebuilt,
re-signed, replaced or restarted.

[`local-preflight.json`](local-preflight.json) preserves the laptop signing
output: both listed identities have `CSSMERR_TP_CERT_REVOKED` annotations, even
though the command's summary says "2 valid identities found". No provisioning
profile exists in the inspected local profile roots. This contradictory summary
does not establish a usable, separate helper signing identity/profile.

The installed Clapperboard app was considered. Its available sibling source is
a SwiftUI lift-triggered black/white toggle, with no raw UITouch/coalesced event
or timestamp logger. That source is not independently bound to the installed
binary; neither observation establishes a suitable existing logger. It was not
launched or modified. See [`clapperboard-inspection.json`](clapperboard-inspection.json).

The laptop's existing Tailscale profile was initially stopped. Reconnecting
using its existing `accept-routes` preference restored rig access. Preferences
were not reset, identities were not changed and no rig service was restarted.

## Frozen workload

[`frozen-plan.json`](frozen-plan.json) digest:
`0c02c970e82d754fc90903dddc92431b678131b5aad9edb71f422a084a1970ed`.
Generated with [`freeze.py`](freeze.py), it contains exact compiler payloads,
quantized coordinates, request identities and randomized balanced repeat order.
All paths meet the existing curved-v1 contract and every quantized segment
passes the protected-control map. Coordinates remain normalized; no offset is
introduced.

| Path | Times (ms) | Shape |
|---|---|---|
| single_400 | 0,400 | One diagonal leg |
| single_800 | 0,800 | Same leg, slower |
| split_800 | 0,400,800 | Same diagonal split into two equal collinear legs |
| corner_600 | 0,300,600 | Horizontal then vertical |
| unequal_600 | 0,150,600 | Same corner, unequal interval durations |
| initial_hold_600 | 0,200,600 | Initial zero-distance move, then diagonal |
| middle_hold_800 | 0,200,400,800 | Diagonal, zero-distance move, diagonal |
| final_hold_600 | 0,400,600 | Diagonal then final zero-distance move |

Helper: 80 scientific requests, ten repeats per path, XR1 only. Each sequential
contact is its own W3C request. Holds are zero-distance `pointerMove` intervals,
never W3C pauses. No retry or replacement is allowed after an unknown outcome.

True Skate: 20 scientific requests, five repeats of `single_800`, `corner_600`,
`unequal_600` and `middle_hold_800`. Two one-minute recordings each contain ten
scientific requests, three die-five calibration requests, two in-recording
resets and one pre-roll reset. **Total cap112 touch requests:** 80 helper,
20 game, six calibration and six resets. All remain unspent.

The plan imports the current guard-aware schedule from
`src/trueskate_ai/model1/curved/collection.py`, whose hash is bound in the plan:
sample slots 4.75/9/13.25/17.5/21.75/32.25/36.5/40.75/45/49.25s;
controls 1.5/29/57s; resets 26/53.5s; prepare lead 1.75s; guard lead 0.25s.
Pre-roll reset/settling happens before recording. Stop targets 59s; 60 s is the
maximum. This replaces the older audit-v3 cadence in the initial draft, before
any plan was frozen or request dispatched.

Future live execution must reuse existing foreground/menu/editor, attachment,
disk, per-recording timing identity/calibration and full native-decode checks.
One failed start/recording stops the invocation; retrieval is attempted once.
No admission threshold is relaxed to complete the workload. Support resets are
explicitly labeled and cannot become training examples. A signed helper's
identity/source/binary and the current game context must be bound before dispatch.

## Logger and clock separation

[`helper/main.m`](helper/main.m) implements an isolated UIKit app with a distinct
bundle ID. During a contact it appends fixed C records to a 16384-entry memory
buffer. At release/cancellation, a copy is handed to a serial writer for JSONL
serialization and disk flush. Overflow is explicit and invalidates the batch.
The logger retains:

- Native B/M/E/cancel callbacks, precise logical positions and raw UITouch
  timestamps, with coalesced samples stored separately and their callback IDs.
- Callback arrival `CACurrentMediaTime` and process uptime, rather than treating
  callback scheduling as input-delivery timing.
- CADisplayLink timestamp, target timestamp and arrival time while contact is
  active. These are scheduled display callbacks, **not measured photons**.
- Launch geometry, OS version, raw launch epoch/uptime and embedded source hash.

The visible dot and instrumentation may affect application work/coalescing.
Buffered IO removes per-movement disk writes; it does not prove zero observer
effect or transfer to True Skate. The helper has not been launched on any device.

Future receipts must independently retain the frozen source/path/payload,
host first/return timestamps and the existing WDA preparation/submission records.
Those clocks are not replaced by logger callbacks. Native video source PTS and
per-segment calibration belong to the game measurement. Glow is never used to
assign UITouch time, contact release or an unobserved hold boundary.

[`helper/build.json`](helper/build.json) records a successful warning-free
arm64 compile against the local iPhoneOS 26.5 SDK, deployment target 18.0.
Source SHA256:
`2a955002ba83825b1ec94501da47927a9fbf20c8bb8c5ddc4be749083e407f44`.
Unsigned executable SHA256:
`17948b3bc5ba092018cb9bda2a7689d0553945139e1f7fcb98370b3ae732202e`.
`codesign -dvv` confirms "code object is not signed at all". The app is preserved
under ignored `tmp/physical-transfer-helper-unsigned-final-20261013`; it was not
installed. Compilation is not runtime validation.

## Analysis and prospective pilot gate

[`analyze.py`](analyze.py) requires explicit unique receipt-to-batch bindings.
It rejects incomplete/cancelled/overflowed contacts, removes duplicate samples,
includes coalesced samples and flags distinct positions with the same timestamp.
It reports duration/release spread, start/end geometry, sample-limited waypoint
occupancy, callback ages and common-absolute-time repeatability. Resampling is
piecewise linear with no duration normalization; native samples remain in output.
Holds/retraces do not silently acquire exact knot timestamps from nearest points.

These **pragmatic pilot gates were frozen before any physical request**:

1. All 80 helper requests complete, with ten complete uncancelled, non-overflowed,
   conflict-free streams per path. Every required waypoint must be observed
   within 2 logical points; each intended hold needs at least two distinct touch
   timestamps spanning 50ms at its coordinate. Unobservable is not a pass.
2. Per path, the 95th percentile of normalized distance from the componentwise
   median delivered trajectory must be≤0.01, with worst deviation≤0.03 on the
   shared absolute-time span. Report individual results and coverage as well.
3. Per path, the 95th percentile absolute release-duration deviation from its
   median must be≤8.333ms. A constant delivery warp is permitted; random spread
   is what this criterion restricts.
4. All 20 game requests must pass unchanged technical admission with verified
   current context. Each family needs at least 4/5 independently assessed,
   interpretable rendered paths; rendered-onset offset range must be no greater
   than one native frame. Ambiguity, menus or missed calibration stop progress.
   Release/hold input times cannot be assigned from glow, and reset/settle does
   not establish identical board state.

These thresholds are neither established label-noise tolerances nor production
qualification. Ten repeats per family do not bound rare failures; interpolation
cannot establish behavior between native samples. The helper-only gate cannot
open the learning pilot without the separate game observation/context gate.

[`synthetic-verification.json`](synthetic-verification.json) records 13 passing
checks, including a 1.2 s callback stall leaving touch duration unchanged,
coalesced sample inclusion, duplicate removal, cancellation/overflow rejection,
same-timestamp conflicts, known 20 ms duration spread, unique batch binding,
partial-cohort gate closure and ideal 80-request helper-only gate opening.
They establish analysis behavior, not physical performance.

## Consequence and next executable step

No physical transformation, repeatability estimate, game fidelity, pilot clips,
training epochs or learned replays can be reported from this experiment.
The next external prerequisite is a valid signing identity/profile for the
separate helper, preserving the healthy WDA runner. Once available, verify that
helper and current scene/context, then execute the fixed helper cohort; only
after its gate passes execute the fixed game cohort and assess the combined gate.
Thresholds and cohorts remain fixed; failures are retained rather than replaced.

No cloud, production corpus, qualification ledger, notifications, rig checkout
source or installed service was changed. No diagnostic server/viewer was started.
[`postflight.json`](postflight.json) confirms healthy WDA, idle recorder,
True Skate frontmost, clean rig source and disabled collectors/watchdogs at the
end. [`gate.json`](gate.json) records the zero-dispatch counts and unmeasured
outcomes; [`provenance.json`](provenance.json) binds the preserved artifact hashes.

## Reproduction

Use the repository's existing `.venv`, from this worktree, with `PYTHONPATH=src`.
`freeze.py --out /absolute/new-plan.json` regenerates the workload (the source
SHA identifies the checkout); `verify_analysis.py --out /absolute/new-check.json`
reruns synthetic checks. `helper/build_unsigned.py --out /absolute/new-build-dir`
performs a local unsigned compile only. `analyze.py --manifest PLAN --log JSONL
--receipts RECEIPTS --out /absolute/new-result.json` analyzes later bound evidence.
None of these commands signs, installs, launches or dispatches a gesture.
