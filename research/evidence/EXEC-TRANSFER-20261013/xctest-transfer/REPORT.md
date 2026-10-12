# XCTest simulator transfer — preserved-result summary

This summary was completed by the coordinating agent from saved measurements
after the primary analyst was interrupted. It distinguishes observations from
inference. No physical phone or rig was operated by this continuation.

## Scope and method

The resumed study used an iPhone XR simulator on iOS 18.6, the WDA fork at
`ae50404aac12d9f8c41f6c3fa8776e97975eaef5`, and a small UIKit logger observing
`UITouch` timestamps, positions, phases and coalesced samples. The researcher
copied WDA for diagnostics rather than changing a production service. Native
compiled records were exported with a diagnostic-only patch. A second logger
buffered events in memory to test callback file-I/O effects.

- Main: 149 commands×5 repeats=745 requests, including all 100 exact round-6 payloads.
- Boundary controls: 51 commands×5 repeats=255 requests.
- Logging control: 6 commands×5 repeats×3groups=90 requests.
- Total: 1,090 requests, not1,090independent devices or sessions.

Main and boundary orders were randomized. Callback groups were synchronous
before, buffered, synchronous after. The main UIKit stream and full boundary
records are preserved as deterministic gzip JSONL. The original scripts retain
their scratch layout; `preservation.json` maps every copied file to its original
path and hash. Callback timing from WDA's256-entry timing buffer was used only
when request timestamps matched; stale cached timing was excluded.

## Source-backed easing

The local Recap runtime selects a cubic timing function with control points
`(0.2620539963245392,0.17945900559425354)` and
`(0.7582859992980957,0.8902440071105957)`. Its touch curve is used by
`_nextTouchLocationFrom:to:at:`. The copied targeted disassembly and extracted
constants support this selection; the original full binary hash is
`db82fc6a55ad90089b56207840d24013137f78ef5372b63f8fa319aef863bb0d`.

Solve the cubic's x coordinate for normalized time, then use its y coordinate
as progress. Comparing at the nearest synthesis cadence, the maximum measured
error for100/400/800/1200 ms single moves is respectively
0.153/0.240/0.246/0.247logicalpt. The broader single-move set has a0.2601 pt
maximum. This is local simulator evidence, not a fitted physical-XR law.

## Repeatability and duration

All149 main commands have identical coordinates at common rounded frame
indices across their five repeats. Raw timestamps follow approximately60 Hz
cadence (median16.667 ms; selected interval range15.475–17.861 ms). Exported
schedules explicitly round timestamps to frame indices; exact tick timing must
not be inferred from those exports. Coordinates fall on a half-point grid.

For round-6 commands, duration error spans−8.502..+101.334 ms, median22.4005 ms
across500 requests.62/100 commands have median error above one16.667 ms frame.
The largest within-command duration SD is 0.3882 ms. Maximum distance from the
command's geometric polyline is 0.3433 pt. Thus repeatable spatial geometry and
nonlinear, segment-dependent time delivery coexist. Matching total duration
would not establish matching intermediate positions or waypoint arrival.

## Boundaries, holds and pauses

All255 boundary requests compile to exact input point/type/timestamp records
(2,555 points checked). Delivery nevertheless differs:

| Command | Median received contact duration |
|---|---:|
| Ten15 ms segments |0 ms|
| Ten16 ms segments |0 ms|
| Ten17/20/25/33 ms segments |333.333 ms for each schedule|
| Twenty17/20 ms segments |666.667 ms|

The15vs16 ms ten-segment controls have identical received phase/position/time
sequences, including coalesced points. This is empirical ambiguity in this
observed input stream. Mixed long/short controls behave differently, so neither
a global33.3 ms-per-segment law nor a simple fixed duration correction follows.

Zero-distance moves preserve stationary intervals, with additional timing
quantization. In contrast, a W3C pause emits no stationary record: an initial
100 ms pause followed by 300 ms movement becomes one400 ms motion rather than a
hold plus move; a middle pause extends following movement. Pauses are not a
safe replacement for zero-distance holds.

The exact clip67 command reaches its final moving endpoint at966.667 ms in the
frame-quantized simulator export, versus934 ms commanded, then releases at1150 ms
versus1106 ms commanded. Its final hold is approximately183.333 ms. This does not
support the old video's supposed170 ms lead or hold collapse. It also cannot
prove the physical phone executed the same schedule.

## Callback observer control

Six cases have20 runs each when combining main 5 with the three logging groups.
Their frame-indexed XY paths are identical. Optional stationary coalesced
samples and duplicate end-position move phases vary; strict phase lists are
not universally identical. Duration SD is at most0.210 ms in this comparison.

No control group has a callback lag above50 ms. Maximum lags are
23.783/21.154/21.620 ms for synchronous-before/buffered/synchronous-after.
The earlier main-run1.376s callback stall did not recur. Its cause remains
unidentified; these controls neither establish backend timing jitter nor show
that synchronous logging caused it. Callback arrival, touch timestamp and
game-display time remain distinct clocks.

## Limits and reproduction

The physical XRs' OS/runtime compatibility was not independently verified.
There is no independently logged True Skate touch-to-render clock here.
The simulator law, half-point coordinates, short-segment effects and corner
geometry must not be generalized to other OS/builds/devices without validation.
Owned WDA/logger/build processes were stopped by the primary analyst; the
preexisting simulator was left booted. No collection or production gate changed.

The original scratch directory disappeared after compact preservation. Movies,
profile caches, full framework disassembly and copied WDA/builds are unavailable
at their former paths. The standalone logger source and buffered variant patch
are preserved. The saved raw event logs,
boundary manifests, summaries, diagnostic patches, analysis scripts and exact
runtime constants support offline reanalysis. Repeating the simulator setup
requires recovering/reconstructing missing build metadata and verifying the
setup separately; this bundle does not claim complete setup reproduction.

The bundle verifier checks counts and preserved hashes without operating a
simulator. The compensation follow-up tests duration-model generalization from
these saved observations; the simulator/video follow-up independently compares
forward schedules with the already-recorded phone movies, with an inconclusive
result and explicit source-availability limits.
