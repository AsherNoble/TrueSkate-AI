# Current research status

Updated 2026-10-09. Behavioural cloning is the development direction.

## Model 1

Current work is calibrated linear clip regression and scaling. The current
best is **91.04%** test recovery: the 64×144 input, seed 2, trained on
`model1_linear_good13100_20260927` and selected on validation (90.94%)
([M1-HALFRES](experiments/M1-HALFRES-20260928.md); this was the second
exposure of that test partition). Development now runs at 64×144 (no worse, ~4× cheaper; 32×72 lost 7.7 pp,
[M1-QUARTERRES](experiments/M1-QUARTERRES-20260929.md)) with a
cosine lr schedule (validation plateau 89.17% against 87.11%
[M1-LRDECAY](experiments/M1-LRDECAY-20260928.md)). The nested-subset curve
shows the model is still data-limited: validation error fell 39.7% for the
last doubling ([M1-SCALE-SUBSETS](experiments/M1-SCALE-SUBSETS-20260928.md)),
and the collected doubling (18,394 training clips, weighted to Kansas City
and Los Angeles, plus three new parks) raised the validation plateau to
**91.37%**, a 20.3% error reduction: still data-limited by the fixed rule,
but only just ([M1-EXPAND](experiments/M1-EXPAND-20260929.md)). Its best
checkpoint (92.16% validation) has not been scored on test. On 907 clips selected in whole recording sessions in the three new parks it scores 87.32%. The first clean-label retrain (128×288, seed 0) scored
**88.50%** on test. The previous
recipe scored 80.05%; same size, park mix and split protocol
([M1-RETRAIN](experiments/M1-RETRAIN-GOOD13100-PLAN-20260927.md)). Remaining
failures are mostly end points falling short, concentrated in Kansas City,
Los Angeles and fast gestures ([M1-DIAG](experiments/M1-DIAG-20260928.md)).
Narrowing the end read's time window recovers ~1 pp on frozen checkpoints
([M1-ENDPROBE](experiments/M1-ENDPROBE-20260930.md)); a retrain with it is
proposed.
The earlier 80.05% run selected seed 0 on validation (80.00%) and exposed the
test split once.
This is command-held-out evidence, not proof of unseen-park generalisation or
the >99.9% certification target. See [evaluation record](experiments/M1-20260904.md).

Hold, per-frame heatmap and recurrent temporal models remain executable
experiments. Their negative results and calibration discoveries are retained
through the [archive](ARCHIVE.md). The linear pipeline is not a substitute for
curved or curved+spin certification.

One useful negative result: [full ungating](experiments/M1-20260719-ungating.md)
collapsed precision in the earlier per-frame heatmap model. Do not repeat that
experiment without a changed hypothesis, or generalise it to the newer linear model.

The [scaling protocol](protocols/model1_scaling.md) defines frozen cohorts,
nested subsets, validation-only selection, interrupted-run resume and separate
certification. Its historical cost figures are dated estimates, not current
quotes or authorization for cloud work.

## Model 2

The sequence policy, causal datasets, stroke assembly and inference are retained
under `model2`. Model 2 remains unfinished. Current BC includes overlapping
action groups and activity masks; the older rig model must not overwrite it.

## Open questions

- Proposed Model 1 replay fine-tuning: automatically grade predicted-gesture
  replays against reference gameplay, after measuring repeatability and the
  required feedback speed. [Discussion note, 2026-10-09](protocols/model1_replay_finetuning.md).
- [XR1 kickflip repeatability](experiments/KICKFLIP-REPEATABILITY-20261009.md):
  XR1 is ready in Workshop. Its first setup attempt stopped before recording/
  gameplay because WDA reported `unversioned`; the operator approved the verified
  signed replacement and setup is continuing with the failed start retained.
  Twenty frozen repeats require explicit
  QuickTime review acceptance first. No gameplay results or game-noise estimate yet.
- How does recovery scale with data volume and domain/session diversity?
- When should linear work expand to curved and curved+spin trajectories?
- Can Model 1 label expert recordings accurately enough for useful Model 2 training?
- How should future collection broaden spatial coverage without contaminating labels?

The frame-alignment sub-problem of self-labelling is validated: WDA's internal
`submitted_to_ios` timestamp aligns gesture onsets to the video within one frame
and beats the uninstrumented host clock ([ALIGN-20260913](experiments/ALIGN-20260913-wda-onset.md)).
One calibration anchor suffices only for short clips (< ~15 s); a per-recording
linear timebase drift (~−1900 ppm, phone wall-clock vs video media-clock) pushes
a single anchor past one frame by ~23 s, so longer recordings need ≥2 anchors
(start and end) with a per-recording offset+rate fit. This is one device, park
and session; it does not certify cross-park/session/device generalisation, the
drift-rate distribution across recordings, or downstream trace quality. The WDA
fork instrumentation remains on its feature branch and was hand-launched; its
validated client/alignment integration is now present on this repository branch.

These are research decisions, not tasks automatically authorized by maintenance.

## Timing and data-quality audit

Three original recordings now have preserved human onset annotations. Two show
false automatic calibration detections; one shows an approximately half-second
within-recording timing shift. These selected surviving originals do not estimate
accepted-corpus error prevalence. The replacement calibration path applies only
to newly collected segments and does not retroactively repair the old corpus.
A human-confirmed spin-contaminated gesture must be excluded before the next
linear training build; exclusion enforcement is still pending. See
[M1-TIMING-20260912](experiments/M1-TIMING-20260912.md).

Bounded timing repeats show return-minus-duration alignment does not reliably
achieve one-frame accuracy. Run03 now captures Appium proxy boundaries: most
call overhead occurs within the WDA round trip. Human labels again yield only
1/6 swipes within one frame using return-minus-duration; using WDA response
instead of client return does not improve that count.

Internal WDA instrumentation is pushed and tested. Its initial deployment was
blocked by training-server signing errors; the later signing/deployment succeeded
and runs05–09 were completed ([ALIGN-20260913](experiments/ALIGN-20260913-wda-onset.md)).
Bundled run04b was rejected after the user observed joined gestures; use separate
requests. Run10 tested
first/last-anchor correction across a 59.10 s recording, with a held-out middle
calibration and eight gestures. All eight gestures were within one 30 fps frame;
the held-out middle calibration missed by 38.239 ms. The user accepted this as
sufficient practical evidence for the linear collection path. The canonical
implementation now brackets each minute with fixed centre-screen controls and
maps WDA `submitted_to_ios.monotonic_s` to video time with those two observed
onsets. It rejects incomplete WDA reports, overlapping controls, and in-recording
resets, and never emits controls as training examples. This code has offline test
coverage but has not yet collected a production segment on this branch.

The classical onset detector now uses local-background subtraction
and short look-ahead confirmation. On the reused development set it matches
23/24 sampled swipe starts exactly, up from 21/24 for the preserved colour-only
baseline; one moving floor graphic still triggers early. This is not held-out
accuracy. It is now the calibration-control detector; fixed centre controls avoid
the observed red-floor edge case.

The 1,109-clip September replacement tranche exposed a separate compact-video
phase defect: FFmpeg resampling paired selected pixels with a synthetic output
time grid. The operator observed a one-slot lead in 35/35 reviewed clips. A
surviving raw recording confirmed that the centre-touch detector chooses the
first visible frame and that the two-anchor WDA fit remains within one native
30 fps frame. An upward-rounding workaround fixed the preserved example but did
not establish a reliable contract on a new clean segment. Compact extraction now
chooses exact source frame numbers and stores those frames' probed source PTS
relative to gesture onset. The existing tranche will be replaced, not repaired.
A bounded XR2 smoke test of the exact-PTS extractor accepted 11/11 payload
clips; every clip decoded to 32 frames with 32 stored source-relative
timestamps, and WDA confirmed True Skate remained frontmost.
The audited 1,109-clip tranche has only one surviving original `.mov`, and that
recording has no emitted clips because alignment was interrupted. It cannot be
faithfully regenerated from original pixels, so it remains an audit artifact;
the next tranche must be collected with the exact extractor enabled.
See [M1-20260921](experiments/M1-20260921-direct-video-phase.md).

The bounded phase-fix validation also exposed an iOS-overlay guard gap. With
Control Center already open, Appium still reported True Skate as foreground and
the collector executed an entire segment against SpringBoard. The raw recordings
were rejected by calibration and retained. Connection and per-gesture guards now
also require WDA's frontmost bundle to be True Skate; unavailable active-app data
falls back to the existing app-state and screenshot guards. Clean validation of
the guard passed the bounded XR2 segments, including the final exact-PTS run
(11/11 strict payload admissions).
A direct XR2 check also validated recovery: with Control Center opened after
connection, WDA reported SpringBoard despite Appium reporting True Skate
foreground; the shared foreground guard reactivated True Skate and WDA then
reported the True Skate bundle. No gesture was recorded in that check.
The subsequent Skateboard GB 2024 audit tranche admitted 310 strict clips with
the exact extractor. All clips decode to 32 frames and carry 32 increasing
source-relative timestamps. During that run, the live foreground guard caught
SpringBoard, restored True Skate and withheld the affected attempt; the segment
then failed calibration and retained its raw recording without emitting clips.
The operator marked 17/17 inspected clips `good`. A full metadata comparison
found that the historical 13,100-sample corpus also used 32-frame windows and a
fixed onset slot, while both corpora vary command duration across the same
0.30–1.20 second range. The exact-PTS tranche therefore preserves those label
controls while fixing the pixel/timestamp alignment defect. See
[M1-LABEL-CONTROL-20260922](experiments/M1-LABEL-CONTROL-20260922.md).

New Model 1 sampling uses the versioned XR control map with a 16-logical-point
margin. The transient bottom navigation row is conservatively treated as present
at all times. Its initial start-only rule failed the first bounded on-device test:
a swipe starting about 20 logical points below the mapped Camera edge and moving
immediately into it opened Replay. The run stopped after 2/300 gestures and was
preserved without replacement. The basic-linear sampler now rejects the entire
straight path if it touches any expanded hitbox. A 100,000-command offline stress
run had no retry exhaustion, and a bounded XR2 segment admitted 11/11 strict,
32-frame exact-PTS clips with zero geometric intersections or detected UI
transitions. Research into the game's path/speed activation mechanism remains
deferred because the conservative sampling rule is sufficient for this corpus.
The replacement XR2 corpus began in Skateboard GB 2024 from revision `b60f19d` after a
strictly admitted 10-clip production segment; its automatic stop target is 1,100
strict clips. On 2026-09-20, all 910 admitted clip records and their segment
manifests were corrected from the erroneous `The Workshop` provenance label;
collection remained paused after the correction. XR1 was unavailable when
collection began. See
[M1-CONTROL-20260917](experiments/M1-CONTROL-20260917.md).

The residual XR2 XCTest attachment from the whole-path smoke test is resolved.
It was a preserved 304-second June orphan; Appium's cleanup RPC falsely reported
deleting it. Removing its directory and restarting testmanagerd while the
directory was absent restored the recorder, after which two consecutive short
recordings completed and self-cleaned. The cleanup wrapper now relists after
deletion and fails if a UUID survives. See
[XCTEST-ATTACHMENT-20260922](experiments/XCTEST-ATTACHMENT-20260922.md).

XR1 recovery completed on 2026-09-22. The operator refreshed Xcode signing and
trusted the replacement runner; the exact instrumented WDA build reports
`b5ace21788b5f5dc4cf0e0759f8bb8a79ab83ae6`. WDA 8100 and Appium 4723 are
healthy. Two consecutive bounded recordings completed, returned the recorder to
idle and self-deleted; a final listing found zero attachments on both XRs. The
fork and rig WDA checkouts now use clean `master` at that SHA. Collection remains
off until an explicitly authorized workload begins.

The linear replacement pool has 13,902 strict clips across five parks: 4,052
Workshop, 3,809 Kansas City, 2,020 Skateboard GB (including its 310-clip
baseline), 2,003 Los Angeles, and 2,018 Super Crown. The two bounded Super Crown
runs stopped at 1,152 XR1 and 866 XR2 clips. Device-by-park cells were not
quotas. A balanced 140-clip human audit marked 124 good, 9 mild and 7
critical. Six critical clips were XR1/Los Angeles, where anomalous calibration
rates closely predict the late visible trace. Across the full pool, 1,063 clips
are in segments whose two-anchor fit rate differs from `1.0` by more than 0.2%;
this is a screening count, not a proven error count. A blinded held-out
24-clip check then confirmed the predicted error pattern in all twelve paired
recordings: 22 first-trace frames were predicted exactly from the
calibration-rate anomaly and two within one frame. This validates the
extreme-rate timing risk, but not a universal rejection cutoff or the visual
cause of a mistaken anchor. The intended deterministic 13,100-sample training
selection remains on hold pending an admission decision. See
[M1-AUDIT-20260924](experiments/M1-AUDIT-20260924.md) and
[M1-ONSET-VALIDATION-20260924](experiments/M1-ONSET-VALIDATION-20260924.md).
Seven new Los Angeles source videos from both XRs are retained for direct
calibration-onset inspection. The bounded run did not reproduce the severe
high-rate case; one XR2 segment has a weaker start detection. See
[M1-LA-ORIGINALS-20260924](experiments/M1-LA-ORIGINALS-20260924.md).
A bounded XR2 probe initially sent only one pointer per attempted pair because
Selenium `ActionChains` discarded the other source. The corrected two-source
command completed in three XR2 repetitions without leaving gameplay. The
operator saw two simultaneous visible marks in every corrected repetition;
no timing-calibration change follows yet. See
[M1-MULTITAP-20260925](experiments/M1-MULTITAP-20260925.md).
A follow-up XR2 four-touch probe omitted the single-touch control and sent
four points in a 66-point square. Three bounded recordings stayed in gameplay;
preliminary inspection shows four visible spots, pending operator review of
their precise timing. See
[M1-MULTITAP4-20260925](experiments/M1-MULTITAP4-20260925.md).
A 7 × 13, 50-point grid sent 91 simultaneous W3C pointer sources on XR2, but
only five requested positions visibly brightened in each of three recordings;
command completion took 7.6–30.5 s. Thus the full-grid marker is not yet a
usable calibration signal despite successful Appium/WDA responses. See
[M1-MULTITAP91-20260925](experiments/M1-MULTITAP91-20260925.md).
Unity documents five concurrent fingers for iPhone. An XR2 eight-touch probe
with three different pointer orders also lit up only five requested positions
per recording. The outgoing payload and deployed WDA construction loop retain
all eight paths, so the practical limit appears downstream; the exact layer
is unresolved. A five-point marker remains a candidate. See
[M1-MULTITAP8-20260925](experiments/M1-MULTITAP8-20260925.md).
A bounded XR2 die-five pilot then recorded twelve five-point calibration
markers in three short videos. The existing local onset detector found every
position on the same first-visible frame for all twelve markers; the existing
centre-only detector also succeeded in this stationary scene. This establishes
executability and local onset visibility, not yet a gain in calibration
accuracy or a fix for the severe two-anchor anomaly. See
[M1-DIE5-20260925](experiments/M1-DIE5-20260925.md).
On 2026-09-23,
the opt-in batch exact-PTS extractor was promoted from clean release `1b8497e`
after isolated one-segment checks on both XRs; previous release `13554c9`
remains for rollback.
See [batch extraction evidence](experiments/M1-BATCH-EXTRACT-20260923.md).
See [M1-RECOLLECT-20260922](experiments/M1-RECOLLECT-20260922.md).

## Calibration false detections (2026-09-25, branch `research/model1-five-touch-calibration`)

The weak or early start-control detections behind the Los Angeles rate
anomalies (90/207 XR1 LA segments) coincide with the board and camera still
moving for ~7–8 s after the pre-segment reset. The branch adds:

- **Five-touch comparison** ([M1-DIE5-COMPARE](experiments/M1-DIE5-COMPARE-20260925.md),
  [100 pt repeat](experiments/M1-DIE5-R100-20260925.md)): scored against blind
  labels.
  - In the heavy-fault 100 pt run, five-touch had significantly fewer gross
    start errors than single-touch (0 vs 9 discordant, p = 0.004). It still
    failed adoption: 3 errors when it answered, and 45% XR1 start rejections.
  - The 100 pt spacing cut post-swipe mid rejections from 12/60 to 4/60
    (descriptive).
- **Timing gate** ([M1-TIMING-GATE](experiments/M1-TIMING-GATE-20260926.md)):
  every labelled control appeared 118–174 ms after its WDA submission, and
  every gross detector error came earlier. Held-out corpus test
  **supported**: 90/90 high-rate segments have an early start anchor, and
  98.7–99.7% of ordinary anchors pass.
- **Settle wait** ([M1-SETTLE](experiments/M1-SETTLE-20260925.md)): an opt-in
  wait before the controls. It is a manipulation-checked precaution, not a
  validated fix.
- **Timing screen** ([M1-SCREEN-DRAFT](experiments/M1-SCREEN-DRAFT-20260925.md)):
  an opt-in per-clip screen for cohort manifests.
- **Multi-anchor calibration** ([M1-MULTIANCHOR](experiments/M1-MULTIANCHOR-20260925.md)):
  an opt-in consensus fit over extra single centre controls. It is a
  smoke-tested candidate that needs no multi-touch.

- **Corpus screen and audit** ([M1-CORPUS-AUDIT](experiments/M1-CORPUS-AUDIT-20260927.md)):
  `corpus-screen-v1` is frozen: whole segments are excluded if the rate is off
  by more than 0.0008 or a control lies outside 85–210 ms after its command.
  It keeps 12,113/13,592 replacement clips (Los Angeles 938). A blind random
  100-clip audit passed 100/100 within ±1 displayed frame, 97 exact.
- **13,100 good labels** ([M1-TOPUP-PLAN](experiments/M1-TOPUP-PLAN-20260927.md)):
  a settle-wait top-up (release `080bb97`) collected 722 clips, all of which
  passed the screen, including Los Angeles (previously 47%).
  `model1_linear_good13100_20260927` (rig
  `tmp/final-manifest-20260927/`, fingerprint `sha256:9fadddee…`) matches the
  historical park counts of the 80.05% corpus. A fresh blind 100-clip audit
  passed 100/100, all exact. It is ready for the controlled retraining
  comparison.

No corpus clip was moved or deleted; exclusion is manifest-only. The rig's stable
release is `189dfde` (rollback `080bb97`), deployed 2026-10-03 for the
notification fixes (PRs #30, #31). `com.trueskate.services` was restarted onto
it and owns its iproxies; the storage guard loads it on each run. The dashboard
still runs the code it loaded earlier, which is unchanged in this release.
Exclusion, recollection and flag adoption await operator decisions.

**Rig note (2026-09-27):** with both recorders idle (`/wda/video` null) and no
collector running, `scripts/recover_remotexpc_attachments.sh --delete` removed
12 XCTest attachments on XR1 (11 from before plus one from the failed
M1-DIE5-R100 segment) and 7 on XR2. Both phones re-listed 0 remaining. One
earlier XR2 dry run failed transiently (`tmp/Attachments` not found); a
repeat succeeded.


## HID gesture execution and recording review

The ESP32 Bluetooth pointer pilot supports separate strokes with short gaps,
but XR2 forces a 15 ms connection interval. Remeasuring its movies with the USB
probe's tracker and FFmpeg gives 5.3% frame-displaced reports with Wi-Fi on and
3.0% off, superseding the original approximately 6% tracker result. USB run-11
gave 4.3% in the matched ten-pass, 300-report comparison: no clear jitter gain.

XR2 accepts Pico hover, a tap and two drags through the powered camera adapter
while WDA control, native XCTest capture, retrieval and attachment cleanup run
over Wi-Fi with no rig USB. Run-09 passed all five cases, including minute
30/60 fps recordings and two-anchor WDA calibration. Run-11 (2026-10-08) passed
a one-minute 60 fps capture (3,580 frames, 59.56 effective fps, zero cleanup
errors) and the Pico completed 1,497/1,497 events. Regular board acceptance and
near-complete visible travel at 3 ms support faster report throughput. At 1 ms,
four of twelve passes exceeded the board lateness gate (maximum 6.608 ms), and
capture gaps clustered around fast motion; clean 1 ms game-input timing remains
unvalidated. Contacts are visible in assistant-reviewed sheets, not a human
fidelity audit. A buffered host command link, exact press/lift/path validation
and production collection remain open; Ethernet and multi-finger control are
separate untested proposals.

The wireless route uses a RemotePairing TCP tunnel (pymobiledevice3, Python
3.13), not the classic CoreDeviceProxy, which XR2 rejects off USB. XR2 carries
instrumented WDA `ae50404a`; production services can reinstall an `unversioned`
runner when rebuilding over USB. Collection remains off. See
[HID-USB-20261006](experiments/HID-USB-20261006.md) and its run-11 evidence.

A bounded preloaded USB curve pilot is implemented. Its one button-up gain
recording completed successfully (3,599 native frames, 59.92 effective fps,
zero cleanup errors). The Pico accepted all 971 reports with maximum board
lateness 32 microseconds. All 36 movement passes and 74 stationary windows
passed the gain checks, with 100% tracking and maximum stationary spread
0.30 logical points; measured gains at 15 and 3 ms differ by at most 0.55%.
Both six-gesture programs are compiled from this measured profile. Batch 1's
one-minute recording completed (3,601 frames, 59.91 effective fps, clean recorder
and attachment cleanup), and its Pico receipt confirms 1,684/1,684 accepted
reports with no clipping and 32 microseconds maximum board lateness. Batch 2
also completed (3,604 frames, 59.91 effective fps, clean recorder/attachment
cleanup), with all 1,684 reports accepted, no clipping and the same maximum
board lateness. The authorized one-gain/two-curve recording workload is complete.
Original movies and timing sectors have verified local copies; sealed native
timing-control reviews are prepared for both batches. Batch 1's human touch-down
marks and operator-confirmed clear preceding frames passed source integrity.
Its held-out middle residual is 16.7487 ms against a 16.667 ms frame limit,
narrowly failing timing qualification (81.7 microseconds beyond the frozen gate;
25.0822 ms carried clock uncertainty). Its curve fidelity remains inconclusive,
with no curve score emitted, individual phase fit or gate relaxation. This does not identify USB jitter
as the cause. Batch 2's verified human control review passes the held-out middle
gate: 14.1529 ms against 16.667 ms, with 22.4864 ms carried clock uncertainty.
Batch 2's position review is imported: 100 cursor-centre marks, each explicitly
declared by the operator to mean active contact at that instant. The unused
orange-confirmation checkbox is superseded by that attestation in a separate
scoring copy; original export bytes and all points/uncertainties remain preserved.
The unchanged orange-control clock comparison has three positional failures
(150 ms arc/S and 600 ms pause) and three inconclusive cases, with zero passes.
Both 600 ms cases have full planned-frame position coverage. Lift boundaries
and interruption assessments are unknown, so duration/continuity are unassessed.
Operator notes report orange feedback at cursor positions from 1–2 earlier
frames in four clips. This cross-signal phase relationship is not quantified;
a diagnostic cursor detector cannot reliably identify the control onsets.
No per-curve retiming or replacement clock was used. These results do not identify
USB jitter as the cause or establish a twelve-curve fidelity pass. All review
viewers are stopped. Capture and board completion do not establish visible
contact/path accuracy. Native-PTS review uses
Pico start/end controls and a held-out middle
control, retaining clock/position uncertainty. The operator kept the 150 ms
reversal stress cases despite their possible 60 Hz measurement floor. The
Pico-only control substitution is specific to this pilot; Model 1 five-touch
requirements and collection-off remain unchanged.
[Gain evidence](evidence/HID-USB-20261006/pico-curve-pilot-gain-01-20261008),
[first curve capture](evidence/HID-USB-20261006/pico-curve-pilot-curve-01-20261008),
[second curve capture/review preparation](evidence/HID-USB-20261006/pico-curve-pilot-curve-02-20261008),
[first timing review](evidence/HID-USB-20261006/pico-curve-pilot-curve-01-timing-20261009),
[second timing qualification and curve review](evidence/HID-USB-20261006/pico-curve-pilot-curve-02-timing-20261009),
[second cursor-position review](evidence/HID-USB-20261006/pico-curve-pilot-curve-02-position-20261009),
[pilot workflow](../hardware/pico_curve_pilot/README.md).

AssistiveTouch supports preset and recorded multi-finger gestures. Independent
live control of two mouse-driven contacts remains unverified. In three short
XR2 diagnostic recordings, the pointer-only and WDA spin-hold controls worked;
the combined condition showed the moving pointer trail while the separate spin
button remained held. This establishes coexistence for that WDA/AssistiveTouch
condition, not physical-pad operation, accurate spin angles or microsecond
touch delivery. A human-finger/pad coexistence test remains necessary.

Existing 60 fps pointer replays contain distinct gameplay changes in all 3,364
audited moving-frame pairs after overlay/compression controls. Effective capture
was 59.30–59.65 fps with approximately 0.9% absent nominal 60 Hz slots, mostly
near startup. Raw size was 75.19 MB/min against 74.75 MB/min for the compared
30 fps recordings; equal bitrate does not establish equal image fidelity.
Use 60 fps for replay diagnostics. Preserve the current collection recipe until
separately authorized one-minute capture/alignment validation and a temporal
sampling decision: the exact-PTS extractor still supplies 32 frames across
2.3 seconds, so raising raw capture rate alone does not increase Model 1's
input sequence length. Collection remains off. See
[HID-REVIEW-20261004](experiments/HID-REVIEW-20261004.md) and the prior
[HID-POINTER-20261004](experiments/HID-POINTER-20261004.md).


The consolidated HID host/firmware source uses protocol v2 with exclusive ownership,
cancellation, fail-closed foreground checks, a one-minute recording budget, neutral
recovery and exact ordered notification-attempt receipts. This source revision is
not deployed; older deployed firmware is intentionally rejected by the new host.
Only XR2 has a measured profile. Historical spin harnesses remain evidence, with no
curved/spin certification or physical-pad validation. See
[HID source operating notes](../hardware/hid_pointer/NOTES.md).

## Curved gesture execution gate

[CURVE-EXEC-20261002](experiments/CURVE-EXEC-20261002.md) introduces an additive
nine-number Bernstein cubic in time, exact rounded cumulative command boundaries
and matching quantized labels. No model or checkpoint schema changed. On twelve
seeded synthetic curves, maximum-duration rules 10/20 ms meet the ≤0.005 command
approximation bound; 40 ms fails two. This is compilation evidence, not observed
execution fidelity. The authorized Inbound pilot stopped after eight XR1 diagnostics. Separate
source-PTS-preserving FFmpeg reanalysis fixes an OpenCV extra-frame mismatch and
passes timing calibration (held-out middle error 5.8 ms), but the blind orange
extractor yields 0/8 evaluable gestures because of competing moving scenery/board
colours and ambiguous centrelines. Execution fidelity and the measurement floor
remain **inconclusive**; XR2, repeats, dense probes, main and confirmation did not
run. Retained native frames and a blinded review export support diagnosis. No
spacing rule is selected; improve measurement before another bounded pilot.
Curve recovery/generation remains a later tranche, with no training authorized.

User-requested [CURVE-SPEED-20261002](experiments/CURVE-SPEED-20261002.md) recorded three wide arcs at 300/200/120 ms on XR1 for interactive human viewing. Post-recording gameplay admission failed; the isolated attempt is preserved and is not fidelity evidence.

[CURVE-JAGGED-20261002](experiments/CURVE-JAGGED-20261002.md) executed a user-requested rounded zigzag at 600/300 ms using five joined cubics in one touch. The isolated recording is available for human viewing; gameplay admission flagged it and no fidelity pass is claimed. This does not change the single-cubic model interface.

Operator review of the jagged clips: 600 ms looked good; 300 ms showed a tap-like mark with a strong gameplay response. Visible trail is therefore insufficient on its own to infer gameplay input fidelity. A rendering/synthesis limit is a hypothesis; duration and segment spacing/count changed together, so no causal threshold is established.

The authorized [LINEAR-SPEED-20261002](experiments/LINEAR-SPEED-20261002.md)
duration-only sweep froze one linear movement per drag, 600/400/300/200/100/50/20/10
ms in reversed repeats on XR1/Inbound. Its first recording aborted on a sleep
overrun before any control or drag. The short original and empty WDA report are
preserved; the second recording was withheld without replacement. **No rendering
transition range was measured.** Collection remains off.

The operator subsequently authorized replacements and requested a usable viewer.
Two complete reversed recordings now preserve all sixteen drags, with 1,767
native frames and nineteen successful WDA requests each. Independent held-out
middle timing residuals are 14.3/9.3 ms. Both original gameplay admissions remain
failed; their first flagged frames visibly show normal gameplay scenery. Assistant
native-frame review finds trails at 50–600 ms, indeterminate feedback at 20 ms
despite strong gameplay response, and tap-like spots at 10 ms. The visible-trail
transition repeats between 20 and 50 ms; no input-collapse threshold follows at
30 fps. The latter half of repeat 2 changes reset location within Inbound.
See the [redo record](experiments/LINEAR-SPEED-20261002.md#authorized-redo-and-native-frame-viewer).

Operator follow-up: visible trace and board movement at 600–50 ms; no visible
trace but board movement at 20 ms; minuscule tap/flicker without board movement
at 10 ms. This primary assessment replaces the assistant's 20 ms uncertainty.
A requested [50–20 ms fine sweep](experiments/LINEAR-SPEED-FINE-20261002.md)
uses 50/45/40/35/30/25/20 ms in reversed repeats with the same path and one move.
The fine sweep completed all fourteen drags; full native decode and timing
checks pass (middle residuals 11.8/6.4 ms). At operator direction, gameplay
assessment uses human vision; automated menu/editor scans are skipped. The operator
review below supplies the visual findings. Collection remains off.

Fine-sweep operator assessment: trace visible in 13/14, except 20 ms R2; board
movement in all 14. No consistent duration-only visibility cutoff is established.
The operator authorized 135 blinded diagnostics: nine durations (50 down to
10 ms by 5), five lengths, three repetitions per cell, with human gameplay review.

[LINEAR-LENGTH-20261003](experiments/LINEAR-LENGTH-20261003.md) completed all
135 diagnostics (45 length/duration cells × 3), randomized across fifteen
recordings. All 315 WDA requests, native decodes and timing checks pass; maximum
held-out middle residual is 25.2 ms. The anonymous viewer is ready with exactly
`flicker` / `hold` / `trace`, Board moved default True, and comments. Condition
keys stay outside the viewer. The completed human assessment follows. Collection stays off.

Blinded operator labels are complete and validated (135/135): trace in all 75
clips at 30–50 ms; 12/15 at 25 ms, 7/15 at both 20 and 15 ms, 0/15 at 10 ms.
Board movement is 120/120 at 15–50 ms and 0/15 at 10 ms, across all lengths.
Observed response transition: 10–15 ms; consistent trace starts at tested 30 ms.
Nineteen flicker clips still move the board. Intermediate-duration length effects
are not consistently monotonic; three repeats/cell do not establish a universal
threshold or input path collapse. [Results](evidence/LINEAR-LENGTH-20261003/operator-review/results.md).

## Curved Model 1 MVP 2.0 requirements

The operator requires arbitrary single-finger paths without spin, with variable
speed and unrestricted gameplay state at gesture start. A single cubic is not
the model's representation limit; an extensible timed piecewise path is the
current design candidate. Next execution review uses a moving circle only,
aligned using five-contact calibration. Model accuracy is position error at
matched frame times, with continued approximately 100-clip held-out human
audits. [Requirements and remaining decisions](protocols/model1_curved_mvp2.md).

MVP refinements: provisional maximum 15 timed waypoints including endpoints;
10-logical-point-radius hollow target ring; broad shape/state coverage; initial
error targets inherited from linear (0.03 normalized position, 0.10 s duration).
Practical test: infer a single-contact expert trick and replay it in Workshop,
checking trick reproduction alongside trajectory accuracy. Model 2 is not
needed for that test. Initial speed/heading/contact state should be comparable.


Research execution/viewer source now emits version 2 content-bound review bundles
and checks foreground/lateness immediately before each scheduled submission. These
synthetic/source checks do not replace human execution fidelity evidence, select a
spacing rule or certify curved/spin collection. Historical v1 reviews retain weaker
provenance and require explicit compatibility when imported by current readers.


## Curved execution audit and replay evidence (2026-10-03–04)

The original direct-waypoint pilot stopped at 0/100 calibrated clips after a
900 ms request took 5.045 s, including 3.751 s of preparation. Its partial video
and timing evidence remain preserved. A subsequent WDA snapshot fix and separately
authorized v2/v3 runs completed the audit: **100/100 human assessments, 66 Good /
21 Minor / 2 Major / 11 Unclear**. See
[CURVE-AUDIT-20261003](experiments/CURVE-AUDIT-20261003.md) for the successive runs
and amendments. Historical statements about pending reruns apply to the pilot.

Exploratory single-rater evidence suggests spacing of about 33 ms or more performs
better; it does not certify a spacing rule, curved executors or spin collection.
Device and park are confounded, and some overlays may be about one frame late.
Bundled, serial, scheduled indexed-path and anchor-finger demo replays all failed
human review; separate scheduled records also hit XCTest's overlapping-record
restriction. Preserve these negative results when considering future executors.
USB HID now has bounded hover/tap/drag evidence; the preloaded curved pilot
awaits live calibration and review. A reusable host command link and physical
spin-pad approaches remain unvalidated proposals.

Current audit/replay source retrieves partial recordings on every post-start exit
and retains failures without retrying failed stop RPCs. New review bundles bind
frozen specifications, successful ordered execution receipts and JPEG bytes;
legacy evidence stays unchanged and requires explicit compatibility. These source
checks authorize no new device run or deployment. Collection remains OFF.
