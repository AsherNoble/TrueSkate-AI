# Current research status

Updated 2026-09-25. Behavioural cloning is the development direction.

## Model 1

Current work is calibrated linear clip regression and scaling. The recorded
13,100-clip evaluation selected seed 0 on validation (80.00% recovery), then
exposed the 1,965-command test split once: 80.05% complete-gesture recovery.
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
  100-clip audit passed 100/100 within ±1 displayed frame, 97 exact. Next:
  top up ~1,480 good clips (mostly Los Angeles), then re-audit a fresh 100.

No corpus clip was excluded and no launcher changed. Exclusion, recollection
and flag adoption await operator decisions.

**Rig note (2026-09-27):** with both recorders idle (`/wda/video` null) and no
collector running, `scripts/recover_remotexpc_attachments.sh --delete` removed
12 XCTest attachments on XR1 (11 from before plus one from the failed
M1-DIE5-R100 segment) and 7 on XR2. Both phones re-listed 0 remaining. One
earlier XR2 dry run failed transiently (`tmp/Attachments` not found); a
repeat succeeded.
