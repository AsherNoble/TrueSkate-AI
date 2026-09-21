# Current research status

Updated 2026-09-22. Behavioural cloning is the development direction.

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
