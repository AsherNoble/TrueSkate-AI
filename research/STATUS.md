# Current research status

Updated 2026-09-12. Behavioural cloning is the development direction.

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

These are research decisions, not tasks automatically authorized by maintenance.

## Timing and data-quality audit

Three original recordings now have preserved human onset annotations. Two show
false automatic calibration detections; one shows an approximately half-second
within-recording timing shift. These selected surviving originals do not estimate
accepted-corpus error prevalence. Shared decoding and calibration remain unchanged.
A human-confirmed spin-contaminated gesture must be excluded before the next
linear training build; exclusion enforcement is still pending. See
[M1-TIMING-20260912](experiments/M1-TIMING-20260912.md).

Bounded timing repeats show return-minus-duration alignment does not reliably
achieve one-frame accuracy. Run03 now captures Appium proxy boundaries: most
call overhead occurs within the WDA round trip. Human labels again yield only
1/6 swipes within one frame using return-minus-duration; using WDA response
instead of client return does not improve that count.

Internal WDA instrumentation is pushed and tested; XR2 deployment is blocked by
training-server Xcode account/certificate/profile errors. Existing WDA remains
running; run05 has no recording yet. Bundled run04b was rejected after the user
observed joined gestures; use separate requests.

Update following Claude handoff: signing/deployment succeeded and runs05–09
were completed (see research/wda-onset-timing-20260913 branch). Run10 now tests
first/last-anchor correction across a 59.10 s recording, with a held-out middle
calibration and eight gestures. Run10 labels evaluated: 8/9 held-out touches within one frame; all eight
gestures pass, but middle calibration error38.239 ms exceeds33.333 ms.
Two-anchor correction improves little over one anchor. Initial gate failed;
XR1/park replications are paused pending discussion.
