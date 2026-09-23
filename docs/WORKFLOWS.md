# Supported and experimental workflows

Run commands from the repository root with `.venv` active. Pass data/checkpoint
paths explicitly. Do not run training against historical held-out sets merely
to test a refactor. `--help` on each entrypoint describes its existing flags.

| Workflow | Status | Entrypoint |
|---|---|---|
| XCTest mixed/hold/linear collection | Supported; linear is current | `scripts/collection/collect_sls_xctest.py` |
| Offline segment alignment | Supported | `scripts/collection/align_xctest_traces.py` |
| MJPEG self-labelled capture | Experimental | `scripts/collection/collect_self_labeled_traces.py` |
| Linear training | Current | `scripts/model1/train_basic_linear_regressor.py` |
| Hold training | Experimental | `scripts/model1/train_basic_hold_regressor.py` |
| Per-frame heatmap training | Experimental | `scripts/model1/train_trace_extractor.py` |
| Recurrent temporal training | Experimental | `scripts/model1/train_temporal_trace_extractor.py` |
| Expert clip labelling | Experimental | `scripts/model2/build_bc_clips.py` |
| Sequence-policy training | Unfinished research | `scripts/model2/train_sequence_model.py` |
| Device policy replay | Experimental; controls phone | `scripts/model2/run_sequence_policy.py` |
| Corpus preview/progress | Supported | `scripts/train_dashboard.py` |

## Offline verification

```bash
python -m pytest -q
python scripts/model1/train_trace_extractor.py --smoke
python scripts/model2/train_sequence_model.py --help
python scripts/ops/check_archive.py --base origin/main
```

For real training, use the relevant trainer's `--help` and provide the existing
corpus and output path. Cloud wrappers (`train_basic_hold_modal.py`,
`train_basic_linear_modal.py`, `train_trace_extractor_modal.py`) now live beside
the Model 1 trainers; they retain their Modal app/volume names. Running Modal
jobs is a separate paid action, not an installation or smoke requirement.

## Current collection

On the rig, the bounded linear wrapper remains
`bash scripts/ops/mvp_collect_linear.sh iPhone_XR /absolute/output/path 1`.
Set `BASIC_LINEAR_PARK` to the actual loaded park; the label does not navigate.
It uses one-minute segments, an exact centre-screen control at the start and
end, instrumented WDA `submitted_to_ios` timestamps, a two-anchor video-time
fit, a reset before recording, and strict aligned video. Set
`BASIC_LINEAR_WDA_TIMING_REVISION` only when deliberately deploying a compatible
instrumented WDA build. For an operator-confirmed gameplay scene with the neutral
bottom row visible, set `BASIC_LINEAR_ALLOW_IDLE_NAVIGATION=1`; replay and editor
guards remain enabled. No reset is sent while recording. The two controls stay in
the raw manifest but the aligner emits no training clips for them. Use separate
outputs for validation.

`BASIC_LINEAR_BATCH_DIRECT_VIDEO=1` opts into decoding each one-minute recording
once for its exact-PTS clips. The existing per-clip extractor remains the default
and is retried automatically if a batch fails validation. See the
[offline comparison](../research/experiments/M1-BATCH-EXTRACT-20260923.md)
before changing a running collection release.

For mixed/hold collection, inspect the collector's `--help`; preserve its
guards and metadata. Label `--park-label` honestly. Keep all trainable samples
separate from calibration controls and exclude contamination markers.

## Corpus and scaling tools

`scripts/data` contains corpus audit/filtering, cohort manifest, shard and
trick-library tools. [The scaling protocol](../research/protocols/model1_scaling.md)
contains the exact frozen experimental procedure. [Trick-library provenance](../trick_libraries/README.md)
explains the retained CMA-ES outputs and their BC use.

## Prediction overlays

Run `python scripts/inspect/render_prediction_overlay.py --corpus /absolute/corpus
--checkpoint /absolute/checkpoint.pth --manifest /absolute/manifest.json
--out /absolute/output --partition train --count 5 --require-miss 1 --no-open`
as one command. FFmpeg/ffprobe are required; decoding defaults to FFmpeg.
The output compares raw frames, the commanded gesture, and the model prediction.
Both animations assume the stored t=0; neither measures onset. The shared
training decoder is unchanged. Images and metadata times retain the historical
independent resampling convention, so a full decode alone does not establish
synchronization. `--require-miss` is an exact quota; this selection is not an
accuracy estimate. The JSON sidecar records every screened sample, including
those not rendered. Test-split inspection must be treated as holdout exposure.

## Original-recording onset annotation

Use `python scripts/inspect/build_onset_viewer.py --video /absolute/original.mov
--out /absolute/empty-output --recording-id session/original.mov
--title "Session name" --export-name timing-labels-session.json` as one command.
Requires FFmpeg/ffprobe. The output includes every original-resolution decoded
frame and its actual presentation timestamp; extraction never changes frame rate.
Open `index.html` in a browser. Arrow keys step one frame, Shift+arrows step 30,
M marks onset, and Download saves labels for comparison with the command manifest.
The browser stores annotations by recording identity. Export labels before
clearing browser data. Output must be empty to protect existing annotations.
Neither this tool nor the overlay contacts the rig or updates training labels.
See [timing findings](../research/experiments/M1-TIMING-20260912.md), including
the human-confirmed training exclusion that must be enforced before the next run.

For last-visible trace frames, run `python scripts/inspect/build_trace_end_viewer.py
--viewer-dir /absolute/existing-viewer --starts /absolute/human-onsets.json` as one
command. This adds `trace-ends.html` beside the original viewer, reuses its frames,
and embeds the original starts unchanged. Select a gesture, mark its last visible
frame with M, then use Next unmarked gesture. End labels are paired by start frame
and exported separately. Ambiguous or recording-truncated endings can be marked
uncertain. Browser storage is separate from onset labels. This measures visible
trace lifetime, not actual contact duration or finger-up.
