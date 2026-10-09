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
| Private XR menu control | Standalone; session discovery required | `scripts/control/serve.py` ([guide](XR_CONTROL.md)) |
| Kickflip repeatability | Bounded diagnostic; operator review before 20 repeats | `scripts/collection/probe_kickflip_repeatability.py` ([procedure](../research/experiments/KICKFLIP-REPEATABILITY-20261009.md)) |
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

## Cubic execution fidelity diagnostic

See [CURVE-EXEC-20261002](../research/experiments/CURVE-EXEC-20261002.md) for the
frozen protocol, evidence and exact commands. `sim/cubic_curve.py` and
`touch_labels.cubic_command_label` share rounded cumulative timing and quantized
coordinates; legacy gesture timing is unchanged. `prepare_curve_exec.py` freezes
commands and checks offline approximation. `probe_cubic_curves.py` runs one bounded
recording on training-server. `measure_curve_exec.py`, `build_curve_audit.py` and
`report_curve_exec.py` provide source-PTS extraction, blinded annotation and
fail-closed stage decisions. Use explicit `PYTHONPATH=src` in an isolated worktree
with the existing `.venv`; never install another virtual environment.

The initial Inbound pilot is inconclusive (0/8 automatically evaluable); no main
or confirmation run is authorized by its stop gate. Do not rerun/replace failed
attempts or route these artifacts into training corpora. The executed manifest
and corrected implementation manifest are separate, preserved versions.

## Linear drag speed diagnostic

`scripts/collection/probe_linear_speed.py --freeze --manifest /absolute/new.json`
freezes the XR1/Inbound duration-only sweep. Execution on training-server takes
`--manifest`, `--wda-revision` and a new isolated `--out` directory, with the
existing `.venv` and isolated source path. It attempts at most two recordings,
gates the second on the first's admission and never replaces a failure.
The authorized attempt aborted before any touch; this command reference does
not authorize another attempt. See [LINEAR-SPEED-20261002](../research/experiments/LINEAR-SPEED-20261002.md).

`scripts/inspect/build_linear_speed_viewer.py --manifest /absolute/frozen.json
--recordings /absolute/recordings --out /absolute/existing-viewer/new-child`
adds duration buttons, slow motion, exact source-frame stepping and separate
visible-feedback/gameplay-response assessments. Unexecuted requests remain
disabled. Source decode counts and remux PTS must match before publishing.

The operator-authorized redo completed the sixteen-clip comparison. `--repeat 1`
or `--repeat 2` runs one separately authorized diagnostic recording with its own
admission result; it is not an admission override or a standing retry policy.
`scripts/inspect/check_linear_speed_recording.py /absolute/recording-directory`
preserves a separate start/end and held-out middle timing diagnostic and the
first gameplay-flag frame, leaving the original admission untouched. Existing
diagnostic output must be preserved. The viewer plays decoded native frames,
supports source-frame stepping, and pairs duration buttons across both repeats.

### Blinded duration × length diagnostic

The frozen [LINEAR-LENGTH-20261003](../research/experiments/LINEAR-LENGTH-20261003.md)
workload contains 135 gestures, not an open-ended collector. Freeze with
`probe_linear_speed.py --profile length --freeze --manifest PATH`. On the rig,
run its verified manifest with `--human-gameplay-review`, isolated `--out` and
`--wda-revision`; optional `--repeat N` selects one of fifteen recordings.
Run `check_linear_speed_recording.py RECORDING_DIR --timing-only` locally for
calibration and full native decode, then `build_linear_length_audit.py` with
`--manifest`, `--recordings` and fresh `--out`. Keep the private key, source map
and raw recordings outside the viewer web root. The public labels are
`flicker` / `hold` / `trace`, a board-movement toggle defaulting True, and comments.


## Content-bound research reviews (source revision 2026-10-05)

New curve, speed and length builders produce version 2 reviews. Their identities
bind the frozen manifest, complete ordered command specifications and payloads,
successful host/WDA execution receipts, source video hashes and displayed JPEG bytes.
New recordings carry `research-execution-v2` receipts; an ID-only or incomplete
historical execution log cannot become a strengthened review through the new builder.
Private maps remain outside the blinded web root. Browsers hash bytes before display
and refuse marks/export when footage changes; saved marks are scoped by bundle hash.

Curve measurement now also requires `--manifest /absolute/frozen/manifest.json`.
Import a new curve review with `build_curve_audit.py --private-map … --export …
--media-root /absolute/review --out …`. Import a new length review with
`report_linear_length_audit.py --export … --evidence … --bundle-map
/absolute/recordings/audit-private-source-map-v2.json --media-root /absolute/review
--out …`. Supply absolute paths for each placeholder and use new output directories.

Historical v1 files stay unchanged. Reading a v1 curve or length export requires
`--allow-legacy`; the reader warns and reports that execution content and displayed
frame bytes are not bound. This option does not upgrade historical evidence.
These commands inspect isolated research artifacts and do not authorize collection.

## Bounded curved execution audit and demo replay

These are isolated diagnostics; running them requires separate bounded authorization.
Freeze an explicit schedule, for example
`python scripts/collection/run_curved_audit.py --freeze --schedule v3 --manifest /absolute/new/manifest.json`.
Execution uses that frozen schedule; an explicitly supplied `--schedule` must match.
On the rig, provide the manifest, new `--out`, pinned `--wda-revision` and optionally
`--segments 10` for an explicitly authorized replacement. v1 is the historical
no-reset schedule; v2/v3 reset before the markers, with v3 adding sample slack.
All retain one-minute recordings, strict foreground guards, native decode checks,
five-point consensus and the independent middle check. Stop failures are recorded
without a second stop RPC; use the documented recovery process before another run.

New builds use full successful v2 execution receipts, not historical v1 logs:
`python scripts/inspect/build_curved_audit.py --manifest /absolute/manifest.json --recordings /absolute/recordings --out /absolute/new/viewer`.
An authorized replacement uses `--replacement /absolute/replacement-manifest.json
/absolute/replacement-recordings 10`. Its complete sample conditions and payloads
must match; schedule slack and calibration resets may differ. Serve only the viewer.
Import with `python scripts/inspect/report_curved_audit.py --mapping
/absolute/recordings/audit-private-source-map-v2.json --assessments
/absolute/export.json --media-root /absolute/viewer --out /absolute/new/results.json`.
The importer verifies JPEG bytes. Historical v1 imports require `--allow-legacy`
and report weaker provenance. Only explicitly saved ratings count; device and park
remain confounded. Shut the temporary viewer server down after the report is returned.

`replay_demo_clip.py` defaults to `--mode scheduled --schedule-mode paths`:
a single indexed-path record. Choose `--mode scheduled --schedule-mode records`
explicitly for separate on-device records, or `--mode separate` for serial HTTP
requests. `--anchor X,Y` requires scheduled paths. These modes retain the negative
replay findings in CURVE-AUDIT-20261003; none certifies curved/spin collection.
Scheduled path starts are intended device offsets, not measured physical delivery.
Every post-start exit attempts one retrieval, saving partial video, request/response
and failure details before disconnecting. No recorder-start or stop retries occur.
