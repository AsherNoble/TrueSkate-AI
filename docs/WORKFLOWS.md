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
It uses one-minute segments, three leading calibration controls, per-segment
reset and strict aligned video. Use separate outputs for validation.

For mixed/hold collection, inspect the collector's `--help`; preserve its
guards and metadata. Label `--park-label` honestly. Keep all trainable samples
separate from calibration controls and exclude contamination markers.

## Corpus and scaling tools

`scripts/data` contains corpus audit/filtering, cohort manifest, shard and
trick-library tools. [The scaling protocol](../research/protocols/model1_scaling.md)
contains the exact frozen experimental procedure. [Trick-library provenance](../trick_libraries/README.md)
explains the retained CMA-ES outputs and their BC use.
