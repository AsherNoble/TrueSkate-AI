# M1-RETRAIN-GOOD13100 — Controlled retraining on the 13,100 good-label corpus

Status: plan only. No Modal spend or upload is authorised by this record.

## Question

With the model, recipe, corpus size, park mix and split method held fixed,
does training on audited clean timing labels improve linear Model 1 recovery
over the 80.05% baseline (M1-20260904)?

## What stays identical (from the seed-0 checkpoint and the archived journal)

- **Model:** `basic_linear_regressor_v3_separate_endpoint_heads_tight_start_prior`,
  temporal mixer on, 16 base channels, 32 frames at 288×128, 2 knots.
- **Timing priors:** start onset 0.24 with sigma 0.05; end onset 0.24.
- **Losses:** no map or trajectory losses, no line fit, no gradient clipping.
- **Training:** AdamW at lr 1e-3 (the default; the checkpoint does not store
  lr), batch 8, 40 epochs, L4 pinned (`MODAL_TRAIN_GPU=L4`).
- **Split:** by exact command, seed 0, 70/15/15. The manifest's command key is
  the trainer's, and `_split_by_key` depends only on the set of keys and the
  seed, so the split is reproduced exactly from the manifest.
- **Seeds and selection:** model seeds 0, 1, 2. Select the seed and epoch on
  validation only (the original picked seed 0 at epoch 37 with 80.00%; seeds 1
  and 2 had 78.42% and 75.83%). Score the chosen checkpoint once on test.

Only the data differ: `model1_linear_good13100_20260927` (fingerprint
`sha256:9fadddee…`, [M1-CORPUS-AUDIT](M1-CORPUS-AUDIT-20260927.md): 200/200
audited clips satisfactory). A known deviation is data loading: shards with
`cache_frames=false`, where the original read a directory with caching. The
protocol requires identical tensors either way, and a smoke check verifies it
(step 3).

## Outcomes (fixed before training)

1. **Headline:** the new model's test recovery against 80.05%. The test sets
   differ (the new one has clean labels), so this mixes better training with
   fairer scoring. At n ≈ 1,965 the 95% CI is about ±1.8 pp.
2. **Controlled (primary):** the old 80.05% checkpoint and the new selected
   checkpoint are both scored on the **same new clean test partition**.
   - A per-clip paired McNemar test isolates the effect of training data.
   - Allowed only if no new test command appears in the old training set.
     Check exact command keys against the local historical corpus; drop any
     overlap from both scores and report it.
3. **Secondary:**
   - the mean of the last 10 validation epochs per seed (old seed 0: 0.7717);
   - start, end and duration recovery;
   - endpoint and duration errors;
   - spread between seeds.

The hypothesis is supported if outcome 2 favours the new model at p < 0.05.

## Steps

1. **Code (branch, tested):**
   - a `command-split` subcommand turning the cohort into
     training/validation/certification cohorts with the trainer's seed-0 split
     (tested for equality with `split_by_command`);
   - an evaluator that scores any checkpoint on an experiment manifest's test
     partition and writes per-clip results for McNemar.
2. **Build on the rig:** `experiment` manifest → `build_model1_shards.py` →
   upload to a new subdirectory of the corpus volume
   (`model1_good13100_20260927/`).
3. **Smoke:** a 1-epoch run on shards. Directory and shard tensors must
   match, and a small batch must decode.
4. **Train:** seeds 0–2 in parallel on L4 with `record_train_metrics`, test
   evaluation disabled during training, and per-epoch resume.
5. **Select on validation, test once, cross-evaluate** the old checkpoint on
   the same test, then run the red-team review before journaling.

## Cost and time

The anchor is 8.42 h per seed at 13.1k on L4 ($1.3166/h billed). Three seeds
cost about **$33**, running in parallel for about 9 h wall time. Sharded
loading without a frame cache has not been benchmarked at this size.
**Approval requested: ceiling $50** (1.5×), for training plus smoke and
evaluation. Per-epoch resume bounds the loss from any provider retry.

## Approval and preparation (2026-09-27)

The operator approved the plan and a **$50 Modal ceiling**, and asked for a
red-team code review before any spend. The prior ~$10 loss was the first
13,100-clip L4 seed: it hit the 8 h function timeout at epoch 38 with no
checkpoint (fixed afterwards by per-epoch resume and a longer timeout).

Done offline, with no spend:
- **Code** `d0b0171`:
  - `command-split`, tested equal to the trainer's `split_by_command` for
    seeds 0 and 3;
  - `evaluate_partition_once`, a manifest-partition evaluator writing per-clip
    results; it refuses to overwrite an existing label.
- **Split** (rig `tmp/final-manifest-20260927/split/`): train 9,170,
  validation 1,965, test 1,965, the same sizes as the original. `check`
  reports zero leakage.
- **Experiment manifest:**
  `experiment.model1_retrain_good13100_20260927.json`, `sha256:eb9b022e…`.
- **Old checkpoint:** the restored 80.05% seed-0 checkpoint loads strictly
  into the current `BasicLinearRegressor` and runs forward.
- **Overlap:** 0 of the 9,170/1,965/1,965 new commands appear among the
  13,100 old corpus commands, so cross-evaluation is valid.

## Runbook (to be reviewed before any spend)

All Modal commands run on the rig from the staged code (`d0b0171`) with
`MODAL_CORPUS_VOLUME=trueskate-model1-good13100-20260927` and
`MODAL_TRAIN_GPU=L4`.

1. `modal volume create trueskate-model1-good13100-20260927`.
2. Shards:
   `build_model1_shards.py --data <runtime root> --experiment-manifest <exp> --out-dir <F>/shards`,
   then `modal volume put <vol> <F>/shards /model1_good13100_20260927`.
3. **Smoke (1 epoch):** `modal run scripts/model1/train_basic_linear_modal.py
   --data-subdir model1_good13100_20260927 --shard-manifest-name shards.json
   --no-cache-frames --temporal-mixer --epochs 1 --no-evaluate-test
   --record-train-metrics --seed 0 --run-label good13100_smoke_e1`.
   - Record the epoch time and project cost.
   - **Budget gate:** launch only if 3 seeds × 40 epochs at the measured rate
     stays under the remaining budget.
4. **Train:** seeds 0–2, same flags with `--epochs 40`, run labels
   `good13100_20260927_seed{0,1,2}`, launched detached on the rig.
5. **Select on validation,** then run
   `evaluate_partition_once --label test_once` on the chosen checkpoint, and
   `--label crosseval_good13100` on `basic_linear_model1_recovered_20260903_seed0.pth`.

## Red-team review and changes (2026-09-27)

Three read-only red-team reviews (Modal execution, data/split, evaluation)
all returned FIXABLE.

**Fixed in `bc291da`/`090011b`:**
- **Spend guards:**
  - `max_hours` stops after a committed epoch once the projected run exceeds
    the cap;
  - a single provider retry (was 6 × 24 h);
  - a finished run label is refused (no retrain or overwrite);
  - a resume at the final epoch finalises;
  - `--required-gpu L4` is checked in the container before any work.
- **Pinned image:** torch 2.12.0, opencv-python-headless 4.13.0.92 (an
  unpinned build now resolves OpenCV 5.x), numpy 2.4.6, scipy 1.17.1. Versions,
  decode mode and checkpoint SHA-256 are recorded in outputs.
- **Decode:** exact-PTS clips hold one keyframe, so per-index seeks cost
  ~1.8 s/clip on the rig, against 0.09 s for a sequential pass, and
  `CAP_PROP_FRAME_COUNT` can under-report by one.
  - An explicit `decode_mode="sequential"` is used for this training; the
    default stays `"seek"`, so old checkpoints evaluate as before.
  - The frame cache is allowed with shards up to 14,000 samples, restoring
    the original recipe's cached loading. `record_train_metrics` is off, as in
    the original run.
- **Test discipline:** `evaluate_test_once` can no longer overwrite the
  historical file (a labelled re-score writes its own file).
  `evaluate_partition_once` requires an explicit partition.

**Scope decision (operator, after debate):**
- **Primary:** the headline, the new test recovery against 80.05% under the
  identical split protocol (95% CI for the difference ≈ ±2.5 pp), reported
  alongside all three seeds' spread.
- **Dropped as primary:** the old-vs-new cross-evaluation on the new test,
  because both it and the reverse design are confounded by sessions shared
  across the command split and by old-clip frame geometry.
- **Known properties:** the command split shares recordings across partitions
  (as in the 80.05% run), so absolute recovery overstates performance on
  unseen recordings; certification should hold out whole sessions. A gain
  reflects the whole new pipeline (clean timing plus correct frame geometry),
  not labels alone.

## Smoke and reproduction (2026-09-27)

- **Upload:** 13,100 samples in 26 shards (649 MB,
  `sha256:4c2bf407…`) on the new volume
  `trueskate-model1-good13100-20260927`.
- **Reproduction check** (volume `trueskate-corpus-v2`,
  `model1_linear_20260902`): the old seed-0 checkpoint scored with
  `evaluate_test_once --label repro_20260927` in the pinned image gives
  **0.8005089 test** (validation at training time 0.8000). This is identical
  to the historical result, so the environment does not move the benchmark.
- **Smoke** (`good13100_smoke_e2`, 2 epochs, L4, pinned image, sequential
  decode, cache on): epoch 1 took 881 s (including decode and cache), epoch 2
  took **561 s**. Projection: 881 + 39 × 561 s ≈ 6.4 h ≈ $8.4 per seed, about
  $25 for three seeds.

## Training launch (2026-09-27)

Seeds 0, 1 and 2 run as `good13100_20260927_seed{0,1,2}`, detached from the
rig with code `090011b`. Settings: 40 epochs, `--max-hours 8.5` (≤ $11.2 per
seed), `--provider-timeout-retries 1`, `--no-evaluate-test`,
`--no-record-train-metrics`. Logs are on the rig under
`tmp/final-manifest-20260927/modal-logs/`.

Budget: smoke + reproduction ≈ $1; training ≈ $25 expected and ≤ $34 at the
caps; evaluation ≈ $1. The worst case stays within the $50 ceiling.
