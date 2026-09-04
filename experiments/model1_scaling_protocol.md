# Model 1 linear scaling protocol

Frozen 2026-09-04. This operationalises the scaling/generalisation plan without
authorising collection or Modal spend.

## Experimental authority

- Directory order is never an experimental split.
- Freeze training, validation, challenge, and certification cohorts with
  `scripts/data/build_model1_scaling_manifests.py cohort`.
- Holdout cohorts require device, park, session, date, accepted calibration,
  clean contamination markers, and exact-command uniqueness.
- Build the training ladder with the `subsets` command. It produces one stable,
  device/park round-robin order; every larger rung contains every smaller rung.
- Bind a training subset and frozen validation cohort with the `experiment`
  command. Add certification only after architecture and hyperparameters freeze.
- `BasicLinearClipDataset` verifies every sample's stored content SHA-256 before
  a manifest-driven run. Resume also binds the experiment fingerprint.
- Challenge (SLS 2016 Munich) is evaluated separately and never used for model
  selection. Once inspected for development, it becomes validation and is
  replaced by a new unseen-park challenge cohort.

## Rungs and decisions

- Sizes: 13,100; 26,200; 52,400; 104,800; 209,600; then doublings.
- Seeds: 0, 1, 2, fixed split seed and recipe.
- Select checkpoints on validation only. Use late-epoch validation
  distributions, not the best single epoch, for the learning curve.
- Continue exactly one doubling while error falls.
- Diagnose a plateau after two consecutive doublings each reduce mean
  validation error by less than 20%.
- If training is low-error while validation stalls, increase domain/session
  diversity. If both stall, test base channels 32 then 64 at the largest rung.
- Gradient clipping is selected only if late-epoch variability falls by at
  least 30% and mean recovery falls by no more than one percentage point.

## Storage and execution

- Pack each sealed experiment with `scripts/data/build_model1_shards.py`.
- Shards preserve original H.264/PNG bytes and are bounded by sample count and
  bytes. Modal verifies each archive, stages it to ephemeral SSD, then runs the
  ordinary loader. Directory-backed and staged-shard tensors must match.
- Use `cache_frames=false` with shards. This avoids the >64 GiB decoded-frame
  cache at 26.2k while removing per-frame Modal Volume/FUSE traffic.
- The Modal wrapper retries timeout/internal-provider interruptions against the
  atomic per-epoch resume checkpoint. A retry can repeat only the in-flight
  epoch.
- Scaling runs enable `record_train_metrics`; checkpoints retain full train and
  validation metrics, confidence bounds, gradient summaries, throughput, and
  accelerator identity.

## Cost estimate before paid rungs

Source rates: Modal public pricing retrieved 2026-09-04. L4 is $0.7992/GPU-hour;
the current function's 64 GiB memory plus default 0.125 CPU brings estimated
total billed compute to $1.316583/hour. Runtime anchor is the observed 13.1k
40-epoch run: 8.42 hours per seed. Costs below assume linear time in sample
count and three seeds.

| training clips | hours/seed | GPU only | billed compute |
|---:|---:|---:|---:|
| 13,100 | 8.42 | $20.19 | $33.26 |
| 26,200 | 16.84 | $40.37 | $66.51 |
| 52,400 | 33.68 | $80.75 | $133.03 |
| 104,800 | 67.36 | $161.50 | $266.06 |
| 209,600 | 134.72 | $323.00 | $532.11 |

The earliest defensible answer needs the next two doublings (26.2k and 52.4k):
**$121.12 GPU-only / $199.54 estimated billed compute**. Request an approval
ceiling of **$299.31** (1.5x) because shard staging and per-epoch training
evaluation have not yet been benchmarked at 26.2k. Stop at 52.4k if both
relative error reductions are below 20%; otherwise approve only the next rung.

Running through 209.6k without an earlier plateau is $605.63 GPU-only / $997.71
additional billed compute beyond the completed 13.1k rung. This is not a single
approval request: each rung gets a fresh estimate and explicit approval.

Reproduce with:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/train/estimate_model1_scaling_cost.py
```

## Certification

- Tune thresholds and checkpoints only on validation.
- Certify linear, curved, and curved+spin independently on at least 30,000
  untouched samples each.
- Pass only when the exact one-sided 95% Clopper-Pearson lower bound is strictly
  above 99.9%. At n=30,000, 20 failures pass and 21 fail.
- Curves require five fixed-time path points within 0.03 and duration within
  0.10 s. Spin additionally requires exact active state, onset/liftoff within
  two 30 fps frames, and no extra, missing, merged, or lost overlap track.
