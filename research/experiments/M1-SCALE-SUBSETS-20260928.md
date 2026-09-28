# M1-SCALE-SUBSETS-20260928 — Does more clean data still help?

Status: plan only. It runs after M1-RETRAIN-GOOD13100 finishes and its test
headline is recorded. No Modal spend is authorised by this record.

## Question

At the 80.05% recipe on clean labels, is linear Model 1 still data-limited?
If it is, collect a doubling of the training set (~9,170 clips, about 1.5–2
days of rig time). If it is not, the ceiling is elsewhere (model capacity, the
32-frame 288×128 input, or label resolution), and more data is not the next
step.

## Design

This follows the [scaling protocol](../protocols/model1_scaling.md), downward
from the existing rung instead of upward.

- **Rungs:** nested training subsets of the good13100 training partition
  (9,170 clips): **2,293 (25%)**, **4,585 (50%)** and the full 9,170.
  - The 9,170 rung is the three completed M1-RETRAIN seeds (the same clip
    set), so it costs nothing new.
  - Built with `subsets --order proportional`: every prefix keeps the full
    set's device/park mix within one clip per stratum. The existing
    `balanced` order equalises strata, which would give the small rungs a
    different park mix from the full set and confound the curve.
- **Held fixed:**
  - the recipe (40 epochs, batch 8, lr 1e-3, 16 base channels, temporal
    mixer, the same priors, sequential decode, frame cache, L4);
  - model seeds 0, 1 and 2;
  - the **same 1,965-clip validation partition** for every rung.
- **Test untouched:** the subset experiment manifests carry no test
  partition. The scaling decision uses validation only.
- **Training-set check:** score the checkpoint selected in M1-RETRAIN once on
  its own 9,170 training clips
  (`evaluate_partition_once --label traincheck --partition train`).

## Metric and decision rule (fixed before training)

For each rung, the metric is the mean across seeds of each seed's mean
validation recovery over the **last 10 epochs** (the protocol's late-epoch
measure, not the best epoch). Validation error is 1 − recovery. The relative
error reduction of a doubling is (e_small − e_large) / e_small.

| 4.6k → 9.2k reduction | 2.3k → 4.6k reduction | Decision |
|---|---|---|
| ≥ 20% | any | **Still data-limited:** collect a doubling, adding all new clips to training with validation and test fixed. |
| < 20% | ≥ 20% | **Flattening:** test capacity first (base channels 32 at 9.2k, per the protocol), then decide on collection. |
| < 20% | < 20% | **Plateau** (the protocol's rule): do not collect. Test capacity and input resolution. |

The training-set check is read alongside the table:
- **Training recovery ≥ validation + 10 pp:** the model memorises what it
  sees, which supports more data.
- **Training within 3 pp of validation:** the model can't fit even its own
  training data, so it is capacity- or input-limited. More data at the same
  size will not help much.

**Known bias:** at a fixed 40 epochs, the small rungs get proportionally
fewer optimiser steps, so they may be undertrained. That makes more data look
more helpful than it is. If a small rung's validation is still rising over
its last 10 epochs, that is reported. This bias is accepted because the
decision it could wrongly favour (collecting) costs only about two days of
rig time.

## Cost (L4 at $1.3166/h)

The full rung measured ~560 s per epoch. Assuming ~80 s of that is the fixed
validation pass:

| Rung | ~s/epoch | ~h per seed | ~$ per seed | 3 seeds |
|---|---:|---:|---:|---:|
| 2,293 | 200 | 2.3 | 3.0 | 9.0 |
| 4,585 | 320 | 3.6 | 4.8 | 14.4 |
| Train check | — | 0.2 | 0.3 | 0.3 |
| **Total** | | | | **≈ $24** |

**Approval requested: ceiling $35** (≈1.5×). Guards as in M1-RETRAIN: a
single provider retry, `--required-gpu L4`, a finished label refused, and
`--max-hours` set to 1.5× the projected run per rung (3.5 h and 5.5 h). The
six runs go in parallel, about 4 h of wall time.

## Steps

1. Rig: `subsets --cohort <split>/…train.json --sizes 2293,4585,9170
   --order proportional --seed 0`, then `experiment --train <subset>
   --validation <split>/…validation.json` for 2,293 and 4,585 (no
   certification). Run `check` for zero leakage.
2. Build shards for each experiment and upload them to new subdirectories of
   `trueskate-model1-good13100-20260927`.
3. After approval, launch the six runs (`good13100_n{2293,4585}_seed{0,1,2}`).
4. Run the training-set check on the selected M1-RETRAIN checkpoint.
5. Apply the decision table and record the results here and in
   STATUS/JOURNAL.
