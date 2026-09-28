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

## Amendment (2026-09-28, before launch): 64×144 and the M1-LRDECAY schedule

Operator-approved with M1-LRDECAY ($20 ceiling for both).

- **Resolution:** all rungs run at **64×144**, the development default after
  [M1-HALFRES](M1-HALFRES-20260928.md).
- **Schedule:** the winner of [M1-LRDECAY](M1-LRDECAY-20260928.md) (cosine or
  constant), fixed before these runs launch.
- **Full rung:** the three existing 64×144 runs with that schedule, so no new
  cost.
- **Labels:** `good13100_n{2293,4585}_{schedule}_seed{0,1,2}`, run with
  `--max-hours 2`. The data go to new volume subdirectories
  `model1_good13100_n2293/` and `model1_good13100_n4585/` (subset + the same
  validation partition, no test).
- **Metric and decision table unchanged,** with per-park failure rates added
  for each rung's best-validation checkpoint (validation autopsy).
- **Cost:** ~0.6 h (2,293) and ~1.0 h (4,585) per seed, **≈ $6.3** for all
  six runs.

## Result (2026-09-28) — still data-limited: collect a doubling

Code `7a0a7d1`, 64×144, cosine, L4. Two 4,585 runs hit the old last-epoch
`max_hours` projection after one stalled epoch (287 s and 336 s against
~70 s). They resumed from their snapshots (epochs 13 and 12) with
`--max-hours 4`; the guard now projects from the median epoch (`2455667`).
Test partition untouched.

| Training clips | Last-10 mean validation (seeds 0 / 1 / 2) | **Mean** | Error |
|---:|---|---:|---:|
| 2,293 | 79.26 / 76.17 / 77.76 | **77.73%** | 22.27% |
| 4,585 | 82.38 / 77.77 / 86.01 | **82.05%** | 17.95% |
| 9,170 | 89.66 / 87.99 / 89.86 (M1-LRDECAY) | **89.17%** | 10.83% |

**Relative error reduction per doubling:**
- 2.3k → 4.6k: **19.4%**
- 4.6k → 9.2k: **39.7%**

**Decision (fixed table):** the 4.6k → 9.2k reduction is ≥ 20%, so the model
is **still data-limited. Collect a doubling:** ~9.2k new clips added to
training, with validation and test fixed.
- The latest doubling gave the larger gain, so there is no sign of a plateau
  yet.
- The 4,585 rung has a wide seed spread (77.8–86.0), but even its best seed is
  below the worst full-size seed (88.0).

**Per park, best-validation checkpoint per rung** (validation failure rate;
[evidence](../evidence/M1-SCALE-SUBSETS-20260928/)):

| Park | 2,293 | 4,585 | 9,170 |
|---|---:|---:|---:|
| Kansas City | 32.9% | 22.8% | 16.3% |
| Los Angeles | 30.2% | 26.4% | 21.4% |
| Super Crown | 23.8% | 16.2% | 10.3% |
| Skateboard GB | 15.3% | 6.3% | 5.6% |
| The Workshop | 5.7% | 2.9% | 2.1% |

**Correction to M1-DIAG:** that record argued that data volume does not
explain the Kansas City gap. The curve shows Kansas City improving steadily
with data (32.9 → 22.8 → 16.3%). It is harder, not immune to data. Los Angeles
improves the slowest; it has the fewest clips in the mix (1,261 of 13,100).

**Caveats:**
- At a fixed 40 epochs, smaller rungs get fewer optimiser steps. That favours
  more data, but it affects both doublings, and the second (larger) reduction
  is the one that decides.
- One seed per rung for the per-park table.
- Recordings are shared across partitions.

**Cost:** ≈ $6 for the six runs plus ≈ $0.7 for the three autopsies.
Together with M1-LRDECAY (≈ $7), the total is ≈ $14 of the $20 ceiling.
