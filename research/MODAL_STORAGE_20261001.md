# Modal storage audit and deletion — 2026-10-01

## Why

Modal bills Volumes at **$0.09 / GiB / month beyond a free 1 TiB**
(modal.com/pricing, checked 2026-10-01). The account held ≈ 1,293 GiB, so
≈ 269 GiB was billable: ≈ **$24/month**.

## Inventory (recursive file sizes, 2026-10-01)

| Volume | Size | Files | Contents |
|---|---:|---:|---|
| `trueskate-corpus` | **929.4 GiB** | 2,370,292 | 52 sessions, 2026-06-25 → 07-18 (uploaded by 07-18): per-gesture PNG frame folders (24 frames), flick / n-slot / spin mixes |
| `trueskate-corpus-v2` | 360.6 GiB | 890,956 | 63 entries, 2026-08-14 → 09-07, same PNG format (32 frames); one session `iPhone_XR_20260814_042825` is 293.6 GiB; also `model1_linear_20260902` (0.5 GiB, the 80.05% model's corpus) |
| `trueskate-model1-good13100-20260927` | 2.2 GiB | 100 | Sequential shards: good13100, the n2293/n4585 subsets, `model1_expand_20260930` |
| `trueskate-mvp` | 0.4 GiB | 6,613 | August basic-hold / basic-linear XCTest experiments |
| `trueskate-models` | 0.1 GiB | 233 | Checkpoints and evaluation outputs |
| four `trueskate-mvp-linear-*` volumes | ≈ 0.3 GiB | ≈ 18.8k | August linear experiment datasets |

## Deleted: `trueskate-corpus` (operator-approved 2026-10-01)

`modal volume delete trueskate-corpus` was run from the rig. The account now
holds ≈ 364 GiB, under the free 1 TiB, so Volume storage costs **$0**.

**Why it was deletable:**
- **Unusable labels.** Its samples predate tap calibration: `meta.json` carries
  no calibration record and `capture_offset_s` is 0, so onset timing is
  unknown. The current pipeline admits only two-anchor calibrated clips, and
  the operator judged recollection easier than repairing the timing
  (probably impossible).
- **No current consumer.** No active experiment reads it. Old scripts still
  default to the name (`train_trace_extractor_modal.py`,
  `corpus_stats_modal.py`, `modal_volume_space.py`, `review_modal_corpus.py`,
  and the basic-linear and basic-hold Modal defaults); they now fail loudly
  rather than read the wrong data. The installed offloader writes to
  `trueskate-corpus-v2`, not here.

**What was lost, permanently:** ≈ 95k gestures, including spin and multi-slot
kinds that later curved/spin work will need. The offloader had deleted the
rig's local copies after upload, and no independent backup existed
([RIG_STORAGE_20260909](RIG_STORAGE_20260909.md)). Recollect with the current
calibrated pipeline instead.

## Kept

`trueskate-corpus-v2` holds the 80.05% reproduction corpus and is the
offloader's target. The rest total under 3 GiB. No further deletion saves
money while the account stays under 1 TiB.
