# MODEL1-TRICK-MVP-20261016 — trick Model 1 MVP v0

- **Question:** can a model read 1–4 fast strokes per two-second clip from 60 fps video?
- **Result:** yes, roughly. On the test clips, scored once, 99% of strokes are found on time at both
  ends; the worst point of each path is within 5% of the screen for 90% and within 2% for 37%.
- **Decision:** v0 clears the plan's "half the strokes" bar if the kickflip tolerates 5%, not if it
  needs 2%. Item 1 (your kickflip position-tolerance review) picks the branch.
- **Next:** at 5%, item 5: label your filmed kickflip with v0 and replay it. At 2%, fix the stroke
  ends first, then retrain on the 1,000-clip batch.
- **Link:** code on branch `research/model1-mvp-v0`; bulk output in the laptop evidence folder
  `MODEL1-TRICK-MVP-20261016/` (`INDEX.md`).

Explore mode; plan item 4, approved 2026-10-16 with Modal capped at US$5.

## What ran

- **Model** (`model1/trick/model.py`): curved v1's network (picture encoder, time layer and finger
  heatmap, unchanged). Its one touch-down/lift choice per clip became three yes/no signals per
  frame: in contact, touch-down here, lift here.
- **Decoder** (`model1/trick/decoder.py`): the best set of up to 4 non-overlapping strokes of
  50–600 ms for those signals, with sub-frame touch-down and lift times. Each stroke's positions are
  resampled every 5 ms and fitted with curved v1's straight-piece search, set to 1–4 pieces of at
  least 40 ms.
- **Data** (`model1/trick/dataset.py`): the 250 admitted trial clips (TRICK-SPEED-20261015, XR1 and
  XR2, labels re-timed to the trail). Split by minute, never by phone: minutes 1–5 train (182 clips,
  402 strokes), minute 6 validation (28, 74), minute 7 test (40, 78). Both phones' copy of a seeded
  minute share its split. New roots, such as the 1,000-clip batch, add in through `freeze`.
- **Training:** one Modal L4, seed 0, AdamW 1e-3. Two runs: 64×144 frames (curved v1's size), batch
  8, 100 epochs; and 128×288, batch 4, 60 epochs. Each run's checkpoint is chosen on validation:
  strokes passing at 5%, then timing, then path error. The same rule chose between the runs, and
  the test split was scored once, on the chosen one.

## How it is scored

Each labelled stroke is matched to the nearest predicted stroke, with both ends within 250 ms.
- **On time:** touch-down and lift both within 2 frames (33 ms) of the label.
- **Path:** positions compared at the same fraction of each stroke's duration, as Euclidean distance
  in normalized screen units (x/414, y/896), the kickflip tolerance test's metric. The worst point
  is the largest distance along the stroke.
- **Pass at a level:** on time, and the worst point within the level (2% or 5% of the screen).

## Choosing the run (validation, 74 strokes)

| Run | Chosen epoch | On time | Pass at 5% | Pass at 2% | Count right |
|---|---:|---:|---:|---:|---:|
| 64×144, batch 8 | 97 of 100 | 99% | 84% | 39% | 26 of 28 clips |
| **128×288, batch 4** | **46 of 60** | **97%** | **86%** | **49%** | **28 of 28** |

Both runs reached about 80% at 5% by epoch 30 and then stayed there. The 64×144 training clips kept
improving (92% and 53% at epoch 97), so more data should help a little.

## Error distributions of the chosen run

| Measure | Split | 10th pct | Median | 90th pct | Share within |
|---|---|---:|---:|---:|---|
| Touch-down error (ms) | validation | −10.0 | +0.2 | +8.3 | 93% within 1 frame, 99% within 2 |
| | test | −7.8 | −0.9 | +7.3 | 99% within 1 frame, 100% within 2 |
| Duration error (ms) | validation | −17.1 | −1.8 | +12.5 | 96% within 2 frames |
| | test | −16.5 | −0.3 | +11.1 | 97% within 2 frames |
| Mean path error (% of screen) | validation | 0.5 | 1.1 | 2.6 | 84% ≤2%, 97% ≤5% |
| | test | 0.4 | 1.0 | 2.0 | 90% ≤2%, 99% ≤5% |
| Worst path point (% of screen) | validation | 0.9 | 2.0 | 5.4 | 50% ≤2%, 89% ≤5% |
| | test | 1.0 | 2.6 | 4.9 | 38% ≤2%, 91% ≤5% |
| End point (% of screen) | validation | 0.4 | 1.4 | 4.3 | 61% ≤2%, 95% ≤5% |
| | test | 0.5 | 1.6 | 4.5 | 57% ≤2%, 94% ≤5% |

- **Pass:** validation 86% of strokes at 5% and 49% at 2%; test 90% and 37%. Every stroke in a clip
  passes in 68% (validation) and 80% (test) of clips at 5%, but only 14–15% at 2%.
- **Count:** right in all 28 validation clips and 39 of 40 test clips; the test miss is a catch
  (a still hold). No extra strokes.
- **By family, test pass at 5% / 2%:** flick 100% / 30% (20), pop 94% / 50% (18), catch 92% / 83%
  (12), push 100% / 25% (4), random 75% / 12% (24). Both phones alike (92% and 87% at 5%).

## Where it fails

- **Stroke ends.** On validation, a worst point over 2% is the end in 20 of 37 strokes, the middle
  in 9 and the start in 8. Predicted ends fall short along the last piece, pops most (median 13 pt;
  89% short). That matches the trail check's pops whose visible trail stops short
  (TRICK-SPEED-20261015), so labels and picture disagree at the end. On the 64×144 run the shortfall
  grew with speed, about 7 ms of last-piece motion. Extending each end by that much (fitted on the
  training clips) lifted validation from 39% to 43% at 2%. That calibration isn't in v0.
- **Many pieces in little time.** A 4-piece, 171 ms stroke came out about 20% off on both phones;
  4-piece strokes passed 4 of 8 at 5% across validation and test, and none at 2%.

## Cost

Modal, from the billing report: US$0.06 smoke run, US$0.36 for the 64×144 run, US$0.85 for the
128×288 run and US$0.01 for the test pass; US$1.28 in all, against the US$5 cap. The two frame
bundles (1.5 GiB) stay on Modal volume `trueskate-trick-model1` for the 1,000-clip retrain.
