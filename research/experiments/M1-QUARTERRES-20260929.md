# M1-QUARTERRES-20260929 — Does quarter resolution (32×72) still hold?

Status: preregistered (2026-09-29), operator-approved single seed (~$1.3).

## Question

[M1-HALFRES](M1-HALFRES-20260928.md) found 64×144 no worse than 128×288
(three-seed last-10 mean 87.11% vs 84.68%). Where is the floor? If quarter
resolution also holds, input resolution is far from the bottleneck, and
development runs get cheaper again. If it fails, 64×144 is the floor.

## Design

- **Changed:** only `--image-width 32 --image-height 72`.
- **Identical** to `good13100_halfres_cosine_seed0`: code `7a0a7d1`, the same
  shards and experiment manifest, cosine lr, temporal mixer, 40 epochs,
  batch 8, lr 1e-3, 16 base channels, seed 0, sequential decode, frame cache,
  L4. Run label `good13100_quarterres_cosine_seed0`, `--max-hours 3`.
- **Test partition untouched** (`--no-evaluate-test`).
- **Pre-flight:** a local forward/backward pass at 32×72 succeeded.

## Risk stated in advance

The encoder has one stride-2 stage, so the score map is 16 cells wide
(~0.0625 each). That is twice the 0.03 recovery tolerance. The run can only
succeed if soft-argmax localises well between cells.

## Decision rule (fixed before launch)

The metric is the last-10-epoch mean validation recovery, against 64×144
cosine seed 0 (**89.66%**).

| Last-10 drop vs 89.66% | Reading | Next |
|---|---|---|
| ≤ 2 pp | Quarter resolution holds | Offer seeds 1–2 (~$2.6) before adopting |
| 2–5 pp | Ambiguous | Keep 64×144; report |
| > 5 pp | Floor reached | Keep 64×144 as the development default |

Also reported (descriptive): best epoch and value, the end component, and
per-park failure rates if a validation autopsy is run.

## Result (2026-09-29) — floor reached; keep 64×144

`good13100_quarterres_cosine_seed0` (code `7a0a7d1`, L4; the payload
confirms 32×72 and cosine; checkpoint SHA-256 `39cdeeb8…`). Epoch 1 took
358 s (decode and cache); later epochs took ~38 s. Total ≈ 0.6 h ≈ **$0.8**.
Test partition untouched.

| Validation, seed 0, cosine | 64×144 | **32×72** |
|---|---:|---:|
| Mean of last 10 epochs | 89.66% | **82.00%** (−7.66 pp) |
| Best (epoch) | 89.92% (40) | 82.39% (34) |
| Start / end / duration component at best | — | 95.1% / 86.2% / 97.2% |

- **Decision (fixed rule):** the drop is > 5 pp, so **64×144 stays the
  development default**, and 32×72 is below the floor. No further seeds.
- The curve had flattened by epoch 31 (81.98–82.39% over the last 10), so
  this is not undertraining.
- Median end error is 0.0138 (start 0.0111), well inside the 0.03 tolerance,
  so the drop is in the tail. That fits the predicted risk: a 16-cell-wide
  score map (~0.0625 per cell) makes soft-argmax interpolate across
  two-tolerance-wide cells.
- **Reading:** resolution stops being free somewhere between 64 and 32 wide.
  One seed only, but the gap is about 4× the 64×144 seed spread (1.9 pp).
