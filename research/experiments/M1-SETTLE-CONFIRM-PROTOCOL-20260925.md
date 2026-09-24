# M1-SETTLE-CONFIRM-PROTOCOL-20260925 — Proposed confirmatory settle test

Status: **proposal only; not run.** It needs operator approval, since it uses
device time and emits isolated (non-corpus) clips. It follows the red-team
recommendation in [M1-SETTLE-20260925](M1-SETTLE-20260925.md).

## Question

In the production single-tap linear configuration, does waiting for a still
centre before the start control reduce start-anchor errors of the production
detector (A) in SLS 2015 Los Angeles?

## Design

- **Configuration:** the unchanged production launcher flags
  (`scripts/ops/mvp_collect_linear.sh`: single 50 ms centre controls, aligner
  on, exact-PTS extraction). There is no `--die-five-experiment`. Output goes
  to an isolated `tmp/` directory with `--retain-mov`, never the corpus.
- **Arms,** interleaved per segment, with the order in each consecutive pair
  set by a seeded coin (seed recorded before running):
  - ON: `--start-settle-threshold 1.5 --start-settle-max-s 15 --start-settle-required`
  - OFF: production default, 1.5 s post-reset sleep only.
- **Size:** 20 segments per arm per XR (80 segments, ~45 min with both XRs in
  parallel). This is the maximum; there is no extension to chase significance.
- **Logging:** `pre_segment_reset_epoch_s` (commit after `0e73bd6`), the WDA
  start-control submission time, `start_settle`, and the pre-start centre
  motion from video.

## Outcomes

- **Primary:** human-labelled start-anchor error of A, counted as a gross error
  when A's frame is more than 1 frame from the first visible frame. Labels are
  blind, arm-hidden and shuffled, using the die-five label viewer builder on
  single markers.
- **Secondary:**
  - A within ±1 frame of the human label;
  - two-anchor fit `abs(rate − 1) > 0.002`;
  - ON-arm skip rate and settle time;
  - the aligner's strict-admission count per arm;
  - reset-to-start-control delay per arm.
- **Mechanism check:** the proportion of starts with pre-start motion >1, per
  arm.

## Analysis and decision

Compare arms with a one-sided Fisher exact test on primary gross errors and
report the full 2×2 per device.

- Because the base rate may be low (~3% anomalies, ~10% one-frame A/B
  disagreements the same day), a null result with 0 OFF-arm errors is
  **inconclusive**, not evidence against the wait.
- Adopt the flags for Los Angeles collection only if ON has no more gross
  errors than OFF, has ≤10% skips, and removes moving starts. A significant
  error reduction would make the claim CONFIRMED.

## Not included

- A forced-fault arm (firing the start control earlier in the post-reset
  transient). The collector cannot currently go below the 1.5 s reset sleep,
  and adding one is a further code change.
- Other parks.
