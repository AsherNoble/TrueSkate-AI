# Recent research and engineering journal

Keep at most 30 dated entries. Before trimming, tag the complete version and
append its commit-pinned recovery entry to [ARCHIVE.md](ARCHIVE.md).
Put substantial experiments in individual records and current facts in STATUS.

## 2026-09-08 — BC reconciliation and repository reorganisation

- Preserved original BC, old main, rig committed history and rig uncommitted
  source/service definitions on GitHub. [Disposition ledger](RIG_RECONCILIATION.md).
- Preserved useful dashboard and all BC variants; extracted shared gesture,
  device and clip utilities, and separated canonical workflow entrypoints.
- Baseline alignment test exposed FFmpeg 9 removal of `-vsync`; replacing it
  with `-fps_mode passthrough` restored the original 248-test suite before moves.
- Local and clean Linux checks pass; staged dashboard HTTP checks pass.
  Both XRs passed bounded calibrated collection and strict loader/frame-count
  checks after an explicit operator-confirmed idle-navigation allowance.
  See [migration validation](MIGRATION_VALIDATION.md)
  for exact revisions, evidence and remaining rollout gates.

## 2026-09-04 — Model 1 evaluation and scaling protocol

- Recorded 80.05% complete-gesture test recovery after validation-only seed
  selection. [M1-20260904](experiments/M1-20260904.md).
- Added frozen manifests, nested subsets, shards, certification and scaling
  analysis. No new paid tranche was authorized. [Protocol](protocols/model1_scaling.md).
