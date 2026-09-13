# Recent research and engineering journal

Keep at most 30 dated entries. Before trimming, tag the complete version and
append its commit-pinned recovery entry to [ARCHIVE.md](ARCHIVE.md).
Put substantial experiments in individual records and current facts in STATUS.

## 2026-09-13 — Instrumented-WDA onset alignment validated

- WDA's internal `submitted_to_ios` timestamp plus one calibration touch per
  recording aligns every gesture onset within one video frame (28/28 across four
  XR2 recordings; true jitter ≤ ~13 ms, the 30 fps labelling floor) and beats the
  uninstrumented host send-clock by up to ~7 frames. Red-team CONFIRMED.
  [ALIGN-20260913](experiments/ALIGN-20260913-wda-onset.md).
- Scope: one device, park and session. Next: a multi-anchor recording to confirm
  one anchor transfers across a whole recording before certifying self-labelling.

## 2026-09-09 — Unintended collection autostart retired

- Confirmed login-time collector jobs, the historical restart loop, and a loaded
  laptop agent-based fixer as automatic start/restart paths. Removed the fixer,
  disabled collector/watchdog jobs, and established collection-off intent.
  [Evidence and operating rule](COLLECTION_AUTOSTART_20260909.md).
- XR1 reboot verification is blocked by an Xcode account/provisioning error;
  no unbounded run or healthy XR2 WDA restart was performed.
- Closed out stale merge/deployment gates and separated release runtime storage
  from the retained dirty checkout. Independent backup coverage is not verified;
  neither same-SSD copy may be deleted. [Storage audit](RIG_STORAGE_20260909.md).

## 2026-09-08 — BC reconciliation and repository reorganisation

- Switched the rig's stable source path to clean merged main, preserved the
  entire dirty checkout, and reloaded application services without restarting
  healthy WDA. [Rollout and rollback](RIG_ROLLOUT_20260908.md).
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
