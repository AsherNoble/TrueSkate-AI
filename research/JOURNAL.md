# Recent research and engineering journal

Keep at most 30 dated entries. Before trimming, tag the complete version and
append its commit-pinned recovery entry to [ARCHIVE.md](ARCHIVE.md).
Put substantial experiments in individual records and current facts in STATUS.

## 2026-09-13 — Minute-scale two-anchor test

- Run10 captured 59.10 s, with 56.02 s between calibration endpoints. Frozen
  endpoint-only fit: all eight gestures pass one-frame target; held-out middle
  calibration misses at38.24 ms. Initial gate fails (8/9); replications paused.
  Preserved raw labels and full comparison in M1-TIMING-20260912/run10 evidence.
- Exploratory local-background and look-ahead trace detection improves the same
  51 labelled examples from 45 to 47 exact and 45 to 48 within one frame. One moving
  floor graphic still triggers early; this is development evidence, not a holdout.
- Canonical linear collection now uses separate start/end 50 ms controls at exact
  screen centre, WDA submitted-to-iOS timestamps and an affine video-time fit.
  Controls stay in manifests and are never emitted as training clips; in-recording
  resets and incomplete timing reports reject the segment. Offline tests pass;
  device/park validation remains to be run before a larger collection.

## 2026-09-13 — Internal WDA timing prepared

- Pushed tested opt-in WDA instrumentation and prepared the separate-request
  probe. Signed rig build blocked by account/certificate/profile errors;
  original XR2 WDA remains running. Joined-gesture batch experiment abandoned.
  [M1-TIMING-20260912](experiments/M1-TIMING-20260912.md).

## 2026-09-12 — Human onset timing audit

- Preserved three original-recording annotation sets, manifests, frame timestamps
  and findings; false calibration detections and a within-recording timing shift
  require investigation. Recorded the user-confirmed spin-control exclusion.
- Bounded timing repeats and human labels reject reliable one-frame return-based
  alignment; logged WDA responses do not resolve the residual.
- Added the reusable frame viewer. No collection or training changes deployed.
  [M1-TIMING-20260912](experiments/M1-TIMING-20260912.md).

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
