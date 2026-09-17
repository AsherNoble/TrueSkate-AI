# Recent research and engineering journal

Keep at most 30 dated entries. Before trimming, tag the complete version and
append its commit-pinned recovery entry to [ARCHIVE.md](ARCHIVE.md).
Put substantial experiments in individual records and current facts in STATUS.

## 2026-09-17 — Gesture starts protected from XR controls

- XR2 mapping produced a versioned 414×896 control-start exclusion profile with
  a 16-point margin. Only moving gestures' touch-down positions are excluded;
  their paths and endpoints may cross controls. Taps and holds remain excluded.
- The bottom Me–Settings row is transient: it appears about one second after the
  stationary board, a reset or a new park, and disappears or stays absent during
  motion. The current profile deliberately blocks it at all times. Its exclusion
  may become state-dependent later only if the lost sampling area matters.
- The preregistered XR2 contamination run failed after 2/300 swipes: a Camera
  probe began outside the 16-point exclusion but opened Replay as it moved into
  the control. The start-only rule is not ready for collection; evidence is
  preserved in [M1-CONTROL-20260917](experiments/M1-CONTROL-20260917.md).

## 2026-09-13 — Instrumented-WDA onset alignment validated

- WDA's internal `submitted_to_ios` timestamp aligns gesture onsets to the video
  within one frame (28/28 across four short XR2 recordings; true jitter ≤ ~13 ms,
  the 30 fps labelling floor) and beats the uninstrumented host send-clock by up
  to ~7 frames. Red-team CONFIRMED.
  [ALIGN-20260913](experiments/ALIGN-20260913-wda-onset.md).
- A multi-anchor recording (run09) then showed one anchor does NOT hold over a
  ~23 s clip: a per-recording linear timebase drift (~−1900 ppm) reaches 1.28
  frames by +22.6 s. A ≥2-anchor offset+rate fit collapses it back to ≤ 0.58
  frame — so long recordings need two anchors (start and end), re-fit per clip.
- Scope: one device, park and session. Not yet certified: cross-park/session/
  device generalisation or the drift-rate distribution across recordings.
- Run10 extended the test to 59.10 s. All eight held-out gestures pass one frame;
  the held-out middle calibration misses by 38.24 ms. The user accepted this
  practical result for rebuilding the linear corpus.
- The local-background V2 onset detector matches 23/24 sampled swipe starts
  exactly on reused development labels. Fixed centre controls avoid its observed
  moving red-floor edge case.
- Canonical linear collection now uses separate start/end 50 ms controls at exact
  screen centre, WDA submission timestamps and an affine video-time fit. Controls
  stay in manifests and are never emitted as training clips; in-recording resets
  and incomplete timing reports reject the segment. Offline tests pass.
  [Timing audit](experiments/M1-TIMING-20260912.md).

## 2026-09-12 — Human onset timing audit

- Preserved three original-recording annotation sets, manifests, frame timestamps
  and findings; false calibration detections and a within-recording timing shift
  motivated the WDA experiments. Recorded the user-confirmed spin exclusion.
- Bounded timing repeats rejected host-call and return-based alignment. Added the
  reusable frame viewer. [M1-TIMING-20260912](experiments/M1-TIMING-20260912.md).

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
