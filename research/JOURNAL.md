# Recent research and engineering journal

Keep at most 30 dated entries. Before trimming, tag the complete version and
append its commit-pinned recovery entry to [ARCHIVE.md](ARCHIVE.md).
Put substantial experiments in individual records and current facts in STATUS.

## 2026-09-21 — Compact-video onset phase corrected

- The operator marked the first visible trace as frame 7 in 35/35 audited clips,
  while synthetic timing placed onset at frame 8. A surviving raw tranche video
  showed both centre-control detections were already on their first visible
  frames and the fitted WDA time missed a swipe onset by only 8.3 ms.
- FFmpeg's default nearest-frame resampling was pulling future source pixels into
  the preceding compact-video slot. Causal upward rounding with an explicit
  zero-time slot moved all seven detector-confirmed preserved swipes from frame
  7 to frame 8 without changing their timing metadata. The extractor and
  regression coverage were updated. The operator chose recollection over repair
  of the 1,109-clip tranche.
- Two bounded validation attempts started underneath an already-open iOS Control
  Center panel. The existing app-state check incorrectly called True Skate
  foreground; raw frame zero disproved any claim that a sampled swipe opened the
  panel. WDA's frontmost-bundle endpoint identified SpringBoard, so connection
  and per-gesture guards now use it to reject this contamination route.
  [Experiment record](experiments/M1-20260921-direct-video-phase.md).

## 2026-09-20 — Replacement corpus park provenance corrected

- The operator identified the XR2 park as Skateboard GB 2024. Corrected the
  erroneous `The Workshop` label in all 910 admitted clip records and their
  segment manifests, and renamed 86 park directories to `skateboard_gb_2024`.
  The paused collector was not relaunched.

## 2026-09-18 — Replacement linear corpus started on XR2

- Linear sampling now keeps both the start and end point outside every expanded
  control hitbox. Intermediate crossings remain provisionally allowed while the
  path/speed activation rule is deferred for later research.
- Two isolated one-minute runs passed strict two-anchor admission. A concurrent
  service launch exposed and fixed a stale screen-recorder binding after Appium
  session recovery (`b60f19d`). The service supervisor was then unloaded because
  it conflicts with the separately launched prebuilt WDA stack.
- XR2 collection started in Skateboard GB 2024 at `b60f19d`, in a new
  `basic_linear_v2_20260918` corpus, with idle-navigation allowance but all
  replay/editor and foreground guards retained. The first production segment
  admitted 10 clips; a strict watcher stops at 1,100. XR1 was unavailable.

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
- Follow-up decision (2026-09-18): control activation may depend on path, speed
  and/or drag-recognition distance. That mechanism is deferred but remains
  important. For the new linear corpus, both gesture start and end points must
  clear the expanded hitboxes; intermediate crossings remain provisionally
  allowed and require later research.

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
