# Recent research and engineering journal

Keep at most 30 dated entries. Before trimming, tag the complete version and
append its commit-pinned recovery entry to [ARCHIVE.md](ARCHIVE.md).
Put substantial experiments in individual records and current facts in STATUS.

## 2026-09-29 — Quarter resolution is the floor

- **32×72** (seed 0, cosine): last-10 validation mean **82.00%**, against
  89.66% at 64×144 (−7.66 pp). By the fixed rule, 64×144 stays the
  development default. ≈ $0.8.
  [M1-QUARTERRES](experiments/M1-QUARTERRES-20260929.md).

## 2026-09-28 — Cosine lr adopted; model still data-limited

- **Cosine decay at 64×144:** the three-seed last-10 validation mean went
  from 87.11% to **89.17%**, and the epoch-to-epoch SD fell 5–12×. Adopted.
  [M1-LRDECAY](experiments/M1-LRDECAY-20260928.md).
- **Nested subsets** (2,293 / 4,585 / 9,170; cosine; 64×144): 77.73% →
  82.05% → 89.17%. Validation error fell 19.4% for the first doubling, then
  39.7% for the second. By the fixed rule, **collect a doubling.** Kansas City
  failures fell 32.9% → 22.8% → 16.3%.
  [M1-SCALE-SUBSETS](experiments/M1-SCALE-SUBSETS-20260928.md).
- The spend guard now projects from the median epoch time; one stalled epoch
  had falsely stopped three runs.

## 2026-09-28 — Half resolution is not worse

- Identical recipe at 64×144 instead of 128×288, seed 0: validation last-10
  mean 86.99% (against 85.32%), best 88.70% (against 87.63%); paired
  McNemar p = 0.13. Kansas City and Los Angeles misses are unchanged. At
  ~4× less compute, resolution is not the bottleneck in this range.
  Seeds 1–2 confirmed it: the three-seed last-10 mean is 87.11% at 64, against
  84.68% at 128; every seed is no worse. 64×144 is adopted for development.
  The best-validation checkpoint (64 seed 2, 90.94%) scored **91.04% on test**
  in the partition's second exposure: McNemar p = 0.0003 against the 88.50%
  model, and +11.0 pp (+8.8 to +13.2) against 80.05%.
  [M1-HALFRES](experiments/M1-HALFRES-20260928.md).

## 2026-09-28 — Clean-label retrain: 88.50% test recovery

- Same recipe, size, park mix and split protocol as the 80.05% model, on
  `model1_linear_good13100_20260927`. All three seeds beat every original
  seed on validation. Selected on validation only: seed 0, epoch 30 (87.63%).
  A single test exposure scored **88.50%** (1,739/1,965): +8.45 pp, 95% CI
  +6.2 to +10.7. [M1-RETRAIN](experiments/M1-RETRAIN-GOOD13100-PLAN-20260927.md).
- **Validation autopsy:** 74% of failures involve the end point, almost all
  falling short along the path. They concentrate in Kansas City and Los
  Angeles (21–24%, against The Workshop's 2.4% on similar data volume) and on
  fast gestures. An equal-weight seed ensemble lowered recovery (86.21%) and
  was not adopted. [M1-DIAG](experiments/M1-DIAG-20260928.md).
- **Planned, not approved:** learning-rate decay (~$25) and nested subset
  scaling (~$24). [M1-SCALE-SUBSETS](experiments/M1-SCALE-SUBSETS-20260928.md).

## 2026-09-27 — 13,100 good-label manifest passes final audit

- A settle-wait top-up on release `080bb97` collected 722 clips (Los Angeles
  331). All passed `corpus-screen-v1`, against 47% for Los Angeles before.
- `park-mix` built `model1_linear_good13100_20260927` with the 80.05% corpus's
  exact park counts.
- A fresh blind 100-clip audit passed 100/100, all exact.
  [Record](experiments/M1-CORPUS-AUDIT-20260927.md).

## 2026-09-27 — Screened corpus passes blind random audit

- `corpus-screen-v1` is frozen as a whole-segment rule: rate within 0.0008 and
  controls within 85–210 ms of their commands. The latency gate was supported
  on the held-out corpus first (90/90 high-rate segments had an early start).
  The screen keeps 12,113/13,592 clips.
- A blind uniform 100-clip audit passed 100/100 within one displayed frame
  (97 exact), bounding bad clips below 3%.
- Plan: top up ~1,480 good clips and audit a fresh 100.
  [Record](experiments/M1-CORPUS-AUDIT-20260927.md).

## 2026-09-26 — Five-touch comparisons scored; timing gate proposed

- Blind labels scored for the 49.5 pt and 100 pt five-touch runs.
  - 49.5 pt: A 1 vs B 0 gross Los Angeles start errors, p = 1.0.
  - 100 pt, a heavy post-reset-motion session: A 12 vs B 3, 9 discordant for
    B, p = 0.004. B's 3 errors fired before the command was sent, and it
    rejected 45% of XR1 starts.
  - Wider spacing cut post-swipe rejections from 12/60 to 4/60.
- Multi-anchor fits cut leave-one-out gross errors from 36 to 9 and the worst
  error from 26 to 2 frames.
- All 120 labelled controls appeared 118–174 ms after WDA submission, and all
  gross errors earlier. A latency gate is preregistered for a held-out corpus
  test. [Record](experiments/M1-DIE5-R100-20260925.md).

## 2026-09-25 — Multi-anchor consensus calibration (smoke)

- An opt-in consensus fit over ≥3 single centre controls rejects anchors that
  disagree. Offline over 44 LA segments it matched the label-free proxy.
  Production-path smoke: 4/4 segments accepted, and one 17-frame mid outlier
  rejected (not human-checked). Red-team: self-consistency only, accuracy
  unproven; screen and fit gates tightened.
  [Record](experiments/M1-MULTIANCHOR-20260925.md).

## 2026-09-25 — Draft corpus timing screen

- A per-clip anchor-error screen (opt-in in the cohort builder) keeps 12,628 or
  12,517 of 13,592 replacement clips at rate thresholds 0.002 or 0.0008.
  Red-team: 0.002 hides up to ~3-frame errors in 74 kept LA clips, so it is not
  recommended. Los Angeles falls below its 1,261 target, so ~320 raw LA clips
  must be recollected. A sealed LA-only blind mid-band check is prepared, not
  labelled. [Record](experiments/M1-SCREEN-DRAFT-20260925.md).

## 2026-09-25 — Post-reset settle wait (smoke)

- Los Angeles boards move for ~7–8 s after the pre-segment reset, and moving
  starts coincided with weak start detections. An opt-in settle wait produced
  still start controls in 17/17 bounded recordings (9/30 moving without).
  Red-team: its effect on anchor errors is untested. A confirmatory interleaved
  test is proposed, not run. [Record](experiments/M1-SETTLE-20260925.md).

## 2026-09-25 — Five-touch vs single-touch calibration (labels pending)

- Preregistered and red-teamed the comparison. 30 Los Angeles recordings were
  captured (20 XR1, 10 XR2). XR1 had no start anomaly, so the gate pauses its
  extension. One XR2 start anomaly (+6,056 ppm): B declined to anchor. A had 7
  false alarms in 157.5 s of no-touch footage; B had none. B rejected 12/60 mid
  markers (post-hoc cause: early upper-corner triggers). The blind 132-item
  viewer is ready. [Record](experiments/M1-DIE5-COMPARE-20260925.md).

## 2026-09-25 — Five-touch die calibration pilot

- Twelve markers across three bounded XR2 recordings lit all five requested
  positions on the same first-visible frame. The single-centre detector also
  succeeded in this stationary scene, so no improvement or calibration change
  is claimed. [Record](experiments/M1-DIE5-20260925.md).

## 2026-09-25 — Eight-touch XR2 limit probe

- Unity documents five concurrent iPhone touches. Three XR2 recordings sent
  eight distinct touch sources in different orders; exactly five requested
  positions brightened each time. The sender and deployed WDA event builder
  retain all eight paths; the limiting downstream layer remains unknown.
  [Record](experiments/M1-MULTITAP8-20260925.md).

## 2026-09-25 — 91-touch XR2 grid probe

- Three isolated recordings captured a 7 × 13, 50-point grid command. WDA
  returned success and gameplay remained visible, but a frame check found
  only five requested positions brightening in each recording. The full grid
  was not delivered visibly; command duration varied from 7.6 to 30.5 s.
  [Record](experiments/M1-MULTITAP91-20260925.md).

## 2026-09-25 — Four simultaneous XR2 touches

- Removed the on-film single-touch control; sent four fingers at the corners
  of a 66-logical-point square in three bounded recordings. All commands
  completed without leaving gameplay; four spots are visible in a preliminary
  frame review. Exact timing awaits operator inspection in the new viewer.
  [Record](experiments/M1-MULTITAP4-20260925.md).

## 2026-09-25 — Two-finger XR2 visibility probe

- The operator found no pair in the first viewer. Selenium `ActionChains` had
  silently retained only the last pointer source, invalidating those recordings
  as a two-finger test. A corrected two-source payload completed in three new
  XR2 originals, and the operator saw two simultaneous marks in each. No
  calibration method was adopted.
  [Record](experiments/M1-MULTITAP-20260925.md).

## 2026-09-24 — Los Angeles originals retained on both XRs

- A bounded four-attempt run per XR preserved seven original one-minute videos
  with WDA reports and strict alignment. XR2 lost one attempt during save. Six
  calibration fits were near 1.0; XR2 segment 3 was 1.001788 with a weak start
  detection (score 10.82). The severe >1.004 case did not recur. All collectors
  stopped at their bounds. [Record](experiments/M1-LA-ORIGINALS-20260924.md).

## 2026-09-24 — Balanced audit exposed calibration outliers

- Both finite Super Crown runs stopped at their bounds, bringing the five-park
  replacement pool to 13,902 strict clips. The operator reviewed 28 random
  clips per park: 124 good, 9 mild, 7 critical. Six critical Los Angeles clips
  have anomalously high two-anchor calibration rates that predict their late
  visible traces; the seventh Super Crown clip has a low-rate anomaly. A
  corpus-wide rate screen flags 1,063 clips for investigation, not automatic
  exclusion. Human marks and selection are preserved in
  [M1-AUDIT-20260924](experiments/M1-AUDIT-20260924.md). Training selection is
  paused pending independent timing validation.
- A blinded, held-out 24-clip onset check is prepared from twelve previously
  unseen recordings, pairing early and late gestures in high-rate Los Angeles,
  ordinary-rate Los Angeles and low-rate other-park segments. The selection is
  frozen before human labels; no collection was started.
  [Protocol](experiments/M1-ONSET-VALIDATION-20260924.md).
- The operator labelled all 24 onsets without uncertainty. High-rate Los
  Angeles pairs had late early-gesture traces that returned to frames 8–9 near
  the end; low-rate other-park pairs showed the reverse; ordinary-rate Los
  Angeles pairs stayed at frame 8. An anchor-error timing model predicted
  22/24 exact onset frames and the other two within one frame. This validates
  the extreme-rate risk signal, not a cutoff for automatic exclusion.
  [Labels and analysis](experiments/M1-ONSET-VALIDATION-20260924.md).

## 2026-09-24 — Replacement park top-up completed

- The second bounded stage stopped at 4,052 XR1/Workshop and 3,809 XR2/Kansas
  City strict clips. One operator-confirmed XR2/Skateboard GB segment added 12
  audited clips, bringing that park to 2,020 including its 310-clip baseline.
  Both recorders returned idle. The operator clarified that the
  replacement should match aggregate park totals, not device-by-park cells;
  Super Crown is the remaining park. After the operator loaded both XRs there,
  two finite collectors started with separate targets of 1,150 and 860. Their
  first segments passed strict audits with 10 and 12 accepted clips.
  [Counts and audit](experiments/M1-RECOLLECT-20260922.md).

## 2026-09-23 — Replacement collection rotated into two new parks

- The first bounded tranche finished with 2,003 XR1/Los Angeles clips and 2,008
  XR2/Skateboard GB clips including its 310-clip baseline. The operator moved
  XR1 to The Workshop and XR2 to SLS 2013 Kansas City and authorized new bounds
  of 4,050 and 3,800 clips. Their persisted seeds continue across the park
  boundary. Initial audits admitted 10 and 11 clips with correct exact-PTS
  metadata and found no duplicate commands across the replacement directory.
  [Plan and evidence](experiments/M1-RECOLLECT-20260922.md).
- The opt-in one-decode exact-PTS extractor matched all 32 decoded pixels and
  source timestamps in ten retained phone clips. Isolated XR1/XR2 segments then
  admitted 10/10 and 9/9 strict clips with accepted calibration, and aligned
  in about 169 seconds versus old-release medians of about 304 seconds. Clean
  release `1b8497e` was promoted between production segments; both bounded
  collectors resumed with prior seeds and targets, and release `13554c9` was
  retained for rollback. Their first production segments admitted 11 XR1 and
  10 XR2 strict clips without batch fallback. [Experiment](experiments/M1-BATCH-EXTRACT-20260923.md).

## 2026-09-22 — Exact-PTS tranche passed audit and preserves label controls

- The operator marked 17/17 inspected exact-PTS clips `good`; the review export
  is preserved with the timing evidence.
- Metadata comparison against all 13,100 historical samples found that both
  corpora use 32-frame clips, the same nominal gesture-onset slot, and varied
  0.30–1.20 second command durations. The new clips retain exact source PTS;
  their small timing variation reflects real frame phase rather than a changed
  label policy. [Comparison](experiments/M1-LABEL-CONTROL-20260922.md).
- Basic-linear sampling now rejects the entire path if it touches an expanded
  control rectangle. A 100,000-command offline stress run had no retry
  exhaustion, and one bounded XR2 segment admitted 11/11 strict, 32-frame,
  exact-PTS clips with zero path intersections or detected UI transitions.
  [Control experiment](experiments/M1-CONTROL-20260917.md).
- A surviving 304-second June XCTest attachment exposed a false-success path in
  Appium's cleanup command. After preserving it, removing its directory and
  restarting XR2 testmanagerd, two consecutive short recordings completed and
  self-cleaned. The wrapper now verifies deletion by relisting.
  [Recovery record](experiments/XCTEST-ATTACHMENT-20260922.md).
- XR1 recovered after the operator refreshed Xcode signing and trusted the new
  runner. Its exact WDA timing build, Appium endpoint and RemoteXPC tunnel passed;
  two consecutive recordings self-cleaned and both XRs ended with zero
  attachments. The WDA fork and rig checkout now use `master` at `b5ace217`.
  Collection remains off.
- The 13,100-sample replacement plan preserves the historical device/park mix.
  Its 310 accepted XR2 Skateboard GB 2024 clips leave 12,790 new clips; XR1's
  loaded park is confirmed as SLS 2015 Los Angeles.
  [Plan](experiments/M1-RECOLLECT-20260922.md).
- The operator authorized a bounded first tranche to 2,000 strict clips per
  device. A simultaneous exact-PTS smoke admitted 11/11 clips on each XR with
  32 source-timed frames per clip. Release `13554c9` now collects XR1 to 2,000
  new clips and XR2 to 1,690 new clips plus its 310-clip baseline. Persisted NTFY
  monitors announce each new ten-percent milestone. XR1's first 598 records were
  corrected from the explicit park placeholder without changing strict count or
  command fingerprint; collection resumed while XR2 continued uninterrupted.

## 2026-09-21 — Compact-video onset phase corrected

- The operator marked the first visible trace as frame 7 in 35/35 audited clips,
  while synthetic timing placed onset at frame 8. A surviving raw tranche video
  showed both centre-control detections were already on their first visible
  frames and the fitted WDA time missed a swipe onset by only 8.3 ms.
- FFmpeg's resampling paired selected pixels with an invented output-time grid.
  Upward rounding fixed the preserved example but did not give a reliable timing
  contract on a new clean segment. Compact extraction now selects exact source
  frame numbers and stores their actual source PTS relative to gesture onset.
  The operator chose recollection over repair of the 1,109-clip tranche.
- Two bounded validation attempts started underneath an already-open iOS Control
  Center panel. The existing app-state check incorrectly called True Skate
  foreground; raw frame zero disproved any claim that a sampled swipe opened the
  panel. WDA's frontmost-bundle endpoint identified SpringBoard, so connection
  and per-gesture guards now use it to reject this contamination route.
- After Control Center was closed, a third bounded segment passed two-anchor
  calibration, strict admission (11/11 payloads), 32-frame decoding and the new
  frontmost-app guard. A final bounded segment using the exact-PTS extractor
  also passed two-anchor calibration (rate `0.99994441`), admitted 11/11
  payloads, decoded every clip at 32 frames with 32 stored source-relative
  timestamps, and left True Skate frontmost. The extractor is now validated on
  device for this collection path.
  An inventory then confirmed that the audited 1,109-clip tranche has only one
  surviving original `.mov`, with no emitted clips from that interrupted
  recording. It cannot be faithfully regenerated; the next corpus will be
  collected with the exact extractor from the outset.
  A direct XR2 guard test then opened Control Center after connection. WDA
  reported SpringBoard while Appium still reported True Skate foreground;
  `ensure_foreground()` reactivated True Skate and WDA confirmed its bundle
  afterward. The recovery mechanism is now directly device-validated.
  A follow-up Skateboard GB 2024 audit tranche admitted 310 strict exact-PTS
  clips. Every clip decodes to 32 frames with 32 increasing stored timestamps.
  The live guard also caught SpringBoard during collection, restored True Skate
  and withheld the affected attempt; the segment then failed calibration and
  retained its raw recording without contributing clips.
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
