# Recent research and engineering journal

One entry per result, newest first: **Ran**, **Result**, **Decision**, at most
six lines. Keep 30 entries and drop the oldest freely; main's history keeps
them. Link a record only when one exists. Current state lives in
[SYSTEM_MAP.md](SYSTEM_MAP.md).


## 2026-10-16 — The kickflip lands with 2% position error, not 5%

- **Ran:** your approved kickflip (19/19) on XR1 in Workshop with its pop and flick moved a fixed distance in a random
  direction: 0, 2, 5 and 10% of the screen, 10 attempts each (KICKFLIP-TOLERANCE-20261015); your blinded review.
- **Result:** landed 10/10 at 0, 8/10 at 2%, 1/9 at 5% (one technical failure), 0/10 at 10%. Both misses at 2% were
  caught a quarter turn late; at 5% half the misses were near misses; at 10% the pop or flick failed (3 pop shove-its).
- **Decision:** labels, Model 1 and the hand need positions within ~2% of the screen; MVP v0 (37% of test gestures at
  2%) misses it, so the pops' end (item 10) comes before the 1,000 clips ([record](experiments/KICKFLIP-TOLERANCE-20261015.md)).

## 2026-10-16 — Live frames: WDA's MJPEG stream tops out at ~28 frames a second; Model 2 takes 30 fps at 207×448

- **Ran:** XR1 Workshop, no gameplay touches: the stream read through 60 fps recorded minutes with die-five markers
  (LIVE-FRAMES-20261016): 10 fps full size, then 30 and 60 fps at 50% and 25% size (six minutes; one never started).
- **Result:** 9.6 fps at the 10 fps setting; 27.4–28.8 at every 30 or 60 fps setting and both sizes, so ~28 is WDA's
  capture ceiling. None stale (0 of 374 spans where the screen moved); the recording held 59.9 fps. A tap shows in the
  stream about as soon as in the recording (0.11–0.16 s); run2's delay creep didn't repeat (−52 to +24 ms/min).
- **Decision:** Model 2 trains on 30 fps frames at 207×448; live play asks for 30 fps at 25%, taking the newest frame.

## 2026-10-16 — Trick Model 1 MVP v0 finds and times fast gestures; placing them is the weak part

- **Ran:** curved v1's network with per-frame touching, touch-down and lift signals and a gesture decoder; trained
  on the 182 trial clips of minutes 1–5 at 64×144 and 128×288 (Modal, US$1.28 in all); minute 6 chose 128×288.
- **Result:** test (minute 7, 78 gestures, scored once): 99% on time at both ends (touch-down −8 to +7 ms, 10–90%);
  worst path point within 5% of the screen for 90%, within 2% for 37%. Predicted ends fall short, pops most.
- **Decision:** clears the "half the gestures" bar at 5%, not at 2%; item 1's tolerance picks the next step
  ([record](experiments/MODEL1-TRICK-MVP-20261016.md)).

## 2026-10-15 — Trick-gesture labels re-timed to the visible trail

- **Ran:** XR2 and XR1 trick-speed trials (TRICK-SPEED-20261015: 125 clips each, 277 direct gestures each, all
  admitted); operator spot check of 20; trail-vs-label timing every 8 ms along each path; 205 pilot gestures.
- **Result:** rig-clock labels sat −23 to +25 ms off the trail (10–90%), mostly per recording; no clear speed
  error once the trail's glow is allowed for. Pilot (phone-timestamped) labels: −16 to +4 ms. Spot check: 9/20 OK.
- **Decision:** admission re-times each gesture to its own trail (misreads take the recording's median). Re-check:
  12/20 OK; the rest are speed-ups seen after bends and pops whose trail stops 10–20 pt short (24% of pops).

## 2026-10-14 — Model 1's 32-frame sampling is too sparse for trick gestures

- **Ran:** filmed kickflip (60 fps) through linear Model 1's frame picks
  (~73 ms apart) at five phases; about an hour, laptop only.
- **Result:** each gesture's motion lands in ~1 picked frame and its trail in
  1–2 (rule: ≥2); at 60 Hz, 3 (pop) and 5 (flick). The flick pauses ~70 ms first.
- **Decision:** trick models read every 60 Hz frame, as curved v1 does.
  [Contact sheet](evidence/SAMPLING-CHECK-20261014/contact-sheet.jpg).

## 2026-10-13 — Free Colab reproduces Model 1 training but not Modal's turnaround

- **What ran:** one Model 1 seed on a free Colab T4, with the reference recipe
  and pins. The frame cache was off, because free Colab's 12.7 GB of RAM cannot
  hold it.
- **Matches Modal:** the Modal checkpoint scored 91.25% on Colab, 0.0 pp from
  its Modal score. Validation over epochs 1–19 averaged −0.08 pp against Modal
  seed 0. Resuming at epoch 6 on a new VM restored the state exactly.
- **Slower:** uncached epochs took 943 s, against Modal's 270 s. Colab ended the
  second session after 4.15 h. With two attempts per UTC day, a 3-seed batch
  would take about 35 GPU-h over five or more days. The operator stopped the run
  at 19/40 epochs (5.94 GPU-h).
- **Effort:** about zero active operator minutes after one-time setup; the agent
  drove the browser.
- **Cost comparison:** Modal's L4 list price for three seeds is $12.46, and 39%
  of that is the 64 GiB memory request. Starter's $30/month credit may cover
  about two runs a month. Colab's paid L4 could fit the cache for $2–5 of units,
  but seeds run serially and need a live session. For occasional training,
  Modal has more pros. The Modal plan and credit balance are still unknown.
  [M1-COLAB-20261006](experiments/M1-COLAB-20261006.md).

## 2026-10-03 — Repeated finding: bundled gestures are conjoined

- **Do not bundle sequential gestures into one request.** A replay of an expert
  demo sent all eight gestures (pushes, two trick flicks, catch) as one W3C
  request with one pointer each. In 9/9 replays, at native, 33 ms and 50 ms point
  spacing, True Skate drew one continuous chain, with each gesture joined to the
  next.
- This was already solved in May 2026, when the `perform_trick_gestures` WDA
  endpoint work found that multi-path records and hover moves both leak touch
  events. It resurfaced as "joined gestures" in September's run04b. The knowledge
  had survived only in archived files; it is now a hard rule in GESTURES.md.
- The one-request-per-gesture replay works, but each request returns about 0.3 s
  after its gesture ends. Flick B started 315–343 ms after flick A instead of
  168 ms. Tighter sequences need on-device scheduling.
  [CURVE-AUDIT-20261003](experiments/CURVE-AUDIT-20261003.md).

## 2026-10-03 — Blinded linear gesture duration × length audit

- Reviewed all 135 XR1/Inbound clips: nine durations from 10 to 50 ms in
  **5 ms increments**, five lengths and three repetitions per combination.
- **Board response:** movement in 120/120 clips at ≥15 ms, versus 0/15 at
  10 ms. The observed transition lies between 10 and 15 ms in this session.
- **Visible trace:** 75/75 at 30–50 ms; 12/15 at 25 ms; 7/15 at both 20 and
  15 ms; 0/15 at 10 ms. Remaining labels were Flicker.
- **Trace visibility is not a reliable proxy for board response:** 19
  Flicker-labelled gestures still moved the board. These observations do not
  establish input-path collapse.
- Length effects were inconsistent across durations. Three repeats per
  combination do not establish a universal threshold. Both comments about
  unusually short traces corresponded to the shortest length at 45 ms.
- Results are explicitly saved human assessments; the viewer preselected Trace
  and Board moved. Full labels, mappings and results are preserved in
  [LINEAR-LENGTH-20261003](experiments/LINEAR-LENGTH-20261003.md).

## 2026-10-02 — Cubic gesture execution gate implemented

- **CURVE-EXEC:** additive nine-number Bernstein cubic, exact-ms compiler and
  matching command labels; legacy defaults and checkpoint formats unchanged.
- Offline twelve-curve bounds admit 10/20 ms spacing; 40 ms fails two curves.
  N=2/4 remain offline. Pilot/main/confirmation ceiling is 204 diagnostics,
  with automatic orange-trail measurement and blinded audits.
- Inbound pilot stopped after eight XR1 diagnostics: native FFmpeg reanalysis
  passes calibration (middle error 5.8 ms), but orange extraction is 0/8
  evaluable. Fidelity/floor inconclusive; main and confirmation withheld.
  [Protocol and evidence](https://github.com/AsherNoble/TrueSkate-AI/blob/3f6709ebeeee42d3acbae0a732b87eec1580382e/research/experiments/CURVE-EXEC-20261002.md).
- Operator viewed clips and requested a wide fast arc: three XR1 commands at
  300/200/120 ms recorded for human viewing. Gameplay admission failed; retained
  without replacement. [Visual speed probe](experiments/CURVE-SPEED-20261002.md).
- Requested jagged organic path: five joined cubics in one touch, recorded at
  600/300 ms. Gameplay check flagged the preserved recording; human viewer ready.
  [Jagged probe](experiments/CURVE-JAGGED-20261002.md).
- Operator review: 600 ms looked good; 300 ms appeared as a tap yet had a strong
  gameplay effect. Rendering/delivery cause unresolved; spacing also changed.
- Frozen duration-only linear sweep (600–10 ms, two reversed repeats) aborted
  on the first sleep overrun before any control/drag. Short video and empty WDA
  report retained; no replacement. [Record](experiments/LINEAR-SPEED-20261002.md).
- Operator overrode replacements: complete reversed pair now records 16 drags;
  native-frame Safari viewer works. Timing middle residuals 14/9 ms; original
  gameplay admissions remain failed (first flags show scenery). Assistant review:
  trail at ≥50 ms, 20 ms indeterminate, 10 ms tap-like; rendering, not input fidelity.

- Operator assessment: ≥50 ms trace/board movement; 20 ms no trace/board movement;
  10 ms tiny flicker/no board movement. Requested finer reversed 50–20 ms sweep.
  All 14 completed; timing 11.8/6.4 ms. Operator requested human gameplay review.
  [Fine sweep](experiments/LINEAR-SPEED-FINE-20261002.md).

- Follow-up (2026-10-03): fine trace visible in 13/14; board moves in all 14.
  The separate 135-clip length experiment and its complete human findings are
  recorded once in the 2026-10-03 entry above.
- MVP 2.0 direction: arbitrary timed single-drag paths, any initial board state,
  five-touch-aligned moving-circle review and frame-position accuracy.
  Provisional 15-point cap, hollow 10 pt ring, linear-equivalent tolerances;
  practical check is expert single-drag trick inference/replay in Workshop.
  [Requirements](protocols/model1_curved_mvp2.md).

## 2026-10-01 — Modal storage audit; deleted the June–July corpus

- The account held ≈ 1,293 GiB of Volumes. Modal charges $0.09/GiB-month
  above a free 1 TiB, so this cost ≈ $24/month. `trueskate-corpus` alone
  was 929 GiB (52 June–July sessions, ≈ 95k gestures, collected before tap
  calibration, so timing is unusable).
- **Deleted `trueskate-corpus`** (operator-approved). ≈ 364 GiB remain, so
  storage is now free. The loss is permanent: no other copy existed.
  Recollect with the calibrated pipeline instead.
  [MODAL_STORAGE_20261001](history/MODAL_STORAGE_20261001.md).

## 2026-10-01 — Doubling the data still helps, but less

- **M1-EXPAND:** 10,131 new screened clips (Kansas City and Los Angeles
  weighted; three new parks), audit 3 passed (99/100 by the frozen scorer,
  100/100 with an operator-attested entry correction). Training doubled to
  18,394; validation and test unchanged.
- **Result** (64×144, cosine, three seeds): last-10 validation **91.37%**
  against 89.17%, a 20.3% error reduction. That clears the fixed 20%
  data-limited line by 0.04 pp; the previous doubling gave 39.7%.
  Los Angeles failures fell 21.4% → 15.4% and Kansas City 16.3% → 12.8%.
  ≈ $12.8. [M1-EXPAND](experiments/M1-EXPAND-20260929.md).
- **New-park holdout** (907 clips selected in whole recording sessions in Portland, Super
  Crown 2016 and Inbound; best checkpoint, scored once): **87.32%**, against
  92.16% validation in the original parks. Mostly end-point misses. The park
  and stricter-split effects are confounded.
- **Free analysis of those results:**
  - Each original park's gain matches its own added data. The new-park
    clips gave no visible lift; Skateboard GB is a weak exception.
  - No detected dose-response with the number of training clips sharing a recording; a step between any sharing and none remains untested.
  - Holdout failures are spread evenly across recordings.
  - The gap is compatible with insufficient per-park data; a shared-recording
    step effect remains unresolved.

## 2026-09-30 — End misses: temporal averaging is part of it

- **Frozen-checkpoint probe** (three cosine seeds, validation only): narrowing
  the end time window from σ 0.15 to 0.05 recovers **+1.09 pp** (89.40 →
  90.48%), with 68 clips gained against 4 lost across seeds. Using the true
  liftoff changes nothing, since the duration estimate is already accurate.
  End-short failures fall only 15%, so most of the shortfall remains
  unexplained. The single-frame variant was an invalid test.
  [M1-ENDPROBE](experiments/M1-ENDPROBE-20260930.md).

## 2026-09-29 — Quarter resolution is the floor

- **32×72** (seed 0, cosine): last-10 validation mean **82.00%**, against
  89.66% at 64×144 (−7.66 pp). By the fixed rule, 64×144 stays the
  development default. ≈ $0.8.
  [M1-QUARTERRES](experiments/M1-QUARTERRES-20260929.md).

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

## 2026-09-27 — Screened corpus passes blind random audit

- `corpus-screen-v1` is frozen as a whole-segment rule: rate within 0.0008 and
  controls within 85–210 ms of their commands. The latency gate was supported
  on the held-out corpus first (90/90 high-rate segments had an early start).
  The screen keeps 12,113/13,592 clips.
- A blind uniform 100-clip audit passed 100/100 within one displayed frame
  (97 exact), bounding bad clips below 3%.
- Plan: top up ~1,480 good clips and audit a fresh 100.
  [Record](experiments/M1-CORPUS-AUDIT-20260927.md).

## 2026-09-27 — 13,100 good-label manifest passes final audit

- A settle-wait top-up on release `080bb97` collected 722 clips (Los Angeles
  331). All passed `corpus-screen-v1`, against 47% for Los Angeles before.
- `park-mix` built `model1_linear_good13100_20260927` with the 80.05% corpus's
  exact park counts.
- A fresh blind 100-clip audit passed 100/100, all exact.
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

## 2026-09-25 — Two-finger XR2 visibility probe

- The operator found no pair in the first viewer. Selenium `ActionChains` had
  silently retained only the last pointer source, invalidating those recordings
  as a two-finger test. A corrected two-source payload completed in three new
  XR2 originals, and the operator saw two simultaneous marks in each. No
  calibration method was adopted.
  [Record](experiments/M1-MULTITAP-20260925.md).

## 2026-09-25 — Post-reset settle wait (smoke)

- Los Angeles boards move for ~7–8 s after the pre-segment reset, and moving
  starts coincided with weak start detections. An opt-in settle wait produced
  still start controls in 17/17 bounded recordings (9/30 moving without).
  Red-team: its effect on anchor errors is untested. A confirmatory interleaved
  test is proposed, not run. [Record](experiments/M1-SETTLE-20260925.md).

## 2026-09-25 — Multi-anchor consensus calibration (smoke)

- An opt-in consensus fit over ≥3 single centre controls rejects anchors that
  disagree. Offline over 44 LA segments it matched the label-free proxy.
  Production-path smoke: 4/4 segments accepted, and one 17-frame mid outlier
  rejected (not human-checked). Red-team: self-consistency only, accuracy
  unproven; screen and fit gates tightened.
  [Record](experiments/M1-MULTIANCHOR-20260925.md).

## 2026-09-25 — Four simultaneous XR2 touches

- Removed the on-film single-touch control; sent four fingers at the corners
  of a 66-logical-point square in three bounded recordings. All commands
  completed without leaving gameplay; four spots are visible in a preliminary
  frame review. Exact timing awaits operator inspection in the new viewer.
  [Record](experiments/M1-MULTITAP4-20260925.md).

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

## 2026-09-25 — Draft corpus timing screen

- A per-clip anchor-error screen (opt-in in the cohort builder) keeps 12,628 or
  12,517 of 13,592 replacement clips at rate thresholds 0.002 or 0.0008.
  Red-team: 0.002 hides up to ~3-frame errors in 74 kept LA clips, so it is not
  recommended. Los Angeles falls below its 1,261 target, so ~320 raw LA clips
  must be recollected. A sealed LA-only blind mid-band check is prepared, not
  labelled. [Record](experiments/M1-SCREEN-DRAFT-20260925.md).

## 2026-09-25 — 91-touch XR2 grid probe

- Three isolated recordings captured a 7 × 13, 50-point grid command. WDA
  returned success and gameplay remained visible, but a frame check found
  only five requested positions brightening in each recording. The full grid
  was not delivered visibly; command duration varied from 7.6 to 30.5 s.
  [Record](experiments/M1-MULTITAP91-20260925.md).

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

## 2026-09-24 — Los Angeles originals retained on both XRs

- A bounded four-attempt run per XR preserved seven original one-minute videos
  with WDA reports and strict alignment. XR2 lost one attempt during save. Six
  calibration fits were near 1.0; XR2 segment 3 was 1.001788 with a weak start
  detection (score 10.82). The severe >1.004 case did not recur. All collectors
  stopped at their bounds. [Record](experiments/M1-LA-ORIGINALS-20260924.md).
