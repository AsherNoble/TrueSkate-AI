# Vision-Grounded Gesture-Sequence Leap — Journal

Tracking the leap from open-loop single-trick CMA-ES to closed-loop, perception-grounded gesture **sequences** (lines, park). Plan: a two-model spine — **Model 1** (learned trace extractor: frame → touch coords, self-labeled by the agent's known touches) feeding **Model 2** (n frames + m gestures → next m gestures, trained on Model-1-labeled expert play). See `/Users/ashernoble/.claude/plans/tender-foraging-creek.md` Part B.

## Data Reality — Verified On Disk (2026-06-14)

Investigated the three candidate data sources. The premise of Asher's idea (reliably extract touch coords from True Skate's orange finger-trace) is **sound but needs the right data + ground truth**:

1. **"True Skate w: my Hand Clips/" (11 clips, ~14 min, 1920×1080) = EXTERNAL CAMERA footage**, filmed over Asher's shoulder — the phone screen is a small, tilted, glare-washed region in the frame. trace_extractor gets **0%** here (no clean full-frame screen to read). **Unusable for trace extraction.** Don't build on these.
2. **`data/extracted_frames/` (6 clips, 727 frames, portrait ~750×1624) = proper SCREEN recordings** — the original BC dataset, the format trace_extractor was built for. True Skate is **portrait** (the landscape hand-clips were just the externally-filmed phone; resolves the orientation confusion — matches the RL device coords 414×896 portrait).
3. **RL self-play frames** are grayscale 210×455 (FrameRecorder does `.convert("L")`) AND discarded (`record_frames=False`) AND captured post-gesture — so no usable color trace corpus exists yet.

## The Orange Finger-Trace Is Real, Distinctive, and Transient

- Confirmed visually: a 360-flip frame (`360_single_flip_001/img_00010.jpg`) shows a bright **orange swoosh** trailing the flick — the finger-trace True Skate renders. It's the signal Model 1 reads.
- **Transient:** present only during/just after an active drag; mid-air/settled frames have no trace (already faded). Labeling must align frames to when the touch is active.

## trace_extractor Reliability — Quantified

Ran the CURRENT `TraceExtractor.process_frame` over each clean clip sequentially (`.reset()` per clip):

| clip | frames | extractor active | median warm-px/frame |
|---|---|---|---|
| 360_single_flip_001 | 95 | **100%** | 9,465 |
| double_kickflip_003 | 91 | **47%** | 4,079 |
| laser_flip_001 | 135 | **95%** | 72,543 |
| gazelle_double_heel_001 | 234 | **100%** | 12,907 |

- The committed `double_kickflip_003_60fps_labels.csv` shows **0%** active — it is **STALE** (old extractor/config); current code gets 47% on the same clip. (So "BC failed on bad labels" is partly a stale-pipeline artifact.)
- **Caveat — the active flag is not the same as positional accuracy.** Warm-pixel floors are huge (laser_flip median 72k px/frame): the warm *board grip + ground reflections* dominate, so the extractor can fire on the board rather than the finger trace. Whether the detected (x,y) is the true touch is **unverified by anything except ground truth.**

## Conclusion → Why Asher's Agent-as-Labeler Is the Right Move

- The trace exists and the extractor *fires*, but its **positional accuracy is unknown** and the hand-tuned HSV+temporal heuristic clearly mislabels in places. There is no ground truth to validate or train against — until we make some.
- **The agent generates touches it knows exactly.** Capture clean screen frames during agent gestures → `(frame, known-touch)` is free ground truth. Use it to (a) **measure** trace_extractor's true positional error, and (b) **train Model 1** (a learned extractor that beats the heuristic). This is the unblocking step for the whole leap.
- **Prerequisite that wasn't obvious:** the corpus must be **color, full-frame screen captures** (like `data/extracted_frames`, or AVFoundation USB capture) — NOT the grayscale 210×455 RL frames and NOT the external hand-clips. The Stage-1 capture must save color screen frames.

## Status / Next
- WDA was **down** on all device ports this session (iproxy tunnels up for both XRs, UDIDs ...3A78002E and ...3A60802E, but no WDA listening; no xcodebuild loop). On-device ground-truth collection (Stage 1) and BIG SPIN OCR verification are blocked on bringing WDA up — flaky to do unattended; left for a supervised session.
- Built device-free deliverables instead: Stage-1 collection script (ready to run once WDA is up), Model-1 training scaffold, Option-D board localizer. See commits on branch `feat/spin-and-vision-sequence-leap`.
