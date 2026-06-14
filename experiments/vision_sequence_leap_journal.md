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

## On-Device Self-Labeling — VALIDATED (2026-06-14, iPhone_XR)

Brought WDA+Appium up (directly, no launcher monitor — see ops note) and ran the new `collect_self_labeled_traces.py`. Color MJPEG capture works: full-res **1792×828 RGB** frames with per-frame timestamps. Findings from ~140 collected gestures:

1. **The trace only renders for on-board FLICKS, not arbitrary drags.** Random screen drags in empty space produce no trace and don't move the board (`warm@label=0`). Fixed the sampler to start on the board (~x∈[0.38,0.62], y∈[0.50,0.80]) and flick outward — every such flick produces a trace (50/50). (Some flicks even land real tricks, e.g. a BACKSIDE 180.)
2. **The orange trace LAGS the flick by ~0.4-0.5s.** The swoosh peaks a median **+0.33s after the gesture END** (50/50 samples, range +0.14..+0.66s) — it's a lingering render of the completed flick path, not coincident with the finger. My first instantaneous-point labeling therefore marked the trace-rich frames "inactive."
3. **Latency-shifted labels align with the trace.** Sweeping `latency_s`: at 0.0s only **1%** of active-frame labels land on the trace; at **0.40s → 78%**, at **0.50s → 85%** (median warm-px at label jumps 0 → ~2000). **`latency_s ≈ 0.45` is the validated offset** — now the default in `train_trace_extractor.py` + a trace-presence gate keeps only aligned frames.

**Conclusion: Asher's agent-as-labeler approach WORKS on-device.** Clean color frames + known flicks + a ~0.45s latency shift → a corpus where labels sit on the rendered trace ~80% of the time. The label naturally lands at the flick's end (swoosh head). For Model 2 the richer target is the whole flick *path* the swoosh encodes (refinement). Corpora on disk: `data/self_labeled_traces/iPhone_XR_20260614_011235` (50, board-flick) + `..._011856` (250) — gitignored under /data/.

## MODEL 1 TRAINED + GENERALIZES (2026-06-14) — the leap's perception foundation is proven

- Trained the `GaussianBumpPredictor` U-Net on the 50-sample board-flick corpus (latency 0.45 + trace gate → 123 trace-aligned positives + negatives = 192 frames, 20 epochs, MPS). Loss fell monotonically **0.044 → 0.00096**.
- **Held-out generalization (243 trace frames from the SEPARATE 250-sample session, seed 23 ≠ train seed 11):** predicted touch → ground-truth-label **median normalised distance 0.032** (≈3% of frame), **100% of predictions land on the orange trace** (median warm-px at prediction 3186 ≥ 2703 at the label). Visual overlays (`tmp/model1_pred/`) show pred (red) and label (green) coincident on the swoosh.
- This is the go-signal: a tiny self-labeled corpus already yields a learned trace extractor that accurately reads the finger trace on unseen data. Scaling the corpus (the 250-sample run, more sessions, more devices) → a production Model 1, then run it on expert play to label Model 2's sequence data. Ops/cache caveat: training stdout was block-buffered (run `python -u`).

## Ops Note — WDA Stability
- WDA bring-up via `launch_services.py` FLAPPED hard (iproxy "died unexpectedly" → auto-restart → port 4723 wedged). Root cause: the launcher started a SECOND iproxy on 8100 conflicting with the pre-existing tunnel, and its restart monitor then wedged.
- **Fix that worked: launch WDA + Appium DIRECTLY** (`xcodebuild ... test-without-building` + `appium --port 4723`), reusing the existing iproxy tunnels, with NO launcher monitor. Stayed rock-solid for ~25 min across 340 gestures + a training run. The launcher's auto-restart was the instability, not WDA itself. Fold a "reuse existing iproxy / don't double-start" guard into launch_services.

## Scaling Pass #1 + the Real Bottleneck: DOMAIN GAP, not data quantity (2026-06-14)

- **Spin button coord CONFIRMED.** Asher's reference images + a gridline-localised screenshot put the spin button (camera + circular-arrows, a HOLD control) at **~(0.055, 0.40)** — essentially the configured (0.0604,0.404). So the coord was right; spin's failure is the base gesture/combo (holding spin over a 360-pop-shove → no recognised trick), which is a True Skate domain question for Asher, not a localisation bug.
- **Expert corpus is real and rich:** `Projects/Robotics & hardware/Training_Data` — 12GB, **264 screen recordings** (RPReplay/ScreenRecording, 750×1624 portrait), `Sorted/Flatground/{kickflips,360_flips,...}` deeply trick-labeled. (The "Hand Clips" were website footage.) This is Model 2's fuel.
- **Scaling infra built:** multi-session dataset (point at the parent dir), in-memory frame+heatmap cache (training 120s→~45s/epoch; full-res-per-item was the bottleneck), tunable `--img` res + `--base-channels`, checkpoint metadata. Corpus grown to ~596 board-flick samples on disk (50+250+296).
- **Honest result — naive scaling did NOT beat v1 on expert transfer, AND the test was confounded.** Trained v2 on 300 samples but (to train fast under Asher's *active* laptop GPU contention) shrank it to 288×128 + base_ch=16 — changing res/arch AND data at once. On 3 expert clips v2 landed on-trace 52-88% (conf ~0.56) vs v1's 57-100% (conf 0.67-0.88). So this says "the smaller/lower-res model is weaker," NOT "more data hurts." A clean data-scaling test must hold arch/res fixed.
- **The deeper finding (the real lever):** v1 was trained ONLY in the agent's warehouse park; the expert clips are in OTHER parks (brighter SLS arena). Neither model has seen that domain. **Expert transfer is a DOMAIN-GAP problem, not a data-quantity problem** — more warehouse data won't close it. The right scaling is **domain-matched collection** (run the agent self-labeling in the same parks the expert clips use — needs True Skate park-switching) and/or semi-supervised training on expert frames. Logged here so the next pass targets domain, not just count.
- Ops: heavy U-Net training contends with the user's live laptop use (epochs swung 45s↔4min). Prefer small models / offload, or train when the laptop is idle.

## Park-Switching Works → Domain-Diverse Collection Unblocked (2026-06-14)

- Navigated True Skate's menus via Appium: bottom **SKATEPARKS** tab (≈0.30,0.95) → **All** tab (≈0.10,0.16) → park row. Confirmed switching loads a new park (state persists across Appium sessions). Installed parks: **The Workshop** (warehouse, the agent's default), **The Glass House** (glass arena), **Skatepark: Underpass** (outdoor urban). SLS-arena parks (the expert clips' domain) are NOT installed — store/download only.
- **Self-labeling transfers across parks unchanged:** on-trace yield at latency 0.45 — Glass House **87%**, Underpass **75%** (vs warehouse ~78%). The trace mechanic + latency are park-invariant; only the background changes. So three distinct visual domains are now collectable.
- Added `--park glasshouse|workshop|underpass` to `collect_self_labeled_traces.py` (`switch_to_park()` does the menu nav; row y-positions are position-dependent — re-check if the installed set changes). Output dirs get a `_<park>` tag.
- **This is the corrected scaling lever** (vs scaling pass #1's same-park data): a multi-park corpus → a domain-robust Model 1 that should generalise to UNSEEN parks (incl. the SLS arena the expert clips use). Multi-park collection (Glass House + Underpass, 150 each) running; next: train a fixed-arch Model 1 on balanced multi-park data and re-test transfer to the SLS-arena expert clips.

## Status / Next
- **DONE this session:** self-labeling pipeline validated end-to-end on-device; Model 1 trained + generalizes; board localizer + reanchor; spin extension (code) — all committed on `feat/spin-and-vision-sequence-leap` (not pushed).
- **Next (supervised):** (1) scale the self-labeled corpus (250+ run on disk; collect on XR2/XS too) and train a production Model 1; (2) run Model 1 on screen-recorded expert play → Model 2 sequence dataset; (3) fix the spin execution (second finger in one W3C payload) + confirm the spin input with Asher; (4) Option-D first 2-trick line using the board localizer + reanchor.
