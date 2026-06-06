# Scene Classifier — "am I still in a True Skate skatepark?"

## Motivation
A recurring 24h-run failure: the agent accidentally exits the skatepark (pause
menu, app backgrounded, an alarm/Clock popup, or the level reloads) and then
spends *hours* firing random swipes at the home screen or another app, burning
thousands of dead evals (see `rl_poc_experiment_journal.md` — the Clock-app
incident, 1800+ wasted evals).

`DeviceWorker.ensure_foreground()` only checks `query_app_state == foreground`.
It catches "True Skate isn't the foreground app" but NOT "True Skate is
foreground but we're on a menu / not actually skating". We need a visual check.

## Approach
A tiny binary CNN: input a frame, output P(in an active skatepark). Run it at
the end of each eval (cheap, gated behind a trained model so it's a no-op until
one exists). If P is low for a device, recover (re-activate app → skip loading →
reset) instead of continuing to swipe into the void.

- Input: grayscale, resized to 64×64, /255. Same transform for MJPEG frames
  (210×455 grayscale) and Appium screenshots (full RGB) — both go through
  `preprocess()`.
- Model: 3 conv blocks → FC → 1 logit. ~tens of k params; CPU inference < few ms.
- Threshold default 0.5; tune on val set.

## Data
**Positives (in skatepark)** — already plentiful in the repo:
- `data/extracted_frames/*/img_*.jpg` — gameplay frames from screen recordings.
- `logs/runs/*/ocr_failures/*/frame_*.png` — frames where a trick notification
  fired (definitionally in-park).
- `logs/runs/*/frames/*/frame_*.png` — eval composites (when frame recording on).
- `tmp/*.png` — ad-hoc eval captures.

**Negatives (NOT in skatepark)** — TODO, collect at home:
- iOS home screen, app switcher, Control Center.
- True Skate pause menu / level select / settings / replay screens.
- Other apps (Clock, Photos, Settings) — the actual failure cases.
- Black/loading frames.
Drop these into `data/scene/negatives/` (any subfolders, jpg/png).

Label noise caveat: some "positive" eval frames may catch a slide-in
notification or a transient menu. Acceptable for a first model; revisit if val
accuracy plateaus low.

## Pipeline
1. `python scripts/data/build_scene_dataset.py --negatives-dir data/scene/negatives`
   → writes `data/scene/manifest.json` (stratified train/val split, balanced).
2. `python scripts/train/train_scene_classifier.py --manifest data/scene/manifest.json`
   → trains, prints val accuracy, saves `notebooks/models/scene_classifier.pth`.
3. Enable at runtime: set `SCENE_GUARD_MODEL=notebooks/models/scene_classifier.pth`
   in `.env`. `DeviceWorker` picks it up via `SceneGuard.from_env()`; with no
   env var / no file it stays disabled (zero overhead).

## Status
- [x] Model + `SceneGuard` inference wrapper (`vision/scene_classifier.py`).
- [x] Dataset builder (positives auto-mined, negatives from a folder).
- [x] Training script.
- [x] Default-off inference + recovery hook in `DeviceWorker.evaluate()`.
- [ ] Collect negatives at home (the only manual step left).
- [ ] Train, set threshold, enable, validate against a deliberate exit-to-home.
- [ ] Tune recovery policy (currently: re-activate → skip loading → reset).

## Open questions / ideas
- Could reuse MJPEG frames already captured during the eval instead of an extra
  screenshot, to avoid added latency when the guard is enabled.
- A 3-class head (skatepark / menu / other-app) would let recovery differ per
  case (menu → tap-back vs other-app → re-activate).
- Longer term this is a natural input to the hierarchical policy idea.
