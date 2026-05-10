# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TrueSkate-AI trains an RL agent to perform skateboarding tricks in the iOS game True Skate. The current approach uses CMA-ES (evolutionary strategy) to optimize continuous gesture parameters executed on a physical iPhone via Appium/WebDriverAgent.

**Current milestone:** Land a 360 flip from a fixed board position.

**Status:** Pipeline is fully operational. The agent has landed pop shove-its, varial flips, 360 flips, and nightmare flips. CMA-ES tends to converge on reliable medium-reward tricks rather than volatile high-reward ones.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install opencv-python numpy torch torchvision scipy pillow appium-python-client matplotlib requests cma pytesseract
```

External tools: Appium (npm), WebDriverAgent (Xcode), ffmpeg, libimobiledevice.

- Device UDID stored in `.env` (not hardcoded)
- WDA project: `~/Projects/WebDriverAgent` (needs `-allowProvisioningUpdates` after Xcode updates)
- Appium: localhost:4723, WDA: localhost:8100
- True Skate bundle ID: `com.trueaxis.skate`

## Architecture
src/trueskate_ai/
├── labeling/       # CV pipeline: video → per-frame touch labels (legacy, pre-RL)
├── vision/         # PyTorch Datasets (TouchDataset, VideoDataset) — legacy
├── rl/             # RL components (action_param, cmaes_optimizer, reward)
├── sim/            # Device interaction (touch_actions, trick_info_reader, known_tricks, execute_trick)
├── utils/          # TrajectorySpline, data_loader
├── eval/           # (placeholder)
└── trajectories/   # (placeholder)
scripts/            # Entry points: launch_services, train_cmaes, build_trick_library, etc.
experiments/        # Experiment journal, standalone experiments
tmp/                # Debug output (gitignored)

### RL Pipeline (active development)

1. **Gesture parameterization** (`src/trueskate_ai/rl/cmaes/action_param.py`): 17-param vector → 2 gesture slots (3 normalised waypoints + duration + easing_power each) + 1 inter-slot delay. Curved drags are essential — straight swipes don't reflect real gameplay. See `GESTURES.md` for coordinate conventions.

2. **Touch execution** (`src/trueskate_ai/sim/touch_actions.py`): `curved_drag` primitives executed via Appium W3C Actions. Slots run as overlapping gestures in a single `perform()` call (parallel, not sequential).

3. **OCR / trick detection** (`src/trueskate_ai/sim/trick_info_reader.py`): `detect_trick()` takes BGR numpy array. pytesseract with 3× upscaling, grayscale, threshold, character whitelist, green/red/white pixel anchoring, fuzzy match against 248-entry `KNOWN_TRICKS` list. Known issues: pytesseract hallucinations — Apple Vision framework replacement planned.

4. **Reward** (`src/trueskate_ai/rl/reward.py`): Tiered scoring. Current v3: 360 FLIP = 1.0, varial/kickflip = 0.6, 360 shove-it/BS 360 = 0.3, everything else = 0.0. Failed multiplier: `base * (base - 0.1)`. FS/FRONTSIDE tricks zeroed. OCR normalizes "540" → "360" before scoring.

5. **CMA-ES optimizer** (`src/trueskate_ai/rl/cmaes_optimizer.py`, entry point `scripts/train_cmaes.py`): Evolutionary optimization over gesture params. JSONL logging. Params clamped to prevent inf/NaN Appium crashes. y-bounds capped at 750 to avoid home indicator zone.

6. **Trick library** (`scripts/data/build_trick_library.py`, `scripts/inspect/execute_trick.py`): Extracts gesture recipes from JSONL logs (median + best params), replays via WDA. See `GESTURES.md` for the full schema and coordinate reference.

### Labeling Pipeline (legacy — pre-RL pivot)

`trace_extractor.py` → `video_labeler.py` → `visualize.py`. CV-based touch label extraction from screen recordings. Superseded by RL approach but code remains.

## Gesture & Coordinate Reference

See `GESTURES.md` at the repo root for the authoritative reference on:
- Terminology (gesture, gesture recipe, gesture parameters)
- Normalised coordinate system and why no y_offset is needed
- Supported device screen ratios and `Y_BOUND_MIN` / `Y_BOUND_MAX`
- Gesture and recipe JSON schema
- Execution flow and code cross-references

## Key Design Decisions

- **Curved gestures required** — Asher's expert domain knowledge; straight swipes don't work
- **CMA-ES over full RL** — 17-dim continuous space suits black-box evolutionary optimization; avoids data requirements of SAC etc.
- **Data throughput is the bottleneck** — True Skate runs at 1× real-time; GPU can't accelerate the live interaction loop. ~15K steps/hour
- **Reward shaping is critical** — partial credit for trick components guides exploration; aggressive tier compression prevents convergence on wrong tricks
- **OCR misreads are a real signal problem** — "360"→"540" misread was causing 1.0 tricks to score 0.6
- **No y_offset** — all supported devices share the 19.5:9 aspect ratio; `scale_to_device(norm_x, norm_y, device_w, device_h)` is the complete coordinate transform. RL gesture y bounds: `Y_BOUND_MIN = 0.12`, `Y_BOUND_MAX = 0.88` (defined in `rl/gestures.py`). Note: iPhone 11 runs with Display Zoom always enabled, which reduces its UIKit logical resolution from 414 × 896 to 375 × 812 — the same as the XS. This is reflected in `DEVICES` in `device_worker.py` and does not break normalised coordinates because the aspect ratio is unchanged.

## Experiment Journal

Located at `experiments/rl_poc_experiment_journal.md`. Read at start of relevant conversations and append key findings, bugs, and decisions. Keep entries brief.

## Conventions

- Debug/temporary output → `tmp/` (gitignored)
- `.venv/` is the sole virtual environment
- `*.pth` model files gitignored; stored in `notebooks/models/`
- Commit messages: 10–20 words, one commit at a time
- Function names should reflect actual behavior precisely (e.g., `reset_position()` not `go_to_waypoint()`)
- Use full absolute paths — tilde expansion (`~/`) is unreliable in tooling

## Known Issues / Next Steps

- Swap OCR to Apple Vision framework (pyobjc) — pytesseract hallucinations worsening
- App-focus check before each eval (agent wasted 1800+ evals screenshotting Clock app)
- Auto-terminate on N consecutive zero-reward evals
- CMA-ES multimodal problem: unimodal Gaussian averages over bimodal landscape → IPOP/BIPOP restarts or novelty bonus needed
- Hard flip reward tier missing (currently scores 0.0)
- Long-term: hierarchical architecture — sequence model over trick names commanding low-level RL policies