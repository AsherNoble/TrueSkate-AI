# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

TrueSkate-AI trains an RL agent to perform skateboarding tricks in the iOS game True Skate. The current approach uses CMA-ES (evolutionary strategy) to optimize continuous gesture parameters executed on a physical iPhone via Appium/WebDriverAgent.

**Current milestone:** Land a 360 flip from a fixed board position.

**Status:** Pipeline is fully operational. The agent has landed pop shove-its, varial flips, 360 flips, and nightmare flips. CMA-ES tends to converge on reliable medium-reward tricks rather than volatile high-reward ones.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

External tools: Appium (npm), WebDriverAgent (Xcode), ffmpeg, libimobiledevice.

- **XR rig access:** By default, and almost always, the XR phones are connected to `training-server`, not this laptop. Run XR device/service commands there via the tailnet (`tailscale ssh training-server@training-server`); only use a local connection when explicitly confirmed.

- Device UDID stored in `.env` (not hardcoded)
- WDA project: `~/Projects/WebDriverAgent` (needs `-allowProvisioningUpdates` after Xcode updates)
- Appium: localhost:4723, WDA: localhost:8100
- True Skate bundle ID: `com.trueaxis.skate`

## Architecture
src/trueskate_ai/
├── labeling/       # CV pipeline: video → per-frame touch labels (legacy, pre-RL)
├── vision/         # CV: vision_ocr, board_localizer, scene_classifier, color_recorder, self_label (legacy datasets removed)
├── rl/             # RL: reward, device_worker, run_logger, worker_pool
│   ├── cmaes/      # CMA-ES (active): action_param, cmaes_optimizer, curriculum
│   └── ppo/        # PPO (experimental track): trainer, policy, collector, buffer, trick_conditioned_action, metrics
├── sim/            # Device interaction: touch_actions, gestures, gesture_recipe, trick_info_reader, known_tricks
├── utils/          # trajectory_spline, notify (data_loader removed)
├── monitoring/     # status
├── eval/           # (placeholder)
└── trajectories/   # (placeholder)
curricula/          # Per-trick CMA-ES reward + dimensionality config (JSON)
trick_libraries/    # Mined gesture-recipe libraries (median + best) per trick
scripts/            # Entry points: launch_services, train/train_cmaes, data/build_trick_library, etc.
experiments/        # Experiment journal, standalone experiments
tmp/                # Debug output (gitignored)

### RL Pipeline (active development)

1. **Gesture parameterization** (`src/trueskate_ai/rl/cmaes/action_param.py`): flat float vector → N gesture slots. `PARAMS_PER_SLOT = 8` (3 normalised waypoints + duration + easing_power) × N, plus N−1 inter-slot delays = 9N−1. Spin-family curricula append a trailing 3-param spin block (`SPIN_PARAMS`: gate, t_start, t_end — a HOLD control) → 9N+2. N=2 no-spin = 17 params; N=2 with spin = 20. N is curriculum-defined; the two length classes are disjoint mod 9, so `infer_layout(len)` recovers `(N, use_spin)` from vector length alone. Curved drags are essential — straight swipes don't reflect real gameplay. See `GESTURES.md` for coordinate conventions.

2. **Touch execution** (`src/trueskate_ai/sim/touch_actions.py`): `curved_drag` primitives executed via Appium W3C Actions. Slots run as overlapping gestures in a single `perform()` call (parallel, not sequential).

3. **OCR / trick detection** (`src/trueskate_ai/sim/trick_info_reader.py`): `detect_trick()` takes a BGR numpy array. Green (landed) / red (failed) pixel anchoring locates the trick-notification cluster, crops above it, then fuzzy-matches OCR lines against the `KNOWN_TRICKS` list — all in `trick_info_reader.py`. Apple Vision (`src/trueskate_ai/vision/vision_ocr.py`, via pyobjc) is the only OCR backend and does raw text-line extraction only; pytesseract and the `TRUESKATE_OCR_BACKEND` backend selector have both been removed.

4. **Reward** (`src/trueskate_ai/rl/cmaes/curriculum.py` + `curricula/*.json`): Reward tiers are no longer a hardcoded tier function. Each target trick has a `curricula/<trick>.json` — a flat `{trick: reward}` dict plus `default_reward` (for recognised-but-unlisted tricks) and a `failure_multiplier` (`"near_miss"` / `"zero"` / constant float). Loaded as a `Curriculum`; `Curriculum.score(result)` does the lookup — landed → base reward, failed/unknown → `failure_multiplier(base)`, combos take the max-scoring component. The target trick auto-gets reward 1.0 if unlisted. `rl/reward.py` now owns only the OCR capture window, the `RepetitionPenalty` (repeat-landing decay `1/(1+count)`, with "360 FLIP" / "BACKSIDE 360 FLIP" exempt), and shared utils: `normalize_trick_name` (applies the "540" → "360" OCR workaround) and `near_miss_multiplier` = `max(0, base*(base-0.1))` (the default failure multiplier).

5. **CMA-ES optimizer** (`src/trueskate_ai/rl/cmaes/cmaes_optimizer.py`, entry point `scripts/train/train_cmaes.py`): Evolutionary optimization over gesture params. JSONL logging. Params clamped via `clamp_params` (NaN/inf → bounds midpoint) to prevent Appium crashes. Gesture y is bounded by `Y_BOUND_MAX = 0.88` (normalised, in `sim/gestures.py`) to stay clear of the home-indicator zone, and x by `X_BOUND_MIN = 0.12` to clear the left-edge in-game buttons.

6. **Trick library** (`scripts/data/build_trick_library.py`, `scripts/inspect/execute_trick.py`): Extracts gesture recipes from JSONL logs (median + best params), replays via WDA. See `GESTURES.md` for the full schema and coordinate reference.

**PPO (experimental, secondary track)** — `src/trueskate_ai/rl/ppo/` (`trainer`, `policy`, `collector`, `buffer`, `trick_conditioned_action`, `metrics`) is a live but secondary trick-conditioned policy-gradient track. Reward there is binary (`compute_conditioned_reward` in `rl/reward.py`). CMA-ES remains the active approach.

### SLS Frame→Gesture Data Collection (XCTest 30fps — active, independent of CMA-ES)

A second track collects a **frame→gesture corpus** (for a future sequence model that predicts the gesture from a clip of frames). It fires random SLS-mix gestures and records the screen — NOT CMA-ES, no reward. Headless 30fps capture is Apple's **XCTest screen recording via Appium** (`src/trueskate_ai/vision/xctest_capture.py`); the CoreMediaIO/DAL screen-mirror path is dead on this rig's macOS/iOS (see memory `ios-dal-screen-capture-wedge`).

- **Collector** `scripts/data/collect_sls_xctest.py` — records bounded `--segment-min` `.mov` segments (default **60s — keep short**: `stop_and_save` returns the whole `.mov` as one base64 HTTP response, which fails above ~114 MB / ~90s) while firing gestures + logging a per-gesture manifest. Hardened with: a **gameplay guard** (`vision/gameplay_filter.is_menu_frame` — skips gestures fired into True Skate's replay/menu, relaunches to exit), per-segment resilience (a failed segment is skipped, never fatal), and a **start-fail cap** (`--max-start-fails`: hammering `rec.start()` re-wedges the recorder, so it exits for a clean supervisor restart). **Spin:** `--spin-frac 0.15` in production — the slice splits 50/50 `spin_flick` (single flick + held spin button, Model-1 trainable; meta carries `spin_active`/`spin_hold_*_s`/`payload_total_s`, see `GESTURES.md`) and nslot-`spin`.
- **Aligner** `scripts/data/align_xctest_traces.py` — slices each gesture's frame window from the `.mov` → per-gesture sample dirs (`frame_NNN.png` + `meta.json`), deletes the `.mov`. Async per segment. Δ (command→pixel offset) is currently 0 (uncalibrated; repro-validated, a constant `frame_times` shift to refine later).
- **Corpus filter** `scripts/data/flag_menu_samples.py` — marks already-collected replay/menu samples with a `.menu` file. **Training loaders MUST exclude any sample dir containing `.menu`.**

**CRITICAL OPERATIONAL PREREQ — the remotexpc tunnel daemon.** The appium-xcuitest driver only auto-deletes on-device XCTest recording attachments when its **remotexpc tunnel registry is reachable**, which needs a ROOT daemon: `sudo appium driver run xcuitest tunnel-creation` (staged as `scripts/ops/com.trueskate.remotexpc-tunnel.plist` → `/Library/LaunchDaemons/`). Without it, attachments accumulate in testmanagerd until recording wedges with `XCTDaemon.ScreenRecordingError Code=7 "Failed to write file"` (despite free disk) — this benched BOTH phones. Recovery: ensure the daemon is up, then `scripts/recover_remotexpc_attachments.sh` (wraps the official `appium … cleanup-videos`, `--dry-run`/`--delete`) clears the backlog. See memory `xctest-recording-attachments-accumulate`.

**Rig ops (launchd):** `com.trueskate.services` (launch_services: Appium+WDA+iproxy, health-aware monitor), `com.trueskate.collect.xr1|xr2` (supervised collectors via `scripts/rig_collect.sh`), `com.trueskate.watchdog.xr1|xr2` (`scripts/collection_watchdog.sh`: one persistent fleet incident alert on state change, then one recovery — never periodic reminders), and the root `com.trueskate.remotexpc-tunnel`. Ports are +3/device (XR1 wda 8100, XR2 wda 8103).

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
- **CMA-ES over full RL** — 17-dim continuous space (canonical 2-gesture, no-spin layout) suits black-box evolutionary optimization; avoids data requirements of SAC etc.
- **Data throughput is the bottleneck** — True Skate runs at 1× real-time; GPU can't accelerate the live interaction loop. ~15K steps/hour
- **Reward shaping is critical** — partial credit for trick components guides exploration; aggressive tier compression prevents convergence on wrong tricks
- **OCR misreads are a real signal problem** — "360"→"540" misread was causing 1.0 tricks to score 0.6
- **No y_offset** — all supported devices share the 19.5:9 aspect ratio; `scale_to_device(norm_x, norm_y, device_w, device_h)` is the complete coordinate transform. RL gesture bounds: `X_BOUND_MIN = 0.12` (left edge hosts in-game buttons), `X_BOUND_MAX = 1.0`, `Y_BOUND_MIN = 0.12`, `Y_BOUND_MAX = 0.88` (defined in `src/trueskate_ai/sim/gestures.py`). Note: iPhone 11 runs with Display Zoom always enabled, which reduces its UIKit logical resolution from 414 × 896 to 375 × 812 — the same as the XS. This is reflected in `DEVICES` in `device_worker.py` and does not break normalised coordinates because the aspect ratio is unchanged.

## Experiment Journal

Located at `experiments/rl_poc_experiment_journal.md`. Read at start of relevant conversations and append key findings, bugs, and decisions. Keep entries brief.

## Conventions

- Debug/temporary output → `tmp/` (gitignored)
- `.venv/` is the sole virtual environment
- `*.pth` model files gitignored; stored in `notebooks/models/`
- Commit messages: use `[type]: [message]` (e.g. `feat: built a new thing`), one commit at a time
- Function names should reflect actual behavior precisely (e.g., `reset_position()` not `go_to_waypoint()`)
- Use full absolute paths — tilde expansion (`~/`) is unreliable in tooling
- Communication style: prefer terse dot points over prose. State what changed and the next step; skip the essay/recap. Less word salad.

## Known Issues / Next Steps

- App-focus check before each eval (agent wasted 1800+ evals screenshotting Clock app)
- Auto-terminate on N consecutive zero-reward evals
- CMA-ES multimodal problem: unimodal Gaussian averages over bimodal landscape → IPOP/BIPOP restarts or novelty bonus needed
- Hard flip reward tier missing (currently scores 0.0)
- Long-term: hierarchical architecture — sequence model over trick names commanding low-level RL policies
- **SLS capture: XCTest recorder depends on the root remotexpc tunnel daemon** — if it's down, attachments accumulate and recording wedges ("Failed to write file"); see the SLS collection section + memory `xctest-recording-attachments-accumulate`.
- **SLS capture spatial coverage** — `reset_position` returns the board to a fixed anchor each gesture, so capture concentrates in one park zone. A "wander" mode (relocate + reset to local waypoints) would broaden it (idea parked, not built).
