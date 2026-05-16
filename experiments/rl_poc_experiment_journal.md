# 360 Flip RL POC — Experiment Journal

## Background
- Project started Feb 2026 as behavioral cloning from gameplay recordings
- Hit data bottleneck: touch-labeling too expensive, U-Net memorized 135 frames
- Pivoted to RL — agent generates own actions via Appium/WDA, knows what it did by definition
- Asher is an expert True Skate player (daily SLS park sessions), guiding design decisions

## Action Parameterization
- Curved drags essential — Asher vetoed straight swipes from domain knowledge
- Started at 23 params (3 uniform curved_drag slots × 7 + 2 delays)
- Asher wanted agent "free to choose" gesture structure, not hardcoded scoop/flick/catch
- Reduced to 17 params: 2 slots × (3 waypoints + duration + easing_power) + 1 delay
- Easing power [0.3, 3.0] — may need widening
- y-bounds capped at 750 (was 896) after accidental app exits via home indicator zone
- Slot 1 initialized as horizontal scoop from tail; slot 2 as NE flick from right-of-center

## OCR Pipeline
- pytesseract with 3× upscaling, grayscale, threshold, character whitelist
- Green pixel anchoring for landed tricks, red for failed (TrickResult namedtuple)
- Fuzzy match against 248-entry KNOWN_TRICKS list
- Fixed: "560"→"360" misread (normalized at OCR level, not reward level)
- Fixed: BACKSIDE fuzzy-matching to FAKIE — added as proper modifier
- Fixed: tesseract merging letters+digits ("BACKSIDE560") — regex space insertion
- Screenshot timing: wait_time=0, 5 captures at 0.25s spacing (settled after debugging)

## Bugs Fixed
- CMA-ES crash when --max-evals cut off mid-generation
- Premature convergence via `while not es.stop()` — now loop on eval budget
- Appium crash on inf/NaN coordinates — clamp_params handles non-finite values
- argparse --wait-time default was 1.5 while get_reward default was 0.0 — aligned to 0.0

## First 500-Eval Run (popsize=12)
- Pipeline fully operational end-to-end
- Agent landed: pop shove-its, varial flips, hard flips, inward heelflips, 360 flips, BS 360 flips
- First 360 FLIP at gen 26 (eval 324); multiple hits by gen 34+
- CMA-ES converged on pop shove-it basin (~0.4 reliable) rather than volatile 1.0
- Mean reward plateaued 0.2–0.35; 360 flips were sporadic outliers
- Key observation: slot 1 evolved into a velocity push before executing a scoop — emergent behavior

## Reward Tuning (post first run)
- Failed multiplier: flat 0.4× → `base * (base - 0.1)` — penalty scales inversely with difficulty (Asher's idea)
- Tiers compressed: flips 0.5, rotations 0.3, 180s 0.2, everything else 0.1
- Shove-it check moved before rotation keywords so "360 POP SHOVE-IT" → 0.3
- Display format :.1f was rounding 0.75→0.8 — use :.2f

## CMA-ES Tuning
- Popsize 12→24 (2× default formula `4 + floor(3·ln(n))` for noisy objective)
- CMA-ES is NOT gradient descent — evolutionary: samples from multivariate Gaussian, updates mean+covariance toward top performers
- Optimizes distribution mean → gravitates to reliable tricks, not best tricks

## Popsize 50 Run (tuned rewards)
- Flips appearing earlier (varial kickflips gen 4, inward heelflip gen 5) — reward tuning working
- Overall rewards much lower: mean ~0.02, 70%+ evals produce nothing
- Shove-its correctly suppressed (0.01–0.05)
- Weak signal: popsize 50 + compressed rewards = noisy selection

## OCR Fixes (round 2)
- "BACKSIDE 360" split across lines → "360" fuzzy-matched "360 FLIP" → false 1.0. Fixed with bare rotation exact-match.
- argparse `--wait-time` default 1.5 vs `get_reward` default 0.0 — aligned to 0.0

## Parallel Touch Execution
- Sequential slots had too much Appium latency even at delay=0
- Switched to overlapping gestures in a single W3C Actions `perform()` call

## Initial Mean Revision (from frame analysis)
- Scoop moved from y=575 (mid-board) to y~680 (tail). Flick changed to upward diagonal from center.
- Slots now geometrically distinct: scoop = horizontal at tail, flick = diagonal upward from center

## Frame Resolution
- 84×84 → 210×455 (preserves 750×1624 aspect ratio, shortest side = 210)

## CMA-ES Multimodal Problem
- Unimodal Gaussian averages over bimodal reward landscape → mean drifts to dead zone
- Practical: tighten reward to suppress competing optima, or IPOP/BIPOP restarts

## Reward Tiers v3 (aggressive focus)
- 360 FLIP: 1.0, varial/kickflip: 0.6, 360 pop shove-it/BS 360: 0.3, everything else: 0.0
- FS/FRONTSIDE tricks zeroed — can never yield 360 flip
- Hard flips zeroed — agent was converging on them

## White Text / Unknown Status OCR
- Added white pixel detection for tricks with white score text
- Bug: `ys.min()` anchors off top of white region, not score line — fix: use `ys.max()` for unknown status

## OCR Fixes (round 3)
- Split trick names: added merge logic to join adjacent candidates ("360" + "POP SHOVE-IT")
- Digit-only filter was dropping "360" before merge — removed
- pytesseract hallucinations worsening ("HARD FLIP" → "LIPSLIDE"). Apple Vision framework identified as replacement.

## Nightmare Flip Convergence Run (~2400 evals)
- Converged on nightmare flip (0.6) — progress over pop shove-it but not target
- Gesture drifted to manual territory at gen 175, then alarm popped Clock app to foreground
- Agent spent 3+ hours screenshotting Clock app — 1800+ dead evals
- Safeguard needed: check foreground app is `com.trueaxis.skate` before each eval
- Auto-terminate on N consecutive zero-reward evals (e.g. 48 = 2 gens)

## Trick Library Pipeline
- `build_trick_library.py`: extracts recipes from JSONL logs (median + best params → decoded `curved_drag` args)
- `execute_trick.py`: replays recipes via Appium
- Extracted: 360 flip (2 samples, 1.0), double kickflip (18, 0.6), nightmare flip (1196, 0.6)
- Hard flip family all scored 0.0 — not matched in reward tiers (bug to fix)

## CMA-ES Revival + Target-Relative Reward (2026-04-29)
- **PPO detour (Apr 19–27):** pivoted to trick-conditioned PPO to escape CMA-ES EV trap; see `rl_neural_net_experiment_journal.md` for full detail. Core issue: match_rate stayed 0 — HER pumped reward=1 experiences that biased V(trick) upward, killing advantage signal for hard targets. Parked PPO; returning to CMA-ES with better reward design.
- **Binary reward failure (confirmed empirically):** ran CMA-ES with reward=1.0 for KICKFLIP only, else 0.0. 6 hits in 4980 evals — covariance update diluted to noise by 59/60 zero-fitness ties per generation. Never converged.
- **`execute_trick.py` coordinate scaling bug fixed:** `execute_recipe` was passing canonical 375×812 coords directly to Appium on non-XS devices. iPhone 11/XR (414×896) need `norm_to_device` scaling — same as `execute_action` already did. XS unaffected (canonical = XS native space). Explains why mined recipes didn't replay reliably.
- **Target-relative reward (`compute_reward_for_target`):** target=1.0, same mechanical family=0.4 (substring-based: any trick containing FLIP → kickflip family), any other recognized trick=0.05, nothing=0.0. Non-landed: `base * (base - 0.1)`. Prevents EV trap (pop shove-it scores 0.05 under flip target) while solving dilution (most evals score >0, giving CMA-ES a real population ranking).
- **Warm-start support added** (`--initial-mean` on `train_cmaes.py`): seeds CMA-ES mean from `best_gestures` in a trick library JSON, coordinate sigmas halved to 20.0.
- **First successful convergence run:** `--target-trick KICKFLIP --initial-mean trick_libraries/kickflip_20260427_105222.json` — CMA-ES converged on 360 FLIP basin (adjacent in param space to kickflip warm-start) within ~20 generations. Back-to-back 360 FLIPs observed for the first time. ~1280 360 FLIP hits logged.


## Next Steps
- App-focus check before each eval
- Auto-terminate on consecutive zero-reward evals
- Run dedicated per-trick convergence runs using `--initial-mean` for each trick in the library — exclude 360 FLIP from kickflip-family curriculum to prevent basin drift

## Reward Refactor + Curriculum System (2026-05-10)
- `reward.py` cleaned up: deleted dead `compute_reward()` (hardcoded 360-flip tiers), deleted hallucinated entries from family frozensets (`360 HEELFLIP`, `HARD HEELFLIP`, `NIGHTMARE HEELFLIP`, `LASER HEELFLIP`, `HARD SHOVE-IT`, `NIGHTMARE SHOVE-IT`, `FRONTSIDE SHOVE-IT`, bare `SHOVE-IT`, `VARIAL HEEL`), moved family taxonomy out to `known_tricks.py` (the canonical trick taxonomy file).
- `compute_reward_for_target()` substring family logic replaced by `Curriculum` class (`src/trueskate_ai/rl/cmaes/curriculum.py`) backed by per-trick JSON files in `curricula/`.
- Curriculum schema: flat `{trick: reward}` dict + `default_reward` for unlisted recognised tricks + togglable `failure_multiplier` (`"near_miss"` (default) | `"zero"` | float constant). New target tricks are now data-only additions.
- `near_miss_multiplier(base) = max(0.0, base * (base - 0.1))` — clamped ≥ 0; previous formula produced small negatives for low-reward bases like `default_reward=0.05`.
- `scripts/train/train_cmaes.py`: `--target-trick` flag replaced by `--curriculum <path>`. `--initial-mean` stays as an explicit override; otherwise defaults from the curriculum's `warm_start` field.
- `known_tricks.py` restructured: family frozensets (`KICKFLIP_FAMILY`, `HEELFLIP_FAMILY`, `SHOVE_IT_FAMILY`, `ROTATION_FAMILY`, `GRIND_SLIDE_FAMILY`, `SPIN_FAMILY`, `DOLPHIN_DRAGON_FAMILY`, `OTHER_TRICKS`) + `MODIFIERS` (now includes `NOLLIE`). `KNOWN_TRICKS` is now the union. OCR matcher (`trick_info_reader.py`) picks up NOLLIE automatically via the import.
## OCR Garbage on Landed Tricks — Sliding Notification (2026-05-16)
- Bug: `trick_info_reader` logged `no match for OCR output 'LU 0'`/`'L 010'`/`''` on genuinely landed tricks → eval scored 0.
- Root cause (confirmed via `tmp/eval*_cap*.png`): the True Skate trick notification slides in from the left, holds ~0.15–0.3s, then slides out right off-screen. `_ocr_above_anchor` crops relative to the live green score-digit position; on slide frames the trick name is clipped by the screen edge → Vision OCR reads truncated fragments that fail fuzzy match. Intermittent because Appium screenshots sample at ~5fps and may catch only slide frames.
- `ContinuousTrickMonitor` was stopped at `action_end_time` (before the notification appears) — never contributed; removed from the CMA-ES eval path (still used by parked PPO collector).
- Fix — "best-N" capture (`reward.capture_and_detect_with_diagnostics`): every frame gets a cheap OCR-free `find_notification_anchor` check (`AnchorInfo` with a `clipped` flag = anchor bbox hugging a frame edge); candidates ranked unclipped-then-most-pixels; only top 3 OCR'd. Adaptive window stops once the notification has appeared and been absent 3 frames (`max_window_s=3.5` cap).
- OCR failures (anchor found, no match) now dump frames + crops + `diagnostics.json` to `logs/runs/<run>/ocr_failures/eval_<n>/`.
- New JSONL keys: `anchor_candidates`, `ocr_calls`; dropped `monitor_frames_checked`, `monitor_elapsed_s`.
