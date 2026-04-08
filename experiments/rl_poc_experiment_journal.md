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

## Next Steps
- Swap OCR to Apple Vision framework (pyobjc)
- App-focus check before each eval
- Auto-terminate on consecutive zero-reward evals
- Fix hard flip reward tier
- Novelty bonus / IPOP restarts to escape convergence traps
- Emulated phone for parallel eval throughput
- Long-term: hierarchical architecture — sequence model commanding low-level RL policies
