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
## Phantom Top-Left Swipe — WDA Drops move-then-pause (2026-05-16)
- Bug: deterministic spurious swipe from top-left downward → menu opens, consistently on the 12th XS eval (candidate_idx=11). Fixed CMA-ES seed makes that param vector recur.
- Root cause: in `execute_n_slot_gestures` (combined branch), a non-primary finger with `starts[i] > 0` was built as `pointerMove(start, dur=0)` → `pause(starts[i])` → `pointer_down` (`include_start_move=False`). WDA drops a standalone zero-duration move when a pause follows it, so `pointer_down` fired at pointer origin (0,0). Eval 12 decodes to a short-delay timing (`starts[1]≈0.02s`) that hits this path — a trigger, not the cause.
- Original design assumed a `dur=0` move survives across a following pause (so no re-move needed). The `force_single_payload` branch already worked around this with `include_start_move=True`; that branch is dead in production (sole caller never sets the flag).
- Fix: `else` branch now uses `include_start_move=True` — re-issues the start move immediately before `pointer_down`, so the down lands at the gesture start. The pre-pause move is kept (harmless). Schedule unchanged (extra `dur=0` action; other fingers padded with `pause(0)`).
## 24h Unattended Rig Hardening (2026-06-06, branch feature/claude-go-crazy)
- **Goal:** continuous 24h collection on XR + XS at home (Intel MacBook `training-server`), remotely run/monitored over Tailscale; iPhone 11 reserved as personal/test phone.
- **Device roles + selection:** `DEVICES` entries carry `role` (`collection`/`personal`); `select_devices`/`resolve_devices` + `--devices/--personal/--all-devices` flags on `train_cmaes.py`, `launch_services.py`, `run_training.py`. Default roster = collection (XR+XS). Ends the comment/uncomment churn on `DEVICES`.
- **CMA-ES reliability bugs fixed:** (1) `ensure_foreground()` in the round loop was unguarded — one dead driver tore down the whole run; now per-worker try/except + `record_failure`. (2) `pool.revive_dead()` / `raise_if_all_dead()` existed but were *only wired into the parked PPO collector* — now called every round; clean `AllWorkersDeadError` abort.
- **launch_services:** replaced fleet-killing `sys.exit` monitor with per-device restart (decaying backoff + wait-for-USB-reappear). One munted XS port no longer stops the rig.
- **Observability:** `monitoring/status.py` writes atomic `logs/status.json` heartbeat each gen; `scripts/status_server.py` serves a self-refreshing dashboard for `tailscale serve`. `utils/notify.py` (stdlib ntfy.sh, no-op when `NTFY_TOPIC` empty — it currently IS empty in `.env`, needs filling) pushes start/device-down/zero-land-stall/all-dead/finish.
- **Supervisor:** `scripts/run_training.py` runs services+status+training under `caffeinate`, restarts crashed pieces, relaunches training with a bumped seed on exit (continuous collection). `deploy/` has a launchd LaunchAgent template + installer.
- **Scene guard (the home-screen-swipe problem):** tiny CNN `vision/scene_classifier.py` + `SceneGuard` (env-gated via `SCENE_GUARD_MODEL`, default-off no-op). Mines positives from existing frames; needs negatives collected at home. Default-off recovery hook in `evaluate()` (re-activate→skip→reset). New JSONL key `in_skatepark`. See `scene_classifier_journal.md`.
## Overnight Multi-Trick Curriculum Pipeline (2026-06-10, branch feat/overnight-curriculum)
- **Goal (demo tomorrow):** 3 phones (XR, new XR2, XS) each run an independent sequential trick queue overnight; auto-advance on land-rate threshold OR eval/time cap; warm starts chained between tricks.
- **Early stop:** `run()` gains `stop_land_rate`/`stop_window` (rolling target-land-rate over last W evals, checked at round boundaries; skips `es.tell` on partial gen). All exits write `run_dir/result.json` (`stop_reason`: max_evals/early_stop/interrupted/all_workers_dead) — machine-readable contract for the orchestrator.
- **Orchestrator:** `scripts/train/overnight_curriculum.py --config configs/overnight/<dev>.json`. Subprocess per trick (SIGINT time-cap → graceful checkpoint), warm-start priority (explicit > session-mined > trick_libraries) with gesture-count guard, mines finished JSONL for own + next target (`--landed-only --min-samples 5`), one retry on infra failures, per-trick log dirs (avoids 3-way status.json collision), ntfy per transition.
- **Param archaeology:** normalisation landed 2026-05-09/10 (PR #13). April logs are canonical-375×812 space, NOT pixels. **iPhone_11 was misconfigured 414×896 all April** (fixed c972b38) → its April params (incl. 1,243 landed 360 FLIPs in r20260429) executed ~10% scaled/offset; `360_flip_20260507_095215.json` was converted with a third (wrong, 414×896) divisor. `scripts/data/convert_legacy_log.py` re-derives normalised coords under both hypotheses (`canonical` = exact for XR runs, `executed-iphone11` = models the bug); both 360-flip variants in `tmp/legacy_libs/` pending replay validation.
- **Mining:** `build_trick_library.py` gains `--landed-only` (failed evals were polluting medians), `--min-samples` (exit 2), `--out-dir`, and combo-component matching ("X + GRIND" counts as X, same as Curriculum.score). Fresh landed-only libraries committed: pop_shove_it (65), hard_flip N=3 (58), varial_kickflip (20), nightmare_flip (7), 360_pop_shove_it (162).
- **Gitignore trap found:** bare `data/` rule silently ignored `scripts/data/` — `build_trick_library.py` had NEVER been tracked. Scoped to `/data/`.
- **pop_size for single-device queues:** 12 (N=2) / 14 (N=3) ≈ CMA default `4+⌊3·ln d⌋`; warm-started runs are local refinement where generation count beats population size.
## Overnight Curriculum — First Full Night (2026-06-11, laptop-hosted, XR1+XR2)
- **Headline: HARD FLIP converged — early stop at 70% rolling land rate (614 evals), first trick ever to hit the threshold.** 360 FLIP reached ~34-36% over 785 evals; POP SHOVE-IT 26% (869 evals).
- **Replay validation (5 trials each, XR1):** 360 FLIP 5/5, HARD FLIP 4/5, POP SHOVE-IT 4/5 — all three libraries demo-ready (`trick_libraries/{360_flip_20260611_045254,hard_flip_20260611_064249,pop_shove_it_20260611_091259}.json`).
- **XR2 (flaky USB):** kickflip 936 evals / 0 pure lands (basin drifted to varial/360 variants — pure kickflip remains elusive); varial + nightmare ran into the morning. WDA on this phone died every ~30-40 min ("connection invalidated"); cable/port suspect — replace before next overnight.
- **Ops lessons:** (1) launch_services' per-device WDA restart wedged twice — supervised `while true; xcodebuild test-without-building` loops per device were the night's real fix; fold that hardening into launch_services. (2) Orchestrator retries must gate on WDA recovery (fixed, committed) — instant retries burned XR2's whole queue in 15 min. (3) Run orchestrators with `python -u`; buffered stdout made failures undebuggable. (4) Error-evals keep the JSONL fresh, so freshness ≠ health. (5) Retry currently re-warm-starts from the original library; resuming from the checkpoint (or re-mining mid-trick) would preserve evolved state — future fix.
- April-params mystery resolved: see 2026-06-10 entry; `executed-iphone11` mapping replay-validated 2/3 and seeded the 360 FLIP run that hit 5/5 replay tonight.
## Night 2 — Deep Refinement + Laser Family (2026-06-12, laptop, XR1+XR2)
- **HARD FLIP 86%** (early stop, 355 evals — broke the 85% ceiling target). **LASER FLIP 70%** (early stop, 405 evals — converged on its FIRST run, warm-chained from the double-laser recipe). 360 FLIP polished 36%→**54%**; 360 DOUBLE FLIP 16%→**48%**; DOUBLE LASER 30% (870 evals total). BACKSIDE 180 failed (0.6%, 858 evals) — rotations join kickflip/nightmare/varial/720/inward-heel on the condemned list.
- **Family-descent works:** double-laser (discovered) → laser (converged in one run). Compositional warm-starting across trick families is the highest-leverage curriculum trick found so far.
- **Unattended ops validated:** stack watchdog (tunnel refresh + appium restart) executed multiple autonomous recoveries incl. one full chain (run death → tunnel refresh → gated retry) with zero human input. Two fixes en route: (1) watchdog must treat fresh JSONL as healthy — /status probes time out mid-gesture and refreshing tunnels mid-eval caused zero-capture rows; (2) per-device xcodebuild DerivedData (-derivedDataPath) — shared DerivedData caused runner crash-looping (22 restarts/90min).
- **Known bug (one manual unstick at 04:30):** train_cmaes can hang after main exits — ThreadPoolExecutor.shutdown(wait=False) leaves a stuck non-daemon worker holding the process; orchestrator polls a corpse until the time cap. Fix: os._exit after final cleanup, or cancel_futures=True. The hang ate ~45 min of XR2's backside-180 slot.
- Converged trick libraries now: HARD FLIP (86%), LASER FLIP (70%), plus high-rate 360 FLIP (54%), 360 DOUBLE FLIP (48%), DOUBLE LASER (replay 4/4).
## Recipes Are Device-Flavored (2026-06-12 morning)
- Replay validation of night-2 libraries: HARD FLIP **5/5 on XR1** (its training device). LASER FLIP **4/5 on XR2** (its training device) but **0/5-as-laser on XR1** (lands 360 FLIPs there — same normalised params, different flip).
- Implication: converged recipes encode device-specific touch-timing/panel characteristics at the margin between adjacent tricks. Validate and demo each library on its training device; treat cross-device transfer as a warm start, not a replay.
- Day-3 queues route accordingly: 360/hard-flip lineage on XR1, laser lineage on XR2. Descent tests running: hard flip→kickflip (XR1), laser→heelflip (XR2); ascent: 360 triple, triple laser.
## CORRECTION: Stance Mirror, Not Device Flavor (2026-06-12 ~09:00)
- **XR2's skater stance was REGULAR (XR1 goofy) until now — Asher caught it.** True Skate labels tricks relative to stance, so every XR2 detection pre-switch is the mirror trick: LASER↔360 FLIP, INWARD HEEL↔HARD FLIP, heel↔kick families, BS↔FS shoves/rotations.
- Supersedes the "device-flavored recipes" entry above: the laser lib replaying as 360 FLIP on XR1 was the mirror, not touch calibration. The "laser 70% convergence" = a second converged 360-flip recipe; the demo's "double laser finale" = a 360 double flip in mirror.
- Flag file with mirror map + affected libraries: `trick_libraries/FLAGGED_stance_mirror_xr2.md`. Pre-switch XR2 JSONLs need label mirroring before mining; post-switch runs are clean. Also re-judge: "inward heelflip 0%" and "backside 180 0.6%" failures were mirror-confused curricula.
- Ops lesson: add stance to the pre-run device checklist (park, camera, stance, DND, Auto-Lock).
## MIRROR THESIS PROVEN (2026-06-12 afternoon)
- **Gesture x-mirror (x -> 1-x) of a converged recipe lands the chiral-mirror trick, converging FAST.** Two parallel experiments, both early-stopped at 70%:
  - LASER FLIP from x-mirror of 360 FLIP recipe → 70% in 625 evals (XR2)
  - INWARD HEELFLIP from x-mirror of 86% HARD FLIP recipe → 70% in **250 evals** (XR1) — faster than the 355-eval hard flip it mirrors.
- **Mirror rule (from trick_vector_curriculum_plan.md, which we are NOT implementing — used only to derive mirrors):** a left-right gesture mirror negates the three chirality axes body_rotation, shove_rotation, kickflip_axis; dolphin_axis unchanged. So HARD FLIP (shove −1, kick +1) ↔ INWARD HEELFLIP (shove +1, kick −1); 360 FLIP (shove +2, kick +1) ↔ LASER FLIP (shove −2, kick −1). Verified empirically both ways.
- **Implication:** every converged trick yields its chiral twin for a few hundred evals of polish. `scripts/data/mirror_library.py` does the transform (x-mirror + bounds-remap). Catalog effectively doubles for free.
- Tooling: `mirror_library.py`. Converged libraries now: HARD FLIP 86%, 360 FLIP, 360 DOUBLE FLIP 70%, LASER FLIP 70%, INWARD HEELFLIP 70%, DOUBLE LASER (replay 4/4), POP SHOVE-IT.
## Negative Result: Thin Priors on Rare Tricks Wander (2026-06-13 overnight)
- DOLPHIN FLIP and DRAGON FLIP, warm-started from thin mined libs (3 and 5 landed samples), both capped at **0% land rate** over ~1160 evals each. Dolphin brushed the family (1 FRONTSIDE DRAGON FLIP combo + 2 DOLPHIN HEELs) but never a clean target; dragon landed 0 family tricks. CMA-ES drifted off the seed into easy neighbours (pop shove-it, FS 360).
- **Conclusion:** a 3-5 sample mined prior is NOT a strong enough anchor for a rare, isolated trick — the seeded gesture region produces easy tricks more often than the target, and zeroing the easy-trap rewards removes the gradient but not the drift. Contrast with the proven winners: MIRROR (x-reflection of a converged recipe) and ASCENT/DESCENT chained from a converged trick both land near-target immediately because the source is a high-quality, many-sample recipe.
- **Roadmap for the dolphin/spin family:** (a) hand-guesstimate a seed (Asher's kickflip approach), or (b) reach them by ascent once we have ONE converged dolphin flip — neither of which we have yet. Thin mining alone is a dead end for these.
- **Spin tricks (big spin / big flip) remain blocked** pending a proper CMA-ES action-vector extension: add the 3 PPO-style spin params (enable + t_start + t_end mirroring SpinControl), thread through bounds/sigma/unpack/execute/warm-start/mining, and verify via OCR before any unattended run. Day task, not an overnight hack.
## Mirror Runs Need the Parent De-Rewarded (2026-06-13)
- A mirrored warm start can slide BACK to the un-mirrored source basin if the curriculum rewards the source trick at all. DOUBLE LASER FLIP (x-mirror of the 70% 360-double recipe) with `360 DOUBLE FLIP: 0.2` in its rewards landed **62 360-FLIPs / 43 360-DOUBLE-FLIPs, 0 lasers** over 331 evals — CMA-ES re-converged on the original, not the mirror.
- **Fix:** zero the source-family rewards (set `360 FLIP` and `360 DOUBLE FLIP` to 0.0). Same run immediately committed to the mirror side — **80 LASER FLIPs + 9 DOUBLE LASER lands** — and started climbing toward the double. The mirrored seed sits close enough to the source basin that any reward gradient on the source pulls it back; removing it forces the commit.
- **Rule:** *mirror the recipe AND zero the parent's reward.* This is the same mechanism as the kickflip descent (parent trick is a comfortable rewarded local optimum) — applies to mirrors too. Single-rotation mirrors (laser←360 flip, inward heel←hard flip) tolerated a small parent reward; the double-rotation mirror did not, because the mirrored gesture is geometrically nearer the source.
- Carries forward: every mirror/descent curriculum should default the source-family reward to 0.0, not a small positive. Update the mirror playbook accordingly.
## Self-Improvement Loop: Re-Seed From a Trick's Own Mine (2026-06-13)
- A trick stuck below convergence can be pushed higher by re-warm-starting it from its OWN mined recipe. 360 TRIPLE FLIP capped at **38%** (1014 evals) off a thin 5-sample prior; re-running it warm-started from that run's mined median jumped to **60% rolling by 409 evals** — over halfway to the 70% stop in under half the evals.
- **Why:** the first run's mined median is a hundreds-of-landed-samples recipe (vs the 5-sample seed it started from), so the second run starts deep in the basin with the coord-sigma already shrunk to local refinement. The thin-prior wandering (see dolphin/dragon negative result) is avoided because the seed is now high quality.
- **Technique:** for any trick that caps in the 30-50% band, mine it and re-run from that library. Iterate until it crosses the early-stop threshold. Cheap, fully automatic, no hand-tuning.
- Distinct from MIRROR (chiral twin of a converged trick) and ASCENT/DESCENT (neighbouring trick in the family tree): this is vertical self-refinement of the SAME trick. The three compose — mine, mirror, and iterate.
## The ~70% Ceiling: Self-Improvement Maxes, Doesn't Break Through (2026-06-13)
- Goal: drive every sub-99% trick to 99% land rate via the self-improvement loop (each trick re-seeded from its own landed-only mine, 3 iterations, stop_land_rate 0.99). Ran 360 FLIP (XR1) and 360 DOUBLE FLIP (XR2) to completion.
- **Iteration-over-iteration (final-window at cap | best 50-eval window ever reached):**
  - 360 FLIP:        i1 26% | 54%  →  i2 66% | 68%  →  i3 66% | 78%
  - 360 DOUBLE FLIP: i1 18% | 34%  →  i2 66% | 78%  →  i3 68% | 72%
- **The first re-seed is the whole lever:** i1→i2 jumps ~40 points (the explicit seed library operates well below the orchestrator's tight landed-only mine). After that it PLATEAUS — i3 adds nothing to the final rate.
- **Intrinsic ceiling ≈ 70% final, with brief peaks to ~78%.** The persistent gap between peak-50 (78%) and final-window (66%) is the real finding: CMA-ES *reaches* the optimum but cannot *hold* it — the population oscillates off a narrow optimum in the sparse multimodal landscape. 99% is NOT reachable by self-improvement alone; the loop maxes a trick at its landability ceiling and stops.
- **Levers to actually break ~70% (untested, ranked):** (1) freeze exploration once peak is found — anneal sigma down hard late / tighten early-stop so it HOLDS 78% instead of drifting to 66%; (2) combo-tolerant reward — count "360 FLIP + NOSE SLIDE" as a land (much of the missing ~30% is the target landing WITH an incidental slide/grind the strict matcher rejects — could lift effective rate 10-15 pts for free); (3) more gesture slots or the spin-timing params for genuinely-missed attempts.
- Decision: stopped the 99% campaign at the ceiling answer rather than grinding all six tricks to the same ~70% wall.
## Spin Action-Vector Extension — SPIN_FAMILY Unblocked (2026-06-14, branch feat/spin-and-vision-sequence-leap)
- **Why:** the 72-trick SPIN_FAMILY (BIG SPIN/BIG FLIP/GAZELLE) was the one hard-blocked family — CMA-ES only emitted curved drags, never the rotate button. Ported the parked PPO `SpinControl(enabled,t_start,t_end)` into the CMA-ES vector.
- **Design:** optional, per-curriculum (`use_spin: true`). 3 spin params appended at the TAIL of the vector → length `8N+(N-1)+3 = 9N+2`. No-spin (≡8 mod 9) and spin (≡2 mod 9) lengths are disjoint, so `infer_layout(len)` recovers `(N, use_spin)` from a vector alone — `device_worker` needs no use_spin flag, only `spin_button_xy`. `use_spin=False` keeps N=2 bit-for-bit 17-dim (zero behaviour change; verified).
- **Execution:** `touch_actions.execute_n_slot_gestures` gained `spin`/`spin_button_pt`; lifts PPO's threaded tap-on/tap-off, and FORCES the combined single-payload path when spin fires (one perform racing the tap thread = PPO's tested topology). Gate sigma 0.4 (CMA can flip it), t sigma 0.2, spin params left OFF the coordinate mask so warm-start keeps spin free.
- **Backward compat (verified):** warm-starting a use_spin run from a legacy (no-`spin`) library appends a neutral spin-off block `[gate=0.0, 0.2, 0.8]` (gate at threshold → ~half the initial population taps the button). `mine_all_tricks` guard fixed (the old `(len+1)%9` check silently dropped EVERY spin eval → zero libraries); group key now `(comp, N, use_spin)` with a `_spin` filename suffix so spin/no-spin never mix into one median. Replay path (`gesture_recipe.py`) fires spin too.
- **First curriculum:** `curricula/big_spin.json` — target BIG SPIN, N=3, warm-started from the 162-sample 360 pop shove-it (board shove reliable; CMA must discover the rotate button to add body rotation). Shove traps ZEROED per the de-rewarded-parent rule.
- **Verified offline:** action_param spin round-trip, optimizer setup (len-29 bounds/sigma/mean for big_spin), warm-start spin-block append, mixed legacy+spin mining round-trip.
- **ON-DEVICE VERIFICATION FAILED THE GATE (2026-06-14, iPhone_XR) — SPIN_FAMILY STILL BLOCKED.** Replayed the 360-pop-shove recipe with vs without spin (5 trials each): no-spin control landed **360 POP SHOVE-IT 4/5** (recipe good); spin-enabled landed **nothing 5/5** (empty/garbage OCR). Enabling spin NULLIFIES the trick rather than adding rotation.
- **Root cause (likely): WDA session conflict, not just a wrong button.** The spin fires a background `driver.execute_script("mobile: tap")` CONCURRENT with the W3C `actions` perform on the SAME session — the tap appears to cancel the in-flight gesture. Same pattern the parked PPO used (its `match_rate` stayed 0). A standalone tap at (0.0604,0.4040) also did nothing visible, so the hardcoded `spin_button_xy` is unconfirmed.
- **Fix (next, supervised):** make the spin a SECOND FINGER inside the SAME `execute_n_slot_gestures` W3C payload (pointer down at t_start·total, up at t_end·total on the button) instead of a concurrent `mobile: tap` thread — exactly how the multi-slot gestures already avoid session conflicts. AND confirm the real spin-trick input with Asher (the coord may be camera-rotate / nonexistent; True Skate may do spins purely by gesture). The spin framework (vector/bounds/mining/curriculum) is correct and committed; only the execution + button need rework + re-verification before any big_spin run.
- **HELD-FINGER FIX IMPLEMENTED + RE-VERIFIED (2026-06-14 follow-up).** Asher confirmed the spin button is a HOLD on the left side (coord he documented = the (0.0604,0.404) in DEVICES). Replaced the threaded `mobile: tap` with a held second finger in the same W3C payload (down at t_start·total, up at t_end·total; re-issues the pre-down move to dodge the WDA phantom-origin bug). Removed the now-dead threading/`_tap_at_time`. **Mechanism is correct:** control (no-spin) lands 360/FS POP SHOVE-IT **5/5** — the held finger does NOT break the gesture. **But spin held t=0..1 (and t=0..0.6) over a 360-pop-shove → 5/5 'none'.** So the gesture stays intact yet holding (0.0604,0.404) yields no recognizable SPIN_FAMILY trick.
- **Still blocked on the BUTTON/GESTURE, not the code.** Open questions for Asher: (1) confirm the exact normalised spin-button coord — the Desktop reference images are macOS-TCC-locked and unreadable by tooling; copy them into the repo or just state the coords; (2) confirm the correct BASE gesture for spins (a 360-pop-shove may be the wrong base — holding spin over it may over-rotate into an invalid/unrecognised trick). A standalone hold on a resting board showed no visible change (expected — spin acts during a trick). Held-finger code is committed; needs the right coord+gesture before a big_spin run.
## Vision-Push Experiment + SLOP Runs (2026-06-14, branch feat/spin-and-vision-sequence-leap)
- **Good finding — bigger push 360-flips the gap.** In SLS Super Crown the flatground 360 catches obstacles, but a vision-guided experiment (board localizer + OCR) showed `PUSH_COUNT=2` builds enough board speed to roll up the runway and cleanly 360-flip the yellow ledge/gap (PUSH_COUNT=1 → "360 FLIP + NOSE SLIDE" combos; =3 overshoots). The static push is now tunable via `PUSH_COUNT` / `PUSH_END_Y` env (`sim/gestures.py`). The earlier "0 360 lands in Super Crown" was a MISCOUNT — the 360 lands as combos (the journal's combo-tolerant-reward case), which the curriculum's max-component scoring credits.
- **SLOP runs (wrong park) — STOPPED, no harm done.** Launched the 360-family (flip/double/triple) self-improvement on BOTH XRs but in SLS OBSTACLE arenas (XR1 Super Crown, XR2 another SLS arena), NOT the 360's clean flatground training park. The obstacle combos are park-/combo-flavored slop. Asher stopped the runs mid-trick; the orchestrator mines only on trick completion, so **NOTHING was mined — `trick_libraries/` is untouched (0 files modified 2026-06-14)**. The 7 run dirs (`logs/overnight/iPhone_XR*/00_360_flip/runs/cmaes_run_20260614_*`) each carry a `SLOP_DO_NOT_MINE.md` marker.
- **Lesson (memory `dont-pollute-well-mined-params`):** self-improvement of a converged recipe must run in its CLEAN TRAINING PARK; verify the recipe lands cleanly there first. Trace-data collection (TRACE_COLLECT) is the separable goal and can run in any park.
- **Salvageable: ~439 SLS-domain trace evals** captured by `TRACE_COLLECT` across those runs (color frame→known-gesture pairs) — valid Model-1 trace data regardless of land rate; the only part worth keeping from the slop runs.
- **Tooling added this session:** `PUSH_COUNT`/`PUSH_END_Y`, `TRACE_COLLECT` (CMA-ES passively saves color trace frames + the gesture vector as label, capped/.noindex'd), board localizer tuned for live in-park frames (deck via tighter ROI + bright-surround + saturated-colour, not the menu bar/ledge), `scripts/inspect/vision_heartbeat.py`. All opt-in / default-off — normal CMA-ES + the well-mined params are unaffected.
## XCTest 30fps Collector: Crash-Loop Root Causes Found + Fixed (2026-06-26, branch feat/dal-capture-prep)
- **Symptom:** both XR collectors looked "running" but produced ZERO completed segments. XR1 had crash-restarted **121 times in ~13h** (a fresh 0B session dir every ~6.5 min), saving nothing. Investigated after asking "why does it crash so frequently — or does the code just think it is?"
- **Ruled out first (evidence, not guesses):** (1) NOT jetsam — `idevicecrashreport` showed zero JetsamEvents in the collection window on either phone (only months-old ones, and they killed True Skate / WebKit, never the WDA runner); (2) NOT a device-side WDA crash — a clean `xcodebuild ... test-without-building` on XR2 emitted `ServerURLHere` in **16s** with no error (the `[User Defaults] KeyboardAutocorrection` lines are benign); (3) NOT `scene_classifier` — `SceneGuard` is a no-op on the rig (no `SCENE_GUARD_MODEL`, no checkpoint), its verdict is logged-only (never gates control flow), and the XCTest collector never calls it or `ensure_foreground`.
- **ROOT CAUSE #1 (the crash-loop): segment payload too large for the retrieval path.** `stop_and_save` pulls the whole `.mov` as ONE base64 HTTP response over Appium/WDA. Measured bitrate with motion ≈ **76 MB/min** at 30fps full-res (not the ~37 the old note assumed). Ladder on XR2: 30s=41 MB ✓, 60s=77 MB ✓, 90s=114 MB ✓ (stop_retrieve 9.7→15.1s) — but a 5-min segment ≈ **380 MB → ~500 MB base64**, which the WDA test-runner can't serialize: the connection aborts (`RemoteDisconnected`) at the stop boundary, every time → the ~6.5-min cadence (5 min record + ~90s dying stop + 30s respawn) and zero saves. **Fix: `--segment-min 1` (60s/~77 MB, comfortable margin).** Lowered the collector default + documented the ~114 MB ceiling in `xctest_capture.py`. With the fix BOTH phones now save a clean ~75 MB segment every 60s and the aligner emits frame→gesture samples.
- **ROOT CAUSE #2 (XR1 wouldn't record at all post-fix): wedged XCTest recording daemon.** After the fix XR1 still failed at `rec.start()` with `XCTDaemon.ScreenRecordingError Code=7 "Failed to write file… make sure there's enough space"` — but the device had **54 GB free**. The 121 crashes left recordings that never cleanly stopped (auto-delete only fires on a clean stop), wedging `testmanagerd`'s recording subsystem (no *active* recording to clear, yet start kept failing). **A headless device reboot (`idevicediagnostics restart`, XR1 has no passcode) cleared it** — recording works immediately after. So crashes are self-amplifying: an oversized-segment crash orphans a recording, and enough orphans wedge the daemon.
- **ROOT CAUSE #3 (why nothing self-recovered): launcher monitor is liveness-only.** `launch_services.py` restarts a device stack only when a tracked proc `poll()`s dead. A wedged-but-alive `xcodebuild` (0% CPU, sleeping, NOT serving its WDA port) reads as healthy, so it's never restarted — XR2 sat like this **12.5h**, and XR1 re-wedged the same way after its reboot until the stale `xcodebuild` was killed by hand (the launcher then brought a clean WDA in ~65s). Ports are +3/device (XR1 wda 8100, XR2 wda **8103** — not 8101). **Recommended fix (not yet done — needs a services restart): health-check the actual `:<wda_port>/status` before declaring WDA up, not just proc liveness.**
- **Net:** both XRs now collecting 30fps→frame/gesture samples on 60s segments (XR2 first to land; XR1 after reboot). Outstanding hardening: launcher health-aware monitor; collector should skip a failed segment + `abort()` the recording instead of crashing (prevents orphan-recording accumulation over a week-long unattended run).
## SLS Collector: Replay-Menu Contamination Found + Guarded (2026-06-26, branch feat/dal-capture-prep)
- **Verifying XR2 alignment surfaced a content (not mechanics) problem.** The align pipeline is mechanically flawless — 300/304 segments aligned (4 = freshest), 0 zero-sample, 0/3622 samples missing frames/meta, 24 frames each at 512×1108, `frame_times` −0.3..+0.87s straddling the gesture, full `meta.json` (17-float params, Δ=0). But **a contiguous ~1300-sample block (~⅓ of one session) captured True Skate's REPLAY/camera-settings menu, not live gameplay** — the random gestures tapped into replay and stuck there.
- **Trap: the skatepark is VISIBLE behind a replay**, so the frame looks like gameplay; thumbnail eyeballing under-counts it. The reliable discriminator is the **bottom button bar** (red `BACK` + teal `SHARE`/`HIDE`/`CAMERA`), which gameplay never shows. `vision/gameplay_filter.py::is_menu_frame` scores saturated red+teal in the bottom 10%: gameplay ≈ (0.0, 0.0), replay ≈ (0.12, 0.12) — validated 13/13 on known frames, resolution-independent. (Cheap heuristic; stands in for the untrained `SceneGuard` CNN.)
- **(b) In-loop guard** (`collect_sls_xctest.py`, default ON): per gesture, screenshot + `is_menu_frame`; if replay/menu → don't log it, and after 2 consecutive hits `terminate_app`+`activate_app`+`skip_loading_screen` to return to gameplay (coordinate-free; BACK isn't safe to tap blindly). Verified live: caught a replay state on startup, relaunched once, resumed clean gameplay (seg 1=16, seg 2=14 gestures); no false-skips, no relaunch loop. The mid-segment relaunch fails that segment's stop → the resilience handler drops it (was replay anyway).
- **(a) Corpus flagger** (`scripts/data/flag_menu_samples.py`): sweeps collected samples, writes a `.menu` marker in each replay/menu sample dir (or `--delete`). **Loaders must exclude dirs containing `.menu`.** Non-destructive default.
- Lesson: random-gesture collection needs a focus/scene guard (the CLAUDE.md "app-focus check" known issue, here as in-app replay). The guard is the going-forward fix; the flagger cleans the backlog.
## XCTest Recorder Wedge: Root Cause = Un-Deleted Attachments; Fixed via remotexpc Tunnel (2026-06-26/27, feat/dal-capture-prep)
- **Both phones wedged with `XCTDaemon.ScreenRecordingError Code=7 "Failed to write file"`** (despite 53 GB free). XR1 had been benched on this for days; XR2 went down after a board-move experiment's collector-interruption tipped it over.
- **Root cause:** the appium-xcuitest driver only auto-deletes on-device XCTest recording attachments when its **remotexpc tunnel registry daemon is running** — it wasn't. So every recording left a stub in testmanagerd's container; they accumulated (XR2 581, XR1 136 ≈ 1/segment) until the store hit its limit and recording failed. A reboot only frees room for ~1 segment, then it re-wedges. And the collector's resilience loop HAMMERING `rec.start()` (~14k retries) re-wedged the daemon even across reboots.
- **Fix:** (1) root LaunchDaemon `com.trueskate.remotexpc-tunnel` running `sudo appium driver run xcuitest tunnel-creation` (needs root; staged `scripts/ops/com.trueskate.remotexpc-tunnel.plist`). (2) `scripts/recover_remotexpc_attachments.sh` (official `cleanup-videos`, dry-run/delete) cleared the backlog (581+136 deleted). (3) Collector `--max-start-fails` cap (exit for clean supervisor restart instead of hammering). Verified: both phones then collected 80+ segments each with the attachment count holding at ~1-2 (auto-delete firing). **XR1 un-benched for the first time.**
- **Also this stretch:** the 5-min-segment payload bug (base64-over-HTTP ceiling → 60s segments); replay-menu contamination (~63% of one session; gameplay guard + `flag_menu_samples` `.menu` markers); the launcher's liveness-only monitor (now health-checks `:port/status`); ntfy collection watchdogs. Δ still 0 (uncalibrated, repro-validated).
- **Process lesson (hard-won):** do NOT interrupt the live workhorse collector for experiments — the bootout/reconnect churn is what tipped XR2 into the wedge.
- **Spatial coverage:** a board-move (a couple of `execute_static_push`) confirmed the boards DO relocate to new park zones and collection captures there (Asher confirmed visually). A "wander" mode to systematically broaden park coverage is PARKED, not built.
## Spin-Family Tricks Need Their OWN Gestures — Not "Cousin + Spin Hold" (2026-07-12, Asher domain knowledge)
- **Resolves the open BASE-gesture question from the 2026-06-14 spin entries (205-216).** Asher (domain expert): taking the EXACT gesture sequence that lands a 360 POP SHOVE-IT and holding the spin button through it does NOT produce a BACKSIDE 360 (or any spin-family cousin). So the on-device "spin over a 360-pop-shove → 5/5 'none'" result was NOT a mechanics/coord bug — the base gesture was simply wrong for the target.
- **Rule:** the required gesture for a spin-family trick is substantially DIFFERENT from its non-spin cousin. You cannot reach a spin trick by adding a spin hold to a working non-spin recipe; the spin trick lives in a different region of gesture space, not "cousin + spin".
- **Implication for `big_spin.json`:** warm-starting a spin curriculum from the 162-sample 360-pop-shove recipe is a WEAK prior (wrong basin). Spin-family tricks need a hand-guessed spin seed (Asher's kickflip approach) or a from-scratch spin search — thin/cousin priors will wander, same as the dolphin/dragon negative result. The spin *framework* (vector/bounds/mining, held-finger execution) is correct and unblocked; only the SEED/base gesture was wrong.
## PPO Spin Mech Unified + SLS Spin Corpus Knob (2026-07-12)
- **PPO spin path was never migrated to the held-finger fix — now done.** `rl/ppo/trick_conditioned_action.execute_gesture_recipe` still fired the spin via a background-thread `mobile: tap` (on at t_start, off at t_end) concurrent with the W3C `perform()` — the exact broken topology from 2026-06-14 (cancels the in-flight gesture on the shared WDA session; PPO `match_rate` stayed 0). Replaced with a HELD finger scheduled inside the SAME single `perform()` (move→pause(t_start·total)→re-move→down→pause(hold)→up), mirroring `touch_actions.execute_n_slot_gestures`. Removed the dead `_tap_at_time`/`threading`/`time` + the no-gesture two-tap branch. Verified offline (mock driver): a spin-enabled 42-dim action emits a `spin` finger with down+hold+up in one perform and NEVER calls `mobile: tap`; spin-disabled emits drag fingers only.
- **SLS corpus: dedicated guaranteed-spin sampler.** The SLS mix already routed `use_spin` through the correct held-finger path (`gesture_sampling` → `execute_gesture_params`), but `--use-spin` was coarse: it only made the nslot branch spin-*length*, ~half gate-OFF, ~12% of fires. Added `sample_spin` (random N-slot base, gate FORCED enabled, hold window ≥ `_SPIN_MIN_HOLD=0.25` so the button is visibly held) + a `spin_frac` mixture slice, tagged `kind="spin"` (distinct corpus label, same execution path). Wired `--spin-frac` into `collect_sls_xctest.py`. Corpus is outcome-agnostic, so a random base + held spin is exactly the (frames→gesture) label the video model needs — no need for a landing spin trick. Verified: `spin_frac=0.2` → ~20% spin samples, every one gate-on with a ≥0.25 hold, layout round-trips; `spin_frac=0` unchanged (no spin).
## Spin ON-DEVICE VERIFIED — Sampler + Fixed PPO + Visual Rotation (2026-07-16, iPhone_XR)
- **Method:** `scripts/inspect/verify_spin_on_device.py` staged to the rig (`~/spin_verify/`) with pre-generated sampler output (40 N=2 + 6 N=3 `sample_spin` vectors, 20 no-spin controls — all offline-certified gate-on/hold≥0.25) so the rig's OWN branch executed them; rig-side precheck confirmed vector/signature compatibility before any fire. XR1's collector stopped via launchd bootout for the test (XR2 untouched), restored after.
- **Results (production stagger env `0.12`):** spin fires **40/40 OK — 0 WDA errors, 0 park-editor, 0 replay-menu**; controls 20/20 identical → the third (spin) finger adds NO editor/menu regression. Fixed-PPO held-finger path (`trick_conditioned_action`, threaded `mobile: tap` fully removed) **2/2 OK**.
- **Visual confirmation (the button ENGAGES):** 360-double-flip recipe fired control vs +spin (`[gate 1.0, 0.05, 0.95]`) with MJPEG frames saved both runs. Control: camera square behind the board, straight run, 9 mph. Spin: mid-trick frames show the camera whirling (straight-down overhead frame, horizon gone), settling ~90-135° rotated facing the side wall at ~1 mph. Same vector except the spin block → rotation only with spin. The held-finger mechanic verifiably drives True Skate's rotate control through the exact collector path.
- **Ship state:** `--spin-frac` is ready for production collection (needs the sampler branch merged to the rig). Ops fixes from the same session: launcher `_coredevice_available` substring bug ("unavailable" contains "available" → absent phones read as found, 2×240s WDA builds burned per cycle), and the verify script's own pgrep guard needed `--devices <name>($| )` anchoring (bare "iPhone_XR" matches the XR2 collector's cmdline). Watchdog blind spot noted: post-reboot it waits for a first segment before arming, so a dead-from-boot rig never alerts.
