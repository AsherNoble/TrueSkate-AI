### Motivation
- CMA-ES is unimodal — converges on dominant reward region, stops exploring
- 1200-eval run with 360 flip = 1.0, everything else = 0.0: landed some tricks, zero 360 flips, no gradient signal
- Previous varial/hard flip convergence confirmed: CMA-ES finds tricks but gets stuck
- Decision: move to trick-conditioned neural network + PPO

### Architecture
- Input: trick index → learned embedding (dim 32)
- Trunk: Linear(32→128) → ReLU → Linear(128→128) → ReLU
- Output: 42 params
  - 4 gesture slots × 9 (x0,y0,x1,y1,x2,y2, duration, easing_power, gate) = 36
  - 3 inter-slot delays = 3
  - Spin control: gate + t_start + t_end = 3
- Slot gate: sigmoid, thresholded at 0.5 inference-time — unused slots fire zero actions
- Spin t_start/t_end: normalized [0,1] over total sequence duration, converted to absolute time at execution

### Training
- Algorithm: PPO (clips updates — prevents unlearning, more stable than REINFORCE)
- Reward: binary 1.0 if OCR matches target trick, 0.0 otherwise (trick-conditioned = no partial credit needed)
- Loop: sample trick from 36-trick list → network outputs params → execute on device → OCR → reward → PPO update
- 3 devices in parallel for rollout collection

### Spin Control
- Reference resolution: 750×1624, button region x:[5,85] y:[615,695]
- Converts to logical points (414×896 iPhone 11): centre ≈ (25, 362) — needs on-device verification
- Execution: schedule two taps (toggle on, toggle off) on absolute timeline interleaved with gestures

### Trick List (36 flatground tricks)
OLLIE, NOLLIE, KICKFLIP, DOUBLE KICKFLIP, TRIPLE KICKFLIP, HEELFLIP, DOUBLE HEELFLIP,
TRIPLE HEELFLIP, POP SHOVE-IT, FS POP SHOVE-IT, 360 POP SHOVE-IT, FS 360 POP SHOVE-IT,
FRONTSIDE 180, BACKSIDE 180, FRONTSIDE 360, BACKSIDE 360, VARIAL KICKFLIP, VARIAL HEELFLIP,
NIGHTMARE FLIP, HARD FLIP, DOUBLE HARD FLIP, 360 HARD FLIP, INWARD HEELFLIP, LASER FLIP,
360 FLIP, 360 DOUBLE FLIP, BACKSIDE FLIP, BACKSIDE DOUBLE FLIP, FRONTSIDE FLIP,
FRONTSIDE DOUBLE FLIP, BACKSIDE HEEL FLIP, FRONTSIDE HEEL FLIP, BACKSIDE 360 FLIP,
FRONTSIDE 360 FLIP, BACKSIDE 360 HEEL, FRONTSIDE 360 HEEL

### Next Steps
- Verify spin button logical-point coordinate on device
- Run short on-device PPO sanity sweep (`--updates 1 --steps-per-update <num_devices>`) and inspect JSONL
- Tune PPO rollout/update ratios after first live run

### Implementation Status (2026-04-19)
- Added trick-conditioned policy module: `src/trueskate_ai/nn/policy.py`
- Added action decode/execute path for 42 params with slot gates + spin timing:
  `src/trueskate_ai/rl/trick_conditioned_action.py`
- Added binary trick-conditioned reward hooks:
  `compute_conditioned_reward()` / `get_conditioned_reward()` in `src/trueskate_ai/rl/reward.py`
- Added parallel rollout collector reusing `DeviceWorker`:
  `src/trueskate_ai/rl/collectors/trick_conditioned_collector.py`
- Added PPO trainer + rollout buffer:
  `src/trueskate_ai/rl/ppo/trainer.py`, `src/trueskate_ai/rl/ppo/buffer.py`
- Added CLI entrypoint and spin calibration utility:
  `scripts/train_ppo.py`, `scripts/calibrate_spin_button.py`
- Added README usage updates for PPO training + spin calibration.

### Live Smoke Test (2026-04-19)
- Started full local stack (`scripts/launch_services.py`) and confirmed Appium/WDA health on all three configured devices.
- Ran:
  `python scripts/train_ppo.py --updates 1 --steps-per-update 3 --epochs-per-update 1 --minibatch-size 3 --checkpoint-every 1 --device-count 3 --settle-time 0.3 --wait-time 0.0`
- First run exposed WDA action-sequencing issues in multi-slot execution:
  - "Actions list cannot be empty"
  - "pause action item must be preceded by pointerMove"
- Fixed execution builder in `trick_conditioned_action.py`:
  - ensure pointerMove precedes pauses
  - submit only active fingers in `ActionChains(..., devices=...)`
- Re-ran smoke test successfully:
  - no rollout execution errors
  - checkpoints written under `logs/runs/ppo_run_20260419_184111/`
  - sample records show `error: null` for all three devices

### Pilot Run (2026-04-19, 10 updates × 12 steps)
- Command:
  `python scripts/train_ppo.py --updates 10 --steps-per-update 12 --epochs-per-update 2 --minibatch-size 12 --checkpoint-every 5 --device-count 3 --settle-time 0.3 --wait-time 0.0`
- Artifacts:
  `logs/runs/ppo_run_20260419_184549/`
- Aggregates from update summaries:
  - avg `detection_rate`: **0.0667**
  - avg `match_rate`: **0.0000**
  - max `detection_rate`: **0.1667**
  - `error_rate`: **0.0** across all updates
- Outcome:
  - runtime stability is now good (no WDA action failures during pilot)
  - signal quality is still the bottleneck (few detections, zero target matches)

### PPO Action Squashing Migration (2026-04-21)
- Switched policy action bounding from **hard clipping** to **tanh squashing** in `src/trueskate_ai/nn/policy.py`.
- Rationale:
  - clipping creates discontinuities at the boundary and mismatched log-prob behavior
  - tanh keeps actions naturally in `[-1, 1]` with smoother gradients
- Implemented corrected squashed log-prob (change-of-variables):
  - sample pre-squash `u ~ Normal(mu, sigma)`
  - action `a = tanh(u)`
  - log-prob uses Jacobian correction term `-log(1 - a^2 + eps)`
- For PPO evaluation on fixed actions:
  - recover pre-squash with stable `atanh(clamp(a, -1+eps, 1-eps))`
  - compute corrected log-prob consistently with rollout sampling path
- Note:
  - entropy bonus still uses base Gaussian entropy (approximation), while PPO ratio uses corrected log-probs.

### Hindsight Relabel Combo Fix (2026-04-21)
- Fixed HER relabeling for combo OCR strings (e.g. `"KICKFLIP + 50-50 GRIND"`).
- Relabel now splits components on `" + "`, normalizes each component, and relabels recognized non-target trick components instead of skipping whole combo strings.

### Resume-Run Feature (2026-04-21)
- Added explicit checkpoint resume support for PPO runs (`--resume-from <checkpoint.pt>`).
- Resume now restores:
  - policy weights
  - optimizer state (if present)
  - trainer counters (`update_idx`, `eval_num`)
- New resumed runs keep a fresh run folder but log lineage explicitly via a `resume_start` JSONL event containing:
  - source checkpoint path
  - source run id (if available in checkpoint metadata)
  - whether optimizer was restored
  - resume start update/eval counters
- Checkpoint payload upgraded to include:
  - `policy_state_dict`
  - `optimizer_state_dict`
  - `trainer_state`
  - `config`
  - `run_metadata`
- Backward compatibility retained for older policy-only checkpoint formats.

### Post-gesture Idle Stall Instrumentation + Fix (2026-04-27)
- Investigated intermittent 3s+ idle windows after gesture execution and before reset in PPO rollouts.
- Added per-sample timing telemetry to PPO JSONL (`sample` + `update_summary`):
  - `action_exec_s`, `reward_eval_s`, `eval_total_s`
  - `post_eval_wait_s`, `reset_s`
  - `capture_attempts`, `skipped_captures`, `detection_capture_idx`, `capture_elapsed_s`
- Added reward-capture diagnostics path in `reward.py`:
  - `capture_and_detect_with_diagnostics()`
  - optional diagnostics returns from `get_conditioned_reward()` / `get_reward()`
- Latency fix in capture scheduler:
  - when `action_start_time` is far in the past, stale capture slots are skipped instead of executing a burst of redundant back-to-back OCR captures.
  - preserves capture-window semantics while reducing tail latency.
- Rollout reset behavior changed in `trick_conditioned_collector.py`:
  - each worker now schedules reset immediately after its eval completes (still waiting for all resets before next batch dispatch).
  - removes unnecessary post-eval board idle caused by waiting for slower peers to finish before starting resets.
- Note: local short-run reproduction command was attempted, but WDA was not reachable (`127.0.0.1:8100`) in this session, so on-device timing confirmation remains to be captured in the next live run.

### XR Live Validation (2026-04-27)
- Brought up XR-only environment using launch script with device list constrained to `iPhone_XR`.
- Ran one-step PPO validation:
  `python scripts/train_ppo.py --updates 1 --steps-per-update 1 --epochs-per-update 1 --minibatch-size 1 --device-count 1 --checkpoint-every 1 --wait-time 0.0 --capture-count 14 --capture-interval 0.15 --settle-time 0.5`
- Run artifact: `logs/runs/ppo_run_20260427_172254/`
- Observed sample telemetry:
  - `action_exec_s`: 6.4537
  - `reward_eval_s`: 0.4166
  - `eval_total_s`: 7.3853
  - `post_eval_wait_s`: 0.0001
  - `reset_s`: 0.8249
  - `capture_attempts`: 1
  - `skipped_captures`: 13
- Interpretation:
  - Post-eval idle before reset is effectively eliminated for this sample (`post_eval_wait_s` near zero).
  - OCR capture burst behavior is avoided (single capture attempt + skipped stale slots), with much lower reward-eval tail time than full 14-capture loop.

### OCR Reliability Hotfix — post-push monitoring (2026-04-27)
- User-observed regression: trick text was being missed too often after the prior capture-skip optimization.
- Fix implemented:
  - OCR monitoring now starts **immediately after static push** (not later) via an explicit `on_post_push` hook.
  - Added `ContinuousTrickMonitor` using MJPEG frames for continuous detection during active eval execution.
  - Post-action capture still runs for the configured window; monitor + post-action detections are merged with de-dup trick concatenation (`" + "`).
- Live validation run:
  - `logs/runs/ppo_run_20260427_183521`
  - `monitor_frames_checked`: `45`
  - `capture_attempts` (monitor + post-action): `51`
  - `monitor_elapsed_s`: `4.865`
- Outcome: screen checking now clearly begins in the post-push phase and continues through eval, restoring dense OCR polling behavior.

### Merge Specificity Fix (2026-04-27)
- Issue observed: one landed trick could be merged as multiple outputs (example: `"HARD FLIP + DOUBLE HARD FLIP"`).
- Fix in `reward.py` merge pipeline:
  - split merged detections into trick components,
  - dedupe by specificity, keeping the most specific overlapping variant,
  - preserve true combos via existing `" + "` formatting.
- Example behavior after fix:
  - `HARD FLIP` + `DOUBLE HARD FLIP` → `DOUBLE HARD FLIP`
  - `KICKFLIP + 50-50 GRIND` + `KICKFLIP` → `KICKFLIP + 50-50 GRIND`

### PPO Warm-Start Plan (2026-05-10)
- PPO is parked. The trick-conditioned net's exploration cliff (HER pumping V(trick) for unsolved targets, killing advantage signal — see 2026-04-29 entry in `rl_poc_experiment_journal.md`) means PPO from-scratch on hard targets is unproductive.
- New plan: use CMA-ES + per-trick curricula (`curricula/<trick>.json`) to find converged params for ~all flatground tricks. Each successful run yields a 17-param point in gesture space, stored as `trick_libraries/<trick>.json` (existing schema).
- Warm-start the trick-conditioned PPO policy by **supervised pretraining** the head: for each trick library, the net's output for that trick's embedding is regressed to the library's params (MSE on the 17-dim vector, before tanh squashing). This seeds the policy in already-good basins per trick.
- After supervised warm-start, unfreeze and run online PPO updates as previously implemented — same collector, same binary reward, same 3-device parallel rollout. The hope is that per-trick basins are stable enough that PPO refines rather than rediscovers, and the shared trunk learns the structure between tricks (e.g. shared scoop kinematics across the kickflip family).
- Prerequisite: ship CMA-ES libraries for at minimum the kickflip family (KICKFLIP, VARIAL KICKFLIP, HARD FLIP, NIGHTMARE FLIP, 360 FLIP) — those are the ones with proven warm-starts already. Heelflip / shove-it families to follow.
- This pivot keeps PPO code intact (no schema breakage) and makes the curriculum work double duty: per-trick reliability now, neural cold start later.
