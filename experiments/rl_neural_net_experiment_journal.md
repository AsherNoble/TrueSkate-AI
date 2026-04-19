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
