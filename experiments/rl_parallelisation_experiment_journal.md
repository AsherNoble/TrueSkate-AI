# RL Parallelisation — Experiment Journal

## Aim
Multiply data collection throughput by evaluating N candidates simultaneously across N physical iPhones. Data collection at 1x real-time is the binding constraint — a GPU can't accelerate the live interaction loop, but more phones can.

## What We Built
- `device_worker.py`: `DeviceWorker` class — one per phone, owns its Appium driver + MJPEG connection, runs `evaluate()` independently. `DEVICES` list is the single source of truth for device configs.
- Refactored `cmaes_optimizer.py`: `run()` creates N workers, dispatches via `ThreadPoolExecutor`. Generation loop processes candidates in rounds of N: parallel execute → parallel reset → next round. Reset is the sync point between rounds.
- `launch_services.py`: starts N WDA + N Appium instances from `DEVICES` config.
- Device configs use `.env` alias names (e.g. `IPHONE_XR_UDID`); raw UDIDs never appear in logs or terminal output.
- Popsize auto-rounds down to nearest multiple of device count; max_evals rounds to nearest multiple of popsize.
- All logging (terminal + JSONL) tagged with `device_id`; frame dirs include device name.
- Coordinate normalization: canonical action space is 375×812pt (XS width, aspect ratio preserved per device). Wider devices get equal top/bottom padding; model always operates in canonical space, `norm_to_device()` in `action_param.py` handles the per-device transform at execution time. `DEVICES` entries carry `logical_w`/`logical_h`; `reset_position` taps `device_w / 2` rather than a hardcoded x.

## Current Devices
- iPhone XR + iPhone 11 (both @2x, 414×896 logical points)
- iPhone XS (@3x, 375×812 logical points) — fully supported via coordinate normalization

## Results
- End-to-end parallelisation confirmed working across all 3 devices — round-based dispatch and reset sync point hold up correctly under real parallel load; no race conditions observed.
- Practical device limit estimated at 4–6 iPhones — bottleneck is USB hub bandwidth, Mac host thread overhead, and MJPEG stream concurrency rather than CMA-ES logic.

## Next Steps
- Monitor for Appium/WDA stability issues under higher device counts

## Update — Post-eval reset scheduling refinement (2026-04-27)
- Addressed idle board time between eval completion and reset by changing reset scheduling from strict end-of-round barriers to immediate per-worker reset submission:
  - PPO collector: `collect_rollouts()` now submits reset for each worker as soon as that worker’s rollout future completes.
  - CMA-ES loop: per-candidate resets are submitted immediately after each evaluation future resolves, instead of waiting for the full round to finish first.
- Added reset-timing telemetry in logs (`post_eval_wait_s`, `reset_s`) to quantify whether synchronization overhead remains.
- Expected effect: better wall-clock utilization and less visible idle board rolling/stationary time on fast-finishing devices while preserving batch semantics for the next dispatch cycle.

## Validation Snapshot (2026-04-27)
- XR live PPO micro-run (`device-count=1`) confirms new timing fields are being emitted and sane.
- Observed from `ppo_run_20260427_172254`:
  - `post_eval_wait_s` ≈ `0.0001`
  - `reset_s` ≈ `0.8249`
- This confirms reset starts immediately after eval completion in the current scheduler, with no meaningful idle barrier wait in the single-worker case.

## Update — OCR monitor start alignment (2026-04-27)
- Added a post-push execution hook in both action paths:
  - CMA-ES: `action_param.execute_action(..., on_post_push=...)`
  - PPO: `trick_conditioned_action.execute_action_vector(..., on_post_push=...)`
- This hook starts a continuous MJPEG-based trick monitor immediately after static push so OCR checking does not wait until action completion.
- Evaluator paths (PPO collector and `DeviceWorker.evaluate`) now merge:
  1. monitor detections collected during execution, and
  2. post-action capture-window detections,
  with duplicate trick names removed and `" + "` concatenation preserved.
