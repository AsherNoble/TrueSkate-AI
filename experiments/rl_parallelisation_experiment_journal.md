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