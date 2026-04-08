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

## Current Devices
- iPhone XR + iPhone 11 (both @2x, 414×896 logical points)
- iPhone XS excluded for now — different coordinate space (@3x, 375×812), will add later with coordinate normalization

## Next Steps
- Coordinate normalization to support iPhone XS as third parallel device
- Test end-to-end with both devices running simultaneously
- Monitor for Appium/WDA stability issues under parallel load