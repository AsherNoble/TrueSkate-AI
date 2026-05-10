# Gesture System Standardisation — Journal

## Motivation

The codebase had accumulated three competing terms ("action", "gesture", "swipe") for the same concept, no single document defining the coordinate system, and stale pixel-space values left over from before coordinate normalisation was introduced. This session audited the full gesture/coordinate stack and standardised it.

## Decisions Made

### Terminology: "gesture" adopted everywhere
- "gesture" = a single touch path (waypoints + duration + easing_power)
- "gesture recipe" = the structured dict `{gestures: [...], delays: [...]}`
- "gesture parameters" = the flat CMA-ES float vector
- "action" and "swipe" retired for these concepts
- Rationale: trick library JSON already used "gestures"; WDA endpoint is `/wda/perform_trick_gestures`; "action" has a specific RL meaning that conflicts

### Coordinate system: normalised [0, 1], no y_offset
- All three supported devices share the 19.5:9 aspect ratio (XR/11: 1:2.1643, XS: 1:2.1653 — < 0.05% difference)
- `scale_to_device(norm_x, norm_y, device_w, device_h)` is the complete and sufficient transform
- No y_offset, no per-device viewport padding needed or added
- `SAFE_Y_MAX = 0.8371` (derived from 750/896) is the hard upper Y bound to avoid the home indicator; applies equivalently to all supported devices

### Source of truth: GESTURES.md
- Created at repo root; absorbs and supersedes `trick_libraries/TRICK_LIBRARY_FORMAT.md`
- Referenced from CLAUDE.md, .github/copilot-instructions.md, README.md
- All source file docstrings updated to point here instead of repeating coordinate/gesture facts inline

## Initial Sweep (initiated 12pm 9/5/2026)

### Bugs Fixed (separate from this session's main scope)

These were fixed prior to this standardisation pass:
- `_COORD_SIGMA = 40.0` → `0.10` in `action_param.py` (pixel-space value crippling CMA-ES coord exploration)
- `cma_stds[coordinate_mask] = 20.0` → `0.05` in `cmaes_optimizer.py` (same issue for library-seeded runs)
- PPO action space bounds (`_X_MAX = 375.0`, `_Y_MIN/MAX` from `812.0`) → normalised `[0, 1]`
- `spin_button_xy` (logical points) passing through `scale_to_device` in PPO → fixed, now normalised `(0.0604, 0.4040)`
- `touch_actions.py` stale module docstring and `DEFAULT_SCALE_FACTOR = 2` / `pixels_to_points` removed

### Renames Applied

| Old name | New name | File |
|---|---|---|
| `unpack_action()` | `unpack_gesture_params()` | `rl/cmaes/action_param.py` |
| `execute_action()` | `execute_gesture_params()` | `rl/cmaes/action_param.py` |
| `ActionPlan` | `GestureRecipe` | `rl/ppo/trick_conditioned_action.py` |
| `decode_action_vector()` | `decode_gesture_params()` | `rl/ppo/trick_conditioned_action.py` |
| `execute_action_plan()` | `execute_gesture_recipe()` | `rl/ppo/trick_conditioned_action.py` |
| `execute_action_vector()` | `execute_gesture_params_vector()` | `rl/ppo/trick_conditioned_action.py` |

## 2nd Sweep (initiated 10am 10/5/2026)

### y-boundary system replaced

- `SAFE_Y_MAX = 0.8371` (upper-only, incorrectly derived) retired
- Replaced with `Y_BOUND_MIN = 0.12` / `Y_BOUND_MAX = 0.88` in `rl/gestures.py` — single source of truth
- Both RL pipelines updated: `action_param.py` `_BOUNDS_RAW` (was `[0.5, 0.8371]`), `trick_conditioned_action.py` `_Y_MIN/_Y_MAX` (was `0.65 / 0.8651`)

### Utility touch functions normalised

- `reset_position(driver, device_w)` → `reset_position(driver, device_w, device_h)` — y from `0.0558 * device_h`
- `skip_loading_screen(driver, x, y)` → `skip_loading_screen(driver, device_w, device_h)` — position `(0.8454, 0.8393)` computed internally
- Per-device `loading_screen_skip_xy` config keys removed from `DEVICES`; `DeviceWorker.loading_screen_skip_xy` property deleted
- All call sites updated: `device_worker.py` (×5), `launch_services.py`, `execute_trick.py`

## Deffered / TODO

- `src/trueskate_ai/rl/gestures.py` is architecturally misplaced in `rl/` — contains execution infrastructure (`scale_to_device`, `execute_static_push`) that belongs in `sim/`. Safe to move but touches several import lines. Deferred.
- `scripts/inspect/execute_trick.py::execute_recipe()` is reusable logic buried in a script — candidate for `sim/gesture_executor.py`. Deferred.
- `touch_actions.py` legacy functions (`swipe`, `flick`, `drag`) not used in RL pipeline. Removal is fix 12, deferred per instructions.
- `trick_libraries/kickflip.json` contains un-normalised logical-pixel coordinates — will break `execute_recipe`. Flagged but not converted (user to handle manual coordinate values per protocol).
