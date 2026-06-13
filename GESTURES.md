# Gesture Reference

Single source of truth for gesture structure, coordinate conventions, device geometry, and the trick library schema. All source files, scripts, and AI assistants working in this repo should defer to this document for anything touch/gesture/coordinate related rather than repeating or deriving these facts inline.

---

## Terminology

| Term | Meaning |
|---|---|
| **gesture** | A single touch path: a list of normalised waypoints, a duration, and an easing profile. The atomic unit of touch input. |
| **gesture recipe** | A structured dict `{"gestures": [...], "delays": [...]}` — one or more gestures with inter-gesture timing. Stored in trick library JSON files. |
| **gesture parameters** | The flat float vector that CMA-ES optimises (17 elements for the current 2-slot layout). Decoded into a gesture recipe at evaluation time. |
| **trick library** | A JSON file containing `median_gestures` and `best_gestures` recipes for a named trick, produced by `scripts/data/build_trick_library.py`. |

**Do not use "action" or "swipe" to describe touch gestures.** "Action" is reserved for RL policy output concepts if they arise. "Swipe" is imprecise and should not appear in new code or documentation.

---

## Coordinate System

All gesture coordinates are **normalised**: a point `[x, y]` is expressed as a fraction of screen width and height, both in `[0.0, 1.0]`.

Conversion to device logical points at execution time:

```python
device_x = norm_x * device_w
device_y = norm_y * device_h
```

This is implemented as `scale_to_device(norm_x, norm_y, device_w, device_h)` in `src/trueskate_ai/sim/gestures.py`. It is the **only** coordinate transform in the pipeline — there is no offset, scaling factor, or per-device adjustment beyond this multiplication.

### Why normalised coordinates work across all devices

All three supported devices share an essentially identical screen aspect ratio (19.5:9):

| Device | Logical size (pts) | Aspect ratio | Notes |
|---|---|---|---|
| iPhone XR | 414 × 896 | 1 : 2.1643 | |
| iPhone 11 | 375 × 812 | 1 : 2.1653 | Display Zoom always on — reduces UIKit logical resolution from 414 × 896 |
| iPhone XS | 375 × 812 | 1 : 2.1653 | |

The difference between any two of these aspect ratios is **< 0.05%** — sub-pixel at any practical resolution. A normalised point `[0.5, 0.7]` lands at the same relative position on all three devices. No y_offset, viewport padding, or per-device correction is needed or should be added.

> If a future device with a materially different aspect ratio (> 0.5% deviation) is added to `DEVICES`, this decision must be revisited before that device is used for training.

### Usable coordinate bounds

```
X:         [0.0,         1.0        ]   full screen width
Y:         [Y_BOUND_MIN, Y_BOUND_MAX]   valid RL gesture area

Y_BOUND_MIN = 0.12   # top of board play area (avoids game controls — reset pos, rewind, etc)
Y_BOUND_MAX = 0.88   # bottom of board play area (avoids home indicator zone & game menu)
```

Defined in `src/trueskate_ai/sim/gestures.py`; imported by both RL pipelines (`action_param.py`, `trick_conditioned_action.py`). These bounds apply only to RL gesture parameters — utility gestures (`skip_loading_screen`, `reset_position`, `execute_static_push`) use their own fixed normalised positions and are not constrained by these values.

---

## Gesture Structure

A single gesture object:

```json
{
  "points": [[x0, y0], [x1, y1], [x2, y2]],
  "duration": 0.35,
  "easing_power": 1.5
}
```

### `points`

A list of `[x, y]` normalised waypoints. Minimum 2 points; most trick gestures use 3 (start → control → end). The path is traversed as a sequence of straight segments — use 3 or more points to produce a curved motion.

### `duration`

Total gesture time in seconds. Typical range: `0.03`–`0.80` s.

- Shorter = faster, more explosive motion.
- Longer = slower, more deliberate motion.

### `easing_power`

Controls the velocity profile across the gesture path. Typical range: `[0.3, 3.0]`.

| Value | Profile | Good for |
|---|---|---|
| `< 1.0` | Decelerating — fast start, slow end | Pop / downward motions |
| `= 1.0` | Linear — constant velocity | Neutral |
| `> 1.0` | Accelerating — slow start, fast end | Flicks, quick finishes |

Mathematically: `velocity ∝ t^easing_power` over normalised progress `t ∈ [0, 1]`.

---

## Gesture Recipe

A gesture recipe pairs an ordered list of gestures with inter-gesture delays:

```json
{
  "gestures": [
    {
      "points": [[x0, y0], [x1, y1], [x2, y2]],
      "duration": 0.35,
      "easing_power": 1.5
    },
    {
      "points": [[x0, y0], [x1, y1], [x2, y2]],
      "duration": 0.40,
      "easing_power": 0.8
    }
  ],
  "delays": [0.12]
}
```

### `gestures`

Ordered list of gesture objects. Most tricks use 2–3 gestures. Gestures are executed **sequentially** (not simultaneously) via the custom WDA endpoint.

### `delays`

List of N−1 floats for N gestures. `delays[i]` is the wait time between the end of gesture `i` and the start of gesture `i+1`.

```
delays: [0.12]         → 2-gesture recipe: wait 0.12 s after gesture 0 then fire gesture 1
delays: [0.12, 0.15]   → 3-gesture recipe: 0.12 s gap then 0.15 s gap
```

Negative delays (inter-gesture overlap) are not currently supported via the WDA endpoint but are handled in the CMA-ES two-slot Appium path.

### `spin` (optional)

Spin-family tricks (BIG SPIN, BIG FLIP, GAZELLE FLIP, …) need True Skate's rotate button, which curved drags can't express. When a curriculum sets `"use_spin": true`, the CMA-ES action vector gains a trailing 3-param spin block and recipes carry a decoded `spin` object:

```json
{
  "gestures": [ ... ],
  "delays": [0.12, 0.15],
  "spin": { "enabled": true, "t_start": 0.20, "t_end": 0.70 }
}
```

- `enabled` — whether the rotate button fires this gesture (CMA-ES param: a gate thresholded at `>= 0`).
- `t_start` / `t_end` — fractions `[0, 1]` of the schedule's total duration; the button is tapped **on** at `t_start·total` and **off** at `t_end·total` from a background thread synchronized with the gesture perform.

The spin block is appended **after** the delays, so the vector length becomes `8N + (N−1) + 3 = 9N + 2` (vs `9N − 1` without spin). The two length classes are disjoint mod 9, so `infer_layout(len)` recovers `(N, use_spin)` from a vector alone — no separate flag is stored in logs. The rotate button's normalised position is per-device (`spin_button_xy` in `DEVICES`, default `(0.0604, 0.4040)`), left of the gesture area (`X_BOUND_MIN = 0.12`) so drags never hit it. Recipes without a `spin` key are pure curved drags (all legacy libraries).

---

## Trick Library File Format

Trick library files live in `trick_libraries/` and are produced by `scripts/data/build_trick_library.py`.

```json
{
  "trick": "360 FLIP",
  "median_gestures": { <gesture recipe> },
  "best_gestures":   { <gesture recipe> },
  "sample_count": 1280,
  "reward_stats": { "min": 0.12, "mean": 0.39, "max": 1.0 },
  "source_log": "logs/runs/cmaes_run_YYYYMMDD_HHMMSS/..."
}
```

| Field | Type | Description |
|---|---|---|
| `trick` | string | Canonical trick name, e.g. `"360 FLIP"` |
| `median_gestures` | gesture recipe | Median params across converged evaluations |
| `best_gestures` | gesture recipe | Params from the single highest-reward evaluation |
| `sample_count` | int | Number of matching evaluations used |
| `reward_stats` | object | `min`, `mean`, `max` reward across matched evals |
| `source_log` | string | Path to the JSONL run log this was built from |

All `points` in stored recipes are normalised `[0, 1]`. Legacy files with raw logical-pixel coordinates must be converted before use (see `kickflip.json` — broken, replaced by `kickflip_2.json`).

Libraries mined by `scripts/data/mine_all_tricks.py` also carry `num_gestures` (int) and `use_spin` (bool). When `use_spin` is true, `median_gestures`/`best_gestures` include a `spin` object (see above) and the filename gets a `_spin` suffix; spin and no-spin variants of the same trick are mined into separate files.

---

## Execution Flow

```
1. PUSH
   Static downward gesture on the right side of the board.
   Constants: PUSH_START, PUSH_END, PUSH_DURATION, PUSH_EASING, PUSH_PRE_DELAY
   Executed via Appium ActionChains (separate perform() call — must not share
   a perform() with trick gestures, or iOS interprets 3+ fingers as a system gesture).

2. TRICK GESTURES
   The gesture recipe fires after PUSH_PRE_DELAY total push time.
   Each gesture is sent as an independent POST to:
       /wda/perform_trick_gestures   (custom WDA endpoint, bypasses Appium)
   WDA's synthesizeEventWithRecord blocks until the gesture completes, so the
   Python-side requests.post returns exactly when the gesture finishes.
   Inter-gesture delay is measured with time.perf_counter() and slept on the Python side.

3. RESET
   reset_position() taps the reset button at (device_w / 2, 50) to return
   the board to its starting position.
```

Push constants are defined in `src/trueskate_ai/sim/gestures.py`. The execution loop is in `src/trueskate_ai/sim/gesture_recipe.py::execute_gesture_recipe()` (library replay) and `src/trueskate_ai/rl/cmaes/action_param.py::execute_gesture_params()` (CMA-ES eval path).

---

## Code Cross-References

| Concept | Location |
|---|---|
| `scale_to_device()` | `src/trueskate_ai/sim/gestures.py` |
| `execute_static_push()`, `PUSH_*` constants | `src/trueskate_ai/sim/gestures.py` |
| `execute_gesture_recipe()` | `src/trueskate_ai/sim/gesture_recipe.py` |
| `build_curved_drag()`, `make_touch_pointer()`, `perform_pointer_actions()` | `src/trueskate_ai/sim/touch_actions.py` |
| CMA-ES gesture parameter bounds, decode, execute | `src/trueskate_ai/rl/cmaes/action_param.py` |
| PPO gesture parameter decode, execute | `src/trueskate_ai/rl/ppo/trick_conditioned_action.py` |
| Library recipe replay | `scripts/inspect/execute_trick.py` |
| Build trick library from JSONL log | `scripts/data/build_trick_library.py` |
| Device configs (`DEVICES`, `logical_w`, `logical_h`) | `src/trueskate_ai/rl/device_worker.py` |
