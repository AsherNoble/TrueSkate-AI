# Gesture Reference

Single source of truth for gesture structure, coordinate conventions, device geometry, and the trick library schema. All source files, scripts, and AI assistants working in this repo should defer to this document for anything touch/gesture/coordinate related rather than repeating or deriving these facts inline.

---

## Terminology

These definitions bind code, docs and messages. Records written before 2026-10-16 may use
the old words in [Stop using](#stop-using); read them through that list.
`tests/test_terminology.py` checks the current docs and the trick, kickflip, Model 2 and
HID pointer code against it.

### Positions and paths

| Term | Meaning |
|---|---|
| **normalised coordinates** | `[x, y]` as fractions of screen width and height, each in `[0, 1]` (see [Coordinate System](#coordinate-system)). |
| **point** | One normalised position on a gesture's path. |
| **path** | A gesture's points, in order. Timed schemas give each point a time in ms from touch-down. |
| **segment** | The straight line between two consecutive points of a path, with its duration. A **moving segment** joins two different points; a **pause** repeats a point, so the finger stays down without moving. |
| **bend** | A point where two moving segments meet at an angle. |

### One gesture

| Term | Meaning |
|---|---|
| **touch** | A finger on the screen. |
| **gesture** | One touch from touch-down to lift: its path and its timing. The atomic unit of touch input. A lift always ends a gesture. |
| **touch-down** | The instant the finger reaches the screen; time 0 of the path. |
| **lift** | The instant the finger leaves the screen, at the end of the last segment. |
| **duration** | Lift time minus touch-down time. |
| **speed profile** | How speed varies along the path: **constant speed**, or the profile set by `easing_power`. "Linear" never describes speed. |

### Kinds of gesture

The kind depends only on the path, never on speed.

| Term | Meaning |
|---|---|
| **linear gesture** | Exactly one segment, and it moves. Linear Model 1 reads these. |
| **curved gesture** | Two or more segments; any of them may be a pause. Still one gesture: its straight parts are segments, not linear gestures. |
| **stationary gesture** | A gesture that never moves: one point, or only pauses. A **tap** lasts under 150 ms; a **hold** lasts 150 ms or more. |
| **simultaneous gestures** | Two or more gestures whose touch-down-to-lift spans overlap. They are sent together in one request. Examples: a **die-five marker** (five simultaneous taps on a die's five spots) and a spin flick (a flick plus a hold on the rotate button by a second finger). |

### Several gestures

| Term | Meaning |
|---|---|
| **gesture sequence** | Sequential gestures in order, with the gap before each one. |
| **gap** | Time from one gesture's lift to the next gesture's touch-down. A negative gap means the two overlap, which makes them simultaneous. |
| **gesture recipe** | A gesture sequence stored as `{"gestures": [...], "delays": [...]}`; each `delays` entry is a gap ([Gesture Recipe](#gesture-recipe)). Trick libraries and kickflip candidates use it. |
| **gesture parameters** | The shared flat float vector used by collection, recipe replay and Model 2 (17 elements for two gestures without spin). Decoded into a gesture recipe for execution. |
| **gesture family** | The sampler category of a training gesture: push, pop, flick or catch (shaped like that part of a kickflip), or random. |
| **trick** | A True Skate move made by a gesture sequence. A kickflip is push, pop, flick, catch. |
| **landed** | The operator's verdict that the intended trick completed and the board landed wheels-down. |
| **trick library** | A JSON file of `median_gestures` and `best_gestures` recipes for a named trick, produced by `scripts/data/build_trick_library.py`. |

### Execution, video and labels

| Term | Meaning |
|---|---|
| **request** | One call to WDA that performs gestures: a W3C `perform()` or a `perform_trick_gestures` call. One request per sequential gesture ([hard rule](#one-request-per-gesture-hard-rule)). |
| **recording** | One XCTest screen recording, one minute long. |
| **clip** | A window of a recording used for training: 120 frames (2 s at 60 fps) for curved and trick Model 1, 32 frames for linear Model 1. |
| **label** | The commanded gestures attached to a clip, in clip time. |
| **trail** | The line True Skate draws under a moving finger. |
| **visible onset** | The first frame that shows a gesture's trail. Never a synonym for touch-down. |
| **marker** | A calibration gesture with a known place and time, used to fit the video clock. An **anchor** is a marker used in that fit. |
| **waypoint** | True Skate's saved spot in a park, where a reset returns the board. Never a point on a path. |

### Measuring gesture error

| Term | Meaning |
|---|---|
| **position error** | Euclidean distance between two positions in normalised coordinates. "2% of the screen" is 0.02: 8.3 pt across or 17.9 pt along a 414 × 896 XR screen. |
| **worst-point error** | The largest position error along a gesture, comparing points at the same fraction of its duration. |
| **timing error** | Predicted minus labelled touch-down; likewise for lift. |
| **passes at X** | Both timing errors within 2 frames (33 ms) and the worst-point error at most X. |

### Stop using

| Don't write | Write |
|---|---|
| stroke | gesture |
| piece, interval or movement (of a path) | segment |
| swipe, drag | gesture; linear gesture when it is straight |
| trace | trail |
| waypoint (a point on a path) | point |
| segment (a recording) | recording |
| press, release, touch-up, onset (of a touch) | touch-down, lift |
| linear (of speed) | constant speed |
| run (the gestures of one clip) | gesture sequence |
| kink | bend |
| action (a touch) | gesture. "Action" stays for RL policy output and external API names such as W3C actions. |

### Legacy names

These predate the vocabulary and stay, so saved data, checkpoints and cited paths keep
working. Each means what the last column says.

| Name | Where | Means |
|---|---|---|
| `strokes`, `stroke`, `GESTURE_EVENT = "stroke"`, `stroke_timing`, `stroke_visible_onsets_s`, `stroke_visible_lags_s`, `stroke_visible_lag_s`, `sent_strokes`, `planned_strokes`, `strokes_sent`, `strokes_dropped`, `stroke_families`, `stroke_ms`, `labelled_strokes`, `predicted_strokes`, `extra_strokes`, `pieces` | `model1_trick_v1` plans, executions, admitted clips, manifests, summaries and results | gestures; one gesture (index, or execution event kind); gesture timing; segments |
| `run`, `runs`, `sample_run` | `model1_trick_v1` | one clip's gesture sequence |
| `strokes`; `stroke_dim`, `stroke_embed`, `stroke_pos`, `stroke_head` | Model 2 `clip.json`; Model 2 checkpoints saved before 2026-10-16, loaded through `upgrade_legacy_checkpoint` | gestures; the gesture size and the policy's gesture layers |
| `strokes` | HID pointer schedule records | the schedule's gestures |
| `onset_*`, `release_*`; the `contact` signal | trick Model 1 results; its per-frame outputs | touch-down; lift; touching |
| `contacts` | kickflip candidates | the kickflip's gestures |
| `waypoints` | trick libraries, SLS `meta.json`, WDA's `perform_trick_gestures` payload | points |
| `delays` | gesture recipes | gaps |
| `press`, `press_slot`, `release_us` | HID pointer schedules | touch-down |
| `segment` in `collect_sls_xctest.py`, `collection/xctest_capture.py` and the linear collector scripts | linear collection code | recording |
| `TimedWaypoints`, `build_curved_drag`, `curved_drag_with_spin_hold`, `trace` in linear Model 1 code | `sim/`, `model1/linear`, `model1/temporal` | timed points; a gesture; trail |
| File names such as `extract_demo_strokes.py`, `align_xctest_traces.py`, `train_trace_extractor.py` | `scripts/` | kept so records' links work |

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
X:         [X_BOUND_MIN, X_BOUND_MAX]   valid RL gesture area
Y:         [Y_BOUND_MIN, Y_BOUND_MAX]   valid RL gesture area

X_BOUND_MIN = 0.12   # coarse left play-area bound; the control endpoint map adds precise exclusions
X_BOUND_MAX = 1.0    # right screen edge
Y_BOUND_MIN = 0.12   # coarse top play-area bound
Y_BOUND_MAX = 0.88   # coarse bottom play-area bound (avoids home indicator and game menu)
```

Defined in `src/trueskate_ai/sim/gestures.py`; imported by both RL pipelines (`action_param.py`, `trick_conditioned_action.py`). These bounds apply only to RL gesture parameters — utility gestures (`skip_loading_screen`, `reset_position`, `execute_static_push`) use their own fixed normalised positions and are not constrained by these values.

Random corpus gestures also use the versioned XR control-endpoint profile in
`src/trueskate_ai/data/control_hitboxes.py`. Its 16-logical-point safety regions
apply to each gesture's first and final points. A moving gesture may still cross a
control region between those endpoints; the exact speed/path conditions that can
activate a crossed control remain unresolved. Stationary gestures (taps and holds)
must remain outside because they never move. Deliberate, labelled spin-button holds
are exempt.
The bottom menu is transient but is conservatively blocked at all times in the
current profile.

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

A list of `[x, y]` normalised points. Minimum 2; most trick gestures use 3 (start → control → end). The path runs along straight segments between consecutive points, so 3 or more points make a curved gesture.

### `duration`

Total gesture time in seconds. Typical range: `0.03`–`0.80` s.

- Shorter = faster, more explosive motion.
- Longer = slower, more deliberate motion.

### `easing_power`

Controls the velocity profile across the gesture path. Typical range: `[0.3, 3.0]`.

| Value | Profile | Good for |
|---|---|---|
| `< 1.0` | Decelerating — fast start, slow end | Pop / downward motions |
| `= 1.0` | Constant speed | Neutral |
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

Ordered list of gesture objects. Most tricks use 2–3 gestures. Slots are scheduled by absolute start time (from the delay chain) and executed via `execute_n_slot_gestures()` (`sim/touch_actions.py`): short or negative inter-gesture gaps are bundled into a single overlapping W3C `perform()` payload; larger positive gaps fall back to separate, delay-compensated `perform()` calls.

### `delays`

List of N−1 floats for N gestures. `delays[i]` is the gap: the wait from the lift of gesture `i` to the touch-down of gesture `i+1`.

```
delays: [0.12]         → 2-gesture recipe: wait 0.12 s after gesture 0 then fire gesture 1
delays: [0.12, 0.15]   → 3-gesture recipe: 0.12 s gap then 0.15 s gap
```

Negative delays (inter-gesture overlap) are supported: `execute_n_slot_gestures()` reorders slots by absolute start time and bundles them into one combined W3C `perform()`, so a delay sweeping through the crossover point stays continuous.

### `spin` (optional)

Spin-family tricks (BIG SPIN, BIG FLIP, GAZELLE FLIP, …) need True Skate's rotate button, which single-finger gestures can't express. When a curriculum sets `"use_spin": true`, the CMA-ES action vector gains a trailing 3-param spin block and recipes carry a decoded `spin` object:

```json
{
  "gestures": [ ... ],
  "delays": [0.12, 0.15],
  "spin": { "enabled": true, "t_start": 0.20, "t_end": 0.70 }
}
```

- `enabled` — whether the rotate button fires this gesture (CMA-ES param: a gate thresholded at `>= 0`).
- `t_start` / `t_end` — fractions `[0, 1]` of the schedule's total duration; the button is held DOWN by an extra finger from `t_start·total` to `t_end·total`, scheduled **inside the same single W3C `perform()`** as the other gestures (a hold, not a tap — an earlier concurrent-tap-from-a-background-thread design cancelled the in-flight gesture on the shared WDA session).

The spin block is appended **after** the delays, so the vector length becomes `8N + (N−1) + 3 = 9N + 2` (vs `9N − 1` without spin). The two length classes are disjoint mod 9, so `infer_layout(len)` recovers `(N, use_spin)` from a vector alone — no separate flag is stored in logs. The rotate button's normalised position is per-device (`spin_button_xy` in `DEVICES`, default `DEFAULT_SPIN_BUTTON_XY = (0.0604, 0.4040)` in `sim/gestures.py`), left of the gesture area (`X_BOUND_MIN = 0.12`) so gestures never hit it. Recipes without a `spin` key hold only single-finger gestures (all legacy libraries).

### `spin_flick` (SLS corpus sample kind)

The SLS collector's Model-1-trainable spin form (`gesture_sampling.sample_spin_flick`, drawn inside the `--spin-frac` slice alongside nslot-`spin`): two simultaneous gestures, a single-finger curved flick plus a hold on the spin button by a second finger — executed via `curved_drag_with_spin_hold()` (`sim/touch_actions.py`) in one W3C payload, **no push**, so every touch in the frames is label-accounted. Unlike the recipe spin block's schedule-fraction `t_start`/`t_end`, the hold window is stored in **absolute seconds from the flick touch-down** and OUTLASTS the flick (real spin play holds the button through the rotation). Sample `meta.json` fields:

- flick fields as usual: `waypoints` (legacy key: always 3 points), `duration`, `easing_power`
- `spin_active: true`
- `spin_hold_start_s` / `spin_hold_end_s` — hold window, seconds from payload start (start ≥ 0.12 s after the flick's touch-down: near-simultaneous multi-finger touch-downs read as the park-editor camera gesture)
- `payload_total_s` — full payload length (= hold end). The end-relative `frame_times` anchor: the gesture call ends when the button's hold lifts, not when the flick does
- `spin_button_xy` — the normalised button coord that actually fired (stamped by the collector)

Params-vector kinds (`spin`, and `nslot`/`recipe` carrying a spin block) emit the same decoded `spin_active`/`spin_hold_start_s`/`spin_hold_end_s`/`payload_total_s` meta fields (nominal, pre-stagger schedule) so no consumer re-derives from the raw trailing block.

### Basic Model 1 hold experiment

The additive basic Model 1 experiment intentionally does **not** use the general gesture schema. Its only learning target is a single stationary hold:

```json
{"point": [0.42, 0.65], "hold_duration_s": 0.80}
```

`point` is normalized screen position and `hold_duration_s` is constrained to `[0.30, 1.50]`. Its corpus loader admits only `gesture_distribution: "hold"`, one non-spin finger, accepted tap-calibration provenance, and no `.menu` marker. Known-position taps can be recorded as timing controls but are never model examples. The regression interface returns `{x, y, dur}` in those native units.

### Basic Model 1 linear experiment

The linear collection target is one constant-speed linear gesture (two points) with
normalised start/end coordinates and duration. Each one-minute recording is
bracketed by two 50 ms timing-control taps at the exact screen centre
`[0.5, 0.5]`. Both controls are separate WDA requests and carry
`calibration_control: true` plus `calibration_role: "start"` or `"end"`.
They remain in the raw manifest for timing provenance; the aligner never emits
them as Model 1 samples. Their first visible frames define an affine mapping
from the instrumented WDA `submitted_to_ios.monotonic_s` clock to video time.
No unlabelled reset may occur during the recording; the canonical wrapper resets
and settles the board before recording begins.

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
   The recipe fires after PUSH_PRE_DELAY total push time, via
   execute_n_slot_gestures() (sim/touch_actions.py) — the SAME canonical path
   the CMA-ES eval path uses. It schedules N slots by absolute start time and
   picks sequential vs a single combined W3C perform() automatically: overlapping
   (negative) delays, all-short delays, or an active spin force one bundled
   perform(). Spin is a held finger inside that same payload.

3. RESET
   reset_position() taps the reset button at (0.5·device_w, 0.0558·device_h)
   to return the board to its starting position.
```

### One request per gesture (hard rule)

Send each **sequential** gesture (touch-down → moves → lift) as its own request.
Never bundle sequential, non-overlapping gestures into one W3C `perform()` or one
`perform_trick_gestures` call. True Skate then joins them into one continuous
chain: the end of gesture *n* is drawn connected to the start of gesture *n + 1*.

- Several `XCPointerEventPath`s in one `XCSynthesizedEventRecord` run as parallel
  finger tracks from t = 0, even when a later path's touch-down is scheduled far
  ahead. This is also how WDA translates a multi-pointer W3C request.
- One path that hover-moves between a lift and the next touch-down still emits real
  touch events, which the game draws as a phantom gesture.
- Found in May 2026 (archived `wda_trick_gestures_plan.md`, tag
  `archive/research-pre-cleanup`), seen again as "joined gestures" in the
  September bundled timing run04b, and **repeated on 2026-10-03**: 9/9 bundled
  demo replays were conjoined at every point spacing
  ([CURVE-AUDIT-20261003](research/experiments/CURVE-AUDIT-20261003.md)).
- Genuinely **simultaneous** gestures are a different case. For example, the
  spin button is held by a second finger while the flick runs. Those still
  share one payload. The `execute_n_slot_gestures` combined path that bundles
  sequential slots for short or negative delays predates this rule and is suspect.
- Cost: each request returns about 0.27–0.33 s after its gesture ends (measured
  2026-10-03 on `ae50404a`). Sequential gestures sent this way cannot start closer
  together than that; tighter timing needs on-device scheduling of separate records.

Push constants are defined in `src/trueskate_ai/sim/gestures.py`. Both replay (`src/trueskate_ai/sim/gesture_recipe.py::execute_gesture_recipe`) and the CMA-ES eval path (`src/trueskate_ai/sim/gesture_params.py::execute_gesture_params`) call the **same** canonical pair: `execute_static_push` (`sim/gestures.py`) then `execute_n_slot_gestures` (`sim/touch_actions.py`). The replay path additionally resets the board afterward.

---

## Code Cross-References

| Concept | Location |
|---|---|
| `scale_to_device()` | `src/trueskate_ai/sim/gestures.py` |
| `execute_static_push()`, `PUSH_*` constants | `src/trueskate_ai/sim/gestures.py` |
| `execute_gesture_recipe()` — delegates to push + `execute_n_slot_gestures` | `src/trueskate_ai/sim/gesture_recipe.py` |
| `execute_n_slot_gestures()` (N-slot scheduler: sequential / combined / spin HOLD) | `src/trueskate_ai/sim/touch_actions.py` |
| `build_curved_drag()`, `make_touch_pointer()`, `perform_pointer_actions()` | `src/trueskate_ai/sim/touch_actions.py` |
| `curved_drag_with_spin_hold()` (spin_flick: a flick plus a hold on the spin button, one payload, no push) | `src/trueskate_ai/sim/touch_actions.py` |
| Shared gesture parameter bounds, decode, execute | `src/trueskate_ai/sim/gesture_params.py` |
| Retired PPO/CMA-ES implementations | `research/ARCHIVE.md` (ARCH-002) |
| Library recipe replay | `scripts/inspect/execute_trick.py` |
| Build trick library from JSONL log | `scripts/data/build_trick_library.py` |
| Device configs (`DEVICES`, `logical_w`, `logical_h`) | `src/trueskate_ai/sim/device.py` |

### Timed-point execution audit

The curved Model 1 collection schema `model1_curved_v1` uses a stricter subset:
2–11 timed points (1–10 segments), each segment at least 100 ms, total
300–1200 ms, one gesture. Exact integer-coordinate commands define
the labels; ideal arc geometry is separate. See the
[current protocol](research/protocols/model1_curved_collection_v1.md).
The broader `TimedWaypoints` interface below remains available to diagnostics.

`sim/timed_waypoints.py::TimedWaypoints` is an additive interface: 2–15 normalized
points with strictly increasing integer millisecond timestamps starting at
zero. Each segment moves at constant speed; a repeated point is a pause.
Existing linear, recipe and cubic interfaces are unchanged. Compilation truncates
normalized coordinates to integer device points and validates **every quantized
segment** against the expanded control map. One request contains an initial
zero-duration positioning move, one pointer-down (touch-down), one positive-duration
move per segment and one pointer-up (lift). No extra subdivisions or gaps are inserted.
The curved execution audit freezes 5/8/11/15-point paths and their exact payloads;
its overlay follows those quantized positions and times, never the observed trail.

### Trick-speed gestures (`model1_trick_v1`)

The [trick-speed batch](research/protocols/trick_speed_batch.md) uses a separate
schema for fast gestures like real play. Its saved keys keep the old word, as in `strokes`
([Legacy names](#legacy-names)).
- **Shape and timing:** 1–4 segments, each at least 40 ms, 50–600 ms in all. A
  repeated point is a pause, sent as a zero-distance move.
- **Safety:** points stay inside the usable bounds, and the whole quantized path
  avoids the expanded control map.
- **Sending:** each gesture is one request to WDA's direct
  `/wda/perform_trick_gestures` endpoint, as in the kickflip runs.
  - A clip's gesture sequence (1–4 gestures) sends each as soon as the previous
    request returns, plus 0–150 ms. Gestures are never bundled into one request.
- **Labels and timing:** labels are the commands sent. Direct requests carry no
  WDA timing record, so gesture times are the rig-clock call start plus the
  recording's median rig-to-WDA offset. Admission then re-times each gesture to
  its own trail.
- Curved v1's contract is unchanged.

## Experimental cubic in time (`cubic_in_time_v1`)

`sim/cubic_curve.py` introduces an additive nine-number description:
`{"schema":"cubic_in_time_v1","coefficients":[[x0,y0],[x1,y1],[x2,y2],[x3,y3]],"duration_s":0.6}`.
Coefficients use the Bernstein basis at normalized time; there is no easing.
Eight evenly timed positions fit with exact endpoints. One cubic cannot represent
all trajectories. Fitting does not certify safety: compilation rejects unsafe
coefficient hulls and every unsafe quantized segment against expanded controls.

The compiler accepts maximum segment duration or explicit N with a minimum
segment-duration floor. It rounds total milliseconds once, rounds cumulative
boundaries, evaluates at those actual times and quantizes to device logical
points. `build_curved_drag(segment_durations_ms=...)` sends one continuous touch.
`touch_labels.cubic_command_label` shares its exact boundaries and quantized
points, avoiding label/executor truncation mismatch. Legacy defaults and labels
are unchanged. Original cubic and executed command remain separate error sources.

This path is diagnostic, not a curved training schema or model certification.
[CURVE-EXEC-20261002](research/experiments/CURVE-EXEC-20261002.md) tests temporal
spacing and visibility before any model recovery/generation tranche. Its primary
0.01 normalized Euclidean metric is anisotropic: 4.14 horizontal or 8.96 vertical
points on an XR. A pilot measurement floor determines whether it is measurable.
