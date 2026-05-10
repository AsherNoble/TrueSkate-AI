# Copilot instructions for TrueSkate-AI

## Quick commands
- Create & activate venv:
  python -m venv .venv && source .venv/bin/activate
- Install runtime deps:
  pip install -r requirements.txt
  (If requirements.txt not present, use: pip install opencv-python numpy torch torchvision scipy pillow appium-python-client matplotlib requests cma pyobjc)
- Launch core services (local Appium + WDA):
  python scripts/launch_services.py (requires Appium & Xcode WDA)
- Train / run CMA-ES entrypoint:
  python scripts/train/train_cmaes.py
- Build trick library / replay:
  python scripts/data/build_trick_library.py

Tests & linting
- No automated test suite detected. To run a single test if tests are added use:
  pytest path/to/test_file.py::test_name
- No lint config detected; common lint command if added:
  flake8 src/ or black --check .

## High-level architecture (big picture)
- src/trueskate_ai/rl: gesture parameterization (17-dim CMA-ES vector + PPO 42-dim), CMA-ES optimizer, reward shaping.
- src/trueskate_ai/sim: device interaction and execution (curved_drag W3C Actions via Appium + custom WDA endpoint), Apple Vision OCR-based trick detection (fuzzy match against KNOWN_TRICKS), and utilities to replay gesture recipes.
- src/trueskate_ai/vision & labeling: legacy CV / dataset code (pre-RL) kept for reference.
- scripts/: entrypoints (train, build library, launch services). Experiments journal in experiments/.

Key runtime constraints:
- Requires physical iPhones, one Appium instance per device (localhost:4723–4725) and one WDA instance per device (localhost:8100–8102). See DEVICES in src/trueskate_ai/rl/device_worker.py for the full per-device port mapping.
- Device UDIDs are read from .env (copy .env.example).

## Key conventions and patterns
- **Gesture terminology**: use "gesture" for a touch path, "gesture recipe" for the structured dict, "gesture parameters" for the CMA-ES flat vector. Do not use "action" or "swipe" for these concepts. See GESTURES.md.
- **Coordinates**: all gesture coordinates are normalised [0, 1]. Conversion to device logical points at execution: `device_x = norm_x * device_w`, `device_y = norm_y * device_h` via `scale_to_device()` in `src/trueskate_ai/rl/gestures.py`. No y_offset needed — all supported devices share the 19.5:9 aspect ratio. Per-device logical dimensions are in `DEVICES` in `device_worker.py`; note that iPhone 11 runs Display Zoom (375 × 812 pts) rather than the spec-sheet standard of 414 × 896.
- **Y_BOUND_MIN = 0.12 / Y_BOUND_MAX = 0.88**: valid RL gesture y range; defined in `src/trueskate_ai/rl/gestures.py`. See GESTURES.md.
- **Curved gestures required**: do not replace with straight swipes — curved multi-waypoint drags are essential for trick physics.
- **Gesture execution**: trick gestures fire as sequential calls to the custom WDA endpoint `/wda/perform_trick_gestures`, bypassing Appium. Push still uses Appium ActionChains (separate perform() call).
- OCR pipeline: Apple Vision framework (pyobjc) → fuzzy match to KNOWN_TRICKS.
- Safety clamps: CMA-ES clamps params to PARAM_BOUNDS to avoid NaNs that crash Appium.
- Debug output: write to tmp/ (gitignored). Model files (*.pth) live in notebooks/models/ (gitignored).
- Use absolute paths (no ~) in tooling and WDA path references.

## Existing assistant files to consult
- CLAUDE.md — repo-specific guidelines and architecture. Consult before proposing changes to core flows.
- GESTURES.md — authoritative reference for gesture terminology, coordinate system, recipe schema, device geometry, and execution flow. Consult before writing any touch/gesture/coordinate code.

## When editing or extending
- Preserve curved gesture primitives; changes to gesture parameterization (action_param.py) or touch execution (touch_actions.py) must be validated on-device.
- All new coordinate values must be normalised [0, 1]. Never hardcode logical pixel values for gestures.
- If adding tests or linters, include commands above and add a requirements.txt for reproducible installs.

---
Update this file if scripts/ or service endpoints change.
