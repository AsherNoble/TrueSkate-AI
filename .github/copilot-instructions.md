# Copilot instructions for TrueSkate-AI

## Quick commands
- Create & activate venv:
  python -m venv .venv && source .venv/bin/activate
- Install runtime deps:
  pip install -r requirements.txt
  (If requirements.txt not present, use: pip install opencv-python numpy torch torchvision scipy pillow appium-python-client matplotlib requests cma pytesseract)
- Launch core services (local Appium + WDA):
  scripts/launch_services (run manually; requires Appium & Xcode WDA)
- Train / run CMA-ES entrypoint:
  python scripts/train_cmaes.py
- Build trick library / replay:
  python scripts/build_trick_library.py

Tests & linting
- No automated test suite detected. To run a single test if tests are added use:
  pytest path/to/test_file.py::test_name
- No lint config detected; common lint command if added:
  flake8 src/ or black --check .

## High-level architecture (big picture)
- src/trueskate_ai/rl: action parameterization (17-dim vector), CMA-ES optimizer, reward shaping.
- src/trueskate_ai/sim: device interaction and execution (curved_drag W3C Actions via Appium), OCR-based trick detection (pytesseract + fuzzy match against KNOWN_TRICKS), and utilities to replay "recipes".
- src/trueskate_ai/vision & labeling: legacy CV / dataset code (pre-RL) kept for reference.
- scripts/: entrypoints (train, build library, launch services). Experiments journal in experiments/.

Key runtime constraints captured here:
- Requires a physical iPhone, Appium (localhost:4723), WebDriverAgent (localhost:8100) and Tesseract OCR.
- Device UDID is read from .env (copy .env.example).

## Key conventions and patterns
- Gesture parameterization: 17 continuous params → two curved multi-waypoint gestures + inter-slot delay. Curved drags are essential; do not replace with straight swipes.
- Touch execution: gestures are performed as overlapping W3C Actions in a single perform() call (parallel execution), not sequential.
- OCR pipeline: screenshots -> 3× upscale -> grayscale -> threshold -> whitelist -> fuzzy match to KNOWN_TRICKS. Be cautious: pytesseract hallucinations occur; CLAUDE.md documents OCR caveats.
- Safety clamps: CMA-ES clamps params (y-bounds capped at 750) to avoid home indicator and prevent NaNs that crash Appium.
- Debug output: write to tmp/ (gitignored). Model files (*.pth) live in notebooks/models/ (gitignored).
- Use absolute paths (no ~) in tooling and WDA path references.

## Existing assistant files to consult
- CLAUDE.md — repo-specific guidelines and architecture (included critical details). Consult before proposing changes to core flows.

## When editing or extending
- Preserve curved gesture primitives; changes to action_param or touch_actions must be validated on-device.
- If adding tests or linters, include commands above and add a requirements.txt for reproducible installs.

---
Created from README.md and CLAUDE.md. Update this file if scripts/ or service endpoints change.
