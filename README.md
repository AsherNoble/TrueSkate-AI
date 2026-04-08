# TrueSkate-AI

Training an AI agent to perform skateboarding tricks in [True Skate](https://apps.apple.com/app/true-skate/id549105915), an iOS mobile game.

## Approach

The project started as behavioral cloning (supervised learning from gameplay recordings) but pivoted to **reinforcement learning** — the agent generates its own touch gestures via Appium on a physical iPhone, so there's no labeling bottleneck.

The current method uses **CMA-ES** (Covariance Matrix Adaptation Evolution Strategy) to optimize a 17-dimensional continuous parameter vector that encodes two curved multi-touch gestures. The agent attempts tricks, reads the result via OCR, and receives a shaped reward.

## Results So Far

The agent has landed **pop shove-its, varial flips, 360 flips, nightmare flips, and others** — confirming the pipeline works end-to-end. CMA-ES tends to converge on reliably landable medium-reward tricks rather than volatile high-reward ones, which is an active area of work.

## How It Works

1. **Action parameterization** — Two `curved_drag` gesture slots (3 waypoints + duration + easing each) plus inter-slot delay → 17 params total
2. **Touch execution** — Gestures run as overlapping W3C Actions via Appium/WebDriverAgent on a physical iPhone
3. **Trick detection** — Screenshot → OCR via pytesseract → fuzzy match against 248 known tricks
4. **Reward** — Tiered scoring based on trick name (360 flip = 1.0 down to 0.0 for irrelevant tricks)
5. **Optimization** — CMA-ES samples parameter vectors, evaluates them on-device, updates the search distribution

## Key Constraint

True Skate runs at **1× real-time** — no way to speed up the simulator. A GPU can't accelerate the live interaction loop. This makes data throughput (~15K steps/hour) the binding constraint, not compute.

## Project Structure
src/trueskate_ai/
├── rl/             # Action parameterization, CMA-ES optimizer, reward function
├── sim/            # Device control (touch_actions, trick_info_reader, known_tricks)
├── labeling/       # Legacy CV pipeline (pre-RL pivot)
├── vision/         # Legacy PyTorch datasets
└── utils/          # Trajectory splines, data loading
scripts/            # Entry points (train_cmaes, launch_services, build_trick_library, etc.)
experiments/        # Experiment journals and standalone experiments

## Requirements

- Python 3.11+ with venv
- Appium + WebDriverAgent (Xcode)
- Physical iPhone with True Skate installed
- pytesseract / Tesseract OCR

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install opencv-python numpy torch torchvision scipy pillow appium-python-client matplotlib requests cma pytesseract
```

Copy `.env.example` to `.env` and set your device UDID.