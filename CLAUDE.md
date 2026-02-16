# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TrueSkate-AI trains a model to play the mobile game True Skate. The pipeline: capture screen recordings → extract per-frame touch labels via computer vision → train a model to predict touch sequences → execute predictions on-device via Appium.

## Setup

No `pyproject.toml` or `setup.py` exists yet. Install dependencies manually:

```bash
python -m venv .venv && source .venv/bin/activate
pip install opencv-python numpy torch torchvision scipy pillow appium-python-client matplotlib requests
```

External tools: Appium (npm), WebDriverAgent (Xcode), ffmpeg, libimobiledevice.

## Architecture

```
src/trueskate_ai/
├── labeling/       # CV pipeline: video → per-frame touch labels (the core implemented module)
├── vision/         # PyTorch Datasets for training (TouchDataset, VideoDataset)
├── utils/          # TrajectorySpline for smooth path fitting, data_loader helper
├── eval/           # (placeholder)
├── sim/            # (placeholder)
└── trajectories/   # (placeholder)
scripts/            # Entry points: launch_services, extract_frames, run_model, etc.
notebooks/          # Experiments, training data, reference images
tmp/                # Debug output (gitignored)
```

### Labeling Pipeline (main implemented module)

`trace_extractor.py` → `video_labeler.py` → `visualize.py`

1. **TraceExtractor** processes individual BGR frames: HSV filtering for orange traces → morphological cleanup → connected component blob detection → solidity + temporal new-pixel-ratio classification → hotspot peak localization → nearest-neighbor touch assignment for temporal consistency. Also detects spin button state via Sobel gradient magnitude on the button icon region.
2. **VideoLabeler** wraps TraceExtractor for full video processing. CLI entry point: `python -m trueskate_ai.labeling.video_labeler <video.mp4>`. Outputs CSV and optionally `.pt` tensors or debug video.
3. **LabelVisualizer** creates annotated debug videos and frame strips.

### Key Data Formats

- **TouchState**: `(frame_number, touch1_active, touch1_x, touch1_y, touch2_active, touch2_x, touch2_y, spin_control_active)` — coordinates normalized to [0, 1]
- **CSV labels**: One row per frame with the TouchState fields
- **Tensor output**: Shape `(N, 7)` float32, excludes frame_number

### Resolution Handling

All TraceExtractor parameters are defined at a reference resolution (1170×2532 for main frame, 750×1624 for spin button region) and scale proportionally to actual frame dimensions.

## Device Configuration

- WebDriverAgent project: `~/Projects/WebDriverAgent`
- Appium: localhost:4723, WDA: localhost:8100
- Training data source: `/Users/ashernoble/Projects/Training_Data/`

## Conventions

- Debug/temporary output goes in `tmp/` (gitignored)
- `.venv/` is the sole virtual environment
- `*.pth` model files are gitignored; stored in `notebooks/models/`
- Notebook outputs (PNGs, JSON) go in `notebooks/outputs/`
