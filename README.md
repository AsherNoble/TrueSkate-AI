# TrueSkate-AI

Behavioural cloning for the iOS game True Skate. Model 1 learns to recover
touch gestures from gameplay clips; Model 2 learns a gameplay policy from
expert recordings labelled by Model 1. Collection runs on physical iPhones
through Appium/WebDriverAgent and bounded XCTest screen recordings.

**Current work:** Model 1 linear gesture recovery and scaling. Hold, heatmap
and temporal variants remain runnable experiments. Model 2 is implemented
but unfinished; this repository does not claim an expert gameplay policy.
CMA-ES and PPO are retired; their history and useful results remain accessible.

## Start here

- [Current research status](research/STATUS.md) and [recent journal](research/JOURNAL.md)
- [Workflow commands](docs/WORKFLOWS.md)
- [Gesture and coordinate contract](GESTURES.md)
- [Rig deployment and rollback](DEPLOYMENT.md)
- [Research archive and recovery](research/ARCHIVE.md)
- [Rig reconciliation](research/RIG_RECONCILIATION.md)

## Setup

Python 3.11+, with `.venv` as the sole virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

For offline training/tests only: `python -m pip install -e '.[training,test]'`.
The `device` extra adds macOS Apple Vision bindings; `cloud` adds Modal.
Physical collection also needs Appium, WebDriverAgent, ffmpeg and
libimobiledevice. Copy `.env.example` to `.env` and configure device UDIDs;
never commit credentials. XR phones normally live on `training-server`.

## Layout

| Package | Responsibility |
|---|---|
| `collection` | XCTest capture, calibration, contamination guards and colour capture |
| `data` | Sampling, clip decoding, labels, corpus audits/manifests and shards |
| `model1` | Hold, heatmap, temporal and linear variants; scaling and certification |
| `model2` | Gesture tokens, stroke assembly, sequence policy and inference |
| `sim` | Device sessions, normalised gestures and touch execution |
| `vision` | Shared scene, board and OCR utilities |
| `monitoring`, `utils` | Monitoring and shared utilities |

Canonical entrypoints are in `scripts/collection`, `scripts/model1`,
`scripts/model2`, `scripts/data`, `scripts/inspect`, and `scripts/ops`.
Old training/cloud/collection script paths are compatibility launchers for
existing deployment callers. Datasets and checkpoints are external artifacts;
the archive index explains what Git preserves and what it does not.
