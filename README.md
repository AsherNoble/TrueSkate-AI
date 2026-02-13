# TrueSkate-AI

This project aims to train an ML to play the mobile game TrueSkate.

## Architecture Plan

1. **Labeling module** (evolved from `synthesis.py`): takes video clips → outputs per-frame `(touch1_active, touch1_x, touch1_y, touch2_active, touch2_x, touch2_y)`
2. **Data pipeline**: sliding windows over clips → each sample is `(N frames + their touch states)` as input, `(next M frames' touch states)` as target
3. **Model input**: grayscale downscaled frames + corresponding touch states for context window
4. **Model output**: predicted touch states for next M frames — continuous values, no categorical labels
5. **Inference loop**: predict short step → execute on device via Appium → capture new frame → slide window forward → repeat

## Current Status
- ✅ iPhone control via Appium + WebDriverAgent
- 🚧 Model training pipeline (in progress)
- 🚧 Data collection scripts (in progress)

## Notes
Early experimental code - messy but functional.