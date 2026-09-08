"""Behavioral-cloning (sequence-leap) package.

The VPT-style second stage: Model 2, a vision-grounded gesture-sequence policy
(n frames + m past strokes -> next strokes), trained on Asher's expert play
auto-labeled by Model 1 (the trace extractor). See
`research/ARCHIVE.md (ARCH-001)` and the plan at
`research/STATUS.md`.

This package is PURE torch/numpy — no Appium/pyobjc/device deps — so it
containerizes and runs unchanged on any cloud GPU provider (Modal first).
"""
