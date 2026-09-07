"""Behavioral-cloning (sequence-leap) package.

The VPT-style second stage: Model 2, a vision-grounded gesture-sequence policy
(n frames + m past strokes -> next strokes), trained on Asher's expert play
auto-labeled by Model 1 (the trace extractor). See
`experiments/vision_sequence_leap_journal.md` and the plan at
`~/.claude/plans/start-investigating-how-we-jazzy-stroustrup.md`.

This package is PURE torch/numpy — no Appium/pyobjc/device deps — so it
containerizes and runs unchanged on any cloud GPU provider (Modal first).
"""
