"""Compatibility launcher; use scripts/model1/train_trace_extractor_modal.py."""
from pathlib import Path
import sys

_target = Path(__file__).resolve().parents[2] / 'scripts/model1/train_trace_extractor_modal.py'
__file__ = str(_target)
exec(compile(_target.read_bytes(), str(_target), "exec"), globals())
