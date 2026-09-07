"""Compatibility launcher; use scripts/collection/align_xctest_traces.py."""
from pathlib import Path
import sys

_target = Path(__file__).resolve().parents[2] / 'scripts/collection/align_xctest_traces.py'
__file__ = str(_target)
exec(compile(_target.read_bytes(), str(_target), "exec"), globals())
