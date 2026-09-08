"""Compatibility launcher; use scripts/model2/run_sequence_policy.py."""
from pathlib import Path
import sys

_target = Path(__file__).resolve().parents[2] / 'scripts/model2/run_sequence_policy.py'
__file__ = str(_target)
exec(compile(_target.read_bytes(), str(_target), "exec"), globals())
