"""Compatibility launcher; use scripts/model1/estimate_model1_scaling_cost.py."""
from pathlib import Path
import sys

_target = Path(__file__).resolve().parents[2] / 'scripts/model1/estimate_model1_scaling_cost.py'
__file__ = str(_target)
exec(compile(_target.read_bytes(), str(_target), "exec"), globals())
