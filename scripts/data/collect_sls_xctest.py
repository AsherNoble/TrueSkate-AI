"""Compatibility launcher; use scripts/collection/collect_sls_xctest.py."""
from pathlib import Path
import sys

_target = Path(__file__).resolve().parents[2] / 'scripts/collection/collect_sls_xctest.py'
__file__ = str(_target)
exec(compile(_target.read_bytes(), str(_target), "exec"), globals())
