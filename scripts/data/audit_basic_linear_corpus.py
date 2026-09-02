#!/usr/bin/env python3
"""Audit a strict calibrated Model-1 linear-drag corpus without Modal access.

Example:
    PYTHONPATH=src python scripts/data/audit_basic_linear_corpus.py \
      --data data/basic_linear_stage1_20260831 --require-device iPhone_XR \
      --require-device iPhone_XR2 --require-park 'The Workshop' --min-per-device 1000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from trueskate_ai.vision.basic_linear_audit import audit_basic_linear_corpus  # noqa: E402


def _gate_errors(report: dict, *, devices: list[str], park: str | None,
                 device_parks: list[str], min_per_device: int,
                 require_unique: bool) -> list[str]:
    errors: list[str] = []
    counts = report["provenance"]["per_device"]
    for device in devices:
        if counts.get(device, 0) < min_per_device:
            errors.append(f"{device}: {counts.get(device, 0)} strict device-provenanced clips < {min_per_device}")
    if park is not None:
        other = {name: count for name, count in report["provenance"]["per_park"].items() if name != park}
        if other:
            errors.append(f"accepted clips outside required park {park!r}: {other}")
    for requirement in device_parks:
        if "=" not in requirement:
            errors.append(f"invalid --require-device-park {requirement!r}; expected DEVICE=PARK")
            continue
        device, expected_park = requirement.split("=", 1)
        device = device.strip()
        expected_park = expected_park.strip()
        observed = set(report["provenance"]["parks_by_device"].get(device, []))
        if not observed:
            errors.append(f"{device}: no accepted clips for required park {expected_park!r}")
        elif observed != {expected_park}:
            errors.append(f"{device}: accepted parks {sorted(observed)!r}, expected only {expected_park!r}")
    if report["provenance"]["missing_device_provenance"]:
        errors.append(f"{report['provenance']['missing_device_provenance']} accepted clips lack device provenance")
    if require_unique and report["duplicates"]["duplicate_groups"]:
        errors.append(f"{len(report['duplicates']['duplicate_groups'])} exact-command duplicate groups")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--position-bins", type=int, default=4)
    parser.add_argument("--numeric-bins", type=int, default=8)
    parser.add_argument("--sparse-cell-max-count", type=int, default=5)
    parser.add_argument("--require-device", action="append", default=[])
    parser.add_argument("--require-park")
    parser.add_argument("--require-device-park", action="append", default=[], metavar="DEVICE=PARK",
                        help="require all accepted clips from DEVICE to have PARK provenance")
    parser.add_argument("--min-per-device", type=int, default=0)
    parser.add_argument("--require-unique-commands", action="store_true")
    args = parser.parse_args()
    if not args.data.is_dir():
        parser.error(f"corpus directory does not exist: {args.data}")
    report = audit_basic_linear_corpus(args.data, position_bins=args.position_bins,
                                       numeric_bins=args.numeric_bins,
                                       sparse_cell_max_count=args.sparse_cell_max_count)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    errors = _gate_errors(report, devices=args.require_device, park=args.require_park,
                          device_parks=args.require_device_park,
                          min_per_device=args.min_per_device,
                          require_unique=args.require_unique_commands)
    if errors:
        print("AUDIT FAILED: " + "; ".join(errors), file=sys.stderr)
        raise SystemExit(2)
    print("AUDIT PASSED", file=sys.stderr)


if __name__ == "__main__":
    main()
