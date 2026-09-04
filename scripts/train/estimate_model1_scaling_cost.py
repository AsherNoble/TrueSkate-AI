#!/usr/bin/env python3
"""Estimate Modal spend for the linear Model 1 scaling-law study."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from trueskate_ai.vision.model1_scaling import (  # noqa: E402
    DEFAULT_LINEAR_RUNGS, estimate_modal_rungs, scaling_status,
)


def _sizes(value: str) -> list[int]:
    try:
        return [int(item) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("sizes must be comma-separated integers") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=_sizes, default=list(DEFAULT_LINEAR_RUNGS))
    parser.add_argument("--base-size", type=int, default=13_100)
    parser.add_argument("--base-run-hours", type=float, default=8.42,
                        help="Observed hours for one 40-epoch seed at base-size.")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--gpu", default="L4")
    parser.add_argument("--memory-gib", type=float, default=64.0)
    parser.add_argument("--cpu-cores", type=float, default=.125)
    parser.add_argument("--approval-contingency", type=float, default=1.5,
                        help="Budget multiplier for shard staging, train-metric passes, and retries.")
    parser.add_argument("--observations-json", type=Path,
                        help="Optional list of training_samples/late_validation_recovery rows.")
    args = parser.parse_args()
    estimate = estimate_modal_rungs(
        args.sizes, base_size=args.base_size, base_run_hours=args.base_run_hours,
        seeds=args.seeds, gpu=args.gpu, memory_gib=args.memory_gib,
        cpu_cores=args.cpu_cores,
    )
    future = [row for row in estimate["rungs"] if row["training_samples"] > args.base_size]
    estimate["minimum_additional_two_doublings_usd"] = sum(
        row["estimated_cost_usd"] for row in future[:2]
    )
    estimate["minimum_additional_two_doublings_gpu_only_usd"] = sum(
        row["gpu_only_cost_usd"] for row in future[:2]
    )
    if args.approval_contingency < 1.0:
        parser.error("approval-contingency must be at least 1.0")
    estimate["approval_contingency"] = args.approval_contingency
    estimate["recommended_two_doubling_approval_ceiling_usd"] = (
        estimate["minimum_additional_two_doublings_usd"] * args.approval_contingency
    )
    if args.observations_json:
        observations = json.loads(args.observations_json.read_text())
        if not isinstance(observations, list):
            parser.error("observations JSON must contain a list")
        estimate["scaling_status"] = scaling_status(observations)
    print(json.dumps(estimate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
