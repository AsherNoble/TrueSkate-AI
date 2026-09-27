#!/usr/bin/env python3
"""Build and verify immutable Model 1 scaling-study manifests."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from trueskate_ai.data.cohort_manifest import (  # noqa: E402
    assert_zero_cohort_leakage, read_manifest, write_manifest,
)
from trueskate_ai.data.timing_screen import TimingScreen  # noqa: E402
from trueskate_ai.model1.scaling import (  # noqa: E402
    DEFAULT_LINEAR_RUNGS, assert_deterministic_nesting,
    build_experiment_manifest, build_linear_cohort_manifest, build_park_mix_cohort,
    build_nested_subset_manifests,
)


def _sizes(value: str) -> list[int]:
    try:
        result = [int(item) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("sizes must be comma-separated integers") from exc
    if not result:
        raise argparse.ArgumentTypeError("at least one size is required")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    cohort = commands.add_parser("cohort", help="freeze a strict linear cohort")
    cohort.add_argument("--data", type=Path, required=True)
    cohort.add_argument("--corpus-root", type=Path,
                        help="Portable path base shared by every cohort (defaults to --data).")
    cohort.add_argument("--out", type=Path, required=True)
    cohort.add_argument("--name", required=True)
    cohort.add_argument("--role", choices=("training", "validation", "challenge", "certification"),
                        required=True)
    cohort.add_argument("--allowed-park", action="append", default=[])
    cohort.add_argument("--allow-missing-provenance", action="store_true")
    cohort.add_argument("--timing-screen-rate", type=float,
                        help="Opt-in calibration timing screen (M1-SCREEN-DRAFT-20260925): "
                             "rate deviation above which the anchor-error formula applies.")
    cohort.add_argument("--timing-screen-max-frames", type=float, default=1.0,
                        help="Maximum predicted onset error in native frames (default 1).")
    cohort.add_argument("--corpus-screen", choices=("corpus-screen-v1",),
                        help="Frozen whole-segment screen (M1-CORPUS-AUDIT-20260927); "
                             "exclusive with --timing-screen-rate.")

    mix = commands.add_parser("park-mix", help="merge frozen cohorts and select exact per-park counts")
    mix.add_argument("--cohort", type=Path, action="append", required=True)
    mix.add_argument("--quota", action="append", required=True, help="PARK=COUNT (repeat per park)")
    mix.add_argument("--seed", type=int, default=0)
    mix.add_argument("--name", required=True)
    mix.add_argument("--role", choices=("training", "validation", "challenge", "certification"),
                     default="training")
    mix.add_argument("--out", type=Path, required=True)

    subsets = commands.add_parser("subsets", help="build deterministic nested training prefixes")
    subsets.add_argument("--cohort", type=Path, required=True)
    subsets.add_argument("--out-dir", type=Path, required=True)
    subsets.add_argument("--sizes", type=_sizes, default=list(DEFAULT_LINEAR_RUNGS))
    subsets.add_argument("--seed", type=int, default=0)

    experiment = commands.add_parser("experiment", help="bind explicit trainer partitions")
    experiment.add_argument("--train", type=Path, required=True)
    experiment.add_argument("--validation", type=Path, required=True)
    experiment.add_argument("--certification", type=Path)
    experiment.add_argument("--name", required=True)
    experiment.add_argument("--out", type=Path, required=True)

    check = commands.add_parser("check", help="check zero leakage between frozen cohorts")
    check.add_argument("manifest", nargs="+", type=Path)

    args = parser.parse_args()
    if args.command == "cohort":
        payload = build_linear_cohort_manifest(
            args.data, cohort=args.name, role=args.role, corpus_root=args.corpus_root,
            require_provenance=not args.allow_missing_provenance,
            allowed_parks=args.allowed_park,
            timing_screen=(None if args.timing_screen_rate is None else TimingScreen(
                rate_threshold=args.timing_screen_rate,
                max_error_frames=args.timing_screen_max_frames)),
            corpus_screen=args.corpus_screen,
        )
        written = write_manifest(args.out, payload)
        print(json.dumps({"path": str(args.out), "sample_count": written["sample_count"],
                          "fingerprint": written["fingerprint"]}, indent=2))
    elif args.command == "park-mix":
        quotas = {}
        for item in args.quota:
            park, _, count = item.rpartition("=")
            if not park or not count.isdigit():
                parser.error(f"--quota must be PARK=COUNT, got {item!r}")
            quotas[park] = int(count)
        payload = build_park_mix_cohort(
            [read_manifest(path) for path in args.cohort], quotas,
            seed=args.seed, cohort=args.name, role=args.role,
        )
        written = write_manifest(args.out, payload)
        print(json.dumps({"path": str(args.out), "sample_count": written["sample_count"],
                          "fingerprint": written["fingerprint"],
                          "available": payload["park_mix"]["available"]}, indent=2))
    elif args.command == "subsets":
        payloads = build_nested_subset_manifests(
            read_manifest(args.cohort), args.sizes, seed=args.seed,
        )
        assert_deterministic_nesting(payloads)
        outputs = []
        for payload in payloads:
            path = args.out_dir / f"linear_train_n{payload['sample_count']}.json"
            written = write_manifest(path, payload)
            outputs.append({"path": str(path), "samples": written["sample_count"],
                            "fingerprint": written["fingerprint"]})
        print(json.dumps(outputs, indent=2))
    elif args.command == "experiment":
        payload = build_experiment_manifest(
            read_manifest(args.train), read_manifest(args.validation),
            certification_cohort=(read_manifest(args.certification)
                                  if args.certification else None),
            name=args.name,
        )
        written = write_manifest(args.out, payload)
        print(json.dumps({"path": str(args.out), "fingerprint": written["fingerprint"],
                          "partition_sizes": {name: len(entries) for name, entries in
                                              written["partitions"].items()}}, indent=2))
    else:
        manifests = [read_manifest(path) for path in args.manifest]
        assert_zero_cohort_leakage(manifests)
        print(json.dumps({"checked": len(manifests), "zero_leakage": True}, indent=2))


if __name__ == "__main__":
    main()
