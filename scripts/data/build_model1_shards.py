#!/usr/bin/env python3
"""Pack one frozen Model 1 experiment into bounded sequential tar shards."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from trueskate_ai.data.sequential_shards import build_sequential_shards  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--experiment-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--max-bytes", type=int, default=2 * 1024**3)
    args = parser.parse_args()
    result = build_sequential_shards(
        args.data, args.experiment_manifest, args.out_dir,
        max_samples=args.max_samples, max_bytes=args.max_bytes,
    )
    print(json.dumps({
        "manifest": str(args.out_dir / "shards.json"),
        "fingerprint": result["fingerprint"],
        "samples": result["sample_count"],
        "shards": len(result["shards"]),
        "bytes": sum(shard["bytes"] for shard in result["shards"]),
    }, indent=2))


if __name__ == "__main__":
    main()
