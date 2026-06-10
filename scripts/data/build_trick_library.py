"""Build a trick library entry from a CMA-ES run log.

Reads a JSONL log produced by scripts/train_cmaes.py, filters rows matching
a given trick name, and computes median and best gesture recipes.

Usage:
    python scripts/data/build_trick_library.py --log <jsonl_path> --trick <name>
        [--landed-only] [--min-samples N] [--out-dir DIR]

Output:
    <out-dir>/<trick_name>_<timestamp>.json   (default out-dir: trick_libraries/)

Exit codes:
    0  library written
    2  fewer than --min-samples matching rows (no file written) — lets callers
       distinguish "not enough data" from real failure
"""
import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from trueskate_ai.rl.cmaes.action_param import (
    build_param_bounds,
    clamp_params,
    infer_num_gestures,
    unpack_gesture_params,
)


def _sanitize_filename(name: str) -> str:
    """Convert a trick name to a safe filename component."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _unpack_to_recipe(params: list[float]) -> dict:
    """Clamp + unpack a param list into a JSON-serializable recipe.

    Infers num_gestures from the vector length (9N - 1 = len(params)),
    so legacy 17-param logs naturally resolve to N=2.
    """
    arr = np.array(params, dtype=np.float64)
    num_gestures = infer_num_gestures(len(arr))
    bounds = build_param_bounds(num_gestures)
    recipe = unpack_gesture_params(clamp_params(arr, bounds), num_gestures)
    for g in recipe["gestures"]:
        g["points"] = [list(p) for p in g["points"]]
    return recipe


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a trick library entry from a CMA-ES run log."
    )
    parser.add_argument("--log", type=Path, required=True,
                        help="Path to the JSONL run log")
    parser.add_argument("--trick", type=str, required=True,
                        help="Trick name to filter for (case-insensitive; matches combo components)")
    parser.add_argument("--landed-only", action="store_true",
                        help="Only include rows with trick_status == 'landed' "
                             "(failed attempts pollute medians)")
    parser.add_argument("--min-samples", type=int, default=1,
                        help="Exit with code 2 (no file) if fewer matching rows (default: 1)")
    parser.add_argument("--out-dir", type=Path, default=_REPO_ROOT / "trick_libraries",
                        help="Output directory (default: trick_libraries/)")
    args = parser.parse_args()

    if not args.log.exists():
        sys.exit(f"ERROR: log file not found: {args.log}")

    # Read and filter matching rows
    trick_lower = args.trick.lower()
    matches = []
    with args.log.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("type") in ("generation_summary", "run_config"):
                continue
            trick_name = row.get("trick_name")
            if not trick_name:
                continue
            if args.landed_only and row.get("trick_status") != "landed":
                continue
            # Combo detections ("KICKFLIP + 50 50 GRIND") count as a match on
            # any component — same semantics as Curriculum.score.
            components = [c.strip().lower() for c in trick_name.split(" + ")]
            if trick_lower in components:
                matches.append(row)

    if len(matches) < max(1, args.min_samples):
        print(
            f"Only {len(matches)} row(s) matching trick '{args.trick}' in {args.log} "
            f"(need {max(1, args.min_samples)}) — no library written."
        )
        sys.exit(2)

    # Extract param vectors and rewards
    param_matrix = np.array([m["params"] for m in matches], dtype=np.float64)
    rewards = np.array([m["reward"] for m in matches], dtype=np.float64)

    # Median params → recipe
    median_params = np.median(param_matrix, axis=0)
    median_recipe = _unpack_to_recipe(median_params.tolist())

    # Best params → recipe (highest reward)
    best_idx = int(np.argmax(rewards))
    best_recipe = _unpack_to_recipe(matches[best_idx]["params"])

    reward_stats = {
        "min": round(float(np.min(rewards)), 4),
        "mean": round(float(np.mean(rewards)), 4),
        "max": round(float(np.max(rewards)), 4),
    }

    output = {
        "trick": args.trick,
        "median_gestures": median_recipe,
        "best_gestures": best_recipe,
        "sample_count": len(matches),
        "landed_only": args.landed_only,
        "reward_stats": reward_stats,
        "source_log": str(args.log),
    }

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _sanitize_filename(args.trick)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{safe_name}_{timestamp}.json"
    out_path.write_text(json.dumps(output, indent=2) + "\n")

    print(f"Trick library entry written to {out_path}")
    print(f"  Trick: {args.trick}")
    print(f"  Matches: {len(matches)}")
    print(f"  Reward: min={reward_stats['min']}, mean={reward_stats['mean']}, max={reward_stats['max']}")


if __name__ == "__main__":
    main()
