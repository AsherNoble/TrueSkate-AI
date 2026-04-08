"""Build a trick library entry from a CMA-ES run log.

Reads a JSONL log produced by scripts/train_cmaes.py, filters rows matching
a given trick name, and computes median and best gesture recipes.

Usage:
    python scripts/build_trick_library.py --log <jsonl_path> --trick <name>

Output:
    trick_libraries/<trick_name>_<timestamp>.json
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
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from trueskate_ai.rl.action_param import clamp_params, unpack_action


def _sanitize_filename(name: str) -> str:
    """Convert a trick name to a safe filename component."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _unpack_to_recipe(params: list[float]) -> dict:
    """Clamp + unpack a 17-element param list into a JSON-serializable recipe."""
    action = unpack_action(clamp_params(np.array(params, dtype=np.float64)))
    for g in action["gestures"]:
        g["points"] = [list(p) for p in g["points"]]
    return action


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a trick library entry from a CMA-ES run log."
    )
    parser.add_argument("--log", type=Path, required=True,
                        help="Path to the JSONL run log")
    parser.add_argument("--trick", type=str, required=True,
                        help="Trick name to filter for (case-insensitive)")
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
            if row.get("type") == "generation_summary":
                continue
            trick_name = row.get("trick_name")
            if trick_name and trick_name.lower() == trick_lower:
                matches.append(row)

    if not matches:
        sys.exit(f"ERROR: no rows matching trick '{args.trick}' found in {args.log}")

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
        "reward_stats": reward_stats,
        "source_log": str(args.log),
    }

    out_dir = _REPO_ROOT / "trick_libraries"
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
