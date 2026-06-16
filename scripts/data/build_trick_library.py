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
from collections import defaultdict
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
    SPIN_PARAMS,
    build_param_bounds,
    clamp_params,
    infer_layout,
    unpack_gesture_params,
)


def _median_recipe_params(
    matrix: np.ndarray, rewards: np.ndarray, use_spin: bool
) -> list[float]:
    """Median the structural params; inherit the spin block from a real sample.

    The spin gate (params[-SPIN_PARAMS]) is a binary on/off decision
    (enabled = gate >= 0), not a continuous value — medianing a mixed on/off
    column drifts it toward 0 and fabricates an arbitrary spin decision. Instead:
    majority-vote the gate, then carry the WHOLE [gate, t_start, t_end] triple
    from the best-reward sample on the winning side, so the timing window stays
    consistent with a configuration that was actually flown. Ties -> enabled.
    """
    median = np.median(matrix, axis=0)
    if not use_spin:
        return median.tolist()
    spin_on = matrix[:, -SPIN_PARAMS] >= 0.0
    enable = spin_on.sum() >= (~spin_on).sum()  # majority vote; ties -> enabled
    side = spin_on if enable else ~spin_on
    rep = int(np.argmax(np.where(side, rewards, -np.inf)))  # best reward on winning side
    median[-SPIN_PARAMS:] = matrix[rep, -SPIN_PARAMS:]      # inherit real spin triple
    return median.tolist()


def _sanitize_filename(name: str) -> str:
    """Convert a trick name to a safe filename component."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _unpack_to_recipe(params: list[float]) -> dict:
    """Clamp + unpack a param list into a JSON-serializable recipe.

    Infers (num_gestures, use_spin) from the vector length, so legacy 17-param
    logs resolve to N=2 no-spin and 20-param spin logs carry a decoded ``spin``
    block in the recipe.
    """
    arr = np.array(params, dtype=np.float64)
    num_gestures, use_spin = infer_layout(len(arr))
    bounds = build_param_bounds(num_gestures, use_spin)
    recipe = unpack_gesture_params(clamp_params(arr, bounds), num_gestures, use_spin)
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

    # Read and filter matching rows. Canonicalize to UPPERCASE — KNOWN_TRICKS and
    # mine_all_tricks key uppercase; lowercase here broke cross-tool exact match.
    trick_upper = args.trick.upper()
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
            components = [c.strip().upper() for c in trick_name.split(" + ")]
            if trick_upper in components:
                matches.append(row)

    # `or not matches` guards the empty case even when --min-samples is 0, so we
    # never reach aggregation (np.array([])/np.argmax) with zero rows.
    if len(matches) < args.min_samples or not matches:
        print(
            f"Only {len(matches)} row(s) matching trick '{args.trick}' in {args.log} "
            f"(need {args.min_samples}) — no library written."
        )
        sys.exit(2)

    # Param vectors only share a length within one (N gestures, spin on/off)
    # layout — infer_layout maps length -> (N, use_spin). A trick landed at both
    # N=2/N=3 or spin/no-spin yields different-length vectors; mixing them into
    # one array is ragged/object (np.median then raises). So bucket by layout and
    # aggregate only the dominant one (most samples; tie -> higher total reward).
    by_layout: dict[tuple[int, bool], list] = defaultdict(list)
    for m in matches:
        try:
            layout = infer_layout(len(m["params"]))
        except ValueError:
            continue  # unrecognised vector length — skip, don't crash the run
        by_layout[layout].append(m)
    if not by_layout:
        print(f"No rows with a recognised param layout for '{args.trick}' — no library written.")
        sys.exit(2)

    (num_gestures, use_spin), layout_matches = max(
        by_layout.items(),
        key=lambda kv: (len(kv[1]), sum(m["reward"] for m in kv[1])),
    )

    # Extract param vectors and rewards (uniform length within this layout)
    param_matrix = np.array([m["params"] for m in layout_matches], dtype=np.float64)
    rewards = np.array([m["reward"] for m in layout_matches], dtype=np.float64)

    # Median structural params; spin gate is decided by vote, not medianed.
    median_recipe = _unpack_to_recipe(
        _median_recipe_params(param_matrix, rewards, use_spin)
    )

    # Best params → recipe (highest reward within the dominant layout)
    best_idx = int(np.argmax(rewards))
    best_recipe = _unpack_to_recipe(layout_matches[best_idx]["params"])

    reward_stats = {
        "min": round(float(np.min(rewards)), 4),
        "mean": round(float(np.mean(rewards)), 4),
        "max": round(float(np.max(rewards)), 4),
    }

    output = {
        "trick": args.trick.upper(),  # canonical uppercase key (matches mine_all_tricks)
        "median_gestures": median_recipe,
        "best_gestures": best_recipe,
        "sample_count": len(layout_matches),  # samples in the aggregated layout
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
