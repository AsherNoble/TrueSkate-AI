"""Mine a trick library for EVERY distinct landed trick across run logs.

Where build_trick_library.py extracts one named trick from one log, this
sweeps a set of JSONL logs and emits a library per (trick component, gesture
count) — capturing the long tail of accidentally-discovered tricks whose only
existing params are a handful of lucky evals.

Combo detections ("HARD FLIP + NOSE SLIDE") credit every component, same as
Curriculum.score. Tricks landed at both N=2 and N=3 produce separate files
(suffixed _2g/_3g) because param vectors aren't interchangeable across N.

Usage:
    python scripts/data/mine_all_tricks.py \
        --logs "logs/overnight/*/*/runs/cmaes_run_*/cmaes_run_*.jsonl" \
        --out-dir trick_libraries/tail_20260611 \
        [--min-samples 1] [--landed-only]

Output schema matches build_trick_library.py (median + best gestures), so
files are directly usable as --initial-mean warm starts and by
execute_trick.py / replay harnesses.
"""
import argparse
import glob
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
    build_param_bounds,
    clamp_params,
    infer_num_gestures,
    unpack_gesture_params,
)


def _sanitize_filename(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _unpack_to_recipe(params: list[float]) -> dict:
    arr = np.array(params, dtype=np.float64)
    num_gestures = infer_num_gestures(len(arr))
    bounds = build_param_bounds(num_gestures)
    recipe = unpack_gesture_params(clamp_params(arr, bounds), num_gestures)
    for g in recipe["gestures"]:
        g["points"] = [list(p) for p in g["points"]]
    return recipe


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine a library per distinct landed trick across many run logs."
    )
    parser.add_argument("--logs", required=True,
                        help="Glob for JSONL run logs (quote it)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-samples", type=int, default=1)
    parser.add_argument("--landed-only", action="store_true", default=True,
                        help="(default on) only status=='landed' rows")
    args = parser.parse_args()

    log_paths = sorted(glob.glob(args.logs))
    if not log_paths:
        sys.exit(f"ERROR: no logs match {args.logs}")

    # (component, num_gestures) -> list of (params, reward, full_trick_name)
    groups: dict[tuple[str, int], list] = defaultdict(list)
    rows = 0
    for path in log_paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("type") in ("run_config", "generation_summary"):
                    continue
                if r.get("trick_status") != "landed" or not r.get("trick_name"):
                    continue
                params = r.get("params")
                if not params or (len(params) + 1) % 9 != 0:
                    continue
                rows += 1
                n = infer_num_gestures(len(params))
                for comp in {c.strip().upper() for c in r["trick_name"].split(" + ") if c.strip()}:
                    groups[(comp, n)].append((params, float(r.get("reward", 0.0)), r["trick_name"]))

    print(f"{rows} landed evals across {len(log_paths)} logs → "
          f"{len(groups)} (trick, N) groups")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    written = 0
    multi_n = {c for (c, _) in groups if sum(1 for (c2, _) in groups if c2 == c) > 1}

    for (comp, n), samples in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if len(samples) < args.min_samples:
            continue
        matrix = np.array([s[0] for s in samples], dtype=np.float64)
        rewards = np.array([s[1] for s in samples])
        best_idx = int(np.argmax(rewards))
        output = {
            "trick": comp,
            "median_gestures": _unpack_to_recipe(np.median(matrix, axis=0).tolist()),
            "best_gestures": _unpack_to_recipe(samples[best_idx][0]),
            "sample_count": len(samples),
            "landed_only": True,
            "num_gestures": n,
            "reward_stats": {
                "min": round(float(rewards.min()), 4),
                "mean": round(float(rewards.mean()), 4),
                "max": round(float(rewards.max()), 4),
            },
            "source_log": args.logs,
        }
        suffix = f"_{n}g" if comp in multi_n else ""
        out_path = args.out_dir / f"{_sanitize_filename(comp)}{suffix}_{timestamp}.json"
        out_path.write_text(json.dumps(output, indent=2) + "\n")
        written += 1
        print(f"  {len(samples):5d} samples  N={n}  {comp}")

    print(f"\n{written} libraries → {args.out_dir}")


if __name__ == "__main__":
    main()
