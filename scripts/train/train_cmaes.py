"""CLI entry point for CMA-ES gesture optimization.

Connects to devices via Appium and runs the parallel CMA-ES loop from
trueskate_ai.rl.cmaes_optimizer. Device connections are managed internally
by run() via DeviceWorker instances.

Usage:
    python scripts/train_cmaes.py [options]

Options:
    --max-evals   Total evaluations before stopping (default: 1800)
    --seed        CMA-ES random seed (default: 42)
    --wait-time   Seconds to wait for trick text after gestures (default: 0.0)
    --settle-time Seconds to wait after reset before next attempt (default: 0.5)
    --pop-size    CMA-ES population size — evals per generation (default: 24)
    --log-dir     Log directory (default: logs/)
    --target-trick Target trick name for conditioned post-hoc rescoring.
    --initial-mean Path to trick library JSON; seeds mean from best_gestures.
"""
import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from trueskate_ai.rl.cmaes.cmaes_optimizer import run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CMA-ES optimization loop for the True Skate 360 flip experiment."
    )
    parser.add_argument("--max-evals", type=int, default=1800,
                        help="Total evaluations before stopping (default: 1800)")
    parser.add_argument("--seed", type=int, default=42,
                        help="CMA-ES random seed (default: 42)")
    parser.add_argument("--wait-time", type=float, default=0.0,
                        help="Seconds to wait for trick text after gestures (default: 0.0)")
    parser.add_argument("--settle-time", type=float, default=0.5,
                        help="Seconds to wait after reset before next attempt (default: 0.5)")
    parser.add_argument("--pop-size", type=int, default=24,
                        help="CMA-ES population size — evals per generation (default: 24)")
    parser.add_argument("--log-dir", type=Path,
                        default=_REPO_ROOT / "logs",
                        help="Log directory (default: logs/)")
    parser.add_argument("--target-trick", type=str, default=None,
                        help="Target trick name for conditioned post-hoc rescoring")
    parser.add_argument("--initial-mean", type=Path, default=None,
                        help="Path to trick library JSON used to seed CMA-ES mean from best_gestures")
    args = parser.parse_args()

    run(
        max_evals=args.max_evals,
        seed=args.seed,
        wait_time=args.wait_time,
        settle_time=args.settle_time,
        pop_size=args.pop_size,
        log_dir=args.log_dir,
        target_trick=args.target_trick,
        initial_mean=args.initial_mean,
    )


if __name__ == "__main__":
    main()
