"""CMA-ES optimization loop for the 360 flip experiment.

Runs candidate gesture sequences on the phone, scores each attempt via
OCR-based trick detection, and feeds rewards back to CMA-ES.

Usage:
    python experiments/rl_poc/run_cmaes.py [options]

Options:
    --max-evals   Total evaluations before stopping (default: 1800)
    --seed        CMA-ES random seed (default: 42)
    --wait-time   Seconds to wait for trick text after gestures (default: 1.5)
    --settle-time Seconds to wait after reset before next attempt (default: 0.5)
    --log-dir     Log directory (default: experiments/rl_poc/logs)
"""
import argparse
import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — must happen before local imports
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
for _p in [str(_HERE), str(_REPO_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
try:
    import cma
except ImportError:
    sys.exit(
        "ERROR: 'cma' package not found. Install it with:\n"
        "    pip install cma"
    )

import numpy as np
from appium import webdriver
from appium.options.ios import XCUITestOptions
from dotenv import load_dotenv

from action_param import INITIAL_MEAN, INITIAL_SIGMA, PARAM_BOUNDS, execute_action
from reward import get_reward
from trueskate_ai.sim.touch_actions import reset_position

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def connect_driver() -> webdriver.Remote:
    """Connect to Appium and launch True Skate.

    Reads IPHONE_UDID from the environment (via .env).

    Returns:
        Appium WebDriver instance.
    """
    load_dotenv(_REPO_ROOT / ".env")
    udid = os.environ.get("IPHONE_UDID")
    if not udid:
        raise RuntimeError(
            "IPHONE_UDID not set. Copy .env.example to .env and fill in your device UDID."
        )

    options = XCUITestOptions()
    options.platform_name = "iOS"
    options.automation_name = "XCUITest"
    options.bundle_id = "com.trueaxis.skate"
    options.udid = udid
    options.wda_local_port = 8100
    options.use_prebuilt_wda = True
    options.skip_log_capture = True

    driver = webdriver.Remote("http://127.0.0.1:4723", options=options)
    print("Connected to True Skate via Appium.")
    return driver


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _open_log(log_dir: Path) -> tuple[Path, object]:
    """Create the run JSONL log file and return (path, file handle)."""
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"run_{run_id}.jsonl"
    return log_path, log_path.open("w", buffering=1)  # line-buffered


def _write_log(fh, record: dict) -> None:
    """Append a JSON record to the log file."""
    fh.write(json.dumps(record) + "\n")


def _save_checkpoint(es, log_dir: Path, generation: int) -> None:
    """Pickle the CMA-ES object to a checkpoint file."""
    path = log_dir / f"checkpoint_gen{generation}.pkl"
    with path.open("wb") as f:
        pickle.dump(es, f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse args, connect to device, and run the CMA-ES optimization loop."""
    parser = argparse.ArgumentParser(
        description="CMA-ES optimization loop for the True Skate 360 flip experiment."
    )
    parser.add_argument("--max-evals", type=int, default=1800,
                        help="Total evaluations before stopping (default: 1800)")
    parser.add_argument("--seed", type=int, default=42,
                        help="CMA-ES random seed (default: 42)")
    parser.add_argument("--wait-time", type=float, default=1.5,
                        help="Seconds to wait for trick text after gestures (default: 1.5)")
    parser.add_argument("--settle-time", type=float, default=0.5,
                        help="Seconds to wait after reset before next attempt (default: 0.5)")
    parser.add_argument("--log-dir", type=Path,
                        default=_HERE / "logs",
                        help="Log directory (default: experiments/rl_poc/logs)")
    args = parser.parse_args()

    log_path, log_fh = _open_log(args.log_dir)
    print(f"Logging to {log_path}")

    driver = connect_driver()

    # CMA-ES bounds format: [list of lower bounds, list of upper bounds]
    bounds = [PARAM_BOUNDS[:, 0].tolist(), PARAM_BOUNDS[:, 1].tolist()]

    es = cma.CMAEvolutionStrategy(
        INITIAL_MEAN.tolist(),
        1.0,  # overall sigma — per-parameter scaling handled by CMA_stds
        {
            "bounds": bounds,
            "CMA_stds": INITIAL_SIGMA.tolist(),
            "seed": args.seed,
            "maxiter": args.max_evals,  # generous ceiling; real stop is max_evals
            "verbose": -9,             # suppress CMA-ES internal printing
        },
    )

    eval_num = 0
    generation = 0
    best_reward = 0.0
    best_trick: str | None = None
    best_params: np.ndarray = INITIAL_MEAN.copy()

    try:
        while not es.stop() and eval_num < args.max_evals:
            solutions = es.ask()
            rewards = []

            for candidate_idx, candidate in enumerate(solutions):
                if eval_num >= args.max_evals:
                    break

                # Wait for board to settle after previous reset
                time.sleep(args.settle_time)

                # Execute gestures on device
                execute_action(driver, np.array(candidate))

                # Capture screenshot and score
                reward, trick_name = get_reward(driver, wait_time=args.wait_time)
                rewards.append(reward)
                eval_num += 1

                # Console progress line
                print(
                    f"[eval {eval_num}/{args.max_evals} | gen {generation}] "
                    f"reward={reward:.1f}  trick={trick_name}"
                )

                # Per-evaluation JSONL record
                _write_log(log_fh, {
                    "generation": generation,
                    "candidate_idx": candidate_idx,
                    "eval_num": eval_num,
                    "reward": reward,
                    "trick_name": trick_name,
                    "params": [round(float(p), 2) for p in candidate],
                    "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                })

                # Track global best
                if reward > best_reward:
                    best_reward = reward
                    best_trick = trick_name
                    best_params = np.array(candidate)

                # Reset board for next candidate
                reset_position(driver)

            # Feed negated rewards to CMA-ES (it minimizes)
            if rewards:
                es.tell(solutions[:len(rewards)], [-r for r in rewards])
                es.disp()

            # Generation summary
            gen_best = max(rewards) if rewards else 0.0
            gen_mean = float(np.mean(rewards)) if rewards else 0.0
            print(
                f"--- gen {generation} complete | "
                f"best={gen_best:.2f} mean={gen_mean:.2f} ---"
            )
            _write_log(log_fh, {
                "type": "generation_summary",
                "generation": generation,
                "best_reward": gen_best,
                "mean_reward": round(gen_mean, 4),
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            })

            # Checkpoint every 10 generations
            if (generation + 1) % 10 == 0:
                _save_checkpoint(es, args.log_dir, generation)
                print(f"Checkpoint saved at generation {generation}.")

            generation += 1

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        # Summary
        print("\n=== Run complete ===")
        print(f"  Total evaluations : {eval_num}")
        print(f"  Best reward       : {best_reward:.2f}")
        print(f"  Best trick        : {best_trick}")
        print(f"  Best params       : {[round(float(p), 2) for p in best_params]}")

        # Final checkpoint
        _save_checkpoint(es, args.log_dir, generation)
        print(f"Final checkpoint saved.")

        log_fh.close()
        driver.quit()
        print("Driver closed.")


if __name__ == "__main__":
    main()
