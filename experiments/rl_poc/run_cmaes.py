"""CMA-ES optimization loop for the 360 flip experiment.

Runs candidate gesture sequences on the phone, scores each attempt via
OCR-based trick detection, and feeds rewards back to CMA-ES.

Usage:
    python experiments/rl_poc/run_cmaes.py [options]

Options:
    --max-evals   Total evaluations before stopping (default: 1800)
    --seed        CMA-ES random seed (default: 42)
    --wait-time   Seconds to wait for trick text after gestures (default: 0.0)
    --settle-time Seconds to wait after reset before next attempt (default: 0.5)
    --pop-size    CMA-ES population size — evals per generation (default: 24)
    --log-dir     Log directory (default: experiments/rl_poc/logs)
"""
import argparse
import io
import json
import logging
import os
import pickle
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from PIL import Image

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
from reward import NoveltyTracker, get_reward
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

_BUNDLE_ID = "com.trueaxis.skate"

# query_app_state() return values (XCUITest / iOS)
_APP_STATE_FOREGROUND = 4


def connect_driver() -> webdriver.Remote:
    """Connect to Appium, reusing True Skate if it is already in the foreground.

    Uses no_reset=True so Appium never stops or reinstalls the app.
    After connecting, queries the app state:
      - Already in foreground (state 4): proceed without touching it.
      - Otherwise: activate it and wait briefly for the UI to settle.

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
    options.bundle_id = _BUNDLE_ID
    options.udid = udid
    options.wda_local_port = 8100
    options.use_prebuilt_wda = True
    options.skip_log_capture = True
    options.no_reset = True  # never stop/reinstall the app

    driver = webdriver.Remote("http://127.0.0.1:4723", options=options)

    state = driver.query_app_state(_BUNDLE_ID)
    if state == _APP_STATE_FOREGROUND:
        print("True Skate is already open — reusing.")
    else:
        print(f"True Skate not in foreground (state={state}) — activating.")
        driver.activate_app(_BUNDLE_ID)
        time.sleep(1.5)  # wait for the game UI to settle

    return driver


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _open_log(log_dir: Path) -> tuple[Path, Path, object]:
    """Create a run folder with JSONL log and frames/ subdir. Returns (run_dir, log_path, file_handle)."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = log_dir / "runs" / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "frames").mkdir(exist_ok=True)
    log_path = run_dir / f"run_{run_id}.jsonl"
    return run_dir, log_path, log_path.open("w", buffering=1)  # line-buffered


def _write_log(fh, record: dict) -> None:
    """Append a JSON record to the log file."""
    fh.write(json.dumps(record) + "\n")


def _save_checkpoint(es, run_dir: Path, generation: int) -> None:
    """Pickle the CMA-ES object to a checkpoint file inside the run folder."""
    path = run_dir / f"checkpoint_gen{generation}.pkl"
    with path.open("wb") as f:
        pickle.dump(es, f)


class FrameRecorder:
    """Captures 84×84 grayscale screenshots on a daemon thread during an eval."""

    def __init__(self):
        self._driver = None
        self._thread: threading.Thread | None = None
        self._stop_flag = False
        self._frames: list[np.ndarray] = []

    def start(self, driver) -> None:
        self._driver = driver
        self._stop_flag = False
        self._frames = []
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        while not self._stop_flag:
            try:
                png_bytes = self._driver.get_screenshot_as_png()
                img = Image.open(io.BytesIO(png_bytes)).convert("L").resize((84, 84), Image.LANCZOS)
                self._frames.append(np.array(img, dtype=np.uint8))
            except Exception:
                pass  # swallow transient capture errors; loop continues until stop_flag

    def stop(self) -> list[np.ndarray]:
        self._stop_flag = True
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        return self._frames


def _save_composites(frames: list[np.ndarray], eval_dir: Path, chunk_size: int = 12) -> int:
    """Max-pool frames into chunks, save each as a grayscale PNG. Returns composite count."""
    n_complete = len(frames) // chunk_size
    if n_complete == 0:
        return 0
    eval_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(n_complete):
        chunk = frames[idx * chunk_size : (idx + 1) * chunk_size]
        composite = np.max(np.stack(chunk, axis=0), axis=0)  # (84, 84) uint8
        Image.fromarray(composite, mode="L").save(eval_dir / f"frame_{idx:02d}.png")
    return n_complete


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
    parser.add_argument("--wait-time", type=float, default=0.0,
                        help="Seconds to wait for trick text after gestures (default: 0.0)")
    parser.add_argument("--settle-time", type=float, default=0.5,
                        help="Seconds to wait after reset before next attempt (default: 0.5)")
    parser.add_argument("--pop-size", type=int, default=24,
                        help="CMA-ES population size — evals per generation (default: 24)")
    parser.add_argument("--log-dir", type=Path,
                        default=_HERE / "logs",
                        help="Log directory (default: experiments/rl_poc/logs)")
    args = parser.parse_args()

    run_dir, log_path, log_fh = _open_log(args.log_dir)
    print(f"Run folder: {run_dir}")
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
            "popsize": args.pop_size,
        },
    )

    eval_num = 0
    generation = 0
    best_reward = 0.0
    best_trick: str | None = None
    best_params: np.ndarray = INITIAL_MEAN.copy()
    novelty_tracker = NoveltyTracker()

    try:
        while eval_num < args.max_evals:
            solutions = es.ask()
            rewards = []

            for candidate_idx, candidate in enumerate(solutions):
                # Wait for board to settle after previous reset
                time.sleep(args.settle_time)

                reward = 0.0
                trick_result = None
                novelty_bonus = 0.0
                recorder = FrameRecorder()
                try:
                    recorder.start(driver)

                    # Execute gestures on device
                    execute_action(driver, np.array(candidate))

                    # Score the attempt
                    reward, trick_result, novelty_bonus = get_reward(
                        driver, wait_time=args.wait_time, tracker=novelty_tracker
                    )
                except Exception as exc:
                    recorder.stop()
                    logging.warning("candidate %d failed: %s", candidate_idx, exc)
                    eval_num += 1
                    print(
                        f"[eval {eval_num}/{args.max_evals} | gen {generation}] "
                        f"ERROR: {exc} — assigning reward=0.0"
                    )
                    rewards.append(0.0)
                    reset_position(driver)
                    continue

                raw_frames = recorder.stop()
                rewards.append(reward)
                eval_num += 1

                # Stack raw frames into max composites and save
                eval_dir_name = f"eval_{eval_num:05d}"
                n_composites = _save_composites(
                    raw_frames, run_dir / "frames" / eval_dir_name
                )

                trick_str = trick_result.trick if trick_result else None
                trick_status = trick_result.status if trick_result else None

                # Console progress line
                print(
                    f"[eval {eval_num}/{args.max_evals} | gen {generation}] "
                    f"reward={reward:.2f}  trick={trick_str}  status={trick_status}  "
                    f"raw_frames={len(raw_frames)} composites={n_composites}"
                )

                # Per-evaluation JSONL record
                _write_log(log_fh, {
                    "generation": generation,
                    "candidate_idx": candidate_idx,
                    "eval_num": eval_num,
                    "reward": reward,
                    "novelty_bonus": round(novelty_bonus, 4),
                    "trick_name": trick_str,
                    "trick_status": trick_status,
                    "params": [round(float(p), 2) for p in candidate],
                    "frame_dir": eval_dir_name,
                    "n_composites": n_composites,
                    "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                })

                # Track global best
                if reward > best_reward:
                    best_reward = reward
                    best_trick = trick_str
                    best_params = np.array(candidate)

                # Reset board for next candidate
                reset_position(driver)

            # Feed negated rewards to CMA-ES (it minimizes)
            es.tell(solutions, [-r for r in rewards])
            es.disp()

            # CMA-ES convergence is informational only — the sparse reward
            # landscape looks flat to CMA-ES long before we've found the trick.
            stop_conditions = es.stop()
            if stop_conditions:
                print(f"WARNING: CMA-ES convergence condition(s) fired (continuing): {stop_conditions}")

            # Generation summary
            gen_best = max(rewards)
            gen_mean = float(np.mean(rewards))
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
                _save_checkpoint(es, run_dir, generation)
                print(f"Checkpoint saved at generation {generation}.")

            generation += 1

            # --max-evals is a minimum: exit cleanly after the generation that crosses it
            if eval_num >= args.max_evals:
                break

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
        _save_checkpoint(es, run_dir, generation)
        print(f"Final checkpoint saved.")

        log_fh.close()
        driver.quit()
        print("Driver closed.")


if __name__ == "__main__":
    main()
