"""CMA-ES optimization loop for True Skate gesture search.

Orchestrates parallel multi-device evaluation via DeviceWorker instances
dispatched through a ThreadPoolExecutor. The CLI entry point is
scripts/train_cmaes.py.

Public API:
    run()  — execute the full CMA-ES optimization loop across multiple devices.
"""
import json
import logging
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from trueskate_ai.rl.cmaes.action_param import INITIAL_MEAN, INITIAL_SIGMA, PARAM_BOUNDS
from trueskate_ai.rl.device_worker import DEVICES, DeviceWorker
from trueskate_ai.rl.reward import compute_reward_for_target
from trueskate_ai.sim.trick_info_reader import TrickResult

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _open_log(log_dir: Path) -> tuple[Path, Path, object]:
    """Create a run folder with JSONL log and frames/ subdir. Returns (run_dir, log_path, file_handle)."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = log_dir / "runs" / f"cmaes_run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "frames").mkdir(exist_ok=True)
    log_path = run_dir / f"cmaes_run_{run_id}.jsonl"
    return run_dir, log_path, log_path.open("w", buffering=1)  # line-buffered


def _write_log(fh, record: dict) -> None:
    """Append a JSON record to the log file."""
    fh.write(json.dumps(record) + "\n")


def _save_checkpoint(es, run_dir: Path, generation: int) -> None:
    """Pickle the CMA-ES object to a checkpoint file inside the run folder."""
    path = run_dir / f"checkpoint_gen{generation}.pkl"
    with path.open("wb") as f:
        pickle.dump(es, f)


def _timed_worker_reset(worker) -> tuple[str, float, float]:
    reset_started_at = time.monotonic()
    worker.reset()
    return worker.device_id, reset_started_at, time.monotonic() - reset_started_at


def _load_initial_mean_from_library(path: Path) -> np.ndarray:
    """Load best_gestures from a trick library JSON and pack into 17 params."""
    data = json.loads(path.read_text())
    best_gestures = data.get("best_gestures")
    if not isinstance(best_gestures, dict):
        raise ValueError(f"Missing or invalid 'best_gestures' in {path}")

    gestures = best_gestures.get("gestures")
    delays = best_gestures.get("delays")
    if not isinstance(gestures, list) or len(gestures) != 2:
        raise ValueError(f"Expected best_gestures.gestures to have exactly 2 entries in {path}")
    if not isinstance(delays, list) or len(delays) != 1:
        raise ValueError(f"Expected best_gestures.delays to have exactly 1 entry in {path}")

    params: list[float] = []
    for gesture_idx, gesture in enumerate(gestures):
        if not isinstance(gesture, dict):
            raise ValueError(f"Gesture {gesture_idx} in {path} is not an object")
        points = gesture.get("points")
        if not isinstance(points, list) or len(points) != 3:
            raise ValueError(f"Gesture {gesture_idx} must contain exactly 3 points in {path}")
        for point_idx, point in enumerate(points):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError(
                    f"Gesture {gesture_idx} point {point_idx} must be [x, y] in {path}"
                )
            params.extend([float(point[0]), float(point[1])])
        params.append(float(gesture["duration"]))
        params.append(float(gesture["easing_power"]))

    params.append(float(delays[0]))
    mean = np.array(params, dtype=np.float64)
    if mean.shape != (17,):
        raise ValueError(f"Expected 17 parameters from best_gestures in {path}, got {mean.shape}")
    return mean


# ---------------------------------------------------------------------------
# Frame composites
# ---------------------------------------------------------------------------

def _save_composites(frames: list[np.ndarray], eval_dir: Path, chunk_size: int = 3) -> int:
    """Max-pool frames into chunks, save each as a grayscale PNG. Returns composite count."""
    n_complete = len(frames) // chunk_size
    if n_complete == 0:
        return 0
    eval_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(n_complete):
        chunk = frames[idx * chunk_size : (idx + 1) * chunk_size]
        composite = np.max(np.stack(chunk, axis=0), axis=0)  # (210, 455) uint8
        Image.fromarray(composite, mode="L").save(eval_dir / f"frame_{idx:02d}.png")
    return n_complete


# ---------------------------------------------------------------------------
# Main optimization loop
# ---------------------------------------------------------------------------

def run(
    *,
    max_evals: int = 1800,
    seed: int = 42,
    wait_time: float = 0.0,
    settle_time: float = 0.5,
    pop_size: int = 24,
    log_dir: Path,
    devices: list[dict] | None = None,
    target_trick: str | None = None,
    initial_mean: Path | None = None,
) -> None:
    """Execute the CMA-ES optimization loop across multiple devices.

    Creates one DeviceWorker per device config, connects them, and
    dispatches candidate evaluations in parallel via ThreadPoolExecutor.

    Args:
        max_evals:   Total evaluations before stopping (rounded down to
                     nearest multiple of adjusted pop_size).
        seed:        CMA-ES random seed.
        wait_time:   Seconds to wait after gestures before first OCR screenshot.
        settle_time: Seconds to wait after board reset before next attempt.
        pop_size:    CMA-ES population size (rounded down to nearest multiple
                     of device count).
        log_dir:     Root directory for run logs and frame composites.
        devices:     List of device config dicts. Defaults to DEVICES from
                     device_worker module.
        target_trick: Optional trick name for post-hoc target-aware rescoring.
        initial_mean: Optional path to trick library JSON; when provided,
                      best_gestures seeds the CMA-ES mean.
    """
    try:
        import cma
    except ImportError:
        raise ImportError(
            "'cma' package not found. Install it with: pip install cma"
        )

    if devices is None:
        devices = DEVICES

    # --- Connect workers ---------------------------------------------------
    workers = [DeviceWorker(cfg) for cfg in devices]
    n_workers = len(workers)

    for worker in workers:
        worker.connect()

    # --- Auto-round pop_size and max_evals ---------------------------------
    original_pop_size = pop_size
    pop_size = max(n_workers, (pop_size // n_workers) * n_workers)
    original_max_evals = max_evals
    max_evals = (max_evals // pop_size) * pop_size
    if pop_size != original_pop_size or max_evals != original_max_evals:
        print(
            f"Adjusted for {n_workers} device(s): "
            f"pop_size {original_pop_size} → {pop_size}, "
            f"max_evals {original_max_evals} → {max_evals}"
        )

    run_dir, log_path, log_fh = _open_log(log_dir)
    print(f"Run folder: {run_dir}")
    print(f"Logging to {log_path}")
    print(f"Workers: {[w.device_id for w in workers]}")

    bounds = [PARAM_BOUNDS[:, 0].tolist(), PARAM_BOUNDS[:, 1].tolist()]
    cma_mean = INITIAL_MEAN.copy()
    cma_stds = INITIAL_SIGMA.copy()
    if initial_mean is not None:
        cma_mean = _load_initial_mean_from_library(initial_mean)
        coordinate_mask = np.ones(17, dtype=bool)
        coordinate_mask[[6, 7, 14, 15, 16]] = False
        cma_stds[coordinate_mask] = 20.0
        print(f"Loaded initial mean from trick library: {initial_mean}")
    if target_trick is not None:
        target_trick = target_trick.strip()
        if not target_trick:
            target_trick = None

    es = cma.CMAEvolutionStrategy(
        cma_mean.tolist(),
        1.0,  # overall sigma — per-parameter scaling handled by CMA_stds
        {
            "bounds": bounds,
            "CMA_stds": cma_stds.tolist(),
            "seed": seed,
            "maxiter": max_evals,  # generous ceiling; real stop is max_evals
            "verbose": -9,         # suppress CMA-ES internal printing
            "popsize": pop_size,
        },
    )

    eval_num = 0
    generation = 0
    best_reward = 0.0
    best_trick: str | None = None
    best_params: np.ndarray = INITIAL_MEAN.copy()

    executor = ThreadPoolExecutor(max_workers=n_workers)

    try:
        while eval_num < max_evals:
            solutions = es.ask()

            # Process candidates in rounds of n_workers.
            # Each round: foreground check → settle → dispatch batch →
            # collect results → reset all devices simultaneously.
            n_rounds = len(solutions) // n_workers
            rewards = []
            device_eval_counts: dict[str, int] = {}

            for round_idx in range(n_rounds):
                batch_start = round_idx * n_workers

                # Ensure True Skate is in the foreground on all devices
                for worker in workers:
                    worker.ensure_foreground()

                # Settle time — all devices settle simultaneously
                time.sleep(settle_time)

                # Dispatch one candidate per worker simultaneously
                futures = {}
                for i, worker in enumerate(workers):
                    cand_idx = batch_start + i
                    cand_eval_num = eval_num + i + 1
                    future = executor.submit(
                        worker.evaluate,
                        solutions[cand_idx], wait_time, cand_eval_num, generation,
                    )
                    futures[future] = (cand_idx, worker)

                # Collect results for this round
                round_results: dict[int, dict] = {}
                completion_times: dict[int, float] = {}
                reset_futures: dict[object, int] = {}
                for future in as_completed(futures):
                    cand_idx, worker = futures[future]
                    try:
                        round_results[cand_idx] = future.result()
                    except Exception as exc:
                        # Safety net — DeviceWorker.evaluate() catches internally,
                        # but guard against unexpected thread-level failures.
                        logging.warning(
                            "Future for candidate %d raised: %s", cand_idx, exc
                        )
                        round_results[cand_idx] = {
                            "reward": 0.0,
                            "trick_name": None,
                            "trick_status": None,
                            "device_id": "unknown",
                            "params": solutions[cand_idx],
                            "raw_frames": [],
                            "n_composites": 0,
                            "app_relaunched": False,
                            "action_exec_s": 0.0,
                            "reward_eval_s": 0.0,
                            "eval_total_s": 0.0,
                            "capture_attempts": 0,
                            "skipped_captures": 0,
                            "detection_capture_idx": None,
                            "capture_elapsed_s": 0.0,
                            "monitor_frames_checked": 0,
                            "monitor_elapsed_s": 0.0,
                        }
                    completion_times[cand_idx] = time.monotonic()
                    reset_future = executor.submit(_timed_worker_reset, worker)
                    reset_futures[reset_future] = cand_idx

                reset_metrics: dict[int, dict[str, float]] = {}
                for reset_future in as_completed(reset_futures):
                    cand_idx = reset_futures[reset_future]
                    try:
                        _, reset_started_at, reset_duration = reset_future.result()
                    except Exception as exc:
                        logging.warning(
                            "Reset future for candidate %d failed: %s", cand_idx, exc
                        )
                        reset_started_at = completion_times.get(cand_idx, time.monotonic())
                        reset_duration = 0.0
                    post_eval_wait = max(
                        0.0,
                        reset_started_at - completion_times.get(cand_idx, reset_started_at),
                    )
                    reset_metrics[cand_idx] = {
                        "post_eval_wait_s": post_eval_wait,
                        "reset_s": reset_duration,
                    }

                # Process round results in candidate order
                for i in range(n_workers):
                    cand_idx = batch_start + i
                    result = round_results[cand_idx]
                    reward = result["reward"]
                    if target_trick is not None:
                        trick_name = result["trick_name"]
                        trick_status = result["trick_status"]
                        trick_result = (
                            TrickResult(trick=trick_name, status=trick_status)
                            if trick_name is not None and trick_status is not None
                            else None
                        )
                        reward = compute_reward_for_target(
                            trick_result,
                            target_trick=target_trick,
                        )
                        result["reward"] = reward
                    rewards.append(reward)
                    eval_num += 1

                    device_id = result["device_id"]
                    device_eval_counts[device_id] = (
                        device_eval_counts.get(device_id, 0) + 1
                    )

                    raw_frames = result["raw_frames"]
                    eval_dir_name = f"eval_{eval_num:05d}_{device_id}"
                    n_composites = _save_composites(
                        raw_frames, run_dir / "frames" / eval_dir_name
                    )

                    _write_log(log_fh, {
                        "generation": generation,
                        "candidate_idx": cand_idx,
                        "eval_num": eval_num,
                        "device_id": device_id,
                        "reward": reward,
                        "trick_name": result["trick_name"],
                        "trick_status": result["trick_status"],
                        "params": [round(float(p), 2) for p in result["params"]],
                        "frame_dir": eval_dir_name,
                        "n_composites": n_composites,
                        "app_relaunched": result["app_relaunched"],
                        "action_exec_s": round(result.get("action_exec_s", 0.0), 4),
                        "reward_eval_s": round(result.get("reward_eval_s", 0.0), 4),
                        "eval_total_s": round(result.get("eval_total_s", 0.0), 4),
                        "post_eval_wait_s": round(
                            reset_metrics.get(cand_idx, {}).get("post_eval_wait_s", 0.0), 4
                        ),
                        "reset_s": round(
                            reset_metrics.get(cand_idx, {}).get("reset_s", 0.0), 4
                        ),
                        "capture_attempts": result.get("capture_attempts", 0),
                        "skipped_captures": result.get("skipped_captures", 0),
                        "detection_capture_idx": result.get("detection_capture_idx"),
                        "capture_elapsed_s": round(result.get("capture_elapsed_s", 0.0), 4),
                        "monitor_frames_checked": result.get("monitor_frames_checked", 0),
                        "monitor_elapsed_s": round(result.get("monitor_elapsed_s", 0.0), 4),
                        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                    })

                    if reward > best_reward:
                        best_reward = reward
                        best_trick = result["trick_name"]
                        best_params = np.array(result["params"])

            # Feed negated rewards to CMA-ES (it minimizes)
            es.tell(solutions, [-r for r in rewards])
            es.disp()

            # CMA-ES convergence is informational only — the sparse reward
            # landscape looks flat to CMA-ES long before we've found the trick.
            stop_conditions = es.stop()
            if stop_conditions:
                print(
                    f"WARNING: CMA-ES convergence condition(s) fired "
                    f"(continuing): {stop_conditions}"
                )

            gen_best = max(rewards)
            gen_mean = float(np.mean(rewards))
            device_counts_str = "  ".join(
                f"{did}={cnt}" for did, cnt in sorted(device_eval_counts.items())
            )
            print(
                f"--- gen {generation} complete | "
                f"best={gen_best:.2f} mean={gen_mean:.2f} | "
                f"{device_counts_str} ---"
            )
            _write_log(log_fh, {
                "type": "generation_summary",
                "generation": generation,
                "best_reward": gen_best,
                "mean_reward": round(gen_mean, 4),
                "device_eval_counts": device_eval_counts,
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            })

            if (generation + 1) % 10 == 0:
                _save_checkpoint(es, run_dir, generation)
                print(f"Checkpoint saved at generation {generation}.")

            generation += 1

            if eval_num >= max_evals:
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        print("\n=== Run complete ===")
        print(f"  Total evaluations : {eval_num}")
        print(f"  Best reward       : {best_reward:.2f}")
        print(f"  Best trick        : {best_trick}")
        print(f"  Best params       : {[round(float(p), 2) for p in best_params]}")

        _save_checkpoint(es, run_dir, generation)
        print("Final checkpoint saved.")

        log_fh.close()

        executor.shutdown(wait=False)
        for worker in workers:
            worker.disconnect()
