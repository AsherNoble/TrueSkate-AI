"""CMA-ES optimization loop for True Skate gesture search.

Provides the reusable infrastructure and main loop extracted from the
run_cmaes experiment script. The CLI entry point is scripts/train_cmaes.py.

Public API:
    connect_driver()  — connect to Appium, activate True Skate if needed.
    run()             — execute the full CMA-ES optimization loop.
"""
import io
import json
import logging
import os
import pickle
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
from appium import webdriver
from appium.options.ios import XCUITestOptions
from dotenv import load_dotenv
from PIL import Image

from trueskate_ai.rl.action_param import INITIAL_MEAN, INITIAL_SIGMA, PARAM_BOUNDS, execute_action
from trueskate_ai.rl.reward import RepetitionPenalty, get_reward
from trueskate_ai.sim.touch_actions import reset_position

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Appium / device constants
# ---------------------------------------------------------------------------

_BUNDLE_ID = "com.trueaxis.skate"
_MJPEG_PORT = 9100
_APP_STATE_FOREGROUND = 4  # XCUITest query_app_state() value for foreground


# ---------------------------------------------------------------------------
# Driver helpers
# ---------------------------------------------------------------------------

def connect_driver() -> tuple[webdriver.Remote, str]:
    """Connect to Appium, reusing True Skate if it is already in the foreground.

    Uses no_reset=True so Appium never stops or reinstalls the app.
    After connecting, queries the app state:
      - Already in foreground (state 4): proceed without touching it.
      - Otherwise: activate it and wait briefly for the UI to settle.

    Reads IPHONE_UDID from the environment (via .env at the project root).

    Returns:
        (driver, mjpeg_url) — Appium WebDriver and WDA MJPEG stream URL.
    """
    _repo_root = Path(__file__).resolve().parents[3]
    load_dotenv(_repo_root / ".env")
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
    options.set_capability("mjpegServerPort", _MJPEG_PORT)

    driver = webdriver.Remote("http://127.0.0.1:4723", options=options)

    state = driver.query_app_state(_BUNDLE_ID)
    if state == _APP_STATE_FOREGROUND:
        print("True Skate is already open — reusing.")
    else:
        print(f"True Skate not in foreground (state={state}) — activating.")
        driver.activate_app(_BUNDLE_ID)
        time.sleep(1.5)  # wait for the game UI to settle

    mjpeg_url = f"http://127.0.0.1:{_MJPEG_PORT}"
    return driver, mjpeg_url


def _ensure_foreground(driver: webdriver.Remote) -> bool:
    """Verify True Skate is in the foreground; relaunch it if not.

    Returns True if the app had to be relaunched, False if it was already running.
    """
    state = driver.query_app_state(_BUNDLE_ID)
    if state == _APP_STATE_FOREGROUND:
        return False
    print(f"True Skate not in foreground (state={state}) — relaunching.")
    driver.activate_app(_BUNDLE_ID)
    time.sleep(2.0)  # wait for game UI to settle after relaunch
    return True


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


# ---------------------------------------------------------------------------
# Frame recording
# ---------------------------------------------------------------------------

class FrameRecorder:
    """Reads 210×455 grayscale frames from WDA's MJPEG stream during an eval.

    Connects to the MJPEG server started by WDA, extracts JPEG frames by
    scanning for SOI/EOI markers (0xFF 0xD8 / 0xFF 0xD9), and decodes each
    to a 210×455 grayscale numpy array. Typical throughput: 30–60 fps.
    """

    def __init__(self):
        self._thread: threading.Thread | None = None
        self._stop_flag = False
        self._frames: list[np.ndarray] = []
        self._response: requests.Response | None = None

    def start(self, mjpeg_url: str) -> None:
        self._stop_flag = False
        self._frames = []
        self._response = None
        self._thread = threading.Thread(
            target=self._capture_loop, args=(mjpeg_url,), daemon=True
        )
        self._thread.start()

    def _capture_loop(self, mjpeg_url: str) -> None:
        buf = b""
        try:
            resp = requests.get(mjpeg_url, stream=True, timeout=5)
            self._response = resp
            for chunk in resp.iter_content(chunk_size=4096):
                if self._stop_flag:
                    break
                buf += chunk
                # Extract complete JPEG frames via SOI (0xFF 0xD8) / EOI (0xFF 0xD9) markers
                while True:
                    start_idx = buf.find(b"\xff\xd8")
                    if start_idx == -1:
                        buf = b""
                        break
                    end_idx = buf.find(b"\xff\xd9", start_idx + 2)
                    if end_idx == -1:
                        buf = buf[start_idx:]  # preserve partial frame
                        break
                    jpeg_bytes = buf[start_idx : end_idx + 2]
                    buf = buf[end_idx + 2:]
                    try:
                        img = Image.open(io.BytesIO(jpeg_bytes)).convert("L").resize((210, 455), Image.LANCZOS)
                        self._frames.append(np.array(img, dtype=np.uint8))
                    except Exception:
                        pass
        except Exception:
            pass
        finally:
            self._response = None

    def stop(self) -> list[np.ndarray]:
        self._stop_flag = True
        resp = self._response
        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        return self._frames


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
    driver: webdriver.Remote,
    mjpeg_url: str,
    *,
    max_evals: int = 1800,
    seed: int = 42,
    wait_time: float = 0.0,
    settle_time: float = 0.5,
    pop_size: int = 24,
    log_dir: Path,
) -> None:
    """Execute the CMA-ES optimization loop.

    Args:
        driver:     Connected Appium WebDriver (from connect_driver()).
        mjpeg_url:  WDA MJPEG stream URL (from connect_driver()).
        max_evals:  Total evaluations before stopping. The loop exits cleanly
                    after the generation that crosses this threshold.
        seed:       CMA-ES random seed.
        wait_time:  Seconds to wait after gestures before first OCR screenshot.
        settle_time: Seconds to wait after board reset before next attempt.
        pop_size:   CMA-ES population size (evals per generation).
        log_dir:    Root directory for run logs and frame composites.
    """
    try:
        import cma
    except ImportError:
        raise ImportError(
            "'cma' package not found. Install it with: pip install cma"
        )

    run_dir, log_path, log_fh = _open_log(log_dir)
    print(f"Run folder: {run_dir}")
    print(f"Logging to {log_path}")

    bounds = [PARAM_BOUNDS[:, 0].tolist(), PARAM_BOUNDS[:, 1].tolist()]

    es = cma.CMAEvolutionStrategy(
        INITIAL_MEAN.tolist(),
        1.0,  # overall sigma — per-parameter scaling handled by CMA_stds
        {
            "bounds": bounds,
            "CMA_stds": INITIAL_SIGMA.tolist(),
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
    repetition_penalty = RepetitionPenalty()

    try:
        while eval_num < max_evals:
            solutions = es.ask()
            rewards = []

            for candidate_idx, candidate in enumerate(solutions):
                # Ensure True Skate is still the foreground app before wasting an eval
                relaunched = _ensure_foreground(driver)

                # Wait for board to settle after previous reset (or relaunch)
                time.sleep(settle_time)

                reward = 0.0
                trick_result = None
                repetition_multiplier = 1.0
                recorder = FrameRecorder()
                try:
                    recorder.start(mjpeg_url)

                    execute_action(driver, np.array(candidate))

                    reward, trick_result, repetition_multiplier = get_reward(
                        driver, wait_time=wait_time, penalty=None
                    )
                except Exception as exc:
                    recorder.stop()
                    logging.warning("candidate %d failed: %s", candidate_idx, exc)
                    eval_num += 1
                    print(
                        f"[eval {eval_num}/{max_evals} | gen {generation}] "
                        f"ERROR: {exc} — assigning reward=0.0"
                    )
                    rewards.append(0.0)
                    reset_position(driver)
                    continue

                raw_frames = recorder.stop()
                rewards.append(reward)
                eval_num += 1

                eval_dir_name = f"eval_{eval_num:05d}"
                n_composites = _save_composites(
                    raw_frames, run_dir / "frames" / eval_dir_name
                )

                trick_str = trick_result.trick if trick_result else None
                trick_status = trick_result.status if trick_result else None

                print(
                    f"[eval {eval_num}/{max_evals} | gen {generation}] "
                    f"reward={reward:.2f}  trick={trick_str}  status={trick_status}  "
                    f"raw_frames={len(raw_frames)} composites={n_composites}"
                )

                _write_log(log_fh, {
                    "generation": generation,
                    "candidate_idx": candidate_idx,
                    "eval_num": eval_num,
                    "reward": reward,
                    "repetition_multiplier": round(repetition_multiplier, 4),
                    "trick_name": trick_str,
                    "trick_status": trick_status,
                    "params": [round(float(p), 2) for p in candidate],
                    "frame_dir": eval_dir_name,
                    "n_composites": n_composites,
                    "app_relaunched": relaunched,
                    "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                })

                if reward > best_reward:
                    best_reward = reward
                    best_trick = trick_str
                    best_params = np.array(candidate)

                reset_position(driver)

            # Feed negated rewards to CMA-ES (it minimizes)
            es.tell(solutions, [-r for r in rewards])
            es.disp()

            # CMA-ES convergence is informational only — the sparse reward
            # landscape looks flat to CMA-ES long before we've found the trick.
            stop_conditions = es.stop()
            if stop_conditions:
                print(f"WARNING: CMA-ES convergence condition(s) fired (continuing): {stop_conditions}")

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
