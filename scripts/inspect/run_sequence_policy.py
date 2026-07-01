"""Deploy Model 2 on-device: receding-horizon gesture-sequence policy.

Closes the last BC gap (step H): load a trained SequencePolicy, and each decision
grab the screen, predict the next stroke-chunk, and execute it via the same
`execute_gesture_params` device path CMA-ES uses. Predicted strokes are committed
back into the policy's history so the next decision conditions on what was played.

    grab screenshot (BGR) -> runner.observe -> runner.act -> to_param_vector
        -> wait pre_delay_s -> execute_gesture_params -> runner.commit -> repeat

The inference core is `trueskate_ai.bc.infer` (pure torch). This script is the
thin device wrapper.

Usage:
    # dry run — full inference + param-vector decode on synthetic frames, NO device
    python scripts/inspect/run_sequence_policy.py --dry-run --steps 5
    python scripts/inspect/run_sequence_policy.py --dry-run --model notebooks/models/sequence_model.pth

    # on device — needs WDA running (python scripts/launch_services.py)
    python scripts/inspect/run_sequence_policy.py --model notebooks/models/sequence_model.pth --steps 20
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from trueskate_ai.bc.infer import SequencePolicyRunner, load_policy  # noqa: E402
from trueskate_ai.bc.model2 import SequencePolicy, SequencePolicyConfig  # noqa: E402


def _device():
    import torch

    return torch.device("mps" if torch.backends.mps.is_available()
                        else "cuda" if torch.cuda.is_available() else "cpu")


def _fmt(strokes: np.ndarray) -> str:
    s = strokes[0]
    return (f"start=({s[0]:.2f},{s[1]:.2f}) end=({s[4]:.2f},{s[5]:.2f}) "
            f"dur={s[6]:.3f} ease={s[7]:.2f} delay={s[8]:.3f}")


def _load_or_fresh(model_path: Path | None, device):
    """Load a checkpoint, or build a fresh (untrained) policy for a --dry-run."""
    if model_path is not None:
        model, cfg = load_policy(model_path, device)
        print(f"loaded {model_path.name}: n_frames={cfg.n_frames} m_past={cfg.m_past} "
              f"m_out={cfg.m_out} img={cfg.img_h}x{cfg.img_w}")
        return model, cfg
    cfg = SequencePolicyConfig()
    print(f"no --model: fresh UNTRAINED policy (dry-run only)  img={cfg.img_h}x{cfg.img_w}")
    return SequencePolicy(cfg).to(device).eval(), cfg


def _dry_run(args) -> None:
    device = _device()
    model, cfg = _load_or_fresh(args.model, device)
    runner = SequencePolicyRunner(model, cfg, device)
    rng = np.random.default_rng(0)
    k = args.execute_k or cfg.m_out
    print(f"dry-run: {args.steps} decisions, executing {k}/{cfg.m_out} strokes each (synthetic frames)\n")
    for step in range(args.steps):
        # synthetic BGR frame at the device's native-ish portrait size
        runner.observe(rng.integers(0, 255, (455, 210, 3), dtype=np.uint8))
        strokes = runner.act()
        vec, n, pre_delay = runner.to_param_vector(strokes[:k])
        print(f"  step {step + 1}: {_fmt(strokes)}  -> {n}-slot vec (len {len(vec)}), pre_delay={pre_delay:.3f}s")
        runner.commit(strokes[:k])
    print("\nDRY-RUN OK — observe -> act -> decode -> param-vector -> commit all run.")


def _active_devices() -> list[dict]:
    import requests
    from trueskate_ai.rl.device_worker import DEVICES

    alive = []
    for d in DEVICES:
        try:
            requests.get(f"http://localhost:{d['wda_port']}/status", timeout=2).raise_for_status()
            alive.append(d)
        except Exception:
            pass
    return alive


def _on_device(args) -> None:
    import cv2
    from trueskate_ai.rl.cmaes.action_param import execute_gesture_params
    from trueskate_ai.rl.device_worker import DeviceWorker

    if args.model is None:
        sys.exit("ERROR: on-device run needs --model (a trained checkpoint).")
    devices = _active_devices()
    if not devices:
        sys.exit("ERROR: no devices with WDA running. Start: python scripts/launch_services.py")

    device = _device()
    model, cfg = _load_or_fresh(args.model, device)
    k = args.execute_k or cfg.m_out

    dev = devices[0]
    print(f"[{dev['name']}] connecting...")
    worker = DeviceWorker(dev, calibrate_touch_on_connect=False)
    worker.connect()
    runner = SequencePolicyRunner(model, cfg, device)
    try:
        for step in range(args.steps):
            png = worker.driver.get_screenshot_as_png()
            frame = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)  # BGR
            runner.observe(frame)
            strokes = runner.act()
            vec, n, pre_delay = runner.to_param_vector(strokes[:k])
            print(f"[{dev['name']}] step {step + 1}: {_fmt(strokes)}  pre_delay={pre_delay:.3f}s")
            if pre_delay > 0:
                time.sleep(min(pre_delay, 1.0))                     # honour predicted inter-stroke wait (capped)
            execute_gesture_params(worker.driver, np.asarray(vec, dtype=np.float64),
                                   worker.device_w, worker.device_h)
            runner.commit(strokes[:k])
    finally:
        worker.disconnect()
    print(f"[{dev['name']}] done: {args.steps} decisions.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Deploy Model 2 (gesture-sequence policy) on-device.")
    ap.add_argument("--model", type=Path, default=None, help="Trained Model 2 checkpoint.")
    ap.add_argument("--steps", type=int, default=10, help="Number of decisions to run.")
    ap.add_argument("--execute-k", type=int, default=None,
                    help="Strokes to execute per decision (default: all m_out). Receding-horizon uses < m_out.")
    ap.add_argument("--dry-run", action="store_true", help="No device; synthetic frames prove the inference path.")
    args = ap.parse_args()

    (_dry_run if args.dry_run else _on_device)(args)


if __name__ == "__main__":
    main()
