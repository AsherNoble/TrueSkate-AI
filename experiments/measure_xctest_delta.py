"""Measure the command->pixel offset Δ for the XCTest 30fps capture path.

Δ centres gesture labels: the collector maps a gesture whose call returned at host
``te`` to video PTS ``(te - started_at) + Δ``. Δ is ~constant per session — measure
once, pass as ``--capture-offset-s`` to ``collect_sls_xctest.py``.

Method — the codebase's NO-APP reset timing (``clapperboard.offset_from_reset``),
since the Clapperboard app's code-signature is expired on this rig (won't launch).
In True Skate (no app-switch): push the board away, let it stop, record a reset, and
time the board's snap-home motion onset in the .mov. We log the reset's call ISSUE and
RETURN; ``offset_from_reset`` gives ``onset - issue``, and Δ (relative to call-RETURN,
matching the collector) is ``(onset - issue) - (return - issue) = onset - return``.

    PYTHONPATH=src .venv/bin/python experiments/measure_xctest_delta.py --devices iPhone_XR
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from trueskate_ai.rl.device_worker import DeviceWorker, resolve_devices  # noqa: E402
from trueskate_ai.sim.gestures import execute_static_push  # noqa: E402
from trueskate_ai.sim.touch_actions import reset_position  # noqa: E402
from trueskate_ai.vision.xctest_capture import XCTestScreenRecorder  # noqa: E402


def _motion_onset(frames, times, issue_v, return_v, thresh=6.0):
    """Δ (onset - return) of the reset's snap-back: first post-issue frame whose
    centre-crop differs from the last settled pre-issue frame by > thresh (mean |Δ|
    over 0..255). No board CV — just localized motion. None if not measurable."""
    pre = [(t, f) for t, f in zip(times, frames) if t < issue_v]
    post = [(t, f) for t, f in zip(times, frames) if issue_v <= t <= issue_v + 1.2]
    if len(pre) < 2 or len(post) < 3:
        return None
    base = pre[-1][1].astype(np.float32)
    for t, f in post:
        if float(np.mean(np.abs(f.astype(np.float32) - base))) > thresh:
            return float(t - return_v)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure XCTest capture Δ via board-reset timing.")
    ap.add_argument("--devices", default="iPhone_XR")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--mov", default="/tmp/delta_xctest.mov")
    args = ap.parse_args()

    cfg = resolve_devices(devices_arg=args.devices)[0]
    w = DeviceWorker(cfg)
    print(f"connecting to {cfg['name']}...")
    w.connect()
    d, dw, dh = w.driver, w.device_w, w.device_h

    rec = XCTestScreenRecorder(d, fps=args.fps)
    rec.start()
    cycles = []  # (issue_epoch, return_epoch)
    for i in range(args.k):
        try:
            execute_static_push(d, device_w=dw, device_h=dh)  # roll board AWAY (push, not flick)
        except Exception as exc:  # noqa: BLE001
            print(f"  push {i} err: {exc}")
        time.sleep(0.8)  # board must fully STOP (stable displaced baseline)
        t_issue = time.time()
        try:
            reset_position(d, dw, dh)  # snap home -> visible motion
        except Exception as exc:  # noqa: BLE001
            print(f"  reset {i} err: {exc}")
            continue
        t_return = time.time()
        cycles.append((t_issue, t_return))
        time.sleep(1.0)  # response window
    res = rec.stop_and_save(args.mov)
    w.disconnect()
    started = res.started_at_epoch_s
    print(f"recorded {res.summary()['mb']}MB, started_at={started:.3f}, cycles={len(cycles)}")

    deltas = []
    for i, (t_issue, t_return) in enumerate(cycles):
        issue_v = t_issue - started
        return_v = t_return - started
        start = max(0.0, issue_v - 0.5)
        tmp = Path(f"/tmp/delta_xc_{i}")
        tmp.mkdir(exist_ok=True)
        for old in tmp.glob("f_*.png"):
            old.unlink()
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-i", args.mov,
             "-t", "2.0", "-vf", f"fps={args.fps},scale=512:-2", "-vsync", "0",
             str(tmp / "f_%03d.png")],
            capture_output=True, text=True,
        )
        files = sorted(tmp.glob("f_*.png"))
        frames = []
        for f in files:
            im = np.asarray(Image.open(f).convert("L"), dtype=np.uint8)
            h, wd = im.shape
            frames.append(im[int(0.45 * h):int(0.88 * h), int(0.25 * wd):int(0.75 * wd)])  # board region
        times = [start + j / args.fps for j in range(len(frames))]
        d_return = _motion_onset(frames, times, issue_v, return_v)
        if d_return is None:
            print(f"  cycle {i:2d}: unmeasurable (no motion onset detected)")
            continue
        deltas.append(d_return)
        print(f"  cycle {i:2d}: swipe={return_v - issue_v:.3f}s  Δ(onset-return)={d_return:+.4f}s")

    if not deltas:
        print("\nNO measurable resets — board CV failed (not in skate view?). Δ unmeasured.")
        raise SystemExit(1)
    arr = np.array(deltas)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    print(f"\nΔ over {len(arr)}/{args.k} resets: median={med:.4f}s  MAD={mad:.4f}s  "
          f"mean={arr.mean():.4f}s  std={arr.std():.4f}s")
    print(f"\n==> pass to the collector:  --capture-offset-s {med:.4f}")


if __name__ == "__main__":
    main()
