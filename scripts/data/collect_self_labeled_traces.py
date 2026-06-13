"""Collect agent self-labeled trace data: (color screen frame -> known touch).

The agent fires diverse calibration drags it knows EXACTLY; True Skate renders
the orange finger-trace; we capture COLOR screen frames timestamped during each
drag. Offline, `vision/self_label.label_frames` turns each frame's timestamp
into the instantaneous ground-truth touch position. This is the corpus that
(a) measures the hand-tuned trace_extractor's true positional error and
(b) trains Model 1 (the learned trace extractor) — the unblocking step for the
vision-grounded sequence leap. See experiments/vision_sequence_leap_journal.md.

IMPORTANT vs. the RL FrameRecorder: that one decodes grayscale 210x455 (no
color for orange-trace detection) and is discarded; this saves full-resolution
COLOR frames with per-frame timestamps.

Prereqs: WDA + Appium up for the target device (python scripts/launch_services.py).

Usage:
    python scripts/data/collect_self_labeled_traces.py --device iPhone_XR \
        --count 300 --out-dir data/self_labeled_traces
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from trueskate_ai.rl.device_worker import DEVICES, DeviceWorker  # noqa: E402
from trueskate_ai.sim.gestures import (  # noqa: E402
    X_BOUND_MAX, X_BOUND_MIN, Y_BOUND_MAX, Y_BOUND_MIN, scale_to_device,
)
from trueskate_ai.sim.touch_actions import curved_drag, reset_position  # noqa: E402


class TimestampedColorRecorder:
    """MJPEG reader that keeps COLOR frames + monotonic capture timestamps."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop = False
        self._frames: list[np.ndarray] = []
        self._times: list[float] = []
        self._resp: requests.Response | None = None

    def start(self, mjpeg_url: str) -> None:
        self._stop = False
        self._frames, self._times = [], []
        self._thread = threading.Thread(target=self._loop, args=(mjpeg_url,), daemon=True)
        self._thread.start()

    def _loop(self, mjpeg_url: str) -> None:
        buf = b""
        try:
            resp = requests.get(mjpeg_url, stream=True, timeout=5)
            self._resp = resp
            for chunk in resp.iter_content(chunk_size=8192):
                if self._stop:
                    break
                buf += chunk
                while True:
                    s = buf.find(b"\xff\xd8")
                    if s == -1:
                        buf = b""
                        break
                    e = buf.find(b"\xff\xd9", s + 2)
                    if e == -1:
                        buf = buf[s:]
                        break
                    jpeg = buf[s:e + 2]
                    buf = buf[e + 2:]
                    try:
                        img = Image.open(io.BytesIO(jpeg)).convert("RGB")
                        self._times.append(time.monotonic())
                        self._frames.append(np.array(img, dtype=np.uint8))
                    except Exception:
                        pass
        except Exception as exc:  # noqa: BLE001
            print(f"  [recorder] MJPEG capture error: {exc}")
        finally:
            self._resp = None

    def stop(self) -> tuple[list[np.ndarray], list[float]]:
        self._stop = True
        if self._resp is not None:
            try:
                self._resp.close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        return self._frames, self._times


# The board sits center-ish (localizer: ~(0.42-0.50, 0.54-0.55)). True Skate only
# renders the orange finger-trace for flicks ON/NEAR the board — arbitrary
# empty-space drags produce no trace and don't move the board (verified
# 2026-06-14). So calibration gestures START on the board and FLICK outward,
# mimicking real trick input, to maximise trace yield while still covering a
# diverse spread of directions/speeds for Model 1 to generalise.
_BOARD_X = (0.38, 0.62)
_BOARD_Y = (0.50, 0.80)


def _sample_gesture(rng: np.random.Generator) -> dict:
    """A board-centered flick: start on the board, flick outward in a random dir."""
    sx = float(rng.uniform(*_BOARD_X))
    sy = float(rng.uniform(*_BOARD_Y))
    ang = float(rng.uniform(0, 2 * np.pi))
    mag = float(rng.uniform(0.18, 0.45))         # flick reach (normalised)
    ex = float(np.clip(sx + mag * np.cos(ang), X_BOUND_MIN, X_BOUND_MAX))
    ey = float(np.clip(sy + mag * np.sin(ang), Y_BOUND_MIN, Y_BOUND_MAX))
    if int(rng.integers(0, 2)) == 0:
        waypoints = [(sx, sy), (ex, ey)]                       # straight flick
    else:
        mx = float(np.clip((sx + ex) / 2 + rng.uniform(-0.08, 0.08), X_BOUND_MIN, X_BOUND_MAX))
        my = float(np.clip((sy + ey) / 2 + rng.uniform(-0.08, 0.08), Y_BOUND_MIN, Y_BOUND_MAX))
        waypoints = [(sx, sy), (mx, my), (ex, ey)]             # curved flick
    duration = float(rng.uniform(0.12, 0.4))     # flicks are fast
    easing_power = float(rng.uniform(0.6, 2.0))
    return {"waypoints": waypoints, "duration": duration, "easing_power": easing_power}


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect agent self-labeled trace data.")
    ap.add_argument("--device", default="iPhone_XR", help="Device name from DEVICES")
    ap.add_argument("--count", type=int, default=200, help="Number of calibration gestures")
    ap.add_argument("--out-dir", type=Path, default=_REPO_ROOT / "data" / "self_labeled_traces")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reset-every", type=int, default=1, help="Reset board every N gestures")
    args = ap.parse_args()

    cfg = next((d for d in DEVICES if d["name"].lower() == args.device.lower()), None)
    if cfg is None:
        raise SystemExit(f"Unknown device {args.device}. Valid: {[d['name'] for d in DEVICES]}")

    worker = DeviceWorker(cfg)
    print(f"Connecting to {cfg['name']} (needs WDA+Appium up; run launch_services.py first)...")
    worker.connect()
    driver, mjpeg_url = worker.driver, worker.mjpeg_url
    dw, dh = worker.device_w, worker.device_h

    session = time.strftime("%Y%m%d_%H%M%S")
    out_root = args.out_dir / f"{cfg['name']}_{session}"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {out_root}")

    rng = np.random.default_rng(args.seed)
    recorder = TimestampedColorRecorder()
    saved = 0
    for i in range(args.count):
        if i % args.reset_every == 0:
            try:
                reset_position(driver, dw, dh)
                time.sleep(0.4)
            except Exception as exc:  # noqa: BLE001
                print(f"  reset failed: {exc}")

        g = _sample_gesture(rng)
        pts_dev = [scale_to_device(x, y, dw, dh) for x, y in g["waypoints"]]
        easing = None if g["easing_power"] == 1.0 else (lambda t, p=g["easing_power"]: t ** p)

        recorder.start(mjpeg_url)
        time.sleep(0.15)  # let the stream warm up so we capture pre-touch frames too
        t0 = time.monotonic()
        try:
            curved_drag(driver, pts_dev, total_duration=g["duration"], easing=easing)
        except Exception as exc:  # noqa: BLE001
            print(f"  gesture {i} failed: {exc}")
            recorder.stop()
            continue
        time.sleep(0.25)  # capture the fading trace tail
        frames, times = recorder.stop()
        if not frames:
            print(f"  gesture {i}: no frames captured")
            continue

        sample_dir = out_root / f"sample_{i:05d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        frame_times = [round(t - t0, 4) for t in times]
        for fi, fr in enumerate(frames):
            Image.fromarray(fr, mode="RGB").save(sample_dir / f"frame_{fi:03d}.png")
        meta = {
            "device": cfg["name"],
            "logical_w": dw,
            "logical_h": dh,
            "waypoints": g["waypoints"],          # normalised [0,1]
            "duration": g["duration"],
            "easing_power": g["easing_power"],
            "spin_active": False,
            "gesture_start_monotonic": t0,
            "frame_times": frame_times,            # relative to gesture start (s)
            "n_frames": len(frames),
        }
        (sample_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        saved += 1
        if i % 10 == 0:
            print(f"  [{i + 1}/{args.count}] {len(frames)} frames, "
                  f"dur={g['duration']:.2f}s, span={frame_times[0]:.2f}..{frame_times[-1]:.2f}s")

    worker.disconnect()
    print(f"\nDone: {saved} samples → {out_root}")
    print("Label them offline with vision/self_label.label_frames (tune latency_s vs. the trace).")


if __name__ == "__main__":
    main()
