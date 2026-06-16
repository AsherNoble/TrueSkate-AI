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

from trueskate_ai.data.gesture_sampling import sample_flick  # noqa: E402
from trueskate_ai.rl.device_worker import DEVICES, DeviceWorker  # noqa: E402
from trueskate_ai.sim.gestures import scale_to_device  # noqa: E402
from trueskate_ai.sim.touch_actions import curved_drag, reset_position  # noqa: E402
from trueskate_ai.vision.color_recorder import TimestampedColorRecorder  # noqa: E402

# Park selection (collect domain-diverse data so Model 1 generalises across the
# parks the expert clips span). Nav: bottom SKATEPARKS tab -> "All" tab -> the
# park row. Row y-positions are for the three INSTALLED parks in their list order
# (position-dependent; re-check with a screenshot if the installed set changes).
_MENU_SKATEPARKS = (0.30, 0.95)
_TAB_ALL = (0.10, 0.16)
_TAB_SLS = (0.55, 0.16)
# park key -> (tab xy, row y-fraction). Row positions are list-order dependent —
# re-check with a screenshot if the installed set changes.
_PARK_NAV = {
    "glasshouse": (_TAB_ALL, 0.34),
    "workshop": (_TAB_ALL, 0.55),
    "underpass": (_TAB_ALL, 0.78),
    "sls_supercrown": (_TAB_SLS, 0.30),
    "sls_newark": (_TAB_SLS, 0.50),
    "sls_munich": (_TAB_SLS, 0.70),
}


def switch_to_park(driver, dw, dh, park: str) -> None:
    """Navigate True Skate's menus to load the named installed park."""
    key = park.strip().lower().replace(" ", "").replace(":", "")
    if key not in _PARK_NAV:
        raise ValueError(f"Unknown park {park!r}. Known: {list(_PARK_NAV)}")
    (tab_x, tab_y), row_y = _PARK_NAV[key]
    driver.execute_script("mobile: tap", {"x": _MENU_SKATEPARKS[0] * dw, "y": _MENU_SKATEPARKS[1] * dh})
    time.sleep(1.2)
    driver.execute_script("mobile: tap", {"x": tab_x * dw, "y": tab_y * dh})
    time.sleep(0.9)
    driver.execute_script("mobile: tap", {"x": 0.5 * dw, "y": row_y * dh})
    time.sleep(3.0)  # park load


# The random board-centered flick sampler lives in
# trueskate_ai.data.gesture_sampling (shared with the SLS collector) — imported
# above as sample_flick.


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect agent self-labeled trace data.")
    ap.add_argument("--device", default="iPhone_XR", help="Device name from DEVICES")
    ap.add_argument("--count", type=int, default=200, help="Number of calibration gestures")
    ap.add_argument("--out-dir", type=Path, default=_REPO_ROOT / "data" / "self_labeled_traces")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reset-every", type=int, default=1, help="Reset board every N gestures")
    ap.add_argument("--park", default=None,
                    help="Switch to this installed park first (glasshouse/workshop/underpass)")
    ap.add_argument("--tag", default=None,
                    help="Label the output dir _<tag> WITHOUT navigating (select the park manually "
                         "first, e.g. an SLS arena). Used by the tonight runner's park auto-include.")
    args = ap.parse_args()

    cfg = next((d for d in DEVICES if d["name"].lower() == args.device.lower()), None)
    if cfg is None:
        raise SystemExit(f"Unknown device {args.device}. Valid: {[d['name'] for d in DEVICES]}")

    worker = DeviceWorker(cfg)
    print(f"Connecting to {cfg['name']} (needs WDA+Appium up; run launch_services.py first)...")
    worker.connect()
    driver, mjpeg_url = worker.driver, worker.mjpeg_url
    dw, dh = worker.device_w, worker.device_h

    if args.park:
        print(f"Switching to park: {args.park}")
        switch_to_park(driver, dw, dh, args.park)

    session = time.strftime("%Y%m%d_%H%M%S")
    tag = (args.park or args.tag or "").lower().replace(" ", "")
    park_tag = f"_{tag}" if tag else ""
    out_root = args.out_dir / f"{cfg['name']}_{session}{park_tag}"
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

        g = sample_flick(rng)
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
