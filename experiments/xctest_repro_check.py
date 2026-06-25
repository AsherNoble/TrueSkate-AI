"""Reproducibility + alignment check for the XCTest 30fps capture path.

Validates the (frame <- gesture) alignment that ``vision/xctest_capture`` enables:
run an identical, fixed gesture sequence TWICE (each gesture from a fresh board
reset), record each run as one XCTest .mov, then for each gesture extract the
frame at its aligned video-time (``gesture_epoch_s - startedAt + lookahead``) from
BOTH runs and compare them.

  match  (low inter-run pixel diff)  -> capture+alignment is reproducible AND the
                                        game is deterministic from a reset.
  mismatch (high diff)               -> alignment drift OR game non-determinism;
                                        inspect the dumped frames in --out.

This is the harness behind the "run gestures, re-run the same gestures, do the
frames match?" experiment. It deliberately resets before EVERY gesture so each
comparison is an independent (reset -> gesture -> response) trial, isolating the
pipeline from cumulative physics divergence.

Run on the rig (WDA+Appium up, Appium started with --allow-insecure
xcuitest:xctest_screen_record):
    PYTHONPATH=src .venv/bin/python experiments/xctest_repro_check.py --out tmp/repro
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
from trueskate_ai.sim.gestures import scale_to_device  # noqa: E402
from trueskate_ai.sim.touch_actions import curved_drag, reset_position  # noqa: E402
from trueskate_ai.vision.xctest_capture import XCTestScreenRecorder  # noqa: E402

# Fixed, varied curved flicks in normalised coords (curved drags — straight swipes
# don't reflect real play). Each is (waypoints, duration_s, label).
GESTURES = [
    ([(0.5, 0.78), (0.46, 0.45), (0.55, 0.2)], 0.18, "ollie-ish"),
    ([(0.5, 0.78), (0.3, 0.5), (0.62, 0.22)], 0.22, "kickflip-ish"),
    ([(0.5, 0.78), (0.7, 0.5), (0.4, 0.22)], 0.22, "heelflip-ish"),
    ([(0.5, 0.8), (0.5, 0.3)], 0.16, "straight-pop"),
    ([(0.45, 0.8), (0.6, 0.4), (0.35, 0.25)], 0.24, "shove-ish"),
]
LOOKAHEADS = (0.20, 0.50, 0.80)  # seconds after gesture call to sample the response


def extract_frame(mov: Path, video_time_s: float, out_png: Path) -> np.ndarray | None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    # Output-seek (-ss after -i) = frame-accurate. Segments are short so this is fine.
    r = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(mov), "-ss", f"{max(0.0, video_time_s):.3f}",
         "-frames:v", "1", str(out_png)],
        capture_output=True, text=True,
    )
    if r.returncode != 0 or not out_png.exists():
        return None
    return np.asarray(Image.open(out_png).convert("L"), dtype=np.float32)


def run_sequence(label: str, rec: XCTestScreenRecorder, worker: DeviceWorker, out_dir: Path):
    d, dw, dh = worker.driver, worker.device_w, worker.device_h
    rec.start()
    events = []
    for k, (wp, dur, name) in enumerate(GESTURES):
        reset_position(d, dw, dh)
        time.sleep(1.0)  # let the board settle into the fixed reset state
        t_call = time.time()
        pts = [scale_to_device(x, y, dw, dh) for x, y in wp]
        curved_drag(d, pts, total_duration=dur)
        events.append({"k": k, "name": name, "t_call_epoch_s": t_call})
        time.sleep(1.0)  # let the trick play out before the next reset
    res = rec.stop_and_save(out_dir / f"{label}.mov")
    print(f"[{label}] {res.summary()}")
    return res, events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", default="iPhone_XR")
    ap.add_argument("--out", default="tmp/repro")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = resolve_devices(devices_arg=args.devices)[0]
    w = DeviceWorker(cfg)
    print(f"connecting to {cfg['name']}...")
    w.connect()
    rec = XCTestScreenRecorder(w.driver, fps=args.fps)

    resA, evA = run_sequence("runA", rec, w, out)
    resB, evB = run_sequence("runB", rec, w, out)

    print("\n=== per-gesture inter-run frame comparison (mean |A-B| over 0..255) ===")
    overall = []
    for (a, b) in zip(evA, evB):
        diffs = []
        for la in LOOKAHEADS:
            va = resA.video_time_for(a["t_call_epoch_s"], la)
            vb = resB.video_time_for(b["t_call_epoch_s"], la)
            fa = extract_frame(resA.mov_path, va, out / f"A_{a['k']}_{int(la*100)}.png")
            fb = extract_frame(resB.mov_path, vb, out / f"B_{b['k']}_{int(la*100)}.png")
            if fa is None or fb is None or fa.shape != fb.shape:
                diffs.append(float("nan"))
                continue
            diffs.append(float(np.mean(np.abs(fa - fb))))
        md = np.nanmean(diffs) if any(not np.isnan(x) for x in diffs) else float("nan")
        overall.append(md)
        pretty = " ".join(f"{la*1000:.0f}ms={d:.1f}" for la, d in zip(LOOKAHEADS, diffs))
        print(f"  g{a['k']} {a['name']:<14} mean={md:5.1f}  [{pretty}]")
    valid = [x for x in overall if not np.isnan(x)]
    grand = float(np.mean(valid)) if valid else float("nan")
    print(f"\nGRAND mean inter-run diff = {grand:.1f} / 255")
    # heuristic verdict; the user inspects the dumped frames for borderline cases
    verdict = "MATCH (reproducible)" if grand < 8 else (
        "CLOSE (minor noise/non-determinism)" if grand < 20 else "MISMATCH — investigate")
    print(f"VERDICT: {verdict}")
    print(f"frames + .movs dumped under: {out}")


if __name__ == "__main__":
    main()
