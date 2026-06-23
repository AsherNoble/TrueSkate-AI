"""Cross-pipeline Δ for the DAL capture — since the Clapperboard is invisible to DAL.

The Clapperboard app renders BLACK under AVFoundation/DAL but flips fine in WDA's
MJPEG (XCTest) path, so we can't clapperboard DAL directly. But:

    Δ_DAL = Δ_MJPEG + (DAL − MJPEG capture-latency gap, G)

Δ_MJPEG is the normal clapperboard over MJPEG (the app IS visible there). G is the
constant latency gap between the two capture pipelines for the SAME physical screen
event — measured here by running BOTH recorders simultaneously across a board RESET
(a True-Skate motion both pipelines see) and differencing the motion-onset frame-time
in each. The onset-detection fuzziness is common to both pipelines, so it cancels in
the difference; averaging over resets tightens G.

Writes Δ_DAL (+ Δ_MJPEG, G, per-reset gaps) to tmp/. Feed the result to the collector
via --capture-offset-s. (Tomorrow's app fix makes DAL clapperboard work directly and
retires this.)

Prereqs: WDA+Appium up, Clapperboard installed, DAL un-wedged, phone unlocked.

Usage:
    python experiments/cross_pipeline_delta.py --device iPhone_XR --resets 10 --k 12
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from trueskate_ai.rl.device_worker import DEVICES, DeviceWorker  # noqa: E402
from trueskate_ai.sim.gestures import execute_static_push  # noqa: E402
from trueskate_ai.sim.touch_actions import reset_position  # noqa: E402
from trueskate_ai.vision.clapperboard import calibrate_via_app  # noqa: E402
from trueskate_ai.vision.color_recorder import TimestampedColorRecorder  # noqa: E402
from trueskate_ai.vision.dal_capture import DalFrameRecorder, resolve_device_name  # noqa: E402


def _to_gray(frame) -> np.ndarray:
    return np.asarray(Image.fromarray(frame).convert("L"), dtype=np.float32)


def motion_onset(frames: list, times: list[float], *, baseline_n: int = 3,
                 k_sigma: float = 5.0, abs_min: float = 2.5) -> float | None:
    """Time of the first frame whose mean-abs-diff from the first (settled) frame
    exceeds the baseline noise floor — i.e. the reset motion onset."""
    if len(frames) < baseline_n + 2:
        return None
    grays = [_to_gray(f) for f in frames]
    base = grays[0]
    diffs = [float(np.mean(np.abs(g - base))) for g in grays]
    seed = diffs[1:baseline_n + 1]
    thresh = max(abs_min, float(np.mean(seed) + k_sigma * (np.std(seed) + 1e-6)))
    for t, d in zip(times[1:], diffs[1:]):
        if d > thresh:
            return t
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-pipeline Δ_DAL = Δ_MJPEG + (DAL−MJPEG gap).")
    ap.add_argument("--device", default="iPhone_XR")
    ap.add_argument("--resets", type=int, default=10, help="Board resets to measure the gap G")
    ap.add_argument("--k", type=int, default=12, help="Clapperboard taps for Δ_MJPEG")
    ap.add_argument("--resize-width", type=int, default=512)
    ap.add_argument("--out", type=Path, default=_REPO_ROOT / "tmp" / "cross_pipeline_delta.json")
    args = ap.parse_args()

    cfg = next((d for d in DEVICES if d["name"].lower() == args.device.lower()), None)
    if cfg is None:
        raise SystemExit(f"Unknown device {args.device}.")
    if not cfg.get("avf_name"):
        raise SystemExit(f"{args.device} has no avf_name in DEVICES.")

    worker = DeviceWorker(cfg)
    print(f"Connecting to {cfg['name']}...")
    worker.connect()
    dw, dh, mjpeg_url = worker.device_w, worker.device_h, worker.mjpeg_url

    avf_name = resolve_device_name(cfg["avf_name"]) or cfg["avf_name"]
    print(f"Opening DAL '{avf_name}' at 30fps...")
    rec_dal = DalFrameRecorder(fps=30, resize_width=args.resize_width)
    rec_dal.open(avf_name)
    if not rec_dal.wait_for_frames(timeout_s=10.0):
        rec_dal.close(); worker.disconnect()
        raise SystemExit(f"No DAL frames (err {rec_dal.last_error!r}). Wedged/locked?")
    rec_mjpeg = TimestampedColorRecorder()

    try:
        # --- Δ_MJPEG: clapperboard over the MJPEG pipeline (app visible there) ---
        print(f"Measuring Δ_MJPEG via clapperboard ({args.k} taps)...")
        est = calibrate_via_app(worker.driver, mjpeg_url, dw, dh, recorder=None, k=args.k)
        worker.ensure_foreground()
        delta_mjpeg = est.offset_s
        print(f"  Δ_MJPEG = {delta_mjpeg} (mad {est.jitter_s}, n {est.n})")

        # --- G: DAL−MJPEG capture-latency gap from a shared reset motion ---
        print(f"Measuring DAL−MJPEG gap over {args.resets} resets...")
        gaps: list[float] = []
        for r in range(args.resets):
            try:
                execute_static_push(worker.driver, device_w=dw, device_h=dh)
                time.sleep(0.7)  # board rolls away + settles
            except Exception:  # noqa: BLE001
                pass
            rec_mjpeg.start(mjpeg_url, resize_width=256)
            t_pre = time.monotonic()
            time.sleep(0.3)  # settled baseline frames in both pipelines
            reset_position(worker.driver, dw, dh)
            time.sleep(0.6)  # capture the snap-back motion
            mj_frames, mj_times = rec_mjpeg.stop()
            dal_frames, dal_times = rec_dal.window(t_pre, time.monotonic())
            t_dal = motion_onset(dal_frames, dal_times)
            t_mj = motion_onset(mj_frames, mj_times)
            if t_dal is not None and t_mj is not None:
                gap = t_dal - t_mj
                gaps.append(gap)
                print(f"  reset {r}: gap={gap:+.4f}s (dal_frames={len(dal_frames)} "
                      f"mjpeg_frames={len(mj_frames)})")
            else:
                print(f"  reset {r}: onset missed (dal={t_dal} mjpeg={t_mj}) — skipped")
            time.sleep(0.2)
    finally:
        rec_dal.close()
        try:
            worker.ensure_foreground()
        except Exception:  # noqa: BLE001
            pass
        worker.disconnect()

    g = statistics.median(gaps) if gaps else None
    g_mad = statistics.median([abs(x - g) for x in gaps]) if len(gaps) > 1 else None
    delta_dal = (delta_mjpeg + g) if (delta_mjpeg is not None and g is not None) else None

    result = {
        "device": cfg["name"], "avf_name": avf_name,
        "delta_mjpeg_s": delta_mjpeg, "delta_mjpeg_mad_s": est.jitter_s, "delta_mjpeg_n": est.n,
        "gap_dal_minus_mjpeg_s": round(g, 4) if g is not None else None,
        "gap_mad_s": round(g_mad, 4) if g_mad is not None else None,
        "gap_n": len(gaps),
        "delta_dal_s": round(delta_dal, 4) if delta_dal is not None else None,
        "gaps": [round(x, 4) for x in gaps],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))

    print("\n=== cross-pipeline Δ ===")
    print(f"  Δ_MJPEG          = {delta_mjpeg} s")
    print(f"  gap (DAL−MJPEG)  = {result['gap_dal_minus_mjpeg_s']} s "
          f"(mad {result['gap_mad_s']}, n {len(gaps)})")
    print(f"  Δ_DAL            = {result['delta_dal_s']} s  ← stamp via --capture-offset-s")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
