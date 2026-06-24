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


def _consec_fracs(grays: list[np.ndarray], change_delta: float) -> list[float]:
    """Frame-to-frame changed-pixel fractions (motion *rate*, not displacement)."""
    return [float(np.mean(np.abs(grays[i] - grays[i - 1]) > change_delta))
            for i in range(1, len(grays))]


def wait_for_quiet(rec, *, quiet_s: float = 0.35, max_wait_s: float = 4.0,
                   change_delta: float = 15.0, frac_thresh: float = 0.006,
                   poll_s: float = 0.05) -> bool:
    """Block until the DAL stream shows no frame-to-frame motion for ``quiet_s``.

    The board must be FULLY at rest before the measured event so BOTH pipelines
    share a clean, identical baseline (a fixed post-push sleep leaves the board
    still settling — DAL then sees motion before the event, corrupting the gap).
    Polls the live DAL buffer's consecutive-frame change rate; quiet when every
    recent inter-frame frac is below ``frac_thresh`` across a ``quiet_s`` span.
    Returns False on timeout (caller proceeds anyway, but the reset may be noisy).
    """
    deadline = time.monotonic() + max_wait_s
    while time.monotonic() < deadline:
        now = time.monotonic()
        frames, times = rec.window(now - (quiet_s + 0.2), now)
        if len(frames) >= 3 and (times[-1] - times[0]) >= quiet_s:
            grays = [_to_gray(f) for f in frames]
            if max(_consec_fracs(grays, change_delta)) < frac_thresh:
                return True
        time.sleep(poll_s)
    return False


def motion_onset(frames: list, times: list[float], t_event: float, *,
                 change_delta: float = 15.0, margin: float = 0.04,
                 search_guard: float = 0.05) -> float | None:
    """Sub-frame time at which the push motion rises ``margin`` above this pipeline's
    OWN pre-event baseline, in the shared monotonic clock.

    Measures the fraction of pixels that moved > ``change_delta`` gray levels from the
    first (settled) frame, then thresholds at ``median(pre-event fracs) + margin`` —
    PER PIPELINE. This matters because the two pipelines have different static floors:
    WDA-MJPEG is ~0, but the DAL USB capture has a slow self-noise drift up to ~0.04
    (vs frame-0; its frame-to-frame diff stays quiet, so the board really is settled —
    verified XR1 tmp/noise_diag). A single ABSOLUTE threshold would make DAL cross
    earlier purely from its higher floor, biasing the gap; a baseline-RELATIVE
    threshold cancels that. The crossing is linearly interpolated between the
    straddling frames for sub-frame timing (so DAL's 30fps and MJPEG's change-driven
    rate stay comparable), and only frames at/after ``t_event - search_guard`` are
    considered so a pre-event DAL noise spike can't masquerade as the onset.

    Returns None if there are too few frames or no pre-event baseline frame (so the
    sample is skipped rather than corrupting the median)."""
    if len(frames) < 4:
        return None
    grays = [_to_gray(f) for f in frames]
    base = grays[0]
    fracs = [float(np.mean(np.abs(g - base) > change_delta)) for g in grays]
    pre = [fr for t, fr in zip(times, fracs) if t < t_event]
    if not pre:
        return None
    thresh = float(np.median(pre)) + margin
    for i in range(1, len(fracs)):
        if times[i] < t_event - search_guard:
            continue
        if fracs[i] > thresh:
            f0, f1, t0, t1 = fracs[i - 1], fracs[i], times[i - 1], times[i]
            if f1 > f0:  # interpolate the crossing for sub-frame timing
                return t0 + (thresh - f0) / (f1 - f0) * (t1 - t0)
            return times[i]
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-pipeline Δ_DAL = Δ_MJPEG + (DAL−MJPEG gap).")
    ap.add_argument("--device", default="iPhone_XR2")
    ap.add_argument("--resets", type=int, default=10, help="Board pushes to measure the gap G")
    ap.add_argument("--warmup", type=int, default=3, help="Discarded warm-up pushes before measuring")
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
    gaps: list[float] = []
    delta_mjpeg = mjpeg_mad = None
    mjpeg_n = 0

    try:
        # ORDER MATTERS: measure the DAL-dependent gap FIRST (DAL live), THEN the
        # Clapperboard. The Clapperboard activate_app is the CMIO-freeze trigger that
        # wedges DAL — fine once we're done reading it, but reversing this order froze
        # DAL and the gap missed every reset (2026-06-22).

        # --- G: DAL−MJPEG capture-latency gap from a shared reset motion (DAL live) ---
        # Shared event = the PUSH impulse. We reset FIRST (recenters AND stops the
        # board — it settles fast), wait for actual rest so both pipelines share a
        # clean baseline, THEN push. (Measuring the reset snap-back instead would
        # need the board settled while DISPLACED, but a pushed board keeps rolling
        # and never goes quiet — wait_for_quiet then times out on every sample.)
        def measure_gap_once() -> tuple[float | None, int, int, bool]:
            try:
                reset_position(worker.driver, dw, dh)  # recenter + stop the board
            except Exception:  # noqa: BLE001
                pass
            settled = wait_for_quiet(rec_dal)
            rec_mjpeg.start(mjpeg_url, resize_width=256)
            t_pre = time.monotonic()
            time.sleep(0.3)  # settled baseline frames in both pipelines
            t_event = time.monotonic()  # push command time (shared monotonic clock)
            try:
                execute_static_push(worker.driver, device_w=dw, device_h=dh)
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.6)  # capture the push motion
            mj_frames, mj_times = rec_mjpeg.stop()
            dal_frames, dal_times = rec_dal.window(t_pre, time.monotonic())
            t_dal = motion_onset(dal_frames, dal_times, t_event)
            t_mj = motion_onset(mj_frames, mj_times, t_event)
            gap = (t_dal - t_mj) if (t_dal is not None and t_mj is not None) else None
            return gap, len(dal_frames), len(mj_frames), settled

        # Warm-up (discarded): the first few cycles read a systematically larger gap
        # (~-0.097 vs steady ~-0.055 on XR1) — the MJPEG stream / board physics settle
        # into steady state after a few iterations. Burn them so the median isn't
        # skewed by a warm-up regime.
        if args.warmup:
            print(f"Warming up ({args.warmup} discarded pushes)...")
            for _ in range(args.warmup):
                measure_gap_once()
                time.sleep(0.2)

        print(f"Measuring DAL−MJPEG gap over {args.resets} pushes...")
        for r in range(args.resets):
            gap, n_dal, n_mj, settled = measure_gap_once()
            if gap is not None:
                gaps.append(gap)
                print(f"  push {r}: gap={gap:+.4f}s (dal_frames={n_dal} "
                      f"mjpeg_frames={n_mj} settled={settled})")
            else:
                print(f"  push {r}: onset missed (dal_frames={n_dal} "
                      f"mjpeg_frames={n_mj} settled={settled}) — skipped")
            time.sleep(0.2)
        rec_dal.close()  # done with DAL — release BEFORE the Clapperboard app-switch

        # --- Δ_MJPEG: Clapperboard over MJPEG (app-switch OK; MJPEG survives it) ---
        print(f"Measuring Δ_MJPEG via clapperboard ({args.k} taps)...")
        est = calibrate_via_app(worker.driver, mjpeg_url, dw, dh, recorder=None, k=args.k)
        worker.ensure_foreground()
        delta_mjpeg, mjpeg_mad, mjpeg_n = est.offset_s, est.jitter_s, est.n
        print(f"  Δ_MJPEG = {delta_mjpeg} (mad {mjpeg_mad}, n {mjpeg_n})")
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
        "delta_mjpeg_s": delta_mjpeg, "delta_mjpeg_mad_s": mjpeg_mad, "delta_mjpeg_n": mjpeg_n,
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
