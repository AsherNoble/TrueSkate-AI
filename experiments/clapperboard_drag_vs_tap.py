"""Drag-vs-tap Δ check — does a GESTURE's call-return anchor Δ like a TAP's?

The SLS collector anchors every sample's frame-times to the gesture's W3C
``perform()`` call-return, but the clapperboard calibrates Δ with an instantaneous
``mobile: tap``. That carries an ASSUMPTION (journal, label-precision analysis): a
drag's return-to-finger-lift lag equals a tap's return-to-flip lag. This validates
it on-device by measuring Δ both ways through the SAME (DAL) pipeline and comparing.

The Clapperboard app is a full-screen black↔white toggle, so any touch flips it:
- TAP: instantaneous; Δ_tap = flip − tap_call_return  (the shipped calibration).
- DRAG: a curved drag whose finger LIFTS at screen-centre; Δ_drag = flip − drag_call_return.

Interpretation:
- |Δ_drag − Δ_tap| ≲ one frame  → the flip fires at the LIFT; the gesture END is
  anchored exactly like a tap. Assumption holds; intermediate path positions inherit
  it (mid-path softness is then only execution-timing fidelity, a separate term).
- Δ_drag ≈ Δ_tap − duration       → the flip fires at touch-DOWN; the visual onset
  leads the call-return by ~duration. Still consistent with the collector's model
  (gesture spans [Δ − duration, Δ]), but tells us the onset, not the lift, is the
  tight anchor — worth knowing before trusting end-anchored labels.

Runs several batches (alternating tap/drag) for validity, pools the raw Δ samples,
and reports per-batch medians + overall median/MAD. Writes JSON to tmp/.

Prereqs: WDA + Appium up (python scripts/launch_services.py --devices iPhone_XR),
the Clapperboard app installed, the DAL stream NOT wedged, phone unlocked.

Usage:
    python experiments/clapperboard_drag_vs_tap.py --device iPhone_XR --batches 4 --k 10
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from trueskate_ai.rl.device_worker import DEVICES, DeviceWorker  # noqa: E402
from trueskate_ai.sim.touch_actions import curved_drag  # noqa: E402
from trueskate_ai.vision.clapperboard import calibrate_via_app  # noqa: E402
from trueskate_ai.vision.dal_capture import DalFrameRecorder, resolve_device_name  # noqa: E402


def _stats(samples: list[float]) -> dict:
    samples = [s for s in samples if s is not None]
    if not samples:
        return {"n": 0, "median": None, "mad": None}
    med = statistics.median(samples)
    mad = statistics.median([abs(s - med) for s in samples]) if len(samples) > 1 else 0.0
    return {"n": len(samples), "median": round(med, 4), "mad": round(mad, 4)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Drag-vs-tap clapperboard Δ check.")
    ap.add_argument("--device", default="iPhone_XR", help="Device name from DEVICES")
    ap.add_argument("--batches", type=int, default=4, help="Alternating tap/drag rounds")
    ap.add_argument("--k", type=int, default=10, help="Flips per condition per batch")
    ap.add_argument("--drag-duration", type=float, default=0.4, help="Drag gesture seconds")
    ap.add_argument("--pipeline", choices=["mjpeg", "dal"], default="mjpeg",
                    help="Capture pipeline that observes the flip. DAL renders the "
                         "Clapperboard app BLACK (verified), so use mjpeg — the "
                         "drag-vs-tap DIFFERENCE is a WDA-command property and transfers "
                         "across pipelines regardless.")
    ap.add_argument("--resize-width", type=int, default=512)
    ap.add_argument("--out", type=Path, default=_REPO_ROOT / "tmp" / "drag_vs_tap_delta.json")
    args = ap.parse_args()

    cfg = next((d for d in DEVICES if d["name"].lower() == args.device.lower()), None)
    if cfg is None:
        raise SystemExit(f"Unknown device {args.device}. Valid: {[d['name'] for d in DEVICES]}")
    if not cfg.get("avf_name"):
        raise SystemExit(f"{args.device} has no avf_name in DEVICES — set it first.")

    worker = DeviceWorker(cfg)
    print(f"Connecting to {cfg['name']} (needs WDA+Appium up)...")
    worker.connect()
    dw, dh = worker.device_w, worker.device_h

    # rec=DalFrameRecorder for --pipeline dal; None for mjpeg (legacy XCTest path,
    # the only one that can see the Clapperboard app — DAL renders it black).
    rec = None
    capture_source = worker.mjpeg_url
    if args.pipeline == "dal":
        avf_name = resolve_device_name(cfg["avf_name"]) or cfg["avf_name"]
        print(f"Opening DAL capture '{avf_name}' at 30fps...")
        rec = DalFrameRecorder(fps=30, resize_width=args.resize_width)
        rec.open(avf_name)
        capture_source = None
        if not rec.wait_for_frames(timeout_s=10.0):
            rec.close()
            worker.disconnect()
            raise SystemExit(
                f"No DAL frames (err: {rec.last_error!r}). Wedged? QuickTime re-activate / "
                f"replug, and ensure the phone is unlocked, screen on.")
    else:
        print(f"Using WDA MJPEG ({capture_source}) — the clapperboard-visible pipeline.")

    # A curved drag that LIFTS at screen-centre (logical pts, like the tap target).
    def drag_action() -> None:
        pts = [(0.30 * dw, 0.70 * dh), (0.42 * dw, 0.56 * dh), (0.50 * dw, 0.50 * dh)]
        curved_drag(worker.driver, pts, total_duration=args.drag_duration)

    tap_samples: list[float] = []
    drag_samples: list[float] = []
    per_batch = []
    try:
        for b in range(args.batches):
            # Alternate which condition leads each batch to decorrelate ordering effects.
            order = ["tap", "drag"] if b % 2 == 0 else ["drag", "tap"]
            row = {"batch": b}
            for cond in order:
                action = None if cond == "tap" else drag_action
                est = calibrate_via_app(worker.driver, capture_source, dw, dh, recorder=rec,
                                        action_fn=action, k=args.k, return_to_trueskate=False)
                (tap_samples if cond == "tap" else drag_samples).extend(
                    [s for s in est.samples if s is not None])
                row[cond] = {"median": est.offset_s, "mad": est.jitter_s, "n": est.n}
                print(f"  batch {b} {cond:>4}: median={est.offset_s} mad={est.jitter_s} n={est.n}")
            per_batch.append(row)
    finally:
        if rec is not None:
            rec.close()
        try:
            worker.ensure_foreground()
        except Exception:  # noqa: BLE001
            pass
        worker.disconnect()

    tap_overall = _stats(tap_samples)
    drag_overall = _stats(drag_samples)
    delta_diff = None
    if tap_overall["median"] is not None and drag_overall["median"] is not None:
        delta_diff = round(drag_overall["median"] - tap_overall["median"], 4)

    frame_s = 1.0 / 30.0
    verdict = "INCONCLUSIVE (insufficient samples)"
    if delta_diff is not None:
        if abs(delta_diff) <= frame_s:
            verdict = (f"LIFT-ANCHORED: Δ_drag ≈ Δ_tap (diff {delta_diff:+.3f}s ≤ 1 frame "
                       f"{frame_s:.3f}s) — gesture END anchored like a tap; assumption HOLDS.")
        elif abs(delta_diff + args.drag_duration) <= 2 * frame_s:
            verdict = (f"DOWN-ANCHORED: Δ_drag ≈ Δ_tap − duration (diff {delta_diff:+.3f}s ≈ "
                       f"−{args.drag_duration:.3f}s) — flip fires at touch-DOWN; onset leads "
                       f"call-return by ~duration.")
        else:
            verdict = (f"UNEXPECTED: Δ_drag − Δ_tap = {delta_diff:+.3f}s (neither ~0 nor "
                       f"−duration {args.drag_duration:.3f}s) — investigate.")

    result = {
        "device": cfg["name"], "pipeline": args.pipeline,
        "batches": args.batches, "k": args.k, "drag_duration_s": args.drag_duration,
        "tap": tap_overall, "drag": drag_overall,
        "delta_drag_minus_tap_s": delta_diff, "verdict": verdict,
        "per_batch": per_batch,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))

    print("\n=== drag-vs-tap Δ ===")
    print(f"  TAP : {tap_overall}")
    print(f"  DRAG: {drag_overall}")
    print(f"  Δ_drag − Δ_tap = {delta_diff}s")
    print(f"  → {verdict}")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
