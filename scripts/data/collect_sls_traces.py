"""Long-running, multi-park RANDOM-GESTURE trace collector for SLS skateparks.

Purpose: build the (frame -> known-gesture) corpus the sequence model needs from
the VISUALLY DIVERSE expert SLS arenas. Unlike CMA-ES trick mining (which needs
the single-reset flatground park so obstacles/varied resets don't contaminate
params), this loop just wants frame/gesture pairs — so it is OBSTACLE-TOLERANT:
it records the executed gesture as the label regardless of whether the board
lands, whiffs, or bumps a wall, and varied board positions are desirable.

Per phone, it collects in one park for --per-park-hours, then ntfy-prompts you to
walk over and load the next SLS park; it KEEPS collecting in the current park
until it detects you actually switched (no idle time if you're away), then resets
the timer and advances. The command->frame pipeline latency is calibrated
continuously off the natural per-iteration reset (the "clapperboard").

Prereqs: WDA + Appium up for the device (python scripts/launch_services.py --personal).

Usage:
    python scripts/data/collect_sls_traces.py --device iPhone_11 --per-park-hours 4
"""
from __future__ import annotations

import argparse
import json
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from trueskate_ai.data.gesture_sampling import load_recipe_vectors, sample_mixture  # noqa: E402
from trueskate_ai.rl.cmaes.action_param import execute_gesture_params  # noqa: E402
from trueskate_ai.rl.device_worker import DEVICES, DeviceWorker  # noqa: E402
from trueskate_ai.sim.gestures import scale_to_device  # noqa: E402
from trueskate_ai.sim.touch_actions import curved_drag, reset_position  # noqa: E402
from trueskate_ai.utils.notify import notify  # noqa: E402
from trueskate_ai.vision.clapperboard import RollingCaptureOffset, board_centroid  # noqa: E402
from trueskate_ai.vision.color_recorder import TimestampedColorRecorder  # noqa: E402
from trueskate_ai.vision.park_change import ParkChangeDetector  # noqa: E402

# The 11 installed SLS arenas, in cycle order. Switching is MANUAL (you load the
# park), so these are labels for prompting + tagging — no brittle menu nav.
DEFAULT_SLS_PARKS = [
    "SLS 2016 Super Crown",
    "SLS 2015 Super Crown",
    "SLS 2015 New Jersey",
    "SLS 2015 Los Angeles",
    "SLS 2013 Super Crown",
    "SLS 2015 Paris",
    "SLS 2016 Newark",
    "SLS 2016 Munich",
    "SLS 2014 Los Angeles",
    "SLS 2013 Portland",
    "SLS 2013 Kansas City",
]

_STOP = False


def _on_sigint(_signum, _frame):
    global _STOP
    _STOP = True
    print("\n[collect_sls] SIGINT — finishing current sample and shutting down...")


def _park_tag(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _start_caffeinate():
    """Keep the Mac awake for the duration; auto-exits when this PID dies."""
    try:
        import os
        p = subprocess.Popen(["caffeinate", "-dimsu", "-w", str(os.getpid())])
        print(f"[collect_sls] caffeinate active (PID {p.pid}).")
        return p
    except FileNotFoundError:
        print("[collect_sls] caffeinate not found — Mac may sleep. (macOS only.)")
        return None


def _capture(rec: TimestampedColorRecorder, mjpeg_url: str, pre_s: float, hold_s: float,
             resize_width: int, action=None):
    """Record a window: warm up pre_s, mark t0, run `action` (if any), hold hold_s.

    Returns (frames, times, t0) where t0 is the monotonic time `action` started.
    """
    rec.start(mjpeg_url, resize_width=resize_width)
    if pre_s > 0:
        time.sleep(pre_s)
    t0 = time.monotonic()
    if action is not None:
        action()
    if hold_s > 0:
        time.sleep(hold_s)
    frames, times = rec.stop()
    return frames, times, t0


def _downsample(items: list, max_n: int) -> list[int]:
    """Evenly-spaced indices selecting at most max_n of len(items)."""
    n = len(items)
    if n <= max_n:
        return list(range(n))
    return [int(round(i * (n - 1) / (max_n - 1))) for i in range(max_n)]


def _execute(worker: DeviceWorker, g) -> None:
    dw, dh = worker.device_w, worker.device_h
    if g.kind == "flick":
        pts = [scale_to_device(x, y, dw, dh) for x, y in g.waypoints]
        easing = None if g.easing_power == 1.0 else (lambda t, p=g.easing_power: t ** p)
        curved_drag(worker.driver, pts, total_duration=g.duration, easing=easing)
    else:  # nslot / recipe — execute_gesture_params pushes first, like a real trick fire
        spin_xy = worker.spin_button_xy if g.use_spin else None
        execute_gesture_params(
            worker.driver, np.asarray(g.params, dtype=np.float64), dw, dh,
            num_gestures=g.num_gestures, use_spin=g.use_spin,
            spin_button_xy=spin_xy, timing_device_key=worker.device_id,
        )


def _save_sample(sample_dir: Path, frames: list, times: list[float], t_gesture: float,
                 g, park: str, park_idx: int, offset: RollingCaptureOffset,
                 dw: float, dh: float, max_frames: int) -> int:
    """Save the gesture-window frames (at/after t_gesture) + meta. Returns n saved."""
    gest = [(t, f) for t, f in zip(times, frames) if t >= t_gesture]
    if not gest:
        return 0
    keep = _downsample(gest, max_frames)
    sample_dir.mkdir(parents=True, exist_ok=True)
    frame_times = []
    for out_i, src_i in enumerate(keep):
        t, fr = gest[src_i]
        Image.fromarray(fr, mode="RGB").save(sample_dir / f"frame_{out_i:03d}.png")
        frame_times.append(round(t - t_gesture, 4))
    meta = {
        "device_logical_w": dw,
        "device_logical_h": dh,
        "park": park,
        "park_change_index": park_idx,
        "spin_active": bool(g.use_spin) if g.kind != "flick" else False,
        "gesture_start_monotonic": t_gesture,
        "frame_times": frame_times,        # relative to gesture start (s)
        "n_frames": len(frame_times),
        # frame at frame_time reflects screen state at frame_time - capture_offset_s
        "capture_offset_s": offset.offset_s,
        "capture_offset_jitter_s": offset.jitter_s,
        **g.meta(),
    }
    (sample_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return len(frame_times)


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-park random-gesture SLS trace collector.")
    ap.add_argument("--device", default="iPhone_11", help="Device name from DEVICES")
    ap.add_argument("--out-dir", type=Path, default=_REPO_ROOT / "data" / "sls_traces")
    ap.add_argument("--per-park-hours", type=float, default=4.0,
                    help="Collect this long in a park before prompting to switch (keeps "
                         "collecting until you actually switch).")
    ap.add_argument("--start-park", default=None,
                    help="Park name or index to start the cycle at (stagger across phones).")
    ap.add_argument("--max-hours", type=float, default=None, help="Global wall-clock cap.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--flick-frac", type=float, default=0.6)
    ap.add_argument("--nslot-frac", type=float, default=0.25)
    ap.add_argument("--recipe-frac", type=float, default=0.15)
    ap.add_argument("--num-gestures", type=int, default=2)
    ap.add_argument("--use-spin", action="store_true")
    ap.add_argument("--resize-width", type=int, default=512)
    ap.add_argument("--max-frames-per-sample", type=int, default=24)
    ap.add_argument("--recipe-dir", type=Path, default=_REPO_ROOT / "trick_libraries")
    ap.add_argument("--warmup-resets", type=int, default=8, help="Clapperboard seeding resets.")
    ap.add_argument("--no-caffeinate", action="store_true")
    ap.add_argument("--no-park-detect", action="store_true",
                    help="Single-park mode: never auto-advance (still prompts on the timer).")
    args = ap.parse_args()

    cfg = next((d for d in DEVICES if d["name"].lower() == args.device.lower()), None)
    if cfg is None:
        raise SystemExit(f"Unknown device {args.device}. Valid: {[d['name'] for d in DEVICES]}")

    # Park cycle, rotated to the requested start.
    cycle = list(DEFAULT_SLS_PARKS)
    if args.start_park is not None:
        if args.start_park.isdigit():
            start = int(args.start_park) % len(cycle)
        else:
            start = next((i for i, p in enumerate(cycle)
                          if _park_tag(p) == _park_tag(args.start_park)), 0)
        cycle = cycle[start:] + cycle[:start]

    signal.signal(signal.SIGINT, _on_sigint)
    caffeinate = None if args.no_caffeinate else _start_caffeinate()

    worker = DeviceWorker(cfg)
    print(f"Connecting to {cfg['name']} (needs WDA+Appium up; run launch_services.py --personal)...")
    worker.connect()
    dw, dh, mjpeg_url, device = worker.device_w, worker.device_h, worker.mjpeg_url, cfg["name"]

    session = time.strftime("%Y%m%d_%H%M%S")
    out_root = args.out_dir / f"{device}_{session}"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {out_root}; park cycle starts at: {cycle[0]}")

    rng = np.random.default_rng(args.seed)
    recipe_vectors = load_recipe_vectors(args.recipe_dir)
    print(f"Loaded {len(recipe_vectors)} packable recipes for the perturbed-recipe share.")
    fracs = (args.flick_frac, args.nslot_frac, args.recipe_frac)

    rec = TimestampedColorRecorder()
    offset = RollingCaptureOffset(window=max(30, args.warmup_resets))
    detector = None if args.no_park_detect else ParkChangeDetector()

    def write_session_meta(saved: int, park_idx: int) -> None:
        (out_root / "session_meta.json").write_text(json.dumps({
            "device": device, "session": session, "logical_w": dw, "logical_h": dh,
            "park_cycle": cycle, "per_park_hours": args.per_park_hours,
            "fracs": {"flick": fracs[0], "nslot": fracs[1], "recipe": fracs[2]},
            "num_gestures": args.num_gestures, "use_spin": args.use_spin,
            "samples_saved": saved, "current_park_index": park_idx,
            **offset.summary(),
        }, indent=2))

    # --- clapperboard warmup: seed Δ before collecting -----------------------
    print(f"Calibrating capture offset ({args.warmup_resets} reset cycles)...")
    for _ in range(args.warmup_resets):
        if _STOP:
            break
        g = sample_mixture(rng, fracs=(1, 0, 0))            # a flick to displace the board
        frames, times, _ = _capture(rec, mjpeg_url, 0.0, 0.3, args.resize_width,
                                    action=lambda: _execute(worker, g))
        frames, times, t_reset = _capture(rec, mjpeg_url, 0.35, 0.9, args.resize_width,
                                           action=lambda: reset_position(worker.driver, dw, dh))
        offset.add_reset(frames, times, t_reset)
    print(f"  capture offset: {offset.summary()}")
    notify(f"[{device}] SLS collection starting in {cycle[0]} "
           f"(offset={offset.offset_s}s). Cycle of {len(cycle)} parks, {args.per_park_hours}h each.",
           title="TrueSkate SLS collect", tags=["camera"])

    park_idx = 0
    deadline = time.monotonic() + args.per_park_hours * 3600.0
    prompted = False
    saved = 0
    i = 0
    global_deadline = (time.monotonic() + args.max_hours * 3600.0) if args.max_hours else None

    try:
        while not _STOP:
            if global_deadline and time.monotonic() > global_deadline:
                print("[collect_sls] global --max-hours reached.")
                break

            # Park reload in progress → don't fight the user's menu nav; just watch.
            if detector is not None and detector.in_transition:
                frames, times, _ = _capture(rec, mjpeg_url, 0.0, 0.6, args.resize_width)
                if detector.feed(frames):
                    park_idx = (park_idx + 1) % len(cycle)
                    deadline = time.monotonic() + args.per_park_hours * 3600.0
                    prompted = False
                    notify(f"[{device}] detected switch — now collecting in {cycle[park_idx]}",
                           title="TrueSkate SLS collect", tags=["white_check_mark"])
                    print(f"[collect_sls] switch detected → {cycle[park_idx]}")
                continue

            # --- one collection iteration: reset (clapperboard) + gesture (sample) ---
            # Capture window 1: reset snap → feeds the clapperboard.
            r_frames, r_times, t_reset = _capture(
                rec, mjpeg_url, 0.35, 0.5, args.resize_width,
                action=lambda: reset_position(worker.driver, dw, dh))
            offset.add_reset(r_frames, r_times, t_reset)
            if detector is not None and detector.feed(r_frames):
                continue  # a reload started during the reset window — loop to monitor it

            # Capture window 2: the gesture → the sample.
            g = sample_mixture(rng, fracs=fracs, num_gestures=args.num_gestures,
                               use_spin=args.use_spin, recipe_vectors=recipe_vectors)
            try:
                g_frames, g_times, t_gesture = _capture(
                    rec, mjpeg_url, 0.0, 0.6, args.resize_width,  # tail captures post-trick motion
                    action=lambda: _execute(worker, g))
            except Exception as exc:  # noqa: BLE001
                print(f"  gesture {i} failed: {exc}")
                continue

            switched = detector.feed(g_frames) if detector is not None else False
            if detector is not None and detector.in_transition:
                # a reload began mid-gesture — discard this (garbage) sample, go monitor
                continue

            n = _save_sample(out_root / _park_tag(cycle[park_idx]) / f"sample_{saved:05d}",
                             g_frames, g_times, t_gesture, g, cycle[park_idx], park_idx,
                             offset, dw, dh, args.max_frames_per_sample)
            if n:
                saved += 1
            i += 1
            if i % 20 == 0:
                print(f"  [{i}] saved={saved} park={cycle[park_idx]} kind={g.kind} "
                      f"offset={offset.offset_s}s frames={n}")
                write_session_meta(saved, park_idx)

            now = time.monotonic()
            if switched:
                park_idx = (park_idx + 1) % len(cycle)
                deadline = now + args.per_park_hours * 3600.0
                prompted = False
                notify(f"[{device}] detected switch — now collecting in {cycle[park_idx]}",
                       title="TrueSkate SLS collect", tags=["white_check_mark"])
                print(f"[collect_sls] switch detected → {cycle[park_idx]}")
            elif detector is not None and not prompted and now >= deadline:
                nxt = cycle[(park_idx + 1) % len(cycle)]
                notify(f"[{device}] {args.per_park_hours}h done in {cycle[park_idx]} — walk over "
                       f"and load: {nxt}. Still collecting until you do.",
                       title="TrueSkate SLS switch park", priority="high", tags=["walking"])
                prompted = True
                print(f"[collect_sls] timer fired — prompted to switch to {nxt}")
    finally:
        write_session_meta(saved, park_idx)
        worker.disconnect()
        if caffeinate and caffeinate.poll() is None:
            caffeinate.terminate()
        notify(f"[{device}] SLS collection stopped: {saved} samples → {out_root.name}",
               title="TrueSkate SLS collect", tags=["checkered_flag"])
        print(f"\nDone: {saved} samples → {out_root}")


if __name__ == "__main__":
    main()
