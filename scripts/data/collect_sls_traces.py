"""Long-running, multi-park RANDOM-GESTURE trace collector for SLS skateparks.

Purpose: build the (frame -> known-gesture) corpus the sequence model needs from
the VISUALLY DIVERSE expert SLS arenas. Unlike CMA-ES trick mining (which needs
the single-reset flatground park so obstacles/varied resets don't contaminate
params), this loop just wants frame/gesture pairs — so it is OBSTACLE-TOLERANT:
it records the executed gesture as the label regardless of whether the board
lands, whiffs, or bumps a wall, and varied board positions are desirable.

Per phone, it collects in one park for --per-park-hours, then ntfy-prompts you to
walk over and load the next SLS park; it KEEPS collecting in the current park
until you tap the ntfy "I switched parks" button (no idle time if you're away),
then it advances + resets the timer. The command->frame pipeline latency (the
"clapperboard") is measured once at startup via the dedicated Clapperboard app
(vision/clapperboard).

Prereqs: WDA + Appium up for the device (python scripts/launch_services.py --devices iPhone_XR).

Device selection mirrors the CMA-ES launcher's shared flags (--devices / --personal
/ --all-devices); no hardcoded default, and the personal iPhone 11 is reachable only
via an explicit --personal. Unlike CMA-ES (multi-device pool), this collector runs ONE
phone per process, so exactly one device must resolve — run a second process for a
second phone.

Usage:
    python scripts/data/collect_sls_traces.py --devices iPhone_XR --per-park-hours 8
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# The Appium client (urllib3) spams "connection pool is full" under our rapid
# request cadence; harmless, but quiet it so the run log stays readable.
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from trueskate_ai.data.gesture_sampling import load_recipe_vectors, sample_mixture  # noqa: E402
from trueskate_ai.rl.cmaes.action_param import execute_gesture_params  # noqa: E402
from trueskate_ai.rl.device_worker import (  # noqa: E402
    DeviceWorker, add_device_selection_args, resolve_devices,
)
from trueskate_ai.sim.gestures import execute_static_push, scale_to_device  # noqa: E402
from trueskate_ai.sim.touch_actions import curved_drag, reset_position  # noqa: E402
from trueskate_ai.utils.notify import confirm_button_action, notify, poll_confirmation  # noqa: E402
from trueskate_ai.vision.clapperboard import RollingCaptureOffset  # noqa: E402
from trueskate_ai.vision.dal_capture import DalFrameRecorder, resolve_device_name  # noqa: E402

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


def _capture(rec: DalFrameRecorder, pre_s: float, hold_s: float, action=None):
    """Mark t0, run `action`, hold, then slice the persistent buffer.

    The DAL recorder streams continuously, so we do NOT start/stop it per sample
    (that would re-open the exclusive device thousands of times and wedge it — see
    vision/dal_capture). Instead we mark the gesture call window and `window()` the
    rolling buffer, including a `pre_s` lead-in that's already buffered for free.

    Returns (frames, times, t0, t_end): t0 is the monotonic time `action` started,
    t_end the monotonic time the synchronous `action` returned.
    """
    t0 = time.monotonic()
    if action is not None:
        action()
    t_end = time.monotonic()  # synchronous action (gesture/reset) has fully executed
    if hold_s > 0:
        time.sleep(hold_s)
    frames, times = rec.window(t0 - pre_s, time.monotonic())
    return frames, times, t0, t_end


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


def _save_sample(sample_dir: Path, frames: list, times: list[float], t_gesture_end: float,
                 g, park: str, park_idx: int, offset: RollingCaptureOffset,
                 dw: float, dh: float, max_frames: int) -> int:
    """Save downsampled frames + meta. frame_times are relative to the gesture
    CALL-RETURN (t_gesture_end), the tight anchor — see vision/clapperboard. Returns n saved."""
    if not frames:
        return 0
    pairs = list(zip(times, frames))
    keep = _downsample(pairs, max_frames)
    sample_dir.mkdir(parents=True, exist_ok=True)
    frame_times = []
    for out_i, src_i in enumerate(keep):
        t, fr = pairs[src_i]
        Image.fromarray(fr, mode="RGB").save(sample_dir / f"frame_{out_i:03d}.png")
        frame_times.append(round(t - t_gesture_end, 4))
    meta = {
        "device_logical_w": dw,
        "device_logical_h": dh,
        "park": park,
        "park_change_index": park_idx,
        "spin_active": bool(g.use_spin) if g.kind != "flick" else False,
        "gesture_end_monotonic": t_gesture_end,
        "frame_times": frame_times,        # relative to the gesture call-return (s)
        "n_frames": len(frame_times),
        # The on-screen gesture spans frame_times [capture_offset_s - duration,
        # capture_offset_s]; a frame at ft shows the effect of a command whose call
        # returned at ft - capture_offset_s. capture_offset_s is small/negative.
        "capture_offset_s": offset.offset_s,
        "capture_offset_jitter_s": offset.jitter_s,
        **g.meta(),
    }
    (sample_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return len(frame_times)


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-park random-gesture SLS trace collector.")
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
    ap.add_argument("--no-clapperboard", action="store_true",
                    help="Explicitly accept a null capture offset (silences the no-offset warning).")
    ap.add_argument("--capture-offset-s", type=float, default=None,
                    help="Stamp this capture offset Δ_DAL directly. Measure it separately via "
                         "experiments/cross_pipeline_delta.py — the in-process Clapperboard "
                         "calibration is disabled under DAL (its app-switch wedges the stream).")
    ap.add_argument("--no-caffeinate", action="store_true")
    ap.add_argument("--no-rotate", action="store_true",
                    help="Single-park mode: collect in one park, never prompt/advance.")
    ap.add_argument("--confirm-poll-s", type=float, default=10.0,
                    help="How often to check ntfy for your 'switched' tap after a prompt.")
    add_device_selection_args(ap)
    args = ap.parse_args()

    # Same shared resolver as the CMA-ES launcher. This collector runs ONE phone per
    # process, so require exactly one device — the no-flag default (whole collection
    # roster) and --all-devices both resolve to >1 and are rejected, and the personal
    # iPhone 11 is reachable only via an explicit --personal. Prevents an unattended
    # job (human or agent) from silently grabbing the wrong phone.
    try:
        devices = resolve_devices(devices_arg=args.devices, personal=args.personal,
                                  all_devices=args.all_devices)
    except ValueError as exc:
        raise SystemExit(str(exc))
    if len(devices) != 1:
        raise SystemExit(
            f"collect_sls_traces runs ONE phone per process; selection resolved to "
            f"{len(devices)} devices {[d['name'] for d in devices]}. Pass a single "
            f"--devices NAME (e.g. --devices iPhone_XR), or --personal for the iPhone 11."
        )
    cfg = devices[0]

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
    print(f"Connecting to {cfg['name']} (needs WDA+Appium up; "
          f"run launch_services.py --devices {cfg['name']})...")
    worker.connect()
    dw, dh, device = worker.device_w, worker.device_h, cfg["name"]

    # Capture via the AVFoundation/DAL screen-mirror (steady 30fps), NOT WDA MJPEG
    # (~10-15fps variable). The recorder streams continuously and we window-slice it
    # per sample — see vision/dal_capture for why per-sample open/close wedges it.
    if not worker.avf_name:
        raise SystemExit(
            f"{device} has no avf_name in DEVICES — can't DAL-capture. Run "
            f"'python scripts/view_device.py --list' and set it (by NAME, not index)."
        )
    avf_name = resolve_device_name(worker.avf_name) or worker.avf_name
    print(f"Opening DAL capture '{avf_name}' at 30fps (resize_width={args.resize_width})...")
    rec = DalFrameRecorder(fps=30, resize_width=args.resize_width)
    rec.open(avf_name)
    if not rec.wait_for_frames(timeout_s=10.0):
        rec.close()
        raise SystemExit(
            f"DAL capture for '{avf_name}' delivered no frames (last ffmpeg err: "
            f"{rec.last_error!r}). The CMIO session is likely wedged — open QuickTime "
            f"> New Movie Recording, pick the device once, quit QuickTime, then retry "
            f"(or replug the phone). The phone must also be unlocked, screen on."
        )
    print(f"  DAL frames flowing (got {rec.frame_count} during warmup).")
    # Frames arriving is NOT enough: a wedged CMIO session delivers frozen duplicate
    # frames that pass wait_for_frames. Force a guaranteed visible change (push+reset)
    # and confirm the decoded content actually moved before starting a run.
    try:
        execute_static_push(worker.driver, device_w=dw, device_h=dh)
        time.sleep(0.6)
        reset_position(worker.driver, dw, dh)
        time.sleep(1.2)
    except Exception as exc:  # noqa: BLE001
        print(f"  (liveness nudge failed: {exc})")
    if (rec.content_stale_age or 99.0) > 2.5:
        rec.close()
        raise SystemExit(
            "DAL stream is FROZEN — frames arrive but content never changes (CMIO wedge). "
            "Open QuickTime > New Movie Recording, select the device once, quit QuickTime "
            "(or replug the phone), then retry. Phone must be unlocked, screen on.")
    print(f"  DAL live (content_stale_age={rec.content_stale_age:.2f}s).")

    session = time.strftime("%Y%m%d_%H%M%S")
    out_root = args.out_dir / f"{device}_{session}"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {out_root}; park cycle starts at: {cycle[0]}")

    rng = np.random.default_rng(args.seed)
    recipe_vectors = load_recipe_vectors(args.recipe_dir)
    print(f"Loaded {len(recipe_vectors)} packable recipes for the perturbed-recipe share.")
    fracs = (args.flick_frac, args.nslot_frac, args.recipe_frac)

    offset = RollingCaptureOffset()

    def write_session_meta(saved: int, park_idx: int) -> None:
        (out_root / "session_meta.json").write_text(json.dumps({
            "device": device, "session": session, "logical_w": dw, "logical_h": dh,
            "park_cycle": cycle, "per_park_hours": args.per_park_hours,
            "fracs": {"flick": fracs[0], "nslot": fracs[1], "recipe": fracs[2]},
            "num_gestures": args.num_gestures, "use_spin": args.use_spin,
            "samples_saved": saved, "current_park_index": park_idx,
            "capture": {"method": "avfoundation_dal", "fps": 30,
                        "resize_width": args.resize_width, "avf_name": avf_name},
            **offset.summary(),
        }, indent=2))

    # --- capture-pipeline offset Δ. The in-process Clapperboard calibration is
    # DELIBERATELY NOT run here: it would activate_app the Clapperboard, and
    # app-switching is the CMIO-freeze trigger that persistently wedges the DAL
    # stream we just opened (verified 2026-06-23 — see vision/dal_capture + journal).
    # So under DAL capture, Δ MUST come from --capture-offset-s; measure it separately
    # via experiments/cross_pipeline_delta.py (gap-first, so DAL stays live).
    if args.capture_offset_s is not None:
        offset.add_sample(args.capture_offset_s)
        print(f"Using provided capture offset Δ_DAL={args.capture_offset_s}s.")
    elif args.no_clapperboard:
        print("Capture offset stamped NULL (--no-clapperboard).")
    else:
        print("WARNING: no --capture-offset-s — capture offset stamped NULL. In-process "
              "Clapperboard calibration is DISABLED under DAL (its app-switch would wedge "
              "the stream). Measure Δ_DAL via experiments/cross_pipeline_delta.py and pass "
              "--capture-offset-s (or --no-clapperboard to silence this).")
    notify(f"[{device}] SLS collection starting in {cycle[0]} "
           f"(offset={offset.offset_s}s). Cycle of {len(cycle)} parks, {args.per_park_hours}h each.",
           title="TrueSkate SLS collect", tags=["camera"])

    park_idx = 0
    deadline = time.monotonic() + args.per_park_hours * 3600.0
    awaiting = False        # prompted to switch, waiting for the user's ntfy tap
    prompt_ts = 0.0
    last_poll = 0.0
    saved = 0
    i = 0
    global_deadline = (time.monotonic() + args.max_hours * 3600.0) if args.max_hours else None

    try:
        while not _STOP:
            if global_deadline and time.monotonic() > global_deadline:
                print("[collect_sls] global --max-hours reached.")
                break

            # --- DAL health. TWO failure modes:
            #   (1) hard stall — no frames at all (ffmpeg died / device gone).
            #   (2) FROZEN — a CMIO wedge keeps delivering BITWISE-IDENTICAL duplicate
            #       frames, so last_frame_age stays ~0 and is_alive stays True; only
            #       content_stale_age catches it. The collector resets+gestures every
            #       iteration, so >8s without any content change means frozen.
            # A restart re-opens ffmpeg but CANNOT clear a CMIO wedge (needs a QuickTime
            # re-activate / replug), so after restarting we FORCE a screen change and
            # confirm the content actually moved — else STOP, never save frozen frames
            # labelled with gestures.
            stale = rec.content_stale_age or 0.0
            if not rec.is_alive() or (rec.last_frame_age or 0.0) > 3.0 or stale > 8.0:
                why = ("dead" if not rec.is_alive()
                       else f"frozen {stale:.0f}s" if stale > 8.0
                       else f"no-frames {rec.last_frame_age:.0f}s")
                print(f"[collect_sls] DAL {why} (err={rec.last_error!r}) — restarting capture...")
                ok = rec.restart()
                if ok:
                    reset_position(worker.driver, dw, dh)  # force a visible change
                    time.sleep(1.2)
                if not ok or (rec.content_stale_age or 99.0) > 2.5:
                    notify(f"[{device}] DAL capture wedged/frozen, can't self-heal — stopping. "
                           f"QuickTime re-activate / replug needed.",
                           title="TrueSkate SLS collect", priority="high", tags=["warning"])
                    print("[collect_sls] DAL did not return LIVE frames after restart — stopping.")
                    break

            # --- one collection iteration: reset the board, then the gesture sample ---
            reset_position(worker.driver, dw, dh)
            time.sleep(0.3)  # board settle

            g = sample_mixture(rng, fracs=fracs, num_gestures=args.num_gestures,
                               use_spin=args.use_spin, recipe_vectors=recipe_vectors)
            try:
                g_frames, g_times, _, t_gesture_end = _capture(
                    rec, 0.2, 0.6,  # 0.2s buffered lead-in + 0.6s tail for post-trick motion
                    action=lambda: _execute(worker, g))
            except Exception as exc:  # noqa: BLE001
                print(f"  gesture {i} failed: {exc}")
                continue

            n = _save_sample(out_root / _park_tag(cycle[park_idx]) / f"sample_{saved:05d}",
                             g_frames, g_times, t_gesture_end, g, cycle[park_idx], park_idx,
                             offset, dw, dh, args.max_frames_per_sample)
            if n:
                saved += 1
            i += 1
            if i % 20 == 0:
                print(f"  [{i}] saved={saved} park={cycle[park_idx]} kind={g.kind} "
                      f"offset={offset.offset_s}s frames={n}")
                write_session_meta(saved, park_idx)

            # --- park rotation: prompt on the timer, advance when you tap "switched" ---
            now = time.monotonic()
            if args.no_rotate:
                continue
            if awaiting:
                if now - last_poll >= args.confirm_poll_s:
                    last_poll = now
                    if poll_confirmation("SWITCHED", since_ts=prompt_ts):
                        park_idx = (park_idx + 1) % len(cycle)
                        deadline = now + args.per_park_hours * 3600.0
                        awaiting = False
                        notify(f"[{device}] confirmed — now collecting in {cycle[park_idx]}",
                               title="TrueSkate SLS collect", tags=["white_check_mark"])
                        print(f"[collect_sls] switch confirmed → {cycle[park_idx]}")
            elif now >= deadline:
                nxt = cycle[(park_idx + 1) % len(cycle)]
                prompt_ts = time.time()
                notify(f"[{device}] {args.per_park_hours}h in {cycle[park_idx]} — please load "
                       f"{nxt}, then tap the button. Still collecting here until you do.",
                       title="TrueSkate: switch park", priority="high", tags=["walking"],
                       actions=confirm_button_action("I switched parks"))
                awaiting = True
                last_poll = now
                print(f"[collect_sls] timer fired — prompted to switch to {nxt}")
    finally:
        write_session_meta(saved, park_idx)
        rec.close()  # release the EXCLUSIVE DAL device — a leaked ffmpeg wedges the next open
        worker.disconnect()
        if caffeinate and caffeinate.poll() is None:
            caffeinate.terminate()
        notify(f"[{device}] SLS collection stopped: {saved} samples → {out_root.name}",
               title="TrueSkate SLS collect", tags=["checkered_flag"])
        print(f"\nDone: {saved} samples → {out_root}")


if __name__ == "__main__":
    main()
