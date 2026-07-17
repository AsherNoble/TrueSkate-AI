"""Segment-based SLS trace collector on the headless 30fps XCTest capture path.

Replaces the wedged DAL real-time capture (``collect_sls_traces.py``) with Apple's
XCTest screen recording (``vision/xctest_capture``): records bounded ``--segment-min``
``.mov`` segments while firing the SLS gesture mix, logging a per-gesture MANIFEST of
host-epoch call times. Each segment's ``.mov`` + ``.json`` manifest are written to THIS
host (the training-server Mac). Frames are aligned to gestures OFFLINE by
``scripts/data/align_xctest_traces.py`` — spawned async after each segment by default,
so a bad segment is caught before its footage is deleted.

Why segments (not one long recording): the recording accrues on-device until ``stop``,
and ``stop`` retrieves the whole thing over Appium (~5 Mbps). Bounding to a few minutes
keeps device free space high and each retrieval cheap. On iOS 18 the driver
auto-deletes the on-device attachment on stop, so device storage doesn't grow.

Park rotation is identical to the DAL collector: collect in one park for
``--per-park-hours``, then ntfy-prompt; advance when you tap "I switched parks". A
park switch closes the current segment so every segment is single-park.

PREREQS: WDA + Appium up, Appium started with ``--allow-insecure
xcuitest:xctest_screen_record`` (see ``scripts/launch_services.py``).

Usage:
    python scripts/data/collect_sls_xctest.py --devices iPhone_XR --segment-min 5
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

# SLS mixture gestures include multi-finger nslot/recipe touches; simultaneous
# finger-downs can trigger True Skate's park editor (see gesture_sampling docs).
# touch_actions reads this env var at import time, so it must be set before the
# import below. CMA-ES training never sets it, so its execution is unaffected.
os.environ.setdefault("TRUESKATE_MIN_FINGER_STAGGER_S", "0.12")

from trueskate_ai.data.gesture_sampling import load_recipe_vectors, sample_mixture  # noqa: E402
from trueskate_ai.rl.cmaes.action_param import execute_gesture_params  # noqa: E402
from trueskate_ai.rl.device_worker import (  # noqa: E402
    BUNDLE_ID, DeviceWorker, add_device_selection_args, resolve_devices,
)
from trueskate_ai.sim.gestures import scale_to_device  # noqa: E402
from trueskate_ai.sim.touch_actions import (  # noqa: E402
    curved_drag, curved_drag_with_spin_hold, reset_position, skip_loading_screen,
)
from trueskate_ai.utils.notify import confirm_button_action, notify, poll_confirmation  # noqa: E402
from trueskate_ai.vision.gameplay_filter import is_editor_frame, is_menu_frame  # noqa: E402
from trueskate_ai.vision.xctest_capture import XCTestScreenRecorder  # noqa: E402

# Same 11 SLS arenas + cycle order as the DAL collector (labels for prompting/tagging).
DEFAULT_SLS_PARKS = [
    "SLS 2016 Super Crown", "SLS 2015 Super Crown", "SLS 2015 New Jersey",
    "SLS 2015 Los Angeles", "SLS 2013 Super Crown", "SLS 2015 Paris",
    "SLS 2016 Newark", "SLS 2016 Munich", "SLS 2014 Los Angeles",
    "SLS 2013 Portland", "SLS 2013 Kansas City",
]

_STOP = False


def _on_sigint(_signum, _frame):
    global _STOP
    _STOP = True
    print("\n[collect_xctest] SIGINT — finishing current segment and shutting down...")


def _park_tag(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _start_caffeinate():
    try:
        p = subprocess.Popen(["caffeinate", "-dimsu", "-w", str(os.getpid())])
        print(f"[collect_xctest] caffeinate active (PID {p.pid}).")
        return p
    except FileNotFoundError:
        print("[collect_xctest] caffeinate not found — Mac may sleep.")
        return None


def _execute(worker: DeviceWorker, g) -> None:
    """Fire one sampled gesture (curved flicks required — straight swipes don't play)."""
    dw, dh = worker.device_w, worker.device_h
    if g.kind == "flick":
        pts = [scale_to_device(x, y, dw, dh) for x, y in g.waypoints]
        easing = None if g.easing_power == 1.0 else (lambda t, p=g.easing_power: t ** p)
        curved_drag(worker.driver, pts, total_duration=g.duration, easing=easing)
    elif g.kind == "spin_flick":
        pts = [scale_to_device(x, y, dw, dh) for x, y in g.waypoints]
        easing = None if g.easing_power == 1.0 else (lambda t, p=g.easing_power: t ** p)
        bx, by = g.spin_button_xy or worker.spin_button_xy
        curved_drag_with_spin_hold(
            worker.driver, pts, total_duration=g.duration, easing=easing,
            spin_button_pt=scale_to_device(bx, by, dw, dh),
            hold_start_s=g.spin_hold_start_s, hold_end_s=g.spin_hold_end_s,
        )
    else:  # nslot / recipe / spin
        spin_xy = worker.spin_button_xy if g.use_spin else None
        execute_gesture_params(
            worker.driver, np.asarray(g.params, dtype=np.float64), dw, dh,
            num_gestures=g.num_gestures, use_spin=g.use_spin,
            spin_button_xy=spin_xy, timing_device_key=worker.device_id,
        )


def _device_free_gb(udid: str) -> float | None:
    """Device free storage in GB via libimobiledevice, or None if unavailable."""
    try:
        r = subprocess.run(
            ["ideviceinfo", "-u", udid, "-q", "com.apple.disk_usage", "-k", "TotalDataAvailable"],
            capture_output=True, text=True, timeout=15,
        )
        if r.returncode == 0 and r.stdout.strip().isdigit():
            return round(int(r.stdout.strip()) / 1e9, 1)
    except Exception:  # noqa: BLE001
        pass
    return None


def _recover_session(worker: DeviceWorker) -> bool:
    """True if the worker has a live WDA session, reconnecting once if it dropped.

    Called after a recording error: a transient XCTDaemon hiccup (e.g. Code=7 "Failed
    to write file") usually leaves the session itself fine — the cheap probe succeeds
    and we just skip the bad segment — but a dropped WDA session needs one reconnect
    before the next segment can record.
    """
    try:
        worker.driver.get_window_size()
        return True
    except Exception:  # noqa: BLE001 — session likely dropped; try one reconnect
        pass
    try:
        return bool(worker._reconnect())
    except Exception:  # noqa: BLE001
        return False


def _exit_replay(worker: DeviceWorker) -> None:
    """Force True Skate back to live skatepark gameplay from a replay/menu state.

    Terminate + relaunch is coordinate-free and reliable (the replay BACK button isn't
    in a stable enough spot to tap blindly, and a wrong tap could open Share). The
    XCTest recording keeps running across this — the loading frames simply aren't logged
    (the guard skips non-gameplay), so they never become samples.
    """
    d = worker.driver
    try:
        d.terminate_app(BUNDLE_ID)
    except Exception:  # noqa: BLE001
        pass
    time.sleep(1.0)
    try:
        d.activate_app(BUNDLE_ID)
        time.sleep(1.0)
        skip_loading_screen(d, worker.device_w, worker.device_h)
        time.sleep(1.0)
    except Exception as exc:  # noqa: BLE001
        print(f"[exit_replay] relaunch error: {exc!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Segment-based XCTest SLS trace collector.")
    ap.add_argument("--out-dir", type=Path, default=_REPO_ROOT / "data" / "sls_xctest")
    ap.add_argument("--segment-min", type=float, default=1.0,
                    help="Max minutes per .mov segment. KEEP SHORT: stop_and_save retrieves the "
                         "whole .mov as ONE base64 HTTP response over Appium/WDA, and gameplay "
                         "motion runs ~76 MB/min at 30fps full-res. Retrieval is reliable to "
                         "~114 MB (~90s); a 5-min segment (~380 MB) aborts the connection "
                         "(RemoteDisconnected) and the segment is lost. 1 min (~77 MB) is safe.")
    ap.add_argument("--per-park-hours", type=float, default=4.0)
    ap.add_argument("--start-park", default=None)
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--flick-frac", type=float, default=0.6)
    ap.add_argument("--nslot-frac", type=float, default=0.25)
    ap.add_argument("--recipe-frac", type=float, default=0.15)
    ap.add_argument("--spin-frac", type=float, default=0.0,
                    help="TRUE share [0,1] of fires that are guaranteed-spin gestures (rotate "
                         "button HELD); the flick/nslot/recipe mix keeps its ratios in the "
                         "remaining share. 0.2 = ~20%% of fires hold the spin button — the "
                         "knob to grow the spin-family corpus.")
    ap.add_argument("--num-gestures", type=int, default=2)
    ap.add_argument("--use-spin", action="store_true",
                    help="Legacy: make the plain nslot branch spin-capable (~half gate-off). "
                         "Prefer --spin-frac for controlled, guaranteed spin coverage.")
    ap.add_argument("--recipe-dir", type=Path, default=_REPO_ROOT / "trick_libraries")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--tail-s", type=float, default=1.0,
                    help="Seconds to wait after each gesture so the trick plays out into the "
                         "recording before the next reset (the aligner's response window).")
    ap.add_argument("--capture-offset-s", type=float, default=None,
                    help="Command->pixel offset Δ stamped into manifests (measure via the "
                         "clapperboard on the XCTest path). None = aligner uses its default.")
    ap.add_argument("--min-free-gb", type=float, default=8.0,
                    help="ntfy-alert if device free storage drops below this.")
    ap.add_argument("--no-align", action="store_true",
                    help="Do NOT auto-spawn the aligner after each segment (save .mov+manifest only).")
    ap.add_argument("--no-caffeinate", action="store_true")
    ap.add_argument("--no-rotate", action="store_true")
    ap.add_argument("--confirm-poll-s", type=float, default=10.0)
    ap.add_argument("--no-gameplay-guard", action="store_true",
                    help="Disable the in-loop replay/menu guard (by default, gestures fired "
                         "while True Skate is in replay/menu are NOT logged, and the app is "
                         "relaunched to return to live gameplay).")
    ap.add_argument("--gameplay-check-every", type=int, default=1,
                    help="Screenshot-check the gameplay state every N gestures (1 = every gesture).")
    ap.add_argument("--menu-recover-after", type=int, default=2,
                    help="Consecutive non-gameplay detections before relaunching True Skate.")
    ap.add_argument("--max-start-fails", type=int, default=8,
                    help="Consecutive recording-start failures before exiting for a clean "
                         "supervisor restart (avoids hammering/re-wedging the XCTest daemon).")
    add_device_selection_args(ap)
    args = ap.parse_args()

    # Validate the gesture mix BEFORE any device contact: a bad value would
    # otherwise surface as a per-gesture ValueError mid-run and crash-loop the
    # launchd job (wedge-adjacent churn on the XCTest recorder).
    spin_frac = min(1.0, max(0.0, args.spin_frac))
    if spin_frac != args.spin_frac:
        print(f"WARNING: --spin-frac {args.spin_frac} outside [0, 1]; clamped to {spin_frac}.")
        args.spin_frac = spin_frac
    if args.flick_frac + args.nslot_frac + args.recipe_frac <= 0 and args.spin_frac <= 0:
        raise SystemExit("All gesture mixture weights are zero — nothing to sample "
                         "(--flick-frac/--nslot-frac/--recipe-frac/--spin-frac).")
    if args.num_gestures < 1:
        raise SystemExit(f"--num-gestures must be >= 1, got {args.num_gestures}")

    try:
        devices = resolve_devices(devices_arg=args.devices, personal=args.personal,
                                  all_devices=args.all_devices)
    except ValueError as exc:
        raise SystemExit(str(exc))
    if len(devices) != 1:
        raise SystemExit(
            f"collect_sls_xctest runs ONE phone per process; selection resolved to "
            f"{len(devices)} {[d['name'] for d in devices]}. Pass a single --devices NAME.")
    cfg = devices[0]

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
    device = cfg["name"]
    print(f"Connecting to {device} (needs WDA+Appium up; run launch_services.py)...")
    worker.connect()
    dw, dh = worker.device_w, worker.device_h
    udid = os.environ.get(cfg.get("env_key", ""), "") or cfg.get("udid", "")

    rec = XCTestScreenRecorder(worker.driver, fps=args.fps)
    session = time.strftime("%Y%m%d_%H%M%S")
    out_root = args.out_dir / f"{device}_{session}"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Saving segments to {out_root}; park cycle starts at: {cycle[0]}")

    rng = np.random.default_rng(args.seed)
    recipe_vectors = load_recipe_vectors(args.recipe_dir)
    print(f"Loaded {len(recipe_vectors)} packable recipes.")
    fracs = (args.flick_frac, args.nslot_frac, args.recipe_frac)
    print(f"Gesture mix: base weights flick={args.flick_frac} nslot={args.nslot_frac} "
          f"recipe={args.recipe_frac}; spin share={args.spin_frac} (guaranteed-hold slice).")

    notify(f"[{device}] XCTest SLS collection starting in {cycle[0]} "
           f"({args.segment_min:.0f}-min segments, {args.per_park_hours}h/park).",
           title="TrueSkate SLS collect", tags=["camera"])

    park_idx = 0
    park_deadline = time.monotonic() + args.per_park_hours * 3600.0
    awaiting = False
    prompt_ts = 0.0
    last_poll = 0.0
    segment_idx = 0
    total_gestures = 0
    total_menu_skips = 0
    start_fail_streak = 0
    global_deadline = (time.monotonic() + args.max_hours * 3600.0) if args.max_hours else None

    def _device_aligner_spawn(manifest_path: Path):
        if args.no_align:
            return
        subprocess.Popen(
            [sys.executable, str(_HERE / "align_xctest_traces.py"),
             "--segment", str(manifest_path), "--delete-mov"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    try:
        while not _STOP:
            if global_deadline and time.monotonic() > global_deadline:
                print("[collect_xctest] global --max-hours reached.")
                break

            cur_park = cycle[park_idx]
            try:
                rec.start()
            except Exception as exc:  # noqa: BLE001
                # XCTest recording can transiently fail (XCTDaemon "Failed to write
                # file" Code=7 — a wedged recording daemon) or the WDA session may have
                # dropped. Abort any partial recording (orphaned on-device recordings
                # pile up and re-wedge the daemon), recover the session, and SKIP this
                # segment — never crash the whole run over one bad segment.
                start_fail_streak += 1
                print(f"[seg {segment_idx}] recording start failed (streak {start_fail_streak}): "
                      f"{exc!r} — abort + skip")
                try:
                    rec.abort()
                except Exception:  # noqa: BLE001
                    pass
                # Don't hammer rec.start(): a tight retry loop on a transient failure
                # RE-WEDGES the XCTest recording daemon — observed ~14k retries keeping it
                # stuck even across a reboot. After a few fails, EXIT so the supervisor
                # restarts us after a pause, giving the daemon a real break instead.
                if start_fail_streak >= args.max_start_fails:
                    notify(f"[{device}] {start_fail_streak} consecutive recording-start failures "
                           f"— exiting for a clean restart (likely a wedged XCTest daemon; "
                           f"a device reboot may be needed).",
                           title="TrueSkate SLS collect", priority="high", tags=["warning"])
                    print(f"[collect_xctest] {start_fail_streak} start-fails — exit for supervisor restart.")
                    break
                if not _recover_session(worker):
                    print("[collect_xctest] session unrecoverable — exit for supervisor restart.")
                    break
                time.sleep(3.0)
                continue
            start_fail_streak = 0  # rec.start() succeeded
            seg_deadline = time.monotonic() + args.segment_min * 60.0
            events: list[dict] = []
            park_switched = False
            seg_aborted = False
            seg_iter = 0
            non_gameplay_streak = 0

            while not _STOP and time.monotonic() < seg_deadline:
                if global_deadline and time.monotonic() > global_deadline:
                    break
                try:
                    reset_position(worker.driver, dw, dh)
                except Exception as exc:  # noqa: BLE001 — WDA session dropped mid-segment
                    print(f"[seg {segment_idx}] reset failed mid-segment: {exc!r} — close segment")
                    seg_aborted = True
                    break
                time.sleep(0.3)  # board settle into the reset state

                # --- gameplay guard: never log a gesture fired into the replay/menu ---
                if not args.no_gameplay_guard and seg_iter % max(1, args.gameplay_check_every) == 0:
                    try:
                        _guard_png = worker.driver.get_screenshot_as_png()
                        _in_editor = is_editor_frame(_guard_png)
                        if _in_editor or is_menu_frame(_guard_png):
                            non_gameplay_streak += 1
                            total_menu_skips += 1
                            _what = "park editor" if _in_editor else "replay/menu"
                            print(f"[seg {segment_idx}] {_what} detected "
                                  f"(streak {non_gameplay_streak}) — skipping gesture (not logged)")
                            if non_gameplay_streak >= args.menu_recover_after:
                                print(f"[seg {segment_idx}] relaunching True Skate to exit {_what}...")
                                _exit_replay(worker)
                                non_gameplay_streak = 0
                            seg_iter += 1
                            continue  # do not fire/log a gesture into the menu/editor
                        non_gameplay_streak = 0
                    except Exception as exc:  # noqa: BLE001 — never let the guard crash the run
                        print(f"[seg {segment_idx}] gameplay check failed: {exc!r} — proceeding")
                seg_iter += 1

                g = sample_mixture(rng, fracs=fracs, spin_frac=args.spin_frac,
                                   num_gestures=args.num_gestures,
                                   use_spin=args.use_spin, recipe_vectors=recipe_vectors)
                if g.kind == "spin_flick" or g.use_spin:
                    # stamp before meta() so the logged coord is the one that fires
                    g.spin_button_xy = worker.spin_button_xy
                t0 = time.time()
                try:
                    _execute(worker, g)
                except Exception as exc:  # noqa: BLE001
                    print(f"  gesture {total_gestures} failed: {exc}")
                    continue
                t1 = time.time()
                # --- post-gesture PARK-EDITOR check ---
                # True Skate's park editor is opened DURING a (multi-finger) gesture, and the
                # NEXT reset_position closes it again — so the pre-gesture guard above never
                # catches it (it checks right after a reset, when the editor is already closed).
                # Check here, right after the gesture: if it landed in the editor, DROP this
                # sample so editor frames never enter the dataset. Persistent editor (if reset
                # ever fails to close it) is still caught + relaunched by the pre-gesture guard.
                if not args.no_gameplay_guard:
                    try:
                        time.sleep(0.35)  # let the editor UI render before scoring
                        if is_editor_frame(worker.driver.get_screenshot_as_png()):
                            total_menu_skips += 1
                            print(f"[seg {segment_idx}] gesture opened the park editor "
                                  f"— dropping sample (not logged)")
                            continue  # do NOT log a gesture that landed in the editor
                    except Exception as exc:  # noqa: BLE001 — never let the guard crash the run
                        print(f"[seg {segment_idx}] post-gesture editor check failed: {exc!r} — proceeding")
                events.append({
                    "gesture_index": total_gestures,
                    "t_call_start_epoch_s": t0,
                    "t_call_end_epoch_s": t1,
                    "park": cur_park,
                    "park_change_index": park_idx,
                    **g.meta(),
                })
                total_gestures += 1
                time.sleep(args.tail_s)  # trick plays out into the recording (response window)

                # --- park rotation (same ntfy mechanism as the DAL collector) ---
                if args.no_rotate:
                    continue
                now = time.monotonic()
                if awaiting:
                    if now - last_poll >= args.confirm_poll_s:
                        last_poll = now
                        if poll_confirmation("SWITCHED", since_ts=prompt_ts):
                            park_idx = (park_idx + 1) % len(cycle)
                            park_deadline = now + args.per_park_hours * 3600.0
                            awaiting = False
                            park_switched = True
                            notify(f"[{device}] confirmed — now collecting in {cycle[park_idx]}",
                                   title="TrueSkate SLS collect", tags=["white_check_mark"])
                            print(f"[collect_xctest] switch confirmed → {cycle[park_idx]}")
                            break  # close the segment so it stays single-park
                elif now >= park_deadline:
                    nxt = cycle[(park_idx + 1) % len(cycle)]
                    prompt_ts = time.time()
                    notify(f"[{device}] {args.per_park_hours}h in {cur_park} — please load "
                           f"{nxt}, then tap the button. Still collecting here until you do.",
                           title="TrueSkate: switch park", priority="high", tags=["walking"],
                           actions=confirm_button_action("I switched parks"))
                    awaiting = True
                    last_poll = now
                    print(f"[collect_xctest] timer fired — prompted to switch to {nxt}")

            # --- close + save the segment (partial segments on STOP/switch are still saved) ---
            if seg_aborted:
                # session dropped mid-segment: discard the partial recording, recover, skip.
                try:
                    rec.abort()
                except Exception:  # noqa: BLE001
                    pass
                if not _recover_session(worker):
                    print("[collect_xctest] session unrecoverable — exit for supervisor restart.")
                    break
                continue
            mov_path = out_root / f"segment_{segment_idx:05d}.mov"
            try:
                res = rec.stop_and_save(mov_path)
            except Exception as exc:  # noqa: BLE001
                # stop/retrieve failed (oversized payload → RemoteDisconnected, or an
                # XCTDaemon write error): the segment is lost, but abort + recover and
                # keep collecting instead of crashing (which would orphan the recording).
                print(f"[seg {segment_idx}] stop_and_save failed: {exc!r} — segment lost, continue")
                try:
                    rec.abort()
                except Exception:  # noqa: BLE001
                    pass
                if not _recover_session(worker):
                    print("[collect_xctest] session unrecoverable — exit for supervisor restart.")
                    break
                segment_idx += 1  # keep segment numbering monotonic even on a lost segment
                continue
            manifest_path = out_root / f"segment_{segment_idx:05d}.json"
            manifest = {
                "device": device, "device_logical_w": dw, "device_logical_h": dh,
                "segment_index": segment_idx, "park": cur_park, "park_change_index": park_idx,
                "started_at_epoch_s": res.started_at_epoch_s,
                "host_start_epoch_s": res.host_start_epoch_s,
                "host_stop_epoch_s": res.host_stop_epoch_s,
                "fps": res.fps, "codec": res.codec,
                "capture_offset_s": args.capture_offset_s,
                "tail_s": args.tail_s,
                # Sampler config, so a corpus session is reconstructable without
                # console logs (mirrors the DAL collector's session_meta.json).
                "mix": {"flick": args.flick_frac, "nslot": args.nslot_frac,
                        "recipe": args.recipe_frac, "spin_frac": args.spin_frac},
                "num_gestures": args.num_gestures, "use_spin": args.use_spin,
                "mov": mov_path.name, "n_gestures": len(events), "gestures": events,
            }
            manifest_path.write_text(json.dumps(manifest, indent=2))
            free_gb = _device_free_gb(udid) if udid else None
            print(f"[seg {segment_idx}] saved {res.summary()['mb']}MB, {len(events)} gestures, "
                  f"park={cur_park}, device_free={free_gb}GB"
                  f"{' (PARK SWITCH)' if park_switched else ''}")
            if free_gb is not None and free_gb < args.min_free_gb:
                notify(f"[{device}] device free storage low: {free_gb}GB (< {args.min_free_gb}).",
                       title="TrueSkate SLS collect", priority="high", tags=["warning"])
            _device_aligner_spawn(manifest_path)
            segment_idx += 1
    finally:
        try:
            rec.abort()  # ensure no recording leaks on the device
        except Exception:  # noqa: BLE001
            pass
        worker.disconnect()
        if caffeinate and caffeinate.poll() is None:
            caffeinate.terminate()
        notify(f"[{device}] XCTest SLS collection stopped: {segment_idx} segments, "
               f"{total_gestures} gestures, {total_menu_skips} menu-skipped → {out_root.name}",
               title="TrueSkate SLS collect", tags=["checkered_flag"])
        print(f"\nDone: {segment_idx} segments, {total_gestures} gestures, "
              f"{total_menu_skips} skipped (replay/menu) → {out_root}")


if __name__ == "__main__":
    main()
