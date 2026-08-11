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

from trueskate_ai.data.gesture_sampling import (  # noqa: E402
    BASIC_HOLD_CALIBRATION_TAP_SHARE, load_recipe_vectors,
    sample_basic_hold_mixture, sample_mixture,
)
from trueskate_ai.rl.cmaes.action_param import execute_gesture_params  # noqa: E402
from trueskate_ai.rl.device_worker import (  # noqa: E402
    BUNDLE_ID, DeviceWorker, add_device_selection_args, resolve_devices,
)
from trueskate_ai.sim.gestures import scale_to_device  # noqa: E402
from trueskate_ai.sim.touch_actions import (  # noqa: E402
    curved_drag, curved_drag_with_spin_hold, long_press, reset_position,
    skip_loading_screen, tap,
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
    if g.kind in ("hold", "tap"):
        # Stationary touch (Model 1 MVP): no path, so nothing to curve. A tap is
        # just a hold of zero length — both render the normal orange mark at the
        # commanded point (measured Stage 0, 2026-07-21).
        x, y = scale_to_device(g.point[0], g.point[1], dw, dh)
        if g.kind == "tap":
            tap(worker.driver, x, y)
        else:
            long_press(worker.driver, x, y, duration=g.hold_duration_s)
    elif g.kind == "flick":
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
    ap.add_argument("--max-segments", type=int, default=None,
                    help="Stop after N recorded segments. Useful for bounded pilot runs "
                         "and for a supervisor that intentionally starts a fresh collector "
                         "process per segment; it does not by itself calibrate timing.")
    ap.add_argument("--no-reset", action="store_true",
                    help="Skip reset_position between gestures. Required for stationary "
                         "runs: reset is a tap, and its own rendered mark would land in "
                         "the next sample's window as an unlabelled touch.")
    ap.add_argument("--park-label", default=None,
                    help="Pin collection to a single named park (e.g. 'The Workshop') "
                         "instead of the SLS rotation. Sets the sample-dir park tag, so "
                         "training can select it with --data-match. Implies --no-rotate.")
    ap.add_argument("--align-video", action="store_true",
                    help="Pass --video to the aligner: one h264 clip per sample instead "
                         "of N PNGs (~150x smaller on static scenes, 1 inode not N).")
    ap.add_argument("--static-frac", type=float, default=0.0,
                    help="TRUE share [0,1] of fires that are STATIONARY touches — holds "
                         "(long_press, 0.1-1.5s) and taps, split ~80/20. These have an "
                         "unambiguous (x,y) plus a known onset and liftoff, with no "
                         "direction/speed ambiguity: the Model 1 MVP arm. 1.0 = a pure "
                         "hold/tap run.")
    ap.add_argument("--basic-holds", action="store_true",
                    help="Collect the additive basic Model-1 experiment: 80%% one-finger "
                         "holds uniformly 0.30-1.50s plus 20%% calibration-only taps. "
                         "No drags, multi-touch, or spin holds are emitted. Use with "
                         "--tap-calibrate and --no-reset.")
    ap.add_argument("--basic-hold-tap-frac", type=float, default=BASIC_HOLD_CALIBRATION_TAP_SHARE,
                    help="Calibration-only tap fraction for --basic-holds (default 0.20). "
                         "Taps are never admitted by the strict hold training loader.")
    ap.add_argument("--tap-calibrate", action="store_true",
                    help="Ask the offline aligner to require per-segment timing calibration "
                         "from the known-position tap arm. Intended for --static-frac MVP "
                         "collection; a rejected calibration preserves the source .mov.")
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
    ap.add_argument("--wait-for-align", action="store_true",
                    help="Run the post-segment aligner in the foreground instead of async. "
                         "Use for a bounded calibration pilot so its go/no-go result is visible.")
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

    if args.no_align and (args.tap_calibrate or args.wait_for_align):
        ap.error("--no-align cannot be combined with --tap-calibrate or --wait-for-align")

    # Validate the gesture mix BEFORE any device contact: a bad value would
    # otherwise surface as a per-gesture ValueError mid-run and crash-loop the
    # launchd job (wedge-adjacent churn on the XCTest recorder).
    spin_frac = min(1.0, max(0.0, args.spin_frac))
    if spin_frac != args.spin_frac:
        print(f"WARNING: --spin-frac {args.spin_frac} outside [0, 1]; clamped to {spin_frac}.")
        args.spin_frac = spin_frac
    static_frac = min(1.0, max(0.0, args.static_frac))
    if static_frac != args.static_frac:
        print(f"WARNING: --static-frac {args.static_frac} outside [0, 1]; clamped to {static_frac}.")
        args.static_frac = static_frac
    if args.spin_frac + args.static_frac > 1.0:
        raise SystemExit(f"--spin-frac ({args.spin_frac}) + --static-frac ({args.static_frac}) "
                         "exceeds 1.0 — they are true shares of all fires.")
    if (args.flick_frac + args.nslot_frac + args.recipe_frac <= 0
            and args.spin_frac <= 0 and args.static_frac <= 0):
        raise SystemExit("All gesture mixture weights are zero — nothing to sample "
                         "(--flick-frac/--nslot-frac/--recipe-frac/--spin-frac/--static-frac).")
    if args.num_gestures < 1:
        raise SystemExit(f"--num-gestures must be >= 1, got {args.num_gestures}")
    if args.basic_holds:
        if args.spin_frac != 0.0 or args.static_frac != 0.0 or args.use_spin:
            raise SystemExit("--basic-holds cannot be combined with --static-frac, --spin-frac, or --use-spin")
        if any(value != default for value, default in (
            (args.flick_frac, 0.6), (args.nslot_frac, 0.25), (args.recipe_frac, 0.15),
        )):
            raise SystemExit("--basic-holds cannot be combined with gesture-mixture fraction overrides")
        if not args.tap_calibrate:
            raise SystemExit("--basic-holds requires --tap-calibrate; uncalibrated hold clips are not admissible")
        if not args.no_reset:
            raise SystemExit("--basic-holds requires --no-reset; reset taps contaminate the next hold clip")
        if not 0.0 <= args.basic_hold_tap_frac < 1.0:
            raise SystemExit("--basic-hold-tap-frac must be in [0, 1) with --basic-holds")

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
    if args.park_label:
        # Single fixed park: the label only tags the sample dirs, so the park must
        # already be loaded on the device (the SLS rotation is prompt-driven and
        # would be meaningless here).
        cycle = [args.park_label]
        args.no_rotate = True
    elif args.start_park is not None:
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
          f"recipe={args.recipe_frac}; spin share={args.spin_frac} (guaranteed-hold slice); "
          f"static share={args.static_frac} (stationary hold/tap).")

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
        cmd = [sys.executable, str(_HERE / "align_xctest_traces.py"),
               "--segment", str(manifest_path), "--delete-mov"]
        if args.align_video:
            cmd.append("--video")
        if args.tap_calibrate:
            cmd.append("--tap-calibrate")
        if args.wait_for_align:
            # The stationary-touch MVP's calibration is an explicit go/no-go gate.
            # Keep it foregrounded so the operator sees accepted/rejected rather
            # than discovering an unaligned .mov after an async child exits.
            result = subprocess.run(cmd)
            if result.returncode != 0:
                raise RuntimeError(f"aligner rejected/failed {manifest_path.name} "
                                   f"(exit {result.returncode})")
            return
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    recovery_exit = False
    try:
        while not _STOP:
            if global_deadline and time.monotonic() > global_deadline:
                print("[collect_xctest] global --max-hours reached.")
                break
            if args.max_segments is not None and segment_idx >= args.max_segments:
                print(f"[collect_xctest] --max-segments {args.max_segments} reached.")
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
                    recovery_exit = True
                    break
                if not _recover_session(worker):
                    print("[collect_xctest] session unrecoverable — exit for supervisor restart.")
                    recovery_exit = True
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

                # --- foreground guard: catches True Skate being fully backgrounded
                # (e.g. a stray gesture opened iOS's share sheet and a later blind tap
                # landed on Files/Home). The pixel-based menu/editor guard below only
                # recognizes True Skate's OWN UI signatures, so it is structurally blind
                # to "a different app is on screen" — this is a cheap OS-level check
                # (no screenshot) that catches it directly. Reuses the same
                # ensure_foreground() the CMA-ES/PPO loops already rely on.
                if not args.no_gameplay_guard:
                    try:
                        if worker.ensure_foreground():
                            print(f"[seg {segment_idx}] True Skate was backgrounded — "
                                  f"relaunched (not logged)")
                            total_menu_skips += 1
                            seg_iter += 1
                            continue
                    except Exception as exc:  # noqa: BLE001 — never let the guard crash the run
                        print(f"[seg {segment_idx}] foreground check failed: {exc!r} — proceeding")

                try:
                    # reset_position is itself a TAP, and a tap renders its own mark
                    # ~1.06s later — which lands inside the next sample's window and
                    # would train the model on an unlabelled touch at (0.5, 0.056).
                    # Stationary gestures never move the board, so --no-reset both
                    # removes that contaminant and speeds the loop up.
                    if not args.no_reset:
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

                g = (sample_basic_hold_mixture(rng, tap_fraction=args.basic_hold_tap_frac) if args.basic_holds else
                     sample_mixture(rng, fracs=fracs, spin_frac=args.spin_frac,
                                    static_frac=args.static_frac,
                                    num_gestures=args.num_gestures,
                                    use_spin=args.use_spin, recipe_vectors=recipe_vectors))
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
                # --- post-gesture foreground check: a gesture that itself backgrounds
                # True Skate (e.g. lands on a share-sheet destination) must not be logged
                # as a clean gameplay sample. Recover immediately rather than waiting for
                # the next loop iteration's guard to notice.
                if not args.no_gameplay_guard:
                    try:
                        if worker.ensure_foreground():
                            total_menu_skips += 1
                            print(f"[seg {segment_idx}] gesture backgrounded True Skate "
                                  f"— relaunched, dropping sample (not logged)")
                            continue
                    except Exception as exc:  # noqa: BLE001 — never let the guard crash the run
                        print(f"[seg {segment_idx}] post-gesture foreground check failed: {exc!r} — proceeding")
                # --- post-gesture menu/editor check ---
                # A gesture can open non-gameplay UI after the pre-gesture guard.  The NEXT
                # reset may close it, so check before logging and drop the contaminated window.
                if not args.no_gameplay_guard:
                    try:
                        time.sleep(0.35)  # let any newly-opened UI render before scoring
                        _post_png = worker.driver.get_screenshot_as_png()
                        _post_editor = is_editor_frame(_post_png)
                        _post_menu = is_menu_frame(_post_png)
                        if _post_editor or _post_menu:
                            total_menu_skips += 1
                            _what = "park editor" if _post_editor else "replay/app menu"
                            print(f"[seg {segment_idx}] gesture opened the {_what} "
                                  f"— dropping sample (not logged)")
                            continue  # do NOT log a gesture that landed in non-gameplay UI
                    except Exception as exc:  # noqa: BLE001 — never let the guard crash the run
                        print(f"[seg {segment_idx}] post-gesture UI check failed: {exc!r} — proceeding")
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
                        "recipe": args.recipe_frac, "spin_frac": args.spin_frac,
                        "static_frac": args.static_frac},
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
    if recovery_exit:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
