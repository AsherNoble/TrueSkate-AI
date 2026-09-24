"""Bounded die-five calibration probe on one XR; never admits training data."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from trueskate_ai.collection.gameplay_filter import is_editor_frame, is_menu_frame
from trueskate_ai.collection.xctest_capture import XCTestScreenRecorder
from trueskate_ai.data.control_hitboxes import point_is_safe
from trueskate_ai.sim.device import BUNDLE_ID, DeviceSession, select_devices
from trueskate_ai.sim.touch_actions import (
    make_touch_pointer,
    perform_pointer_actions,
    reset_position,
    skip_loading_screen,
)


FIVE_POINTS = ((172, 413), (242, 413), (207, 448), (172, 483), (242, 483))
EXPECTED_SIZE = (414, 896)
HOLD_S = 0.08
RECORDINGS = 3
MARKERS_PER_RECORDING = 4
MAX_RECORDING_S = 50.0


def gameplay_ok(png: bytes) -> bool:
    return not is_editor_frame(png) and not is_menu_frame(png, allow_idle_navigation=True)


def send_five(driver, hold_s: float = HOLD_S) -> None:
    fingers = []
    for x, y in FIVE_POINTS:
        finger = make_touch_pointer("calibration_five")
        finger.create_pointer_move(x=x, y=y, duration=0)
        finger.create_pointer_down()
        finger.create_pause(hold_s)
        finger.create_pointer_up(0)
        fingers.append(finger)
    perform_pointer_actions(driver, fingers)


def relaunch_game(worker: DeviceSession) -> None:
    driver = worker.driver
    driver.terminate_app(BUNDLE_ID)
    time.sleep(1)
    driver.activate_app(BUNDLE_ID)
    time.sleep(1)
    skip_loading_screen(driver, worker.device_w, worker.device_h)
    time.sleep(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--device", default="iPhone_XR2", choices=("iPhone_XR", "iPhone_XR2"))
    parser.add_argument("--hold-s", type=float, default=HOLD_S)
    parser.add_argument("--recordings", type=int, default=RECORDINGS)
    parser.add_argument("--experiment", default="die-five-calibration-v1")
    args = parser.parse_args()
    if not all(point_is_safe((x / 414, y / 896)) for x, y in FIVE_POINTS):
        parser.error("a calibration point intersects a protected control region")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    worker = DeviceSession(select_devices(names=[args.device])[0])
    worker.connect()
    driver = worker.driver
    try:
        if (worker.device_w, worker.device_h) != EXPECTED_SIZE:
            raise RuntimeError(f"{args.device} logical size changed: {(worker.device_w, worker.device_h)}")
        for recording in range(1, args.recordings + 1):
            if worker.ensure_foreground():
                print(f"[recording {recording}] True Skate was reactivated", flush=True)
            before = driver.get_screenshot_as_png()
            (args.out_dir / f"recording_{recording}_before.png").write_bytes(before)
            if not gameplay_ok(before):
                relaunch_game(worker)
                before = driver.get_screenshot_as_png()
                (args.out_dir / f"recording_{recording}_before_relaunch.png").write_bytes(before)
                if not gameplay_ok(before):
                    raise RuntimeError("True Skate did not return to gameplay; stopping")

            reset_position(driver, worker.device_w, worker.device_h)
            time.sleep(1.5)  # reset remains outside the recording
            recorder = XCTestScreenRecorder(driver, fps=30)
            mov = args.out_dir / f"recording_{recording}.mov"
            events = []
            try:
                recorder.start()
                segment_start = time.monotonic()
                time.sleep(2.5)  # video can start ~1.7 s after start() returns
                for marker in range(1, MARKERS_PER_RECORDING + 1):
                    if time.monotonic() - segment_start > MAX_RECORDING_S - 8.0:
                        raise RuntimeError("bounded segment is approaching one minute")
                    call_start = time.time()
                    send_five(driver, args.hold_s)
                    events.append({
                        "kind": "simultaneous_five", "marker": marker,
                        "points_logical": FIVE_POINTS,
                        "call_start_epoch_s": call_start,
                        "call_end_epoch_s": time.time(),
                    })
                    time.sleep(1.0 if marker < MARKERS_PER_RECORDING else 2.0)
                after = driver.get_screenshot_as_png()
                (args.out_dir / f"recording_{recording}_after.png").write_bytes(after)
                if not gameplay_ok(after):
                    raise RuntimeError("True Skate left gameplay; withholding recording")
                if time.monotonic() - segment_start > MAX_RECORDING_S:
                    raise RuntimeError("bounded recording exceeded 50 seconds")
                result = recorder.stop_and_save(mov)
            except Exception:
                recorder.abort()
                raise
            meta = {
                "experiment": args.experiment,
                "device": worker.device_id,
                "park": "SLS 2015 Los Angeles (last operator-confirmed; screenshot saved)",
                "recording_number": recording,
                "hold_s": args.hold_s,
                "events": events,
                "gameplay_after": gameplay_ok(after),
                "recording": result.summary(),
            }
            (args.out_dir / f"recording_{recording}.json").write_text(
                json.dumps(meta, indent=2) + "\n"
            )
            print(f"[recording {recording}] saved {result.summary()['mb']} MB, "
                  f"{len(events)} markers", flush=True)
    finally:
        worker.disconnect()


if __name__ == "__main__":
    main()
