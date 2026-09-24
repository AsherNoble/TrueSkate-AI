"""Three isolated XR2 four-finger visibility trials; never admits training data."""
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

# Exact XR2 logical points: a 66-point square centred at (207, 448).
FOUR_POINTS = ((174, 415), (240, 415), (174, 481), (240, 481))
EXPECTED_SIZE = (414, 896)
HOLD_S = 0.08


def gameplay_ok(png: bytes) -> bool:
    return not is_editor_frame(png) and not is_menu_frame(png, allow_idle_navigation=True)


def send_four(driver) -> None:
    fingers = []
    for x, y in FOUR_POINTS:
        finger = make_touch_pointer("calibration_four")
        finger.create_pointer_move(x=x, y=y, duration=0)
        finger.create_pointer_down()
        finger.create_pause(HOLD_S)
        finger.create_pointer_up(0)
        fingers.append(finger)
    # Equal-length pointer chains place all four downs on the same W3C action tick.
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
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.repetitions != 3:
        parser.error("this experiment is bounded to exactly three repetitions")
    if not all(point_is_safe((x / 414, y / 896)) for x, y in FOUR_POINTS):
        parser.error("a test point intersects a protected control region")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = select_devices(names=["iPhone_XR2"])[0]
    worker = DeviceSession(cfg)
    worker.connect()
    driver = worker.driver
    try:
        if (worker.device_w, worker.device_h) != EXPECTED_SIZE:
            raise RuntimeError(f"XR2 logical size changed: {(worker.device_w, worker.device_h)}")
        for rep in range(1, args.repetitions + 1):
            print(f"[repeat {rep}] preflight", flush=True)
            if worker.ensure_foreground():
                print(f"[repeat {rep}] True Skate was reactivated", flush=True)
            before = driver.get_screenshot_as_png()
            (args.out_dir / f"repeat_{rep}_before.png").write_bytes(before)
            if not gameplay_ok(before):
                print(f"[repeat {rep}] non-gameplay preflight; relaunching once", flush=True)
                relaunch_game(worker)
                before = driver.get_screenshot_as_png()
                (args.out_dir / f"repeat_{rep}_before_relaunch.png").write_bytes(before)
                if not gameplay_ok(before):
                    raise RuntimeError("True Skate did not return to gameplay; stopping")

            reset_position(driver, worker.device_w, worker.device_h)
            time.sleep(1.5)  # reset mark stays outside the recording
            recorder = XCTestScreenRecorder(driver, fps=30)
            events: list[dict] = []
            mov = args.out_dir / f"repeat_{rep}.mov"
            try:
                recorder.start()
                # XCTest can report a video start ~1.7 s after start() was called.
                # Leave enough lead-in for the four-finger command to be on film.
                time.sleep(2.5)
                four_start = time.time()
                send_four(driver)
                events.append({"kind": "simultaneous_four",
                               "points_logical": FOUR_POINTS,
                               "call_start_epoch_s": four_start,
                               "call_end_epoch_s": time.time()})
                time.sleep(2.0)
                after = driver.get_screenshot_as_png()
                (args.out_dir / f"repeat_{rep}_after.png").write_bytes(after)
                result = recorder.stop_and_save(mov)
            except Exception:
                recorder.abort()
                raise
            meta = {
                "experiment": "simultaneous-four-tap-visibility-v1",
                "device": worker.device_id,
                "park": "SLS 2015 Los Angeles (last operator-confirmed for preceding probe; screenshot saved)",
                "repetition": rep,
                "hold_s": HOLD_S,
                "events": events,
                "gameplay_after": gameplay_ok(after),
                "editor_after": is_editor_frame(after),
                "menu_after": is_menu_frame(after, allow_idle_navigation=True),
                "recording": result.summary(),
            }
            (args.out_dir / f"repeat_{rep}.json").write_text(json.dumps(meta, indent=2) + "\n")
            print(f"[repeat {rep}] saved {result.summary()['mb']} MB; "
                  f"gameplay_after={meta['gameplay_after']}, editor_after={meta['editor_after']}",
                  flush=True)
            if not meta["gameplay_after"] and rep < args.repetitions:
                print(f"[repeat {rep}] restoring gameplay before next attempt", flush=True)
                relaunch_game(worker)
    finally:
        worker.disconnect()


if __name__ == "__main__":
    main()
