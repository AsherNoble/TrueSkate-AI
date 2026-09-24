"""Three isolated XR2 two-finger visibility trials; never admits training data."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from selenium.webdriver.common.action_chains import ActionChains

from trueskate_ai.collection.gameplay_filter import is_editor_frame, is_menu_frame
from trueskate_ai.collection.xctest_capture import XCTestScreenRecorder
from trueskate_ai.data.control_hitboxes import point_is_safe
from trueskate_ai.sim.device import BUNDLE_ID, DeviceSession, select_devices
from trueskate_ai.sim.touch_actions import long_press, make_touch_pointer, reset_position, skip_loading_screen

BASELINE = (0.5, 0.5)
PAIR = ((0.46, 0.5), (0.54, 0.5))
HOLD_S = 0.08


def gameplay_ok(png: bytes) -> bool:
    return not is_editor_frame(png) and not is_menu_frame(png, allow_idle_navigation=True)


def send_pair(driver, *, width: float, height: float) -> None:
    fingers = []
    for x, y in PAIR:
        finger = make_touch_pointer("calibration_pair")
        finger.create_pointer_move(x=x * width, y=y * height, duration=0)
        finger.create_pointer_down()
        finger.create_pause(HOLD_S)
        finger.create_pointer_up(0)
        fingers.append(finger)
    # Equal-length pointer chains place both downs on the same W3C action tick.
    ActionChains(driver, devices=fingers).perform()


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
    if not all(point_is_safe(point) for point in (BASELINE, *PAIR)):
        parser.error("a test point intersects a protected control region")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = select_devices(names=["iPhone_XR2"])[0]
    worker = DeviceSession(cfg)
    worker.connect()
    driver = worker.driver
    try:
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
                # Leave enough lead-in that even the baseline touch is on film.
                time.sleep(2.5)
                baseline_start = time.time()
                long_press(driver, BASELINE[0] * worker.device_w,
                           BASELINE[1] * worker.device_h, duration=HOLD_S)
                events.append({"kind": "single_control", "points": [BASELINE],
                               "call_start_epoch_s": baseline_start,
                               "call_end_epoch_s": time.time()})
                time.sleep(2.0)
                pair_start = time.time()
                send_pair(driver, width=worker.device_w, height=worker.device_h)
                events.append({"kind": "simultaneous_pair", "points": PAIR,
                               "call_start_epoch_s": pair_start,
                               "call_end_epoch_s": time.time()})
                time.sleep(2.0)
                after = driver.get_screenshot_as_png()
                (args.out_dir / f"repeat_{rep}_after.png").write_bytes(after)
                result = recorder.stop_and_save(mov)
            except Exception:
                recorder.abort()
                raise
            meta = {
                "experiment": "simultaneous-two-tap-visibility-v1",
                "device": worker.device_id,
                "park": "SLS 2015 Los Angeles (operator-confirmed; screenshot saved)",
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
