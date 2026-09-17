"""Bounded XR2 control-contamination test for the Model 1 linear sampler.

The experiment sends exactly ``--count`` separate linear swipes. Touch-downs
sit just outside the canonical safety regions while paths deliberately cross
and end inside controls. Recordings and screenshots are diagnostic evidence;
nothing is admitted to a training corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from trueskate_ai.collection.gameplay_filter import (  # noqa: E402
    is_bolt_modal_frame,
    is_editor_frame,
    is_menu_frame,
)
from trueskate_ai.collection.wda_action_timing import (  # noqa: E402
    WDAActionTimingCapture,
    validate_action_timing_report,
)
from trueskate_ai.collection.xctest_capture import XCTestScreenRecorder  # noqa: E402
from trueskate_ai.data.control_hitboxes import (  # noqa: E402
    CONTROL_MAP_LOGICAL_SIZE,
    CONTROL_START_EXCLUSIONS,
    CONTROL_START_MAP_VERSION,
    EFFECTIVE_CONTROL_HITBOXES,
    start_is_safe,
)
from trueskate_ai.sim.device import BUNDLE_ID, DEVICES, DeviceSession  # noqa: E402
from trueskate_ai.sim.gestures import scale_to_device  # noqa: E402
from trueskate_ai.sim.touch_actions import curved_drag, long_press, reset_position  # noqa: E402


DEFAULT_COUNT = 300
DEFAULT_SEGMENT_GESTURES = 25
MAX_RECORDING_S = 55.0
END_RESERVE_S = 4.0
RESET_SETTLE_S = 1.5
POST_GESTURE_S = 0.35
CALIBRATION_HOLD_S = 0.05


@dataclass(frozen=True)
class ProbeGesture:
    index: int
    target_control: str
    start: tuple[float, float]
    end: tuple[float, float]
    duration_s: float


def _outside_direction(name: str) -> tuple[float, float]:
    hitbox = next(item for item in EFFECTIVE_CONTROL_HITBOXES if item.name == name)
    x0, y0, _x1, _y1 = hitbox.rect
    if hitbox.transient:
        return 0.0, -1.0
    if y0 == 0.0:
        return 0.0, 1.0
    if x0 == 0.0:
        return 1.0, 0.0
    raise ValueError(f"unknown control-edge orientation: {name}")


def generate_boundary_sequence(count: int, seed: int) -> list[ProbeGesture]:
    """Generate balanced, reproducible swipes from safety to control interiors."""
    if count < 1:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    effective = {item.name: item for item in EFFECTIVE_CONTROL_HITBOXES}
    exclusions = {item.name: item for item in CONTROL_START_EXCLUSIONS}
    names = [item.name for item in EFFECTIVE_CONTROL_HITBOXES]
    order: list[str] = []
    while len(order) < count:
        cycle = list(names)
        rng.shuffle(cycle)
        order.extend(cycle)

    width, height = CONTROL_MAP_LOGICAL_SIZE
    results: list[ProbeGesture] = []
    for index, name in enumerate(order[:count]):
        hitbox = effective[name]
        exclusion = exclusions[name]
        x0, y0, x1, y1 = hitbox.rect
        # Keep endpoints away from inferred cell borders while covering different
        # parts of each effective control region.
        end_x = float(rng.uniform(x0 + 0.25 * (x1 - x0), x1 - 0.25 * (x1 - x0)))
        end_y = float(rng.uniform(y0 + 0.25 * (y1 - y0), y1 - 0.25 * (y1 - y0)))
        dx, dy = _outside_direction(name)
        offset_points = float(rng.uniform(3.0, 14.0))
        start_x = end_x + dx * offset_points / width
        start_y = end_y + dy * offset_points / height
        if dx > 0:
            start_x = max(start_x, exclusion.rect[2] + offset_points / width, end_x + 0.06)
        elif dy > 0:
            start_y = max(start_y, exclusion.rect[3] + offset_points / height, end_y + 0.06)
        else:
            start_y = min(start_y, exclusion.rect[1] - offset_points / height, end_y - 0.06)

        # The More button's inward ray overlaps the continuous left-side strip.
        # Continue toward play space until the origin clears every control.
        step_x, step_y = dx * 4.0 / width, dy * 4.0 / height
        for _ in range(256):
            if start_is_safe((start_x, start_y)):
                break
            start_x += step_x
            start_y += step_y
        else:
            raise RuntimeError(f"could not place safe start for {name}")
        if not (0.0 <= start_x <= 1.0 and 0.0 <= start_y <= 1.0):
            raise RuntimeError(f"safe start escaped screen for {name}: {(start_x, start_y)}")
        if not hitbox.contains((end_x, end_y)):
            raise AssertionError(f"endpoint missed target hitbox {name}")
        if math.hypot(end_x - start_x, end_y - start_y) < 0.05:
            raise AssertionError(f"probe path too short for {name}")
        results.append(ProbeGesture(
            index=index,
            target_control=name,
            start=(start_x, start_y),
            end=(end_x, end_y),
            duration_s=(0.25, 0.35, 0.50)[index % 3],
        ))
    return results


def _collector_running() -> str | None:
    result = subprocess.run(
        ["pgrep", "-f", r"collect_sls_xctest\.py.*--devices[= ]iPhone_XR2($| )"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None


def _editor_toolbar_frame(frame: bytes) -> bool:
    values = np.asarray(Image.open(io.BytesIO(frame)).convert("RGB"))
    roi = values[round(values.shape[0] * 0.84):, :, :3].astype(np.int16)
    red = (
        (roi[..., 0] >= 105)
        & (roi[..., 0] - roi[..., 1] >= 35)
        & (roi[..., 0] - roi[..., 2] >= 35)
    )
    return float(red.mean()) >= 0.018


def _screen_state(driver, png: bytes) -> dict:
    return {
        "foreground": driver.query_app_state(BUNDLE_ID) == 4,
        "menu": bool(is_menu_frame(png, allow_idle_navigation=True)),
        "editor": bool(is_editor_frame(png) or _editor_toolbar_frame(png)),
        "bolt_modal": bool(is_bolt_modal_frame(png)),
    }


def _is_contaminated(state: dict) -> bool:
    return (
        not state["foreground"]
        or state["menu"]
        or state["editor"]
        or state["bolt_modal"]
    )


def _write_review_index(out: Path, segments: list[dict]) -> None:
    rows = []
    for segment in segments:
        video = segment.get("video")
        if video:
            rows.append(
                f'<h2>Segment {segment["segment_index"]}</h2>'
                f'<video controls preload="metadata" width="414" src="{video}"></video>'
            )
        rows.append("<table><tr><th>Gesture</th><th>Target</th><th>Before</th><th>After</th><th>Flags</th></tr>")
        for event in segment.get("gestures", []):
            before = event["before_screenshot"]
            after = event["after_screenshot"]
            flags = json.dumps(event["post_state"], sort_keys=True)
            rows.append(
                "<tr>"
                f'<td>{event["gesture_index"]}</td><td>{event["target_control"]}</td>'
                f'<td><img loading="lazy" src="{before}" width="207"></td>'
                f'<td><img loading="lazy" src="{after}" width="207"></td>'
                f"<td><code>{flags}</code></td>"
                "</tr>"
            )
        rows.append("</table>")
    html = f"""<!doctype html><meta charset="utf-8">
<title>XR2 control-contamination review</title>
<style>body{{font:14px system-ui;background:#151515;color:#eee}}table{{border-collapse:collapse}}
td,th{{border:1px solid #555;padding:5px;vertical-align:top}}code{{white-space:pre-wrap}}</style>
<h1>XR2 control-contamination review</h1>
<p>Diagnostic only. Starts remain outside the safety map; paths end inside the named control.</p>
{''.join(rows)}
"""
    (out / "index.html").write_text(html)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--park", required=True)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--segment-gestures", type=int, default=DEFAULT_SEGMENT_GESTURES)
    parser.add_argument("--wda-timing-revision", required=True)
    args = parser.parse_args()
    if args.count != DEFAULT_COUNT:
        raise SystemExit(f"This approved experiment requires exactly {DEFAULT_COUNT} gestures")
    if not 1 <= args.segment_gestures <= DEFAULT_SEGMENT_GESTURES:
        raise SystemExit(f"--segment-gestures must be in [1, {DEFAULT_SEGMENT_GESTURES}]")
    if args.out.exists():
        raise SystemExit("Use a new output directory; experiment evidence is never overwritten")
    running = _collector_running()
    if running:
        raise SystemExit(f"XR2 collector is running: {running}")
    tunnel = subprocess.check_output(
        ["launchctl", "print", "system/com.trueskate.remotexpc-tunnel"], text=True
    )
    if "state = running" not in tunnel:
        raise SystemExit("Required recording tunnel is not running")

    sequence = generate_boundary_sequence(args.count, args.seed)
    args.out.mkdir(parents=True)
    (args.out / "screenshots").mkdir()
    cfg = next(item for item in DEVICES if item["name"] == "iPhone_XR2")
    worker = DeviceSession(cfg)
    segments: list[dict] = []
    overall_failure: dict | None = None
    next_index = 0
    try:
        worker.connect()
        driver = worker.driver
        if driver.query_app_state(BUNDLE_ID) != 4:
            raise RuntimeError("True Skate must be foreground before the test")
        preflight = driver.get_screenshot_as_png()
        (args.out / "preflight.png").write_bytes(preflight)
        preflight_state = _screen_state(driver, preflight)
        if _is_contaminated(preflight_state):
            raise RuntimeError(f"blocking preflight state: {preflight_state}")

        while next_index < len(sequence) and overall_failure is None:
            segment_index = len(segments)
            reset_position(driver, worker.device_w, worker.device_h)
            time.sleep(RESET_SETTLE_S)
            reset_cleared = driver.get_screenshot_as_png()
            reset_path = args.out / "screenshots" / f"segment_{segment_index:03d}_after_reset.png"
            reset_path.write_bytes(reset_cleared)
            reset_state = _screen_state(driver, reset_cleared)
            if _is_contaminated(reset_state):
                overall_failure = {"reason": "blocking state after pre-recording reset",
                                   "state": reset_state, "segment_index": segment_index}
                break

            recorder = XCTestScreenRecorder(driver, fps=30)
            timing = WDAActionTimingCapture(
                wda_port=int(cfg["wda_port"]),
                expected_revision=args.wda_timing_revision,
            )
            segment_dir = args.out / f"segment_{segment_index:03d}"
            segment_dir.mkdir()
            events: list[dict] = []
            action_count = 0
            timing_report = None
            capture_elapsed_s = None
            started = time.monotonic()
            recorder.start()  # one attempt only
            try:
                timing.start()
                long_press(
                    driver,
                    0.5 * worker.device_w,
                    0.5 * worker.device_h,
                    duration=CALIBRATION_HOLD_S,
                )
                action_count += 1
                segment_limit = min(len(sequence), next_index + args.segment_gestures)
                payload_deadline = started + MAX_RECORDING_S - END_RESERVE_S
                while next_index < segment_limit and time.monotonic() < payload_deadline:
                    spec = sequence[next_index]
                    before = driver.get_screenshot_as_png()
                    before_rel = f"screenshots/g{spec.index:03d}_before.png"
                    (args.out / before_rel).write_bytes(before)
                    before_state = _screen_state(driver, before)
                    if _is_contaminated(before_state):
                        overall_failure = {
                            "reason": "blocking state before gesture",
                            "gesture_index": spec.index,
                            "state": before_state,
                        }
                        break
                    points = [
                        scale_to_device(*spec.start, worker.device_w, worker.device_h),
                        scale_to_device(*spec.end, worker.device_w, worker.device_h),
                    ]
                    call_start = time.time()
                    curved_drag(driver, points, total_duration=spec.duration_s, easing=None)
                    call_end = time.time()
                    action_count += 1
                    time.sleep(POST_GESTURE_S)
                    after = driver.get_screenshot_as_png()
                    after_rel = f"screenshots/g{spec.index:03d}_after.png"
                    (args.out / after_rel).write_bytes(after)
                    post_state = _screen_state(driver, after)
                    event = {
                        "gesture_index": spec.index,
                        "target_control": spec.target_control,
                        "waypoints": [spec.start, spec.end],
                        "duration_s": spec.duration_s,
                        "t_call_start_epoch_s": call_start,
                        "t_call_end_epoch_s": call_end,
                        "before_screenshot": before_rel,
                        "after_screenshot": after_rel,
                        "pre_state": before_state,
                        "post_state": post_state,
                    }
                    events.append(event)
                    next_index += 1
                    if _is_contaminated(post_state):
                        overall_failure = {
                            "reason": "control contamination detected after gesture",
                            "gesture_index": spec.index,
                            "target_control": spec.target_control,
                            "state": post_state,
                        }
                        break
                if overall_failure is None:
                    long_press(
                        driver,
                        0.5 * worker.device_w,
                        0.5 * worker.device_h,
                        duration=CALIBRATION_HOLD_S,
                    )
                    action_count += 1
                timing_report = timing.stop()
                validate_action_timing_report(
                    timing_report,
                    expected_revision=args.wda_timing_revision,
                    expected_count=action_count,
                )
            finally:
                if timing.active:
                    timing.cleanup()
                capture_elapsed_s = time.monotonic() - started
                video_path = segment_dir / "segment.mov"
                if recorder.is_recording:
                    recording = recorder.stop_and_save(video_path)
                else:
                    recording = None
            if capture_elapsed_s > 60.0:
                overall_failure = {
                    "reason": "recording exceeded one-minute cap",
                    "segment_index": segment_index,
                    "elapsed_s": capture_elapsed_s,
                }
            if timing_report is not None:
                (segment_dir / "wda-action-timings.json").write_text(
                    json.dumps(timing_report, indent=2) + "\n"
                )
            segment_record = {
                "segment_index": segment_index,
                "video": f"segment_{segment_index:03d}/segment.mov",
                "recording": recording.summary() if recording else None,
                "capture_elapsed_s": capture_elapsed_s,
                "reset_state": reset_state,
                "action_count": action_count,
                "gestures": events,
            }
            (segment_dir / "manifest.json").write_text(
                json.dumps(segment_record, indent=2) + "\n"
            )
            segments.append(segment_record)
            _write_review_index(args.out, segments)

        result = {
            "experiment": "xr2-control-start-contamination",
            "device": "iPhone_XR2",
            "park": args.park,
            "training_admission": "never; bounded diagnostic only",
            "control_start_map_version": CONTROL_START_MAP_VERSION,
            "seed": args.seed,
            "planned_gestures": args.count,
            "completed_gestures": next_index,
            "segments": len(segments),
            "automatic_contamination_failures": 0 if overall_failure is None else 1,
            "failure": overall_failure,
            "visual_review": "pending",
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        (args.out / "sequence.json").write_text(
            json.dumps([asdict(item) for item in sequence], indent=2) + "\n"
        )
        print(json.dumps(result, indent=2), flush=True)
        if overall_failure is not None or next_index != args.count:
            raise SystemExit(2)
    finally:
        worker.disconnect()


if __name__ == "__main__":
    main()
