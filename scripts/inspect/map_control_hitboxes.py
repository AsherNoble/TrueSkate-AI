"""Map True Skate control hitboxes with isolated stationary-hold probes.

The tool is diagnostic only.  It never starts XCTest recording and never emits
training examples.  Each probe holds one point while the MJPEG stream records
the control's pressed state.  A paired hold in the nearby play area establishes
that control's own background-motion baseline.  The boundary search uses the
resulting control-crop difference and preserves a review sheet for every choice.

Run on the rig while all collectors are stopped::

    python scripts/inspect/map_control_hitboxes.py \
      --device iPhone_XR2 --out /absolute/new/output
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from trueskate_ai.collection.color_recorder import TimestampedColorRecorder
from trueskate_ai.collection.gameplay_filter import (
    is_bolt_modal_frame,
    is_editor_frame,
    is_menu_frame,
)
from trueskate_ai.data.control_hitboxes import (
    CONTROL_SAFETY_MARGIN_POINTS,
    CONTROL_START_MAP_VERSION,
)
from trueskate_ai.sim.device import BUNDLE_ID, DEVICES, DeviceSession
from trueskate_ai.sim.touch_actions import long_press, reset_position, skip_loading_screen


HISTORICAL_SPIN_POINT = (0.1832222850, 0.4213833183)
PRESS_SCORE_GAP_MIN = 2.0
BOTTOM_PRESS_SCORE_GAP_MIN = 8.0
SAFETY_MARGIN_POINTS = CONTROL_SAFETY_MARGIN_POINTS


@dataclass(frozen=True)
class ControlRay:
    name: str
    group: str
    center: tuple[float, float]
    axis: str
    inside: float
    outside: float


# The centres come from the live 414 x 896 XR2 screenshot.  The upper-left menu
# is not probed downwards because its ray intersects the Bolt control beneath it;
# it inherits the conservative maximum top-row boundary in the rendered map.
CONTROL_RAYS = (
    ControlRay("park_editor", "top", (0.273, 0.069), "y", 0.069, 0.200),
    ControlRay("reset", "top", (0.500, 0.069), "y", 0.069, 0.200),
    ControlRay("camera", "top", (0.725, 0.069), "y", 0.069, 0.200),
    ControlRay("rewind", "top", (0.940, 0.069), "y", 0.069, 0.200),
    ControlRay("bolt_challenges", "left", (0.058, 0.212), "x", 0.058, 0.300),
    ControlRay("fast_forward", "left", (0.058, 0.313), "x", 0.058, 0.300),
    ControlRay("spin", "left", (0.0604, 0.4040), "x", 0.0604, 0.300),
    ControlRay("me", "bottom", (0.100, 0.952), "y", 0.952, 0.780),
    ControlRay("skateparks", "bottom", (0.300, 0.952), "y", 0.952, 0.780),
    ControlRay("community", "bottom", (0.500, 0.952), "y", 0.952, 0.780),
    ControlRay("shop", "bottom", (0.700, 0.952), "y", 0.952, 0.780),
    ControlRay("settings", "bottom", (0.900, 0.952), "y", 0.952, 0.780),
)


def _collector_running(device: str) -> str | None:
    result = subprocess.run(
        ["pgrep", "-f", rf"collect_sls_xctest\.py.*--devices[= ]{device}($| )"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None


def _point(ray: ControlRay, value: float) -> tuple[float, float]:
    return (value, ray.center[1]) if ray.axis == "x" else (ray.center[0], value)


def _control_crop(ray: ControlRay, height: int, width: int) -> tuple[int, int, int, int]:
    if ray.group == "top":
        x0, x1 = ray.center[0] - 0.040, ray.center[0] + 0.040
        y0, y1 = 0.025, 0.110
    elif ray.group == "left":
        x0, x1 = 0.005, 0.115
        y0, y1 = ray.center[1] - 0.045, ray.center[1] + 0.045
    else:
        x0, x1 = ray.center[0] - 0.075, ray.center[0] + 0.075
        y0, y1 = 0.910, 0.995
    return (
        max(0, round(x0 * width)),
        max(0, round(y0 * height)),
        min(width, round(x1 * width)),
        min(height, round(y1 * height)),
    )


def press_scores(
    frames: list[np.ndarray],
    times: list[float],
    action_started: float,
    action_ended: float,
    ray: ControlRay,
) -> tuple[list[float], float]:
    if not frames:
        return [], 0.0
    before = [frame for frame, stamp in zip(frames, times) if stamp < action_started]
    reference = np.median(before or frames[:1], axis=0).astype(np.float32)
    x0, y0, x1, y1 = _control_crop(ray, reference.shape[0], reference.shape[1])
    reference_crop = reference[y0:y1, x0:x1]
    scores: list[float] = []
    for frame, stamp in zip(frames, times):
        release_tail = 0.30 if ray.group == "bottom" else 0.0
        if stamp < action_started or stamp > action_ended + release_tail:
            scores.append(0.0)
            continue
        delta = np.abs(frame[y0:y1, x0:x1].astype(np.float32) - reference_crop)
        # The 90th-percentile pixel change is sensitive to an icon highlight but
        # less dominated by a few compression blocks than a maximum.
        scores.append(float(np.percentile(delta.max(axis=2), 90)))
    return scores, max(scores, default=0.0)


def _save_review_sheet(
    path: Path,
    frames: list[np.ndarray],
    scores: list[int],
    ray: ControlRay,
    point: tuple[float, float],
    title: str,
) -> None:
    if not frames:
        Image.new("RGB", (828, 220), "#222").save(path)
        return
    picks = sorted({0, max(range(len(scores)), key=scores.__getitem__), len(frames) - 1})
    panes = []
    for index in picks:
        image = Image.fromarray(frames[index]).resize((207, 448))
        draw = ImageDraw.Draw(image)
        draw.ellipse(
            (
                point[0] * 207 - 4,
                point[1] * 448 - 4,
                point[0] * 207 + 4,
                point[1] * 448 + 4,
            ),
            outline="#44ddff",
            width=2,
        )
        x0, y0, x1, y1 = _control_crop(ray, 448, 207)
        draw.rectangle((x0, y0, x1, y1), outline="#ffdd33", width=1)
        draw.rectangle((0, 0, 207, 24), fill="#111")
        draw.text((4, 4), f"frame {index} score {scores[index]}", fill="white")
        panes.append(image)
    sheet = Image.new("RGB", (207 * len(panes), 486), "#111")
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 462), title, fill="white")
    for column, pane in enumerate(panes):
        sheet.paste(pane, (column * 207, 0))
    sheet.save(path, quality=90)


def _relaunch(worker: DeviceSession) -> None:
    driver = worker.driver
    try:
        driver.terminate_app(BUNDLE_ID)
    except Exception:
        pass
    time.sleep(0.8)
    driver.activate_app(BUNDLE_ID)
    time.sleep(1.2)
    skip_loading_screen(driver, worker.device_w, worker.device_h)
    time.sleep(1.0)


def _editor_toolbar_frame(frame: bytes | np.ndarray) -> bool:
    if isinstance(frame, bytes):
        values = np.asarray(Image.open(io.BytesIO(frame)).convert("RGB"))
    else:
        values = np.asarray(frame)
    roi = values[round(values.shape[0] * 0.84):, :, :3].astype(np.int16)
    red = (
        (roi[..., 0] >= 105)
        & (roi[..., 0] - roi[..., 1] >= 35)
        & (roi[..., 0] - roi[..., 2] >= 35)
    )
    return float(red.mean()) >= 0.018


def _ensure_gameplay(worker: DeviceSession) -> None:
    driver = worker.driver
    if driver.query_app_state(BUNDLE_ID) != 4:
        _relaunch(worker)
    screenshot = driver.get_screenshot_as_png()
    if _editor_toolbar_frame(screenshot):
        # The editor's close X is at the right edge of its red toolbar.
        driver.tap([(0.973 * worker.device_w, 0.862 * worker.device_h)])
        time.sleep(0.8)
        screenshot = driver.get_screenshot_as_png()
    if (
        _editor_toolbar_frame(screenshot)
        or is_editor_frame(screenshot)
        or is_menu_frame(screenshot, allow_idle_navigation=True)
        or is_bolt_modal_frame(screenshot)
    ):
        _relaunch(worker)
        screenshot = driver.get_screenshot_as_png()
        if _editor_toolbar_frame(screenshot):
            driver.tap([(0.973 * worker.device_w, 0.862 * worker.device_h)])
            time.sleep(0.8)
            screenshot = driver.get_screenshot_as_png()
    if (
        driver.query_app_state(BUNDLE_ID) != 4
        or _editor_toolbar_frame(screenshot)
        or is_editor_frame(screenshot)
        or is_menu_frame(screenshot, allow_idle_navigation=True)
        or is_bolt_modal_frame(screenshot)
    ):
        raise RuntimeError("Could not restore verified gameplay before the next hitbox probe")


def run_probe(
    worker: DeviceSession,
    ray: ControlRay,
    value: float,
    probe_index: int,
    out: Path,
) -> dict:
    _ensure_gameplay(worker)
    point = _point(ray, value)
    recorder = TimestampedColorRecorder()
    recorder.start(worker.mjpeg_url, resize_width=414)
    time.sleep(0.25)
    action_started = time.monotonic()
    error = None
    try:
        long_press(
            worker.driver,
            point[0] * worker.device_w,
            point[1] * worker.device_h,
            duration=0.70,
        )
    except Exception as exc:  # preserve failed diagnostic evidence
        error = f"{type(exc).__name__}: {exc}"
    action_ended = time.monotonic()
    time.sleep(0.35)
    frames, times = recorder.stop()
    scores, peak = press_scores(frames, times, action_started, action_ended, ray)
    post = worker.driver.get_screenshot_as_png()
    editor_toolbar_during = any(_editor_toolbar_frame(frame) for frame in frames)
    blocking = {
        "menu": bool(is_menu_frame(post, allow_idle_navigation=True)),
        "editor": bool(is_editor_frame(post) or _editor_toolbar_frame(post)),
        "editor_toolbar_during": editor_toolbar_during,
        "bolt_modal": bool(is_bolt_modal_frame(post)),
        "foreground": worker.driver.query_app_state(BUNDLE_ID) == 4,
    }
    name = f"{probe_index:03d}-{ray.name}-{ray.axis}{value:.6f}"
    _save_review_sheet(
        out / "reviews" / f"{name}.jpg",
        frames,
        scores,
        ray,
        point,
        f"{ray.name} {ray.axis}={value:.6f} press-score={peak:.2f}",
    )
    record = {
        "id": name,
        "control": ray.name,
        "group": ray.group,
        "axis": ray.axis,
        "value": value,
        "point": point,
        "press_score": round(peak, 3),
        "frame_count": len(frames),
        "blocking": blocking,
        "error": error,
    }
    print(json.dumps(record), flush=True)
    # UI-consumed bottom touches and persistent overlays need a full return to
    # gameplay.  Otherwise a reset prevents accumulated board movement from
    # changing the next probe's conditions.
    if ray.group == "bottom":
        _relaunch(worker)
    elif any(
        blocking[key] for key in ("menu", "editor", "bolt_modal")
    ) or not blocking["foreground"]:
        _ensure_gameplay(worker)
    else:
        reset_position(worker.driver, worker.device_w, worker.device_h)
        time.sleep(0.35)
        _ensure_gameplay(worker)
    return record


def _rectangle_map(boundaries: dict[str, float]) -> list[dict]:
    top_bottom = max(boundaries[name] for name in ("park_editor", "reset", "camera", "rewind"))
    top_centres = (0.058, 0.273, 0.500, 0.725, 0.940)
    top_names = ("more", "park_editor", "reset", "camera", "rewind")
    separators = [(a + b) / 2 for a, b in zip(top_centres, top_centres[1:])]
    x_edges = (0.0, *separators, 1.0)
    rects = []
    for i, name in enumerate(top_names):
        bottom = top_bottom if name == "more" else boundaries[name]
        rects.append({
            "name": name,
            "source": "measured lower edge; horizontal split inferred from icon centres",
            "rect": [x_edges[i], 0.0, x_edges[i + 1], bottom],
        })
    left_names = ("bolt_challenges", "fast_forward", "spin")
    left_centres = (0.212, 0.313, 0.404)
    y_edges = (top_bottom, (left_centres[0] + left_centres[1]) / 2,
               (left_centres[1] + left_centres[2]) / 2, 0.465)
    for i, name in enumerate(left_names):
        right = boundaries[name]
        if name == "spin":
            right = max(right, HISTORICAL_SPIN_POINT[0])
            y_edges = (*y_edges[:-1], max(y_edges[-1], HISTORICAL_SPIN_POINT[1]))
        rects.append({"name": name, "source": "measured right edge; vertical split inferred from icon centres",
                      "rect": [0.0, y_edges[i], right, y_edges[i + 1]]})
    bottom_names = ("me", "skateparks", "community", "shop", "settings")
    for i, name in enumerate(bottom_names):
        rects.append({"name": name,
                      "source": "conservative upper edge inferred; transient row kept strict",
                      "rect": [i / 5, boundaries[name], (i + 1) / 5, 1.0]})
    return rects


def _inflate(rect: list[float], width: float, height: float) -> list[float]:
    x0, y0, x1, y1 = rect
    return [
        max(0.0, x0 - SAFETY_MARGIN_POINTS / width),
        max(0.0, y0 - SAFETY_MARGIN_POINTS / height),
        min(1.0, x1 + SAFETY_MARGIN_POINTS / width),
        min(1.0, y1 + SAFETY_MARGIN_POINTS / height),
    ]


def render_overlay(baseline: bytes, rects: list[dict], width: float, height: float) -> Image.Image:
    base = Image.open(io.BytesIO(baseline)).convert("RGBA")
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 21)
    except OSError:
        font = ImageFont.load_default()
        title_font = font
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for item in rects:
        measured = item["rect"]
        safety = _inflate(measured, width, height)
        item["safety_rect"] = safety
        measured_px = tuple(round(v * (base.width if i % 2 == 0 else base.height))
                            for i, v in enumerate(measured))
        safety_px = tuple(round(v * (base.width if i % 2 == 0 else base.height))
                          for i, v in enumerate(safety))
        draw.rectangle(measured_px, fill=(255, 55, 55, 42), outline=(255, 55, 55, 255), width=4)
        # Pillow has no dashed rectangle primitive; draw each edge explicitly.
        x0, y0, x1, y1 = safety_px
        dash = 12
        for x in range(x0, x1, dash * 2):
            draw.line((x, y0, min(x + dash, x1), y0), fill=(255, 220, 40, 255), width=4)
            draw.line((x, y1, min(x + dash, x1), y1), fill=(255, 220, 40, 255), width=4)
        for y in range(y0, y1, dash * 2):
            draw.line((x0, y, x0, min(y + dash, y1)), fill=(255, 220, 40, 255), width=4)
            draw.line((x1, y, x1, min(y + dash, y1)), fill=(255, 220, 40, 255), width=4)
        label = item["name"] + "*"
        label_x = measured_px[0] + 5
        label_y = min(base.height - 25, measured_px[1] + 5)
        text_box = draw.textbbox((label_x, label_y), label, font=font)
        draw.rectangle((text_box[0] - 3, text_box[1] - 2, text_box[2] + 3, text_box[3] + 2),
                       fill=(0, 0, 0, 190))
        draw.text((label_x, label_y), label, fill="white", font=font)
    composed = Image.alpha_composite(base, overlay)
    panel_width = 780
    legend_height = 172
    legend = Image.new("RGBA", (base.width + panel_width, legend_height), (18, 18, 18, 255))
    legend_draw = ImageDraw.Draw(legend)
    legend_draw.text((18, 10), "XR2 CONTROL START EXCLUSION MAP",
                     fill="white", font=title_font)
    legend_draw.line((18, 43, 90, 43), fill=(255, 55, 55), width=4)
    legend_draw.text((100, 34), "solid red = effective hitbox estimate", fill="white", font=font)
    for x in range(18, 90, 24):
        legend_draw.line((x, 72, min(x + 12, 90), 72), fill=(255, 220, 40), width=4)
    legend_draw.text((100, 63), "dashed yellow = enforced +16-point start exclusion",
                     fill="white", font=font)
    legend_draw.text((18, 92), "Only the touch-down must stay outside yellow; a moving gesture may cross or end inside.",
                     fill=(100, 220, 255), font=font)
    legend_draw.text((18, 116), "Bottom row is transient, but this profile blocks it at all times.",
                     fill=(255, 225, 80), font=font)
    legend_draw.text((18, 140), "* One axis is measured; the other follows the visible control-cell split. More inherits top depth.",
                     fill=(255, 170, 220), font=font)
    canvas = Image.new("RGBA", (base.width + panel_width, base.height + legend.height),
                       (18, 18, 18, 255))
    canvas.paste(legend, (0, 0))
    canvas.paste(composed, (0, legend.height))
    panel_draw = ImageDraw.Draw(canvas)
    panel_x = base.width + 22
    panel_draw.text((panel_x, legend.height + 16), "Normalised start exclusions", fill="white", font=title_font)
    y = legend.height + 56
    for item in rects:
        measured = ", ".join(f"{value:.4f}" for value in item["rect"])
        safety = ", ".join(f"{value:.4f}" for value in item["safety_rect"])
        panel_draw.text((panel_x, y), item["name"] + "*", fill=(255, 130, 130), font=font)
        panel_draw.text((panel_x, y + 23), f"hitbox  [{measured}]", fill="white", font=font)
        panel_draw.text((panel_x, y + 46), f"safety  [{safety}]", fill=(255, 225, 80), font=font)
        if item.get("uncertainty"):
            panel_draw.text((panel_x, y + 69), "UNCERTAIN: " + item["uncertainty"][:46],
                            fill=(255, 140, 220), font=font)
            y += 98
        else:
            y += 78
    return canvas.convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="iPhone_XR2")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=7)
    args = parser.parse_args()
    if args.device != "iPhone_XR2":
        raise SystemExit("This bounded experiment is approved for iPhone_XR2 only")
    if args.out.exists():
        raise SystemExit("Use a new output path; diagnostic evidence is never overwritten")
    running = _collector_running(args.device)
    if running:
        raise SystemExit(f"Collector is running for {args.device}: {running}")
    args.out.mkdir(parents=True)
    (args.out / "reviews").mkdir()
    cfg = next(d for d in DEVICES if d["name"] == args.device)
    worker = DeviceSession(cfg)
    records: list[dict] = []
    try:
        worker.connect()
        if worker.driver.query_app_state(BUNDLE_ID) != 4:
            raise RuntimeError("True Skate must be foreground before probing")
        baseline = worker.driver.get_screenshot_as_png()
        (args.out / "baseline.png").write_bytes(baseline)
        probe_index = 0
        boundaries: dict[str, float] = {}
        uncertainty: dict[str, str] = {}
        for ray in CONTROL_RAYS:
            outside_record = run_probe(worker, ray, ray.outside, probe_index, args.out)
            records.append(outside_record); probe_index += 1
            inside_record = run_probe(worker, ray, ray.inside, probe_index, args.out)
            records.append(inside_record); probe_index += 1
            outside_score = outside_record["press_score"]
            inside_score = inside_record["press_score"]
            if ray.name == "park_editor":
                outside_record["activated"] = bool(
                    outside_record["blocking"]["editor_toolbar_during"]
                    or outside_record["blocking"]["editor"]
                )
                inside_record["activated"] = bool(
                    inside_record["blocking"]["editor_toolbar_during"]
                    or inside_record["blocking"]["editor"]
                )
                outside_score = 0.0 if not outside_record["activated"] else 100.0
                inside_score = 100.0 if inside_record["activated"] else 0.0
            score_gap = inside_score - outside_score
            required_gap = (
                BOTTOM_PRESS_SCORE_GAP_MIN if ray.group == "bottom" else PRESS_SCORE_GAP_MIN
            )
            if score_gap < required_gap:
                uncertainty[ray.name] = (
                    f"control-centre score {inside_score:.2f} did not separate from "
                    f"play-area score {outside_score:.2f}; "
                    "boundary is conservative and requires human review"
                )
                if ray.axis == "x":
                    boundaries[ray.name] = max(ray.inside, ray.outside)
                elif ray.group == "top":
                    boundaries[ray.name] = max(ray.inside, ray.outside)
                else:
                    # The visible navigation strip begins at ~0.92.  Use 0.90
                    # as a conservative inferred touch edge; its later 16-point
                    # margin reaches ~0.882, matching the current 0.88 guard.
                    boundaries[ray.name] = 0.90
                continue
            threshold = (inside_score + outside_score) / 2
            inside_record["activated"] = True
            outside_record["activated"] = False
            inside_record["activation_threshold"] = threshold
            outside_record["activation_threshold"] = threshold
            inside, outside = ray.inside, ray.outside
            for _ in range(args.iterations):
                value = (inside + outside) / 2
                record = run_probe(worker, ray, value, probe_index, args.out)
                records.append(record); probe_index += 1
                record["activation_threshold"] = threshold
                record["activated"] = (
                    bool(
                        record["blocking"]["editor_toolbar_during"]
                        or record["blocking"]["editor"]
                    )
                    if ray.name == "park_editor"
                    else record["press_score"] >= threshold
                )
                if record["activated"]:
                    inside = value
                else:
                    outside = value
            # Midpoint is the observed transition; top/left hitboxes occupy the
            # lower coordinate side, while the bottom navigation occupies the upper.
            boundaries[ray.name] = (inside + outside) / 2
        rects = _rectangle_map(boundaries)
        for item in rects:
            if item["name"] in uncertainty:
                item["uncertainty"] = uncertainty[item["name"]]
            elif item["name"] == "more":
                item["uncertainty"] = "top extent inherited; direct vertical ray would cross Bolt"
        overlay = render_overlay(baseline, rects, worker.device_w, worker.device_h)
        overlay.save(args.out / "xr2-hitbox-review.png")
        result = {
            "experiment": "xr2-control-hitbox-map",
            "training_admission": "never; diagnostic only",
            "control_start_map_version": CONTROL_START_MAP_VERSION,
            "enforcement": "gesture origin only; paths may cross or end inside controls",
            "bottom_navigation_policy": "transient in game; conservatively enforced at all times",
            "device": args.device,
            "logical_size": [worker.device_w, worker.device_h],
            "press_score_gap_min": PRESS_SCORE_GAP_MIN,
            "bottom_press_score_gap_min": BOTTOM_PRESS_SCORE_GAP_MIN,
            "safety_margin_logical_points": SAFETY_MARGIN_POINTS,
            "historical_spin_point": HISTORICAL_SPIN_POINT,
            "boundaries": boundaries,
            "uncertainty": uncertainty,
            "hitboxes": rects,
            "probes": records,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        (args.out / "results.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({"out": str(args.out), "boundaries": boundaries,
                          "uncertainty": uncertainty}, indent=2), flush=True)
    finally:
        worker.disconnect()


if __name__ == "__main__":
    main()
