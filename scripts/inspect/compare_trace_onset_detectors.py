"""Compare trace-onset detector versions on a frozen wide-window review set."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from scripts.inspect.trace_onset_detector import detect_v2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def load_frames(directory: Path) -> np.ndarray:
    paths = sorted(directory.glob("f*.png"))
    if not paths:
        raise ValueError(f"no decoded frames in {directory}")
    return np.asarray([np.asarray(Image.open(path).convert("RGB")) for path in paths])


def counts(rows: list[dict], kind: str, field: str) -> dict[str, int]:
    errors = [row[field] for row in rows if row["label"]["kind"] == kind]
    return {
        "count": len(errors),
        "exact": sum(error == 0 for error in errors),
        "within_one": sum(error is not None and abs(error) <= 1 for error in errors),
        "missed": sum(error is None for error in errors),
        "early_by_more_than_one": sum(error is not None and error < -1 for error in errors),
    }


def make_strip(
    frames: np.ndarray,
    absolute_start: int,
    human: int,
    detected: int | None,
    title: str,
) -> Image.Image:
    focus = human if detected is None else detected
    first = max(absolute_start, min(focus, human) - 5)
    last = min(absolute_start + len(frames) - 1, max(focus, human) + 3)
    shown = list(range(first, last + 1))
    sheet = Image.new("RGB", (100 * len(shown), 145), "#181818")
    draw = ImageDraw.Draw(sheet)
    draw.text((4, 3), title, fill="white")
    for column, frame_number in enumerate(shown):
        image = Image.fromarray(frames[frame_number - absolute_start]).resize((96, 96))
        sheet.paste(image, (column * 100, 42))
        tags = []
        if frame_number == human:
            tags.append("H")
        if frame_number == detected:
            tags.append("V2")
        colour = "#7fff7f" if frame_number == human else "#ffcc66"
        draw.text(
            (column * 100 + 2, 24),
            f"f{frame_number} {'/'.join(tags)}",
            fill=colour if tags else "white",
        )
    return sheet


def main() -> None:
    args = parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError(f"output directory is not empty: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    baseline_rows = json.loads((args.baseline_dir / "results.json").read_text())
    selection = json.loads((args.baseline_dir / "selection.json").read_text())
    rows = []
    for baseline in baseline_rows:
        source = args.baseline_dir / baseline["id"]
        frames = load_frames(source)
        gesture = baseline["gesture"]
        point = gesture.get("point") or gesture["waypoints"][0]
        crop_x, crop_y = baseline["crop_xywh"][:2]
        center_x = point[0] * 828 - crop_x
        center_y = point[1] * 1792 - crop_y
        relative = detect_v2(frames, center_x, center_y)
        start = baseline["window_frames"][0]
        detected = None if relative is None else start + relative
        human = baseline["label"]["frame"]
        row = {
            "id": baseline["id"],
            "kind": baseline["label"]["kind"],
            "human_frame": human,
            "v1_frame": baseline["detected_frame"],
            "v1_error_frames": baseline["error_frames"],
            "v2_frame": detected,
            "v2_error_frames": None if detected is None else detected - human,
            "window_frames": baseline["window_frames"],
        }
        rows.append({**baseline, **row})
        directory = args.out / baseline["id"]
        directory.mkdir()
        make_strip(
            frames,
            start,
            human,
            detected,
            f"{baseline['id']} | V2 {detected}; human {human}",
        ).save(directory / "sheet.png")

    summary = {
        version: {kind: counts(rows, kind, f"{version}_error_frames") for kind in ("calibration", "gesture")}
        for version in ("v1", "v2")
    }
    result = {
        "frozen_selection": selection,
        "v2_method": {
            "core_radius_pixels": 10,
            "background_ring_pixels": [20, 40],
            "immediate_local_brightness_increase": 4.0,
            "history_frames": 3,
            "lookahead_frames": 3,
            "confirmed_local_contrast_increase": 6.0,
        },
        "summary": summary,
        "rows": [
            {key: row[key] for key in ("id", "kind", "human_frame", "v1_frame", "v1_error_frames", "v2_frame", "v2_error_frames", "window_frames")}
            for row in rows
        ],
    }
    (args.out / "comparison.json").write_text(json.dumps(result, indent=2))

    page = """<!doctype html><meta charset=\"utf-8\"><title>Onset detector V1/V2</title>
<style>body{background:#181818;color:#eee;font:16px system-ui;margin:24px}section{border-top:1px solid #666;padding-top:10px;margin-top:28px}.strip{overflow:auto}img{image-rendering:pixelated}pre{white-space:pre-wrap}</style>
<h1>Trace-onset detector comparison</h1>
<p>The same 27 calibrations and frozen random sample of 24 swipes are used for both versions. Green H is the human-labelled first visible trace frame; amber V2 is the new detector. V2 subtracts nearby background brightness and uses three frames of look-ahead to confirm a persistent local glow while returning its candidate start.</p>
"""
    page += "<pre>" + html.escape(json.dumps(summary, indent=2)) + "</pre>"
    for row in rows:
        page += (
            f"<section><h2>{html.escape(row['id'])}</h2>"
            f"<p>Human {row['human_frame']}; V1 {row['v1_frame']} "
            f"({row['v1_error_frames']}); V2 {row['v2_frame']} "
            f"({row['v2_error_frames']}).</p>"
            f"<div class=\"strip\"><img src=\"{row['id']}/sheet.png\"></div></section>"
        )
    (args.out / "index.html").write_text(page)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
