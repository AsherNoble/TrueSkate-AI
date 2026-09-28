#!/usr/bin/env python3
"""Build a viewer of end-point misses for visual classification (M1-DIAG).

Draws a seeded random sample of validation end misses (end error > 0.03) from
chosen parks in an ``autopsy_failures`` report, mixed with recovered controls
from the same parks, in shuffled order. Each clip shows real frames around the
commanded end time, with the commanded path and end (green) and the predicted
end (red). The operator classifies what the trace does at the green end and
exports JSON. Which clips are misses is written to a separate key file and is
not shown in the page.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import random
from pathlib import Path

import cv2
import numpy as np

CATEGORIES = [
    ("reaches", "Trace clearly reaches the green end"),
    ("faint", "Trace tip is faint or blends into the floor near the green end"),
    ("hidden", "Trace near the green end is hidden (board, skater, UI)"),
    ("short", "Trace visibly stops short of the green end"),
    ("other", "Other / unclear (add a note)"),
]
SCALE = 2
MAX_TILES = 10


def read_frames(sample: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(sample / "frames.mp4"))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames


def panel(sample: Path, record: dict) -> bytes:
    meta = json.loads((sample / "meta.json").read_text())
    frames = read_frames(sample)
    times = meta["frame_times"][:len(frames)]
    duration = float(meta["duration"])
    # The trail fades within a frame or two, so show the whole swipe: every
    # frame from just before onset to just after the commanded end (evenly
    # thinned to MAX_TILES), always keeping the frames either side of the end.
    window = [i for i, t in enumerate(times) if -0.1 <= t <= duration + 0.2]
    end_index = next((i for i, t in enumerate(times) if t >= duration), len(frames) - 1)
    keep = {window[0], window[-1], max(0, end_index - 1), end_index}
    if len(window) > MAX_TILES:
        picks = np.linspace(0, len(window) - 1, MAX_TILES - 2).round()
        keep |= {window[int(k)] for k in picks}
    else:
        keep |= set(window)
    chosen = sorted(keep)
    c, p = record["commanded"], record["predicted"]
    tiles = []
    for index in chosen:
        frame = cv2.resize(frames[index], None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_LINEAR)
        height, width = frame.shape[:2]

        def point(x: float, y: float) -> tuple[int, int]:
            return int(round(x * width)), int(round(y * height))

        start, end = point(c[0], c[1]), point(c[-3], c[-2])
        cv2.line(frame, start, end, (0, 200, 0), 1, cv2.LINE_AA)
        cv2.circle(frame, end, 9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.drawMarker(frame, point(p[-3], p[-2]), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
        label = f"{times[index]:+.2f}s"
        cv2.rectangle(frame, (0, 0), (width, 18), (0, 0, 0), -1)
        cv2.putText(frame, label, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1,
                    cv2.LINE_AA)
        tiles.append(frame)
    gap = np.full((tiles[0].shape[0], 4, 3), 40, np.uint8)
    strip = np.concatenate([part for tile in tiles for part in (tile, gap)][:-1], axis=1)
    ok, encoded = cv2.imencode(".jpg", strip, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise RuntimeError(f"could not encode {sample}")
    return encoded.tobytes()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--autopsy", type=Path, required=True)
    ap.add_argument("--corpus-root", type=Path, required=True)
    ap.add_argument("--park", action="append", required=True)
    ap.add_argument("--misses-per-park", type=int, default=15)
    ap.add_argument("--controls-per-park", type=int, default=5)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    report = json.loads(args.autopsy.read_text())
    if report["partition"] != "validation":
        raise SystemExit("only validation autopsies may be viewed")
    records = report["all_records"]
    rng = random.Random(args.seed)
    chosen = []
    for park in args.park:
        in_park = [r for r in records if r["park"] == park]
        misses = [r for r in in_park if r["end_error"] > .03]
        controls = [r for r in in_park if r["recovered"]]
        chosen += [("miss", r) for r in rng.sample(misses, args.misses_per_park)]
        chosen += [("control", r) for r in rng.sample(controls, args.controls_per_park)]
    rng.shuffle(chosen)

    args.out_dir.mkdir(parents=True, exist_ok=False)
    key, cards = [], []
    for order, (kind, record) in enumerate(chosen, 1):
        image = base64.b64encode(panel(args.corpus_root / record["sample"], record)).decode()
        key.append({"order": order, "kind": kind, "sample": record["sample"], "park": record["park"],
                    "device": record["device"], "gesture_duration": record["gesture_duration"],
                    "end_error": record["end_error"], "end_along": record["end_along"]})
        options = "".join(
            f'<label><input type="radio" name="c{order}" value="{value}"> {html.escape(text)}</label>'
            for value, text in CATEGORIES)
        cards.append(f'<section data-order="{order}"><h2>Clip {order}</h2>'
                     f'<img src="data:image/jpeg;base64,{image}" alt="clip {order}">'
                     f'<div class="opts">{options}</div>'
                     f'<input class="note" placeholder="note (optional)"></section>')
    (args.out_dir / "key.json").write_text(json.dumps({"seed": args.seed, "autopsy": report["checkpoint"],
                                                       "clips": key}, indent=1) + "\n")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>End Miss Review</title>
<style>
:root {{ --bg:#fafafa; --fg:#1a1a1a; --card:#fff; --line:#ddd; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#141414; --fg:#eee; --card:#1f1f1f; --line:#333; }} }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.4 system-ui, sans-serif; margin:0 auto;
       max-width:1400px; padding:16px; }}
section {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px;
          margin:0 0 16px; }}
img {{ max-width:100%; height:auto; display:block; margin:8px 0; }}
.opts label {{ display:block; padding:2px 0; }}
.note {{ width:100%; max-width:600px; margin-top:6px; padding:4px; }}
header {{ position:sticky; top:0; background:var(--bg); padding:8px 0; border-bottom:1px solid var(--line);
         margin-bottom:12px; }}
button {{ font-size:15px; padding:6px 14px; }}
</style></head><body>
<header><strong>End-point miss review</strong> — {len(chosen)} clips (Kansas City / Los Angeles, validation).
Green line and circle: the commanded path and end. Red ×: the model's predicted end. Frames run from
just before the swipe starts to just after it ends; labels are seconds since the swipe started. For each
clip, choose what the <em>trace</em> (the orange finger trail) does near the green end in any frame. <span id="count"></span> <button id="export">Export JSON</button></header>
{''.join(cards)}
<script>
function collect() {{
  return [...document.querySelectorAll('section')].map(s => {{
    const o = s.dataset.order, c = s.querySelector('input[type=radio]:checked');
    return {{order: +o, category: c ? c.value : null, note: s.querySelector('.note').value}};
  }});
}}
function update() {{
  const done = collect().filter(r => r.category).length;
  document.getElementById('count').textContent = done + '/{len(chosen)} classified';
}}
document.addEventListener('change', update); update();
document.getElementById('export').onclick = () => {{
  const blob = new Blob([JSON.stringify({{seed: {args.seed}, labels: collect()}}, null, 1)],
                        {{type: 'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = 'model1-end-miss-review-{args.seed}.json'; a.click();
}};
</script></body></html>
"""
    (args.out_dir / "viewer").mkdir()
    (args.out_dir / "viewer" / "index.html").write_text(page)
    print(json.dumps({"clips": len(chosen), "out": str(args.out_dir)}))


if __name__ == "__main__":
    main()
