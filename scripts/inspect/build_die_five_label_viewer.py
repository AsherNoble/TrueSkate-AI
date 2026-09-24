#!/usr/bin/env python3
"""Select M1-DIE5-COMPARE markers for blind human labelling and build the viewer.

Selection (protocol phase 3 as amended): every start/end marker, every mid
die-five marker where detectors A and B disagree or either has no result, a seeded 25% (min 40) of agreeing die-five
markers, and a seeded 25% (min 20) of single-touch markers for the corner check.
Each item is a frame-exact native-resolution crop around the screen centre.
Window placement depends only on WDA submission time plus seeded jitter, never
on detector output. The answer key is written outside the served directory.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BEFORE_S = 0.75
AFTER_S = 2.0
JITTER_S = 0.3
CROP_PT = 90  # half-width in logical points around (207, 448)


def select(analysis: list[dict], rng: random.Random) -> list[dict]:
    disagree, agree, single = [], [], []
    for seg in analysis:
        for m in seg["markers"]:
            item = {"segment": seg["segment"], **m}
            if m["marker"] == "single":
                single.append(item)
            elif m["role"] in ("start", "end"):
                disagree.append(item | {"stratum": "anchor_all_labelled"})
            elif m.get("a_frame") is None or m.get("b_frame") is None or m["a_frame"] != m["b_frame"]:
                disagree.append(item | {"stratum": "disagree_or_no_result"})
            else:
                agree.append(item | {"stratum": "agree"})
    def sample(pool, frac, minimum, stratum):
        k = min(len(pool), max(minimum, math.ceil(frac * len(pool))))
        return [p | {"stratum": stratum} for p in rng.sample(pool, k)]
    chosen = disagree + sample(agree, 0.25, 40, "agree") + sample(single, 0.25, 20, "single")
    rng.shuffle(chosen)
    return chosen


def frame_times(mov: Path) -> list[float]:
    """The aligner's own probe, so global frame indices match the analysis."""
    import importlib.util
    path = HERE.parent / "collection" / "align_xctest_traces.py"
    spec = importlib.util.spec_from_file_location("die5_viewer_aligner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    times = module._probe_video_frame_times(mov)
    count = int(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(mov)]).strip())
    if len(times) != count:
        raise RuntimeError(f"{mov}: {len(times)} timestamps for {count} decoded frames")
    return times


def extract(mov: Path, first: int, last: int, dest: Path) -> int:
    dest.mkdir(parents=True)
    w, h = 828, 1792
    half = CROP_PT * 2
    crop = f"crop={2 * half}:{2 * half}:{w // 2 - half}:{h // 2 - half}"
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-i", str(mov),
        "-vf", f"select=between(n\\,{first}\\,{last}),{crop}", "-fps_mode", "passthrough",
        "-q:v", "2", str(dest / "%03d.jpg")], check=True)
    count = len(list(dest.glob("*.jpg")))
    if count != last - first + 1:
        raise RuntimeError(f"{mov}: extracted {count} frames, expected {last - first + 1}")
    return count


def build(analysis_path: Path, out: Path, key_path: Path, seed: int) -> int:
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"output must be empty: {out}")
    analysis = json.loads(analysis_path.read_text())
    rng = random.Random(seed)
    chosen = select(analysis, rng)
    times_cache: dict[str, list[float]] = {}
    samples, key = [], []
    for order, item in enumerate(chosen):
        mov = Path(item["segment"]).with_suffix(".mov")
        times = times_cache.setdefault(str(mov), frame_times(mov))
        jitter = rng.uniform(-JITTER_S, JITTER_S)
        start_s = item["approx_video_s"] - BEFORE_S + jitter
        stop_s = item["approx_video_s"] + AFTER_S + jitter
        idx = [i for i, t in enumerate(times) if start_s <= t <= stop_s]
        item_id = f"{order:03d}"
        n = extract(mov, idx[0], idx[-1], out / "assets" / item_id)
        samples.append({"id": item_id, "frames": [f"assets/{item_id}/{i:03d}.jpg" for i in range(1, n + 1)]})
        key.append({"id": item_id, "first_global_frame": idx[0], "n_frames": n,
                    "segment": item["segment"], "wda_action_sequence": item["wda_action_sequence"],
                    "role": item["role"], "marker": item["marker"], "stratum": item["stratum"],
                    "a_frame": item.get("a_frame"), "b_frame": item.get("b_frame")})
    payload = json.dumps({"seed": str(seed), "samples": samples}, separators=(",", ":")).replace("<", "\\u003c")
    (out / "index.html").write_text((HERE / "die_five_label_viewer.html").read_text().replace("__PAYLOAD__", payload))
    key_path.write_text(json.dumps({"schema": "die5-compare-selection-v1", "seed": seed,
                                    "analysis": str(analysis_path), "items": key}, indent=1) + "\n")
    return len(samples)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--key", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    if args.key.resolve().is_relative_to(args.out.resolve()):
        raise SystemExit("--key must be outside the served --out directory")
    n = build(args.analysis.resolve(), args.out.resolve(), args.key.resolve(), args.seed)
    print(f"Wrote {args.out / 'index.html'} with {n} blinded items; key {args.key}")


if __name__ == "__main__":
    main()
