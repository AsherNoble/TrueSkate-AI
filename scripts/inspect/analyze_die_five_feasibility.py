"""Phase 1 of M1-DIE5-COMPARE: die-five visibility (detector B) and scene motion per marker."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze_five_touch_probe import decode  # noqa: E402

from trueskate_ai.collection.die_five_calibration import detect_die_five_onset  # noqa: E402


def scene_change(frames, a: int, b: int) -> float:
    """Mean absolute grey difference outside a box around the marker (logical ±90 pt)."""
    ga, gb = (np.asarray(frames[i], dtype=np.float32) for i in (a, b))
    if ga.ndim == 3:
        ga, gb = ga[..., :3].mean(axis=2), gb[..., :3].mean(axis=2)
    h, w = ga.shape
    mask = np.ones_like(ga, dtype=bool)
    mask[int(h * (448 - 90) / 896):int(h * (448 + 90) / 896), int(w * (207 - 90) / 414):int(w * (207 + 90) / 414)] = False
    return float(np.abs(ga - gb)[mask].mean())


def analyze(metadata: Path, sheet_dir: Path) -> dict:
    data = json.loads(metadata.read_text())
    frames, times = decode(metadata.with_suffix(".mov"))
    started = data["recording"]["started_at_epoch_s"]
    search_s = data["events"][0]["call_start_epoch_s"] - started + 0.5
    markers = []
    for event in data["events"]:
        result = detect_die_five_onset(frames, times, command_s=search_s)
        record = {"marker": event["marker"], "votes": result.votes,
                  "point_frames": result.point_frames, "onset_frame": result.onset_frame}
        if result.accepted:
            f = result.onset_frame
            record["motion_after"] = round(scene_change(frames, f - 1, min(len(frames) - 1, f + 15)), 3)
            record["motion_before"] = round(scene_change(frames, max(0, f - 17), f - 1), 3)
            tiles = [cv2.resize(frames[i], (207, 448)) for i in (f - 1, f, min(len(frames) - 1, f + 15))]
            cv2.imwrite(str(sheet_dir / f"{metadata.parent.name}_{metadata.stem}_m{event['marker']}.png"), np.hstack(tiles))
            search_s = times[f] + 1.0
        markers.append(record)
    return {"recording": f"{metadata.parent.name}/{metadata.stem}", "hold_s": data["hold_s"],
            "gameplay_after": data["gameplay_after"], "frame_count": len(frames), "markers": markers}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    sheets = args.root / "sheets"
    sheets.mkdir(exist_ok=True)
    results = [analyze(p, sheets) for p in sorted(args.root.glob("xr*/recording_*.json"))]
    args.out.write_text(json.dumps(results, indent=1) + "\n")
    for r in results:
        for m in r["markers"]:
            print(r["recording"], m)


if __name__ == "__main__":
    main()
