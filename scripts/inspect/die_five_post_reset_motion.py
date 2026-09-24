#!/usr/bin/env python3
"""Exploratory: centre-area scene motion just before each start control.

For every segment manifest under the given roots, decode the 0.75 s before the
start control's WDA submission (the single-touch detector's reference span) with
the aligner's own windowing, and report mean absolute frame-to-frame grey change
inside ±90 logical points of the screen centre. Works for production two-control
segments and die-five experiment segments. No labels, no admission decisions.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

_path = Path(__file__).resolve().parents[1] / "collection" / "align_xctest_traces.py"
_spec = importlib.util.spec_from_file_location("die5_motion_aligner", _path)
ALIGN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ALIGN)

REFERENCE_S = 0.75
HALF_PT = 90


def centre_motion(manifest_path: Path) -> dict | None:
    manifest = json.loads(manifest_path.read_text())
    start = next((e for e in manifest["gestures"] if e.get("calibration_role") == "start"), None)
    mov = manifest_path.with_suffix(".mov")
    if start is None or not mov.exists() or "wda_submitted_epoch_s" not in start:
        return None
    approx = start["wda_submitted_epoch_s"] - manifest["started_at_epoch_s"]
    times = ALIGN._probe_video_frame_times(mov)
    frames, _ = ALIGN._decode_calibration_window(
        mov, command_video_s=approx, fps=30, reference_window_s=REFERENCE_S,
        search_after_s=0.0, resize_width=256, source_frame_times=times,
    )
    if len(frames) < 3:
        return None
    grey = [f.astype(np.float32).mean(axis=2) for f in frames]
    h, w = grey[0].shape
    ry, rx = int(h * HALF_PT / 896), int(w * HALF_PT / 414)
    crops = [g[h // 2 - ry:h // 2 + ry, w // 2 - rx:w // 2 + rx] for g in grey]
    diffs = [float(np.abs(b - a).mean()) for a, b in zip(crops, crops[1:])]
    return {"segment": str(manifest_path), "device": manifest["device"], "park": manifest["park"],
            "start_approx_video_s": round(approx, 3), "frames": len(frames),
            "centre_motion_mean": round(float(np.mean(diffs)), 3),
            "centre_motion_max": round(float(np.max(diffs)), 3)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = []
    for root in args.roots:
        for manifest in sorted(root.rglob("segment_*.json")):
            if manifest.name.endswith(".wda-action-timings.json"):
                continue
            row = centre_motion(manifest)
            if row is not None:
                rows.append(row)
                print(manifest.parent.name, manifest.stem, row["centre_motion_mean"], row["centre_motion_max"], flush=True)
    args.out.write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
