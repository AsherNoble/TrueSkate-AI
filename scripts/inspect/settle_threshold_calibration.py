#!/usr/bin/env python3
"""Reproduce the settle-threshold calibration (M1-SETTLE-20260925).

For each recording in a pre-start motion JSON (``die_five_post_reset_motion.py``),
decode frames from video start to the start control and compute the
screenshot-scale centre difference (``scene_settle.centre_grey``) between frames
8 apart (~0.25 s), sampled every 4 frames. Reports min/last/max per recording.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from trueskate_ai.collection.scene_settle import centre_difference, centre_grey


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("motion_json", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = []
    for r in json.loads(args.motion_json.read_text()):
        cap = cv2.VideoCapture(str(Path(r["segment"]).with_suffix(".mov")))
        greys = []
        for _ in range(max(1, int(r["start_approx_video_s"] * 30))):
            ok, image = cap.read()
            if not ok:
                break
            greys.append(centre_grey(cv2.imencode(".png", image)[1].tobytes()))
        diffs = [centre_difference(greys[i], greys[i + 8]) for i in range(0, len(greys) - 8, 4)]
        rows.append({"segment": r["segment"], "device": r["device"],
                     "video_scale_motion_mean": r["centre_motion_mean"],
                     "screenshot_scale_0p25s_min": round(min(diffs), 3),
                     "screenshot_scale_0p25s_last": round(diffs[-1], 3),
                     "screenshot_scale_0p25s_max": round(max(diffs), 3)})
    args.out.write_text(json.dumps(rows, indent=1) + "\n")
    for row in sorted(rows, key=lambda x: x["video_scale_motion_mean"]):
        print(row["device"], row["video_scale_motion_mean"], row["screenshot_scale_0p25s_min"],
              row["screenshot_scale_0p25s_last"], row["screenshot_scale_0p25s_max"])


if __name__ == "__main__":
    main()
