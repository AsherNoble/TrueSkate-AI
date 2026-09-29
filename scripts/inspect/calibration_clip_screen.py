#!/usr/bin/env python3
"""Draft per-clip timing screen from two-anchor calibration rates (read-only).

M1-ONSET-VALIDATION-20260924 found that, assuming a true clock rate of ~1.0,
a falsely early start anchor delays a clip's real onset by
``(rate - 1) * (end_wda - gesture_wda)`` and a falsely early end anchor by
``(1 - rate) * (gesture_wda - start_wda)``; this matched 22/24 blind labels
exactly and 24/24 within one frame. Segments with ``abs(rate - 1)`` inside the
screen threshold are treated as ordinary (their spread matches ±1-frame anchor
quantisation). For flagged segments this reports each clip's predicted onset
error, so a clip-level screen can be compared with whole-segment exclusion.
Standard library only. Proposes nothing; writes a report.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

FRAME_S = 1 / 30


def clip_rows(roots: list[Path], threshold: float) -> list[dict]:
    rows = []
    for root in roots:
        for meta_path in root.rglob("meta.json"):
            meta = json.loads(meta_path.read_text())
            cal = meta.get("tap_calibration") or {}
            if cal.get("method") != "wda-submitted-two-centre-controls-v2":
                continue
            anchors = {d["role"]: d["submitted_to_ios_monotonic_s"] for d in cal["detections"]}
            rate = cal["rate"]
            g = meta["wda_submitted_monotonic_s"]
            if rate - 1 > threshold:
                predicted = (rate - 1) * (anchors["end"] - g)
                implicated = "start"
            elif 1 - rate > threshold:
                predicted = (1 - rate) * (g - anchors["start"])
                implicated = "end"
            else:
                predicted, implicated = 0.0, None
            rows.append({
                "path": str(meta_path.parent),
                "device": meta.get("device"), "park": meta.get("park"),
                "segment": f"{meta.get('session')}/{meta.get('segment_index')}",
                "rate": rate, "implicated_anchor": implicated,
                "predicted_onset_error_s": round(predicted, 5),
            })
    return rows


def summarise(rows: list[dict]) -> dict:
    cells: dict[str, dict] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        for name in (r["park"], "ALL"):
            c = cells[name]
            c["clips"] += 1
            if r["implicated_anchor"] is not None:
                c["in_flagged_segments"] += 1
                frames = r["predicted_onset_error_s"] / FRAME_S
                c["flagged_predicted_le_0_5_frame"] += frames <= 0.5
                c["flagged_predicted_le_1_frame"] += frames <= 1.0
            c["clean_if_segment_excluded"] = c["clips"] - c["in_flagged_segments"]
            c["clean_if_clip_screen_1_frame"] = c["clean_if_segment_excluded"] + c["flagged_predicted_le_1_frame"]
    return {k: dict(v) for k, v in sorted(cells.items())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--threshold", type=float, default=0.002)
    ap.add_argument("--rows-out", type=Path)
    args = ap.parse_args()
    rows = clip_rows(args.roots, args.threshold)
    if args.rows_out:
        args.rows_out.write_text(json.dumps(rows) + "\n")
    print(json.dumps({"threshold": args.threshold, "roots": [str(r) for r in args.roots],
                      "cells": summarise(rows)}, indent=1))


if __name__ == "__main__":
    main()
