#!/usr/bin/env python3
"""Count two-anchor calibration-fit anomalies per device, park and implicated anchor.

Read-only. Groups admitted clip ``meta.json`` files by recording segment and
reports how many segments/clips have ``abs(rate - 1) > threshold``. A high rate
implicates an early *start*-control detection; a low rate implicates an early
*end*-control detection (M1-ONSET-VALIDATION-20260924). Segments that failed
calibration never emitted clips, so this undercounts detector failures.
Standard library only, so it runs on the rig's system Python.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def collect(roots: list[Path]) -> dict:
    segments: dict[tuple, dict] = {}
    for root in roots:
        for meta_path in root.rglob("meta.json"):
            meta = json.loads(meta_path.read_text())
            cal = meta.get("tap_calibration") or {}
            if cal.get("method") != "wda-submitted-two-centre-controls-v2":
                continue
            key = (str(meta.get("session")), meta.get("segment_index"), str(meta.get("device")))
            seg = segments.setdefault(key, {
                "device": meta.get("device"),
                "park": meta.get("park"),
                "rate": cal["rate"],
                "scores": {d["role"]: d["detector_score"] for d in cal.get("detections", [])},
                "clips": 0,
            })
            seg["clips"] += 1
    return segments


def summarise(segments: dict, threshold: float) -> dict:
    cells: dict[str, dict] = defaultdict(lambda: {
        "segments": 0, "clips": 0, "high_rate_segments": 0, "high_rate_clips": 0,
        "low_rate_segments": 0, "low_rate_clips": 0,
    })
    for seg in segments.values():
        for name in (f"{seg['device']}|{seg['park']}", "ALL"):
            cell = cells[name]
            cell["segments"] += 1
            cell["clips"] += seg["clips"]
            if seg["rate"] - 1.0 > threshold:
                cell["high_rate_segments"] += 1
                cell["high_rate_clips"] += seg["clips"]
            elif 1.0 - seg["rate"] > threshold:
                cell["low_rate_segments"] += 1
                cell["low_rate_clips"] += seg["clips"]
    return dict(sorted(cells.items()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--threshold", type=float, default=0.002)
    args = ap.parse_args()
    segments = collect(args.roots)
    print(json.dumps({
        "threshold": args.threshold,
        "roots": [str(r) for r in args.roots],
        "cells": summarise(segments, args.threshold),
        "segments": [
            {"session": k[0], "segment_index": k[1], **v} for k, v in sorted(segments.items(), key=str)
        ],
    }, indent=1))


if __name__ == "__main__":
    main()
