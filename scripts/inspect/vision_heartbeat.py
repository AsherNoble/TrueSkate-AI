"""Vision heartbeat: periodically report where the board is + what's happening
in the latest CMA-ES run — without touching the device (reads the run's saved
color trace frames + JSONL, so it runs safely alongside the CMA-ES that owns
the Appium session).

Usage:
    python scripts/inspect/vision_heartbeat.py [--interval 60] [--device iPhone_XR]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import cv2

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from trueskate_ai.vision.board_localizer import locate_board  # noqa: E402


def _latest_run(device: str | None) -> Path | None:
    pat = f"logs/overnight/{device or '*'}/*/runs/cmaes_run_*/"
    runs = glob.glob(str(_REPO / pat))
    if not runs:
        return None
    return Path(max(runs, key=lambda p: Path(p).stat().st_mtime))  # most recently active


def _status(run_dir: Path) -> str:
    # board pose from the most recent trace frame
    frames = sorted(glob.glob(str(run_dir / "trace_data" / "eval_*" / "frame_*.png")))
    pose = locate_board(cv2.imread(frames[-1])) if frames else None
    board = f"({pose.cx:.2f},{pose.cy:.2f}) {pose.angle_deg:+.0f}deg" if pose else "NA"
    # recent run stats from the JSONL
    land50, recent, total = "?", [], 0
    jls = glob.glob(str(run_dir / "*.jsonl"))
    if jls:
        rows = [json.loads(l) for l in open(jls[0]) if l.strip()]
        evs = [r for r in rows if "eval_num" in r]
        total = len(evs)
        last = evs[-50:]
        landed = sum(1 for r in last if r.get("trick_status") == "landed")
        land50 = f"{100 * landed / max(1, len(last)):.0f}%"
        recent = [r.get("trick_name") for r in evs[-3:] if r.get("trick_name")]
    target = run_dir.parent.parent.name
    return (f"trick={target} evals={total} land(last50)={land50} "
            f"board={board} trace_frames={len(frames)} recent={recent}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Vision heartbeat for CMA-ES runs.")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--device", default=None)
    ap.add_argument("--once", action="store_true", help="print one status and exit")
    args = ap.parse_args()
    while True:
        rd = _latest_run(args.device)
        ts = time.strftime("%H:%M:%S")
        print(f"[heartbeat {ts}] " + (_status(rd) if rd else "no CMA-ES run found"), flush=True)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
