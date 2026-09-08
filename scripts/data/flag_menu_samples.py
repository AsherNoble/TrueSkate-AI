"""Flag (or delete) already-collected samples whose frames are True Skate's replay/menu
rather than live skatepark gameplay.

The random-gesture XCTest collector could tap the game into REPLAY and stay there for
long stretches (the park is visible behind the replay, so it *looks* like gameplay) —
those ``(menu-frame, random-gesture)`` pairs are noise for a frame->gesture model. The
collector now guards against this in-loop (``--no-gameplay-guard`` to disable); this
script cleans what was collected *before* the guard.

Uses ``vision.gameplay_filter.is_menu_frame`` across each sample. Replay state is
normally stable, but the gray app-hub navigation can fade in/out within a sample,
so checking only the middle frame leaves partially contaminated samples behind.
NON-DESTRUCTIVE by default: writes a ``.menu`` marker in each flagged sample dir
so a training loader can exclude any sample dir containing ``.menu``. ``--delete``
removes flagged dirs instead.

    python scripts/data/flag_menu_samples.py [--device iPhone_XR2] [--dry-run] [--delete]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from trueskate_ai.vision.gameplay_filter import is_menu_frame  # noqa: E402


def _sample_has_menu_frame(sample_dir: Path) -> bool:
    """Inspect all retained PNG frames or all frames in the compact MP4 variant."""
    pngs = sorted(sample_dir.glob("frame_*.png"))
    if pngs:
        # Probe likely hits first; retain the exhaustive sweep for a menu that
        # appears only in the lead-in/tail of an otherwise gameplay sample.
        mid = len(pngs) // 2
        ordered = list(dict.fromkeys((pngs[mid], pngs[0], pngs[-1])))
        priority = set(ordered)
        ordered.extend(frame for frame in pngs if frame not in priority)
        return any(is_menu_frame(frame) for frame in ordered)

    video = sample_dir / "frames.mp4"
    if not video.is_file():
        raise ValueError(f"{sample_dir}: no retained frames")
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError(f"{sample_dir}: unreadable frames.mp4")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            # OpenCV decodes BGR; gameplay_filter's array path expects RGB.
            if is_menu_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)):
                return True
    finally:
        capture.release()
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Flag/delete replay-menu samples in the collected corpus.")
    ap.add_argument("--data", type=Path, default=_REPO / "data" / "sls_xctest")
    ap.add_argument("--device", default="", help="Substring filter on session dir name (e.g. iPhone_XR2).")
    ap.add_argument("--dry-run", action="store_true", help="Report only; write no markers, delete nothing.")
    ap.add_argument("--delete", action="store_true", help="Delete flagged sample dirs (else write a .menu marker).")
    args = ap.parse_args()

    sessions = sorted(p for p in args.data.glob("*") if p.is_dir() and args.device in p.name)
    grand_menu = grand_total = 0
    for sess in sessions:
        sds = sorted(p for p in sess.glob("**/sample_*") if p.is_dir())
        if not sds:
            continue
        menu = 0
        for sd in sds:
            try:
                flagged = _sample_has_menu_frame(sd)
            except Exception:  # noqa: BLE001 — a bad frame shouldn't abort the sweep
                continue
            if flagged:
                menu += 1
                if not args.dry_run:
                    if args.delete:
                        shutil.rmtree(sd, ignore_errors=True)
                    else:
                        (sd / ".menu").write_text("replay/menu frame, not gameplay\n")
        grand_menu += menu
        grand_total += len(sds)
        print(f"{sess.name}: {menu}/{len(sds)} menu ({100*menu/len(sds):.0f}%)", flush=True)

    action = "deleted" if (args.delete and not args.dry_run) else (
        "flagged (.menu marker)" if not args.dry_run else "would flag")
    print(f"\nTOTAL: {grand_menu}/{grand_total} samples are replay/menu "
          f"({100*grand_menu/max(1,grand_total):.1f}%)  gameplay={grand_total-grand_menu}  [{action}]")
    if not args.dry_run and not args.delete:
        print("Exclude any sample dir containing a '.menu' file in your training loader.")


if __name__ == "__main__":
    main()
