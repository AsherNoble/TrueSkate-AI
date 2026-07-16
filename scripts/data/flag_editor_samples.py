"""Flag already-collected samples whose frames are True Skate's park editor.

This mirrors ``flag_menu_samples.py`` but targets park-editor contamination in
the SLS frame->gesture corpus. It samples each ``sample_*`` dir's middle frame,
runs ``vision.gameplay_filter.is_editor_frame``, and writes a ``.editor`` marker
for flagged dirs (idempotent: existing markers are counted and skipped).

    python scripts/data/flag_editor_samples.py [--root data/sls_xctest] [--device iPhone_XR2] [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import traceback

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from trueskate_ai.vision.gameplay_filter import is_editor_frame  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Flag park-editor-contaminated samples in the collected corpus.")
    ap.add_argument("--root", type=Path, default=_REPO / "data" / "sls_xctest")
    ap.add_argument("--device", default="", help="Substring filter on session dir name (e.g. iPhone_XR2).")
    ap.add_argument("--dry-run", action="store_true", help="Report only; write no markers.")
    args = ap.parse_args()

    sessions = sorted(p for p in args.root.glob("*") if p.is_dir() and args.device in p.name)
    grand_scanned = grand_flagged = grand_already = grand_errors = 0
    for sess in sessions:
        sample_dirs = sorted(p for p in sess.glob("**/sample_*") if p.is_dir())
        if not sample_dirs:
            continue

        scanned = flagged = already_flagged = errors = 0
        for sample_dir in sample_dirs:
            scanned += 1
            marker = sample_dir / ".editor"
            if marker.exists():
                already_flagged += 1
                continue

            frames = sorted(sample_dir.glob("frame_*.png"))
            if not frames:
                errors += 1
                continue

            try:
                is_editor = is_editor_frame(frames[len(frames) // 2])
            except Exception as e:  # log traceback to stderr and continue
                errors += 1
                print(f"Error processing frame in {sample_dir}: {e}", file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)
                continue

            if is_editor:
                flagged += 1
                if not args.dry_run:
                    marker.write_text("park-editor frame, not gameplay\n")

        grand_scanned += scanned
        grand_flagged += flagged
        grand_already += already_flagged
        grand_errors += errors
        print(
            f"{sess.name}: scanned={scanned}  flagged={flagged}  already-flagged={already_flagged}  errors={errors}",
            flush=True,
        )

    mode = "would flag" if args.dry_run else "flagged (.editor marker)"
    print(
        f"\nTOTAL: scanned={grand_scanned}  flagged={grand_flagged}  already-flagged={grand_already}  errors={grand_errors}"
        f"  [{mode}]"
    )
    if not args.dry_run:
        print("Exclude any sample dir containing a '.editor' file in your training loader.")

    # Exit with non-zero status if any errors occurred during the sweep
    if grand_errors > 0:
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
