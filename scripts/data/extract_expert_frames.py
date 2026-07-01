"""Extract expert videos -> per-clip frame dirs for the BC pipeline (front end).

The expert corpus is raw .mp4/.mov (e.g. Training_Data/Sorted/**/<trick>.mp4);
`build_bc_clips.py` consumes clip dirs of `frame_%06d.png`. This is the missing
decode step between them: each video -> `<out>/<relpath-no-ext>/frame_%06d.png`,
resampled to a target fps. Pass the SAME --fps to build_bc_clips.

Streaming (constant memory) and mirrors the input tree, so a whole corpus maps to
a parallel tree of clip dirs. Pure cv2 — no device deps.

Usage:
    python scripts/data/extract_expert_frames.py \
        --videos-root ~/Projects/Robotics\\ \\&\\ hardware/Training_Data/Sorted \
        --out data/expert/Sorted --fps 30
    python scripts/data/extract_expert_frames.py --smoke
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

_VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi"}


def extract_video(video: Path, out_dir: Path, *, target_fps: float,
                  resize: tuple[int, int] | None = None, grayscale: bool = False) -> int:
    """Decode `video` -> frame_%06d.png in out_dir at ~target_fps. Returns #frames."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    native = cap.get(cv2.CAP_PROP_FPS) or target_fps
    step = max(1.0, native / target_fps)          # >1 downsamples; clamped so we never upsample
    out_dir.mkdir(parents=True, exist_ok=True)

    src_idx, next_keep, kept = 0, 0.0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if src_idx >= next_keep:
            if resize is not None:
                frame = cv2.resize(frame, resize)
            if grayscale:
                frame = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            cv2.imwrite(str(out_dir / f"frame_{kept:06d}.png"), frame)
            kept += 1
            next_keep += step
        src_idx += 1
    cap.release()
    return kept


def _run(args) -> None:
    root = args.videos_root
    vids = sorted(p for p in root.rglob("*") if p.suffix.lower() in _VIDEO_EXTS) \
        if root.is_dir() else [root]
    if not vids:
        raise SystemExit(f"no videos under {root}")
    resize = tuple(args.resize) if args.resize else None
    print(f"extracting {len(vids)} video(s) @ {args.fps}fps"
          + (f", resize={resize}" if resize else "") + (", grayscale" if args.grayscale else ""))
    total = 0
    for v in vids:
        rel = v.relative_to(root).with_suffix("") if root.is_dir() else Path(v.stem)
        out = args.out / rel
        n = extract_video(v, out, target_fps=args.fps, resize=resize, grayscale=args.grayscale)
        total += n
        print(f"  {rel}: {n} frames -> {out}")
    print(f"done: {len(vids)} clips, {total} frames. Next: build_bc_clips.py --clips-root {args.out} --fps {args.fps}")


def _smoke() -> None:
    """Synthesize a short video, extract at half fps, confirm the frame_%06d.png layout."""
    import tempfile

    import cv2
    import numpy as np

    with tempfile.TemporaryDirectory() as td:
        vid = Path(td) / "clip0.mp4"
        w, h, src_fps, n = 96, 208, 30.0, 30
        vw = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))
        if not vw.isOpened():                       # codec fallback for headless cv2 builds
            vid = Path(td) / "clip0.avi"
            vw = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"MJPG"), src_fps, (w, h))
        for i in range(n):
            vw.write(np.full((h, w, 3), i * 8 % 255, dtype=np.uint8))
        vw.release()

        out = Path(td) / "out"
        kept = extract_video(vid, out, target_fps=15.0)   # half -> ~15 frames
        pngs = sorted(out.glob("frame_*.png"))
        assert kept == len(pngs) >= 1, (kept, len(pngs))
        assert pngs[0].name == "frame_000000.png", pngs[0].name
        assert 12 <= kept <= 18, f"expected ~15 frames at half-fps, got {kept}"
        # frame_%06d.png is exactly what build_bc_clips._sorted_frames globs.
    print(f"SMOKE OK — {kept} frame_%06d.png extracted at half fps; layout matches build_bc_clips")


def main() -> None:
    ap = argparse.ArgumentParser(description="Decode expert videos into per-clip frame_%06d.png dirs.")
    ap.add_argument("--smoke", action="store_true", help="Synthetic video -> extract round-trip; no corpus.")
    ap.add_argument("--videos-root", type=Path, help="Video file or dir tree of .mp4/.mov.")
    ap.add_argument("--out", type=Path, help="Output root; mirrors --videos-root structure.")
    ap.add_argument("--fps", type=float, default=30.0, help="Target sample rate (never upsamples past native).")
    ap.add_argument("--resize", type=int, nargs=2, metavar=("W", "H"), default=None,
                    help="Optional output frame size, e.g. --resize 96 208.")
    ap.add_argument("--grayscale", action="store_true", help="Write luma-only frames (stored as 3ch).")
    args = ap.parse_args()

    if args.smoke:
        _smoke()
        return
    if not (args.videos_root and args.out):
        ap.error("real mode needs --videos-root and --out (or use --smoke)")
    _run(args)


if __name__ == "__main__":
    main()
