"""Align XCTest segment recordings to gestures -> (frame, gesture) training samples.

Consumes a segment written by ``collect_sls_xctest.py`` (a ``.mov`` + a ``.json``
manifest of host-epoch gesture call times) and slices a frame window around each
gesture out of the ``.mov``, writing per-gesture sample dirs that match the DAL
collector's on-disk format (``frame_NNN.png`` + ``meta.json`` with ``frame_times``).

Alignment: the manifest's ``started_at_epoch_s`` is video t0 (same epoch clock as the
gesture call times), so a gesture whose call returned at host ``te`` maps to video PTS
``gv = (te - started_at) + Δ`` where Δ (``capture_offset_s``) is the command->pixel
latency (measure via the clapperboard on the XCTest path; defaults to 0). We extract
``[gv - pre, gv + window]`` at the native fps (fast INPUT-seek per gesture, so a 5-min
segment isn't re-decoded N times), downscale to ``--resize-width``, evenly downsample
to ``--max-frames``, and stamp each kept frame's ``frame_time`` (video PTS - gv).

Run async by the collector (``--segment <manifest> --delete-mov``), or by hand over a
session dir (``--session <dir>`` processes every not-yet-aligned segment).

    python scripts/data/align_xctest_traces.py --segment data/sls_xctest/<sess>/segment_00000.json
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


def _park_tag(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _even_indices(n: int, max_n: int) -> list[int]:
    if n <= max_n:
        return list(range(n))
    return [round(i * (n - 1) / (max_n - 1)) for i in range(max_n)]


def align_segment(manifest_path: Path, *, pre_s: float, window_s: float, fps: int,
                  resize_width: int, max_frames: int, delta_override: float | None,
                  delete_mov: bool) -> int:
    manifest = json.loads(manifest_path.read_text())
    seg_dir = manifest_path.parent
    mov = seg_dir / manifest["mov"]
    if not mov.exists():
        print(f"[align] {manifest_path.name}: .mov missing ({mov.name}) — skip")
        return 0
    started_at = float(manifest["started_at_epoch_s"])
    dw, dh = manifest["device_logical_w"], manifest["device_logical_h"]
    delta = delta_override if delta_override is not None else (manifest.get("capture_offset_s") or 0.0)
    gestures = manifest.get("gestures", [])

    saved = 0
    for ev in gestures:
        te = float(ev["t_call_end_epoch_s"])
        gv = (te - started_at) + delta            # gesture-end video PTS
        start = max(0.0, gv - pre_s)
        dur = pre_s + window_s
        sample_dir = seg_dir / _park_tag(ev.get("park", "park")) / f"sample_{ev['gesture_index']:06d}"
        raw = sample_dir / "_raw"
        raw.mkdir(parents=True, exist_ok=True)
        # INPUT-seek (-ss before -i): fast keyframe seek + accurate decode-to-pos in modern
        # ffmpeg; output PTS reset to 0 at `start`, so frame i is at video PTS start + i/fps.
        r = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-i", str(mov),
             "-t", f"{dur:.3f}", "-vf", f"fps={fps},scale={resize_width}:-2",
             "-vsync", "0", str(raw / "f_%04d.png")],
            capture_output=True, text=True,
        )
        frames = sorted(raw.glob("f_*.png"))
        if r.returncode != 0 or not frames:
            print(f"  g{ev['gesture_index']}: no frames (ffmpeg rc={r.returncode}) {r.stderr[:120]}")
            shutil.rmtree(sample_dir, ignore_errors=True)
            continue
        times = [start + i / fps for i in range(len(frames))]   # absolute video PTS
        keep = _even_indices(len(frames), max_frames)
        frame_times = []
        for out_i, src_i in enumerate(keep):
            frames[src_i].rename(sample_dir / f"frame_{out_i:03d}.png")
            frame_times.append(round(times[src_i] - gv, 4))      # relative to gesture-end PTS
        shutil.rmtree(raw, ignore_errors=True)
        meta = {
            **{k: v for k, v in ev.items()},                     # gesture params + call times + park
            "device_logical_w": dw, "device_logical_h": dh,
            "gesture_video_time_s": round(gv, 4),
            "capture_offset_s": delta,
            "frame_times": frame_times,
            "n_frames": len(frame_times),
            "segment_index": manifest.get("segment_index"),
            "session": seg_dir.name,
        }
        (sample_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        saved += 1

    # mark processed + optionally drop the (large) .mov to free host space
    (seg_dir / (manifest_path.stem + ".aligned")).write_text(
        json.dumps({"samples": saved, "delta_s": delta}))
    print(f"[align] {manifest_path.name}: {saved}/{len(gestures)} samples "
          f"(Δ={delta}s) -> {seg_dir}")
    if delete_mov and saved > 0:
        mov.unlink(missing_ok=True)
        print(f"[align] deleted {mov.name} (freed host space)")
    return saved


def main() -> None:
    ap = argparse.ArgumentParser(description="Align XCTest segment recordings to gestures.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--segment", type=Path, help="A single segment manifest .json.")
    g.add_argument("--session", type=Path, help="A session dir; aligns every not-yet-aligned segment.")
    ap.add_argument("--pre-s", type=float, default=0.3, help="Seconds of lead-in before the gesture PTS.")
    ap.add_argument("--window-s", type=float, default=0.9, help="Seconds of response after the gesture PTS.")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--resize-width", type=int, default=512)
    ap.add_argument("--max-frames", type=int, default=24)
    ap.add_argument("--delta-s", type=float, default=None,
                    help="Override the command->pixel Δ (else use the manifest's capture_offset_s, else 0).")
    ap.add_argument("--delete-mov", action="store_true", help="Delete the .mov after a successful align.")
    args = ap.parse_args()

    if args.segment:
        manifests = [args.segment]
    else:
        done = {p.stem for p in args.session.glob("*.aligned")}
        manifests = sorted(p for p in args.session.glob("segment_*.json") if p.stem not in done)
        if not manifests:
            print(f"[align] no un-aligned segments under {args.session}")
            return

    total = 0
    for m in manifests:
        try:
            total += align_segment(
                m, pre_s=args.pre_s, window_s=args.window_s, fps=args.fps,
                resize_width=args.resize_width, max_frames=args.max_frames,
                delta_override=args.delta_s, delete_mov=args.delete_mov)
        except Exception as exc:  # noqa: BLE001
            print(f"[align] {m.name} FAILED: {exc}")
    print(f"[align] done: {total} samples across {len(manifests)} segment(s)")


if __name__ == "__main__":
    main()
