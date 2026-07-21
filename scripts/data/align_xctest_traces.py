"""Align XCTest segment recordings to gestures -> (frame, gesture) training samples.

Consumes a segment written by ``collect_sls_xctest.py`` (a ``.mov`` + a ``.json``
manifest of host-epoch gesture call times) and slices a frame window around each
gesture out of the ``.mov``, writing per-gesture sample dirs that match the DAL
collector's on-disk format (``frame_NNN.png`` + ``meta.json`` with ``frame_times``).

Alignment: the manifest's ``started_at_epoch_s`` is video t0 (same epoch clock as the
gesture call times), so a gesture whose call STARTED at host ``ts`` maps to video PTS
``gv = (ts - started_at) + Δ``, where Δ is the measured command->pixel offset
(``_DELTA_BY_KIND``). ``frame_time`` 0 is therefore the moment the touch's first
pixels land. We extract ``[gv - pre, gv + window]`` at the native fps (fast INPUT-seek
per gesture, so a 5-min segment isn't re-decoded N times), downscale to
``--resize-width``, evenly downsample to ``--max-frames``, and stamp each kept frame's
``frame_time`` (video PTS - gv).

HISTORY — why the anchor moved (2026-07-21). This previously anchored on
``t_call_end_epoch_s`` with Δ=0 and 0.3s of lead-in. That is when Appium's HTTP
``perform()`` RETURNED, which trails the actual touch by the whole call wall — median
1.7s on the SLS corpus against ~0.23s of real gesture payload. Consequence: on 43/100
sampled corpus samples the trace was ALREADY drawn in ``frame_000``, and on ~half of
those it was at full strength and only decaying, i.e. the stroke finished before the
window opened. Δ has since been measured (see memory ``xctest-command-to-pixel-delta``)
and is stable to well under one frame, so start-anchoring is now both possible and
correct. ``--anchor end`` restores the old behaviour for re-aligning old segments.

CONSUMER NOTE: with ``--anchor start`` the emitted meta carries
``gesture_start_monotonic``, which flips ``temporal_trace_dataset._is_end_relative()``
to the START-relative branch — the correct reading of these ``frame_times``. Because Δ
is already applied here, the trainer's own ``latency_s`` should be **0** for this data;
leaving it at the legacy 0.2 would double-count the compensation.

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


# Measured command->pixel offsets, seconds after `t_call_start_epoch_s` (XR1,
# 2026-07-21, frame-differencing known touch points across 4 recordings; see memory
# `xctest-command-to-pixel-delta`). Stable to <1 frame at 30fps: within-recording std
# 0.017s, between-recording std 0.007s — so a CONSTANT Δ is valid, it does not drift
# per segment.
#
# Δ depends on the Appium dispatch path, not just the rig: `mobile: tap` is ~0.12s
# slower than the ActionChains path. Measured directly for tap (1.112, n=16) and
# long_press (~0.98, n=8). Drag kinds are ActionChains like long_press, so they take
# that value — INFERRED for drags, not directly measured. Re-measure with
# `tmp/delta_stability.py` if the rig, iOS, or appium-xcuitest version changes.
_DELTA_TAP_S = 1.11
# Refined from 0.98 by the end-to-end anchor check: with 0.98 the 9 hold samples all
# landed +0.067..+0.10s late while taps landed exactly on 0, i.e. the ActionChains path
# is ~0.08s slower than the coarse Stage 0 estimate. Verified back to ~0 residual.
_DELTA_ACTIONCHAINS_S = 1.06
_DELTA_BY_KIND = {
    "tap": _DELTA_TAP_S,
    "hold": _DELTA_ACTIONCHAINS_S,
    "flick": _DELTA_ACTIONCHAINS_S,
    "spin_flick": _DELTA_ACTIONCHAINS_S,
    "nslot": _DELTA_ACTIONCHAINS_S,
    "recipe": _DELTA_ACTIONCHAINS_S,
    "spin": _DELTA_ACTIONCHAINS_S,
}


def _park_tag(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _delta_for(ev: dict, override: float | None, manifest_delta: float | None) -> float:
    if override is not None:
        return override
    if manifest_delta:
        return float(manifest_delta)
    return _DELTA_BY_KIND.get(str(ev.get("gesture_distribution", "")).casefold(),
                              _DELTA_ACTIONCHAINS_S)


def _encode_sample_video(sample_dir: Path, n_frames: int, fps: int, crf: int) -> bool:
    """Pack a sample's frame_NNN.png into one h264 clip, deleting the PNGs.

    ~30x smaller than the PNG sequence at crf 20 (benchmarked on 333 real corpus
    frames: 4.5% error in the path-glow statistic, so the orange trace survives),
    and 1 inode instead of N — which matters as much as bytes, the corpus volume
    having hit 2.37M files against a 500k inode limit.

    Returns False and leaves the PNGs untouched if encoding fails: a sample that
    exists as PNGs is strictly better than one lost to a bad encode.
    """
    out = sample_dir / "frames.mp4"
    r = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-framerate", str(fps),
         "-i", str(sample_dir / "frame_%03d.png"),
         "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", str(out)],
        capture_output=True, text=True,
    )
    if r.returncode != 0 or not out.exists():
        print(f"  {sample_dir.name}: video encode failed ({r.stderr[:120]}) — keeping PNGs")
        out.unlink(missing_ok=True)
        return False
    for i in range(n_frames):
        (sample_dir / f"frame_{i:03d}.png").unlink(missing_ok=True)
    return True


def _even_indices(n: int, max_n: int) -> list[int]:
    if n <= max_n:
        return list(range(n))
    return [round(i * (n - 1) / (max_n - 1)) for i in range(max_n)]


def align_segment(manifest_path: Path, *, pre_s: float, window_s: float, fps: int,
                  resize_width: int, max_frames: int, delta_override: float | None,
                  delete_mov: bool, anchor: str = "start",
                  video: bool = False, video_crf: int = 20) -> int:
    manifest = json.loads(manifest_path.read_text())
    seg_dir = manifest_path.parent
    mov = seg_dir / manifest["mov"]
    if not mov.exists():
        print(f"[align] {manifest_path.name}: .mov missing ({mov.name}) — skip")
        return 0
    # CLAIM the segment before doing any work. The collector spawns one aligner per
    # segment asynchronously, so a concurrent `--session` sweep would otherwise
    # re-align segments already in flight and, with --delete-mov, pull the .mov out
    # from under the running process (observed: 3/9 samples survived). The .aligned
    # marker is only written on completion, so it cannot prevent this on its own.
    claim = seg_dir / (manifest_path.stem + ".aligning")
    try:
        claim.touch(exist_ok=False)
    except FileExistsError:
        print(f"[align] {manifest_path.name}: already being aligned "
              f"({claim.name} exists) — skip. Delete it if a previous run died.")
        return 0
    started_at = float(manifest["started_at_epoch_s"])
    dw, dh = manifest["device_logical_w"], manifest["device_logical_h"]
    manifest_delta = manifest.get("capture_offset_s")
    gestures = manifest.get("gestures", [])

    saved = 0
    deltas_used: set[float] = set()
    try:
        for ev in gestures:
            delta = _delta_for(ev, delta_override, manifest_delta)
            deltas_used.add(delta)
            if anchor == "start":
                # Anchor on when the touch's FIRST PIXELS land: t_call_start + Δ. The old
                # `t_call_end` anchor was when Appium's HTTP perform() RETURNED, which
                # trails the actual touch by the whole call wall (median 1.7s on the SLS
                # corpus vs ~0.23s of real payload) — so with only 0.3s of lead-in the
                # stroke was frequently over before the window even opened.
                gv = (float(ev["t_call_start_epoch_s"]) - started_at) + delta
            else:
                gv = (float(ev["t_call_end_epoch_s"]) - started_at) + delta
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
                # frame_time 0 == the touch's first pixels (start anchor + Δ)
                frame_times.append(round(times[src_i] - gv, 4))
            shutil.rmtree(raw, ignore_errors=True)
            meta = {
                **{k: v for k, v in ev.items()},                     # gesture params + call times + park
                "device_logical_w": dw, "device_logical_h": dh,
                "gesture_video_time_s": round(gv, 4),
                "capture_offset_s": delta,
                "anchor": anchor,
                "frame_times": frame_times,
                "n_frames": len(frame_times),
                "segment_index": manifest.get("segment_index"),
                "session": seg_dir.name,
            }
            if anchor == "start":
                # temporal_trace_dataset._is_end_relative() keys off this: its presence
                # switches the label scheduler to the START-relative branch, which is what
                # these frame_times now are. Without it the scheduler would add the payload
                # duration and place every touch a full stroke too late.
                meta["gesture_start_monotonic"] = float(ev["t_call_start_epoch_s"])
            if video:
                meta["frames_format"] = ("mp4" if _encode_sample_video(
                    sample_dir, len(frame_times), fps, video_crf) else "png")
            (sample_dir / "meta.json").write_text(json.dumps(meta, indent=2))
            saved += 1

    finally:
        claim.unlink(missing_ok=True)

    # mark processed + optionally drop the (large) .mov to free host space
    (seg_dir / (manifest_path.stem + ".aligned")).write_text(
        json.dumps({"samples": saved, "anchor": anchor,
                    "delta_s": sorted(deltas_used)}))
    print(f"[align] {manifest_path.name}: {saved}/{len(gestures)} samples "
          f"(anchor={anchor}, Δ={sorted(deltas_used)}s) -> {seg_dir}")
    if delete_mov and saved > 0:
        mov.unlink(missing_ok=True)
        print(f"[align] deleted {mov.name} (freed host space)")
    return saved


def main() -> None:
    ap = argparse.ArgumentParser(description="Align XCTest segment recordings to gestures.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--segment", type=Path, help="A single segment manifest .json.")
    g.add_argument("--session", type=Path, help="A session dir; aligns every not-yet-aligned segment.")
    ap.add_argument("--anchor", choices=("start", "end"), default="start",
                    help="Which command time frame_time 0 is pinned to, plus Δ. 'start' "
                         "(default) = t_call_start + Δ, i.e. the touch's first pixels. "
                         "'end' reproduces the old t_call_end behaviour — only for "
                         "re-aligning pre-2026-07-21 segments.")
    ap.add_argument("--pre-s", type=float, default=0.5,
                    help="Seconds of pre-touch lead-in. 0.5 is ample now that Δ is "
                         "measured to <1 frame; a bigger window would only dilute "
                         "temporal resolution once --max-frames downsamples it.")
    ap.add_argument("--window-s", type=float, default=1.8,
                    help="Seconds after touch onset. Default covers the longest 1.5s "
                         "hold plus its ~0.2s fade tail.")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--resize-width", type=int, default=512)
    ap.add_argument("--max-frames", type=int, default=32)
    ap.add_argument("--delta-s", type=float, default=None,
                    help="Override the command->pixel Δ for every gesture. Default is "
                         "per-kind measured values (see _DELTA_BY_KIND).")
    ap.add_argument("--delete-mov", action="store_true", help="Delete the .mov after a successful align.")
    ap.add_argument("--video", action="store_true",
                    help="Store each sample as one frames.mp4 instead of N PNGs: ~30x "
                         "smaller and 1 inode instead of N. The dataset loader reads "
                         "either format transparently.")
    ap.add_argument("--video-crf", type=int, default=20,
                    help="x264 CRF for --video. 20 keeps the orange trace intact "
                         "(measured 4.5%% error in the path-glow statistic).")
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
                delta_override=args.delta_s, delete_mov=args.delete_mov,
                anchor=args.anchor, video=args.video, video_crf=args.video_crf)
        except Exception as exc:  # noqa: BLE001
            print(f"[align] {m.name} FAILED: {exc}")
    print(f"[align] done: {total} samples across {len(manifests)} segment(s)")


if __name__ == "__main__":
    main()
