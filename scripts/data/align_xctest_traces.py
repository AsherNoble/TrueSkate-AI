"""Align XCTest segment recordings to gestures -> (frame, gesture) training samples.

Consumes a segment written by ``collect_sls_xctest.py`` (a ``.mov`` + a ``.json``
manifest of host-epoch gesture call times) and slices a frame window around each
gesture out of the ``.mov``, writing per-gesture sample dirs that match the DAL
collector's on-disk format (``frame_NNN.png`` + ``meta.json`` with ``frame_times``).

Alignment: the manifest's ``started_at_epoch_s`` is video t0 (same epoch clock as the
gesture call times), so a gesture whose call STARTED at host ``ts`` maps to video PTS
``gv = (ts - started_at) + Δ``. The per-kind offsets in ``_DELTA_BY_KIND`` are a
fallback; ``--tap-calibrate`` instead derives the segment shift from the known-position
tap arm of the stationary-touch MVP. ``frame_time`` 0 is therefore the moment the
touch's first pixels land. We extract ``[gv - pre, gv + window]`` at the native fps
(fast INPUT-seek per gesture, so a 5-min segment isn't re-decoded N times), downscale to
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
import math
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from trueskate_ai.vision.tap_timing_calibration import (  # noqa: E402
    detect_tap_onset,
    fit_tap_offsets,
)


# Fallback command->pixel offsets, seconds after `t_call_start_epoch_s` (XR1,
# 2026-07-21). They remain useful for legacy/non-MVP segments, but are not assumed
# segment-stable: the stationary-touch MVP found large unexplained drift.  When
# --tap-calibrate is enabled, its fitted tap offset shifts these values together and
# preserves the measured tap-vs-ActionChains relative difference.
_DELTA_TAP_S = 1.11
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


class TapCalibrationRejected(RuntimeError):
    """A requested per-segment calibration did not meet its evidence gate."""


def _park_tag(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _delta_for(ev: dict, override: float | None, manifest_delta: float | None) -> float:
    if override is not None:
        return override
    if manifest_delta is not None:
        return float(manifest_delta)
    return _DELTA_BY_KIND.get(str(ev.get("gesture_distribution", "")).casefold(),
                              _DELTA_ACTIONCHAINS_S)


def _tap_point(ev: dict) -> tuple[float, float] | None:
    """Return a valid normalised stationary-tap point, otherwise abstain."""
    raw = ev.get("point")
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    try:
        point = float(raw[0]), float(raw[1])
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in point):
        return None
    return point


def _decode_calibration_window(
    mov: Path,
    *,
    command_video_s: float,
    fps: int,
    reference_window_s: float,
    search_after_s: float,
    resize_width: int,
) -> tuple[list, list[float]]:
    """Decode a small tap-search window without retaining a whole segment in RAM."""
    start = max(0.0, command_video_s - reference_window_s)
    duration = (command_video_s - start) + search_after_s
    with tempfile.TemporaryDirectory(prefix="trueskate-tap-cal-") as tmp:
        raw = Path(tmp)
        result = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-i", str(mov),
             "-t", f"{duration:.3f}", "-vf", f"fps={fps},scale={resize_width}:-2",
             "-vsync", "0", str(raw / "f_%04d.png")],
            capture_output=True, text=True,
        )
        paths = sorted(raw.glob("f_*.png"))
        if result.returncode != 0 or not paths:
            return [], []
        frames = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in paths]
        frames = [frame for frame in frames if frame is not None]
    return frames, [start + index / fps for index in range(len(frames))]


def _tap_calibration(
    *,
    manifest: dict,
    mov: Path,
    started_at: float,
    fps: int,
    delta_override: float | None,
    manifest_delta: float | None,
    min_taps: int,
    max_mad_s: float,
    search_after_s: float,
    resize_width: int,
) -> tuple[dict, float | None]:
    """Fit a segment timing shift from its manifest-known tap marks.

    The shift applies to all gesture kinds.  This retains the small measured
    dispatch-path difference between a ``mobile: tap`` and ActionChains while
    correcting the segment-level recorder/transport drift that the handover found.
    """
    tap_events = [
        ev for ev in manifest.get("gestures", [])
        if str(ev.get("gesture_distribution", "")).casefold() == "tap"
    ]
    # A short held control uses the ActionChains dispatch route but remains a
    # manifest `tap` so data loaders exclude it.  Its reference latency must
    # match that route; otherwise its fitted segment shift would bias every
    # following drag by the mobile-tap vs ActionChains difference.
    if tap_events and tap_events[0].get("calibration_execution") == "short_hold":
        reference_tap_delta = _DELTA_ACTIONCHAINS_S
    else:
        reference_tap_delta = (
            _delta_for(tap_events[0], delta_override, manifest_delta)
            if tap_events else _DELTA_TAP_S
        )
    offsets: list[float] = []
    detections: list[dict] = []
    skipped = 0
    for ev in tap_events:
        point = _tap_point(ev)
        try:
            command_s = float(ev["t_call_start_epoch_s"]) - started_at
        except (KeyError, TypeError, ValueError):
            skipped += 1
            continue
        if point is None or not math.isfinite(command_s) or command_s < 0.0:
            skipped += 1
            continue
        frames, times = _decode_calibration_window(
            mov,
            command_video_s=command_s,
            fps=fps,
            reference_window_s=0.5,
            search_after_s=search_after_s,
            resize_width=resize_width,
        )
        onset = detect_tap_onset(
            frames, times, point_xy=point, command_s=command_s,
        )
        if onset is None:
            skipped += 1
            continue
        offset = onset.onset_s - command_s
        offsets.append(offset)
        detections.append({
            "gesture_index": ev.get("gesture_index"),
            "onset_video_s": round(onset.onset_s, 4),
            "offset_s": round(offset, 4),
            "score": onset.score,
            "threshold": onset.threshold,
        })

    fit = fit_tap_offsets(offsets, min_taps=min_taps, max_mad_s=max_mad_s)
    info = {
        "method": "known-point-local-frame-difference",
        "tap_events": len(tap_events),
        "tap_detections": len(offsets),
        "tap_skipped": skipped,
        "reference_tap_delta_s": round(reference_tap_delta, 4),
        "candidate_offsets_s": [round(value, 4) for value in fit.candidate_offsets_s],
        "inlier_offsets_s": [round(value, 4) for value in fit.inlier_offsets_s],
        "tap_offset_s": None if fit.offset_s is None else round(fit.offset_s, 4),
        "mad_s": None if fit.mad_s is None else round(fit.mad_s, 4),
        "accepted": fit.accepted,
        "reason": fit.reason,
        "detections": detections,
    }
    if not fit.accepted or fit.offset_s is None:
        return info, None
    shift = fit.offset_s - reference_tap_delta
    info["shift_s"] = round(shift, 4)
    return info, shift


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


def _extract_sample_video(mov: Path, sample_dir: Path, *, start_s: float, duration_s: float,
                          resize_width: int, output_fps: float, max_frames: int,
                          crf: int, source_fps: float = 30.0) -> bool:
    """Slice, downsample, and encode a compact sample video in one ffmpeg pass.

    The former compact-video path first decoded every requested frame to PNG and
    then encoded those PNGs back to H.264.  For a fixed clip-level regressor
    that is needless disk I/O and roughly doubles the alignment wall time.  This
    direct path has exactly the same source window and resize contract while
    avoiding temporary files entirely.

    **Tail margin and frame-count assertion (EQ-018).**  ``output_fps`` places
    the final slot at ``duration_s - 1/source_fps``, which left only 0.4 output
    slots of margin; ``-ss`` input-seek quantises to the source's frame grid and
    ate it, so the ``fps`` filter flushed at EOF one frame short.  Every clip in
    the 3,040-sample MVP corpus came out with 31 frames while its synthesised
    ``frame_times`` asserted 32 — and nothing noticed, because the loader
    stretches whatever frames exist across the requested count.  Two changes:
    request a couple of source frames of extra tail (``-frames:v`` still bounds
    the output at ``max_frames``), and verify the produced count, failing the
    sample loudly rather than emitting a clip whose pixels and labels disagree.
    """
    sample_dir.mkdir(parents=True, exist_ok=True)
    out = sample_dir / "frames.mp4"
    margin_s = 2.0 / max(source_fps, 1e-6)
    r = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{start_s:.3f}", "-i", str(mov),
         "-t", f"{duration_s + margin_s:.3f}",
         "-vf", f"fps={output_fps:.8f},scale={resize_width}:-2",
         "-frames:v", str(max_frames), "-c:v", "libx264", "-crf", str(crf),
         "-pix_fmt", "yuv420p", str(out)],
        capture_output=True, text=True,
    )
    if r.returncode != 0 or not out.exists() or out.stat().st_size == 0:
        print(f"  {sample_dir.name}: direct video extract failed ({r.stderr[:120]})")
        out.unlink(missing_ok=True)
        return False
    produced = _video_frame_count(out)
    if produced != max_frames:
        # A short clip is silently stretched by the loader, so it must never be
        # written: the labels would assert content the pixels do not contain.
        print(f"  {sample_dir.name}: extract produced {produced} frames, expected {max_frames}")
        out.unlink(missing_ok=True)
        return False
    return True


def _video_frame_count(path: Path) -> int:
    """Frames actually decodable from a clip.

    ``CAP_PROP_FRAME_COUNT`` is frequently estimated from duration x fps rather
    than counted, and the defect this guards against is exactly one frame — so
    the header is not trustworthy at the precision required.  Decode and count.
    """
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return -1
    frames = 0
    while True:
        ok, _frame = capture.read()
        if not ok:
            break
        frames += 1
    capture.release()
    return frames


def _even_indices(n: int, max_n: int) -> list[int]:
    if n <= max_n:
        return list(range(n))
    return [round(i * (n - 1) / (max_n - 1)) for i in range(max_n)]


def align_segment(manifest_path: Path, *, pre_s: float, window_s: float, fps: int,
                  resize_width: int, max_frames: int, delta_override: float | None,
                  delete_mov: bool, anchor: str = "start",
                  video: bool = False, video_crf: int = 20,
                  direct_video: bool = False,
                  tap_calibrate: bool = False, tap_calibration_min_taps: int = 2,
                  tap_calibration_max_mad_s: float = 0.10,
                  tap_calibration_search_s: float = 4.0,
                  tap_calibration_width: int = 256) -> int:
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
    calibration_info: dict | None = None
    calibration_shift_s = 0.0
    try:
        if tap_calibrate:
            calibration_info, shift = _tap_calibration(
                manifest=manifest,
                mov=mov,
                started_at=started_at,
                fps=fps,
                delta_override=delta_override,
                manifest_delta=manifest_delta,
                min_taps=tap_calibration_min_taps,
                max_mad_s=tap_calibration_max_mad_s,
                search_after_s=tap_calibration_search_s,
                resize_width=tap_calibration_width,
            )
            if shift is None:
                # A failed calibration is a data-quality failure, not a reason to
                # manufacture labels from the fallback constants.  Preserve the .mov
                # for diagnosis and leave no .aligned marker so it can be retried.
                rejected = seg_dir / (manifest_path.stem + ".calibration_rejected.json")
                rejected.write_text(json.dumps(calibration_info, indent=2))
                print(f"[align] {manifest_path.name}: tap calibration rejected "
                      f"({calibration_info['reason']}); preserving {mov.name} "
                      f"and wrote {rejected.name}")
                raise TapCalibrationRejected(str(calibration_info["reason"]))
            calibration_shift_s = shift
            print(f"[align] {manifest_path.name}: tap calibration accepted "
                  f"(Δtap={calibration_info['tap_offset_s']}s, "
                  f"shift={calibration_shift_s:+.3f}s, "
                  f"n={len(calibration_info['inlier_offsets_s'])}, "
                  f"MAD={calibration_info['mad_s']}s)")
        for ev in gestures:
            delta = _delta_for(ev, delta_override, manifest_delta) + calibration_shift_s
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
            if direct_video:
                # Keep 32 evenly spaced time samples over the same window the
                # PNG path used.  The dataset selects across decoded frames, so
                # the explicit count both bounds storage and preserves temporal
                # coverage for the clip-level endpoint model.
                output_fps = (max_frames - 1) / max(dur - 1 / fps, 1 / fps)
                if not _extract_sample_video(
                    mov, sample_dir, start_s=start, duration_s=dur,
                    resize_width=resize_width, output_fps=output_fps,
                    max_frames=max_frames, crf=video_crf, source_fps=fps,
                ):
                    shutil.rmtree(sample_dir, ignore_errors=True)
                    continue
                frame_times = [round(i / output_fps - pre_s, 4) for i in range(max_frames)]
                frames_format = "mp4"
            else:
                frames_format = None
            if not direct_video:
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
                frames_format = ("mp4" if video and _encode_sample_video(
                    sample_dir, len(frame_times), fps, video_crf) else "png")
            meta = {
                **{k: v for k, v in ev.items()},                     # gesture params + call times + park
                "device": manifest.get("device"),
                "device_logical_w": dw, "device_logical_h": dh,
                "gesture_video_time_s": round(gv, 4),
                "capture_offset_s": delta,
                "capture_offset_source": (
                    "tap_self_calibrated" if calibration_info is not None
                    else "per_kind_fallback"
                ),
                "anchor": anchor,
                "frame_times": frame_times,
                "n_frames": len(frame_times),
                "segment_index": manifest.get("segment_index"),
                "session": seg_dir.name,
            }
            if calibration_info is not None:
                # Keep provenance with every sample; the marker below retains the
                # full segment report too.  Timing comes from rendered taps, while
                # the touch coordinates remain the command manifest's labels.
                meta["tap_calibration"] = calibration_info
            if anchor == "start":
                # temporal_trace_dataset._is_end_relative() keys off this: its presence
                # switches the label scheduler to the START-relative branch, which is what
                # these frame_times now are. Without it the scheduler would add the payload
                # duration and place every touch a full stroke too late.
                meta["gesture_start_monotonic"] = float(ev["t_call_start_epoch_s"])
            if video or direct_video:
                meta["frames_format"] = frames_format
            (sample_dir / "meta.json").write_text(json.dumps(meta, indent=2))
            saved += 1

    finally:
        claim.unlink(missing_ok=True)

    # mark processed + optionally drop the (large) .mov to free host space
    (seg_dir / (manifest_path.stem + ".aligned")).write_text(
        json.dumps({"samples": saved, "anchor": anchor,
                    "delta_s": sorted(deltas_used),
                    "tap_calibration": calibration_info}))
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
    ap.add_argument("--tap-calibrate", action="store_true",
                    help="Require per-segment timing calibration from known-position "
                         "stationary taps. Uses local frame differencing only for timing, "
                         "preserves the .mov and writes no samples when taps disagree.")
    ap.add_argument("--tap-calibration-min-taps", type=int, default=2,
                    help="Minimum detected tap marks required by --tap-calibrate (default: 2).")
    ap.add_argument("--tap-calibration-max-mad-s", type=float, default=0.10,
                    help="Maximum allowed robust tap-offset MAD in seconds (default: 0.10).")
    ap.add_argument("--tap-calibration-search-s", type=float, default=4.0,
                    help="Seconds after each tap command to search for its rendered mark (default: 4).")
    ap.add_argument("--tap-calibration-width", type=int, default=256,
                    help="Decode width for tap calibration only; sample output still uses --resize-width.")
    ap.add_argument("--delete-mov", action="store_true", help="Delete the .mov after a successful align.")
    ap.add_argument("--video", action="store_true",
                    help="Store each sample as one frames.mp4 instead of N PNGs: ~30x "
                         "smaller and 1 inode instead of N. The dataset loader reads "
                         "either format transparently.")
    ap.add_argument("--direct-video", action="store_true",
                    help="Slice and encode compact videos directly from the segment MOV. "
                         "Avoids temporary PNGs; implies --video.")
    ap.add_argument("--video-crf", type=int, default=20,
                    help="x264 CRF for --video. 20 keeps the orange trace intact "
                         "(measured 4.5%% error in the path-glow statistic).")
    args = ap.parse_args()
    if args.tap_calibration_min_taps < 1:
        ap.error("--tap-calibration-min-taps must be >= 1")
    if args.tap_calibration_max_mad_s < 0.0:
        ap.error("--tap-calibration-max-mad-s must be >= 0")
    if args.tap_calibration_search_s <= 0.0:
        ap.error("--tap-calibration-search-s must be > 0")
    if args.tap_calibration_width < 8:
        ap.error("--tap-calibration-width must be >= 8")
    if args.direct_video:
        args.video = True

    if args.segment:
        manifests = [args.segment]
    else:
        done = {p.stem for p in args.session.glob("*.aligned")}
        manifests = sorted(p for p in args.session.glob("segment_*.json") if p.stem not in done)
        if not manifests:
            print(f"[align] no un-aligned segments under {args.session}")
            return

    total = 0
    rejected = 0
    for m in manifests:
        try:
            total += align_segment(
                m, pre_s=args.pre_s, window_s=args.window_s, fps=args.fps,
                resize_width=args.resize_width, max_frames=args.max_frames,
                delta_override=args.delta_s, delete_mov=args.delete_mov,
                anchor=args.anchor, video=args.video, video_crf=args.video_crf,
                direct_video=args.direct_video,
                tap_calibrate=args.tap_calibrate,
                tap_calibration_min_taps=args.tap_calibration_min_taps,
                tap_calibration_max_mad_s=args.tap_calibration_max_mad_s,
                tap_calibration_search_s=args.tap_calibration_search_s,
                tap_calibration_width=args.tap_calibration_width)
        except TapCalibrationRejected as exc:
            rejected += 1
            print(f"[align] {m.name} CALIBRATION REJECTED: {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"[align] {m.name} FAILED: {exc}")
    print(f"[align] done: {total} samples across {len(manifests)} segment(s)")
    if rejected:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
