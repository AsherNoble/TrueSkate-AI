"""60 Hz native-frame calibration and exact-index, random-onset extraction."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from trueskate_ai.collection.die_five_calibration import detect_die_five_onset, die_five_points
from trueskate_ai.collection.gameplay_filter import is_editor_frame, is_menu_frame
from trueskate_ai.collection.wda_action_timing import validate_action_timing_report
from trueskate_ai.research.audit_calibration import fit_markers
from trueskate_ai.research.audit_video import _decode_source_frames
from .contract import FRAME_COUNT, SCHEMA, Gesture, save_json

# The neutral idle navigation bar slides over live gameplay when a touch begins or
# the board idles; replay/menu colour, editor and foreground gates still apply.
# Operator decision 2026-10-11 (round 3 evidence in M1-CURVED-20261011).
MENU_POLICY = "allow_idle_navigation"


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            value.update(block)
    return value.hexdigest()


def verify_sample(path: Path):
    """Check persisted admission proof, including pixels, before recovering a dispatch."""
    meta = json.loads((path/"meta.json").read_text())
    if any((path/m).exists() for m in (".menu", ".trace_mismatch", ".contaminated")):
        raise ValueError("contaminated sample")
    admission = meta.get("admission", {})
    if (meta.get("schema") != SCHEMA or not meta.get("calibration", {}).get("accepted")
            or admission.get("accepted") is not True or admission.get("frame_count") != 120
            or admission.get("menu_checked_frames") != 120 or admission.get("native_frame_selection") is not True
            or admission.get("menu_policy") != MENU_POLICY
            or file_sha256(path/"clip.mp4") != meta.get("clip_sha256")):
        raise ValueError("incomplete or changed sample admission evidence")
    Gesture.from_dict(meta["gesture"])
    times = np.asarray(meta["frame_times_s"])
    source = np.asarray(meta["source_frame_pts_s"])
    numbers = meta["source_frame_numbers"]
    encoded = probe_pts(path/"clip.mp4")
    if (times.shape != (120,) or source.shape != (120,) or encoded.shape != (120,)
            or len(numbers) != 120 or any(b-a != 1 for a, b in zip(numbers, numbers[1:]))
            or not np.allclose(source-source[0], times, atol=1e-8)
            or not np.allclose(encoded-encoded[0], times, atol=.001)
            or np.max(np.diff(times)) > .0500001
            or not times[4] < meta["onset_s"] < meta["release_s"] < times[-5]):
        raise ValueError("invalid native sample timeline")
    _decode_source_frames(path/"clip.mp4", encoded, keep=False)
    return meta


def probe_pts(video: Path, *, recording: bool = True) -> np.ndarray:
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                             "-show_frames", "-show_entries", "frame=best_effort_timestamp_time",
                             "-of", "json", str(video)], check=True, capture_output=True, text=True)
    pts = np.asarray([float(f["best_effort_timestamp_time"])
                      for f in json.loads(result.stdout)["frames"]])
    if len(pts) < 2 or not np.isfinite(pts).all() or np.any(np.diff(pts) <= 0):
        raise ValueError("missing/nonmonotonic native timestamps")
    if recording:
        effective = (len(pts)-1)/(pts[-1]-pts[0])
        median = float(np.median(np.diff(pts)))
        if not 58 <= effective <= 62 or not .015 <= median <= .018:
            raise ValueError("capture is not native approximately 60 fps")
    return pts


def source_time_base(video: Path) -> str:
    """The recording's native stream time base as an ffmpeg ratio, e.g. ``1:600``."""
    value = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=time_base", "-of", "csv=p=0", str(video)],
                           check=True, capture_output=True, text=True).stdout.strip()
    num, den = (int(part) for part in value.split("/"))
    if num <= 0 or den <= 0:
        raise ValueError("invalid native time base")
    return f"{num}:{den}"


def window_start(pts, onset: float, release: float, *, contacts, own: int, seed: int) -> int:
    """Uniform among valid *source-frame starts*, never among invented times."""
    pts = np.asarray(pts, dtype=float)
    starts = []
    first = int(np.searchsorted(pts, onset, side="left"))
    after = int(np.searchsorted(pts, release, side="right"))
    lo = max(0, after+5-FRAME_COUNT)
    hi = min(len(pts)-FRAME_COUNT, first-5)
    for start in range(lo, hi+1):
        times = pts[start:start+FRAME_COUNT]
        if np.max(np.diff(times)) > .0500001:
            continue
        if any(i != own and begin <= times[-1] and end >= times[0]
               for i, (begin, end) in enumerate(contacts)):
            continue
        starts.append(start)
    if not starts:
        raise ValueError("no 120-frame single-contact crop with five-frame buffers")
    return int(np.random.default_rng(seed).choice(starts))


def calibrate_recording(out: Path, revision: str):
    out = Path(out)
    planned = json.loads((out/"planned.json").read_text())
    execution = json.loads((out/"execution.json").read_text())
    if execution["error"]:
        raise ValueError("failed recording cannot contribute clips")
    commands = planned["commands"]
    records = validate_action_timing_report(json.loads((out/"wda-timing.json").read_text()),
                                            expected_revision=revision, expected_count=len(commands))
    pts = probe_pts(out/"original.mov")
    # Check every decoded frame against ffprobe, without retaining minute-long pixels.
    _decode_source_frames(out/"original.mov", pts, keep=False)
    stamps, onsets, detections = {}, {}, {}
    for index, command in enumerate(commands):
        if command["kind"] != "control":
            continue
        stamp = records[index]["submitted_to_ios"]
        approx = stamp["epoch_s"]-execution["video"]["started_at_epoch_s"]
        frames, times, numbers = _decode_source_frames(out/"original.mov", pts,
                                                       start=approx-.75, end=approx+.75)
        detection = detect_die_five_onset(frames, times, command_s=approx,
                                          points=die_five_points(71), reference_window_s=.75)
        role = command["role"]
        detections[role] = dict(**asdict(detection), source_frames=numbers)
        if not detection.accepted:
            raise ValueError("die-five calibration failed: "+role)
        stamps[role], onsets[role] = stamp["monotonic_s"], detection.onset_s
    fit, middle_error = fit_markers(stamps, onsets, float(np.median(np.diff(pts))))
    evidence = dict(accepted=True, fit=asdict(fit), heldout_middle_error_s=middle_error,
                    detections=detections, decoded_frame_count=len(pts),
                    effective_fps=(len(pts)-1)/(pts[-1]-pts[0]),
                    native_pts_sha256=hashlib.sha256(pts.tobytes()).hexdigest(),
                    marker_offset_pt=71)
    save_json(out/"calibration.json", evidence)
    return planned, records, pts, fit, evidence


def extract_recording(out: Path, ledger, *, revision: str) -> list[Path]:
    """One source decode for clip encodes. No temporal sampling or synthesized PTS."""
    out = Path(out)
    planned, records, pts, fit, evidence = calibrate_recording(out, revision)
    commands = planned["commands"]
    contacts = []
    for spec, report in zip(commands, records):
        begin = fit.video_time_s(report["submitted_to_ios"]["monotonic_s"])
        duration = spec["gesture"]["times_ms"][-1]/1000 if spec["kind"] == "sample" else .05
        contacts.append((begin, begin+fit.rate*duration))
    jobs = []
    for index, command in enumerate(commands):
        if command["kind"] != "sample":
            continue
        candidate_id = command["candidate_id"]
        try:
            start = window_start(pts, *contacts[index], contacts=contacts, own=index,
                                 seed=planned["seed"]+candidate_id*9173)
        except ValueError as exc:
            ledger.settle(candidate_id, reason=str(exc))
            continue
        sample = out/f"sample-{candidate_id:06}"
        sample.mkdir(exist_ok=False)
        jobs.append((index, command, start, sample))
    if not jobs:
        return []
    filters = [f"[0:v]split={len(jobs)}"+''.join(f"[in{i}]" for i in range(len(jobs)))]
    for i, (_, _, start, _) in enumerate(jobs):
        filters.append(f"[in{i}]select=between(n\\,{start}\\,{start+FRAME_COUNT-1}),setpts=PTS-STARTPTS[out{i}]")
    args = ["ffmpeg", "-v", "error", "-i", str(out/"original.mov"),
            "-filter_complex", ";".join(filters)]
    # Without an explicit encoder time base, libx264 snaps PTS to a 1/60 s grid and
    # shifts every frame after a native 15 ms interval by 1.67 ms.
    time_base = source_time_base(out/"original.mov")
    for i, (_, _, _, sample) in enumerate(jobs):
        args += ["-map", f"[out{i}]", "-an", "-fps_mode", "passthrough",
                 "-enc_time_base", time_base,
                 "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                 "-threads", "2", str(sample/"clip.mp4")]
    subprocess.run(args, check=True, capture_output=True)
    source_hash = file_sha256(out/"original.mov")
    admitted = []
    for index, command, start, sample in jobs:
        candidate_id = command["candidate_id"]
        try:
            clip_pts = probe_pts(sample/"clip.mp4", recording=False)
            if len(clip_pts) != FRAME_COUNT:
                raise ValueError("clip must decode to 120 native frames")
            contaminated = []

            def inspect(bgr):
                rgb = bgr[..., ::-1]
                if is_editor_frame(rgb) or is_menu_frame(rgb, allow_idle_navigation=True):
                    contaminated.append(True)

            _decode_source_frames(sample/"clip.mp4", clip_pts, keep=False, inspect=inspect)
            if contaminated:
                (sample/".menu").touch(exist_ok=False)
                raise ValueError("menu/editor contamination")
            selected_pts = pts[start:start+FRAME_COUNT]
            if not np.allclose(clip_pts-clip_pts[0], selected_pts-selected_pts[0], atol=.001):
                raise ValueError("encoded clip changed native timestamps")
            session = ledger.session(planned["session"])
            meta = dict(schema=SCHEMA, candidate_id=candidate_id,
                        session_id=session["id"], park=session["park"],
                        phase=session["phase"], split=session["split"],
                        device=planned["device"], fingerprint=planned["fingerprint"],
                        waypoint=planned["waypoint"], settings_note=planned["settings_note"],
                        gesture=command["gesture"], clock_rate=fit.rate,
                        onset_s=contacts[index][0]-selected_pts[0],
                        release_s=contacts[index][1]-selected_pts[0],
                        frame_times_s=(selected_pts-selected_pts[0]).tolist(),
                        source_frame_numbers=list(range(start, start+FRAME_COUNT)),
                        source_frame_pts_s=selected_pts.tolist(),
                        source_video=str((out/"original.mov").resolve()),
                        source_video_sha256=source_hash,
                        clip_sha256=file_sha256(sample/"clip.mp4"),
                        source_recording_id=out.name, calibration=evidence,
                        admission=dict(accepted=True, frame_count=FRAME_COUNT,
                                       menu_checked_frames=FRAME_COUNT,
                                       menu_policy=MENU_POLICY,
                                       native_frame_selection=True))
            save_json(sample/"meta.json", meta)
            ledger.settle(candidate_id, sample_path=sample)
            admitted.append(sample)
        except Exception as exc:
            save_json(sample/"rejected.json", dict(reason=f"{type(exc).__name__}: {exc}"))
            ledger.settle(candidate_id, reason=str(exc))
    return admitted
