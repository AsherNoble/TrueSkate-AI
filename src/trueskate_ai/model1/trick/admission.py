"""Laptop admission: calibrate each recording, then cut one 120-frame clip per run.

Kept from curved v1: native 60 fps PTS and decoded frame counts, three die-five markers
with the held-out middle within two frames, exact source frames with no padding, and
clip PTS within 1 ms of the source. No menu or editor checks (operator decision
2026-10-15; see schedule.py). A clip holds exactly one run's strokes and no other
contact, with five frames either side.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import subprocess

import numpy as np

from trueskate_ai.collection.die_five_calibration import detect_die_five_onset, die_five_points
from trueskate_ai.collection.wda_action_timing import validate_action_timing_report
from trueskate_ai.model1.curved.capture import file_sha256, probe_pts, source_time_base
from trueskate_ai.model1.curved.contract import save_json
from trueskate_ai.research.audit_calibration import fit_markers
from trueskate_ai.research.audit_video import _decode_source_frames
from .contract import EXPERIMENT, FRAME_COUNT, SCHEMA, SEED

TAP_S = .05  # markers and resets are 50 ms taps
TIMING_SOURCE = "rig clock + median rig-to-WDA offset of this recording's instrumented requests"


def calibrate(out: Path, planned: dict, execution: dict):
    """Video clock from the three die-five markers (instrumented W3C requests)."""
    performed = [e for e in execution["events"] if not e.get("skipped")]
    instrumented = [e for e in performed if e["instrumented"]]
    records = validate_action_timing_report(json.loads((out/"wda-timing.json").read_text()),
                                            expected_revision=planned["wda_revision"],
                                            expected_count=len(instrumented))
    pts = probe_pts(out/"original.mov")
    _decode_source_frames(out/"original.mov", pts, keep=False)  # every frame decodes, counts agree
    stamps, onsets, detections = {}, {}, {}
    started = execution["video"]["started_at_epoch_s"]
    for event, record in zip(instrumented, records):
        if event["kind"] != "control":
            continue
        approx = record["submitted_to_ios"]["epoch_s"] - started
        frames, times, numbers = _decode_source_frames(out/"original.mov", pts, start=approx-.75, end=approx+.75)
        detection = detect_die_five_onset(frames, times, command_s=approx, points=die_five_points(71),
                                          reference_window_s=.75)
        detections[event["role"]] = dict(**asdict(detection), source_frames=numbers)
        if not detection.accepted:
            raise ValueError("die-five calibration failed: " + event["role"])
        stamps[event["role"]], onsets[event["role"]] = record["submitted_to_ios"]["monotonic_s"], detection.onset_s
    fit, middle_error = fit_markers(stamps, onsets, float(np.median(np.diff(pts))))
    # Direct strokes have no WDA timing record: map their rig-clock call starts with the
    # median rig-to-WDA offset of the instrumented requests (the kickflip runs' estimate).
    offset = float(np.median([r["request_entered"]["monotonic_s"] - e["call_start_monotonic_s"]
                              for e, r in zip(instrumented, records)]))
    evidence = dict(accepted=True, fit=asdict(fit), heldout_middle_error_s=middle_error, detections=detections,
                    rig_to_wda_offset_s=offset, decoded_frame_count=len(pts),
                    effective_fps=(len(pts)-1)/(pts[-1]-pts[0]), marker_offset_pt=71)
    contacts = []  # (video begin, video end, run number or None)
    instrumented_records = iter(records)
    for event in performed:
        if event["instrumented"]:
            begin = fit.video_time_s(next(instrumented_records)["submitted_to_ios"]["monotonic_s"])
            contacts.append((begin, begin + fit.rate*TAP_S, None, event))
        else:
            stroke = next(r for r in planned["runs"] if r["number"] == event["run"])["strokes"][event["stroke"]]
            begin = fit.video_time_s(event["call_start_monotonic_s"] + offset)
            contacts.append((begin, begin + fit.rate*stroke["times_ms"][-1]/1000, event["run"], event))
    return pts, fit, evidence, contacts


def window_start(pts, onset: float, release: float, others, *, seed) -> int:
    """Uniform among source-frame starts that hold the run alone, with five frames either side."""
    pts = np.asarray(pts, dtype=float)
    first = int(np.searchsorted(pts, onset, side="left"))
    after = int(np.searchsorted(pts, release, side="right"))
    starts = [start for start in range(max(0, after+5-FRAME_COUNT), min(len(pts)-FRAME_COUNT, first-5)+1)
              if np.max(np.diff(pts[start:start+FRAME_COUNT])) <= .0500001
              and not any(b <= pts[start+FRAME_COUNT-1] and e >= pts[start] for b, e in others)]
    if not starts:
        raise ValueError("no 120-frame window holds this run alone with five frames either side")
    return int(np.random.default_rng(seed).choice(starts))


def encode_clips(original: Path, jobs) -> None:
    """One decode of the original; each clip keeps the source frames and time base."""
    filters = [f"[0:v]split={len(jobs)}" + "".join(f"[in{i}]" for i in range(len(jobs)))]
    for i, (start, _) in enumerate(jobs):
        filters.append(f"[in{i}]select=between(n\\,{start}\\,{start+FRAME_COUNT-1}),setpts=PTS-STARTPTS[out{i}]")
    args = ["ffmpeg", "-v", "error", "-i", str(original), "-filter_complex", ";".join(filters)]
    time_base = source_time_base(original)
    for i, (_, clip) in enumerate(jobs):
        args += ["-map", f"[out{i}]", "-an", "-fps_mode", "passthrough", "-enc_time_base", time_base,
                 "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-threads", "2", str(clip)]
    subprocess.run(args, check=True, capture_output=True)


def admit_recording(out: Path) -> dict:
    """Admit one recording's runs; failures are recorded per run and never retried silently."""
    out = Path(out)
    planned = json.loads((out/"planned.json").read_text())
    execution = json.loads((out/"execution.json").read_text())
    if execution["error"]:
        raise ValueError("failed recording cannot contribute clips: " + execution["error"])
    pts, fit, evidence, contacts = calibrate(out, planned, execution)
    rejected, jobs = {}, []
    for run in planned["runs"]:
        own = [c for c in contacts if c[2] == run["number"]]
        if not own:
            rejected[run["number"]] = "not sent"
            continue
        try:
            start = window_start(pts, min(c[0] for c in own), max(c[1] for c in own),
                                 [c[:2] for c in contacts if c[2] != run["number"]], seed=[SEED, run["number"]])
        except ValueError as exc:
            rejected[run["number"]] = str(exc)
            continue
        folder = out/f"run-{run['number']:05d}"
        folder.mkdir(exist_ok=False)
        jobs.append((run, own, start, folder))
    if jobs:
        encode_clips(out/"original.mov", [(start, folder/"clip.mp4") for _, _, start, folder in jobs])
    source_hash = file_sha256(out/"original.mov")
    admitted = []
    for run, own, start, folder in jobs:
        try:
            clip_pts = probe_pts(folder/"clip.mp4", recording=False)
            selected = pts[start:start+FRAME_COUNT]
            if len(clip_pts) != FRAME_COUNT:
                raise ValueError("clip must decode to 120 native frames")
            if not np.allclose(clip_pts-clip_pts[0], selected-selected[0], atol=.001):
                raise ValueError("encoded clip changed native timestamps")
            strokes = []
            for begin, end, _, event in own:
                stroke = {k: v for k, v in run["strokes"][event["stroke"]].items()}
                strokes.append(dict(stroke, onset_s=begin-selected[0], release_s=end-selected[0],
                                    timing_source=TIMING_SOURCE,
                                    call_s=event["call_end_monotonic_s"]-event["call_start_monotonic_s"]))
            meta = dict(schema=SCHEMA, experiment=EXPERIMENT, recording=out.name, run=run["number"],
                        run_kind=run["kind"], strokes=strokes, sent_strokes=len(own),
                        planned_strokes=len(run["strokes"]), device=planned["device"], park=planned["park"],
                        settings_note=planned["settings_note"], clock_rate=fit.rate,
                        frame_times_s=(selected-selected[0]).tolist(),
                        source_frame_numbers=list(range(start, start+FRAME_COUNT)),
                        source_frame_pts_s=selected.tolist(), source_video_sha256=source_hash,
                        clip_sha256=file_sha256(folder/"clip.mp4"), calibration=evidence,
                        admission=dict(accepted=True, frame_count=FRAME_COUNT, menu_checks="none (2026-10-15)"))
            save_json(folder/"meta.json", meta)
            admitted.append(run["number"])
        except Exception as exc:
            save_json(folder/"rejected.json", dict(reason=f"{type(exc).__name__}: {exc}"))
            rejected[run["number"]] = str(exc)
    return dict(accepted=True, admitted=admitted, rejected={str(k): v for k, v in rejected.items()},
                calibration=evidence)


def _admit_one(out: str) -> tuple[str, dict]:
    out = Path(out)
    try:
        result = admit_recording(out)
    except Exception as exc:
        result = dict(accepted=False, error=f"{type(exc).__name__}: {exc}", admitted=[], rejected={})
    save_json(out/"admission.json", result)
    return out.name, result


def admit_all(root: Path, *, jobs: int = 4) -> dict:
    """Admit every recording that has no admission yet (rerun a failure by moving its admission.json)."""
    pending = sorted(str(p.parent) for p in Path(root).glob("recordings/*/execution.json")
                     if not (p.parent/"admission.json").exists())
    with ProcessPoolExecutor(max_workers=max(1, jobs)) as pool:
        return dict(pool.map(_admit_one, pending))
