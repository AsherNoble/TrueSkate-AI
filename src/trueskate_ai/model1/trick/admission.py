"""Laptop admission: calibrate each recording, then cut one 120-frame clip per run.

Kept from curved v1: native 60 fps PTS and decoded frame counts, three die-five markers
with the held-out middle within two frames, exact source frames with no padding, and
clip PTS within 1 ms of the source. No menu or editor checks (operator decision
2026-10-15; see schedule.py). A clip holds exactly one run's strokes and no other
contact, with five frames either side.

Stroke labels are re-timed to the visible trail. A direct stroke has no WDA timing record,
so its first estimate is the rig clock plus an Appium-measured offset; that sat 2-3 frames
off the trail on the XR2 trial's faulted clips (TRICK-SPEED-20261015 spot check). The same
decode now times when the trail reaches points along each commanded path, and each stroke
moves by the median glow-corrected offset; a stroke too faint to read takes its recording's
median shift.
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
from .contract import EXPERIMENT, FRAME_COUNT, HEIGHT, SCHEMA, SEED, WIDTH

TAP_S = .05  # markers and resets are 50 ms taps
TIMING_SOURCE = "rig clock + median rig-to-WDA offset of this recording's instrumented requests"
TIMING_SOURCES = dict(
    trail="visible trail along the commanded path (median of glow-corrected probes)",
    recording="rig-clock estimate shifted by this recording's median visible-trail offset",
    estimate=TIMING_SOURCE)
# Trail probes: the kickflip timing detector's disc-versus-ring test (detect_tap_onset) at
# points every PATH_STEP_MS of commanded time. The trail's glow reaches a probe GLOW_PT ahead
# of the finger, so a probe reads GLOW_PT/speed early; fitted 2026-10-15 from how that lead
# grows on slower stretches (6 pt on the XR2 trick trial, 11 pt on the learn-replay pilot).
# Touch-down shows at the finger, so the start probe needs no correction. Probes the path,
# or an earlier stroke of the run, already passed within REACH_PT of are skipped.
PROBE_HALF_WINDOW_S = .35
PROBE_LEAD_S = .3
PATH_STEP_MS = 8.
GLOW_PT = 8.
MIN_SPEED_PT_MS = .5
REACH_PT = 20.
MIN_PROBES = 3
PROBE_SPREAD_S = .06  # probes further than this from a stroke's median are misreads


def probe_geometry(width: int, height: int, point):
    """Crop and disc/ring masks at a normalized point, with detect_tap_onset's radii."""
    cx, cy = point[0]*(width-1), point[1]*(height-1)
    scale = max(width/828., .25)
    core_radius = max(2., 10.*scale)
    inner = max(core_radius+1., 10., 20.*scale)
    outer = max(inner+2., 20., 40.*scale)
    x0, x1 = max(0, int(cx-outer)), min(width, int(cx+outer)+2)
    y0, y1 = max(0, int(cy-outer)), min(height, int(cy+outer)+2)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    squared = (xx-cx)**2 + (yy-cy)**2
    core, ring = squared <= core_radius**2, (squared > inner**2) & (squared <= outer**2)
    return ((slice(y0, y1), slice(x0, x1)), core, ring) if core.any() and ring.any() else None


def first_brightening(contrast, times, start_s: float, *, immediate=4., confirmation=6., history=3, lookahead=3):
    """detect_tap_onset's decision on a precomputed contrast series; None if nothing persists."""
    contrast, times = np.asarray(contrast, dtype=float), np.asarray(times, dtype=float)
    changes = np.diff(contrast)
    for index in np.flatnonzero(times[1:] >= start_s) + 1:
        if changes[index-1] < immediate:
            continue
        before = np.median(contrast[max(0, index-history):index])
        after = np.median(contrast[index:index+lookahead])
        if after - before >= confirmation:
            return float(times[index])
    return None


def path_samples(stroke: dict, step_ms: float = PATH_STEP_MS):
    """Commanded time (ms), normalized point and speed (points per ms) every step_ms along a stroke."""
    times = np.asarray(stroke["times_ms"], dtype=float)
    points = np.asarray(stroke["command_points"], dtype=float)
    ms = np.arange(0., times[-1] + 1e-9, step_ms)
    xy = np.stack([np.interp(ms, times, points[:, 0]), np.interp(ms, times, points[:, 1])], axis=1)
    piece = np.clip(np.searchsorted(times, ms, side="right") - 1, 0, len(times) - 2)
    lengths = np.hypot(*((points[piece+1] - points[piece])*[WIDTH, HEIGHT]).T)
    return ms, xy, lengths/np.maximum(times[piece+1] - times[piece], 1e-9)


def trail_shift(rows) -> tuple[float, int] | None:
    """Median glow-corrected offset (s) of a stroke's probes, given (ms, seen - estimate, speed) rows."""
    lags = [lag if ms == 0 else lag + GLOW_PT/speed/1000
            for ms, lag, speed in rows if ms == 0 or speed >= MIN_SPEED_PT_MS]
    if len(lags) < MIN_PROBES:
        return None
    middle = float(np.median(lags))
    kept = [lag for lag in lags if abs(lag - middle) <= PROBE_SPREAD_S]
    return (float(np.median(kept)), len(kept)) if len(kept) >= MIN_PROBES else None


def calibrate(out: Path, planned: dict, execution: dict):
    """Video clock from the three die-five markers (instrumented W3C requests)."""
    performed = [e for e in execution["events"] if not e.get("skipped")]
    instrumented = [e for e in performed if e["instrumented"]]
    records = validate_action_timing_report(json.loads((out/"wda-timing.json").read_text()),
                                            expected_revision=planned["wda_revision"],
                                            expected_count=len(instrumented))
    pts = probe_pts(out/"original.mov")
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
    # One full decode checks that every frame decodes and that counts agree, and times when
    # each stroke's trail reaches points along its commanded path.
    probes, by_frame, run_paths, run_release = [], {}, {}, {}
    for begin, end, run, event in contacts:
        if run is None:
            continue
        stroke = next(r for r in planned["runs"] if r["number"] == run)["strokes"][event["stroke"]]
        key = f"{run}:{event['stroke']}"
        dense = path_samples(stroke, 2.)[1]*[WIDTH, HEIGHT]
        earlier = run_paths.setdefault(run, [])
        for ms, point, speed in zip(*path_samples(stroke)):
            at = point*[WIDTH, HEIGHT]
            passed = dense[:max(0, int((ms - 30)/2))]
            if (len(passed) and np.hypot(*(passed - at).T).min() < REACH_PT) or any(
                    np.hypot(*(path - at).T).min() < REACH_PT for path in earlier):
                continue
            estimate = begin + ms/1000*fit.rate
            lo, hi = (int(np.searchsorted(pts, estimate + d)) for d in (-PROBE_HALF_WINDOW_S, PROBE_HALF_WINDOW_S))
            # The run's previous stroke still draws until its release, so search after it.
            probe = dict(key=key, ms=float(ms), speed=float(speed), point=point.tolist(), estimate=estimate,
                         search=max(estimate - PROBE_LEAD_S, run_release.get(run, -np.inf)),
                         lo=lo, hi=hi, values=[], geometry=False)
            probes.append(probe)
            for i in range(lo, hi):
                by_frame.setdefault(i, []).append(probe)
        earlier.append(dense)
        run_release[run] = end + .05
    count = iter(range(len(pts)))

    def inspect(frame):
        for probe in by_frame.get(next(count), ()):
            if probe["geometry"] is False:
                probe["geometry"] = probe_geometry(frame.shape[1], frame.shape[0], probe["point"])
            if probe["geometry"] is not None:
                crop, core, ring = probe["geometry"]
                patch = frame[crop].astype(np.float32).mean(axis=2)
                probe["values"].append(float(patch[core].mean() - patch[ring].mean()))

    _decode_source_frames(out/"original.mov", pts, keep=False, inspect=inspect)
    rows, visible = {}, {}
    for p in probes:
        seen = (first_brightening(p["values"], pts[p["lo"]:p["hi"]], p["search"])
                if len(p["values"]) == p["hi"] - p["lo"] > 1 else None)
        if p["ms"] == 0:
            visible[p["key"]] = seen
        if seen is not None:
            rows.setdefault(p["key"], []).append((p["ms"], seen - p["estimate"], p["speed"]))
    shifts = {key: shift for key, shift in ((k, trail_shift(r)) for k, r in rows.items()) if shift is not None}
    fallback = float(np.median([shift for shift, _ in shifts.values()])) if shifts else None
    timing, retimed = {}, []
    for begin, end, run, event in contacts:
        if run is not None:
            key = f"{run}:{event['stroke']}"
            if key in shifts:
                shift, used, source = *shifts[key], "trail"
            else:
                shift, used, source = (fallback, 0, "recording") if fallback is not None else (0., 0, "estimate")
            timing[key] = dict(estimated_onset_s=begin, shift_s=shift, probes=used, source=source)
            begin, end = begin + shift, end + shift
        retimed.append((begin, end, run, event))
    # Diagnostic: where each start point first shows, against the rig-clock estimate.
    lags = {k: v - timing[k]["estimated_onset_s"] for k, v in visible.items() if v is not None}
    evidence.update(stroke_visible_onsets_s=visible, stroke_visible_lags_s=lags,
                    stroke_visible_lag_s=dict(measured=len(lags), strokes=len(timing),
                                              median=float(np.median(list(lags.values()))) if lags else None),
                    stroke_timing=timing,
                    trail_timing=dict(measured=len(shifts), strokes=len(timing), median_shift_s=fallback,
                                      glow_pt=GLOW_PT, step_ms=PATH_STEP_MS))
    return pts, fit, evidence, retimed


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
    # -y replaces a partial clip left by a crashed attempt (admit_recording never re-encodes a settled run).
    args = ["ffmpeg", "-v", "error", "-y", "-i", str(original), "-filter_complex", ";".join(filters)]
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
        # A crashed earlier attempt can leave this folder with a partial clip, which is
        # re-encoded; a run that attempt already admitted or rejected is never redone.
        if (folder/"meta.json").exists() or (folder/"rejected.json").exists():
            raise ValueError(f"{folder.name} was already settled by an earlier admission")
        folder.mkdir(exist_ok=True)
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
            visible = evidence.get("stroke_visible_onsets_s", {})
            for begin, end, _, event in own:
                stroke = {k: v for k, v in run["strokes"][event["stroke"]].items()}
                key = f"{run['number']}:{event['stroke']}"
                seen = visible.get(key)
                timing = evidence.get("stroke_timing", {}).get(key) or dict(
                    estimated_onset_s=begin, shift_s=0., probes=0, source="estimate")
                strokes.append(dict(stroke, onset_s=begin-selected[0], release_s=end-selected[0],
                                    timing_source=TIMING_SOURCES[timing["source"]],
                                    estimated_onset_s=timing["estimated_onset_s"]-selected[0],
                                    retimed_by_s=timing["shift_s"], trail_probes=timing["probes"],
                                    visible_onset_s=None if seen is None else seen-selected[0],
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
    if jobs <= 1:
        return dict(map(_admit_one, pending))
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        return dict(pool.map(_admit_one, pending))
