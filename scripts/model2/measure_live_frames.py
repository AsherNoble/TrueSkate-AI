"""Measure WDA's MJPEG stream as Model 2's live frame feed (LIVE-FRAMES-20261016, plan item 6).

``run`` (rig only): for each MJPEG config, one recorded minute on XR1 in The Workshop through the
trick runner's ``record_minute``: five die-five markers, a reset 3 s before each later marker, and
no gameplay strokes. The stream is read from a raw socket throughout, and each JPEG is stamped
with the rig's monotonic clock when its last byte arrives. (A ``requests`` reader blocks until a
full chunk arrives, so a frame's tail would wait for the next frame.) Every frame's arrival, size
and hash are saved, plus the JPEGs around each marker. The reader stops before the recording is
fetched. MJPEG settings go back to their old values in ``finally``. The run stops at the first
failure and never retries a recording or restarts a service.

``analyse`` (laptop): per marker, rig call → visible in the stream, WDA submit → visible in the
recording, and the stream minus the recording; per minute, stream and recording health.

``stream-check`` (rig): read the stream for a few seconds with no session (settings check).

    PYTHONPATH=src $RT/.venv/bin/python scripts/model2/measure_live_frames.py run \\
        --root /Users/training-server/trueskate-evidence/LIVE-FRAMES-20261016/run1 \\
        --env-file $RT/.env --ready-note '<who said XR1 is ready, and how it was checked>' \\
        --source-commit <sha> --configs current 30:50 60:50
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import statistics
import subprocess
import sys
import threading
import time

import numpy as np

EXPERIMENT = "LIVE-FRAMES"
DEVICE = "iPhone_XR"
PARK = "The Workshop"
SETTINGS_NOTE = "Workshop saved flatground waypoint; die-five markers and resets only, no gameplay strokes"
BUSY_LOCK = Path("/Users/training-server/trueskate-ai-runtime/tmp/RIG-BUSY-curved-recording")
MARKER_SLOTS = (("start", 1.5), ("m1", 12.), ("middle", 29.), ("m2", 45.), ("end", 57.))
RESET_SLOTS = (9., 26., 42., 54.)  # 3 s before each later marker, as in the trick minute
MJPEG_KEYS = ("mjpegServerFramerate", "mjpegScalingFactor", "mjpegServerScreenshotQuality")
STREAM_SETTLE_S = 2.     # read the stream this long before the recorder starts
WINDOW_S = (-.75, 1.5)   # stream JPEGs kept around each marker call
FRESH_SPAN_S = .3        # after a stream onset the screen changes; count repeated frames there
PART_HEADER = re.compile(rb"Content-Length:\s*(\d+)\r\n\r\n", re.IGNORECASE)
GATE = dict(min_fps=20., min_recording_fps=59.)


def parse_config(text: str) -> dict | None:
    """'current' keeps the session's settings; 'FPS:SCALE[:QUALITY]' sets framerate, scaling % and quality."""
    if text == "current":
        return None
    values = [int(v) for v in text.split(":")]
    if len(values) not in (2, 3) or not 1 <= values[0] <= 60 or not 1 <= values[1] <= 100 or (
            len(values) == 3 and not 1 <= values[2] <= 100):
        raise ValueError(f"config {text!r}: FPS:SCALE[:QUALITY] with fps 1–60, scale and quality 1–100")
    return dict(zip(MJPEG_KEYS, values))


def split_parts(buf: bytearray) -> list[bytes]:
    """Remove and return each complete multipart JPEG at the front of ``buf`` (WDA sends Content-Length)."""
    parts = []
    while True:
        match = PART_HEADER.search(buf)
        if not match:
            return parts
        start = match.end()
        end = start + int(match.group(1))
        if len(buf) < end:
            return parts
        parts.append(bytes(buf[start:end]))
        del buf[:end]


def jpeg_shape(data: bytes) -> list[int] | None:
    """[height, width] from the first start-of-frame marker."""
    i = 2
    while i + 9 < len(data) and data[i] == 0xFF:
        if data[i + 1] in (0xC0, 0xC1, 0xC2):
            return [int.from_bytes(data[i + 5:i + 7], "big"), int.from_bytes(data[i + 7:i + 9], "big")]
        i += 2 + int.from_bytes(data[i + 2:i + 4], "big")
    return None


class StreamReader:
    """Read WDA's MJPEG stream from a raw socket; stamp each JPEG when its last byte arrives."""

    def __init__(self, port: int, host: str = "127.0.0.1"):
        self.address = (host, port)
        self.arrivals: list[float] = []
        self.jpegs: list[bytes] = []
        self.error: Exception | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        try:
            with socket.create_connection(self.address, timeout=5) as sock:
                sock.sendall(b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
                buf, header = bytearray(), False
                while not self._stop.is_set():
                    chunk = sock.recv(1 << 16)
                    arrived = time.monotonic()
                    if not chunk:
                        raise ConnectionError("MJPEG stream closed")
                    buf += chunk
                    if not header:
                        end = buf.find(b"\r\n\r\n")
                        if end < 0:
                            continue
                        if b" 200 " not in bytes(buf[:end]).split(b"\r\n", 1)[0] + b" ":
                            raise ConnectionError(f"MJPEG server answered {bytes(buf[:end])[:60]!r}")
                        del buf[:end + 4]
                        header = True
                    for jpeg in split_parts(buf):
                        self.arrivals.append(arrived)
                        self.jpegs.append(jpeg)
                    if len(buf) > 8_000_000:
                        raise RuntimeError("MJPEG stream has no Content-Length parts")
        except Exception as exc:  # noqa: BLE001 - reported with the minute
            self.error = exc


class StoppingRecorder:
    """The XCTest recorder, but the stream reader stops before the recording is fetched."""

    def __init__(self, recorder, stream: StreamReader):
        self.recorder, self.stream = recorder, stream

    def start(self):
        return self.recorder.start()

    def stop_and_save(self, path):
        self.stream.stop()
        return self.recorder.stop_and_save(path)


def stream_stats(arrivals, jpegs, lo: float | None = None, hi: float | None = None) -> dict:
    """Frame rate, gaps, repeats, size and resolution of the frames that arrived in [lo, hi]."""
    arrivals = np.asarray(arrivals, dtype=float)
    keep = np.ones(len(arrivals), bool)
    if lo is not None:
        keep &= arrivals >= lo
    if hi is not None:
        keep &= arrivals <= hi
    times = arrivals[keep]
    chosen = [j for j, k in zip(jpegs, keep) if k]
    gaps = np.diff(times) * 1000
    hashes = [hashlib.sha1(j).digest() for j in chosen]
    return dict(frames=len(times),
                fps=(len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else None,
                gap_ms=dict(median=float(np.median(gaps)), p90=float(np.percentile(gaps, 90)),
                            p99=float(np.percentile(gaps, 99)), max=float(gaps.max())) if len(gaps) else None,
                repeated_share=(sum(a == b for a, b in zip(hashes, hashes[1:])) / (len(hashes) - 1)
                                if len(hashes) > 1 else None),
                jpeg_kb_median=statistics.median(len(j) for j in chosen) / 1024 if chosen else None,
                resolution_hw=jpeg_shape(chosen[0]) if chosen else None)


def save_stream(out: Path, stream: StreamReader, events: list) -> None:
    """Every frame's arrival, size and hash; the JPEG bytes only around each marker."""
    arrivals = np.asarray(stream.arrivals, dtype=float)
    keep = np.zeros(len(arrivals), bool)
    for event in events:
        if event.get("kind") == "control":
            call = event["call_start_monotonic_s"]
            keep |= (arrivals >= call + WINDOW_S[0]) & (arrivals <= call + WINDOW_S[1])
    saved = np.flatnonzero(keep)
    blobs = [stream.jpegs[i] for i in saved]
    np.savez_compressed(out/"stream.npz", arrivals=arrivals,
                        sizes=np.asarray([len(j) for j in stream.jpegs], dtype=np.int64),
                        hashes=np.asarray([hashlib.sha1(j).hexdigest()[:16] for j in stream.jpegs], dtype="S16"),
                        saved_index=saved.astype(np.int64),
                        saved_offsets=np.cumsum([0] + [len(b) for b in blobs]).astype(np.int64),
                        saved_bytes=np.frombuffer(b"".join(blobs), dtype=np.uint8))


def run(args, configs) -> None:
    from dotenv import load_dotenv
    load_dotenv(args.env_file, override=True)
    from trueskate_ai.collection.wda_action_timing import WDAActionTimingCapture, _http_json
    from trueskate_ai.collection.xctest_capture import XCTestScreenRecorder
    from trueskate_ai.model1.curved.capture import probe_pts
    from trueskate_ai.model1.curved.contract import save_json
    from trueskate_ai.model1.trick import live
    from trueskate_ai.model1.trick.recording import record_minute
    from trueskate_ai.research.curved_audit import marker_payload, reset_payload
    from trueskate_ai.sim.device import BUNDLE_ID, DEVICES, DeviceSession
    cfg = dict(next(d for d in DEVICES if d["name"] == DEVICE), appium_port=args.appium_port)
    udid = os.environ.get(cfg["env_key"])
    if not udid:
        raise ValueError(f"{DEVICE} UDID must come from the rig .env ({cfg['env_key']})")
    base = f"http://127.0.0.1:{cfg['wda_port']}"
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    # The trick runner's preflight (live.collect), unchanged.
    tunnel = subprocess.check_output(["launchctl", "print", "system/com.trueskate.remotexpc-tunnel"], text=True)
    if "state = running" not in tunnel:
        raise RuntimeError("root recording tunnel must be running; no start attempted")
    live.attachments_clear(udid, DEVICE)
    if _http_json(base + "/status").get("sessionId"):
        raise RuntimeError(f"{DEVICE} WDA session is already owned; do not disturb it")
    if _http_json(base + "/wda/activeAppInfo")["value"]["bundleId"] != BUNDLE_ID:
        raise RuntimeError("True Skate must already be frontmost")
    if _http_json(base + "/wda/video").get("value") is not None:
        raise RuntimeError("recorder is not idle; no start attempted")

    def sessions():
        value = _http_json(f"http://127.0.0.1:{args.appium_port}/appium/sessions").get("value")
        if not isinstance(value, list) or any(not isinstance(s, dict) or not s.get("id") for s in value):
            raise RuntimeError("Appium session discovery must be enabled and return a session list")
        return [s["id"] for s in value]

    if sessions():
        raise RuntimeError(f"{DEVICE} Appium session is already owned; do not disturb it")
    with (root/"sessions.jsonl").open("a") as handle:
        handle.write(json.dumps(dict(started=dt.datetime.now(dt.timezone.utc).isoformat(), device=DEVICE, park=PARK,
                                     ready_note=args.ready_note, configs=args.configs,
                                     source_commit=args.source_commit, pid=os.getpid())) + "\n")
    worker = DeviceSession(cfg, new_command_timeout_s=3600)
    original = None
    try:
        worker.connect()
        driver = worker.driver
        owned = _http_json(base + "/status").get("sessionId")
        if (driver.capabilities.get("udid") or driver.capabilities.get("appium:udid")) != udid:
            raise RuntimeError(f"Appium session does not confirm the configured {DEVICE} UDID")
        timing_state = _http_json(f"{base}/session/{owned}/wda/actionTiming").get("value")
        if (not isinstance(timing_state, dict) or timing_state.get("schema_version") != 1
                or timing_state.get("build_revision") != live.WDA_REVISION or timing_state.get("enabled")):
            raise RuntimeError("instrumented WDA identity/idle check failed; no start attempted")
        original = {k: driver.get_settings().get(k) for k in MJPEG_KEYS}
        print("MJPEG_SETTINGS_ORIGINAL", json.dumps(original), flush=True)

        def ownership():
            if sessions() != [driver.session_id] or not owned or _http_json(base + "/status").get("sessionId") != owned:
                raise RuntimeError("Appium/WDA session ownership changed")
            if driver.query_app_state(BUNDLE_ID) != 4 or worker._active_bundle_id() != BUNDLE_ID:
                raise RuntimeError("True Skate foreground lost")

        def guard():  # before every marker and reset, as before every trick run
            if _http_json(base + "/wda/activeAppInfo").get("value", {}).get("bundleId") != BUNDLE_ID:
                raise RuntimeError("True Skate foreground lost")

        def perform(payload):
            return driver.execute("actions", payload)

        def no_strokes(payload):
            raise RuntimeError("this measurement sends no gameplay strokes")

        commands = sorted([dict(kind="control", role=role, slot_s=slot, payload=marker_payload())
                           for role, slot in MARKER_SLOTS]
                          + [dict(kind="reset", slot_s=slot, payload=reset_payload()) for slot in RESET_SLOTS],
                          key=lambda c: c["slot_s"])
        for index, (text, config) in enumerate(zip(args.configs, configs), 1):
            if shutil.disk_usage(root).free < live.MIN_FREE_BYTES:
                raise RuntimeError("less than 8 GiB free; capture stopped")
            if config:
                driver.update_settings(config)
            settings = {k: driver.get_settings().get(k) for k in MJPEG_KEYS}
            if config and any(settings[k] != v for k, v in config.items()):
                raise RuntimeError(f"MJPEG settings not applied: asked {config}, got {settings}")
            live.attachments_clear(udid, DEVICE)
            ownership()
            if _http_json(base + "/wda/video").get("value") is not None:
                raise RuntimeError("recorder is not idle")
            perform(reset_payload())
            time.sleep(live.PRE_ROLL_SETTLE_S)
            ownership()
            stream = StreamReader(cfg["mjpeg_port"])
            stream.start()
            time.sleep(STREAM_SETTLE_S)
            if stream.error is not None or not stream.arrivals:
                stream.stop()
                raise RuntimeError(f"MJPEG stream gave no frames before recording ({stream.error!r})")
            out = root/"recordings"/f"m{index:04d}"
            plan = dict(commands=commands, runs=[], skipped_runs=[], next_run=0)
            metadata = dict(experiment=EXPERIMENT, recording=out.name, device=DEVICE, park=PARK,
                            park_source="ready note (operator's waypoint, agent screenshot check)",
                            settings_note=SETTINGS_NOTE, fps=60, mjpeg_config=text, mjpeg_settings=settings,
                            ready_note=args.ready_note, source_commit=args.source_commit)
            events = []
            try:
                events = record_minute(out=out, plan=plan,
                                       recorder=StoppingRecorder(XCTestScreenRecorder(driver, fps=60), stream),
                                       timing=WDAActionTimingCapture(wda_port=cfg["wda_port"],
                                                                     expected_revision=live.WDA_REVISION),
                                       perform=perform, perform_direct=no_strokes, guard=guard,
                                       revision=live.WDA_REVISION, metadata=metadata)
            finally:
                stream.stop()
                if (out/"execution.json").exists():
                    events = events or json.loads((out/"execution.json").read_text())["events"]
                    save_stream(out, stream, events)
            ownership()
            calls = [e["call_start_monotonic_s"] for e in events if e.get("kind") == "control"]
            summary = dict(recording=out.name, config=text, mjpeg=settings,
                           stream=stream_stats(stream.arrivals, stream.jpegs, calls[0] - 1.5, calls[-1] + 1.5),
                           stream_error=None if stream.error is None else repr(stream.error))
            try:
                pts = probe_pts(out/"original.mov", recording=False)
                summary["recording"] = dict(decoded_frames=len(pts), fps=(len(pts) - 1)/(pts[-1] - pts[0]))
            except Exception as exc:  # noqa: BLE001 - the laptop analysis repeats this
                summary["recording"] = f"not probed on the rig: {type(exc).__name__}: {exc}"
            save_json(out/"rig-summary.json", summary)
            print("MINUTE", json.dumps(summary), flush=True)
            del stream
    finally:
        try:
            if original is not None:
                worker.driver.update_settings({k: v for k, v in original.items() if v is not None})
                now = {k: worker.driver.get_settings().get(k) for k in MJPEG_KEYS}
                print("MJPEG_SETTINGS_RESTORED" if now == original else "MJPEG_RESTORE_MISMATCH",
                      json.dumps(dict(original=original, now=now)), flush=True)
        finally:
            worker.disconnect()


def _ms(value):
    return None if value is None else round(value * 1000, 1)


def _spread(values):
    values = [v for v in values if v is not None]
    return dict(n=len(values), min=min(values), median=statistics.median(values), max=max(values)) if values else None


def analyse_minute(out: Path) -> dict:
    """Marker delays in the stream and the recording, plus stream and recording health."""
    import cv2
    from trueskate_ai.collection.die_five_calibration import detect_die_five_onset, die_five_points
    from trueskate_ai.collection.wda_action_timing import validate_action_timing_report
    from trueskate_ai.model1.curved.capture import probe_pts
    from trueskate_ai.research.audit_video import _decode_source_frames
    planned = json.loads((out/"planned.json").read_text())
    execution = json.loads((out/"execution.json").read_text())
    report = dict(minute=out.name, config=planned["mjpeg_config"], mjpeg_settings=planned["mjpeg_settings"],
                  error=execution["error"])
    if execution["error"]:
        return report
    instrumented = [e for e in execution["events"] if e["instrumented"]]
    records = validate_action_timing_report(json.loads((out/"wda-timing.json").read_text()),
                                            expected_revision=planned["wda_revision"],
                                            expected_count=len(instrumented))
    # Phone monotonic minus rig monotonic. One-sided (the trick admission's estimate) carries the
    # Appium hop; two-sided (NTP-style) cancels it when the way there and back take equally long.
    one_sided = statistics.median(r["request_entered"]["monotonic_s"] - e["call_start_monotonic_s"]
                                  for e, r in zip(instrumented, records))
    two_sided = statistics.median(((r["request_entered"]["monotonic_s"] - e["call_start_monotonic_s"])
                                   + (r["request_finished"]["monotonic_s"] - e["call_end_monotonic_s"])) / 2
                                  for e, r in zip(instrumented, records))
    pts = probe_pts(out/"original.mov", recording=False)
    started = execution["video"]["started_at_epoch_s"]
    stream = np.load(out/"stream.npz")
    arrivals, saved = stream["arrivals"], stream["saved_index"]
    offsets, blob = stream["saved_offsets"], stream["saved_bytes"]
    grey = {int(i): cv2.imdecode(blob[offsets[k]:offsets[k + 1]], cv2.IMREAD_GRAYSCALE) for k, i in enumerate(saved)}
    hashes = stream["hashes"]
    points = die_five_points(71)
    markers = []
    for event, record in zip(instrumented, records):
        if event["kind"] != "control":
            continue
        call, call_end = event["call_start_monotonic_s"], event["call_end_monotonic_s"]
        approx = record["submitted_to_ios"]["epoch_s"] - started  # video seconds, as in admission.calibrate
        frames, times, _ = _decode_source_frames(out/"original.mov", pts, start=approx - .75, end=approx + .75)
        seen_rec = detect_die_five_onset(frames, times, command_s=approx, points=points, reference_window_s=.75)
        index = [int(i) for i in np.flatnonzero((arrivals >= call + WINDOW_S[0]) & (arrivals <= call + WINDOW_S[1]))
                 if int(i) in grey]
        detector, seen_stream = "default", None
        if len(index) > 4:
            args = ([grey[i] for i in index], arrivals[index])
            seen_stream = detect_die_five_onset(*args, command_s=call, points=points, reference_window_s=.75)
            if not seen_stream.accepted:  # a sparse stream may hold the glow for fewer than 3 frames
                detector = "one-frame lookahead"
                seen_stream = detect_die_five_onset(*args, command_s=call, points=points, reference_window_s=.75,
                                                    lookahead_frames=1, history_frames=2)
        stream_s = seen_stream.onset_s if seen_stream is not None and seen_stream.accepted else None
        if stream_s is not None and stream_s < call:
            stream_s, detector = None, "rejected: before the call"
        rec_delay = None if seen_rec.onset_s is None else seen_rec.onset_s - approx
        row = dict(role=event["role"], recording_votes=seen_rec.votes,
                   stream_votes=None if seen_stream is None else seen_stream.votes, stream_detector=detector,
                   call_return_ms=_ms(call_end - call), call_to_stream_ms=_ms(None if stream_s is None else stream_s - call),
                   submit_to_recording_ms=_ms(rec_delay))
        for name, offset in (("one_sided", one_sided), ("two_sided", two_sided)):
            submit_rig = record["submitted_to_ios"]["monotonic_s"] - offset
            to_stream = None if stream_s is None else stream_s - submit_rig
            row[f"call_to_submit_ms_{name}"] = _ms(submit_rig - call)
            row[f"submit_to_stream_ms_{name}"] = _ms(to_stream)
            row[f"stream_minus_recording_ms_{name}"] = _ms(None if to_stream is None or rec_delay is None
                                                           else to_stream - rec_delay)
        if stream_s is not None:  # the screen changes here, so a repeated frame is a stale capture
            span = np.flatnonzero((arrivals >= stream_s) & (arrivals <= stream_s + FRESH_SPAN_S))
            pairs = list(zip(span, span[1:]))
            row["repeats_after_onset"] = f"{sum(hashes[a] == hashes[b] for a, b in pairs)}/{len(pairs)}"
        markers.append(row)
    calls = [e["call_start_monotonic_s"] for e in instrumented if e["kind"] == "control"]
    inside = (arrivals >= calls[0] - 1.5) & (arrivals <= calls[-1] + 1.5)
    times = arrivals[inside]
    gaps = np.diff(times) * 1000
    marker_keys = [k for k in markers[0] if "_ms" in k] if markers else []
    report.update(
        rig_to_wda_offset_s=dict(one_sided=one_sided, two_sided=two_sided, difference_ms=_ms(one_sided - two_sided)),
        stream=dict(frames=int(inside.sum()), fps=(len(times) - 1) / (times[-1] - times[0]),
                    gap_ms=dict(median=float(np.median(gaps)), p90=float(np.percentile(gaps, 90)),
                                p99=float(np.percentile(gaps, 99)), max=float(gaps.max())),
                    repeated_share=float(np.mean(hashes[inside][1:] == hashes[inside][:-1])),
                    jpeg_kb_median=float(np.median(stream["sizes"][inside])) / 1024,
                    resolution_hw=list(next(iter(grey.values())).shape) if grey else None),
        recording=dict(decoded_frames=len(pts), fps=(len(pts) - 1) / (pts[-1] - pts[0]),
                       median_frame_ms=float(np.median(np.diff(pts))) * 1000, max_gap_ms=float(np.diff(pts).max()) * 1000),
        markers=markers, marker_summary={k: _spread([m[k] for m in markers]) for k in dict.fromkeys(marker_keys)})
    fps, frame_ms = report["stream"]["fps"], 1000 / report["stream"]["fps"]
    spread = report["marker_summary"].get("submit_to_stream_ms_two_sided")
    report["gate"] = dict(
        fps_ok=bool(fps >= GATE["min_fps"]), recording_ok=bool(report["recording"]["fps"] >= GATE["min_recording_fps"]),
        delay_steady=None if not spread or spread["n"] < 3 else bool(spread["max"] - spread["min"] <= frame_ms),
        stream_frame_ms=float(frame_ms))
    report["gate"]["passes"] = all(report["gate"][k] is True for k in ("fps_ok", "recording_ok", "delay_steady"))
    return report


def analyse(root: Path) -> None:
    from trueskate_ai.model1.curved.contract import save_json
    rows = []
    for out in sorted((root/"recordings").glob("m[0-9][0-9][0-9][0-9]")):
        report = analyse_minute(out)
        save_json(out/"live-frames.json", report)
        rows.append(report)
        if report["error"]:
            print(out.name, report["config"], "ERROR", report["error"])
            continue
        s, r, m = report["stream"], report["recording"], report["marker_summary"]
        print(f"{out.name} {report['config']:>8} {s['resolution_hw']} stream {s['fps']:.1f} fps "
              f"gap p50/p90/max {s['gap_ms']['median']:.0f}/{s['gap_ms']['p90']:.0f}/{s['gap_ms']['max']:.0f} ms "
              f"repeats {s['repeated_share']:.2f} jpeg {s['jpeg_kb_median']:.0f} KB | recording {r['fps']:.2f} fps "
              f"| gate {report['gate']}")
        for key in ("call_to_stream_ms", "submit_to_recording_ms", "submit_to_stream_ms_two_sided",
                    "stream_minus_recording_ms_two_sided", "stream_minus_recording_ms_one_sided"):
            print(f"    {key}: {m.get(key)}")
        for row in report["markers"]:
            print("    ", json.dumps(row))
    save_json(root/"summary.json", dict(experiment=EXPERIMENT, minutes=rows))


def stream_check(seconds: float, port: int) -> None:
    """Read the stream with no session: what WDA's MJPEG settings are now."""
    stream = StreamReader(port)
    stream.start()
    time.sleep(seconds)
    stream.stop()
    print("STREAM_CHECK", json.dumps(dict(stream_stats(stream.arrivals, stream.jpegs),
                                          error=None if stream.error is None else repr(stream.error))), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="record the configs on XR1 (rig only)")
    r.add_argument("--root", type=Path, required=True, help="absolute run directory on the rig")
    r.add_argument("--env-file", type=Path, required=True)
    r.add_argument("--ready-note", required=True, help="who said XR1 is ready, and how it was checked")
    r.add_argument("--source-commit", required=True)
    r.add_argument("--appium-port", type=int, default=4725, help="session-discovery Appium (the temporary 4725)")
    r.add_argument("--configs", nargs="+", default=["current"], help="'current' and/or FPS:SCALE[:QUALITY]")
    r.add_argument("--busy-lock", type=Path, default=BUSY_LOCK)
    a = sub.add_parser("analyse", help="analyse a fetched run (laptop)")
    a.add_argument("--root", type=Path, required=True)
    c = sub.add_parser("stream-check", help="read the stream for a few seconds, no session (rig)")
    c.add_argument("--seconds", type=float, default=3.)
    c.add_argument("--port", type=int, default=9100)
    args = ap.parse_args()
    if args.command == "analyse":
        return analyse(args.root)
    if args.command == "stream-check":
        return stream_check(args.seconds, args.port)
    if not args.root.is_absolute() or not args.env_file.is_absolute():
        ap.error("--root and --env-file must be absolute")
    if not args.env_file.is_file():
        ap.error("rig .env file not found")
    configs = [parse_config(c) for c in args.configs]
    from trueskate_ai.model1.trick import live
    live.require_rig()  # before the lock: its default path exists only on the rig
    try:
        with live.busy_lock(args.busy_lock, owner=f"{EXPERIMENT} MJPEG measurement on {DEVICE}", out=args.root):
            run(args, configs)
    except BaseException as exc:
        print("LIVE_FRAMES_FAILED", f"{type(exc).__name__}: {exc}", flush=True)
        raise
    print("LIVE_FRAMES_DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
