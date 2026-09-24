"""Evaluate five independent onset detections in bounded die-five recordings."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2

from scripts.inspect.probe_five_simultaneous_taps import FIVE_POINTS
from trueskate_ai.collection.tap_timing_calibration import detect_tap_onset


def decode(video: Path) -> tuple[list, list[float]]:
    payload = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries", "frame=best_effort_timestamp_time", "-of", "json",
        str(video),
    ]))
    times = [float(frame["best_effort_timestamp_time"]) for frame in payload["frames"]]
    capture = cv2.VideoCapture(str(video))
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(cv2.resize(frame, (414, 896)), cv2.COLOR_BGR2GRAY))
    finally:
        capture.release()
    if len(frames) != len(times):
        raise ValueError(f"decoded {len(frames)} frames but found {len(times)} PTS in {video}")
    return frames, times


def analyze_recording(metadata: Path) -> dict:
    data = json.loads(metadata.read_text())
    frames, times = decode(metadata.with_suffix(".mov"))
    started = data["recording"]["started_at_epoch_s"]
    results = []
    # The host call is not the pixel time. Search for each visible marker in
    # chronological order, advancing only after a real three-point consensus.
    search_s = data["events"][0]["call_start_epoch_s"] - started + 0.5
    for event in data["events"]:
        command_s = event["call_start_epoch_s"] - started
        detections = []
        for x, y in FIVE_POINTS:
            onset = detect_tap_onset(
                frames, times, point_xy=(x / 414, y / 896), command_s=search_s,
                reference_window_s=0.5,
            )
            if onset is None:
                detections.append({"point": [x, y], "frame": None})
                continue
            frame_index = min(range(len(times)), key=lambda j: abs(times[j] - onset.onset_s))
            detections.append({
                "point": [x, y], "frame": frame_index,
                "video_s": round(onset.onset_s, 6), "score": round(onset.score, 3),
            })
        found = [d["frame"] for d in detections if d["frame"] is not None]
        consensus = None
        votes = 0
        for candidate in sorted(set(found)):
            count = sum(abs(candidate - value) <= 1 for value in found)
            if count > votes:
                consensus, votes = candidate, count
        accepted = votes >= 3
        if not accepted:
            raise ValueError(f"no three-point onset consensus for marker {event['marker']} in {metadata}")
        search_s = times[consensus] + 1.0
        results.append({
            "marker": event["marker"],
            "command_start_video_s": round(command_s, 6),
            "detections": detections,
            "consensus_frame": consensus,
            "consensus_video_s": round(times[consensus], 6),
            "host_call_start_to_visible_s": round(times[consensus] - command_s, 6),
            "consensus_votes_within_one_frame": votes,
            "detected_count": len(found),
        })
    return {
        "recording": metadata.stem,
        "frame_count": len(frames),
        "markers": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    paths = sorted(args.recordings_dir.glob("recording_?.json"))
    if len(paths) != 3:
        parser.error(f"expected exactly three recordings; found {len(paths)}")
    result = {
        "method": "unchanged production centre/ring detector at five positions; "
                  "chronological search advances after each marker; "
                  "exploratory consensus requires three onsets within one frame",
        "recordings": [analyze_recording(path) for path in paths],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for recording in result["recordings"]:
        print(recording["recording"], [(m["detected_count"], m["consensus_votes_within_one_frame"],
               m["consensus_frame"]) for m in recording["markers"]])


if __name__ == "__main__":
    main()
