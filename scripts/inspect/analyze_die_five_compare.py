"""M1-DIE5-COMPARE phase 2: detector A vs B on every marker, fits, and null windows.

Detector A reproduces production exactly: the aligner's calibration window
(0.75 s before the WDA-submitted time, 4 s after, 256 px decode, exact source
frame times) and the centre-only ``detect_tap_onset``. Detector B runs the frozen
die-five consensus on the same decoded window. No human labels are used here.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from trueskate_ai.collection.die_five_calibration import (
    DIE_FIVE_OFFSET_PT,
    DIE_FIVE_POINTS,
    detect_die_five_onset,
    die_five_points,
)
from trueskate_ai.collection.tap_timing_calibration import (
    detect_tap_onset,
    fit_two_anchor_timeline,
)

REFERENCE_S = 0.75
SEARCH_AFTER_S = 4.0
DECODE_WIDTH = 256
NULL_CLEAR_BEFORE_S = 3.0
NULL_CLEAR_AFTER_S = 1.5
NULL_WINDOW_AFTER_S = 1.0


def _aligner():
    path = Path(__file__).parents[1] / "collection" / "align_xctest_traces.py"
    spec = importlib.util.spec_from_file_location("die5_compare_aligner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ALIGN = _aligner()


def _frame(all_times: list[float], onset_s: float | None) -> int | None:
    if onset_s is None:
        return None
    return min(range(len(all_times)), key=lambda i: abs(all_times[i] - onset_s))


def detect_both(mov: Path, all_times: list[float], approx_s: float, *, after_s: float = SEARCH_AFTER_S,
                points=DIE_FIVE_POINTS) -> dict:
    frames, times = ALIGN._decode_calibration_window(
        mov, command_video_s=approx_s, fps=30, reference_window_s=REFERENCE_S,
        search_after_s=after_s, resize_width=DECODE_WIDTH, source_frame_times=all_times,
    )
    if not frames:
        return {"decoded": False}
    a = detect_tap_onset(frames, times, point_xy=(0.5, 0.5), command_s=approx_s,
                         reference_window_s=REFERENCE_S)
    b = detect_die_five_onset(frames, times, command_s=approx_s, points=points,
                              reference_window_s=REFERENCE_S)
    first = all_times.index(times[0])
    return {
        "decoded": True,
        "a_frame": _frame(all_times, None if a is None else a.onset_s),
        "a_score": None if a is None else round(a.score, 3),
        "b_frame": None if b.onset_frame is None else first + b.onset_frame,
        "b_votes": b.votes,
        "b_point_frames": [None if f is None else first + f for f in b.point_frames],
    }


def analyze_segment(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    mov = manifest_path.with_suffix(".mov")
    offset_pt = manifest.get("die_five_offset_pt") or DIE_FIVE_OFFSET_PT
    points = die_five_points(offset_pt)
    all_times = ALIGN._probe_video_frame_times(mov)
    started = manifest["started_at_epoch_s"]
    submits = [e["wda_submitted_epoch_s"] - started for e in manifest["gestures"]]
    markers = []
    for event in manifest["gestures"]:
        if not event.get("calibration_control"):
            continue
        approx = event["wda_submitted_epoch_s"] - started
        markers.append({
            "wda_action_sequence": event["wda_action_sequence"],
            "role": event["calibration_role"],
            "marker": event.get("calibration_marker", "die_five"),
            "wda_submitted_monotonic_s": event["wda_submitted_monotonic_s"],
            "approx_video_s": round(approx, 4),
            **detect_both(mov, all_times, approx, points=points),
        })

    fits = {}
    by_role = {m["role"]: m for m in markers if m["role"] in ("start", "end")}
    for name in ("a", "b"):
        start, end = by_role.get("start", {}), by_role.get("end", {})
        if start.get(f"{name}_frame") is None or end.get(f"{name}_frame") is None:
            fits[name] = None
            continue
        try:
            fit = fit_two_anchor_timeline(
                start["wda_submitted_monotonic_s"], all_times[start[f"{name}_frame"]],
                end["wda_submitted_monotonic_s"], all_times[end[f"{name}_frame"]],
            )
        except ValueError as exc:
            fits[name] = {"rejected": str(exc)}
            continue
        fits[name] = {"rate": fit.rate, "intercept_s": fit.intercept_s,
                      "mid_predicted_s": {str(m["wda_action_sequence"]): fit.video_time_s(m["wda_submitted_monotonic_s"])
                                          for m in markers if m["role"] == "mid"}}

    # Null windows: no WDA submission from 3 s before to 1.5 s after the anchor.
    nulls = []
    t = REFERENCE_S + 0.1
    while t + NULL_WINDOW_AFTER_S < all_times[-1]:
        if all(not (t - NULL_CLEAR_BEFORE_S <= s <= t + NULL_CLEAR_AFTER_S) for s in submits):
            result = detect_both(mov, all_times, t, after_s=NULL_WINDOW_AFTER_S, points=points)
            nulls.append({"anchor_video_s": round(t, 3), **result})
            t += REFERENCE_S + NULL_WINDOW_AFTER_S
        else:
            t += 0.25
    # Post-reset null window: recording start to just before the start marker
    # is submitted. The board may still be moving here (the observed fault state).
    start_marker = next((m for m in markers if m["role"] == "start"), None)
    post_reset = None
    if start_marker is not None and start_marker["approx_video_s"] - 0.1 > all_times[0] + 0.2:
        span = start_marker["approx_video_s"] - 0.1 - REFERENCE_S
        post_reset = {"scanned_s": round(start_marker["approx_video_s"] - 0.1 - all_times[0], 3),
                      **detect_both(mov, all_times, REFERENCE_S, after_s=max(0.05, span),
                                   points=points)}
    return {
        "segment": str(manifest_path),
        "device": manifest["device"],
        "park": manifest["park"],
        "die_five_offset_pt": offset_pt,
        "frame_count": len(all_times),
        "markers": markers,
        "fits": fits,
        "null_windows": nulls,
        "null_scanned_s": round(len(nulls) * (REFERENCE_S + NULL_WINDOW_AFTER_S), 3),
        "post_reset_null": post_reset,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    results = []
    for manifest in sorted(args.root.rglob("segment_*.json")):
        if manifest.name.endswith(".wda-action-timings.json"):
            continue
        results.append(analyze_segment(manifest))
        seg = results[-1]
        summary = [(m["role"][0], m["marker"][0], m.get("a_frame"), m.get("b_frame")) for m in seg["markers"]]
        rates = {k: (None if v is None else v.get("rate", "rej")) for k, v in seg["fits"].items()}
        print(manifest.parent.name, rates, summary, flush=True)
    args.out.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
