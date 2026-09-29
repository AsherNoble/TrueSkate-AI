"""Exploratory B' (all-candidates die-five consensus) on the Phase 2 analysis windows.

Post-hoc, found after the preregistered B rejected 12/60 mid markers. Uses the
same decoded windows as ``analyze_die_five_compare.py``; results are exploratory
and need a separate confirmatory test.
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

from trueskate_ai.collection.die_five_calibration import (
    DIE_FIVE_POINTS, earliest_candidate_consensus, tap_onset_candidates,
)
from scripts.inspect.analyze_die_five_compare import ALIGN, REFERENCE_S, SEARCH_AFTER_S

WINDOW_AFTER_NULL_S = 1.0


def b_prime(mov: Path, all_times: list[float], approx_s: float, after_s: float) -> dict:
    frames, times = ALIGN._decode_calibration_window(
        mov, command_video_s=approx_s, fps=30, reference_window_s=REFERENCE_S,
        search_after_s=after_s, resize_width=256, source_frame_times=all_times,
    )
    if not frames:
        return {"decoded": False}
    first = all_times.index(times[0])
    candidates = [tap_onset_candidates(frames, times, point_xy=p, command_s=approx_s,
                                       reference_window_s=REFERENCE_S) for p in DIE_FIVE_POINTS]
    frame, votes = earliest_candidate_consensus(candidates)
    return {"decoded": True, "bp_frame": None if frame is None else first + frame, "bp_votes": votes,
            "bp_candidates": [[first + f for f in c] for c in candidates]}


def run_segment(seg: dict) -> dict:
    mov = Path(seg["segment"]).with_suffix(".mov")
    all_times = ALIGN._probe_video_frame_times(mov)
    out = {"segment": seg["segment"], "markers": [], "null_windows": [], "post_reset_null": None}
    for m in seg["markers"]:
        if m["marker"] == "die_five":
            out["markers"].append({"wda_action_sequence": m["wda_action_sequence"], "role": m["role"],
                                   **b_prime(mov, all_times, m["approx_video_s"], SEARCH_AFTER_S)})
    for n in seg["null_windows"]:
        out["null_windows"].append({"anchor_video_s": n["anchor_video_s"],
                                    **b_prime(mov, all_times, n["anchor_video_s"], WINDOW_AFTER_NULL_S)})
    start = next((m for m in seg["markers"] if m["role"] == "start"), None)
    if seg.get("post_reset_null") and start:
        span = start["approx_video_s"] - 0.1 - REFERENCE_S
        out["post_reset_null"] = b_prime(mov, all_times, REFERENCE_S, max(0.05, span))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    segments = json.loads(args.analysis.read_text())
    with Pool(args.workers) as pool:
        results = pool.map(run_segment, segments)
    args.out.write_text(json.dumps(results, indent=1) + "\n")
    print(f"wrote {len(results)} segments to {args.out}")


if __name__ == "__main__":
    main()
