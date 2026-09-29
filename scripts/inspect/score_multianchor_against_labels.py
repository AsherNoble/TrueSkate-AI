#!/usr/bin/env python3
"""Score multi-anchor consensus decisions against Phase 3 human labels (frozen pre-labels).

For each Phase 2 segment, the consensus fit is run on detector A's frames for
every marker (the production detector). Against blind human first-touch frames:

1. outlier decision vs A gross error (>1 frame from the human frame);
2. for each labelled marker, the error (frames) of the time predicted by A's
   two-anchor start+end fit versus the multi-anchor fit, leaving that marker
   out of the multi-anchor fit so it is not scored against itself.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from trueskate_ai.collection.tap_timing_calibration import fit_multi_anchor_timeline

FPS = 30.0
_path = Path(__file__).resolve().parents[1] / "collection" / "align_xctest_traces.py"
_spec = importlib.util.spec_from_file_location("multianchor_score_aligner", _path)
ALIGN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ALIGN)


def human_frames(key: dict, labels: dict) -> dict[tuple, int]:
    out = {}
    for item in key["items"]:
        label = labels.get(item["id"])
        if label and not label.get("uncertain") and label.get("frame_index_0based") is not None:
            out[(item["segment"], item["wda_action_sequence"])] = item["first_global_frame"] + int(label["frame_index_0based"])
    return out


def score(analysis: list[dict], human: dict[tuple, int], frame_times=ALIGN._probe_video_frame_times) -> dict:
    confusion = {"outlier_and_gross": 0, "outlier_not_gross": 0, "inlier_and_gross": 0, "inlier_not_gross": 0}
    errors = {"two_anchor": [], "multi_anchor_loo": []}
    rejected_segments = 0
    for seg in analysis:
        times = frame_times(Path(seg["segment"]).with_suffix(".mov"))
        markers = [m for m in seg["markers"] if m.get("a_frame") is not None]
        wda = [m["wda_submitted_monotonic_s"] for m in markers]
        video = [times[m["a_frame"]] for m in markers]
        try:
            fit = fit_multi_anchor_timeline(wda, video)
        except ValueError:
            rejected_segments += 1
            continue
        for index, m in enumerate(markers):
            h = human.get((seg["segment"], m["wda_action_sequence"]))
            if h is None:
                continue
            gross = abs(m["a_frame"] - h) > 1
            outlier = index in fit.outlier_indices
            confusion[f"{'outlier' if outlier else 'inlier'}_{'and_gross' if gross else 'not_gross'}"] += 1
            two = seg["fits"].get("a")
            if two and "rate" in two:
                errors["two_anchor"].append((two["intercept_s"] + two["rate"] * m["wda_submitted_monotonic_s"] - times[h]) * FPS)
            rest = [i for i in range(len(markers)) if i != index]
            try:
                loo = fit_multi_anchor_timeline([wda[i] for i in rest], [video[i] for i in rest])
                errors["multi_anchor_loo"].append((loo.video_time_s(m["wda_submitted_monotonic_s"]) - times[h]) * FPS)
            except ValueError:
                pass

    def summary(values):
        if not values:
            return None
        absolute = sorted(abs(v) for v in values)
        return {"n": len(values), "within_1_frame": sum(v <= 1.0 for v in absolute),
                "gross_gt_1": sum(v > 1.0 for v in absolute), "max_abs_frames": round(absolute[-1], 2)}

    return {"segments": len(analysis), "multi_anchor_rejected_segments": rejected_segments,
            "outlier_vs_A_gross": confusion,
            "prediction_error_frames": {k: summary(v) for k, v in errors.items()}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--key", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    labels = json.loads(args.labels.read_text())
    key = json.loads(args.key.read_text())
    if str(key["seed"]) != str(labels["selection_seed"]):
        raise SystemExit("labels were made for a different selection")
    result = score(json.loads(args.analysis.read_text()), human_frames(key, labels["labels"]))
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
