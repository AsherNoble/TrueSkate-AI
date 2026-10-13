"""Descriptive summary of a finished helper cohort. Reads analysis output; changes no gate.

Hold diagnosis compares each run's samples after the hold with the path's other runs:
a run whose post-hold samples match the others exactly, but which lacks the stationary
end-of-hold sample, differs in observability rather than in delivered motion.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def describe(values):
    a = np.asarray(values, dtype=float)
    return dict(n=len(a), median=float(np.median(a)), sd_population=float(np.std(a)),
                min=float(a.min()), max=float(a.max()), p05=float(np.quantile(a, .05)),
                p95=float(np.quantile(a, .95)))


def hold_diagnosis(plan, result):
    paths = {p["path_id"]: p for p in plan["paths"]}
    rows = []
    for name, path in paths.items():
        points = path["gesture"]["command_points"]
        holds = [i for i, (a, b) in enumerate(zip(points, points[1:])) if a == b]
        if not holds:
            continue
        hold_end_ms = path["gesture"]["times_ms"][holds[0]+1]
        runs = [r for r in result["requests"] if r["path_id"] == name]
        # Samples strictly after the hold end plus one digitizer interval (8.33 ms).
        after = {r["request_id"]: [(t, (round(x, 3), round(y, 3)))
                                   for t, (x, y) in zip(r["touch_times_from_B_ms"], r["positions_pt"])
                                   if t > hold_end_ms+8.4] for r in runs}
        positions = [tuple(p for _, p in samples) for samples in after.values()]
        modal = max(set(positions), key=positions.count)
        modal_times = np.median([[t for t, _ in samples] for samples in after.values()
                                 if tuple(p for _, p in samples) == modal], axis=0)
        for r in runs:
            w = r["waypoints"][holds[0]]
            first, last = w["first_within_half_point_ms"], w["last_within_half_point_ms"]
            samples = after[r["request_id"]]
            same = tuple(p for _, p in samples) == modal
            rows.append(dict(request_id=r["request_id"], path_id=name, hold_knot=holds[0],
                             hold_end_command_ms=hold_end_ms,
                             observed_span_ms=None if first is None else last-first,
                             post_hold_positions_equal_modal_run=same,
                             post_hold_max_time_difference_from_modal_ms=float(np.max(np.abs(
                                 np.asarray([t for t, _ in samples])-modal_times))) if same and samples else None,
                             unique_touch_samples=r["unique_touch_samples"]))
    return rows


def summarize(plan, result, bound):
    groups = {}
    for name, g in result["groups"].items():
        runs = [r for r in result["requests"] if r["path_id"] == name]
        rep = g.get("repeatability_interpolated_common_time", {})
        groups[name] = dict(
            admitted=g["admitted"], attempted=g["attempted"],
            requested_duration_ms=runs[0]["requested_duration_ms"],
            delivered_duration_ms=g["delivered_duration_ms"],
            p95_absolute_release_deviation_from_median_ms=g.get("p95_absolute_release_deviation_from_median_ms"),
            p95_repeat_deviation_normalized=rep.get("p95_repeat_deviation_normalized"),
            worst_repeat_deviation_normalized=rep.get("worst_repeat_deviation_normalized"),
            max_spatial_sd_pt=rep.get("max_spatial_sd_pt"),
            start_error_pt_max=g["start_error_pt"]["max"], end_error_pt_max=g["end_error_pt"]["max"],
            median_max_deviation_from_linear_command_pt=float(np.median(
                [r["sampled_command_deviation_pt"]["max"] for r in runs])),
            median_unique_touch_samples=float(np.median([r["unique_touch_samples"] for r in runs])))
    onset = [b["touch_began_minus_submitted_s"]*1000 for b in bound]
    return dict(schema="physical-transfer-helper-summary-v1", gate=result["helper_gate"],
                attempted=result["attempted"], unbound_batches=result["unbound_batches"],
                missing_requests=result["missing_requests"], groups=groups,
                touch_began_minus_wda_submitted_ms=describe(onset),
                hold_diagnosis=hold_diagnosis(plan, result),
                interpretation=["descriptive only; the frozen helper gate above is the decision record",
                                "touch-began minus WDA submitted_to_ios uses the phone's own clocks"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True, help="bound receipts.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(json.loads(args.plan.read_text()), json.loads(args.result.read_text()),
                        json.loads(args.receipts.read_text()))
    with args.out.open("x") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
