"""Post-hoc hold criterion (operator-approved 2026-10-13, after the frozen gate failed).

The frozen helper gate stays the decision of record. This replaces only its hold
occupancy check, which required UIKit to emit a stationary end-of-hold sample:
a hold passes if its point is observed, no sample inside the commanded hold window
is more than 0.5 pt from it, and the run's post-hold motion (samples after the
window that have left the hold point) equals the path's modal run (timestamps
within 1 ms). Motion compares distinct consecutive positions,
because UIKit's stationary repeats are the artefact being amended; that choice was
made after seeing one run with an extra stationary endpoint sample, and the literal
sample-by-sample comparison is reported alongside.
"""
import argparse
import json
from pathlib import Path

import numpy as np

HOLD_FAILURE = ": hold occupancy unmeasured/too short"
POINT_PT, TIME_MS = .5, 1.


def motion(samples):
    """Distinct consecutive positions with the time each was first reached."""
    out = []
    for t, p in samples:
        if not out or out[-1][1] != p:
            out.append((t, p))
    return out


def hold_checks(plan, result):
    rows = []
    for path in plan["paths"]:
        points = path["gesture"]["command_points"]
        times = path["gesture"]["times_ms"]
        holds = [i for i, (a, b) in enumerate(zip(points, points[1:])) if a == b]
        runs = [r for r in result["requests"] if r["path_id"] == path["path_id"]]
        for i in holds:
            target = np.asarray(points[i])*[414, 896]
            start, end = times[i], times[i+1]
            # Post-hold motion: samples after the hold window that have left the hold point.
            # Samples at the hold point are the stationary ones whose presence varies.
            after = {}
            for r in runs:
                samples = [(t, (round(x, 3), round(y, 3)))
                           for t, (x, y) in zip(r["touch_times_from_B_ms"], r["positions_pt"])
                           if t > end and np.hypot(x-target[0], y-target[1]) > POINT_PT]
                after[r["request_id"]] = samples
            literal = [tuple(p for _, p in s) for s in after.values()]
            collapsed = [tuple(p for _, p in motion(s)) for s in after.values()]
            literal_modal = max(set(literal), key=literal.count)
            modal = max(set(collapsed), key=collapsed.count)
            modal_times = np.median([[t for t, _ in motion(s)] for s in after.values()
                                     if tuple(p for _, p in motion(s)) == modal], axis=0)
            for r in runs:
                t = np.asarray(r["touch_times_from_B_ms"])
                distance = np.linalg.norm(np.asarray(r["positions_pt"])-target, axis=1)
                window = (t >= start) & (t <= end)
                moved = motion(after[r["request_id"]])
                same = tuple(p for _, p in moved) == modal
                lag = float(np.max(np.abs(np.asarray([x for x, _ in moved])-modal_times))) if same and moved else 0.
                literal_same = tuple(p for _, p in after[r["request_id"]]) == literal_modal
                checks = dict(observed=bool(np.any(distance <= POINT_PT)),
                              stationary_in_window=bool(np.all(distance[window] <= POINT_PT)),
                              post_hold_motion_equals_modal=same,
                              post_hold_time_within_1ms=lag <= TIME_MS)
                rows.append(dict(request_id=r["request_id"], path_id=path["path_id"], hold_knot=i,
                                 window_ms=[start, end], samples_in_window=int(window.sum()),
                                 post_hold_max_time_difference_ms=lag, **checks, passed=all(checks.values()),
                                 literal_post_hold_samples_equal_modal=literal_same))
    return rows


def amend(plan, result):
    frozen = result["helper_gate"]
    rows = hold_checks(plan, result)
    kept = [f for f in frozen["failures"] if not f.endswith(HOLD_FAILURE)]
    failures = kept+[r["request_id"]+": amended hold check failed" for r in rows if not r["passed"]]
    literal = kept+[r["request_id"]+": literal post-hold samples differ" for r in rows
                    if not (r["passed"] and r["literal_post_hold_samples_equal_modal"])]
    return dict(schema="physical-transfer-helper-gate-amended-v1",
                frozen_gate=frozen, frozen_gate_is_decision_of_record=True,
                amendment="post hoc, operator-approved 2026-10-13; see module docstring",
                amended_gate=dict(passed=not failures, failures=failures),
                literal_wording_gate=dict(passed=not literal, failures=literal),
                holds=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    amended = amend(json.loads(args.plan.read_text()), json.loads(args.result.read_text()))
    with args.out.open("x") as handle:
        json.dump(amended, handle, indent=2)
        handle.write("\n")
    print(json.dumps(dict(amended=amended["amended_gate"], literal=amended["literal_wording_gate"]), indent=2))
