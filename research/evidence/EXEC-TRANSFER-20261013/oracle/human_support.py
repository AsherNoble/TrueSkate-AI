"""Audit support and timing sensitivity on an existing eight-stroke extraction.

These are trail-derived estimates from one demonstration, not touch telemetry.
No sampling, model, decoder or contract is changed by this diagnostic.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def simplify(p, t, tolerance, minimum_ms):
    n = len(t)
    cost = np.full((n, n), np.inf)
    for i in range(n-1):
        for j in range(i+1, n):
            if t[j]-t[i] < minimum_ms-1e-6:
                continue
            u = (t[i:j+1]-t[i])/(t[j]-t[i])
            cost[i, j] = np.linalg.norm(p[i:j+1]-(p[i]+u[:, None]*(p[j]-p[i])), axis=1).max()
    best = np.full((11, n), np.inf)
    best[0, 0] = 0
    for k in range(1, 11):
        for j in range(1, n):
            best[k, j] = np.min(np.maximum(best[k-1, :j], cost[:j, j]))
        if best[k, -1] <= tolerance:
            return dict(segments=k, residual=float(best[k, -1]))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strokes", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    raw = json.loads(args.strokes.read_text())
    intervals = [(r[0]["t"]*1000, r[-1]["t"]*1000) for r in raw.values()]
    results = {}
    for k, (name, rows) in enumerate(raw.items()):
        unique = {r["t"]: (r["x"], r["y"]) for r in rows}
        keys = sorted(unique)
        t = np.array(keys)*1000
        p = np.array([unique[v] for v in keys])
        t -= t[0]
        dt = np.diff(t)
        norm_speed = np.linalg.norm(np.diff(p, axis=0), axis=1)/dt*1000
        logical_speed = np.linalg.norm(np.diff(p, axis=0)*[414, 896], axis=1)/dt*1000
        a, b = intervals[k]
        # 120 frames span119 intervals. Require a non-negative video start;
        # source end is unknown here, so report a necessary feasible range only.
        before = intervals[k-1][1] if k else 0.
        after = intervals[k+1][0] if k+1 < len(intervals) else float("inf")
        span, margin = 119*1000/60, 5*1000/60
        low = max(0., b+margin-span, before+1e-6 if k else 0.)
        high = min(a-margin, after-span-1e-6)
        fits = {f"tol{tol}_min{minimum}": simplify(p, t, tol, minimum)
                for tol in (.01, .03) for minimum in (0., 16., 33., 100.)}
        results[name] = dict(
            duration_ms=float(t[-1]), distinct_samples=len(t),
            within_current_duration=bool(300 <= t[-1] <= 1200),
            median_speed_pt_s=float(np.median(logical_speed)),
            peak_speed_pt_s=float(logical_speed.max()),
            median_local_003_timing_budget_ms=float(np.median(.03/norm_speed*1000)),
            tightest_local_003_timing_budget_ms=float(np.min(.03/norm_speed*1000)),
            segments=fits,
            isolated_window_start_range_ms=[low, high] if low <= high else None,
            isolated_window_caveat="Source end must still be verified; negative starts forbidden.",
        )
    out = dict(source_sha256=hashlib.sha256(args.strokes.read_bytes()).hexdigest(),
               normalized_tolerance_logical_semiaxes_pt=[.03*414, .03*896],
               input_pixel_logical_pt=[414/64, 896/144],
               heatmap_grid_normalized_step=[1/31, 1/71],
               interpretation="Durations are trail-derived visible moving spans, not independently measured contact durations; endpoint holds are unknown. Window support is conditional on the extracted event partition. Local timing budgets assume exact path/speed, not empirical tolerance or error estimates.",
               samples=results)
    args.out.write_text(json.dumps(out, indent=2)+"\n")
    for name, row in results.items():
        print(name, round(row["duration_ms"], 1), "ms; tightest .03 phase budget",
              round(row["tightest_local_003_timing_budget_ms"], 2), "ms; .03 minimum33",
              row["segments"]["tol0.03_min33.0"], "; isolated", row["isolated_window_start_range_ms"])


if __name__ == "__main__":
    main()
