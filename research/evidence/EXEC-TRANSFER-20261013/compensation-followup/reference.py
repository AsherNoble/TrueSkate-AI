"""Offline easing reference and retrospective command-holdout duration check.

No WDA requests, simulator commands, dataset reads or production imports.
The split/candidates are declared in PLAN-FROZEN.md. This is not a new blind
experiment. Duration-model error alone cannot validate trajectory compensation.
"""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
XCT = ROOT.parent / "xctest-transfer"


def load(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def cubic(u, a, b):
    return 3 * (1-u)**2 * u * a + 3 * (1-u) * u**2 * b + u**3


def solve_cubic(value, a, b):
    value = np.asarray(value, dtype=float)
    if np.any((value < 0) | (value > 1)):
        raise ValueError("Normalized progress/time must be in [0,1]")
    lo, hi = np.zeros_like(value), np.ones_like(value)
    for _ in range(55):
        u = (lo+hi)/2
        lower = cubic(u, a, b) < value
        lo, hi = np.where(lower, u, lo), np.where(lower, hi, u)
    return (lo+hi)/2


def ease(phase, cp):
    return cubic(solve_cubic(phase, cp[0], cp[2]), cp[1], cp[3])


def command_phase_for_progress(progress, cp):
    """Inverse local single-leg curve, not a compiled W3C compensation action."""
    return cubic(solve_cubic(progress, cp[1], cp[3]), cp[0], cp[2])


def statistics(values):
    a = np.asarray(values)
    return {"n_commands": len(a), "mae_ms": float(np.mean(np.abs(a))),
            "p95_abs_ms": float(np.percentile(np.abs(a), 95)),
            "max_abs_ms": float(np.max(np.abs(a)))}


def main():
    paths = [XCT / "resume-20261013/summary.json.gz",
             XCT / "resume-20261013/r6-transfer.json.gz",
             XCT / "resume-20261013/recap-control-points.json",
             XCT / "boundary-20261013/summary.json",
             XCT / "boundary-20261013/manifest.json",
             XCT / "scripts/probes.py", ROOT / "PLAN-FROZEN.md"]
    summary, transfer, constants, boundary, manifest = map(load, paths[:5])
    spec = importlib.util.spec_from_file_location("research_probes", paths[5])
    probes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probes)
    records = {}
    for name, ((payload, cmd), kind) in probes.probes().items():
        records[name] = (np.asarray(cmd), summary["by_command"][name]["observed_duration_ms"]["median"], kind)
    for name, value in transfer["commands"].items():
        records[name] = (np.asarray(value["command"]), summary["by_command"][name]["observed_duration_ms"]["median"], "r6")
    for name, value in manifest["probes"].items():
        records[name] = (np.asarray(value["cmd"]), boundary[name]["delivered_median_ms"], "boundary")
    assert len(records) == 200
    development = {"single_100", "single_400", "single_800", "single_1200",
                   "sub_300_n2", "sub_300_n3", "boundary_d17_n2"}
    tick = 1000/60
    x = np.array([[1, len(records[n][0])-1] for n in sorted(development)])
    y = np.array([records[n][1]-records[n][0][-1, 0] for n in sorted(development)])
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    candidates = {"identity": {}, "nearest_total_tick": {},
                  "sum_rounded_segment_ticks": {}, "sum_ceil_segment_ticks": {},
                  "segment_count_linear": {}}
    details = {}
    for name, (cmd, observed, kind) in records.items():
        durations = np.diff(cmd[:, 0]); total = cmd[-1, 0]
        predicted = {"identity": total,
                     "nearest_total_tick": np.floor(total/tick+0.5)*tick,
                     "sum_rounded_segment_ticks": np.sum(np.floor(durations/tick+0.5))*tick,
                     "sum_ceil_segment_ticks": np.sum(np.ceil(durations/tick-1e-10))*tick,
                     "segment_count_linear": total+beta[0]+beta[1]*len(durations)}
        cohort = "development" if name in development else ("round6_holdout" if kind == "r6" else "boundary_holdout" if kind == "boundary" else "main_probe_holdout")
        details[name] = {"cohort": cohort, "command_duration_ms": float(total),
                         "segments": len(durations), "observed_duration_ms": observed,
                         "prediction_error_ms": {k: float(v-observed) for k, v in predicted.items()}}
        for model, value in predicted.items():
            candidates[model].setdefault(cohort, []).append(value-observed)
    cp = constants["touch_control_points"]
    grid = np.linspace(0, 1, 10001)
    inverse = command_phase_for_progress(grid, cp)
    back = ease(inverse, cp)
    gap = ease(grid, cp)-grid
    result = {
        "scope": "Offline retrospective command holdouts; no new execution, no physical transfer claim.",
        "development_commands": sorted(development),
        "duration_error_sign": "predicted minus observed contact duration; command-weighted, five-repeat medians",
        "linear_coefficients_intercept_and_per_segment_ms": beta.tolist(),
        "duration_models": {model: {group: statistics(values) for group, values in groups.items()} for model, groups in candidates.items()},
        "inverse_easing": {"control_points": cp,
                           "forward_inverse_max_abs": float(np.max(np.abs(back-grid))),
                           "strictly_monotonic_on_grid": bool(np.all(np.diff(inverse)>0)),
                           "min_progress_minus_linear": float(gap.min()),
                           "max_progress_minus_linear": float(gap.max()),
                           "limitation": "Mathematical inverse of local single-leg curve; not an executable compensation payload or whole-stream inverse."},
        "command_results": details,
        "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (ROOT / "results.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("command_results", "input_sha256")}, indent=2))


if __name__ == "__main__":
    main()
