"""Research-only decoder-floor diagnostic. Never imports training or touches the rig.

The reviewed audit is reused only for representability diagnosis. Synthetic
cases are deterministic, independent inputs; no holdout is treated as a test.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import gzip
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
from scipy.optimize import least_squares
import scipy


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def poly_predict(pred, times):
    p = np.asarray(pred["points"])
    return np.stack([np.interp(times-pred["onset_s"],
                               np.asarray(pred["times_ms"])/1000, p[:, a])
                     for a in (0, 1)], axis=-1)


def quantized(pred):
    if pred is None:
        return None
    out = dict(pred)
    integers = np.trunc(np.asarray(pred["points"])*[414, 896]).astype(int)
    out["points"] = (integers/[414, 896]).tolist()
    out["executor_integer_points"] = integers.tolist()
    return out


def refit_grid_coordinates(xy, times, a, b, pred):
    """Only refit knot coordinates; keep baseline's frame-locked knot times."""
    knot_t = np.asarray(pred["times_ms"])
    rel = (times[a:b+1]-times[a])*1000
    # Same target endpoint convention as baseline. Rounded end time otherwise
    # changes the final observation by <.5 ms.
    rel[-1] = knot_t[-1]
    n = len(knot_t)
    basis = np.stack([np.interp(rel, knot_t, np.eye(n)[:, j]) for j in range(n)], axis=-1)
    endpoints = xy[[a,b]]
    rhs = xy[a:b+1] - basis[:, [0]]*endpoints[0] - basis[:, [-1]]*endpoints[1]
    inner = np.linalg.lstsq(basis[:, 1:-1], rhs, rcond=None)[0]
    p = np.vstack((endpoints[0], np.clip(inner, 0., 1.), endpoints[1]))
    residual = float(np.linalg.norm(basis@p-xy[a:b+1], axis=1).max())
    return dict(pred, points=p.tolist(), fit_residual=residual, fit_met_tolerance=residual <= .01)


def compatibility(pred, segment_is_safe):
    if pred is None:
        return None
    t, p = np.asarray(pred["times_ms"]), np.asarray(pred["points"])
    q = (np.asarray(pred["executor_integer_points"])/[414,896] if "executor_integer_points" in pred
         else np.trunc(p*[414,896])/[414,896])
    return dict(integer_ms=bool(np.all(t == np.rint(t))),
                duration_300_1200=bool(300 <= t[-1] <= 1200),
                min_interval_100=bool(np.diff(t).min() >= 100-1e-7),
                segment_cap_10=bool(1 <= len(t)-1 <= 10),
                normalized_bounds=bool(np.all((p >= 0) & (p <= 1))),
                quantized_segments_safe=all(segment_is_safe(a,b) for a,b in zip(q,q[1:])),
                minimum_interval_ms=float(np.diff(t).min()))


def interval_boundary_decode(xy, times, a, b, initial, *, max_nfev=180):
    """Unprivileged research fit: nearest-frame class means a timing interval.

    Optimize boundaries within their Voronoi intervals and internal free knots.
    Unlike privileged boundary ablations, ALL samples, including endpoint frame
    overwrites, are scored at their actual native times in the objective.
    Every segment stays >=100 ms and duration300–1200 ms. Boundaries are not
    identifiable in all cases; this is a feasibility probe, not measured timing.
    """
    lo_a, hi_a = (times[a-1]+times[a])/2, (times[a]+times[a+1])/2
    lo_b, hi_b = (times[b-1]+times[b])/2, (times[b]+times[b+1])/2
    obs, native = xy[a:b+1], times[a:b+1]
    cap = min(10, int(np.floor((hi_b-lo_a+1e-10)/.1)))
    best = initial
    for k in range(min(initial["segment_count"],cap),cap+1):
        min_duration = max(.3,k*.1)
        max_a = min(hi_a,hi_b-min_duration)
        if max_a < lo_a:
            continue
        for phase_start in (.25,.75):
            knots0 = np.linspace(0, initial["times_ms"][-1], k+1)
            pts0 = np.stack([np.interp(knots0,(native-native[0])*1000,obs[:,q]) for q in (0,1)],axis=-1)
            theta = np.r_[phase_start,phase_start,np.zeros(k-1),pts0[1:-1].ravel()]

            def unpack(v):
                down = lo_a+v[0]*(max_a-lo_a)
                min_b, max_b = max(lo_b,down+min_duration),min(hi_b,down+1.2)
                up = min_b+v[1]*(max_b-min_b)
                duration = (up-down)*1000
                logits = np.r_[v[2:2+k-1],0.]
                weights = np.exp(logits-logits.max())
                durations = 100+(duration-100*k)*weights/weights.sum()
                knots = np.r_[0.,np.cumsum(durations)]
                pts = np.vstack((obs[0],v[2+k-1:].reshape(-1,2),obs[-1]))
                return down,up,knots,pts

            def residuals(v):
                down,up,knots,pts = unpack(v)
                estimate = np.stack([np.interp((native-down)*1000,knots,pts[:,q]) for q in (0,1)],axis=-1)
                return (estimate-obs).ravel()

            bounds=(np.r_[0.,0.,np.full(k-1,-20.),np.zeros(2*(k-1))],
                    np.r_[1.,1.,np.full(k-1,20.),np.ones(2*(k-1))])
            fit=least_squares(residuals,np.clip(theta,*bounds),bounds=bounds,max_nfev=max_nfev,
                              ftol=1e-9,xtol=1e-9,gtol=1e-9)
            down,up,knots,pts=unpack(fit.x)
            residual=float(np.linalg.norm(residuals(fit.x).reshape(-1,2),axis=1).max())
            pred=dict(onset_frame=a,release_frame=b,onset_s=float(down),release_s=float(up),
                      points=pts.tolist(),times_ms=knots.tolist(),segment_count=k,
                      fit_residual=residual,fit_met_tolerance=residual<=.01,
                      optimizer_nfev=fit.nfev,optimizer_status=fit.status)
            if residual < best["fit_residual"]:
                best=pred
        if best["fit_residual"]<=.01:
            break
    return best


def grid_decode(xy, times, a, b, *, minimum=100., integer=True,
                interval_rate=1., endpoint_times=None):
    """Same minimax objective as frozen decoder, explicit independent ablations.

    endpoint_times is privileged exact timing, used ONLY to locate floor sources.
    The nearest boundary points remain forced endpoint targets for that ablation.
    """
    points = xy[a:b+1].copy()
    span = times[a:b+1].copy()
    if endpoint_times is not None:
        span[0], span[-1] = endpoint_times
    relative = (span-span[0])*1000
    if integer:
        relative = np.rint(relative)
    n = len(span)
    cost = np.full((n, n), np.inf)
    for i in range(n-1):
        for j in range(i+1, n):
            if relative[j]-relative[i] < minimum*interval_rate-1e-8:
                continue
            u = (span[i:j+1]-span[i])/(span[j]-span[i])
            line = points[i]+u[:, None]*(points[j]-points[i])
            cost[i, j] = np.linalg.norm(points[i:j+1]-line, axis=1).max()
    limit = min(10, int(np.floor((relative[-1]+1e-8)/(minimum*interval_rate))))
    best, previous = np.full((limit+1, n), np.inf), np.full((limit+1, n), -1, int)
    best[0, 0] = 0
    chosen = None
    for k in range(1, limit+1):
        for j in range(1, n):
            values = np.maximum(best[k-1, :j], cost[:j, j])
            i = int(np.argmin(values))
            best[k, j], previous[k, j] = values[i], i
        if best[k, -1] <= .01:
            chosen = k
            break
    if chosen is None:
        chosen = int(np.argmin(best[1:, -1]))+1
    residual = float(best[chosen, -1])
    if not np.isfinite(residual):
        return None
    indices, k = [n-1], chosen
    while k:
        indices.append(int(previous[k, indices[-1]]))
        k -= 1
    indices.reverse()
    return dict(onset_frame=a, release_frame=b, onset_s=float(span[0]),
                release_s=float(span[-1]), points=points[indices].tolist(),
                times_ms=relative[indices].tolist(), segment_count=chosen,
                fit_residual=residual, fit_met_tolerance=residual <= .01)


def integer_schedule(t, duration, minimum=100):
    """Round fractional knots while preserving every minimum interval exactly."""
    k = len(t)-1
    result = np.rint(t).astype(int)
    result[0], result[-1] = 0, int(duration)
    for i in range(1, k):
        result[i] = max(result[i], result[i-1]+minimum)
    for i in range(k-1, 0, -1):
        result[i] = min(result[i], result[i+1]-minimum)
    assert np.all(np.diff(result) >= minimum)
    return result


def integer_interval_prediction(pred, native_times):
    """Round command time while keeping release in its classified interval."""
    out = dict(pred)
    b, down = pred["release_frame"], pred["onset_s"]
    low = (native_times[b-1]+native_times[b])/2
    high = (native_times[b]+native_times[b+1])/2
    min_total = max(300,100*pred["segment_count"],int(np.ceil((low-down)*1000+1e-8)))
    max_total = min(1200,int(np.floor((high-down)*1000-1e-8)))
    assert min_total <= max_total
    duration = int(np.clip(np.rint(pred["times_ms"][-1]),min_total,max_total))
    out["times_ms"] = integer_schedule(np.asarray(pred["times_ms"]),duration).tolist()
    out["release_s"] = down+duration/1000
    assert int(np.argmin(abs(native_times-out["release_s"]))) == b
    return out


def fractional_decode(xy, times, a, b, initial, *, endpoint_times=None,
                      minimum=100., round_output=False, max_nfev=180):
    """Unprivileged fitted knots between frames with minimum-duration constraints.

    Fix endpoints to the same observed boundary positions as baseline. The
    duration simplex guarantees >=minimum without relaxed penalties. Try the
    baseline's chosen segment count through the existing ten-segment cap; two
    deterministic starts. Least-squares fitting is a heuristic, NOT an oracle
    global minimax solver. No original command knots/coordinates are read.
    """
    down, up = (times[a], times[b]) if endpoint_times is None else endpoint_times
    duration = (up-down)*1000
    if round_output:
        duration = float(np.rint(duration))
    observed = xy[a:b+1]
    rel = (times[a:b+1]-down)*1000
    # The target contract explicitly pins boundary-frame positions. Interpret
    # these as observations of the boundary knots in this fitted variant.
    rel[0], rel[-1] = 0., duration
    cap = min(10, int(np.floor((duration+1e-8)/minimum)))
    if cap < 1:
        return None
    lower = min(cap, initial["segment_count"])
    best = initial if endpoint_times is None else None
    for k in range(lower, cap+1):
        slack = max(0., duration-k*minimum)
        starts = [np.linspace(0, duration, k+1)]
        if initial["segment_count"] == k:
            candidate = np.asarray(initial["times_ms"], float)
            candidate = candidate/candidate[-1]*duration
            if np.all(np.diff(candidate) >= minimum-1e-6):
                starts.insert(0, candidate)
        for seed_t in starts:
            seed_p = np.stack([np.interp(seed_t, rel, observed[:, axis])
                               for axis in (0, 1)], axis=-1)
            # Positive extra-duration weights. The final weight is fixed at 1
            # to remove the softmax location indeterminacy.
            weights = np.maximum(np.diff(seed_t)-minimum, .001)
            z = np.log(weights[:-1]/weights[-1])
            theta = np.r_[z, seed_p[1:-1].ravel()]

            def unpack(v):
                logits = np.r_[v[:k-1], 0.]
                weights = np.exp(logits-logits.max())
                durations = minimum+slack*weights/weights.sum()
                knots = np.r_[0., np.cumsum(durations)]
                pts = np.vstack((observed[0], v[k-1:].reshape(-1, 2), observed[-1]))
                return knots, pts

            def residuals(v):
                knots, pts = unpack(v)
                estimate = np.stack([np.interp(rel, knots, pts[:, axis])
                                     for axis in (0, 1)], axis=-1)
                return (estimate-observed).ravel()

            if len(theta):
                bounds = (np.r_[np.full(k-1, -20.), np.zeros(2*(k-1))],
                          np.r_[np.full(k-1, 20.), np.ones(2*(k-1))])
                fit = least_squares(residuals, np.clip(theta, *bounds), bounds=bounds,
                                    max_nfev=max_nfev, ftol=1e-9, xtol=1e-9, gtol=1e-9)
                knot_t, knot_p = unpack(fit.x)
                nfev, status = fit.nfev, fit.status
            else:
                knot_t, knot_p = unpack(theta)
                nfev, status = 0, 1
            if round_output:
                knot_t = integer_schedule(knot_t, duration, int(minimum))
            estimate = np.stack([np.interp(rel, knot_t, knot_p[:, axis]) for axis in (0, 1)], axis=-1)
            residual = float(np.linalg.norm(estimate-observed, axis=1).max())
            pred = dict(onset_frame=a, release_frame=b, onset_s=float(down),
                        release_s=float(down+duration/1000 if round_output else up), points=knot_p.tolist(), times_ms=knot_t.tolist(),
                        segment_count=k, fit_residual=residual,
                        fit_met_tolerance=residual <= .01, optimizer_nfev=nfev,
                        optimizer_status=status)
            if best is None or residual < best["fit_residual"]:
                best = pred
        if best["fit_residual"] <= .01:
            break
    return best


def synthetic_cases(Gesture):
    """Independent factorial cases: phase, fast movement, kink timing, holds/rate.

    All actual commands meet frozen 100 ms/300–1200 ms restrictions. The cases
    are not samplings from the audit, nor meaningful prevalence estimates.
    """
    cases = []
    for shape in ("line", "kink", "zigzag", "hold"):
        for segment_ms in (100, 103, 117, 151):
            for amplitude in (.08, .25, .48):
                for phase in (0., 2., 5., 8., 11., 14.):
                    for rate in (.999, 1., 1.001):
                        x0, x1, yc = .25, .25+amplitude, .65
                        if shape == "line":
                            p, t = [(x0,yc),(x1,yc)], [0, max(300,3*segment_ms)]
                        elif shape == "kink":
                            p = [(x0,yc),(x1,yc+.08),(x0,yc+.16)]
                            t = [0, segment_ms, max(300, 3*segment_ms)]
                        elif shape == "zigzag":
                            p = [(x0,yc),(x1,yc+.03),(x0,yc+.06),(x1,yc+.09)]
                            t = [0, segment_ms, 2*segment_ms, 3*segment_ms]
                        else:
                            p = [(x0,yc),(x1,yc+.03),(x1,yc+.03),(x0,yc+.06)]
                            t = [0, segment_ms, 2*segment_ms, 3*segment_ms]
                        g = Gesture(tuple(p), tuple(t), "random_chain")
                        onset = .3+phase/1000
                        meta = dict(frame_times_s=(np.arange(120)/60).tolist(),
                                    onset_s=onset, release_s=onset+rate*t[-1]/1000,
                                    gesture=g.to_dict(), clock_rate=rate)
                        ident = f"{shape}-s{segment_ms}-a{amplitude}-p{phase}-r{rate}"
                        cases.append((ident, meta, dict(shape=shape, segment_ms=segment_ms,
                                                       amplitude=amplitude, phase_ms=phase, rate=rate)))
    return cases


def summary(rows):
    result = {}
    for variant in sorted({r["variant"] for r in rows}):
        rr = [r for r in rows if r["variant"] == variant]
        finite = [r for r in rr if r.get("prediction") is not None]
        result[variant] = dict(n=len(rr), feasible=len(finite),
                               passed=sum(r.get("passed", False) for r in rr),
                               fit_001=sum(r["prediction"].get("fit_met_tolerance", False) for r in finite),
                               max_fit_residual=max((r["prediction"]["fit_residual"] for r in finite), default=None),
                               failed_ids=[r["id"] for r in rr if not r.get("passed", False)],
                               median_good_contact_fraction=float(np.median([r["good_contact_fraction"] for r in finite])) if finite else None)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--mirror", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--synthetic-stride", type=int, default=1,
                    help="Fit expensive fractional variants to deterministic every nth synthetic case")
    args = ap.parse_args()
    sys.path.insert(0, str(args.source / "src"))
    from trueskate_ai.model1.curved.contract import Gesture
    from trueskate_ai.model1.curved.dataset import targets
    from trueskate_ai.model1.curved.decoder import decode, evaluate_path
    from trueskate_ai.data.control_hitboxes import segment_is_safe
    start = time.monotonic()
    manifest_path = args.mirror / "audit-6/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    audit = [(s["id"], json.loads((args.mirror/s["sample_dir"]/"meta.json").read_text()), {})
             for s in manifest["samples"]]
    all_rows = {}
    endpoint_diagnostics = []
    timing_records, repeat_checks = [], []
    for cohort, cases in (("audit", audit), ("synthetic", synthetic_cases(Gesture))):
        rows = []
        for ci, (ident, meta, factors) in enumerate(cases):
            g = Gesture.from_dict(meta["gesture"])
            times = np.asarray(meta["frame_times_s"])
            label = targets(meta)
            xy = np.asarray(label["xy"], float)
            a, b = int(label["onset"]), int(label["release"])
            onset, release = np.full(120, -1e6), np.full(120, -1e6)
            onset[a] = release[b] = 0.
            tick = time.perf_counter()
            baseline = decode(xy, onset, release, times)
            baseline_seconds = time.perf_counter()-tick
            independent = grid_decode(xy, times, a, b)
            assert np.allclose(independent["points"], baseline["points"])
            assert independent["times_ms"] == baseline["times_ms"]
            assert abs(independent["fit_residual"]-baseline["fit_residual"]) < 1e-12
            variants = {"baseline": baseline}
            tick = time.perf_counter()
            variants["refit_grid_coords_100"] = refit_grid_coordinates(xy, times, a, b, baseline)
            coordinate_seconds = time.perf_counter()-tick
            variants["refit_grid_coords100_quantized"] = quantized(variants["refit_grid_coords_100"])
            variants["fractional_grid_100"] = grid_decode(xy, times, a, b, integer=False)
            variants["rounded_grid_rate100"] = grid_decode(xy, times, a, b, interval_rate=meta["clock_rate"])
            variants["fractional_grid_rate100"] = grid_decode(xy, times, a, b, integer=False, interval_rate=meta["clock_rate"])
            variants["exact_boundary_grid_100_ORACLE"] = grid_decode(xy, times, a, b, integer=False, endpoint_times=(meta["onset_s"], meta["release_s"]))
            for floor in (99, 95, 83, 66, 33):
                variants[f"grid_min{floor}_INCOMPATIBLE"] = grid_decode(xy, times, a, b, minimum=floor)
            truth_pred = dict(onset_frame=a, release_frame=b,
                              onset_s=meta["onset_s"], release_s=meta["release_s"],
                              points=g.command_points().tolist(),
                              times_ms=(np.asarray(g.times_ms)*meta["clock_rate"]).tolist(),
                              segment_count=len(g.times_ms)-1, fit_residual=0., fit_met_tolerance=True)
            residual = np.linalg.norm(poly_predict(truth_pred, times[a:b+1])-xy[a:b+1], axis=1)
            truth_pred["fit_residual"] = float(residual.max())
            truth_pred["fit_met_tolerance"] = bool(residual.max() <= .01)
            variants["exact_command_ORACLE"] = truth_pred
            if cohort == "audit" or ci % args.synthetic_stride == 0:
                tick = time.perf_counter()
                variants["fitted_fractional_100"] = fractional_decode(xy, times, a, b, baseline)
                fractional_seconds = time.perf_counter()-tick
                tick = time.perf_counter()
                variants["fitted_integer_100"] = fractional_decode(xy, times, a, b, baseline, round_output=True)
                integer_seconds = time.perf_counter()-tick
                variants["fitted_integer100_quantized"] = quantized(variants["fitted_integer_100"])
                tick = time.perf_counter()
                variants["interval_boundaries_fitted100"] = interval_boundary_decode(xy, times, a, b, baseline)
                interval_seconds = time.perf_counter()-tick
                interval_integer = integer_interval_prediction(variants["interval_boundaries_fitted100"],times)
                variants["interval_boundaries_integer100"] = interval_integer
                variants["interval_boundaries_integer100_quantized"] = quantized(interval_integer)
                variants["exact_boundary_fitted100_ORACLE"] = fractional_decode(xy, times, a, b, baseline, endpoint_times=(meta["onset_s"], meta["release_s"]))
                variants["exact_boundary_rate_fitted100_ORACLE"] = fractional_decode(xy, times, a, b, baseline, endpoint_times=(meta["onset_s"], meta["release_s"]), minimum=100*meta["clock_rate"])
                timing_records.append(dict(cohort=cohort, id=ident, baseline_seconds=baseline_seconds,
                                           coordinate_seconds=coordinate_seconds,
                                           fractional_seconds=fractional_seconds, integer_seconds=integer_seconds,
                                           interval_seconds=interval_seconds))
                if cohort == "audit" and (ci % 10 == 0 or ident in (51,77,83,113,133)):
                    repeated = fractional_decode(xy.copy(), times.copy(), a, b, baseline, round_output=True)
                    repeat_checks.append(dict(id=ident, exact_equal=repeated == variants["fitted_integer_100"]))
            if cohort == "audit":
                raw = g.positions(times-meta["onset_s"], rate=meta["clock_rate"])
                endpoint_diagnostics.append(dict(id=ident, clock_rate=meta["clock_rate"],
                                                 command_min_interval_ms=int(np.diff(g.times_ms).min()),
                                                 onset_phase_ms=1000*(meta["onset_s"]-times[a]),
                                                 release_phase_ms=1000*(meta["release_s"]-times[b]),
                                                 onset_target_override=float(np.linalg.norm(xy[a]-raw[a])),
                                                 release_target_override=float(np.linalg.norm(xy[b]-raw[b]))))
            for name, pred in variants.items():
                if pred is not None and "quantized" in name:
                    residual = float(np.linalg.norm(poly_predict(pred, times[a:b+1])-xy[a:b+1], axis=1).max())
                    pred["fit_residual"], pred["fit_met_tolerance"] = residual, residual <= .01
                record = dict(id=ident, variant=name, prediction=pred, factors=factors)
                if pred is not None:
                    record.update(evaluate_path(pred, meta))
                    record["compatibility"] = compatibility(pred, segment_is_safe)
                    if name in ("refit_grid_coords100_quantized", "fitted_integer100_quantized", "interval_boundaries_integer100_quantized"):
                        source_pred = variants[dict(refit_grid_coords100_quantized="refit_grid_coords_100",
                                                    fitted_integer100_quantized="fitted_integer_100",
                                                    interval_boundaries_integer100_quantized="interval_boundaries_integer100")[name]]
                        try:
                            command = Gesture(tuple(tuple(p) for p in source_pred["points"]),
                                              tuple(int(t) for t in source_pred["times_ms"]), "random_chain")
                            payload = command.payload()
                            assert command.command_points().tolist() == pred["points"]
                            record["compiled_gesture_payload_sha256"] = hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
                            record["compiled_gesture_valid"] = True
                        except ValueError as exc:
                            record["compiled_gesture_valid"] = False
                            record["compile_error"] = str(exc)
                rows.append(record)
            if ci % 100 == 0:
                print(cohort, ci, "/", len(cases), "elapsed", round(time.monotonic()-start), flush=True)
        all_rows[cohort] = rows
        args.out.mkdir(parents=True, exist_ok=True)
        result_bytes=(json.dumps(dict(summary=summary(rows), samples=rows), separators=(",", ":"))+"\n").encode()
        (args.out/f"{cohort}-results.json.gz").write_bytes(gzip.compress(result_bytes,mtime=0))
        (args.out/f"{cohort}-summary.json").write_text(json.dumps(summary(rows),indent=2)+"\n")
        print(cohort, {k: (v["passed"],v["n"],v["fit_001"]) for k,v in summary(rows).items()}, flush=True)
    provenance = dict(source_path=str(args.source), mirror_path=str(args.mirror),
                      source_sha=subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip(),
                      manifest_sha256=digest(manifest_path), script_sha256=digest(__file__),
                      implementation={n: digest(args.source/"src/trueskate_ai/model1/curved"/n) for n in ("contract.py", "dataset.py", "decoder.py")},
                      audit_meta_sha256={str(s["id"]): digest(args.mirror/s["sample_dir"]/"meta.json") for s in manifest["samples"]},
                      python=sys.version, numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform(),
                      elapsed_s=time.monotonic()-start, argv=sys.argv,
                      notes=["Audit reuse is diagnostic, not a refactor test, training split or certification.",
                             "All evaluation uses the unchanged frozen .03 / 95% and +/-2 boundary-frame metric.",
                             "Every _ORACLE variant reads privileged command/timing metadata.",
                             "Every _INCOMPATIBLE floor is outside the >=100 ms curved-v1 contract.",
                             "Fractional fitting reads only frame xy, native times and selected boundaries; no command knots.",
                             "Fractional knot fitting is local least-squares, not guaranteed minimax/global optimization."])
    (args.out/"provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    (args.out/"endpoint-diagnostics.json").write_text(json.dumps(endpoint_diagnostics, indent=2)+"\n")
    (args.out/"timings-and-repeats.json").write_text(json.dumps(dict(timings=timing_records, deterministic_repeats=repeat_checks), indent=2)+"\n")


if __name__ == "__main__":
    main()
