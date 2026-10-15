"""Deterministic free-breakpoint minimax polyline decoding on the native timeline."""
from __future__ import annotations

import numpy as np


def decode(xy, onset_logits, release_logits, times, *, tolerance=.01):
    xy, times = np.asarray(xy, float), np.asarray(times, float)
    onset_logits, release_logits = np.asarray(onset_logits), np.asarray(release_logits)
    if (xy.shape != (len(times), 2) or len(times) != 120 or
            onset_logits.shape != times.shape or release_logits.shape != times.shape
            or not all(np.isfinite(a).all() for a in (xy, times, onset_logits, release_logits))
            or np.any(np.diff(times) <= 0)):
        raise ValueError("invalid decoder inputs")
    elapsed_ms = np.rint((times[None, :]-times[:, None])*1000).astype(int)
    valid = (elapsed_ms >= 300) & (elapsed_ms <= 1200)
    scores = onset_logits[:, None]+release_logits[None, :]
    scores = np.where(valid, scores, -np.inf)
    down, up = np.unravel_index(int(np.argmax(scores)), scores.shape)
    span = times[down:up+1]
    points = xy[down:up+1]
    relative_ms = np.rint((span-span[0])*1000).astype(int)
    n = len(span)
    cost = np.full((n, n), np.inf)
    for i in range(n-1):
        for j in range(i+1, n):
            if relative_ms[j]-relative_ms[i] < 100:
                continue
            u = (span[i:j+1]-span[i])/(span[j]-span[i])
            line = points[i]+u[:, None]*(points[j]-points[i])
            cost[i, j] = np.linalg.norm(points[i:j+1]-line, axis=1).max()
    limit = min(10, relative_ms[-1]//100)
    best = np.full((limit+1, n), np.inf)
    previous = np.full((limit+1, n), -1, dtype=int)
    best[0, 0] = 0
    chosen = None
    for k in range(1, limit+1):
        for j in range(1, n):
            values = np.maximum(best[k-1, :j], cost[:j, j])
            i = int(np.argmin(values))
            best[k, j], previous[k, j] = values[i], i
        if best[k, -1] <= tolerance:
            chosen = k
            break
    if chosen is None:
        chosen = int(np.argmin(best[1:, -1]))+1
    residual = float(best[chosen, -1])
    if not np.isfinite(residual):
        raise ValueError("no feasible decoded path")
    indices = [n-1]
    k = chosen
    while k:
        indices.append(int(previous[k, indices[-1]]))
        k -= 1
    indices.reverse()
    return dict(onset_frame=int(down), release_frame=int(up),
                onset_s=float(times[down]), release_s=float(times[up]),
                points=points[indices].tolist(), times_ms=relative_ms[indices].tolist(),
                segment_count=chosen, fit_residual=residual, fit_met_tolerance=residual <= tolerance)


def evaluate_path(pred, meta):
    from .contract import Gesture
    times = np.asarray(meta["frame_times_s"])
    target = Gesture.from_dict(meta["gesture"])
    a = int(np.argmin(abs(times-meta["onset_s"])))
    b = int(np.argmin(abs(times-meta["release_s"])))
    contact = np.arange(a, b+1)
    truth = target.positions(times[contact]-meta["onset_s"], rate=meta["clock_rate"])
    truth[0], truth[-1] = target.command_points()[[0, -1]]
    predicted = np.stack([np.interp(times[contact]-pred["onset_s"],
                                    np.asarray(pred["times_ms"])/1000,
                                    np.asarray(pred["points"])[:, axis]) for axis in (0, 1)], axis=-1)
    error = np.linalg.norm(predicted-truth, axis=1)
    logical = np.linalg.norm((predicted-truth)*[414, 896], axis=1)
    missing = (contact < pred["onset_frame"]) | (contact > pred["release_frame"])
    good = (error <= .03) & ~missing
    fraction = float(good.mean())
    boundaries = abs(pred["onset_frame"]-a) <= 2 and abs(pred["release_frame"]-b) <= 2
    return dict(passed=bool(boundaries and fraction >= .95), good_contact_fraction=fraction,
                onset_error_frames=abs(pred["onset_frame"]-a), release_error_frames=abs(pred["release_frame"]-b),
                onset_error_ms=1000*abs(pred["onset_s"]-meta["onset_s"]),
                release_error_ms=1000*abs(pred["release_s"]-meta["release_s"]),
                mean_path_error=float(np.mean(np.where(missing, np.sqrt(2), error))),
                path_error_p95=float(np.quantile(np.where(missing, np.sqrt(2), error), .95)),
                mean_logical_error_pt=float(np.mean(np.where(missing, np.hypot(414, 896), logical))),
                logical_error_p95_pt=float(np.quantile(np.where(missing, np.hypot(414, 896), logical), .95)),
                missing_contact_frames=int(missing.sum()), fit_residual=pred["fit_residual"])
