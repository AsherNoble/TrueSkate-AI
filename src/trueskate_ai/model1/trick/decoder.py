"""Gestures from per-frame signals, straight-segment fits, and scoring against the trail labels."""
from __future__ import annotations

import numpy as np

from .contract import MAX_SEGMENTS, MAX_GESTURE_MS, MAX_GESTURES, MIN_SEGMENT_MS, MIN_GESTURE_MS

LOGIT_LIMIT = 9.  # probabilities kept about 1e-4 away from 0 and 1
FIT_TOLERANCE = .01  # curved v1's straight-segment tolerance, normalized screen units
TIMING_TOLERANCE_S = 2/60  # ±2 frames at each end
# Kickflip position-tolerance levels (KICKFLIP-TOLERANCE-20261015): every point moved this far,
# Euclidean in normalized screen coordinates (x/414, y/896).
LEVELS = (.02, .05)
MATCH_WINDOW_S = .25
FIT_STEP_MS = 5.
PROGRESS = np.linspace(0., 1., 41)


def gesture_spans(contact, down, up, times, *, max_gestures=MAX_GESTURES) -> list[tuple[int, int]]:
    """Best non-overlapping gestures [a, b] for per-frame logits.

    A gesture scores its frames' contact log-odds plus the touch-down log-odds at a and the
    lift log-odds at b; no gesture scores 0. Each gesture lasts 50–600 ms, give or take a
    frame, and the next starts after it ends.
    """
    contact, down, up, times = (np.asarray(v, dtype=float) for v in (contact, down, up, times))
    n = len(times)
    if (n < 2 or any(v.shape != (n,) for v in (contact, down, up))
            or not all(np.isfinite(v).all() for v in (contact, down, up, times)) or np.any(np.diff(times) <= 0)):
        raise ValueError("invalid decoder inputs")
    lc, ld, lu = (np.clip(v, -LOGIT_LIMIT, LOGIT_LIMIT) for v in (contact, down, up))
    prefix = np.concatenate([[0.], np.cumsum(lc)])
    ms = times*1000
    slack = float(np.median(np.diff(ms))) + 1.  # one frame, plus jitter in native frame times
    starts = [(int(np.searchsorted(ms, ms[b] - MAX_GESTURE_MS - slack, side="left")),
               int(np.searchsorted(ms, ms[b] - MIN_GESTURE_MS + slack, side="right")) - 1) for b in range(n)]
    # best[k, i]: best score with exactly k gestures inside frames [0, i); back: the last one's start.
    best = np.full((max_gestures + 1, n + 1), -np.inf)
    best[0] = 0.
    back = np.full((max_gestures + 1, n + 1), -1, dtype=int)
    for k in range(1, max_gestures + 1):
        for i in range(1, n + 1):
            b = i - 1
            best[k, i] = best[k, i - 1]
            lo, hi = starts[b]
            hi = min(hi, b - 1)
            if hi < lo:
                continue
            a = np.arange(lo, hi + 1)
            scores = best[k - 1, a] + prefix[b + 1] - prefix[a] + ld[a] + lu[b]
            j = int(np.argmax(scores))
            if scores[j] > best[k, i]:
                best[k, i], back[k, i] = scores[j], a[j]
    k, i, spans = int(np.argmax(best[:, n])), n, []
    while k > 0:
        if back[k, i] < 0:
            i -= 1
            continue
        spans.append((int(back[k, i]), i - 1))
        i, k = int(back[k, i]), k - 1
    return spans[::-1]


def fit_segments(times, points, *, tolerance=FIT_TOLERANCE):
    """Curved v1's free-breakpoint minimax polyline, set to 1–4 segments of at least 40 ms.

    Returns the chosen frame indices, the frames' ms from the first, and the residual.
    A gesture shorter than one segment is one segment.
    """
    times, points = np.asarray(times, dtype=float), np.asarray(points, dtype=float)
    relative_ms = np.rint((times - times[0])*1000).astype(int)
    n = len(times)
    shortest = min(MIN_SEGMENT_MS, relative_ms[-1])
    cost = np.full((n, n), np.inf)
    for i in range(n - 1):
        for j in range(i + 1, n):
            if relative_ms[j] - relative_ms[i] < shortest:
                continue
            u = (times[i:j+1] - times[i])/(times[j] - times[i])
            line = points[i] + u[:, None]*(points[j] - points[i])
            cost[i, j] = np.linalg.norm(points[i:j+1] - line, axis=1).max()
    limit = max(1, min(MAX_SEGMENTS, relative_ms[-1]//MIN_SEGMENT_MS))
    best = np.full((limit + 1, n), np.inf)
    previous = np.full((limit + 1, n), -1, dtype=int)
    best[0, 0] = 0
    chosen = None
    for k in range(1, limit + 1):
        for j in range(1, n):
            values = np.maximum(best[k - 1, :j], cost[:j, j])
            i = int(np.argmin(values))
            best[k, j], previous[k, j] = values[i], i
        if best[k, -1] <= tolerance:
            chosen = k
            break
    if chosen is None:
        chosen = int(np.argmin(best[1:, -1])) + 1
    residual = float(best[chosen, -1])
    if not np.isfinite(residual):
        raise ValueError("no feasible straight-segment fit")
    indices = [n - 1]
    for k in range(chosen, 0, -1):
        indices.append(int(previous[k, indices[-1]]))
    return indices[::-1], relative_ms, residual


def refine(times, logits, i: int) -> float:
    """Sub-frame time of an event at frame i: the vertex of a parabola through log-probabilities."""
    times = np.asarray(times, dtype=float)
    if not 0 < i < len(times) - 1:
        return float(times[i])
    y = -np.logaddexp(0., -np.clip(np.asarray(logits[i-1:i+2], dtype=float), -LOGIT_LIMIT, LOGIT_LIMIT))
    curvature = y[0] - 2*y[1] + y[2]
    if curvature >= 0:
        return float(times[i])
    offset = float(np.clip(.5*(y[0] - y[2])/curvature, -.5, .5))
    step = times[i+1] - times[i] if offset > 0 else times[i] - times[i-1]
    return float(times[i] + offset*step)


def sample(times, points, at) -> np.ndarray:
    """Positions at times `at`: linear between frames, extended along the end segments' motion."""
    times, points, at = (np.asarray(v, dtype=float) for v in (times, points, at))
    out = np.stack([np.interp(at, times, points[:, axis]) for axis in (0, 1)], axis=-1)
    if len(times) > 1:
        before, after = at < times[0], at > times[-1]
        out[before] = points[0] + (at[before] - times[0])[:, None]*(points[1] - points[0])/(times[1] - times[0])
        out[after] = points[-1] + (at[after] - times[-1])[:, None]*(points[-1] - points[-2])/(times[-1] - times[-2])
    return out


def decode(xy, contact, down, up, times) -> list[dict]:
    """Timed gestures: touch-down and lift in clip seconds, points and ms since touch-down.

    Saved results keep model1_trick_v1's key names: onset_* is touch-down and release_* is
    lift. Each gesture runs between its sub-frame touch-down and lift times. Its frames'
    positions are resampled every FIT_STEP_MS, so a segment boundary (a bend) can fall
    between frames.
    """
    xy, times = np.asarray(xy, dtype=float), np.asarray(times, dtype=float)
    if xy.shape != (len(times), 2) or not np.isfinite(xy).all():
        raise ValueError("invalid decoder inputs")
    gestures = []
    for a, b in gesture_spans(contact, down, up, times):
        onset, release = refine(times, down, a), refine(times, up, b)
        if release <= onset:
            onset, release = float(times[a]), float(times[b])
        grid = np.linspace(onset, release, max(2, round((release - onset)*1000/FIT_STEP_MS) + 1))
        dense = sample(times[a:b+1], xy[a:b+1], grid)
        indices, relative_ms, residual = fit_segments(grid, dense)
        gestures.append(dict(onset_frame=a, release_frame=b, onset_s=onset, release_s=release,
                            points=dense[indices].tolist(), times_ms=relative_ms[indices].tolist(),
                            pieces=len(indices) - 1, fit_residual=residual,
                            contact=float((1/(1 + np.exp(-np.asarray(contact[a:b+1], dtype=float)))).mean())))
    return gestures


def path_at(times_ms, points, progress=PROGRESS) -> np.ndarray:
    """Positions at fractions of a gesture's duration."""
    times_ms, points = np.asarray(times_ms, dtype=float), np.asarray(points, dtype=float)
    at = np.asarray(progress)*times_ms[-1]
    return np.stack([np.interp(at, times_ms, points[:, axis]) for axis in (0, 1)], axis=-1)


def match(predicted, labels) -> dict[int, int]:
    """Label index → predicted index, nearest ends first, both ends within MATCH_WINDOW_S."""
    pairs = sorted((abs(p["onset_s"] - t["onset_s"]) + abs(p["release_s"] - t["release_s"]), j, i)
                   for i, p in enumerate(predicted) for j, t in enumerate(labels)
                   if abs(p["onset_s"] - t["onset_s"]) <= MATCH_WINDOW_S
                   and abs(p["release_s"] - t["release_s"]) <= MATCH_WINDOW_S)
    result, used = {}, set()
    for _, j, i in pairs:
        if j not in result and i not in used:
            result[j] = i
            used.add(i)
    return result


def score_clip(predicted, meta) -> dict:
    """Per labelled gesture: timing errors, path error along the gesture, and passes per level.

    Path error compares positions at the same fraction of each gesture's duration, so it
    measures where the finger goes; timing is scored separately at both ends.
    """
    labels = meta["strokes"]
    matched = match(predicted, labels)
    gestures = []
    for j, label in enumerate(labels):
        row = dict(family=label["family"], pieces=len(label["times_ms"]) - 1, duration_ms=label["times_ms"][-1],
                   matched=j in matched, timing_ok=False, **{f"pass_{level:g}": False for level in LEVELS})
        if j in matched:
            guess = predicted[matched[j]]
            onset = 1000*(guess["onset_s"] - label["onset_s"])
            release = 1000*(guess["release_s"] - label["release_s"])
            error = np.linalg.norm(path_at(guess["times_ms"], guess["points"])
                                   - path_at(label["times_ms"], label["command_points"]), axis=1)
            timing = max(abs(onset), abs(release)) <= 1000*TIMING_TOLERANCE_S + 1e-6
            row.update(onset_error_ms=onset, release_error_ms=release, duration_error_ms=release - onset,
                       path_mean=float(error.mean()), path_p95=float(np.quantile(error, .95)),
                       path_max=float(error.max()), start_error=float(error[0]), end_error=float(error[-1]),
                       timing_ok=bool(timing),
                       **{f"pass_{level:g}": bool(timing and error.max() <= level) for level in LEVELS})
        gestures.append(row)
    count_ok = len(predicted) == len(labels)
    return dict(labelled_strokes=len(labels), predicted_strokes=len(predicted), count_ok=count_ok,
                extra_strokes=len(predicted) - len(matched), strokes=gestures,
                **{f"clip_pass_{level:g}": bool(count_ok and all(s[f"pass_{level:g}"] for s in gestures))
                   for level in LEVELS})
