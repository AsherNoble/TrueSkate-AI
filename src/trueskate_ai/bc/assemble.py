"""Gesture assembly — per-frame touch track -> stroke recipes (the inverse of self_label).

Model 1 (the trace extractor) emits an instantaneous touch `(active, x, y)` per
frame of a continuous expert clip. To label Asher's play for Model 2 we must turn
that per-frame track back into the discrete *strokes* he made: segment the active
runs, then fit each run to one curved drag — 3 waypoints + duration + easing —
i.e. the inverse of `vision/self_label.label_frames`.

This is the "+ gesture-assembly" half of Asher's "frame sequence -> gesture"
labeler. Pure numpy; validated by a round-trip against `self_label` (synthesize a
known gesture's per-frame touches -> assemble -> recover the gesture).

Output strokes are NATIVE units (see `gesture_tokens.STROKE_DIM`): a stream of
`[x0,y0,x1,y1,x2,y2,duration,easing,delay_before]` plus each stroke's
(t_start, t_end) in clip time, so the sequence dataset can window frames per stroke.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trueskate_ai.bc.gesture_tokens import STROKE_DIM
from trueskate_ai.rl.cmaes.action_param import build_param_bounds

_BOUNDS = build_param_bounds(1, use_spin=False)  # (8,2) one slot
_DUR_LO, _DUR_HI = float(_BOUNDS[6, 0]), float(_BOUNDS[6, 1])      # 0.03, 0.8
_EAS_LO, _EAS_HI = float(_BOUNDS[7, 0]), float(_BOUNDS[7, 1])      # 0.3, 3.0


@dataclass
class Stroke:
    """One assembled stroke in native units + its clip-time span."""
    params: np.ndarray   # (STROKE_DIM,) [x0,y0,x1,y1,x2,y2,dur,easing,delay_before]
    t_start: float
    t_end: float


def _runs(active: np.ndarray, times: np.ndarray, *, max_gap_s: float, min_frames: int) -> list[np.ndarray]:
    """Indices of contiguous active runs, bridging gaps <= max_gap_s, len >= min_frames."""
    idx = np.nonzero(active)[0]
    if idx.size == 0:
        return []
    runs, cur = [], [idx[0]]
    for prev, i in zip(idx[:-1], idx[1:]):
        if (times[i] - times[prev]) <= max_gap_s:
            cur.append(i)
        else:
            runs.append(np.array(cur)); cur = [i]
    runs.append(np.array(cur))
    return [r for r in runs if r.size >= min_frames]


def _fit_easing(s: np.ndarray, tau: np.ndarray) -> float:
    """Recover easing_power p from progress s ≈ tau**p (position_progress = τ^p)."""
    m = (tau > 1e-3) & (tau < 1.0 - 1e-3) & (s > 1e-3)
    if m.sum() < 3:
        return 1.0
    slope = np.polyfit(np.log(tau[m]), np.log(s[m]), 1)[0]
    return float(np.clip(slope, _EAS_LO, _EAS_HI))


def fit_stroke(t: np.ndarray, x: np.ndarray, y: np.ndarray, delay_before: float) -> np.ndarray:
    """Fit a dense (t,x,y) touch run to one native stroke vector (STROKE_DIM)."""
    p0 = np.array([x[0], y[0]])
    p1 = np.array([x[-1], y[-1]])
    pts = np.stack([x, y], axis=1)
    # control waypoint = max perpendicular deviation from the start->end chord
    chord = p1 - p0
    clen = float(np.hypot(*chord))
    if clen < 1e-6 or len(t) < 3:
        ctrl = (p0 + p1) / 2
    else:
        n = np.array([-chord[1], chord[0]]) / clen
        dev = (pts[1:-1] - p0) @ n
        ctrl = pts[1:-1][int(np.argmax(np.abs(dev)))]
    duration = float(np.clip(t[-1] - t[0], _DUR_LO, _DUR_HI))
    # arc-length progress vs normalised time -> easing
    seg = np.hypot(np.diff(x), np.diff(y))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    span = max(t[-1] - t[0], 1e-6)
    easing = _fit_easing(cum / total, (t - t[0]) / span) if total > 1e-6 else 1.0
    return np.array([p0[0], p0[1], ctrl[0], ctrl[1], p1[0], p1[1], duration, easing, delay_before],
                    dtype=np.float64)


def assemble_strokes(
    active: np.ndarray, x: np.ndarray, y: np.ndarray, times: np.ndarray,
    *, max_gap_s: float = 0.10, min_frames: int = 2,
) -> list[Stroke]:
    """Per-frame touch track -> ordered list of assembled Strokes."""
    active = np.asarray(active, dtype=bool)
    x, y, times = (np.asarray(a, dtype=np.float64) for a in (x, y, times))
    strokes: list[Stroke] = []
    prev_end: float | None = None
    for run in _runs(active, times, max_gap_s=max_gap_s, min_frames=min_frames):
        tr, xr, yr = times[run], x[run], y[run]
        delay = 0.0 if prev_end is None else float(np.clip(tr[0] - prev_end, _BOUNDS[6, 0] - 1, 0.8))
        params = fit_stroke(tr, xr, yr, delay)
        strokes.append(Stroke(params=params, t_start=float(tr[0]), t_end=float(tr[-1])))
        prev_end = float(tr[-1])
    return strokes


# --- round-trip self-test --------------------------------------------------
if __name__ == "__main__":
    from trueskate_ai.vision.self_label import label_frames

    # a curved 3-waypoint drag, sampled densely at 30 fps, with two strokes in a clip
    g1 = dict(waypoints=[(0.30, 0.72), (0.46, 0.50), (0.74, 0.44)], total_duration=0.40, easing_power=1.0)
    g2 = dict(waypoints=[(0.60, 0.40), (0.50, 0.60), (0.40, 0.80)], total_duration=0.30, easing_power=1.0)
    fps = 30.0
    # stroke 1 over [0.1,0.5], gap, stroke 2 over [0.8,1.1] in clip time
    t1 = np.arange(0.10, 0.10 + g1["total_duration"] + 1e-9, 1 / fps)
    t2 = np.arange(0.80, 0.80 + g2["total_duration"] + 1e-9, 1 / fps)
    clip_t = np.concatenate([np.arange(0.0, 0.10, 1 / fps), t1, np.arange(0.55, 0.80, 1 / fps), t2,
                             np.arange(1.15, 1.30, 1 / fps)])
    # build per-frame touch track from BOTH gestures (each labels its own window)
    l1 = label_frames(g1["waypoints"], g1["total_duration"], g1["easing_power"], list(clip_t - t1[0]))
    l2 = label_frames(g2["waypoints"], g2["total_duration"], g2["easing_power"], list(clip_t - t2[0]))
    active = np.array([a.active or b.active for a, b in zip(l1, l2)])
    xs = np.array([a.x if a.active else b.x for a, b in zip(l1, l2)])
    ys = np.array([a.y if a.active else b.y for a, b in zip(l1, l2)])

    strokes = assemble_strokes(active, xs, ys, clip_t)
    assert len(strokes) == 2, f"expected 2 strokes, got {len(strokes)}"
    s1, s2 = strokes[0].params, strokes[1].params
    # endpoints recovered (tol ≈ 1 frame of motion — discrete sampling truncates
    # the run at the last ACTIVE frame, up to ~1/fps short of the nominal end)
    assert abs(s1[0] - 0.30) < 0.03 and abs(s1[1] - 0.72) < 0.03, s1
    assert abs(s1[4] - 0.74) < 0.06 and abs(s1[5] - 0.44) < 0.06, s1
    assert abs(s2[0] - 0.60) < 0.03 and abs(s2[5] - 0.80) < 0.06, s2
    # durations recovered (within ~1 frame)
    assert abs(s1[6] - 0.40) < 0.05 and abs(s2[6] - 0.30) < 0.05, (s1[6], s2[6])
    # control captures the bend (deviates from the chord midpoint)
    assert abs(s1[2] - 0.46) < 0.06 and abs(s1[3] - 0.50) < 0.06, s1
    # delay_before of stroke 2 ≈ gap (0.80 - 0.50 = 0.30)
    assert abs(s2[8] - 0.30) < 0.06, s2[8]
    print("assemble round-trip OK:",
          " s1=", [round(v, 3) for v in s1],
          " s2=", [round(v, 3) for v in s2])
