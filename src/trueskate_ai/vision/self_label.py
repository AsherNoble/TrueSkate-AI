"""Ground-truth touch labels for agent self-labeled trace capture.

The agent fires gestures it knows EXACTLY (the waypoints it sent). True Skate
renders an orange finger-trace; capturing color screen frames during the
gesture yields (frame, known-touch) pairs for free — the corpus that trains
Model 1 (the learned trace extractor) and lets us MEASURE the hand-tuned
trace_extractor's true positional error. See experiments/vision_sequence_leap_journal.md.

This module is the pure, device-free label core: given a gesture's waypoints +
timing and a frame's timestamp (relative to the gesture start), it returns the
instantaneous touch position the finger was at when that frame was rendered.
Capture stores raw (frames + timestamps + gesture spec); labels are computed
here so the MJPEG/render latency offset stays a tunable hyperparameter rather
than being baked into the data.
"""
from __future__ import annotations

from dataclasses import dataclass

from trueskate_ai.sim.touch_actions import easing_to_segment_durations


@dataclass(frozen=True)
class TouchLabel:
    """Instantaneous ground-truth touch for one frame (normalised [0, 1])."""
    active: bool
    x: float
    y: float


def segment_durations_s(num_waypoints: int, total_duration: float, easing_power: float) -> list[float]:
    """Per-segment durations (seconds) matching build_curved_drag's execution.

    Mirrors the W3C pointer schedule: easing reshapes the time spent per equal-
    index segment, and the pointer moves linearly within each segment.
    """
    n_segments = max(1, num_waypoints - 1)
    total_ms = int(total_duration * 1000)
    easing = None if easing_power == 1.0 else (lambda t, p=easing_power: t ** p)
    if easing is None:
        ms = [max(1, total_ms // n_segments)] * n_segments
    else:
        ms = easing_to_segment_durations(n_segments, total_ms, easing)
    return [m / 1000.0 for m in ms]


def position_at(
    waypoints: list[tuple[float, float]],
    seg_durations: list[float],
    t: float,
) -> tuple[float, float]:
    """Touch position at gesture-relative time t (seconds), piecewise-linear.

    Clamped: t <= 0 → first waypoint (touch down); t >= total → last waypoint
    (just before touch up).
    """
    if t <= 0:
        return waypoints[0]
    cum = 0.0
    for k, dur in enumerate(seg_durations):
        if t < cum + dur or k == len(seg_durations) - 1:
            frac = 0.0 if dur <= 0 else min(1.0, (t - cum) / dur)
            (x0, y0), (x1, y1) = waypoints[k], waypoints[k + 1]
            return (x0 + (x1 - x0) * frac, y0 + (y1 - y0) * frac)
        cum += dur
    return waypoints[-1]


def label_frames(
    waypoints: list[tuple[float, float]],
    total_duration: float,
    easing_power: float,
    frame_times: list[float],
    *,
    latency_s: float = 0.0,
    spin_active: bool = False,
) -> list[TouchLabel]:
    """Label each captured frame with the instantaneous touch position.

    Args:
        waypoints: gesture waypoints in normalised [0, 1].
        total_duration: gesture duration (seconds).
        easing_power: easing exponent used at execution.
        frame_times: capture timestamps, each RELATIVE to the gesture start
            (frame_ts - gesture_start_monotonic), seconds.
        latency_s: subtracted from each frame time to compensate the
            capture/render pipeline delay (frame shown at T reflects state at
            T - latency). Tunable offline against trace overlays.
        spin_active: whether the rotate button was held during this gesture.

    A frame whose compensated time falls within [0, total_duration] is an active
    touch; outside that window the finger is up → inactive (x, y = -1).
    """
    seg = segment_durations_s(len(waypoints), total_duration, easing_power)
    total = sum(seg)
    labels: list[TouchLabel] = []
    for ft in frame_times:
        t = ft - latency_s
        if 0.0 <= t <= total:
            x, y = position_at(waypoints, seg, t)
            labels.append(TouchLabel(active=True, x=float(x), y=float(y)))
        else:
            labels.append(TouchLabel(active=False, x=-1.0, y=-1.0))
    _ = spin_active  # carried through by the dataset for the spin-state channel
    return labels


# --- self-test -------------------------------------------------------------
if __name__ == "__main__":
    wps = [(0.2, 0.8), (0.5, 0.5), (0.8, 0.2)]  # diagonal 3-waypoint drag
    dur, ep = 1.0, 1.0
    seg = segment_durations_s(len(wps), dur, ep)
    assert abs(sum(seg) - dur) < 0.05, seg
    # midpoint of a linear, equal-segment drag → centre waypoint
    mid = position_at(wps, seg, sum(seg) / 2)
    assert abs(mid[0] - 0.5) < 0.02 and abs(mid[1] - 0.5) < 0.02, mid
    # endpoints clamp
    assert position_at(wps, seg, -1) == (0.2, 0.8)
    assert position_at(wps, seg, 99) == (0.8, 0.2)
    # frame labelling: before/at-start/mid/after
    labs = label_frames(wps, dur, ep, [-0.1, 0.0, 0.5, 1.0, 1.2])
    assert labs[0].active is False and labs[1].active is True
    assert abs(labs[2].x - 0.5) < 0.02, labs[2]
    assert labs[4].active is False
    # latency shift: a frame at t=0.05 with latency 0.05 → start
    shifted = label_frames(wps, dur, ep, [0.05], latency_s=0.05)[0]
    assert abs(shifted.x - 0.2) < 1e-6 and abs(shifted.y - 0.8) < 1e-6, shifted
    print("self_label tests OK; seg=", [round(s, 3) for s in seg], " mid=", tuple(round(v, 3) for v in mid))
