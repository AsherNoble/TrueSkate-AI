"""Resample an executed gesture command to positions at evenly-spaced times.

This is the keystone of the MVP-3 representation. Instead of predicting semantic
waypoints plus an easing power, Model 1 predicts *where the finger was* at K
fixed fractions of the gesture. That choice is what lets one architecture cover a
straight drag, a basic curve, and eventually a Z: only K changes.

Two properties make it work, and both are properties of the executor rather than
of this module:

* ``curved_drag`` chains linear pointer moves through the waypoints, so an
  executed gesture is exactly a polyline in space and piecewise-linear in time.
* With ``easing=None`` it splits the duration equally across segments. So the K
  points returned here, replayed as a polyline with no easing, reproduce the
  sampled trajectory — the representation is directly executable.

Consequently a command with *any* ``easing_power`` and *any* number of waypoints
resamples into the same fixed-width target, and easing never needs predicting.
"""
from __future__ import annotations

import numpy as np

from trueskate_ai.sim.touch_actions import easing_to_segment_durations


def command_knot_times(waypoints, duration: float, easing_power: float = 1.0) -> np.ndarray:
    """Return the time, in seconds, at which the drag reaches each waypoint.

    Mirrors ``easing_to_segment_durations`` exactly, including its integer-ms
    quantisation, so the labels match what the device actually executed rather
    than an idealised continuous model of it.
    """
    points = np.asarray(waypoints, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("waypoints must be a sequence of at least two (x, y) pairs")
    if not np.isfinite(points).all():
        raise ValueError("waypoints must be finite")
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be positive and finite")
    segments = len(points) - 1
    total_ms = int(duration * 1000)
    if easing_power == 1.0:
        durations = [max(1, total_ms // segments)] * segments
    else:
        if not np.isfinite(easing_power) or easing_power <= 0:
            raise ValueError("easing_power must be positive and finite")
        durations = easing_to_segment_durations(segments, total_ms, lambda t: t ** easing_power)
    return np.concatenate(([0.0], np.cumsum(np.asarray(durations, dtype=float) / 1000.0)))


def resample_command_at_times(waypoints, duration: float, *, knots: int,
                              easing_power: float = 1.0) -> np.ndarray:
    """Positions at ``knots`` evenly-spaced times spanning the whole gesture.

    Returns an array of shape ``[knots, 2]``. The first and last rows are the
    commanded start and end exactly; interior rows are linear interpolations
    within whichever executed segment covers that instant.
    """
    if knots < 2:
        raise ValueError("knots must be at least 2")
    points = np.asarray(waypoints, dtype=float)
    boundaries = command_knot_times(points, duration, easing_power)
    # The quantised segment durations rarely sum to exactly `duration`; stretch
    # the boundary times onto the true span so the last knot lands on the
    # commanded endpoint instead of drifting a few ms short.
    span = float(boundaries[-1])
    if span <= 0:
        raise ValueError("executed gesture has zero duration")
    boundaries = boundaries / span
    targets = np.linspace(0.0, 1.0, knots)
    sampled = np.empty((knots, 2), dtype=float)
    for index, fraction in enumerate(targets):
        segment = int(np.searchsorted(boundaries, fraction, side="right") - 1)
        segment = min(max(segment, 0), len(points) - 2)
        start_time, end_time = boundaries[segment], boundaries[segment + 1]
        width = end_time - start_time
        local = 0.0 if width <= 0 else (fraction - start_time) / width
        local = min(max(local, 0.0), 1.0)
        sampled[index] = points[segment] + (points[segment + 1] - points[segment]) * local
    return sampled
