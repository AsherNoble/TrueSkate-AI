"""Construct feasible random chains and straight-to-half-circle approximations."""
from __future__ import annotations

from functools import lru_cache
import math

import numpy as np

from trueskate_ai.data.control_hitboxes import CONTROL_START_EXCLUSIONS
from .contract import (ARC_TOLERANCE_PT, HEIGHT, MAX_SEGMENTS, MIN_SEGMENT_MS,
                       WIDTH, Gesture)


@lru_cache(maxsize=1)
def safe_rectangles() -> tuple[tuple[float, float, float, float], ...]:
    """Maximal empty rectangles, inset two logical points against truncation."""
    bounds = (.12, .12, 1., .88)
    obstacles = [h.rect for h in CONTROL_START_EXCLUSIONS]
    xs = sorted({bounds[0], bounds[2], *(max(bounds[0], min(bounds[2], x))
                  for r in obstacles for x in (r[0], r[2]))})
    candidates = []
    for i, left in enumerate(xs):
        for right in xs[i+1:]:
            blocked = sorted((max(bounds[1], r[1]), min(bounds[3], r[3]))
                             for r in obstacles if r[0] < right and r[2] > left
                             and r[3] > bounds[1] and r[1] < bounds[3])
            bottom = bounds[1]
            for lo, hi in blocked + [(bounds[3], bounds[3])]:
                if lo > bottom:
                    candidates.append((left, bottom, right, lo))
                bottom = max(bottom, hi)
    maximal = [r for r in candidates if not any(
        other != r and other[0] <= r[0] and other[1] <= r[1]
        and other[2] >= r[2] and other[3] >= r[3] for other in candidates)]
    result = tuple((a*WIDTH+2, b*HEIGHT+2, c*WIDTH-2, d*HEIGHT-2)
                   for a, b, c, d in maximal
                   if (c-a)*WIDTH >= 36 and (d-b)*HEIGHT >= 36)
    if not result:
        raise ValueError("control map has no usable construction region")
    return result


def allocate_ms(total: int, weights) -> tuple[int, ...]:
    weights = np.asarray(weights, dtype=float)
    if (weights.ndim != 1 or not len(weights) or not np.isfinite(weights).all()
            or np.any(weights <= 0) or total < MIN_SEGMENT_MS * len(weights)):
        raise ValueError("infeasible duration allocation")
    raw = (total - MIN_SEGMENT_MS*len(weights)) * weights / weights.sum()
    extra = np.floor(raw).astype(int)
    missing = total - MIN_SEGMENT_MS*len(weights) - int(extra.sum())
    order = np.argsort(-(raw-extra), kind="stable")
    extra[order[:missing]] += 1
    return tuple(int(x+MIN_SEGMENT_MS) for x in extra)


def timing(rng, count: int, total: int, *, pause: bool = False):
    profile = str(rng.choice(("uniform", "accelerating", "decelerating",
                              "slow_fast_slow", "random")))
    if profile == "uniform":
        weights = np.ones(count)
    elif profile == "accelerating":
        weights = np.linspace(1.8, .3, count)
    elif profile == "decelerating":
        weights = np.linspace(.3, 1.8, count)
    elif profile == "slow_fast_slow":
        weights = 1 + 2*np.abs(np.linspace(-1, 1, count))
    else:
        weights = rng.uniform(.1, 2., count)
    return ("pause" if pause else profile), (0, *np.cumsum(allocate_ms(total, weights)).tolist())


def _region(rng):
    regions = safe_rectangles()
    areas = np.asarray([(r[2]-r[0])*(r[3]-r[1]) for r in regions])
    return regions[int(rng.choice(len(regions), p=areas/areas.sum()))]


def sample_gesture(seed: int, sampler: str) -> Gesture:
    rng = np.random.default_rng(seed)
    duration = int(rng.integers(300, 1201))
    budget = min(MAX_SEGMENTS, duration // MIN_SEGMENT_MS)
    x0, y0, x1, y1 = _region(rng)
    if sampler == "random_chain":
        count = int(rng.integers(1, budget+1))
        points = rng.uniform([x0, y0], [x1, y1], (count+1, 2))
        pause = count >= 2 and rng.random() < .2
        if pause:
            index = int(rng.integers(1, count+1))
            points[index] = points[index-1]
        profile, times = timing(rng, count, duration, pause=pause)
        ideal = None
    elif sampler == "arc":
        theta = 0. if rng.random() < .1 else float(rng.uniform(1e-4, math.pi))
        orientation = float(rng.uniform(-math.pi, math.pi))
        if theta == 0.:
            direction = np.array([math.cos(orientation), math.sin(orientation)])
            maximum = min((x1-x0)/max(abs(direction[0]), 1e-12),
                          (y1-y0)/max(abs(direction[1]), 1e-12))
            length = float(rng.uniform(8., maximum))
            base = np.asarray([[0., 0.], direction*length])
            count = 1
            ideal = dict(kind="line", length_pt=length, orientation=orientation)
        else:
            # Exact rotated arc bounding box: include every x/y extremum.
            extrema = [0., theta]
            for k in range(-4, 5):
                for a in (k*math.pi-orientation, math.pi/2+k*math.pi-orientation):
                    if 0. < a < theta:
                        extrema.append(a)
            angles = np.asarray(extrema)
            envelope = np.stack((np.sin(angles+orientation)-math.sin(orientation),
                                 math.cos(orientation)-np.cos(angles+orientation)), axis=1)
            span = np.ptp(envelope, axis=0)
            maximum = min((x1-x0)/max(span[0], 1e-12),
                          (y1-y0)/max(span[1], 1e-12),
                          ARC_TOLERANCE_PT/(1-math.cos(theta/(2*budget))))
            radius = float(rng.uniform(min(8., maximum/2), maximum))
            count = next(n for n in range(1, budget+1)
                         if radius*(1-math.cos(theta/(2*n))) <= ARC_TOLERANCE_PT+1e-10)
            angles = np.linspace(0., theta, count+1)
            base = radius*np.stack((np.sin(angles+orientation)-math.sin(orientation),
                                    math.cos(orientation)-np.cos(angles+orientation)), axis=1)
            # Translate using the ideal arc envelope, not just sampled vertices.
            low, high = envelope.min(axis=0)*radius, envelope.max(axis=0)*radius
            offset = rng.uniform([x0, y0]-low, [x1, y1]-high)
            points = base+offset
            ideal = dict(kind="arc", radius_pt=radius, angle_rad=theta,
                         orientation=orientation, offset_pt=offset.tolist(),
                         maximum_deviation_pt=radius*(1-math.cos(theta/(2*count))))
        if theta == 0.:
            low, high = base.min(axis=0), base.max(axis=0)
            points = base+rng.uniform([x0, y0]-low, [x1, y1]-high)
        profile, times = timing(rng, count, duration)
    else:
        raise ValueError("unknown sampler")
    gesture = Gesture(tuple(map(tuple, (points/[WIDTH, HEIGHT]).tolist())),
                      tuple(times), sampler, profile, ideal)
    gesture.payload()  # Validate every integer-quantized segment; no retries.
    return gesture
