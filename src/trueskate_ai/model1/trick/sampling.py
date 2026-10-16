"""Kickflip-shaped and random trick-speed gestures, and short runs of them.

About half the runs follow the kickflip's order (push, pop, flick, catch, or part of
it) with each gesture jittered around the operator's filmed kickflip; the rest mix
families freely. Durations are drawn log-uniformly, so most gestures are short.
"""
from __future__ import annotations

import math

import numpy as np

from trueskate_ai.model1.curved.sampling import safe_rectangles
from .contract import (FAMILIES, HEIGHT, MAX_SEGMENTS, MAX_GESTURE_MS, MIN_SEGMENT_MS, MIN_GESTURE_MS,
                       SEED, WIDTH, Gesture)

# The filmed kickflip as the approved candidate_19 replays it (KICKFLIP-REPEATABILITY-20261009),
# in normalized coordinates.
PUSH = ((.813, .27), (.74, .522))
POP = ((.505, .56), (.555, .815))
FLICK = ((.537, .51), (.57, .54), (.715, .56))
CATCH = (.5, .505)
KICKFLIP_RUNS = ((("pop", "flick"), .3), (("pop", "flick", "catch"), .25), (("push", "pop", "flick"), .2),
                 (("push", "pop", "flick", "catch"), .15), (("pop",), .05), (("flick",), .05))
MIXED_FAMILIES = (("random", .5), ("push", .125), ("pop", .125), ("flick", .125), ("catch", .125))
RUN_LENGTHS = ((1, .3), (2, .3), (3, .25), (4, .15))
# Extra wait after a gesture's request returns. The return alone (~0.3 s) already
# exceeds the filmed pop-to-flick gap (0.13 s), so keep the extra small.
MAX_GAP_MS = 150
MAX_SPEED_PT_S = 4000.  # the filmed pop moves ~3,000 logical points per second
MAX_REDRAWS = 200


def _points(p) -> np.ndarray:
    return np.array([p[0] * WIDTH, p[1] * HEIGHT], dtype=float)


def _rotate(vector, degrees) -> np.ndarray:
    a = math.radians(degrees)
    return np.array([vector[0] * math.cos(a) - vector[1] * math.sin(a),
                     vector[0] * math.sin(a) + vector[1] * math.cos(a)])


def _log_ms(rng, low, high) -> int:
    return int(round(math.exp(rng.uniform(math.log(low), math.log(high)))))


def _split(rng, total, count) -> list[int]:
    """Random segment durations, each at least MIN_SEGMENT_MS, summing exactly to ``total``."""
    weights = rng.uniform(.3, 1.7, count)
    segment_ms = np.floor((total - MIN_SEGMENT_MS * count) * weights / weights.sum()).astype(int) + MIN_SEGMENT_MS
    segment_ms[:total - int(segment_ms.sum())] += 1
    return segment_ms.tolist()


def _push(rng):
    start = _points(PUSH[0]) + rng.normal(0., 30., 2)
    move = _rotate((_points(PUSH[1]) - _points(PUSH[0])) * rng.uniform(.6, 1.4), rng.uniform(-25, 25))
    return [start, start + move], [_log_ms(rng, 80, 300)]


def _pop(rng):
    start = _points(POP[0]) + rng.normal(0., 24., 2)
    move = _rotate((_points(POP[1]) - _points(POP[0])) * rng.uniform(.4, 1.3), rng.uniform(-30, 30))
    points, segment_ms = [start], []
    if rng.random() < .25:  # a short pause before the snap ("ppppOP")
        points.append(start)
        segment_ms.append(int(rng.integers(MIN_SEGMENT_MS, 121)))
    return points + [start + move], segment_ms + [_log_ms(rng, MIN_SEGMENT_MS, 200)]


def _flick(rng):
    a, b, c = (_points(p) for p in FLICK)
    start = a + rng.normal(0., 24., 2)
    arc = _rotate((b - a) * rng.uniform(.3, 2.), rng.uniform(-40, 40))
    out = _rotate((c - b) * rng.uniform(.6, 1.5), rng.uniform(-35, 35))
    if rng.random() < .3:  # flick the other way too
        arc[0], out[0] = -arc[0], -out[0]
    points, segment_ms = [start], []
    if rng.random() < .5:  # the filmed flick holds ~70 ms before moving
        points.append(start)
        segment_ms.append(int(rng.integers(MIN_SEGMENT_MS, 141)))
    points += [start + arc, start + arc + out]
    return points, segment_ms + [int(rng.integers(MIN_SEGMENT_MS, 81)), int(rng.integers(MIN_SEGMENT_MS, 101))]


def _catch(rng):
    point = _points(CATCH) + rng.normal(0., 30., 2)
    return [point, point], [_log_ms(rng, 150, MAX_GESTURE_MS)]


def _random(rng):
    regions = safe_rectangles()
    areas = np.asarray([(r[2] - r[0]) * (r[3] - r[1]) for r in regions])
    x0, y0, x1, y1 = regions[int(rng.choice(len(regions), p=areas / areas.sum()))]
    count = int(rng.integers(1, MAX_SEGMENTS + 1))
    segment_ms = _split(rng, _log_ms(rng, max(MIN_GESTURE_MS, MIN_SEGMENT_MS * count), MAX_GESTURE_MS), count)
    points = [rng.uniform([x0, y0], [x1, y1])]
    hold = int(rng.integers(0, count)) if count >= 2 and rng.random() < .2 else None
    for index, duration in enumerate(segment_ms):
        if index == hold:
            points.append(points[-1].copy())
            continue
        speed = math.exp(rng.uniform(math.log(300.), math.log(MAX_SPEED_PT_S)))
        points.append(points[-1] + _rotate((speed * duration / 1000., 0.), rng.uniform(-180, 180)))
    return points, segment_ms


SAMPLERS = dict(push=_push, pop=_pop, flick=_flick, catch=_catch, random=_random)
assert set(SAMPLERS) == set(FAMILIES)


def sample_gesture(rng, family: str) -> tuple[Gesture, int]:
    """A feasible gesture and the number of infeasible draws it took."""
    for redraw in range(MAX_REDRAWS):
        points, segment_ms = SAMPLERS[family](rng)
        times = (0, *(int(t) for t in np.cumsum(segment_ms)))
        try:
            return Gesture(tuple((float(x / WIDTH), float(y / HEIGHT)) for x, y in points), times, family), redraw
        except ValueError:
            continue
    raise ValueError(f"no feasible {family} gesture in {MAX_REDRAWS} draws")


def _pick(rng, options):
    values, weights = zip(*options)
    return values[int(rng.choice(len(values), p=np.asarray(weights) / sum(weights)))]


def sample_run(number: int) -> dict:
    """Run ``number`` of the batch: its gestures, in order, and the extra wait after each."""
    rng = np.random.default_rng([SEED, number])
    if rng.random() < .5:
        kind, families = "kickflip_shaped", _pick(rng, KICKFLIP_RUNS)
    else:
        kind = "mixed"
        families = tuple(_pick(rng, MIXED_FAMILIES) for _ in range(_pick(rng, RUN_LENGTHS)))
    gestures, redraws = [], 0
    for family in families:
        gesture, count = sample_gesture(rng, family)
        gestures.append(gesture.to_dict())
        redraws += count
    gaps = [int(rng.integers(0, MAX_GAP_MS + 1)) for _ in gestures[1:]]
    return dict(number=number, seed=[SEED, number], kind=kind, strokes=gestures, gaps_ms=gaps, redraws=redraws)
