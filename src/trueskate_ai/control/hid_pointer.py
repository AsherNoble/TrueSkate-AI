"""Plan board-timed schedules for the ESP32 Bluetooth pointer (iOS AssistiveTouch).

Measured on XR2 (iOS 18.7.10, AssistiveTouch defaults) on 2026-10-04, 49 drags:
each relative report c = (dx, dy) counts moves the cursor D(|c|) points along c.
D does not depend on report timing (15/30/60 ms gave the same result) or on earlier
reports, and x and y behave the same. D is piecewise linear in |c|:

    |c| < 3     D = 0.2017 |c|
    |c| >= 3    D = min(max(1.0009|c| - 2.107, 2.4485|c| - 39.405), 0.965|c| + 66.997)

The fit is 0.14 pt rms per report (max 0.44). Reports reach the phone on the 15 ms
Bluetooth connection interval, so schedules sit on a 15 ms grid: a whole schedule
then shifts by one constant delay and the gaps inside it survive.

A lift must be its own report: a report that both moves and releases loses its
move (the touch ends one step short). "immediate" lifts follow the last move by
1 ms, so both usually travel in the same connection event; "separate" lifts wait
for the next slot, which holds the finger still for 15 ms before it leaves.

A schedule is a list of (t_us, dx, dy, buttons) events for the firmware's E command.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

STEP_US = 15_000
LIFT_DELAY_US = 1_000
LIFT_MODES = ('immediate', 'separate')
MAX_COUNT = 127
SLOW_LIMIT = 3.0
SLOW_GAIN = 0.2017
LINES = ((1.0009, -2.107), (2.4485, -39.405), (0.965, 66.997))
# Home: 25 reports of (-127, -127) pin the cursor in the top-left corner. Park: 40
# reports of (3, 11) then put it here, measured to +/-0.5 pt across 60 drags.
HOME_REPORTS = 25
PARK_STEP = (3, 11)
PARK_REPORTS = 40
PARK_POINT = (109.1, 368.2)


def gain_distance(m: float) -> float:
    """Points moved by one report whose count vector has length m."""
    if m <= 0:
        return 0.0
    if m < SLOW_LIMIT:
        return SLOW_GAIN * m
    (a1, b1), (a2, b2), (a3, b3) = LINES
    return min(max(a1 * m + b1, a2 * m + b2), a3 * m + b3)


def displacement(dx: int, dy: int) -> tuple[float, float]:
    """Cursor movement (points) for one report of (dx, dy) counts."""
    m = math.hypot(dx, dy)
    if m == 0:
        return 0.0, 0.0
    d = gain_distance(m) / m
    return dx * d, dy * d


def counts_length(distance: float) -> float:
    """Inverse of gain_distance for distances the fast region covers (>= D(3))."""
    lo, hi = SLOW_LIMIT, float(MAX_COUNT)
    if distance <= gain_distance(lo):
        return lo
    if distance >= gain_distance(hi):
        return hi
    for _ in range(60):                                   # gain_distance is increasing on [3, 127]
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if gain_distance(mid) < distance else (lo, mid)
    return (lo + hi) / 2


def best_report(vx: float, vy: float) -> tuple[int, int]:
    """Integer report whose modelled movement is closest to (vx, vy) points."""
    dist = math.hypot(vx, vy)
    if dist < 0.1:
        return 0, 0
    if dist < gain_distance(SLOW_LIMIT):
        seeds = [(vx / SLOW_GAIN, vy / SLOW_GAIN)]
        m = counts_length(dist)
        seeds.append((vx / dist * m, vy / dist * m))
    else:
        m = counts_length(dist)
        seeds = [(vx / dist * m, vy / dist * m)]
    best, best_err = (0, 0), dist
    for sx, sy in seeds:
        for cx in range(math.floor(sx) - 1, math.floor(sx) + 3):
            for cy in range(math.floor(sy) - 1, math.floor(sy) + 3):
                if max(abs(cx), abs(cy)) > MAX_COUNT or math.hypot(cx, cy) > MAX_COUNT:
                    continue
                px, py = displacement(cx, cy)
                err = math.hypot(px - vx, py - vy)
                if err < best_err:
                    best, best_err = (cx, cy), err
    return best


def follow(start: tuple[float, float], targets: list[tuple[float, float]]):
    """Reports that walk the cursor through targets, one report per target.

    Each report aims at its target from the modelled position so far, so errors do
    not accumulate. Returns (reports, modelled positions after each report).
    """
    x, y = start
    reports, positions = [], []
    for tx, ty in targets:
        cx, cy = best_report(tx - x, ty - y)
        px, py = displacement(cx, cy)
        x, y = x + px, y + py
        reports.append((cx, cy))
        positions.append((x, y))
    return reports, positions


def hover_slots(start: tuple[float, float], end: tuple[float, float]) -> int:
    """Fewest reports that can carry the cursor from start to end."""
    return max(1, math.ceil(math.hypot(end[0] - start[0], end[1] - start[1]) / gain_distance(MAX_COUNT)))


@dataclass
class Stroke:
    """One touch contact: samples are (t_s, x_pt, y_pt); press at the first, lift at the last.

    lift overrides build_schedule's lift mode for this stroke when set.
    """
    name: str
    samples: list[tuple[float, float, float]]
    lift: str | None = None


def _position_at(samples, t):
    if t <= samples[0][0]:
        return samples[0][1:]
    for (ta, xa, ya), (tb, xb, yb) in zip(samples, samples[1:]):
        if t <= tb:
            u = (t - ta) / (tb - ta) if tb > ta else 1.0
            return xa + u * (xb - xa), ya + u * (yb - ya)
    return samples[-1][1:]


def grid_phase(strokes: list[Stroke], step_s: float = STEP_US / 1e6) -> float:
    """Offset (s) of the 15 ms grid that best fits every stroke's press and lift times."""
    times = [t for s in strokes for t in (s.samples[0][0], s.samples[-1][0])]
    candidates = [k * step_s / 30 for k in range(30)]
    return min(candidates, key=lambda ph: sum(((t - ph) / step_s - round((t - ph) / step_s)) ** 2 for t in times))


def build_schedule(strokes: list[Stroke], lead_s: float = 1.0, lift: str = 'immediate'):
    """Home, park, then play the strokes on the 15 ms grid with hover moves between them.

    Stroke times are kept relative to each other (rounded to the grid); the first press
    comes lead_s after parking ends. The press is a still report; moves follow on each
    slot and reach the last sample on the slot of its time. The lift is a still report
    1 ms after the last move ("immediate") or on the next slot ("separate").
    Returns (events, info) where events are (t_us, dx, dy, buttons) and info lists each
    stroke's slots and modelled path.
    """
    if not strokes:
        raise ValueError('no strokes')
    if lift not in LIFT_MODES or any(s.lift not in (None,) + LIFT_MODES for s in strokes):
        raise ValueError(f'lift must be one of {LIFT_MODES}')
    if not math.isfinite(lead_s) or not 0 <= lead_s <= 60:
        raise ValueError('invalid schedule lead time')
    for stroke in strokes:
        if len(stroke.samples) < 2:
            raise ValueError('each stroke requires at least two samples')
        previous = -1.0
        for sample in stroke.samples:
            if len(sample) != 3 or not all(math.isfinite(v) for v in sample):
                raise ValueError('finite stroke samples required')
            t, x, y = sample
            if not 0 <= t <= 60 or t <= previous or not 0 <= x <= 414 or not 0 <= y <= 896:
                raise ValueError('invalid stroke time or XR2 coordinates')
            previous = t
    strokes = sorted(strokes, key=lambda s: s.samples[0][0])
    events: list[tuple[int, int, int, int]] = []
    slot = 0
    for _ in range(HOME_REPORTS):
        events.append((slot * STEP_US, -MAX_COUNT, -MAX_COUNT, 0)); slot += 1
    slot += 20                                                # 300 ms still in the corner
    for _ in range(PARK_REPORTS):
        events.append((slot * STEP_US, PARK_STEP[0], PARK_STEP[1], 0)); slot += 1
    park_end = slot
    phase = grid_phase(strokes)
    step_s = STEP_US / 1e6
    origin_slot = round((strokes[0].samples[0][0] - phase) / step_s)
    base = park_end + round(lead_s / step_s)

    def to_slot(t):                                       # clip time -> slot, on the best-fitting grid
        return base + round((t - phase) / step_s) - origin_slot

    pos = PARK_POINT
    free_from = park_end                                      # first slot not yet used
    info = []
    for s in strokes:
        press = to_slot(s.samples[0][0])
        lift_slot = max(press + 1, to_slot(s.samples[-1][0]))
        start_pt = s.samples[0][1:]
        need = hover_slots(pos, start_pt)
        if press - free_from < need:
            raise ValueError(f'{s.name}: {press - free_from} slots to reach the start, need {need}')
        # spread the hover evenly over the slots just before the press (cursor arrives one slot early)
        hover_n = min(press - free_from, max(need, 4))
        first = press - hover_n
        hover_targets = [(pos[0] + (start_pt[0] - pos[0]) * (k + 1) / hover_n,
                          pos[1] + (start_pt[1] - pos[1]) * (k + 1) / hover_n) for k in range(hover_n)]
        reports, path = follow(pos, hover_targets)
        for k, (cx, cy) in enumerate(reports):
            events.append(((first + k) * STEP_US, cx, cy, 0))
        pos = path[-1]
        events.append((press * STEP_US, 0, 0, 1))
        # moves on the slots after the press, reaching the last sample on the lift slot
        mode = s.lift or lift
        t_press = s.samples[0][0]
        move_slots = list(range(press + 1, lift_slot + 1))
        targets = [_position_at(s.samples, t_press + (k - press) * step_s) for k in move_slots]
        targets[-1] = s.samples[-1][1:]
        reports, path = follow(pos, targets)
        for k, (cx, cy) in zip(move_slots, reports):
            events.append((k * STEP_US, cx, cy, 1))
        pos = path[-1]
        release_us = lift_slot * STEP_US + LIFT_DELAY_US if mode == 'immediate' else (lift_slot + 1) * STEP_US
        events.append((release_us, 0, 0, 0))
        free_from = lift_slot + (1 if mode == 'immediate' else 2)
        info.append(dict(name=s.name, press_slot=press, last_move_slot=lift_slot, release_us=release_us, lift=mode,
                         start=start_pt, modelled_end=pos,
                         target_end=s.samples[-1][1:], hover_slots=hover_n,
                         path=[(round(x, 2), round(y, 2)) for x, y in path]))
    events.sort(key=lambda e: e[0])
    return events, dict(phase_s=phase, park_end_slot=park_end, first_press_slot=base, strokes=info)
