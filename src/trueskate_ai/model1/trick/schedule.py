"""One recorded minute: die-five markers and resets as instrumented W3C requests, runs between.

There is no screenshot before each touch. Menu checks were dropped for this batch
(operator decision 2026-10-15): every gesture path avoids the protected controls, the
foreground check runs before every run, and a full-screen menu would also fail the
calibration markers, which rejects the whole recording.
"""
from __future__ import annotations

from trueskate_ai.research.curved_audit import marker_payload, reset_payload
from .contract import FRAME_COUNT
from .sampling import sample_run

CONTROL_SLOTS = (("start", 1.5), ("middle", 29.), ("end", 57.))
RESET_SLOTS = (26., 53.5)  # 3 s or more of settling before the next marker, as in curved v1
# Runs (guard included) start and finish inside these spans: after a marker's W3C call
# has returned (≤1.22 s measured) and at least 0.8 s before the next reset.
SPANS = ((4., 25.2), (32., 52.7))
GUARD_S = .4             # foreground query: 80–360 ms measured
RETURN_BUDGET_S = .6     # direct requests return ~0.27–0.33 s after the gesture (GESTURES.md)
NOMINAL_RETURN_S = .3
STOP_S = 59.
# A run must fit one 120-frame clip (~1.98 s at 60 fps) with five frames either side.
MAX_RUN_SPAN_S = 1.6
assert (FRAME_COUNT - 10) / 60 - MAX_RUN_SPAN_S >= .2


def run_budget_s(run: dict) -> float:
    """Worst case from the run's foreground check to its last request returning."""
    return (GUARD_S + sum(s["times_ms"][-1] for s in run["strokes"]) / 1000
            + len(run["strokes"]) * RETURN_BUDGET_S + sum(run["gaps_ms"]) / 1000)


def nominal_span_s(run: dict) -> float:
    """Expected time from the first touch-down to the last lift."""
    return (sum(s["times_ms"][-1] for s in run["strokes"]) / 1000
            + (len(run["strokes"]) - 1) * NOMINAL_RETURN_S + sum(run["gaps_ms"]) / 1000)


def plan_minute(first_run: int) -> dict:
    """Pack runs ``first_run``, ``first_run + 1``, … into the minute's spans, in order.

    A run too long for one clip is skipped (never sent). A run that no longer fits a
    span moves to the next span; the next minute starts with the first unused run.
    """
    commands = [dict(kind="control", role=role, slot_s=slot, payload=marker_payload())
                for role, slot in CONTROL_SLOTS]
    commands += [dict(kind="reset", slot_s=slot, payload=reset_payload()) for slot in RESET_SLOTS]
    runs, skipped, number = [], [], first_run
    for begin, end in SPANS:
        cursor = begin
        while True:
            run = sample_run(number)
            if nominal_span_s(run) > MAX_RUN_SPAN_S:
                skipped.append(number)
                number += 1
                continue
            budget = run_budget_s(run)
            if cursor + budget > end:
                break
            runs.append(dict(run, slot_s=round(cursor + GUARD_S, 3), budget_s=round(budget, 3)))
            cursor += budget
            number += 1
    return dict(commands=sorted(commands, key=lambda c: c["slot_s"]), runs=runs,
                skipped_runs=skipped, next_run=number)
