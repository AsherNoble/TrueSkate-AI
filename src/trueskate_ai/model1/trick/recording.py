"""Run one planned minute on XR1; stop on the first failure and keep the evidence.

Markers and resets go through Appium's W3C actions (timing-instrumented). Each gesture
goes to WDA's direct endpoint, sent as soon as the previous request returns plus the
run's small extra gap, with no guard between gestures (as in the kickflip runs).
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import time

from trueskate_ai.collection.wda_action_timing import validate_action_timing_report
from trueskate_ai.model1.curved.contract import digest, save_json
from trueskate_ai.research.audit_execution import deadline_sleep
from .contract import GESTURE_EVENT, Gesture
from .schedule import GUARD_S, RETURN_BUDGET_S, STOP_S

LATE_S = .1
RECORDER_START_LIMIT_S = 1.


def _response_error(response) -> bool:
    return isinstance(response, dict) and bool(
        response.get("error") or isinstance(response.get("value"), dict) and response["value"].get("error"))


def record_minute(*, out, plan, recorder, timing, perform, perform_direct, guard, revision, metadata,
                  clock=time.monotonic, sleep=deadline_sleep, epoch=time.time):
    out = Path(out)
    payloads = {(run["number"], j): Gesture.from_dict(gesture).payload()
                for run in plan["runs"] for j, gesture in enumerate(run["strokes"])}
    items = sorted([("command", c) for c in plan["commands"]] + [("run", r) for r in plan["runs"]],
                   key=lambda item: item[1]["slot_s"])
    out.mkdir(parents=True, exist_ok=False)
    save_json(out/"planned.json", dict(**metadata, wda_revision=revision, commands=plan["commands"],
                                       runs=plan["runs"], skipped_runs=plan["skipped_runs"]))
    events, failures = [], []
    video = report = None
    started = False

    def call(kind, payload, *, direct, **info):
        event = dict(kind=kind, instrumented=not direct, payload_sha256=digest(payload), success=False,
                     call_start_monotonic_s=clock(), call_start_epoch_s=epoch(), **info)
        events.append(event)
        response = (perform_direct if direct else perform)(payload)
        event.update(response=response, call_end_monotonic_s=clock(), call_end_epoch_s=epoch())
        if _response_error(response):
            raise RuntimeError(f"{'gesture' if kind == GESTURE_EVENT else kind} response reports an error")
        event["success"] = True

    try:
        guard()
        timing.start()
        start_call = clock()
        recorder.start()  # One attempt; no retry or service restart.
        started = True
        origin = clock()
        if origin - start_call > RECORDER_START_LIMIT_S:
            raise RuntimeError("recorder start exceeded its reserved lead")
        for index, (item, spec) in enumerate(items):
            target = origin + spec["slot_s"]
            following = origin + (items[index+1][1]["slot_s"] if index+1 < len(items) else STOP_S)
            sleep(max(0., target - GUARD_S - clock()))
            guard()
            sleep(max(0., target - clock()))
            lateness = clock() - target
            if lateness > LATE_S:
                raise RuntimeError(f"schedule overrun: {lateness:.3f} s late")
            if item == "command":
                call(spec["kind"], spec["payload"], direct=False, role=spec.get("role"),
                     slot_s=spec["slot_s"], lateness_s=lateness)
            else:
                for j, gesture in enumerate(spec["strokes"]):
                    if j:
                        sleep(spec["gaps_ms"][j-1] / 1000)
                        # Never run into the next slot: drop the run's remaining gestures instead.
                        if clock() + gesture["times_ms"][-1] / 1000 + RETURN_BUDGET_S + GUARD_S > following:
                            events.append(dict(kind=GESTURE_EVENT, run=spec["number"], stroke=j, skipped=True,
                                               reason="would overrun the next slot"))
                            break
                    call(GESTURE_EVENT, payloads[spec["number"], j], direct=True, run=spec["number"], stroke=j,
                         slot_s=spec["slot_s"], lateness_s=lateness if j == 0 else None)
            if clock() >= following:
                raise RuntimeError("execution overran the next slot")
        sleep(max(0., start_call + STOP_S - clock()))
        guard()
    except BaseException as exc:
        failures.append(dict(phase="execution", error=f"{type(exc).__name__}: {exc}"))
    finally:
        if started:
            try:
                result = recorder.stop_and_save(out/"original.mov")
                video = {k: str(v) if isinstance(v, Path) else v for k, v in asdict(result).items()}
            except BaseException as exc:
                failures.append(dict(phase="recorder_stop", error=f"{type(exc).__name__}: {exc}"))
        if timing.active:
            try:
                report = timing.stop()
                save_json(out/"wda-timing.json", report)
            except BaseException as exc:
                failures.append(dict(phase="timing_stop", error=f"{type(exc).__name__}: {exc}"))
        if not failures:
            try:
                validate_action_timing_report(report, expected_revision=revision,
                                              expected_count=sum(e.get("instrumented", False) for e in events))
            except BaseException as exc:
                failures.append(dict(phase="timing_validation", error=f"{type(exc).__name__}: {exc}"))
        error = failures[0]["error"] if failures else None
        save_json(out/"execution.json", dict(events=events, video=video, failures=failures, error=error))
    if error:
        raise RuntimeError(error)
    return events
