"""Frozen two-minute linear drag diagnostic; never admits training examples."""
from __future__ import annotations

from dataclasses import asdict
import time
from pathlib import Path

from trueskate_ai.data.control_hitboxes import segment_is_safe, CONTROL_START_MAP_VERSION
from trueskate_ai.research.curve_protocol import digest, save_new
from trueskate_ai.sim.touch_actions import make_touch_pointer
from trueskate_ai.collection.wda_action_timing import validate_action_timing_report

EXPERIMENT = 'LINEAR-SPEED-20261002'
PATH = ((.27, .78), (.91, .30))
DURATIONS_MS = (600, 400, 300, 200, 100, 50, 20, 10)
SLOTS = (6., 12., 18., 24., 36., 42., 48., 54.)
LATE_S = .1


def deadline_sleep(seconds):
    """Wake early, then approach the deadline with short sleeps.

    macOS can coalesce a single long sleep beyond the 100 ms schedule limit.
    The caller still measures and rejects any actual overrun.
    """
    target = time.monotonic() + seconds
    while (remaining := target-time.monotonic()) > 0:
        time.sleep(remaining-.25 if remaining > .26 else min(.002, remaining))


def payload(spec):
    finger = make_touch_pointer('linear_speed')
    finger.name = 'linear_speed'  # Stable ID: freeze payload bytes before execution.
    point = PATH[0] if spec['kind'] == 'diagnostic' else (
        (.5, .0558) if spec['kind'] == 'reset' else (.5, .5))
    finger.create_pointer_move(x=point[0]*414, y=point[1]*896, duration=0)
    finger.create_pointer_down()
    if spec['kind'] == 'diagnostic':
        finger.create_pointer_move(x=PATH[1][0]*414, y=PATH[1][1]*896,
                                   duration=spec['duration_ms'])
    else:
        finger.create_pause(.05)
    finger.create_pointer_up(0)
    return {'actions': [finger.encode()]}


def manifest(profile="broad"):
    if profile not in ("broad", "fine"):
        raise ValueError("unknown linear speed profile")
    durations_ms = DURATIONS_MS if profile == "broad" else (50,45,40,35,30,25,20)
    slots = SLOTS if profile == "broad" else SLOTS[:-1]
    experiment = EXPERIMENT if profile == "broad" else "LINEAR-SPEED-FINE-20261002"
    if not segment_is_safe(*PATH):
        raise ValueError('whole drag intersects protected controls')
    recordings = []
    for repeat, durations in enumerate((durations_ms, durations_ms[::-1]), 1):
        commands = [dict(kind='control', role=role, slot_s=slot)
                    for role, slot in (('start', 1.), ('middle', 30.), ('end', 57.))]
        for slot, duration in zip(slots, durations):
            commands += [dict(kind='reset', slot_s=slot-3., for_duration_ms=duration),
                         dict(kind='diagnostic', slot_s=slot, duration_ms=duration,
                              repeat=repeat, command_id=f'r{repeat}-{duration}ms')]
        commands.sort(key=lambda c: c['slot_s'])
        for command in commands:
            command['payload'] = payload(command)
        recordings.append(commands)
    result = dict(experiment=experiment, device='iPhone_XR', park='Inbound',
                  path=PATH, durations_ms=durations_ms, recordings=recordings,
                  control_map_version=CONTROL_START_MAP_VERSION, whole_path_safe=True,
                  preparation_lead_s=3., late_tolerance_s=LATE_S, stop_s=59.,
                  settle_threshold=2., settle_consecutive=2,
                  training_admission=False)
    result['sha256'] = digest(result)
    return result


def verify_manifest(value):
    expected = manifest("fine" if value.get("experiment") == "LINEAR-SPEED-FINE-20261002" else "broad")
    if digest(value) != digest(expected):
        raise ValueError('frozen manifest changed')


def run_recording(*, recorder, timing, commands, perform, guard, settle, out,
                  revision, metadata, clock=time.monotonic, sleep=deadline_sleep,
                  epoch=time.time):
    """Reset at slot−3, settle by slot; stop on first failure and retain evidence."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    save_new(out/'planned.json', dict(**metadata, commands=commands, training_admission=False))
    events = []
    schedule = []
    video = None
    error = None
    started = False
    try:
        guard()
        timing.start()
        recorder.start()  # One attempt; no replacement or service restart.
        started = True
        origin = clock()
        for index, spec in enumerate(commands):
            target = origin + spec['slot_s']
            if clock() > target + LATE_S:
                raise RuntimeError('preparation/execution schedule overrun')
            sleep(max(0., target-clock()))
            woke = clock()
            schedule.append(dict(kind=spec['kind'], slot_s=spec['slot_s'],
                                 woke_monotonic_s=woke, lateness_s=woke-target))
            if woke > target + LATE_S:
                raise RuntimeError('sleep overrun')
            event = dict(spec=spec, call_start_monotonic_s=clock(), call_start_epoch_s=epoch())
            events.append(event)
            perform(spec)
            event.update(call_end_monotonic_s=clock(), call_end_epoch_s=epoch())
            deadline = origin + (commands[index+1]['slot_s'] if index+1 < len(commands) else 59.)
            if clock() >= deadline:
                raise RuntimeError('execution overrun')
            if spec['kind'] == 'reset':
                # Reserve time for the foreground/gameplay guard after settling.
                remaining = deadline-clock()
                result = settle(max(0., remaining-.3))
                event['settle'] = result.summary()
                if not result.settled:
                    raise RuntimeError('reset did not settle within preparation window')
            guard()
            if clock() >= deadline:
                raise RuntimeError('preparation/execution overrun after guard')
        sleep(max(0., origin+59.-clock()))
        if clock() > origin+60.:
            raise RuntimeError('one-minute recording overrun')
    except (Exception, KeyboardInterrupt) as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        if started:
            try:
                result = recorder.stop_and_save(out/'original.mov')
                video = {k: str(v) if isinstance(v, Path) else v for k,v in asdict(result).items()}
            except Exception as exc:
                error = error or f'recorder stop failed: {exc}'
        report = None
        if timing.active:
            try:
                report = timing.stop()
                save_new(out/'wda-timing.json', report)
            except Exception as exc:
                error = error or f'timing stop failed: {exc}'
        timing.cleanup()
        # Do not issue a second stop RPC after a failed retrieval; preserve state
        # for the documented recorder recovery procedure.
        if not error:
            try:
                validate_action_timing_report(report, expected_revision=revision,
                                             expected_count=len(commands))
            except Exception as exc:
                error = f'timing validation failed: {exc}'
        save_new(out/'execution.json', dict(**metadata, events=events, video=video,
                                          schedule=schedule, error=error, training_admission=False))
    if error:
        raise RuntimeError(error)
    return video, report
