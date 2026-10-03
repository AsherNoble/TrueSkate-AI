"""Bounded recording lifecycle adapted from the 135-clip audit."""
from dataclasses import asdict
from pathlib import Path
import time
from trueskate_ai.research.curved_audit import save_new
from trueskate_ai.collection.wda_action_timing import validate_action_timing_report
LATE_S=.1

def deadline_sleep(seconds):
    """Wake early, then approach the deadline with short sleeps.

    macOS can coalesce a single long sleep beyond the 100 ms schedule limit.
    The caller still measures and rejects any actual overrun.
    """
    target = time.monotonic() + seconds
    while (remaining := target-time.monotonic()) > 0:
        time.sleep(remaining-.25 if remaining > .26 else min(.002, remaining))


def run_recording(*, recorder, timing, commands, perform, guard, out,
                  revision, metadata, clock=time.monotonic, sleep=deadline_sleep,
                  epoch=time.time):
    """Run one fixed minute; stop on first failure and retain complete/partial evidence."""
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
