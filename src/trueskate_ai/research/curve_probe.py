"""One-minute fail-closed diagnostic execution; no collection jobs or retries."""
from __future__ import annotations
from dataclasses import asdict
import time
from pathlib import Path
from trueskate_ai.research.curve_protocol import PROTOCOL,save_new,equivalent_commands,digest
from trueskate_ai.sim.cubic_curve import CubicInTime,compile_curve,curve_pointer
from trueskate_ai.sim.touch_actions import make_touch_pointer
from trueskate_ai.collection.wda_action_timing import validate_action_timing_report


def scheduled_commands(rows):
    if not 1 <= len(rows) <= 8:
        raise ValueError('one to eight diagnostics per recording required')
    slots=PROTOCOL['diagnostic_slots_s']
    # Keep diagnostics on both sides of the independent middle control.
    chosen=slots[:(len(rows)+1)//2]+slots[4:4+len(rows)//2]
    commands=[dict(kind='control',role=role,slot_s=slot) for role,slot in [('start',1.),('middle',30.),('end',57.)]]
    commands += [dict(kind='diagnostic',slot_s=slot,**row) for slot,row in zip(chosen,rows)]
    return sorted(commands,key=lambda r:r['slot_s'])


def pointer_for(spec,device_size):
    if spec['kind']=='diagnostic':
        kwargs={'n':spec['requested_n']} if spec['requested_n'] is not None else {'max_segment_duration_ms':spec['rule_ms']}
        compiled=compile_curve(CubicInTime.from_dict(spec['curve']),device_size=device_size,**kwargs)
        if not equivalent_commands(compiled.to_dict(),spec['compiled']):
            raise ValueError('compiled command differs from frozen manifest')
        return curve_pointer(compiled)
    finger=make_touch_pointer('control')
    finger.create_pointer_move(x=round(.5*device_size[0]),y=round(.5*device_size[1]),duration=0)
    finger.create_pointer_down()
    finger.create_pause(.05)
    finger.create_pointer_up(0)
    return finger


def command_payload(spec, device_size=(414,896)):
    finger = pointer_for(spec, device_size)
    finger.name = spec.get('command_id', 'control-' + spec.get('role', 'unknown'))
    return {'actions': [finger.encode()]}


def run_recording(*,recorder,timing,commands,perform,guard,out,revision,metadata,
                  clock=time.monotonic,sleep=time.sleep,epoch=time.time):
    """Always retain partial evidence and stop after any single failed attempt."""
    metadata=dict(metadata,wda_revision=revision)
    out=Path(out)
    out.mkdir(parents=True,exist_ok=False)
    events=[]; timing_report=None; video=None; error=None; started=False
    save_new(out/'planned.json',dict(**metadata,commands=commands,training_admission=False))
    try:
        guard()
        timing.start()
        recorder.start() # Exactly one attempt. No automatic replacement.
        started=True
        origin=clock()
        for spec in commands:
            guard()
            target=origin+spec['slot_s']
            now=clock()
            if now>target+PROTOCOL['late_tolerance_s']:
                raise RuntimeError('Late command; schedule overrun')
            sleep(max(0,target-now))
            payload=command_payload(spec,metadata.get('device_size',(414,896)))
            guard()
            if clock()>target+PROTOCOL['late_tolerance_s']:
                raise RuntimeError('Late command after foreground guard; schedule overrun')
            event=dict(spec=spec,payload=payload,payload_sha256=digest(payload),call_start_monotonic_s=clock(),call_start_epoch_s=epoch())
            events.append(event)
            response=perform(spec)
            event.update(response=response,success=True,call_end_monotonic_s=clock(),call_end_epoch_s=epoch())
            if clock()>origin+59.:
                raise RuntimeError('Recording schedule overrun')
            guard()
        sleep(max(0,origin+59.-clock()))
        if clock()>origin+60.:
            raise RuntimeError('Recording schedule overrun')
    except (Exception,KeyboardInterrupt) as exc:
        error=f'{type(exc).__name__}: {exc}'
    finally:
        if started:
            try:
                result=recorder.stop_and_save(out/'original.mov')
                video={k:str(v) if isinstance(v,Path) else v for k,v in asdict(result).items()}
            except Exception as exc:
                error=error or f'recorder stop failed: {exc}'
        if timing.active:
            try:
                timing_report=timing.stop()
                save_new(out/'wda-timing.json',timing_report)
            except Exception as exc:
                error=error or f'timing stop failed: {exc}'
        timing.cleanup()
        # A failed retrieval leaves recorder state unknown; never issue a second stop.
        if not error:
            try:
                validate_action_timing_report(timing_report,expected_revision=revision,expected_count=len(commands))
            except Exception as exc:
                error=f'timing validation failed: {exc}'
        save_new(out/'execution.json',dict(**metadata,events=events,video=video,error=error,
                                         execution_schema='research-execution-v2',training_admission=False))
    if error:
        raise RuntimeError(error)
    return video,timing_report
