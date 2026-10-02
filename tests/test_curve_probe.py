from dataclasses import dataclass
import pytest
from trueskate_ai.research.curve_probe import run_recording,scheduled_commands
from trueskate_ai.research.curve_protocol import command_manifest
from trueskate_ai.collection.wda_action_timing import BOUNDARIES

@dataclass
class Video:
    mov_path:str='original.mov'

class Recorder:
    is_recording=False
    starts=0;stops=0;aborts=0
    fail_start=False;fail_stop=False
    def start(self):
        self.starts+=1
        if self.fail_start:raise RuntimeError('failed recording')
        self.is_recording=True
    def stop_and_save(self,path):
        self.stops+=1
        if self.fail_stop:raise RuntimeError('failed stop')
        self.is_recording=False
        return Video(str(path))
    def abort(self):self.aborts+=1;self.is_recording=False

class Timing:
    active=False
    missing=False
    def __init__(self):self.records=[]
    def start(self):self.active=True
    def stop(self):
        self.active=False
        return dict(schema_version=1,build_revision='test',dropped_records=0,records=self.records[:-1] if self.missing else self.records)
    def cleanup(self):self.active=False
    def add(self,now):
        row=dict(sequence=len(self.records),outcome='success',session_id='test',missing_ios_callback=False,ios_callback_result=True)
        row.update({name:dict(monotonic_s=now+i*.001,epoch_s=1000+now+i*.001) for i,name in enumerate(BOUNDARIES)})
        self.records.append(row)


def harness(tmp_path,mode='success'):
    now=[0.];recorder=Recorder();timing=Timing();calls=[]
    recorder.fail_start=mode=='start';recorder.fail_stop=mode=='stop';timing.missing=mode=='timing'
    def perform(spec):
        calls.append(spec);timing.add(now[0]);now[0]+=10 if mode=='late' else .3
    def guard():
        if mode=='foreground' and calls:raise RuntimeError('foreground lost')
    def sleep(d):now[0]+=d
    commands=scheduled_commands(command_manifest()['devices']['iPhone_XR']['pilot'][:8])
    kwargs=dict(recorder=recorder,timing=timing,commands=commands,perform=perform,guard=guard,out=tmp_path/'recording',
                revision='test',metadata={},clock=lambda:now[0],sleep=sleep,epoch=lambda:1000+now[0])
    return kwargs,recorder,timing,calls


def test_bounded_schedule(tmp_path):
    kwargs,recorder,timing,calls=harness(tmp_path)
    run_recording(**kwargs)
    assert len(calls)==11 and recorder.starts==recorder.stops==1 and recorder.aborts==0
    assert [c['slot_s'] for c in calls]==[1,6,12,18,24,30,36,42,48,54,57]


@pytest.mark.parametrize('mode',['late','start','stop','timing','foreground'])
def test_failure_is_preserved_cleaned_and_never_replaced(tmp_path,mode):
    kwargs,recorder,timing,calls=harness(tmp_path,mode)
    with pytest.raises((RuntimeError,ValueError)):run_recording(**kwargs)
    assert recorder.starts==1 and not recorder.is_recording and not timing.active
    assert (tmp_path/'recording'/'execution.json').exists()
    if mode=='start':assert not calls and recorder.stops==0
    else:assert recorder.stops==1
    if mode=='stop':assert recorder.aborts==1


def test_partial_recordings_keep_spread_controls():
    rows=command_manifest()['devices']['iPhone_XR']['pilot']
    for n in range(1,9):
        specs=scheduled_commands(rows[:n]);assert len(specs)==n+3
        assert [s['role'] for s in specs if s['kind']=='control']==['start','middle','end']
