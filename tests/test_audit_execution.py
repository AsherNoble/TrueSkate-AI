import json
from dataclasses import dataclass
from pathlib import Path
import pytest
import trueskate_ai.research.audit_execution as execution

class Clock:
    now=0.
    def __call__(self):return self.now
    def sleep(self,s):self.now+=s
class Timing:
    active=False
    def start(self):self.active=True
    def stop(self):self.active=False;return {}
    def cleanup(self):pass
@dataclass
class Video:
    mov_path:Path
class Recorder:
    starts=0
    stops=0
    def start(self):self.starts+=1
    def stop_and_save(self,p):self.stops+=1;p.write_bytes(b'partial');return Video(p)

@pytest.mark.parametrize('fail',[False,True])
def test_bounded_lifecycle_preserves_partial(tmp_path,monkeypatch,fail):
    clock=Clock();rec=Recorder();calls=[]
    monkeypatch.setattr(execution,'validate_action_timing_report',lambda *a,**k:None)
    def perform(spec):
        calls.append(clock())
        if fail:raise ValueError('execution failure')
        clock.sleep(.1)
    commands=[dict(kind='sample',slot_s=5,payload={'actions':[]}),dict(kind='sample',slot_s=54,payload={'actions':[]})]
    args=dict(recorder=rec,timing=Timing(),commands=commands,perform=perform,guard=lambda:None,
              out=tmp_path/'run',revision='fixture',metadata={},clock=clock,sleep=clock.sleep,epoch=clock)
    if fail:
        with pytest.raises(RuntimeError):execution.run_recording(**args)
    else:execution.run_recording(**args)
    assert rec.starts==rec.stops==1
    assert (tmp_path/'run/original.mov').read_bytes()==b'partial'
    result=json.loads((tmp_path/'run/execution.json').read_text())
    assert bool(result['error'])==fail
    assert calls==([5] if fail else [5,54])
    assert clock.now==(5 if fail else 59)


@pytest.mark.parametrize('mode', ['foreground', 'late_guard', 'interrupt', 'exit', 'stop', 'cleanup', 'multiple'])
def test_failed_exit_retrieves_once_and_preserves_all_failures(tmp_path,monkeypatch,mode):
    clock=Clock();rec=Recorder();calls=[];guards=0
    monkeypatch.setattr(execution,'validate_action_timing_report',lambda *a,**k:None)
    class BrokenTiming(Timing):
        def stop(self):
            if mode=='multiple':raise OSError('timing retrieval')
            return super().stop()
        def cleanup(self):
            if mode in ('cleanup','multiple'):raise OSError('cleanup diagnostic')
    if mode in ('stop','multiple'):
        def stop(path):rec.stops+=1;path.write_bytes(b'partial');raise OSError('video retrieval')
        rec.stop_and_save=stop
    def guard():
        nonlocal guards
        guards+=1
        if guards==2 and mode=='foreground':raise RuntimeError('foreground changed during sleep')
        if guards==2 and mode=='late_guard':clock.sleep(.2)
    def perform(spec):
        calls.append(spec)
        if mode in ('interrupt','exit','multiple'):
            raise {'interrupt':KeyboardInterrupt,'exit':SystemExit,'multiple':ValueError}[mode]('primary')
        clock.sleep(.1);return {'value':None}
    with pytest.raises(RuntimeError):
        execution.run_recording(recorder=rec,timing=BrokenTiming(),commands=[dict(kind='sample',slot_s=5,payload={'actions':[]})],
            perform=perform,guard=guard,out=tmp_path/'run',revision='fixture',metadata={},clock=clock,sleep=clock.sleep,epoch=clock)
    assert rec.starts==rec.stops==1
    result=json.loads((tmp_path/'run/execution.json').read_text())
    assert result['error'] and result['failures']
    assert (tmp_path/'run/original.mov').read_bytes()==b'partial'
    if mode in ('foreground','late_guard'):assert not calls and result['events']==[]
    if mode=='multiple':
        assert [f['phase'] for f in result['failures']]==['execution','recorder_stop','timing_stop','timing_cleanup']
        assert 'primary' in result['error']


def test_start_failure_is_not_retried_or_stopped(tmp_path):
    rec=Recorder()
    def start():rec.starts+=1;raise OSError('start rejected')
    rec.start=start
    with pytest.raises(RuntimeError,match='start rejected'):
        execution.run_recording(recorder=rec,timing=Timing(),commands=[dict(kind='sample',slot_s=5,payload={})],
            perform=lambda s:None,guard=lambda:None,out=tmp_path/'run',revision='fixture',metadata={})
    assert rec.starts==1 and rec.stops==0


def test_recorder_start_latency_does_not_extend_minute(tmp_path,monkeypatch):
    clock=Clock();rec=Recorder();calls=[]
    monkeypatch.setattr(execution,'validate_action_timing_report',lambda *a,**k:None)
    def start():rec.starts+=1;clock.sleep(.5)
    rec.start=start
    execution.run_recording(recorder=rec,timing=Timing(),commands=[dict(kind='sample',slot_s=5,payload={})],
        perform=lambda s:calls.append(clock()),guard=lambda:None,out=tmp_path/'run',revision='fixture',metadata={},
        clock=clock,sleep=clock.sleep,epoch=clock)
    assert calls==[5] and clock.now==59 and rec.stops==1


@pytest.mark.parametrize('slots', [[-1],[60],[float('nan')],[5,5],[5,4],[1,100000]])
def test_invalid_schedule_is_rejected_before_recorder_start(tmp_path,slots):
    rec=Recorder()
    with pytest.raises(ValueError):
        execution.run_recording(recorder=rec,timing=Timing(),commands=[dict(kind='sample',slot_s=s,payload={}) for s in slots],
            perform=lambda s:None,guard=lambda:None,out=tmp_path/'run',revision='fixture',metadata={})
    assert rec.starts==rec.stops==0
