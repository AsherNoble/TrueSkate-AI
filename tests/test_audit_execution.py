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
    commands=[dict(kind='sample',slot_s=5),dict(kind='sample',slot_s=54)]
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
