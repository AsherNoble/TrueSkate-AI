"""Offline replay lifecycle regressions; no device or recording fixtures."""
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/collection/replay_demo_clip.py'
spec=importlib.util.spec_from_file_location('demo_replay',SCRIPT)
replay=importlib.util.module_from_spec(spec);spec.loader.exec_module(replay)


class Clock:
    now=0.
    def __call__(self):return self.now
    def sleep(self,s):self.now+=s


@dataclass
class Video:
    mov_path:Path


class Recorder:
    starts=0
    stops=0
    def start(self):self.starts+=1
    def stop_and_save(self,path):self.stops+=1;path.write_bytes(b'partial');return Video(path)


def requests():
    return replay.gesture_requests({'push':[(0.,.5,.5),(.1,.51,.52)],
                                    'flick':[(.5,.6,.6),(.6,.61,.62)]},None)


@pytest.mark.parametrize('mode,schedule_mode', [('separate','paths'),('scheduled','paths'),('scheduled','records')])
def test_valid_modes_retain_exact_payloads_and_responses(tmp_path,mode,schedule_mode):
    clock=Clock();rec=Recorder();posts=[]
    def post(path,payload,*,timeout_s):
        posts.append((path,payload));assert 0<timeout_s<=10
        clock.sleep(.2)
        return {'value':dict(complete=True,completed_s=.8,gestures=[dict(submitted_s=i*.5,completed_s=i*.5+.2) for i in range(2)])}
    result=replay.replay_trial(recorder=rec,requests=requests(),mode=mode,schedule_mode=schedule_mode,anchor=None,
        guard=lambda:None,post=post,out=tmp_path/'trial',metadata={},clock=clock,sleep=clock.sleep)
    assert result['error'] is None and rec.starts==rec.stops==1
    assert len(posts)==(2 if mode=='separate' else 1)
    assert all(c['success'] and c['response'] for c in result['calls'])
    assert (tmp_path/'trial/original.mov').read_bytes()==b'partial'
    assert json.loads((tmp_path/'trial/execution.json').read_text())['calls']==result['calls']
    if mode=='scheduled' and schedule_mode=='paths':
        assert all(t['delivery_measured'] is False and 'start_s' not in t for t in result['gesture_timings'])


@pytest.mark.parametrize('failure', ['foreground','http','incomplete','interrupt','exit','stop','multiple','sleep_overrun'])
def test_every_post_start_failure_retrieves_partial_once(tmp_path,failure):
    clock=Clock();rec=Recorder();guards=0;posts=[]
    def guard():
        nonlocal guards
        guards+=1
        if guards==2 and failure=='foreground':raise RuntimeError('foreground lost during lead-in')
    def post(path,payload,**kw):
        posts.append(payload)
        if failure in ('http','multiple'):raise OSError('HTTP failure')
        if failure=='interrupt':raise KeyboardInterrupt('cancelled')
        if failure=='exit':raise SystemExit('cancelled')
        return {'value':dict(complete=failure!='incomplete',error='partial schedule' if failure=='incomplete' else None,completed_s=.8)}
    if failure in ('stop','multiple'):
        def stop(path):rec.stops+=1;path.write_bytes(b'partial');raise OSError('stop failed')
        rec.stop_and_save=stop
    def sleep(seconds):clock.sleep(70 if failure=='sleep_overrun' else seconds)
    result=replay.replay_trial(recorder=rec,requests=requests(),mode='scheduled',schedule_mode='paths',anchor=None,
        guard=guard,post=post,out=tmp_path/'trial',metadata={},clock=clock,sleep=sleep)
    assert result['error'] and rec.starts==rec.stops==1
    saved=json.loads((tmp_path/'trial/execution.json').read_text())
    assert saved['failures'] and (tmp_path/'trial/original.mov').read_bytes()==b'partial'
    if failure in ('foreground','sleep_overrun'):assert not posts
    if failure=='incomplete':assert saved['schedule_response']['value']['error']=='partial schedule'
    if failure=='multiple':assert [f['phase'] for f in saved['failures']]==['execution','recorder_stop']


def test_start_failure_and_sparse_overlong_schedule_never_retry(tmp_path):
    rec=Recorder()
    def start():rec.starts+=1;raise OSError('start failed')
    rec.start=start
    args=dict(recorder=rec,requests=requests(),mode='scheduled',schedule_mode='paths',anchor=None,
              guard=lambda:None,post=lambda *a,**k:None,metadata={})
    result=replay.replay_trial(**args,out=tmp_path/'trial')
    assert result['error'] and rec.starts==1 and rec.stops==0
    args['requests']=[(name,1000+start,payload) for name,start,payload in requests()]
    with pytest.raises(ValueError,match='budget'):replay.replay_trial(**args,out=tmp_path/'overlong')
    assert rec.starts==1 and not (tmp_path/'overlong').exists()


def test_recorder_start_latency_consumes_deadline_before_playback(tmp_path):
    clock=Clock();rec=Recorder();posts=[]
    def start():rec.starts+=1;clock.sleep(58)
    rec.start=start
    result=replay.replay_trial(recorder=rec,requests=requests(),mode='scheduled',schedule_mode='paths',anchor=None,
        guard=lambda:None,post=lambda *a,**k:posts.append(a),out=tmp_path/'trial',metadata={},clock=clock,sleep=clock.sleep)
    assert result['error'] and not posts and rec.starts==rec.stops==1 and clock.now==59


def test_anchor_preserves_indexed_paths_and_overlapping_contacts():
    result=replay.schedule_payload(requests(),'paths',(.96,.60))
    assert result['mode']=='paths' and len(result['gestures'])==3
    assert [g['start_ms'] for g in result['gestures']]==[100,600,0]
    anchor=result['gestures'][-1]['waypoints']
    assert anchor[0]['x']==round(.96*414) and anchor[-1]['duration_ms']==800
    with pytest.raises(ValueError):replay.schedule_payload(requests(),'records',(.96,.60))


@pytest.mark.parametrize('rows', [[],[(0,.5,.5)],[(0,.5,.5),(0,.6,.6)],[(0,.5,.5),(float('inf'),.6,.6)],
                                   [(0,.5,.5),(.1,-.1,.6)]])
def test_invalid_source_samples_rejected(rows):
    with pytest.raises(ValueError):replay.gesture_requests({'gesture':rows},None)


def test_failure_evidence_written_before_disconnect(tmp_path,monkeypatch):
    out=tmp_path/'batch';gestures=tmp_path/'gestures.json'
    gestures.write_text(json.dumps({'flick':[dict(t=0.,x=.5,y=.5),dict(t=.1,x=.51,y=.52)]}))
    monkeypatch.setattr(sys,'argv',[str(SCRIPT),'--gestures',str(gestures),'--out',str(out),'--repeats','1','--variants','native'])
    monkeypatch.setattr(replay.socket,'gethostname',lambda:'training-server')
    monkeypatch.setattr(replay.subprocess,'check_output',lambda *a,**k:'state = running')
    monkeypatch.setattr(replay.time,'sleep',lambda s:None)
    driver=SimpleNamespace(query_app_state=lambda b:4,execute=lambda *a:None)
    class Worker:
        def __init__(self,cfg):self.driver=driver
        def connect(self):pass
        def _active_bundle_id(self):return 'game'
        def disconnect(self):
            assert json.loads((out/'trial_01/execution.json').read_text())['error']
            assert json.loads((out/'runs.json').read_text())['runs'][0]['schedule_response']['value']['error']=='failed'
    monkeypatch.setitem(sys.modules,'trueskate_ai.sim.device',SimpleNamespace(DeviceSession=Worker,DEVICES=[dict(name='iPhone_XR2',wda_port=8103)],BUNDLE_ID='game'))
    import trueskate_ai.collection.wda_action_timing as timing
    monkeypatch.setattr(timing,'_http_json',lambda url:dict(value=dict(bundleId='game')) if url.endswith('activeAppInfo') else dict(value=None))
    monkeypatch.setitem(sys.modules,'trueskate_ai.collection.xctest_capture',SimpleNamespace(XCTestScreenRecorder=lambda *a,**k:Recorder()))
    monkeypatch.setattr(replay,'request_json',lambda *a,**k:dict(value=dict(complete=False,error='failed')))
    real=replay.replay_trial;clock=Clock()
    monkeypatch.setattr(replay,'replay_trial',lambda **kw:real(**kw,clock=clock,sleep=clock.sleep))
    with pytest.raises(RuntimeError,match='incomplete'):replay.main()
