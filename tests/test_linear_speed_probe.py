import json
import pytest
from trueskate_ai.research.linear_speed_probe import manifest, verify_manifest, run_recording
from test_curve_probe import Recorder, Timing
from trueskate_ai.collection.scene_settle import SettleResult


def test_deadline_wait_survives_coalesced_long_sleep(monkeypatch):
    from trueskate_ai.research import linear_speed_probe as module
    now = [0.]
    waits = []
    def sleep(seconds):
        waits.append(seconds)
        now[0] += seconds + (.146 if seconds > .1 else .001)
    monkeypatch.setattr(module.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(module.time, 'sleep', sleep)
    module.deadline_sleep(1.)
    assert waits[0] == .75
    assert 1. <= now[0] < 1.1


def test_frozen_payload_and_geometry():
    frozen = manifest()
    verify_manifest(json.loads(json.dumps(frozen)))
    for repeat, commands in enumerate(frozen['recordings']):
        drags = [c for c in commands if c['kind'] == 'diagnostic']
        assert [c['duration_ms'] for c in drags] == ([600,400,300,200,100,50,20,10] if repeat == 0 else [10,20,50,100,200,300,400,600])
        for c in drags:
            a = c['payload']['actions'][0]['actions']
            assert [x['type'] for x in a] == ['pointerMove','pointerDown','pointerMove','pointerUp']
            assert a[0]['duration'] == 0 and a[2]['duration'] == c['duration_ms']
            assert (a[0]['x'], a[0]['y']) == (int(.27*414), int(.78*896))
            assert (a[2]['x'], a[2]['y']) == (int(.91*414), int(.30*896))
        assert len(commands) == 19
    frozen['recordings'][0][2]['duration_ms'] = 601
    with pytest.raises(ValueError):
        verify_manifest(frozen)


@pytest.mark.parametrize('mode', ['success','settle','preparation','execution','start','timing','sleep'])
def test_schedule_failure_retention(tmp_path, mode):
    now = [0.]
    recorder, timing = Recorder(), Timing()
    recorder.fail_start = mode == 'start'
    timing.missing = mode == 'timing'
    calls = []
    def perform(spec):
        calls.append((spec, now[0]))
        timing.add(now[0])
        now[0] += 4. if mode == 'execution' and spec['kind'] == 'diagnostic' else .2
    def settle(limit):
        now[0] += 4. if mode == 'preparation' else .6
        return SettleResult(mode != 'settle', .6)
    kwargs = dict(recorder=recorder, timing=timing, commands=manifest()['recordings'][0],
                  perform=perform, guard=lambda: None, settle=settle, out=tmp_path/'run',
                  revision='test', metadata={}, clock=lambda: now[0],
                  sleep=lambda s: now.__setitem__(0,now[0]+s+(.2 if mode == 'sleep' else 0)), epoch=lambda: 1000+now[0])
    if mode == 'success':
        run_recording(**kwargs)
        assert len(calls) == 19 and now[0] == 59.
        for i, (spec, when) in enumerate(calls):
            assert when == spec['slot_s']
            if spec['kind'] == 'diagnostic':
                assert calls[i-1][0]['kind'] == 'reset'
                assert when - calls[i-1][1] == 3.
    else:
        with pytest.raises(RuntimeError):
            run_recording(**kwargs)
        execution = json.loads((tmp_path/'run'/'execution.json').read_text())
        assert execution['error']
        assert recorder.starts == 1
        assert recorder.stops == (0 if mode == 'start' else 1)
        if mode in ('settle', 'preparation'):
            assert not any(s['kind'] == 'diagnostic' for s,_ in calls)
        if mode == 'sleep':
            assert not calls and execution['schedule'][0]['lateness_s'] == pytest.approx(.2)
