import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('probe', Path(__file__).resolve().parents[1] / 'scripts/collection/probe_gesture_timing.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_measured_sequence_contains_only_six_calls_and_seven_waits():
    operations = []
    now = [0.0]
    class Command:
        def perform(self):
            operations.append('touch')
            now[0] += 1.5
    def sleep(duration):
        assert duration == 1.0
        operations.append('wait')
        now[0] += duration
    events, waits = probe.run_sequence([Command() for _ in range(6)], sleep=sleep,
                                       epoch=lambda: 1000 + now[0], monotonic=lambda: now[0])
    assert operations == ['wait'] + ['touch', 'wait'] * 6
    assert waits == [1.0] * 7
    assert len(events) == 6
    assert all(e['t_call_end_monotonic_s'] - e['t_call_start_monotonic_s'] == 1.5 for e in events)


def test_duration_repeat_has_three_controls_and_two_duration_cycles():
    specs = probe.sequence('duration-repeat')
    assert len(specs) == 9
    assert [s['duration'] for s in specs] == [.05, .05, .05, .3, .6, 1., .3, .6, 1.]
    calls = []
    class Command:
        def perform(self):
            calls.append('touch')
    events, waits = probe.run_sequence([Command() for _ in specs],
        sleep=lambda d: calls.append(('wait', d)), epoch=lambda: 0., monotonic=lambda: 0.)
    assert calls == [('wait', 1.)] + ['touch', ('wait', 1.)] * 9
    assert len(events) == 9 and len(waits) == 10


def test_bundle_sends_one_request_with_released_pauses_and_exact_schedule():
    import pytest
    class Driver:
        def __init__(self): self.calls = []
        def execute(self, command, params): self.calls.append((command, params))
    driver = Driver()
    command, starts, duration = probe.build_bundle(driver, probe.sequence('duration-repeat'))
    command.perform()
    assert len(driver.calls) == 1
    sources = driver.calls[0][1]['actions']
    assert len(sources) == 1 and sources[0]['parameters']['pointerType'] == 'touch'
    elapsed = 0; down = False; observed = []; gaps = []
    for action in sources[0]['actions']:
        if action['type'] == 'pointerDown':
            assert not down
            down = True; observed.append(elapsed / 1000)
        elif action['type'] == 'pointerUp':
            assert down
            down = False
        elif action['type'] == 'pause' and not down:
            gaps.append(action['duration'])
        elapsed += action.get('duration', 0)
    assert not down and gaps == [1000] * 10
    assert observed == pytest.approx([1, 2.05, 3.1, 4.15, 5.45, 7.05, 9.05, 10.35, 11.95])
    assert starts == pytest.approx(observed)
    assert duration == pytest.approx(13.95) == elapsed / 1000


def test_wda_timing_validation_rejects_incomplete_or_misattributed_records():
    import copy
    import pytest
    boundaries = ('request_entered', 'preparation_started', 'preparation_finished',
                  'submitted_to_ios', 'ios_completion_callback', 'stability_wait_started',
                  'stability_wait_finished', 'request_finished')
    record = dict(sequence=0, outcome='success', session_id='session-a',
                  missing_ios_callback=False, ios_callback_result=True)
    record.update({name: {'monotonic_s': float(i)} for i, name in enumerate(boundaries)})
    report = dict(schema_version=1, build_revision='abc', dropped_records=0, records=[record])
    assert probe.validate_wda_timings(report, 'abc', 1)
    for key, value in [('sequence', 1), ('outcome', 'error'), ('missing_ios_callback', True),
                       ('ios_callback_result', False), ('session_id', '')]:
        bad = copy.deepcopy(report); bad['records'][0][key] = value
        with pytest.raises(ValueError): probe.validate_wda_timings(bad, 'abc', 1)
    bad = copy.deepcopy(report); bad['records'][0]['submitted_to_ios']['monotonic_s'] = 99
    with pytest.raises(ValueError): probe.validate_wda_timings(bad, 'abc', 1)
    with pytest.raises(ValueError): probe.validate_wda_timings(report, 'wrong', 1)
    with pytest.raises(ValueError): probe.validate_wda_timings(report, 'abc', 2)
    bad = copy.deepcopy(report); bad['dropped_records'] = 1
    with pytest.raises(ValueError): probe.validate_wda_timings(bad, 'abc', 1)


def test_minute_schedule_keeps_separate_calls_and_spread_anchors():
    import pytest
    specs = probe.random_sequence(8, 1010, 3, True)
    assert [i for i,s in enumerate(specs) if s['kind']=='calibration'] == [0,5,10]
    now = [0.]; calls = []
    class Command:
        def perform(self): calls.append(now[0]); now[0] += 1.8
    def sleep(d): now[0] += d
    events, waits = probe.run_minute_sequence([Command() for _ in specs], sleep=sleep,
        epoch=lambda: now[0]+1000, monotonic=lambda: now[0])
    assert calls == pytest.approx(probe.MINUTE_SLOTS)
    assert len(events)==11 and len(waits)==12 and now[0]<60
    now[0]=0
    class Slow:
        def perform(self): now[0]+=10
    with pytest.raises(RuntimeError, match='Late command'):
        probe.run_minute_sequence([Slow() for _ in specs], sleep=sleep,
            epoch=lambda:now[0], monotonic=lambda:now[0])
