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
