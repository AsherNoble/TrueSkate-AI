"""Synthetic diagnostic lifecycle and operator-review gates; no phone/holdouts."""
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import numpy as np

import pytest

from trueskate_ai.research import kickflip_repeatability as module
from trueskate_ai.collection.scene_settle import SettleResult
from trueskate_ai.research.curve_protocol import save_new

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def experiment(tmp_path):
    save_new(tmp_path / 'manifest.json', module.prepare_manifest(REPO))
    save_new(tmp_path / 'context.json', dict(settings_note='synthetic waypoint/settings'))
    return tmp_path


class Clock:
    def __init__(self):
        self.now = 0.
    def __call__(self):
        return self.now
    def sleep(self, seconds):
        assert seconds >= 0
        self.now += seconds


@dataclass
class Video:
    mov_path: Path
    started_at_epoch_s: float = 1000.


class Recorder:
    def __init__(self):
        self.starts = self.stops = 0
    def start(self):
        self.starts += 1
    def stop_and_save(self, path):
        self.stops += 1
        path.write_bytes(b'synthetic movie')
        return Video(path)


class Timing:
    def __init__(self):
        self.active = False
        self.records = []
        self.stops = 0
    def start(self):
        self.active = True
    def stop(self):
        self.active = False
        self.stops += 1
        return dict(records=self.records)


def test_candidates_preserve_push_and_never_bundle_contacts(experiment):
    manifest = module.load_manifest(experiment)
    assert len(manifest['candidates']) == 22 and manifest['rejected'] == []
    first = manifest['candidates'][0]
    assert [c['name'] for c in first['contacts']] == ['push', 'pop', 'flick']
    push = first['contacts'][0]
    assert push['encoded_duration_s'] == .02
    assert push['points'] == [[.7658, .3044], [.7658, .6797]]
    for candidate in manifest['candidates']:
        assert candidate['pop_to_flick_gap_s'] >= .4
        assert candidate['contacts'][0] == push
        for contact in candidate['contacts']:
            assert len(contact['payload']['actions']) == 1
            actions = contact['payload']['actions'][0]['actions']
            assert sum(a['type'] == 'pointerDown' for a in actions) == 1
            assert sum(a['type'] == 'pointerUp' for a in actions) == 1
    first['contacts'][0]['payload']['actions'][0]['actions'][2]['duration'] = 21
    with pytest.raises(ValueError, match='hash'):
        module.verify_seal(first)


@pytest.mark.parametrize('points', [[(.5, .5), (.05, .4)], [(.5, .5), (float('nan'), .6)], [(.5, .5), (1.1, .6)]])
def test_unsafe_geometry_rejected(points):
    with pytest.raises(ValueError):
        module.contact_payload('bad', points, .1)


@pytest.mark.parametrize('failure', [None, 'start', 'slow_start', 'execution', 'interrupt', 'exit',
                                    'foreground', 'settle', 'sleep', 'stop', 'timing'])
def test_trial_retains_one_retrieval_and_all_failures(tmp_path, monkeypatch, failure):
    clock, recorder, timing = Clock(), Recorder(), Timing()
    candidate = module.prepare_manifest(REPO)['candidates'][0]
    calls = []
    def perform(payload):
        calls.append((payload, clock()))
        timing.records.append({'synthetic': True})
        if len(calls) == 3 and failure in ('execution', 'interrupt', 'exit'):
            raise {'execution': OSError, 'interrupt': KeyboardInterrupt, 'exit': SystemExit}[failure]('synthetic failure')
        actions = payload['actions'][0]['actions']
        clock.sleep(sum(a.get('duration', 0) for a in actions)/1000 + .3)
        return {'value': None}
    def guard():
        if failure == 'foreground' and clock() >= 1:
            raise RuntimeError('foreground lost')
    def settle(reserve):
        clock.sleep(.5)
        return SettleResult(failure != 'settle', .5)
    original_start = recorder.start
    def start():
        original_start()
        if failure == 'start':
            raise RuntimeError('start failed')
        if failure == 'slow_start':
            clock.sleep(1.1)
    recorder.start = start
    if failure == 'stop':
        def stop(path):
            recorder.stops += 1
            raise RuntimeError('retrieval failed')
        recorder.stop_and_save = stop
    def validate(report, **kwargs):
        if failure == 'timing':
            raise ValueError('timing incomplete')
        assert len(report['records']) == kwargs['expected_count']
    monkeypatch.setattr(module, 'validate_action_timing_report', validate)
    def sleep(seconds):
        clock.sleep(seconds + (.2 if failure == 'sleep' else 0))
    kwargs = dict(out=tmp_path, candidate=candidate, recorder=recorder, timing=timing,
                  perform=perform, guard=guard, settle=settle, revision='synthetic', context={},
                  clock=clock, sleep=sleep, epoch=lambda: 1000+clock())
    if failure:
        with pytest.raises(RuntimeError):
            module.run_trial(**kwargs)
    else:
        module.run_trial(**kwargs)
        execution = module.read_json(tmp_path / 'execution.json')
        assert len(calls) == 9 and clock() == 59
        gameplay = [e for e in execution['events'] if e['kind'] == 'gesture']
        assert [e['role'] for e in gameplay] == ['push', 'pop', 'flick']
        assert gameplay[1]['call_start_monotonic_s'] - gameplay[0]['call_end_monotonic_s'] == pytest.approx(.48)
        assert gameplay[2]['call_start_monotonic_s'] - gameplay[1]['call_start_monotonic_s'] == pytest.approx(.179+.62)
        assert [e['intended_s'] for e in execution['events'] if e['kind'] == 'control'] == [1, 30, 57]
    saved = module.read_json(tmp_path / 'execution.json')
    assert bool(saved['error']) == bool(failure)
    assert recorder.starts == 1 and recorder.stops == (0 if failure == 'start' else 1)
    assert timing.stops == 1 and not timing.active
    if failure in ('start', 'slow_start', 'foreground', 'sleep'):
        assert not calls


def add_trial(root, index, candidate_id='candidate_01', *, landed=True, assessed=True):
    trial = root / 'setup' / f'trial_{index:02d}'
    trial.mkdir(parents=True)
    (trial / 'original.mov').write_bytes(f'synthetic {index}'.encode())
    sha = module.file_sha(trial / 'original.mov')
    save_new(trial / 'reservation.json', dict(candidate_id=candidate_id))
    save_new(trial / 'admission.json', dict(accepted=True, video_sha256=sha))
    if assessed:
        save_new(trial / 'assessment.json', dict(trick='KICKFLIP', status='landed' if landed else 'failed', video_sha256=sha))
    return trial


def test_attempt_cap_and_failed_start_cannot_be_replaced(experiment):
    for i in range(1, 25):
        add_trial(experiment, i)
    with pytest.raises(ValueError, match='limit'):
        module.reserve_setup(experiment, 'candidate_01')
    trial = experiment / 'setup/trial_24/admission.json'
    trial.unlink()
    assert len(module.setup_trials(experiment)) == 24


def test_incomplete_attempt_stops_setup(experiment):
    trial, candidate = module.reserve_setup(experiment, 'candidate_01')
    assert trial.name == 'trial_01'
    with pytest.raises(ValueError, match='technical failure'):
        module.reserve_setup(experiment, 'candidate_01')


def test_review_cannot_omit_failed_or_unassessed_runs(experiment):
    add_trial(experiment, 1, landed=False)
    add_trial(experiment, 2, landed=False)
    add_trial(experiment, 3)
    add_trial(experiment, 4)
    with pytest.raises(ValueError, match='first three'):
        module.qualified_trials(experiment, 'candidate_01')
    (experiment / 'setup/trial_01/assessment.json').unlink()
    with pytest.raises(ValueError, match='all candidate attempts'):
        module.qualified_trials(experiment, 'candidate_01')


def install_review(root):
    manifest = module.load_manifest(root)
    candidate = manifest['candidates'][0]
    sources = []
    for i in range(1, 4):
        trial = add_trial(root, i, landed=i != 2)
        sources.append(dict(trial=str(trial.relative_to(root)), video_sha256=module.file_sha(trial / 'original.mov')))
    review_dir = root / 'review/candidate_01'
    review_dir.mkdir(parents=True)
    (review_dir / 'preview.mov').write_bytes(b'synthetic preview')
    review = module.seal(dict(candidate_id=candidate['id'], candidate_sha256=candidate['sha256'],
        manifest_sha256=manifest['sha256'], context_sha256=module.digest(module.read_json(root / 'context.json')),
        sources=sources, preview='review/candidate_01/preview.mov', preview_sha256=module.file_sha(review_dir / 'preview.mov')))
    save_new(review_dir / 'review.json', review)


@pytest.mark.parametrize('tamper', [None, 'preview', 'source', 'settings', 'manifest'])
def test_explicit_approval_binds_exact_recipe_context_and_movies(experiment, tamper):
    with pytest.raises(FileNotFoundError):
        module.approved_candidate(experiment)
    install_review(experiment)
    with pytest.raises(ValueError, match='explicit'):
        module.approve_review(experiment, 'candidate_01', ' ')
    module.approve_review(experiment, 'candidate_01', 'Operator: this is satisfactory')
    assert module.approved_candidate(experiment)['id'] == 'candidate_01'
    if tamper == 'preview':
        (experiment / 'review/candidate_01/preview.mov').write_bytes(b'changed')
    elif tamper == 'source':
        (experiment / 'setup/trial_01/original.mov').write_bytes(b'changed')
    elif tamper == 'settings':
        (experiment / 'context.json').write_text(json.dumps(dict(settings_note='changed')))
    elif tamper == 'manifest':
        m = module.load_manifest(experiment)
        m.pop('sha256')
        m['fps'] = 30
        (experiment / 'manifest.json').write_text(json.dumps(module.seal(m)))
    if tamper:
        with pytest.raises(ValueError, match='changed'):
            module.approved_candidate(experiment)
    with pytest.raises(ValueError, match='frozen'):
        module.reserve_setup(experiment, 'candidate_02')


def test_report_counts_failures_and_unassessed_movies_without_faking_completion(experiment):
    for i in (1, 2):
        trial = experiment / 'repeats' / f'trial_{i:02d}'
        trial.mkdir(parents=True)
        if i == 2:
            continue
        (trial / 'original.mov').write_bytes(b'synthetic')
        gestures = [dict(role=role, submitted_monotonic_s=t, encoded_duration_s=.05,
                         ios_callback_latency_s=.35) for role, t in zip(('push', 'pop', 'flick'), (10., 10.8, 11.4))]
        save_new(trial / 'admission.json', dict(accepted=True, video_sha256=module.file_sha(trial / 'original.mov'),
                 gestures=gestures, heldout_middle_error_s=.01, native_frame_s=1/60, max_frame_gap_s=.033))
    result = module.report(experiment)
    assert result['attempted'] == 2 and result['admitted'] == 1 and not result['complete']
    assert result['outcomes'] == {'unassessed': 1}
    assert result['technical_failures'] == ['repeats/trial_02']
    assert result['timing_distributions']['pop_to_flick']['median_s'] == pytest.approx(.6)


@pytest.mark.parametrize('failure', [None, 'dimensions', 'fps', 'pts', 'decode_count', 'control', 'middle'])
def test_admission_requires_native_decode_and_independent_calibration(tmp_path, monkeypatch, failure):
    roles = ['start', None, 'push', 'pop', 'flick', None, 'middle', None, 'end']
    stamps = [1., 3., 8., 8.8, 9.6, 27., 30., 54., 57.]
    boundaries = module.validate_action_timing_report.__globals__['BOUNDARIES']
    records = [dict(sequence=i, outcome='success', session_id='synthetic', missing_ios_callback=False,
                    ios_callback_result=True, **{b: dict(monotonic_s=t+j*.01, epoch_s=1000+t+j*.01)
                    for j, b in enumerate(boundaries)}) for i, t in enumerate(stamps)]
    events = [dict(kind='control' if role in ('start', 'middle', 'end') else 'gesture' if role else 'reset',
                   role=role, encoded_duration_s=.05, lateness_s=0.) for role in roles]
    save_new(tmp_path / 'execution.json', dict(error=None, events=events, video=dict(started_at_epoch_s=1000.)))
    save_new(tmp_path / 'wda-timing.json', dict(schema_version=1, build_revision='synthetic', records=records, dropped_records=0))
    (tmp_path / 'original.mov').write_bytes(b'synthetic')
    pts = np.arange(0, 59, 1/(30 if failure == 'fps' else 60))
    if failure == 'pts':
        pts[1] = pts[0]
    probe = dict(streams=[dict(width=827 if failure == 'dimensions' else 828, height=1792)],
                 frames=[dict(best_effort_timestamp_time=float(t)) for t in pts])
    monkeypatch.setattr(module.subprocess, 'check_output', lambda *a, **k: json.dumps(probe).encode())
    def decode(*args, **kwargs):
        if failure == 'decode_count':
            raise ValueError('decoded frame count differs from original PTS selection')
    monkeypatch.setattr(module, '_decode_source_frames', decode)
    monkeypatch.setattr(module, 'read_native_frames', lambda *a, **k: ([np.zeros((2, 2, 3))], [.0], [0]))
    def detection(*args, command_s, **kwargs):
        if failure == 'control':
            return None
        residual = .05 if failure == 'middle' and 29 < command_s < 31 else 0.
        return SimpleNamespace(onset_s=command_s+residual)
    monkeypatch.setattr(module, 'detect_tap_onset', detection)
    if failure:
        with pytest.raises(ValueError):
            module.admit_trial(tmp_path, 'synthetic')
        assert module.read_json(tmp_path / 'admission.json')['accepted'] is False
    else:
        result = module.admit_trial(tmp_path, 'synthetic')
        assert result['accepted'] and result['heldout_middle_error_s'] == pytest.approx(0.)
        assert [g['role'] for g in result['gestures']] == ['push', 'pop', 'flick']


def load_cli():
    spec = importlib.util.spec_from_file_location('kickflip_cli', REPO / 'scripts/collection/probe_kickflip_repeatability.py')
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


@pytest.mark.parametrize('failure_at', [None, 4])
def test_live_batch_is_exactly_twenty_and_stops_without_replacement(experiment, monkeypatch, failure_at):
    import base64
    import trueskate_ai.sim.device as device
    import trueskate_ai.collection.wda_action_timing as timing
    import trueskate_ai.collection.xctest_capture as capture
    import trueskate_ai.collection.scene_settle as scene
    import trueskate_ai.collection.gameplay_filter as gameplay
    cli = load_cli()
    install_review(experiment)
    module.approve_review(experiment, 'candidate_01', 'Operator approves preview')
    env_file = experiment / 'synthetic.env'
    env_file.write_text('IPHONE_XR_UDID=synthetic\n')
    monkeypatch.setattr(cli.socket, 'gethostname', lambda: 'training-server')
    monkeypatch.setattr(cli.subprocess, 'check_output', lambda *a, **k: 'state = running')
    monkeypatch.setattr(cli.signal, 'signal', lambda *a: None)
    runs, resets, connections = [], [], []
    class Worker:
        def __init__(self, *a, **k):
            self.driver = None
        def connect(self):
            connections.append('connect')
            self.driver = SimpleNamespace(capabilities={'udid': 'synthetic'},
                query_app_state=lambda b: 4, execute=lambda action, payload: resets.append(payload))
        def disconnect(self):
            self.driver = None
        def _active_bundle_id(self):
            return device.BUNDLE_ID
    def http(url):
        if url.endswith('activeAppInfo'):
            return dict(value=dict(bundleId=device.BUNDLE_ID))
        if url.endswith('screenshot'):
            return dict(value=base64.b64encode(b'synthetic screenshot').decode())
        return dict(value=None)
    monkeypatch.setattr(device, 'DeviceSession', Worker)
    monkeypatch.setattr(timing, '_http_json', http)
    monkeypatch.setattr(timing, 'WDAActionTimingCapture', lambda **kw: None)
    monkeypatch.setattr(capture, 'XCTestScreenRecorder', lambda *a, **kw: None)
    monkeypatch.setattr(scene, 'wait_for_centre_settle', lambda *a, **k: SettleResult(True, .5))
    monkeypatch.setattr(gameplay, 'is_menu_frame', lambda *a, **k: False)
    monkeypatch.setattr(gameplay, 'is_editor_frame', lambda *a, **k: False)
    def run(**kw):
        runs.append(kw['candidate']['sha256'])
        if len(runs) == failure_at:
            raise RuntimeError('synthetic live failure')
    monkeypatch.setattr(module, 'run_trial', run)
    monkeypatch.setattr(module, 'admit_trial', lambda p, rev: save_new(p / 'admission.json', dict(accepted=True)))
    kwargs = dict(root=experiment, candidate_id=None, repeatability=True, env_file=env_file,
                  ready_note='operator confirms ready', settings_note='synthetic waypoint/settings')
    if failure_at:
        with pytest.raises(RuntimeError, match='live failure'):
            cli.live_batch(**kwargs)
        assert len(runs) == failure_at
        assert (experiment / 'repeats/trial_04/live-failure.json').exists()
    else:
        cli.live_batch(**kwargs)
        assert len(runs) == module.REPEATS == len(connections) == len(resets)
    assert len(set(runs)) == 1
    with pytest.raises(ValueError, match='already started'):
        cli.live_batch(**kwargs)


def test_unapproved_batch_never_contacts_phone(experiment, monkeypatch):
    cli = load_cli()
    monkeypatch.setattr(cli.socket, 'gethostname', lambda: 'training-server')
    monkeypatch.setattr(cli.subprocess, 'check_output', lambda *a, **k: pytest.fail('phone preflight before review'))
    with pytest.raises(FileNotFoundError):
        cli.live_batch(experiment, candidate_id=None, repeatability=True, env_file=experiment / 'missing.env',
                       ready_note='ready', settings_note='synthetic waypoint/settings')
