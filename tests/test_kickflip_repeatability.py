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


def test_candidates_preserve_push_and_give_each_trick_stroke_one_finger(experiment):
    manifest = module.load_manifest(experiment)
    assert len(manifest['candidates']) == 13 and manifest['rejected'] == []
    first = manifest['candidates'][0]
    assert first['varied'] == [] and [c['name'] for c in first['contacts']] == ['push', 'trick']
    push = first['contacts'][0]
    assert push['encoded_duration_s'] == .02
    assert push['points'] == [[.7658, .3044], [.7658, .6797]]
    assert len(push['payload']['actions']) == 1
    for candidate in manifest['candidates'][1:]:
        assert candidate['contacts'][0] == push and candidate['varied']
        assert {k: v for k, v in candidate['parameters'].items() if k not in candidate['varied']} == \
            {k: v for k, v in module.KICKFLIP.items() if k not in candidate['varied']}
    for candidate in manifest['candidates']:
        sources = candidate['contacts'][1]['payload']['actions']
        assert [f['role'] for f in candidate['contacts'][1]['fingers']] == ['pop', 'flick', 'catch']
        assert len({len(source['actions']) for source in sources}) == 1
        for source in sources:
            assert sum(a['type'] == 'pointerDown' for a in source['actions']) == 1
            assert sum(a['type'] == 'pointerUp' for a in source['actions']) == 1
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
        if failure == 'foreground' and recorder.starts:
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
        assert len(calls) == 8 and clock() == 59
        gameplay = [e for e in execution['events'] if e['kind'] == 'gesture']
        assert [e['role'] for e in gameplay] == ['push', 'trick']
        assert gameplay[1]['call_start_monotonic_s'] - gameplay[0]['call_end_monotonic_s'] == pytest.approx(.48)
        assert gameplay[1]['encoded_duration_s'] == candidate['contacts'][1]['encoded_duration_s']
        assert [e['intended_s'] for e in execution['events'] if e['kind'] == 'control'] == [1.5, 30, 57]
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
                         ios_callback_latency_s=.35) for role, t in zip(('push', 'trick'), (10., 10.9))]
        save_new(trial / 'admission.json', dict(accepted=True, video_sha256=module.file_sha(trial / 'original.mov'),
                 gestures=gestures, heldout_middle_error_s=.01, native_frame_s=1/60, max_frame_gap_s=.033))
    result = module.report(experiment)
    assert result['attempted'] == 2 and result['admitted'] == 1 and not result['complete']
    assert result['outcomes'] == {'unassessed': 1}
    assert result['technical_failures'] == ['repeats/trial_02']
    assert result['timing_distributions']['push_to_trick']['median_s'] == pytest.approx(.9)


@pytest.mark.parametrize('failure', [None, 'dimensions', 'fps', 'pts', 'decode_count', 'control', 'middle'])
def test_admission_requires_native_decode_and_independent_calibration(tmp_path, monkeypatch, failure):
    roles = ['start', None, 'push', 'trick', None, 'middle', None, 'end']
    stamps = [1.5, 3., 14., 14.9, 22., 30., 49., 57.]
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
        assert [g['role'] for g in result['gestures']] == ['push', 'trick']


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
    runs, resets, connections, cleanup_checks = [], [], [], []
    monkeypatch.setattr(cli, 'recording_cleanup_ready', lambda udid: cleanup_checks.append(len(connections)))
    class Worker:
        def __init__(self, *a, **k):
            self.driver = None
        def connect(self):
            connections.append('connect')
            self.driver = SimpleNamespace(capabilities={'udid': 'synthetic'},
                session_id='synthetic-appium', query_app_state=lambda b: 4,
                execute=lambda action, payload: resets.append(payload))
        def disconnect(self):
            self.driver = None
        def _active_bundle_id(self):
            return device.BUNDLE_ID
    def http(url):
        if url.endswith('/wda/actionTiming'):
            return dict(value=dict(schema_version=1, build_revision=module.WDA_REVISION, enabled=False))
        if url.endswith('/appium/sessions'):
            return dict(value=[dict(id='synthetic-appium')] if len(connections) > len(runs) else [])
        if url.endswith('/status'):
            return dict(sessionId='synthetic-wda' if len(connections) > len(runs) else None)
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
    monkeypatch.setattr(gameplay, '_to_rgb01', lambda *a: np.zeros((2, 2, 3), np.float32))
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
        # Checked before the first connection and before every later reconnect.
        assert cleanup_checks == list(range(module.REPEATS))
    assert len(set(runs)) == 1
    with pytest.raises(ValueError, match='already started'):
        cli.live_batch(**kwargs)


def test_unapproved_batch_never_contacts_phone(experiment, monkeypatch):
    cli = load_cli()
    monkeypatch.setattr(cli.socket, 'gethostname', lambda: 'training-server')
    monkeypatch.setattr(cli.subprocess, 'check_output', lambda *a, **k: pytest.fail('phone preflight before review'))
    monkeypatch.setattr(cli, 'recording_cleanup_ready', lambda udid: pytest.fail('phone preflight before review'))
    with pytest.raises(FileNotFoundError):
        cli.live_batch(experiment, candidate_id=None, repeatability=True, env_file=experiment / 'missing.env',
                       ready_note='ready', settings_note='synthetic waypoint/settings')


def test_screenshot_guard_latency_is_reserved_before_submission(tmp_path, monkeypatch):
    clock, recorder, timing = Clock(), Recorder(), Timing()
    candidate = module.prepare_manifest(REPO)['candidates'][0]
    def perform(payload):
        timing.records.append({})
        clock.sleep(sum(a.get('duration', 0) for a in payload['actions'][0]['actions']) / 1000 + .1)
    monkeypatch.setattr(module, 'validate_action_timing_report', lambda *a, **k: [])
    module.run_trial(out=tmp_path, candidate=candidate, recorder=recorder, timing=timing,
        perform=perform, guard=lambda: clock.sleep(.2), settle=lambda reserve: SettleResult(True, 0.),
        revision='synthetic', context={}, clock=clock, sleep=clock.sleep, epoch=clock)
    events = module.read_json(tmp_path / 'execution.json')['events']
    assert all(e['lateness_s'] == pytest.approx(0.) for e in events)


def test_pre_recording_failure_history_preserves_budget_and_evidence(experiment, tmp_path):
    import shutil
    trial = experiment / 'setup/trial_01'
    trial.mkdir(parents=True)
    save_new(trial / 'reservation.json', dict(candidate_id='candidate_01'))
    save_new(trial / 'execution.json', dict(error='identity mismatch', events=[], video=None))
    successor = tmp_path / 'successor'
    manifest = module.prepare_manifest(REPO, experiment)
    assert manifest['prior_setup_attempts'] == 1
    shutil.copytree(experiment, successor / 'history/predecessor')
    save_new(successor / 'manifest.json', manifest)
    assert module.setup_attempt_count(successor) == 1
    reserved, _ = module.reserve_setup(successor, 'candidate_01')
    assert reserved.name == 'trial_02'
    assert module.setup_attempt_count(successor) == 2
    (successor / 'history/predecessor/setup/trial_01/execution.json').write_text('{}')
    with pytest.raises(ValueError, match='predecessor evidence changed'):
        module.load_manifest(successor)


@pytest.mark.parametrize('partial', ['video', 'actions', 'approval'])
def test_migration_cannot_bypass_recorded_or_reviewed_failures(experiment, partial):
    trial = experiment / 'setup/trial_01'
    trial.mkdir(parents=True)
    save_new(trial / 'execution.json', dict(error='failure',
        events=[{}] if partial == 'actions' else [], video={} if partial == 'video' else None))
    if partial == 'approval':
        save_new(experiment / 'approval.json', {})
    with pytest.raises(ValueError, match='cannot migrate|only failures'):
        module.prepare_manifest(REPO, experiment)


def test_retrieved_pre_gameplay_movie_can_migrate_but_gameplay_cannot(experiment):
    trial = experiment / 'setup/trial_01'
    trial.mkdir(parents=True)
    (trial / 'original.mov').write_bytes(b'retained short recording')
    execution = dict(error='control deadline', events=[], video=dict(n_bytes=(trial / 'original.mov').stat().st_size))
    save_new(trial / 'execution.json', execution)
    assert module.prepare_manifest(REPO, experiment)['prior_setup_attempts'] == 1
    execution['events'] = [dict(kind='gesture', role='push')]
    (trial / 'execution.json').write_text(json.dumps(execution))
    with pytest.raises(ValueError, match='before gameplay'):
        module.prepare_manifest(REPO, experiment)


def test_full_screen_guards_do_not_stretch_push_to_trick_wait(tmp_path, monkeypatch):
    clock, recorder, timing = Clock(), Recorder(), Timing()
    candidate = module.prepare_manifest(REPO)['candidates'][0]
    def perform(payload):
        timing.records.append({})
        clock.sleep(sum(a.get('duration', 0) for a in payload['actions'][0]['actions']) / 1000 + .3)
    monkeypatch.setattr(module, 'validate_action_timing_report', lambda *a, **k: [])
    module.run_trial(out=tmp_path, candidate=candidate, recorder=recorder, timing=timing,
        perform=perform, guard=lambda: clock.sleep(1.4), foreground_guard=lambda: clock.sleep(.18),
        settle=lambda reserve: (clock.sleep(4.6) or SettleResult(True, 4.6)), revision='synthetic', context={},
        clock=clock, sleep=clock.sleep, epoch=clock)
    events = module.read_json(tmp_path / 'execution.json')['events']
    gestures = [e for e in events if e['kind'] == 'gesture']
    assert all(e['lateness_s'] == pytest.approx(0.) for e in events)
    assert gestures[1]['call_start_monotonic_s'] - gestures[0]['call_end_monotonic_s'] == pytest.approx(.48)
    assert clock() - 1.4 < 60  # Initial full guard occurs before recorder.start().


def test_slow_wda_returns_never_overrun_the_trick(tmp_path, monkeypatch):
    # Trial 06: every call took ~0.69 s; a contact timed from a previous start was already late.
    clock, recorder, timing = Clock(), Recorder(), Timing()
    candidate = module.prepare_manifest(REPO)['candidates'][0]
    def perform(payload):
        timing.records.append({})
        clock.sleep(sum(a.get('duration', 0) for a in payload['actions'][0]['actions']) / 1000 + .65)
    monkeypatch.setattr(module, 'validate_action_timing_report', lambda *a, **k: [])
    module.run_trial(out=tmp_path, candidate=candidate, recorder=recorder, timing=timing,
        perform=perform, guard=lambda: clock.sleep(1.4), foreground_guard=lambda: clock.sleep(.18),
        settle=lambda reserve: (clock.sleep(4.6) or SettleResult(True, 4.6)), revision='synthetic', context={},
        clock=clock, sleep=clock.sleep, epoch=clock)
    events = module.read_json(tmp_path / 'execution.json')['events']
    push, trick = [e for e in events if e['kind'] == 'gesture']
    assert all(e['lateness_s'] == pytest.approx(0.) for e in events)
    assert trick['call_start_monotonic_s'] - push['call_end_monotonic_s'] == pytest.approx(.48)


@pytest.mark.parametrize('state', ['ready', 'no_tunnel', 'other_device', 'leftover', 'unlistable'])
def test_recording_cleanup_requires_tunnel_and_zero_attachments(monkeypatch, state):
    import io
    import urllib.error
    cli = load_cli()
    def urlopen(url, timeout):
        assert url.endswith('/synthetic-udid')
        if state == 'no_tunnel':
            raise urllib.error.HTTPError(url, 404, 'Not Found', {}, io.BytesIO(b''))
        body = b'{"udid": "synthetic-other"}' if state == 'other_device' else b'{"udid": "synthetic-udid"}'
        return io.BytesIO(body)
    monkeypatch.setattr(cli.urllib.request, 'urlopen', urlopen)
    count = 3 if state == 'leftover' else 0
    listing = SimpleNamespace(returncode=1 if state == 'unlistable' else 0, stderr='',
        stdout=f'Found {count} UUID-shaped attachment{"" if count == 1 else "s"} in testmanagerd Attachments')
    monkeypatch.setattr(cli.subprocess, 'run', lambda *a, **k: listing)
    if state == 'ready':
        cli.recording_cleanup_ready('synthetic-udid')
    else:
        with pytest.raises(RuntimeError, match='tunnel|registry|remain|list'):
            cli.recording_cleanup_ready('synthetic-udid')


def supersede(root):
    """Rewrite a predecessor as an older recipe set (different candidate hashes)."""
    manifest = module.load_manifest(root)
    manifest.pop('sha256')
    old = {k: v for k, v in manifest['candidates'][0].items() if k != 'sha256'}
    old['parameters'] = {**old['parameters'], 'flick_gap_s': .48}
    manifest['candidates'] = [module.seal(old)]
    (root / 'manifest.json').write_text(json.dumps(module.seal(manifest)))


def add_gameplay_trial(root, index, *, error=None, assessed=True):
    trial = root / 'setup' / f'trial_{index:02d}'
    trial.mkdir(parents=True)
    (trial / 'original.mov').write_bytes(f'synthetic gameplay {index}'.encode())
    save_new(trial / 'reservation.json', dict(candidate_id='candidate_01'))
    save_new(trial / 'execution.json', dict(error=error, events=[dict(kind='gesture', role='push')],
             video=dict(n_bytes=(trial / 'original.mov').stat().st_size)))
    if not error:
        sha = module.file_sha(trial / 'original.mov')
        save_new(trial / 'admission.json', dict(accepted=True, video_sha256=sha))
        if assessed:
            save_new(trial / 'assessment.json', dict(trick='360 FLIP', status='landed', video_sha256=sha))


def test_superseded_procedure_carries_gameplay_attempts_only_with_a_reason(experiment, tmp_path):
    import shutil
    supersede(experiment)
    add_gameplay_trial(experiment, 1)
    add_gameplay_trial(experiment, 2, error='gesture schedule overrun')
    with pytest.raises(ValueError, match='superseded-procedure reason'):
        module.prepare_manifest(REPO, experiment)
    manifest = module.prepare_manifest(REPO, experiment, 'flick gap now follows the pop return')
    history = manifest['prior_setup']
    assert manifest['prior_setup_attempts'] == 2 and history['superseded_reason']
    assert history['trials'] == {'trial_01': 'superseded-procedure outcome: 360 FLIP / landed',
                                 'trial_02': 'technical failure during gameplay'}
    successor = tmp_path / 'successor'
    shutil.copytree(experiment, successor / 'history/predecessor')
    save_new(successor / 'manifest.json', manifest)
    reserved, _ = module.reserve_setup(successor, 'candidate_01')
    assert reserved.name == 'trial_03'
    # Old outcomes stay in history and never enter a new candidate's review pool.
    assert [p.name for p in module.setup_trials(successor)] == ['trial_03']


def test_superseded_procedure_still_requires_assessed_gameplay(experiment):
    supersede(experiment)
    add_gameplay_trial(experiment, 1, assessed=False)
    with pytest.raises(ValueError, match='admitted and assessed'):
        module.prepare_manifest(REPO, experiment, 'reason')


def test_operator_kickflip_timeline_is_exact_in_one_payload():
    trick = module.trick_payload(module.KICKFLIP)
    fingers = {f['role']: f for f in trick['fingers']}
    pop_up = fingers['pop']['up_s']
    # One continuous 0.8 s pop (ms truncation may shorten it slightly); no separate hold.
    assert fingers['pop']['down_s'] == 0. and .785 <= pop_up <= .8
    assert fingers['flick']['down_s'] == pytest.approx(pop_up + .05)
    assert fingers['flick']['up_s'] == pytest.approx(pop_up + .11)
    assert fingers['catch']['down_s'] == pytest.approx(pop_up + .36)
    assert fingers['catch']['up_s'] == pytest.approx(pop_up + .66) == trick['encoded_duration_s']
    pop, flick, catch = trick['payload']['actions']
    kinds = [a['type'] for a in pop['actions']]
    assert kinds[:2] == ['pointerMove', 'pointerDown'] and 'pause' not in kinds[:kinds.index('pointerUp')]
    moves = [a['duration'] for a in pop['actions'] if a['type'] == 'pointerMove'][1:]
    # Equal distances with shrinking durations: exponential speed-up, mostly a slow creep.
    assert len(moves) == 16 and all(a >= b for a, b in zip(moves, moves[1:]))  # ms truncation ties
    assert moves[0] / sum(moves) > .5 and moves[-1] <= 10
    assert [a['y'] for a in pop['actions'] if a['type'] == 'pointerMove'][-1] == int(.67 * 896 + 150)
    assert {(a['x'], a['y']) for a in catch['actions'] if a['type'] == 'pointerMove'} == {(207, 448)}
    # Replaying WDA's path rules yields exactly one touch per finger at the planned times.
    touches = {f['role']: wda_touch_paths(source) for f, source in zip(trick['fingers'], trick['payload']['actions'])}
    ms = lambda seconds: int(round(seconds * 1000))
    assert touches == {'pop': [(0, ms(pop_up), (207, 600))],
                       'flick': [(ms(pop_up + .05), ms(pop_up + .11), (207, 448))],
                       'catch': [(ms(pop_up + .36), ms(pop_up + .66), (207, 448))]}


def wda_touch_paths(source):
    """FBW3CActionsSynthesizer at ae50404a: a move with no touch starts one at its end;
    a down is skipped only at index 1 after a move; otherwise a down starts a new touch."""
    touches, current, offset, position = [], None, 0, None
    for index, action in enumerate(source['actions']):
        duration = action.get('duration', 0)
        if action['type'] == 'pointerMove':
            position = (action['x'], action['y'])
            if current is None:
                current = [offset + duration, None, position]
                touches.append(current)
        elif action['type'] == 'pointerDown':
            if not (current is not None and index == 1 and source['actions'][0]['type'] == 'pointerMove'):
                current = [offset, None, position]
                touches.append(current)
        elif action['type'] == 'pointerUp':
            current[1] = offset
        offset += duration
    return [tuple(t) for t in touches]


def test_leading_pause_pattern_creates_a_phantom_touch():
    # The executor/spin-hold pattern: move(0), pause, move(0), down. WDA touches at t = 0
    # from the first move and never lifts it, then adds the intended touch.
    source = dict(actions=[dict(type='pointerMove', duration=0, x=207, y=448), dict(type='pause', duration=900),
                           dict(type='pointerMove', duration=0, x=207, y=448), dict(type='pointerDown'),
                           dict(type='pointerMove', duration=60, x=331, y=448), dict(type='pointerUp')])
    assert wda_touch_paths(source) == [(0, None, (207, 448)), (900, 960, (207, 448))]


def test_pop_cannot_reach_protected_bottom_controls():
    with pytest.raises(ValueError, match='protected'):
        module.trick_payload({**module.KICKFLIP, 'pop_length_pt': 200.})
