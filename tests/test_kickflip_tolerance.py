"""Synthetic checks for the kickflip position-tolerance plan, run mode and review; no phone."""
import base64
import importlib.util
import json
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from trueskate_ai.collection.scene_settle import SettleResult
from trueskate_ai.research import kickflip_repeatability as recipe
from trueskate_ai.research import kickflip_review as review
from trueskate_ai.research import kickflip_tolerance as module
from trueskate_ai.research.curve_protocol import save_new

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def plan():
    return module.build_plan(REPO)


@pytest.fixture
def experiment(tmp_path):
    root = tmp_path / 'experiment'
    module.prepare(root, REPO, source_commit='synthetic')
    return root


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / 'scripts/collection' / f'{name}.py')
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


def test_plan_is_deterministic_blocked_and_changes_only_pop_and_flick(plan):
    assert module.build_plan(REPO) == plan
    base = next(c for c in recipe.prepare_manifest(REPO)['candidates'] if c['id'] == module.BASE_CANDIDATE)
    assert base['sha256'] == plan['base_sha256'] == module.BASE_SHA256
    levels = [a['level'] for a in plan['attempts']]
    assert len(levels) == 40 and [a['number'] for a in plan['attempts']] == list(range(1, 41))
    # Every block of four holds each distance once, so drift or an early stop hits all alike.
    assert all(sorted(levels[i:i + 4]) == plan['levels'] for i in range(0, 40, 4))
    assert levels[:4] != sorted(levels[:4]) or levels[4:8] != sorted(levels[4:8])
    for attempt in plan['attempts']:
        candidate = recipe.verify_seal(attempt['candidate'])
        push, pop, flick, catch = candidate['contacts']
        assert (push, catch) == (base['contacts'][0], base['contacts'][3])
        assert (candidate['pop_wait_s'], candidate['catch_wait_s']) == (base['pop_wait_s'], base['catch_wait_s'])
        assert [s['name'] for s in candidate['contacts']] == ['push', 'pop', 'flick', 'catch']
        if attempt['level'] == 0:
            assert candidate['contacts'] == base['contacts']
    assert module.build_plan(REPO, seed=1)['attempts'] != plan['attempts']


def test_every_pop_and_flick_point_moves_exactly_the_distance(plan):
    for attempt in plan['attempts']:
        for key, stroke, count in zip(('pop', 'flick'), attempt['candidate']['contacts'][1:3], (2, 3)):
            jitter = attempt['candidate']['jitter'][key]
            assert len(stroke['points']) == len(jitter['base_points']) == count
            for moved, original in zip(stroke['points'], jitter['base_points']):
                assert math.dist(moved, original) == pytest.approx(attempt['level'], abs=1e-12)
            recipe.safe_scaled_path(stroke['points'])  # On screen and clear of protected controls.


class Directions:
    """Fake generator that returns the given direction sets in turn."""
    def __init__(self, *draws):
        self.draws = list(draws)
    def uniform(self, low, high, size):
        return np.array(self.draws.pop(0), float)


@pytest.mark.parametrize('points,bad', [([[.5, .82], [.5, .85]], math.pi / 2),  # Down into the bottom row.
                                        ([[.03, .6], [.2, .6]], math.pi)])          # Left off the screen.
def test_jitter_redraws_directions_that_leave_the_screen_or_hit_controls(points, bad):
    up = 3 * math.pi / 2
    stroke, jitter = module.jitter_stroke('pop', points, [.075], .05, Directions([bad, bad], [up, up]))
    assert jitter['redraws'] == 1 and jitter['angles_rad'] == [up, up]
    recipe.safe_scaled_path(stroke['points'])
    for moved, original in zip(stroke['points'], points):
        assert moved == pytest.approx([original[0], original[1] - .05])


def test_jitter_gives_up_rather_than_clipping(monkeypatch):
    monkeypatch.setattr(module, 'MAX_REDRAWS', 20)
    # Inside the reset button every direction stays inside a control.
    with pytest.raises(ValueError, match='no valid directions'):
        module.jitter_stroke('pop', [[.5, .03], [.5, .04]], [.075], .01, np.random.default_rng(0))


def test_changed_base_recipe_is_refused(monkeypatch):
    monkeypatch.setattr(module, 'BASE_SHA256', '0' * 64)
    with pytest.raises(ValueError, match='approved kickflip recipe'):
        module.build_plan(REPO)


def record(root, number, *, error=None, live_failure=False):
    trial = module.attempt_dir(root, number)
    trial.mkdir(parents=True)
    save_new(trial / 'execution.json', dict(error=error, events=[]))
    if live_failure:
        save_new(trial / 'live-failure.json', dict(error='synthetic', no_replacements=True))
    return trial


def test_failures_stop_the_run_until_an_explicit_reason_and_are_never_rerun(experiment):
    plan = module.load_plan(experiment)
    assert [a['number'] for a in module.pending_attempts(experiment, plan)] == list(range(1, 41))
    record(experiment, 1)
    record(experiment, 2, error='recorder stop failed', live_failure=True)
    with pytest.raises(ValueError, match=r'technical failure in attempt\(s\) \[2\]'):
        module.pending_attempts(experiment, plan)
    with pytest.raises(ValueError, match='explicit reason'):
        module.pending_attempts(experiment, plan, '  ')
    pending = module.pending_attempts(experiment, plan, 'operator: retrieval failed; continue the plan')
    assert pending[0]['number'] == 3
    # Acknowledged once; a new failure needs a new reason.
    assert module.pending_attempts(experiment, plan)[0]['number'] == 3
    module.attempt_dir(experiment, 3).mkdir(parents=True)  # Reserved, never finished.
    with pytest.raises(ValueError, match=r'\[3\]'):
        module.pending_attempts(experiment, plan)
    log = [json.loads(line) for line in (experiment / 'continuations.jsonl').read_text().splitlines()]
    assert log[0]['acknowledged_failures'] == [2] and log[0]['next_attempt'] == 3


def test_attempts_run_in_plan_order_and_none_twice(experiment):
    plan = module.load_plan(experiment)
    record(experiment, 2)
    with pytest.raises(ValueError, match='plan order'):
        module.pending_attempts(experiment, plan)
    for number in [1, *range(3, 41)]:
        record(experiment, number)
    with pytest.raises(ValueError, match='every planned attempt has run'):
        module.pending_attempts(experiment, plan)
    with pytest.raises(FileExistsError):
        module.reserve_attempt(experiment, plan, 1)


@pytest.mark.parametrize('failure', [None, 'attempt', 'reconnect'])
def test_live_tolerance_run_keeps_every_guard_and_stops_without_replacement(experiment, monkeypatch, failure):
    import trueskate_ai.collection.gameplay_filter as gameplay
    import trueskate_ai.collection.scene_settle as scene
    import trueskate_ai.collection.wda_action_timing as timing
    import trueskate_ai.collection.xctest_capture as capture
    import trueskate_ai.sim.device as device
    cli = load_script('probe_kickflip_repeatability')
    plan = module.load_plan(experiment)
    env_file = experiment / 'synthetic.env'
    env_file.write_text('IPHONE_XR_UDID=synthetic\n')
    monkeypatch.setattr(cli.socket, 'gethostname', lambda: 'training-server')
    monkeypatch.setattr(cli.subprocess, 'check_output', lambda *a, **k: 'state = running')
    monkeypatch.setattr(cli.signal, 'signal', lambda *a: None)
    runs, resets, connections, cleanup_checks = [], [], [], []
    def cleanup(udid):
        cleanup_checks.append(len(connections))
        if failure == 'reconnect' and len(cleanup_checks) == 3:
            raise RuntimeError('synthetic attachment left on the phone')
    monkeypatch.setattr(cli, 'recording_cleanup_ready', cleanup)
    class Worker:
        def __init__(self, *a, **k):
            self.driver = None
        def connect(self):
            connections.append('connect')
            self.driver = SimpleNamespace(capabilities={'udid': 'synthetic'}, session_id='synthetic-appium',
                query_app_state=lambda b: 4, execute=lambda action, payload: resets.append(payload))
        def disconnect(self):
            self.driver = None
        def _active_bundle_id(self):
            return device.BUNDLE_ID
    def http(url):
        if url.endswith('/wda/actionTiming'):
            return dict(value=dict(schema_version=1, build_revision=recipe.WDA_REVISION, enabled=False))
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
    def run_trial(**kw):
        runs.append(kw['candidate']['sha256'])
        assert kw['revision'] == recipe.WDA_REVISION and kw['foreground_guard'] is not None
        error = 'synthetic recorder failure' if failure == 'attempt' and len(runs) == 4 else None
        save_new(kw['out'] / 'execution.json', dict(error=error, events=[]))
        if error:
            raise RuntimeError(error)
    monkeypatch.setattr(recipe, 'run_trial', run_trial)
    monkeypatch.setattr(recipe, 'admit_trial', lambda *a: pytest.fail('admission belongs on the laptop'))
    kwargs = dict(root=experiment, candidate_id=None, repeatability=False, tolerance=True, env_file=env_file,
                  ready_note='operator confirms Workshop waypoint', settings_note='synthetic waypoint/settings',
                  appium_port=4725)
    expected = [a['candidate']['sha256'] for a in plan['attempts']]
    if failure is None:
        cli.live_batch(**kwargs)
        assert runs == expected and len(connections) == len(resets) == 40
        # Checked before the first connection and before every reconnect.
        assert cleanup_checks == list(range(40))
        assert set(module.states(experiment, plan).values()) == {'recorded'}
        assert recipe.read_json(experiment / 'context.json')['appium_port'] == 4725
        with pytest.raises(ValueError, match='every planned attempt has run'):
            cli.live_batch(**kwargs)
        return
    with pytest.raises(RuntimeError, match='synthetic'):
        cli.live_batch(**kwargs)
    if failure == 'reconnect':
        # The cleanup check failed before attempt 3 was reserved: attempts 1-2 stay good, nothing is lost.
        assert runs == expected[:2]
        assert [module.attempt_state(module.attempt_dir(experiment, n)) for n in (1, 2, 3)] == \
            ['recorded', 'recorded', 'pending']
        return
    assert runs == expected[:4]
    assert module.attempt_state(module.attempt_dir(experiment, 4)) == 'technical failure'
    assert (module.attempt_dir(experiment, 4) / 'live-failure.json').exists()
    with pytest.raises(ValueError, match='explicit reason'):
        cli.live_batch(**kwargs)
    assert len(runs) == 4
    cli.live_batch(**kwargs, after_failure_reason='operator: continue after the recorder failure')
    assert runs == expected[:4] + expected[4:]  # Attempt 4 is never rerun.


def test_changed_settings_or_port_need_a_new_experiment(experiment, monkeypatch):
    cli = load_script('probe_kickflip_repeatability')
    monkeypatch.setattr(cli.socket, 'gethostname', lambda: 'training-server')
    save_new(experiment / 'context.json', dict(settings_note='first waypoint', appium_port=4725))
    monkeypatch.setattr(cli.subprocess, 'check_output', lambda *a, **k: pytest.fail('phone contacted'))
    with pytest.raises(ValueError, match='waypoint/settings changed'):
        cli.live_batch(experiment, candidate_id=None, repeatability=False, tolerance=True,
                       env_file=experiment / 'missing.env', ready_note='ready', settings_note='another waypoint',
                       appium_port=4725)


def test_run_holds_the_busy_lock_and_refuses_when_someone_else_does(experiment, monkeypatch):
    script = load_script('probe_kickflip_tolerance')
    lock = experiment / 'RIG-BUSY'
    seen = []
    monkeypatch.setattr(script.socket, 'gethostname', lambda: 'training-server')
    monkeypatch.setattr(script, 'load_runner', lambda: SimpleNamespace(
        live_batch=lambda *a, **k: seen.append((json.loads(lock.read_text())['out'], k['tolerance']))))
    kwargs = dict(env_file=Path('/rig/.env'), ready_note='ready', settings_note='waypoint', appium_port=4725,
                  after_failure_reason=None, lock=lock)
    script.run(experiment, **kwargs)
    assert seen == [(str(experiment), True)] and not lock.exists()
    lock.write_text(json.dumps(dict(owner='learn-replay pilot', pid=1)))
    with pytest.raises(RuntimeError, match='another recording'):
        script.run(experiment, **kwargs)
    assert len(seen) == 1 and json.loads(lock.read_text())['pid'] == 1  # Someone else's lock is left alone.
    monkeypatch.setattr(script.socket, 'gethostname', lambda: 'laptop')
    lock.unlink()
    with pytest.raises(ValueError, match='training-server'):
        script.run(experiment, **kwargs)
    assert not lock.exists()


def admitted(root, number):
    trial = module.attempt_dir(root, number)
    if not trial.exists():
        record(root, number)
    (trial / 'original.mov').write_bytes(f'synthetic {number}'.encode())
    pts = (np.arange(1200) / 60).tolist()
    gestures = [dict(role=r, video_start_s=t, encoded_duration_s=d) for r, t, d in
                (('push', 14., .13), ('pop', 14.6, .075), ('flick', 15.0, .1), ('catch', 15.6, .52))]
    save_new(trial / 'admission.json', dict(accepted=True, video_sha256=recipe.file_sha(trial / 'original.mov'),
             original_pts_s=pts, native_frame_s=1 / 60, gestures=gestures))
    return trial


def test_verdicts_are_checked_together_and_bound_to_the_video(experiment):
    for number in (1, 2, 3):
        admitted(experiment, number)
    record(experiment, 4, error='synthetic', live_failure=True)
    for bad, message in (('01X', 'codes look like'), ('1L 1F', 'twice'), ('41L', 'not in the plan'),
                         ('01L 04F', 'no admitted recording'), ('01L and 02F', 'codes look like'), ('', 'codes look like')):
        with pytest.raises(ValueError, match=message):
            module.record_verdicts(experiment, bad)
        assert not (module.attempt_dir(experiment, 1) / 'verdict.json').exists()
    assert module.record_verdicts(experiment, '01L, 2f 03u') == 3
    verdict = recipe.read_json(module.attempt_dir(experiment, 2) / 'verdict.json')
    assert verdict['code'] == 'F' and verdict['verdict'] == 'not a landed kickflip'
    with pytest.raises(ValueError, match='already has a verdict'):
        module.record_verdicts(experiment, '02L')
    module.record_verdicts(experiment, '02L', replace=True)
    summary = module.summarize(experiment)
    plan = module.load_plan(experiment)
    level_of = {a['number']: f"{a['level']:g}" for a in plan['attempts']}
    total = {k: sum(v[k] for v in summary['levels'].values()) for k in ('planned', 'run', 'technical_failures', 'L', 'F', 'U')}
    assert total == dict(planned=40, run=4, technical_failures=1, L=2, F=0, U=1)
    assert summary['levels'][level_of[1]]['L'] >= 1
    (module.attempt_dir(experiment, 1) / 'original.mov').write_bytes(b'another video')
    admission = recipe.read_json(module.attempt_dir(experiment, 1) / 'admission.json')
    admission['video_sha256'] = recipe.file_sha(module.attempt_dir(experiment, 1) / 'original.mov')
    (module.attempt_dir(experiment, 1) / 'admission.json').write_text(json.dumps(admission))
    with pytest.raises(ValueError, match='another video'):
        module.summarize(experiment)
    assert '| Distance |' in module.summary_markdown(summary)


def real_attempt(root, number):
    """An admitted attempt whose original.mov is a real 1,200-frame 60 fps video (testsrc2)."""
    trial = admitted(root, number)
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=s=32x64:r=60', '-frames:v', '1200',
                    '-c:v', 'libx264', '-threads', '1', '-y', str(trial / 'original.mov')], check=True)
    admission = recipe.read_json(trial / 'admission.json')
    admission['video_sha256'] = recipe.file_sha(trial / 'original.mov')
    (trial / 'admission.json').write_text(json.dumps(admission))
    return trial


def test_review_manifest_is_blind_and_bound_to_the_videos(experiment):
    for number in (1, 2):
        admitted(experiment, number)
    record(experiment, 3, error='synthetic', live_failure=True)
    manifest, sources = review.review_manifest(experiment)
    assert [s['code'] for s in manifest['samples']] == ['01', '02'] and sorted(sources) == [1, 2]
    sample, pts = manifest['samples'][0], np.arange(1200) / 60
    # From 0.5 s before the pop (14.6 s) to 3 s after the flick ends (15.1 s), every native frame.
    expected = [t for t in pts if 14.6 - .5 <= t <= 15.0 + .1 + 3.]
    assert len(sample['frame_times_s']) == len(expected) == sources[1]['last'] - sources[1]['first'] + 1
    assert sample['frame_times_s'][0] == 0 and [m['role'] for m in sample['marks']] == ['pop', 'flick']
    assert sample['marks'][0]['start_s'] == pytest.approx(.5, abs=1 / 60)
    assert sample['marks'][1]['end_s'] == pytest.approx(1., abs=1 / 60)
    # Blind: nothing in the manifest or the page names or carries the distance.
    text = json.dumps(manifest)
    for word in ('level', 'distance', 'jitter', 'points', 'candidate'):
        assert word not in text
    page = (REPO / 'src/trueskate_ai/research/kickflip_review.html').read_text().lower()
    assert 'level' not in page and 'distance' not in page
    # The hash binds the videos: a changed video makes a different review.
    trial = module.attempt_dir(experiment, 1)
    (trial / 'original.mov').write_bytes(b'another video')
    admission = recipe.read_json(trial / 'admission.json')
    admission['video_sha256'] = recipe.file_sha(trial / 'original.mov')
    (trial / 'admission.json').write_text(json.dumps(admission))
    assert review.review_manifest(experiment)[0]['sha256'] != manifest['sha256']


def test_review_server_serves_native_frames_and_saves_a_checked_export(experiment):
    import socket
    import threading
    import urllib.request
    from urllib.error import HTTPError
    for number in (1, 2):
        real_attempt(experiment, number)
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    result = []
    server = threading.Thread(target=lambda: result.append(review.serve(experiment, port=port)), daemon=True)
    server.start()
    base = f'http://127.0.0.1:{port}'
    for _ in range(200):
        try:
            page = urllib.request.urlopen(base + '/', timeout=5).read()
            break
        except OSError:
            server.join(.05)
    assert b'Kickflip review' in page and b'Save &amp; next' in page
    manifest = json.loads(urllib.request.urlopen(base + '/manifest').read())
    last = len(manifest['samples'][0]['frame_times_s']) - 1
    frames = [urllib.request.urlopen(f"{base}/frame/1/{n}?m={manifest['sha256'][:16]}").read() for n in (0, last)]
    assert all(f.startswith(b'\xff\xd8') for f in frames)
    for bad in (f'/frame/1/{last + 1}', '/frame/3/0'):
        with pytest.raises(HTTPError):
            urllib.request.urlopen(base + bad)

    def post(report):
        return urllib.request.urlopen(urllib.request.Request(
            base + '/export', data=json.dumps(report).encode(), headers={'Content-Type': 'application/json'}))
    report = dict(experiment=module.EXPERIMENT, manifest_sha256=manifest['sha256'],
                  attempts=[dict(code='01', verdict='L', note=''), dict(code='02', verdict='F', note='')])
    with pytest.raises(HTTPError) as refused:
        post(report)  # Not landed needs a note; the server says why and keeps running.
    assert refused.value.code == 400 and server.is_alive()
    report['attempts'][1]['note'] = 'bailed on the catch'
    saved = Path(json.loads(post(report).read())['path'])
    server.join(5)
    assert not server.is_alive() and result == [saved]
    assert saved.parent == experiment.resolve() / 'review' and json.loads(saved.read_text()) == report
    assert not list(experiment.glob('kickflip-review-*'))  # The frame cache is removed.
    assert review.import_report(experiment, saved) == 2
    verdict = recipe.read_json(module.attempt_dir(experiment, 2) / 'verdict.json')
    assert (verdict['code'], verdict['note'], verdict['source']) == ('F', 'bailed on the catch', saved.name)
    with pytest.raises(ValueError, match='already have verdicts'):
        review.serve(experiment, port=0)


def test_import_report_needs_every_attempt_and_a_note_unless_landed(experiment, tmp_path):
    for number in (1, 2):
        admitted(experiment, number)
    sha = review.review_manifest(experiment)[0]['sha256']
    path = tmp_path / 'report.json'

    def exported(*rows, digest=sha):
        path.write_text(json.dumps(dict(manifest_sha256=digest, attempts=[dict(zip(('code', 'verdict', 'note'), r))
                                                                        for r in rows])))
        return path
    for rows, digest, message in (((('01', 'L'), ('02', 'L')), '0' * 64, 'another set'),
                                  ((('01', 'L'),), sha, 'misses attempt'),
                                  ((('01', 'L'), ('01', 'L'), ('02', 'L')), sha, 'listed twice'),
                                  ((('01', 'L'), ('03', 'L')), sha, 'not under review'),
                                  ((('01', 'L'), ('02', 'X')), sha, 'no Landed'),
                                  ((('01', 'L'), ('02', 'U', '  ')), sha, 'needs a note')):
        with pytest.raises(ValueError, match=message):
            review.import_report(experiment, exported(*rows, digest=digest))
    assert not (module.attempt_dir(experiment, 1) / 'verdict.json').exists()
    assert review.import_report(experiment, exported(('01', 'L'), ('02', 'U', 'banner hidden'))) == 2
    with pytest.raises(ValueError, match='already has a verdict'):
        review.import_report(experiment, path)
    assert review.import_report(experiment, exported(('01', 'F', 'fell'), ('02', 'L')), replace=True) == 2
    total = {k: sum(v[k] for v in module.summarize(experiment)['levels'].values()) for k in ('L', 'F', 'U')}
    assert total == dict(L=1, F=1, U=0)


def test_cli_serves_the_review_and_imports_reports():
    import sys
    script = REPO / 'scripts/collection/probe_kickflip_tolerance.py'
    run = lambda *args: subprocess.run([sys.executable, str(script), '--root', '/tmp/x', *args],
                                       capture_output=True, text=True, check=True).stdout
    assert '--report' in run('verdicts', '--help') and '--port' in run('review', '--help')


def test_admit_all_uses_the_plan_revision_for_recorded_attempts_only(experiment, monkeypatch):
    record(experiment, 1)
    record(experiment, 2, error='synthetic', live_failure=True)
    calls = []
    def admit(trial, revision):
        calls.append((Path(trial).name, revision))
        if Path(trial).name == 'attempt_03':
            save_new(Path(trial) / 'admission.json', dict(accepted=False, error='disk full'))
            raise ValueError('disk full')
        save_new(Path(trial) / 'admission.json', dict(accepted=True))
        return dict(heldout_middle_error_s=.004)
    monkeypatch.setattr(recipe, 'admit_trial', admit)
    record(experiment, 3)
    results = module.admit_all(experiment, jobs=1)
    assert calls == [('attempt_01', recipe.WDA_REVISION), ('attempt_03', recipe.WDA_REVISION)]
    assert results['attempt_01']['admitted'] and not results['attempt_03']['admitted']
    assert module.admit_all(experiment, jobs=1) == {}  # Nothing is re-admitted unless asked.
    module.admit_all(experiment, jobs=1, retry_failed=True)
    assert calls[-1] == ('attempt_03', recipe.WDA_REVISION) and len(calls) == 3
    assert (module.attempt_dir(experiment, 3) / 'admission-failed-1.json').exists()
